//! 주입 채택률(L2) — `kmd util`.
//!
//! "검색이 찾았나"(L1)를 넘어 "주입된 컨텍스트를 답변이 실제로 썼나"를 잰다.
//! rag.jsonl(주입 기록)과 Claude Code transcript(실제 답변)를 조인해서,
//! 각 주입 프롬프트에 대한 그 다음 assistant 답변이 주입 스니펫의 토큰을
//! 얼마나 재사용했는지 오버랩으로 근사한다.
//!
//! 완벽한 인과 측정은 아니지만(우연 일치·공통어 포함), 상대 비교와 추세
//! 관측용으로 충분하다. 지표:
//!   - matched: transcript에서 답변을 찾은 주입 엔트리 수
//!   - utilized: 답변이 주입 스니펫 토큰을 유의하게(>=임계) 재사용한 수
//!   - 평균 오버랩(주입 스니펫 대비 답변에 등장한 고유 토큰 비율)
//!
//! transcript 위치: ~/.claude/projects/<slug>/<uuid>.jsonl
//! rag.jsonl의 session_id == transcript 파일의 sessionId 로 조인.

use crate::rag::{RagLogEntry, rag_log_path};
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

fn projects_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME not set")).join(".claude/projects")
}

/// transcript에서 (session_id, 정규화된 프롬프트) → 그 다음 assistant 텍스트.
/// 여러 transcript 파일을 훑어 맵을 만든다.
fn build_answer_map() -> HashMap<(String, String), String> {
    let mut map = HashMap::new();
    let root = projects_dir();
    let Ok(projects) = std::fs::read_dir(&root) else {
        return map;
    };
    for proj in projects.flatten() {
        let Ok(files) = std::fs::read_dir(proj.path()) else {
            continue;
        };
        for f in files.flatten() {
            let path = f.path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            let Ok(raw) = std::fs::read_to_string(&path) else {
                continue;
            };
            scan_transcript(&raw, &mut map);
        }
    }
    map
}

/// 한 transcript 파일을 순회하며 user 프롬프트 뒤 첫 assistant 텍스트를 잇는다.
fn scan_transcript(raw: &str, map: &mut HashMap<(String, String), String>) {
    let mut pending: Option<(String, String)> = None; // (session_id, prompt_key)
    for line in raw.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let ty = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
        let sid = v
            .get("sessionId")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string();
        match ty {
            "user" => {
                if let Some(text) = extract_text(&v) {
                    // 슬래시/툴결과 등은 건너뛰고 실제 프롬프트만
                    if !text.trim().is_empty() {
                        pending = Some((sid, norm_key(&text)));
                    }
                }
            }
            "assistant" => {
                if let (Some((psid, pkey)), Some(text)) = (pending.clone(), extract_text(&v)) {
                    if !text.trim().is_empty() {
                        // 프롬프트 뒤 첫 텍스트 답변만 기록
                        map.entry((psid, pkey)).or_insert(text);
                        pending = None;
                    }
                }
            }
            _ => {}
        }
    }
}

/// user/assistant 메시지에서 텍스트 본문 추출.
fn extract_text(v: &serde_json::Value) -> Option<String> {
    let content = v.get("message")?.get("content")?;
    if let Some(s) = content.as_str() {
        return Some(s.to_string());
    }
    if let Some(arr) = content.as_array() {
        let mut out = String::new();
        for b in arr {
            if b.get("type").and_then(|t| t.as_str()) == Some("text") {
                if let Some(t) = b.get("text").and_then(|t| t.as_str()) {
                    out.push_str(t);
                    out.push(' ');
                }
            }
        }
        if !out.is_empty() {
            return Some(out);
        }
    }
    None
}

/// 프롬프트 조인 키 — 앞부분을 소문자·공백정규화해서 rag.jsonl과 맞춘다.
fn norm_key(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
        .chars()
        .take(80)
        .collect()
}

/// 토큰화 — 한글은 2-gram, 영숫자는 단어 단위. 스니펫/답변 오버랩용.
fn tokens(s: &str) -> HashSet<String> {
    let mut set = HashSet::new();
    let lower = s.to_lowercase();
    // 영숫자 단어
    let mut word = String::new();
    let flush = |w: &mut String, set: &mut HashSet<String>| {
        if w.chars().count() >= 2 {
            set.insert(std::mem::take(w));
        } else {
            w.clear();
        }
    };
    let hangul: Vec<char> = lower
        .chars()
        .filter(|c| ('\u{AC00}'..='\u{D7A3}').contains(c))
        .collect();
    for c in lower.chars() {
        if c.is_ascii_alphanumeric() {
            word.push(c);
        } else {
            flush(&mut word, &mut set);
        }
    }
    flush(&mut word, &mut set);
    // 한글 2-gram
    for w in hangul.windows(2) {
        set.insert(w.iter().collect());
    }
    set
}

/// 채택률 임계 — 답변이 재사용한 스니펫 고유토큰 비율이 이 이상이면 "활용됨".
const UTILIZED_THRESHOLD: f64 = 0.15;

pub fn print_utilization(since_secs: Option<u64>, json: bool) -> Result<()> {
    let log = rag_log_path();
    let raw = match std::fs::read_to_string(&log) {
        Ok(r) => r,
        Err(_) => {
            println!("no rag.jsonl at {}", log.display());
            return Ok(());
        }
    };
    let cutoff = since_secs.map(|s| now_secs().saturating_sub(s));
    let entries: Vec<RagLogEntry> = raw
        .lines()
        .filter_map(|l| serde_json::from_str::<RagLogEntry>(l).ok())
        .filter(|e| e.injected > 0)
        .filter(|e| match cutoff {
            Some(c) => e.ts.parse::<u64>().map(|t| t >= c).unwrap_or(true),
            None => true,
        })
        .collect();

    if entries.is_empty() {
        println!("no injected rag entries to evaluate");
        return Ok(());
    }

    let answers = build_answer_map();

    let injected_total = entries.len();
    let mut matched = 0usize;
    let mut utilized = 0usize;
    let mut overlap_sum = 0.0f64;
    let mut korean_inj = 0usize;
    let mut korean_util = 0usize;
    let mut examples: Vec<(f64, String)> = Vec::new();

    for e in &entries {
        // 주입 스니펫은 로그에 파일만 남으므로, 조인 키로 답변을 찾은 뒤
        // 프롬프트+쿼리를 스니펫 대용 신호로 사용... 대신 여기서는 답변이
        // 주입 파일의 "쿼리 토큰"을 얼마나 재사용했는지로 근사한다.
        // 더 정확히 하려면 rag.jsonl에 snippet을 남겨야 한다(향후).
        let key = norm_key(&e.prompt);
        let Some(answer) = answers.get(&(e.session_id.clone(), key)) else {
            continue;
        };
        matched += 1;
        if e.hangul {
            korean_inj += 1;
        }

        // 신호 소스: 주입된 스니펫이 로그에 있으면 그 토큰을(정밀), 없으면
        // 주입 쿼리 토큰을(근사) 답변이 얼마나 재사용했는지로 채택을 측정한다.
        let logged_snippets: String = e
            .hits
            .iter()
            .map(|h| h.snippet.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let signal = if logged_snippets.trim().is_empty() {
            e.query.clone().unwrap_or_default()
        } else {
            logged_snippets
        };
        // 프롬프트 자체 토큰은 답변과 당연히 겹치므로 신호에서 제외 —
        // "주입이 답변에 기여했나"를 보려면 프롬프트에 없던 토큰만 카운트.
        let prompt_tokens = tokens(&e.prompt);
        let sig_tokens: HashSet<String> = tokens(&signal)
            .into_iter()
            .filter(|t| !prompt_tokens.contains(t))
            .collect();
        if sig_tokens.is_empty() {
            continue;
        }
        let ans_tokens = tokens(answer);
        let hit = sig_tokens.iter().filter(|t| ans_tokens.contains(*t)).count();
        let overlap = hit as f64 / sig_tokens.len() as f64;
        overlap_sum += overlap;
        if overlap >= UTILIZED_THRESHOLD {
            utilized += 1;
            if e.hangul {
                korean_util += 1;
            }
        }
        examples.push((overlap, format!("{:.0}% {}", overlap * 100.0, short(&e.prompt))));
    }

    examples.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    if json {
        let out = serde_json::json!({
            "injected_total": injected_total,
            "matched_to_answer": matched,
            "utilized": utilized,
            "utilization_rate_pct": pct(utilized, matched),
            "avg_overlap_pct": if matched>0 { 100.0*overlap_sum/matched as f64 } else {0.0},
            "korean_injected": korean_inj,
            "korean_utilized": korean_util,
            "note": "overlap approximates utilization via query-token reuse in the following answer; add snippet logging for precision",
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        println!("kmd util — 주입 채택률 (L2)\n");
        println!("주입 엔트리:        {}", injected_total);
        println!(
            "답변 매칭:          {}  ({:.0}% of injected — 나머지는 transcript 미발견/벤치 세션)",
            matched,
            pct(matched, injected_total)
        );
        println!(
            "활용됨(≥{:.0}% overlap): {}  → 채택률 {:.0}%",
            UTILIZED_THRESHOLD * 100.0,
            utilized,
            pct(utilized, matched)
        );
        println!(
            "평균 오버랩:        {:.0}%",
            if matched > 0 { 100.0 * overlap_sum / matched as f64 } else { 0.0 }
        );
        println!("한글 주입/활용:     {} / {}", korean_inj, korean_util);
        if !examples.is_empty() {
            println!("\n상위 활용 예시:");
            for (_, s) in examples.iter().take(8) {
                println!("  {}", s);
            }
        }
        println!("\n주: 오버랩은 '주입 쿼리 토큰의 답변 내 재사용'으로 채택을 근사한다.");
        println!("   정밀 측정하려면 rag.jsonl에 snippet을 남겨야 한다(로드맵).");
    }
    Ok(())
}

fn short(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(48).collect()
}

fn pct(n: usize, d: usize) -> f64 {
    if d == 0 { 0.0 } else { 100.0 * n as f64 / d as f64 }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_hangul_bigrams_and_words() {
        let t = tokens("인증서 SAN check");
        assert!(t.contains("인증")); // 한글 2-gram
        assert!(t.contains("증서"));
        assert!(t.contains("san")); // 소문자 단어
        assert!(t.contains("check"));
        assert!(!t.contains("a")); // 1글자 단어는 버림
    }

    #[test]
    fn norm_key_lowercases_and_collapses() {
        assert_eq!(norm_key("  Hello   World  "), "hello world");
    }

    #[test]
    fn overlap_excludes_prompt_tokens() {
        // 프롬프트에 이미 있던 토큰은 신호에서 빠지고, 주입 고유 토큰만 남아야 한다.
        let prompt = tokens("인증서 확인");
        let signal: std::collections::HashSet<String> = tokens("인증서 doppler externalsecret")
            .into_iter()
            .filter(|t| !prompt.contains(t))
            .collect();
        // "인증"/"증서"는 프롬프트에 있으니 제외, doppler/externalsecret는 남음.
        assert!(signal.contains("doppler"));
        assert!(!signal.contains("인증"));
    }
}
