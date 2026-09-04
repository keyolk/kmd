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

/// 한 transcript 파일을 순회하며, user 프롬프트 뒤 **다음 사용자 프롬프트 전까지**의
/// 모든 assistant 텍스트를 이어붙인다(그 턴의 답변 전체). 첫 서두만 잡으면 오버랩이
/// 0에 수렴하므로 턴 전체를 모아야 실제 활용을 측정할 수 있다.
/// tool_result(role=user, content=array)와 슬래시 커맨드는 프롬프트로 치지 않는다.
fn scan_transcript(raw: &str, map: &mut HashMap<(String, String), String>) {
    let mut cur: Option<(String, String)> = None; // (session_id, prompt_key)
    let mut buf = String::new();
    let flush = |cur: &Option<(String, String)>, buf: &mut String, map: &mut HashMap<(String, String), String>| {
        if let Some(key) = cur {
            if !buf.trim().is_empty() {
                map.entry(key.clone()).or_insert_with(|| std::mem::take(buf));
            }
        }
        buf.clear();
    };
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
                // 실제 텍스트 프롬프트만 새 턴으로 — tool_result 등은 무시.
                if let Some(text) = extract_text(&v) {
                    if !text.trim().is_empty() {
                        // 이전 턴 답변을 확정하고 새 턴 시작
                        flush(&cur, &mut buf, map);
                        cur = Some((sid, norm_key(&text)));
                    }
                }
            }
            "assistant" => {
                if cur.is_some() {
                    if let Some(text) = extract_text(&v) {
                        if !text.trim().is_empty() {
                            buf.push_str(&text);
                            buf.push(' ');
                        }
                    }
                }
            }
            _ => {}
        }
    }
    flush(&cur, &mut buf, map);
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

/// 히트 파일의 컬렉션 이름. 옛 `qmd://` 스킴도 받는다(과거 로그가 그 스킴이다).
fn collection_of(file: &str) -> String {
    file.strip_prefix("qmd://")
        .or_else(|| file.strip_prefix("kmd://"))
        .and_then(|rest| rest.split('/').next())
        .unwrap_or("?")
        .to_string()
}

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
    // 무엇이 증강되나 — 컬렉션별 주입 파일 수
    let mut coll_count: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for e in &entries {
        for h in &e.hits {
            *coll_count.entry(collection_of(&h.file)).or_default() += 1;
        }
    }
    // 컬렉션별 활용 기여 — 활용된 프롬프트에 그 컬렉션이 주입됐는지.
    //
    // 주입 점유율만으로는 배분이 옳은지 알 수 없다. 실측 learnings가 주입의
    // 89%를 차지하는데, 코퍼스 자체가 1,431건 대 정제 지식 98건(93% 대 7%)이라
    // 그 점유율은 규모를 그대로 따른 것일 수 있다. 활용된 프롬프트에서의 분포와
    // 비교해야 "많이 주입돼서 많이 쓰인 것"과 "쓸모가 있어서 쓰인 것"이 갈린다.
    let mut coll_utilized: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();

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
            // 한 프롬프트에 같은 컬렉션이 여러 번 주입될 수 있으므로 dedup —
            // 세는 단위는 "그 컬렉션이 기여한 프롬프트 수"다.
            let mut seen = HashSet::new();
            for h in &e.hits {
                let coll = collection_of(&h.file);
                if seen.insert(coll.clone()) {
                    *coll_utilized.entry(coll).or_default() += 1;
                }
            }
        }
        examples.push((overlap, format!("{:.0}% {}", overlap * 100.0, short(&e.prompt))));
    }

    examples.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let utilization_rate = pct(utilized, matched);
    let avg_overlap = if matched > 0 { 100.0 * overlap_sum / matched as f64 } else { 0.0 };
    let metrics = serde_json::json!({
        "injected_total": injected_total,
        "matched_to_answer": matched,
        "utilized": utilized,
        "utilization_rate_pct": utilization_rate,
        "avg_overlap_pct": avg_overlap,
        "korean_injected": korean_inj,
        "korean_utilized": korean_util,
        "injected_by_collection": &coll_count,
        "note": "overlap approximates utilization via query-token reuse in the following answer; add snippet logging for precision",
    });
    crate::evaluation_log::record(
        "utilization",
        format!("{utilized}/{matched} utilized · {utilization_rate:.0}% · {injected_total} injected"),
        metrics.clone(),
    );

    if json {
        println!("{}", serde_json::to_string_pretty(&metrics)?);
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
        let total_files: usize = coll_count.values().sum();
        println!("\n무엇이 증강되나 (컬렉션별 주입 파일 {}건):", total_files);
        let mut ranked: Vec<_> = coll_count.iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(a.1));
        for (coll, n) in ranked {
            println!("  {:<18} {:>4}  ({:.0}%)", coll, n, pct(*n, total_files));
        }
        if utilized > 0 {
            // 주입 점유율과 활용 기여율을 나란히 놓는다. 코퍼스 규모 차이 때문에
            // 점유율만으로는 배분이 옳은지 알 수 없다 (learnings 1,431건 대 정제
            // 지식 98건). 활용 쪽 비율이 더 높은 컬렉션은 규모에 비해 값을 하는
            // 것이고, 낮은 컬렉션은 자리만 차지하는 것이다.
            println!(
                "\n활용된 {}개 프롬프트에 어느 컬렉션이 기여했나 (컬렉션당 프롬프트 수):",
                utilized
            );
            let mut util_ranked: Vec<_> = coll_utilized.iter().collect();
            util_ranked.sort_by(|a, b| b.1.cmp(a.1));
            for (coll, n) in util_ranked {
                let injected_share = coll_count.get(coll).copied().unwrap_or(0);
                println!(
                    "  {:<18} {:>4}  ({:.0}% of utilized | {:.0}% of injected files)",
                    coll,
                    n,
                    pct(*n, utilized),
                    pct(injected_share, total_files)
                );
            }
        }
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

// ------------------------------------------------------- 세션 드릴다운 ----

/// 실사용(사람) 세션인지 — session_id가 uuid 형태(8-4-…)면 실사용,
/// bench/test/hook 등 합성 세션은 제외.
fn is_real_session(sid: &str) -> bool {
    let b = sid.as_bytes();
    b.len() >= 9
        && b[..8].iter().all(|c| c.is_ascii_hexdigit())
        && b[8] == b'-'
}

/// 사람이 실제로 물은 프롬프트인지 — task-notification/tool-result 등 자동
/// 주입성 프롬프트는 관측 가치가 낮으므로 걸러낸다.
fn is_human_prompt(p: &str) -> bool {
    let t = p.trim_start();
    !(t.starts_with("<task-notification>")
        || t.starts_with("<command-")
        || t.starts_with("<tool-use")
        || t.starts_with("<local-command")
        || t.starts_with("Caveat:"))
}

/// `kmd show` — 실사용 세션에서 증강된 프롬프트→파일→(답변 활용 여부)를 훑는다.
/// session이 주어지면 그 세션만, 아니면 최근 실사용 프롬프트 위주로.
pub fn show(session: Option<&str>, limit: usize, all: bool, json: bool) -> Result<()> {
    let raw = std::fs::read_to_string(rag_log_path())?;
    let mut entries: Vec<RagLogEntry> = raw
        .lines()
        .filter_map(|l| serde_json::from_str::<RagLogEntry>(l).ok())
        .filter(|e| e.injected > 0)
        .filter(|e| all || is_real_session(&e.session_id))
        .filter(|e| all || is_human_prompt(&e.prompt))
        .collect();

    if let Some(sid) = session {
        entries.retain(|e| e.session_id.starts_with(sid));
    }

    // 동일 (session, 프롬프트 앞부분) 중복 제거 — 데몬+폴백 이중 로깅 방지.
    let mut seen = HashSet::new();
    entries.retain(|e| seen.insert((e.session_id.clone(), norm_key(&e.prompt))));

    // 최근 것부터
    entries.reverse();
    entries.truncate(limit);
    entries.reverse();

    if entries.is_empty() {
        println!("표시할 실사용 증강 기록이 없습니다 (--all 로 합성/자동 포함).");
        return Ok(());
    }

    let answers = build_answer_map();

    if json {
        let out: Vec<_> = entries
            .iter()
            .map(|e| {
                let ans = answers.get(&(e.session_id.clone(), norm_key(&e.prompt)));
                let used = ans.map(|a| answer_uses(e, a));
                serde_json::json!({
                    "session": e.session_id,
                    "prompt": e.prompt,
                    "hangul": e.hangul,
                    "query": e.query,
                    "files": e.hits.iter().map(|h| h.file.replace("qmd://","").replace("kmd://","")).collect::<Vec<_>>(),
                    "answered": ans.is_some(),
                    "used_ratio": used,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }

    let mut cur = String::new();
    for e in &entries {
        if e.session_id != cur {
            cur = e.session_id.clone();
            println!("\n━━ session {} ━━", &cur[..cur.len().min(8)]);
        }
        let ans = answers.get(&(e.session_id.clone(), norm_key(&e.prompt)));
        let mark = match ans.map(|a| answer_uses(e, a)) {
            Some(r) if r >= 0.15 => format!("✓ 활용 {:.0}%", r * 100.0),
            Some(r) => format!("· 답변있음 {:.0}%", r * 100.0),
            None => "  (답변 미발견)".to_string(),
        };
        let flag = if e.hangul { "KO" } else { "en" };
        println!("\n  [{}] {}  {}", flag, short(&e.prompt), mark);
        if let Some(q) = &e.query {
            println!("      q: {}", q.chars().take(64).collect::<String>());
        }
        for h in &e.hits {
            println!("      → {}", h.file.replace("qmd://", "").replace("kmd://", "").chars().take(78).collect::<String>());
        }
    }
    println!();
    Ok(())
}

/// 답변이 이 엔트리의 주입 신호(snippet 우선, 없으면 query)를 얼마나 재사용했나(0..1).
fn answer_uses(e: &RagLogEntry, answer: &str) -> f64 {
    let logged: String = e.hits.iter().map(|h| h.snippet.as_str()).collect::<Vec<_>>().join(" ");
    let signal = if logged.trim().is_empty() {
        e.query.clone().unwrap_or_default()
    } else {
        logged
    };
    let prompt_tokens = tokens(&e.prompt);
    let sig: HashSet<String> = tokens(&signal).into_iter().filter(|t| !prompt_tokens.contains(t)).collect();
    if sig.is_empty() {
        return 0.0;
    }
    let ans = tokens(answer);
    sig.iter().filter(|t| ans.contains(*t)).count() as f64 / sig.len() as f64
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

    #[test]
    fn real_session_detection() {
        assert!(is_real_session("a4d9c131-8b8c-404b-9ef6"));
        assert!(!is_real_session("bench-daemon"));
        assert!(!is_real_session("test-session-1"));
        assert!(!is_real_session("hook-e2e"));
    }

    #[test]
    fn human_prompt_filters_auto() {
        assert!(is_human_prompt("soda의 모든 로그 소스 싱크 경로 리스트해줘"));
        assert!(!is_human_prompt("<task-notification>\n<task-id>abc</task-id>"));
        assert!(!is_human_prompt("<command-message>clear</command-message>"));
        assert!(!is_human_prompt("Caveat: The messages below..."));
    }
}
