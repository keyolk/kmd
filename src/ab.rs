//! A/B 블라인드 비교(L3) — `kmd ab`.
//!
//! 같은 프롬프트 집합에 대해 kmd와 qmd가 각각 주입할 컨텍스트를 생성하고,
//! A/B를 블라인드로 섞어 판정용 산출물(JSONL)을 낸다. 최종 답변 품질 판정은
//! LLM/사람 judge의 몫이지만, 오늘 당장 돌아가도록 검색 관련도 프록시 판정을
//! 내장한다(프롬프트 키워드가 각 컨텍스트에 얼마나 담겼는지).
//!
//! 사용:
//!   kmd ab --prompts eval/prompts.txt        # 블라인드 페어 + 프록시 판정 리포트
//!   kmd ab --prompts eval/prompts.txt --emit pairs.jsonl   # 판정용 페어만 출력
//!   kmd ab --judge verdicts.jsonl            # 외부 판정 취합 → win/tie/loss
//!
//! verdicts.jsonl 한 줄: {"id": <n>, "winner": "A"|"B"|"tie"}

use crate::rag;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

const K: usize = 3;

/// 블라인드 페어 — A/B가 어느 엔진인지는 map에만 남기고 판정자에겐 숨긴다.
#[derive(Serialize)]
struct BlindPair {
    id: usize,
    prompt: String,
    hangul: bool,
    context_a: String,
    context_b: String,
    // 판정 후 집계를 위해 유지하되, --emit 시엔 제거해 블라인드 보장
    #[serde(skip_serializing_if = "Option::is_none")]
    a_engine: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    b_engine: Option<String>,
}

#[derive(Deserialize)]
struct Verdict {
    id: usize,
    /// "A" | "B" | "tie"
    winner: String,
}

fn load_prompts(path: &Path) -> Result<Vec<String>> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("cannot read prompts {}", path.display()))?;
    Ok(raw
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(String::from)
        .collect())
}

/// kmd 컨텍스트 — 실제 rag 파이프라인 그대로(claude 컬렉션 필터 포함).
fn kmd_context(prompt: &str) -> Result<String> {
    let outcome = rag::run_pipeline(prompt)?;
    Ok(outcome.context.unwrap_or_default())
}

/// qmd 컨텍스트 — 같은 게이팅/키워드로 qmd search 결과를 <qmd-context> 형태로.
fn qmd_context(prompt: &str) -> String {
    let query = match rag::gate(prompt) {
        Ok(q) => q,
        Err(_) => return String::new(),
    };
    let out = std::process::Command::new("qmd")
        .args(["search", &query, "-n", &K.to_string(), "--json"])
        .output();
    let Ok(out) = out else {
        return String::new();
    };
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(&out.stdout) else {
        return String::new();
    };
    let Some(arr) = v.as_array() else {
        return String::new();
    };
    let chunks: Vec<String> = arr
        .iter()
        .filter_map(|h| {
            let file = h.get("file").and_then(|f| f.as_str())?;
            // rag와 동일하게 claude 컬렉션만
            let coll = file.strip_prefix("qmd://").and_then(|r| r.split('/').next()).unwrap_or("");
            if !rag::CLAUDE_COLLECTIONS.contains(&coll) {
                return None;
            }
            let title = h.get("title").and_then(|t| t.as_str()).unwrap_or(file);
            let snippet = h.get("snippet").and_then(|s| s.as_str()).unwrap_or("");
            Some(format!("[QMD] {}\n{}", title, snippet))
        })
        .take(K)
        .collect();
    if chunks.is_empty() {
        String::new()
    } else {
        format!(
            "<qmd-context>\nRelevant knowledge from your QMD index:\n\n{}\n</qmd-context>",
            chunks.join("\n---\n")
        )
    }
}

/// 결정적 A/B 스왑 — id 홀짝으로 배정(판정자에겐 무작위, 집계엔 재현 가능).
fn swap(id: usize) -> bool {
    id % 2 == 1
}

/// 프롬프트 키워드가 컨텍스트에 얼마나 담겼나 — 프록시 관련도 점수(0..1).
fn relevance(prompt: &str, context: &str) -> f64 {
    let q = rag::extract_keywords(prompt);
    let terms: Vec<String> = q.split_whitespace().map(|t| t.to_lowercase()).collect();
    if terms.is_empty() {
        return 0.0;
    }
    let hay = context.to_lowercase();
    let hit = terms.iter().filter(|t| hay.contains(*t)).count();
    hit as f64 / terms.len() as f64
}

pub fn run(
    prompts_path: Option<&Path>,
    emit: Option<&Path>,
    judge: Option<&Path>,
    json: bool,
) -> Result<()> {
    // 판정 취합 모드
    if let Some(vpath) = judge {
        return tally(vpath, prompts_path, json);
    }

    let path = prompts_path.context("--prompts <file> required (or --judge)")?;
    let prompts = load_prompts(path)?;
    if prompts.is_empty() {
        anyhow::bail!("no prompts in {}", path.display());
    }

    let mut pairs = Vec::new();
    for (i, prompt) in prompts.iter().enumerate() {
        let kc = kmd_context(prompt)?;
        let qc = qmd_context(prompt);
        // 둘 다 비면 판정 의미 없음 — 건너뛴다
        if kc.is_empty() && qc.is_empty() {
            continue;
        }
        let id = i;
        let (a_eng, a_ctx, b_eng, b_ctx) = if swap(id) {
            ("qmd", qc.clone(), "kmd", kc.clone())
        } else {
            ("kmd", kc.clone(), "qmd", qc.clone())
        };
        pairs.push(BlindPair {
            id,
            prompt: prompt.clone(),
            hangul: rag::has_hangul(prompt),
            context_a: a_ctx,
            context_b: b_ctx,
            a_engine: Some(a_eng.into()),
            b_engine: Some(b_eng.into()),
        });
    }

    // 판정용 산출물만 출력 (엔진 라벨 제거 → 블라인드)
    if let Some(out) = emit {
        use std::io::Write;
        let mut f = std::fs::File::create(out)?;
        for p in &pairs {
            let blind = serde_json::json!({
                "id": p.id,
                "prompt": p.prompt,
                "context_a": p.context_a,
                "context_b": p.context_b,
            });
            writeln!(f, "{}", serde_json::to_string(&blind)?)?;
        }
        // 정답 키(map)는 별도 파일로 — 취합 시 사용
        let key_path = out.with_extension("key.jsonl");
        let mut kf = std::fs::File::create(&key_path)?;
        for p in &pairs {
            writeln!(
                kf,
                "{}",
                serde_json::json!({
                    "id": p.id,
                    "a_engine": p.a_engine,
                    "b_engine": p.b_engine,
                })
            )?;
        }
        eprintln!(
            "wrote {} blind pairs → {}\nkey → {}\njudge them, then: kmd ab --judge <verdicts.jsonl> --prompts {}",
            pairs.len(),
            out.display(),
            key_path.display(),
            out.display(),
        );
        return Ok(());
    }

    // 내장 프록시 판정 — 검색 관련도로 win/tie/loss 근사
    proxy_report(&pairs, json)
}

/// 프록시 판정 — 각 페어에서 kmd/qmd 컨텍스트의 프롬프트 관련도를 비교.
fn proxy_report(pairs: &[BlindPair], json: bool) -> Result<()> {
    let (mut kmd_win, mut qmd_win, mut tie) = (0usize, 0usize, 0usize);
    let (mut k_ko, mut q_ko, mut t_ko) = (0usize, 0usize, 0usize);
    let (mut kmd_cov, mut qmd_cov) = (0usize, 0usize); // 컨텍스트를 낸 횟수
    let mut kmd_rel = 0.0;
    let mut qmd_rel = 0.0;

    for p in pairs {
        let (kc, qc) = if p.a_engine.as_deref() == Some("kmd") {
            (&p.context_a, &p.context_b)
        } else {
            (&p.context_b, &p.context_a)
        };
        if !kc.is_empty() {
            kmd_cov += 1;
        }
        if !qc.is_empty() {
            qmd_cov += 1;
        }
        let kr = relevance(&p.prompt, kc);
        let qr = relevance(&p.prompt, qc);
        kmd_rel += kr;
        qmd_rel += qr;
        // 관련도 + 커버리지(빈 컨텍스트는 패배) 종합
        let ks = if kc.is_empty() { -1.0 } else { kr };
        let qs = if qc.is_empty() { -1.0 } else { qr };
        let margin = 0.001;
        if (ks - qs).abs() <= margin {
            tie += 1;
            if p.hangul {
                t_ko += 1;
            }
        } else if ks > qs {
            kmd_win += 1;
            if p.hangul {
                k_ko += 1;
            }
        } else {
            qmd_win += 1;
            if p.hangul {
                q_ko += 1;
            }
        }
    }
    let n = pairs.len();
    if json {
        let out = serde_json::json!({
            "pairs": n,
            "kmd_win": kmd_win, "qmd_win": qmd_win, "tie": tie,
            "kmd_win_pct": pct(kmd_win, n), "qmd_win_pct": pct(qmd_win, n),
            "korean": {"kmd_win": k_ko, "qmd_win": q_ko, "tie": t_ko},
            "kmd_coverage": kmd_cov, "qmd_coverage": qmd_cov,
            "kmd_avg_relevance": kmd_rel / n as f64,
            "qmd_avg_relevance": qmd_rel / n as f64,
            "judge": "built-in proxy (prompt-keyword coverage in context)",
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        println!("kmd ab — A/B 비교 (L3, 내장 프록시 판정)\n");
        println!("페어: {}", n);
        println!(
            "kmd 승 {} ({:.0}%)   qmd 승 {} ({:.0}%)   무승부 {}",
            kmd_win, pct(kmd_win, n), qmd_win, pct(qmd_win, n), tie
        );
        println!(
            "  한글만:  kmd {} · qmd {} · tie {}",
            k_ko, q_ko, t_ko
        );
        println!(
            "컨텍스트 제공: kmd {}/{}   qmd {}/{}",
            kmd_cov, n, qmd_cov, n
        );
        println!(
            "평균 관련도:   kmd {:.2}   qmd {:.2}",
            kmd_rel / n as f64,
            qmd_rel / n as f64
        );
        println!("\n주: 내장 판정은 '프롬프트 키워드가 컨텍스트에 담긴 비율' 프록시다.");
        println!("   최종 답변 품질은 --emit 후 LLM/사람 블라인드 판정으로 확정하라.");
    }
    Ok(())
}

/// 외부 판정 취합 — verdicts.jsonl(A/B/tie) + key로 win/tie/loss.
fn tally(vpath: &Path, prompts_path: Option<&Path>, json: bool) -> Result<()> {
    let vraw = std::fs::read_to_string(vpath)
        .with_context(|| format!("cannot read verdicts {}", vpath.display()))?;
    let verdicts: Vec<Verdict> = vraw
        .lines()
        .filter_map(|l| serde_json::from_str::<Verdict>(l).ok())
        .collect();

    // key 파일 추론: --prompts pairs.jsonl → pairs.key.jsonl
    let key_path = prompts_path
        .context("--prompts <pairs.jsonl> required to resolve the .key.jsonl mapping")?
        .with_extension("key.jsonl");
    let kraw = std::fs::read_to_string(&key_path)
        .with_context(|| format!("cannot read key {}", key_path.display()))?;
    use std::collections::HashMap;
    let mut key: HashMap<usize, (String, String)> = HashMap::new();
    for l in kraw.lines() {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(l) {
            let id = v.get("id").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
            let a = v.get("a_engine").and_then(|x| x.as_str()).unwrap_or("").to_string();
            let b = v.get("b_engine").and_then(|x| x.as_str()).unwrap_or("").to_string();
            key.insert(id, (a, b));
        }
    }

    let (mut kmd_win, mut qmd_win, mut tie) = (0usize, 0usize, 0usize);
    for v in &verdicts {
        let Some((a, b)) = key.get(&v.id) else {
            continue;
        };
        let winner_engine = match v.winner.as_str() {
            "A" => a.as_str(),
            "B" => b.as_str(),
            _ => "tie",
        };
        match winner_engine {
            "kmd" => kmd_win += 1,
            "qmd" => qmd_win += 1,
            _ => tie += 1,
        }
    }
    let n = kmd_win + qmd_win + tie;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "pairs_judged": n, "kmd_win": kmd_win, "qmd_win": qmd_win, "tie": tie,
                "kmd_win_pct": pct(kmd_win, n), "qmd_win_pct": pct(qmd_win, n),
                "judge": "external blind verdicts",
            }))?
        );
    } else {
        println!("kmd ab — 외부 블라인드 판정 취합\n");
        println!("판정된 페어: {}", n);
        println!(
            "kmd 승 {} ({:.0}%)   qmd 승 {} ({:.0}%)   무승부 {}",
            kmd_win, pct(kmd_win, n), qmd_win, pct(qmd_win, n), tie
        );
    }
    Ok(())
}

fn pct(n: usize, d: usize) -> f64 {
    if d == 0 { 0.0 } else { 100.0 * n as f64 / d as f64 }
}
