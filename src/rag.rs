//! RAG 훅 파이프라인 — UserPromptSubmit stdin JSON을 받아
//! 게이팅 → 키워드 추출 → BM25 검색 → <qmd-context> 출력.
//! 모든 결정을 JSONL로 로깅한다 (실사용 관측이 목적).

use crate::bm25::{self, SearchHit};
use crate::config;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

pub const MIN_PROMPT_LENGTH: usize = 20;
pub const MAX_RESULTS: usize = 3;
const SKIP_PREFIXES: &[&str] = &["/", "yes", "no", "ok", "sure", "thanks", "thank"];

/// 주입 대상 컬렉션. memory/rules는 CLAUDE.md로 상시 로드되므로 제외 대상 후보지만
/// 지금은 qmd_rag.py와 동일하게 유지 — stats로 실측 후 조정한다.
pub const CLAUDE_COLLECTIONS: &[&str] = &[
    "claude-memory",
    "claude-rules",
    "claude-agents",
    "claude-skills",
    "claude-contexts",
    "learnings",
];

const STOP_WORDS: &[&str] = &[
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for", "of", "and", "or", "but", "not",
    "with", "from", "by", "as", "do", "does", "did", "has", "have", "had", "be", "been", "being",
    "am", "are", "was", "were", "will", "would", "could", "should", "may", "might", "can", "shall",
    "this", "that", "these", "those", "i", "you", "he", "she", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their", "what", "which", "who", "whom",
    "how", "when", "where", "why", "if", "then", "so", "very", "just", "about", "also", "up",
    "out", "all", "some", "any", "each", "every", "no", "into", "through", "during", "before",
    "after", "above", "below", "between", "there", "here", "much", "many", "more", "most",
    "other", "like", "need", "want", "get", "make", "use", "try", "know",
];

#[derive(Debug, Deserialize)]
struct HookInput {
    #[serde(default)]
    prompt: String,
    #[serde(default)]
    session_id: String,
}

/// rag.jsonl 한 줄 — 파이프라인의 모든 결정을 담는다.
#[derive(Debug, Serialize, Deserialize)]
pub struct RagLogEntry {
    pub ts: String,
    pub session_id: String,
    /// 프롬프트 원문 (관측용 — 로컬 파일이므로 그대로 저장)
    pub prompt: String,
    pub prompt_len: usize,
    pub hangul: bool,
    /// gated | searched
    pub stage: String,
    /// 게이팅 사유: too_short | skip_prefix | query_too_short | (searched면 없음)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gate_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query: Option<String>,
    pub injected: usize,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub hits: Vec<RagHitLog>,
    pub latency_ms: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RagHitLog {
    pub file: String,
    pub score: f32,
    /// 주입된 스니펫 (L2 채택률 측정용 — 없으면 빈 문자열)
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub snippet: String,
}

pub fn state_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("KMD_STATE_DIR") {
        return PathBuf::from(dir);
    }
    PathBuf::from(std::env::var("HOME").expect("HOME not set")).join(".local/state/kmd")
}

pub fn rag_log_path() -> PathBuf {
    state_dir().join("rag.jsonl")
}

pub fn has_hangul(s: &str) -> bool {
    s.chars().any(|c| ('\u{AC00}'..='\u{D7A3}').contains(&c))
}

/// qmd_rag.py의 extract_keywords와 동일 로직 + 한글 인지.
/// 한글 토큰은 조사 제거를 tantivy 쪽 형태소 분석이 하므로 그대로 통과.
pub fn extract_keywords(prompt: &str) -> String {
    let strip: &[char] = &['?', '.', ',', '!', ':', ';', '"', '\'', '(', ')', '[', ']', '{', '}'];
    prompt
        .to_lowercase()
        .split_whitespace()
        .map(|w| w.trim_matches(strip))
        .filter(|w| !STOP_WORDS.contains(w) && w.chars().count() > 2)
        .take(12)
        .collect::<Vec<_>>()
        .join(" ")
}

/// 게이팅. 통과하면 검색 쿼리를 반환.
pub fn gate(prompt: &str) -> std::result::Result<String, &'static str> {
    let p = prompt.trim();
    if p.chars().count() < MIN_PROMPT_LENGTH {
        return Err("too_short");
    }
    // 사람이 실제로 물은 게 아닌 구조적 노이즈는 검색하지 않는다.
    // (task-notification/tool-result/command 태그, 붙여넣은 터미널 출력 등)
    if is_structural_noise(p) {
        return Err("noise");
    }
    let lower = p.to_lowercase();
    // "y"/"n" 단독 답변은 too_short에 이미 걸리므로 접두사만 확인
    if SKIP_PREFIXES.iter().any(|pre| lower.starts_with(pre)) {
        return Err("skip_prefix");
    }
    let q = extract_keywords(p);
    if q.chars().count() < 5 {
        return Err("query_too_short");
    }
    Ok(q)
}

/// 자동 주입성/비질의 프롬프트 판별 — 사람이 타이핑한 질문이 아닌 것.
fn is_structural_noise(p: &str) -> bool {
    // 1) 자동 태그로 시작 (Claude Code가 주입하는 알림/커맨드/툴 결과)
    const NOISE_PREFIXES: &[&str] = &[
        "<task-notification",
        "<command-",
        "<tool-use",
        "<local-command",
        "<system-reminder",
        "caveat:",
    ];
    let lower_start: String = p.chars().take(24).collect::<String>().to_lowercase();
    if NOISE_PREFIXES.iter().any(|pre| lower_start.starts_with(pre)) {
        return true;
    }
    // 2) URL 단독 (첫 토큰이 URL이고 남는 텍스트가 거의 없음)
    let first = p.split_whitespace().next().unwrap_or("");
    if (first.starts_with("http://") || first.starts_with("https://"))
        && p.split_whitespace().count() <= 2
    {
        return true;
    }
    // 3) 붙여넣은 터미널 출력/프롬프트 (셸 프롬프트·로그 라인으로 시작)
    let t = p.trim_start();
    if t.starts_with('❯') || t.starts_with("$ ") || t.starts_with("> ") {
        return true;
    }
    // 4) 이전에 주입된 컨텍스트를 되붙인 경우 (재귀 오염 방지)
    if t.starts_with("<qmd-context")
        || t.starts_with("[QMD]")
        || t.starts_with("━━ session")
    {
        return true;
    }
    false
}

/// 검색 결과를 claude 컬렉션으로 필터 + 파일 단위 dedupe.
pub fn filter_hits(hits: Vec<SearchHit>) -> Vec<SearchHit> {
    let mut seen = std::collections::HashSet::new();
    hits.into_iter()
        .filter(|h| {
            let coll = h
                .file
                .strip_prefix("qmd://")
                .and_then(|r| r.split('/').next())
                .unwrap_or("");
            CLAUDE_COLLECTIONS.contains(&coll) && seen.insert(h.file.clone())
        })
        .take(MAX_RESULTS)
        .collect()
}

/// 주입 컨텍스트 포맷 — 기존 qmd_rag.py 출력과 동일한 <qmd-context> 형태.
pub fn format_context(hits: &[SearchHit]) -> Option<String> {
    let chunks: Vec<String> = hits
        .iter()
        .filter_map(|h| {
            let snippet = h.snippet.as_deref()?;
            if snippet.is_empty() {
                return None;
            }
            let display = h.file.strip_prefix("qmd://").unwrap_or(&h.file);
            let mut header = format!("[QMD] {}", if h.title.is_empty() { display } else { &h.title });
            if let Some(ctx) = &h.context {
                header.push_str(&format!(" ({})", ctx));
            }
            Some(format!("{}\n{}", header, snippet))
        })
        .collect();
    if chunks.is_empty() {
        return None;
    }
    Some(format!(
        "<qmd-context>\nRelevant knowledge from your QMD index:\n\n{}\n</qmd-context>",
        chunks.join("\n---\n")
    ))
}

fn append_log(entry: &RagLogEntry) {
    let path = rag_log_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(line) = serde_json::to_string(entry) {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
            let _ = writeln!(f, "{}", line);
        }
    }
}

fn now_iso() -> String {
    // chrono 없이 초 단위 epoch → 로그 분석은 stats 쪽에서 처리
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", d.as_secs())
}

/// 파이프라인 실행 결과 — sim/TUI에서 재사용.
pub struct RagOutcome {
    pub gate_reason: Option<&'static str>,
    pub query: Option<String>,
    pub hits: Vec<SearchHit>,
    pub context: Option<String>,
    pub latency_ms: u64,
}

/// 프롬프트 하나에 대해 전체 파이프라인 실행 (로깅 없이).
pub fn run_pipeline(prompt: &str) -> Result<RagOutcome> {
    let started = Instant::now();
    let query = match gate(prompt) {
        Ok(q) => q,
        Err(reason) => {
            return Ok(RagOutcome {
                gate_reason: Some(reason),
                query: None,
                hits: vec![],
                context: None,
                latency_ms: started.elapsed().as_millis() as u64,
            });
        }
    };
    let cfg = config::load()?;
    let raw = bm25::search(&config::tantivy_dir(), &cfg, &query, 10, None)?;
    let hits = filter_hits(raw);
    let context = format_context(&hits);
    Ok(RagOutcome {
        gate_reason: None,
        query: Some(query),
        hits,
        context,
        latency_ms: started.elapsed().as_millis() as u64,
    })
}

/// 파이프라인 결과를 rag.jsonl에 기록 (훅/데몬 공용).
pub fn log_outcome(prompt: &str, session_id: &str, outcome: &RagOutcome) {
    append_log(&RagLogEntry {
        ts: now_iso(),
        session_id: session_id.to_string(),
        prompt: prompt.to_string(),
        prompt_len: prompt.chars().count(),
        hangul: has_hangul(prompt),
        stage: if outcome.gate_reason.is_some() { "gated" } else { "searched" }.into(),
        gate_reason: outcome.gate_reason.map(String::from),
        query: outcome.query.clone(),
        injected: outcome.context.as_ref().map(|_| outcome.hits.len()).unwrap_or(0),
        hits: outcome
            .hits
            .iter()
            .map(|h| RagHitLog {
                file: h.file.clone(),
                score: h.score,
                snippet: h.snippet.clone().unwrap_or_default(),
            })
            .collect(),
        latency_ms: outcome.latency_ms,
    });
}

/// `kmd rag --hook`: stdin JSON → stdout 컨텍스트 (+ rag.jsonl 로깅).
/// 데몬이 떠 있으면 데몬 경유(빠름), 아니면 인프로세스 폴백.
/// 훅은 절대 실패하면 안 되므로 모든 에러는 빈 출력 + exit 0.
pub fn run_hook() -> Result<()> {
    let mut raw = String::new();
    std::io::stdin().read_to_string(&mut raw)?;
    let input: HookInput = match serde_json::from_str(&raw) {
        Ok(i) => i,
        Err(_) => return Ok(()),
    };
    if input.prompt.is_empty() {
        return Ok(());
    }

    // 1) 데몬 경유 시도
    if let Some(resp) = crate::daemon::try_request(&serde_json::json!({
        "cmd": "rag",
        "prompt": input.prompt,
        "session_id": input.session_id,
    })) {
        if resp.get("ok").and_then(|v| v.as_bool()) == Some(true) {
            if let Some(ctx) = resp.get("context").and_then(|v| v.as_str()) {
                println!("{}", ctx);
            }
            return Ok(());
        }
    }

    // 2) 인프로세스 폴백
    let outcome = match run_pipeline(&input.prompt) {
        Ok(o) => o,
        Err(_) => return Ok(()), // 인덱스 없음 등 — 훅은 조용히 통과
    };
    log_outcome(&input.prompt, &input.session_id, &outcome);

    if let Some(ctx) = outcome.context {
        println!("{}", ctx);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_rejects_structural_noise() {
        assert_eq!(gate("<task-notification>\n<task-id>abc</task-id> <summary>monitor done</summary>"), Err("noise"));
        assert_eq!(gate("<command-message>clear</command-message> more text here"), Err("noise"));
        assert_eq!(gate("<system-reminder> some long injected reminder text here"), Err("noise"));
        assert_eq!(gate("Caveat: The messages below were generated by the user"), Err("noise"));
        assert_eq!(gate("https://github.com/sendbird/ops-k8s/pull/7357"), Err("noise"));
        assert_eq!(gate("❯ ccproxy: warning could not persist refreshed kiro token keychain"), Err("noise"));
        assert_eq!(gate("━━ session cf3f0edd ━━ [KO] 그리고 지금 한시적으로 안정성을"), Err("noise"));
        assert_eq!(gate("[QMD] Session: 8e79893a some pasted context block here"), Err("noise"));
    }

    #[test]
    fn gate_accepts_real_prompts() {
        assert!(gate("soda-cell-a 내 vs가 같은 네임스페이스로만 expose하고 있는데 뭔가 문제").is_ok());
        assert!(gate("왜 envoy/istio 기반으로 안만들고 rust로 따로 만들었는지 알아보자").is_ok());
        // URL이 포함돼도 뒤에 실제 질의가 있으면 통과
        assert!(gate("https://github.com/sendbird/ops-k8s/pull/7357 이 코멘트가 왜 반복되는지 봐줘").is_ok());
    }
}
