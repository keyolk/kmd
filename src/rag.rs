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

/// `kmd rag --hook`: stdin JSON → stdout 컨텍스트 (+ rag.jsonl 로깅).
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

    let outcome = match run_pipeline(&input.prompt) {
        Ok(o) => o,
        Err(_) => return Ok(()), // 인덱스 없음 등 — 훅은 조용히 통과
    };

    append_log(&RagLogEntry {
        ts: now_iso(),
        session_id: input.session_id.clone(),
        prompt: input.prompt.clone(),
        prompt_len: input.prompt.chars().count(),
        hangul: has_hangul(&input.prompt),
        stage: if outcome.gate_reason.is_some() { "gated" } else { "searched" }.into(),
        gate_reason: outcome.gate_reason.map(String::from),
        query: outcome.query.clone(),
        injected: outcome.context.as_ref().map(|_| outcome.hits.len()).unwrap_or(0),
        hits: outcome
            .hits
            .iter()
            .map(|h| RagHitLog { file: h.file.clone(), score: h.score })
            .collect(),
        latency_ms: outcome.latency_ms,
    });

    if let Some(ctx) = outcome.context {
        println!("{}", ctx);
    }
    Ok(())
}
