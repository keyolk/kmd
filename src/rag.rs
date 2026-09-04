//! RAG 훅 파이프라인 — UserPromptSubmit stdin JSON을 받아
//! 게이팅 → 키워드 추출 → BM25 검색 → <kmd-context> 출력.
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
    "after", "above", "below", "between", "there", "here", "much", "many", "more", "most", "other",
    "like", "need", "want", "get", "make", "use", "try", "know",
];

#[derive(Debug, Deserialize)]
struct HookInput {
    #[serde(default)]
    prompt: String,
    #[serde(default)]
    session_id: String,
    #[serde(default)]
    cwd: String,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    let strip: &[char] = &[
        '?', '.', ',', '!', ':', ';', '"', '\'', '(', ')', '[', ']', '{', '}',
    ];
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
    if NOISE_PREFIXES
        .iter()
        .any(|pre| lower_start.starts_with(pre))
    {
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
    // 4) 이전에 주입된 컨텍스트를 되붙인 경우 (재귀 오염 방지).
    //    옛 <qmd-context>/[QMD] 표기도 계속 인식한다 — 과거 세션에서 복사해 온
    //    텍스트가 그 형태로 남아 있다.
    if t.starts_with("<kmd-context")
        || t.starts_with("<qmd-context")
        || t.starts_with("[KMD]")
        || t.starts_with("[QMD]")
        || t.starts_with("━━ session")
    {
        return true;
    }
    false
}

/// 컬렉션 가중치 — 정제 지식(memory/skills/agents/rules/contexts)은 답변에
/// 직접 반영될 가치가 높아 부스트하고, learnings(과거 세션 로그)는 배경 참고
/// 성격이라 약간 낮춘다. kmd show/util 실측에서 주입의 97%가 learnings로 쏠려
/// 정제 지식이 묻히는 걸 완화한다. BM25 점수에 곱해 재정렬한다.
fn collection_weight(coll: &str) -> f32 {
    match coll {
        "claude-memory" | "claude-skills" | "claude-agents" | "claude-rules"
        | "claude-contexts" => 1.3,
        "learnings" => 0.75,
        _ => 1.0,
    }
}

fn collection_of(file: &str) -> &str {
    file.strip_prefix("kmd://")
        .and_then(|r| r.split('/').next())
        .unwrap_or("")
}

/// L0 page index가 커버하는 축에 속한 learnings hit은 RAG 주입에서 skip.
///
/// `kmd page --cover` 실측: RAG가 주입한 learnings hit의 68%가 L0 축 안에
/// 있다. 이 영역은 세션 시작 시 L0 인덱스로 이미 "여기 이 축에 N세션" 형태로
/// 주입되므로, 매 프롬프트마다 다시 BM25로 밀어넣는 건 중복이자 90% 낭비의
/// 주원인이다. L0가 담당하는 영역은 RAG에서 빼서 주입 건수를 줄인다.
///
/// 미커버 32%(`**Files**:` 메타에서 repo 추출 불가 세션)는 여전히 RAG가
/// 담당한다 — page index의 근본 한계 경계를 건드리지 않는다.
///
/// `pageindex::repo_of`와 동일한 분해 규칙을 써서 주입과 측정이 같은 기준을
/// 공유하게 한다. file이 `kmd://learnings/<date>-<id8>.md` 형태일 때만
/// 검사하고, pageindex가 load한 id→축 맵에 id8이 있으면 skip.
fn is_l0_covered(
    file: &str,
    id_axes: &std::collections::HashMap<String, std::collections::HashSet<String>>,
) -> bool {
    if !file.contains("learnings") {
        return false;
    }
    let Some(stem) = file.rsplit('/').next() else {
        return false;
    };
    let stem = stem.strip_suffix(".md").unwrap_or(stem);
    let id8 = stem.rsplit('-').next().unwrap_or(stem);
    id_axes.get(id8).is_some_and(|ax| !ax.is_empty())
}

/// 검색 결과를 claude 컬렉션으로 필터 + 파일 단위 dedupe + 컬렉션 가중 재정렬.
/// 후보 검색 — 임베더가 있으면 hybrid, 없으면 BM25.
fn search_candidates(
    cfg: &config::IndexConfig,
    query: &str,
    limit: usize,
    #[cfg(feature = "embed")] warm: Option<&crate::embed::Embedder>,
    #[cfg(not(feature = "embed"))] warm: Option<&()>,
) -> Result<Vec<SearchHit>> {
    #[cfg(feature = "embed")]
    if let Some(embedder) = warm {
        let store = crate::store::Store::open(&config::store_path())?;
        let hybrid = crate::embed::hybrid_with(
            &store,
            &config::tantivy_dir(),
            cfg,
            query,
            limit,
            Some(CLAUDE_COLLECTIONS),
            Some(embedder),
        );
        // hybrid가 실패하면(임베딩 테이블 없음 등) BM25로 내려간다. 훅은
        // 검색 품질보다 응답 자체가 먼저다.
        if let Ok(hits) = hybrid {
            return Ok(hits);
        }
    }
    #[cfg(not(feature = "embed"))]
    let _ = warm;
    bm25::search_in(
        &config::tantivy_dir(),
        cfg,
        query,
        limit,
        Some(CLAUDE_COLLECTIONS),
    )
}

pub fn filter_hits(hits: Vec<SearchHit>) -> Vec<SearchHit> {
    // L0 page index가 커버하는 learnings hit은 주입에서 제외 — 중복 주입 비용을
    // 줄인다. pageindex::load_sessions가 파일을 읽어야 하므로 매 호출마다 약간의
    // I/O 비용이 있지만, 훅 1회당 한 번이고 181세션 메타 파싱은 ~10ms 수준이다.
    let id_axes = crate::pageindex::load_session_axes().unwrap_or_default();
    let mut seen = std::collections::HashSet::new();
    let mut kept: Vec<SearchHit> = hits
        .into_iter()
        .filter(|h| {
            let coll = collection_of(&h.file);
            if !CLAUDE_COLLECTIONS.contains(&coll) {
                return false;
            }
            if coll == "learnings" && is_l0_covered(&h.file, &id_axes) {
                return false;
            }
            seen.insert(h.file.clone())
        })
        .collect();
    // 컬렉션 가중을 적용한 유효 점수로 재정렬 (원 score는 표시용으로 보존).
    kept.sort_by(|a, b| {
        let wa = a.score * collection_weight(collection_of(&a.file));
        let wb = b.score * collection_weight(collection_of(&b.file));
        wb.partial_cmp(&wa).unwrap_or(std::cmp::Ordering::Equal)
    });
    kept.truncate(MAX_RESULTS);
    kept
}

/// 주입 컨텍스트 포맷 — `<kmd-context>` 블록. 항목 머리표는 `[KMD]`.
pub fn format_context(hits: &[SearchHit]) -> Option<String> {
    let chunks: Vec<String> = hits
        .iter()
        .filter_map(|h| {
            let snippet = h.snippet.as_deref()?;
            if snippet.is_empty() {
                return None;
            }
            let display = h.file.strip_prefix("kmd://").unwrap_or(&h.file);
            let mut header = format!(
                "[KMD] {}",
                if h.title.is_empty() {
                    display
                } else {
                    &h.title
                }
            );
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
        "<kmd-context>\nRelevant knowledge from your kmd index:\n\n{}\n</kmd-context>",
        chunks.join("\n---\n")
    ))
}

fn combine_contexts(activity: Option<&str>, rag: Option<&str>) -> Option<String> {
    match (activity, rag) {
        (Some(activity), Some(rag)) => Some(format!("{}\n\n{}", activity, rag)),
        (Some(activity), None) => Some(activity.to_string()),
        (None, Some(rag)) => Some(rag.to_string()),
        (None, None) => None,
    }
}

fn append_log(entry: &RagLogEntry) {
    let path = rag_log_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(line) = serde_json::to_string(entry) {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
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
    run_pipeline_with(prompt, None)
}

/// 상주 임베더를 받는 파이프라인. 데몬이 넘긴다.
///
/// 임베더가 있으면 hybrid(BM25 + 벡터 RRF)로, 없으면 BM25로 검색한다. 훅은
/// 매 프롬프트에 도는 경로이므로 hybrid로 바꾸는 근거를 남긴다:
///
/// - 품질: gold 49케이스(문서 어휘를 쓰지 않는 패러프레이즈 질의)에서 hybrid가
///   BM25를 크게 앞선다 — 전체 R@5 43% 대 33%, 한국어 33% 대 22%, MRR 0.328 대
///   0.251. 훅의 실제 프롬프트가 이 모양이다.
/// - 지연: 상주 임베더 + session 축으로 좁힌 hybrid가 276ms인데, 지금 BM25
///   파이프라인이 425ms다(같은 프롬프트, 살아 있는 데몬에 직접 측정). 벡터가
///   느리다는 통념은 모델 로드(220~420ms)를 매번 내는 CLI 경로에서 온 것이고,
///   데몬에서는 그 비용이 없다.
///
/// 임베더가 없으면(모델 미설치, embed 피처 없는 빌드) BM25로 조용히 내려간다 —
/// 훅은 절대 실패하면 안 되는 경로다.
pub fn run_pipeline_with(
    prompt: &str,
    #[cfg(feature = "embed")] warm: Option<&crate::embed::Embedder>,
    #[cfg(not(feature = "embed"))] warm: Option<&()>,
) -> Result<RagOutcome> {
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
    let timing = std::env::var("KMD_RAG_TIMING").is_ok();
    let t_gate = started.elapsed();
    let cfg = config::load()?;
    // 컬렉션 가중 재정렬이 실효를 내려면 후보를 넉넉히 가져와야 한다.
    // (정제 지식이 원 top-3 밖이어도 부스트로 올라올 여지 확보)
    //
    // 검색 범위를 CLAUDE_COLLECTIONS로 좁힌다. filter_hits가 어차피 그 밖을
    // 버리므로 결과는 같지만, 벡터 경로에서는 스캔량이 곧 지연이다 — 전
    // 임베딩(295k 청크) 대 이 범위(82k)에서 hybrid가 718ms 대 276ms였다.
    let t_cfg = started.elapsed();
    let raw = search_candidates(&cfg, &query, 30, warm)?;
    let t_search = started.elapsed();
    let hits = filter_hits(raw);
    let t_filter = started.elapsed();
    let context = format_context(&hits);
    if timing {
        eprintln!(
            "rag timing: gate {:?}, cfg {:?}, search {:?}, filter {:?}, format {:?}",
            t_gate,
            t_cfg - t_gate,
            t_search - t_cfg,
            t_filter - t_search,
            started.elapsed() - t_filter
        );
    }
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
        stage: if outcome.gate_reason.is_some() {
            "gated"
        } else {
            "searched"
        }
        .into(),
        gate_reason: outcome.gate_reason.map(String::from),
        query: outcome.query.clone(),
        injected: outcome
            .context
            .as_ref()
            .map(|_| outcome.hits.len())
            .unwrap_or(0),
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

    // Live activity is independent of BM25: even gated prompts should see recent
    // work from other sessions, while the current session is always excluded.
    let activity = crate::activity::render_recent(&input.session_id, &input.cwd, 6)
        .ok()
        .flatten();

    // 1) 데몬 경유 시도
    if let Some(resp) = crate::daemon::try_request(&serde_json::json!({
        "cmd": "rag",
        "prompt": input.prompt,
        "session_id": input.session_id,
    })) {
        if resp.get("ok").and_then(|v| v.as_bool()) == Some(true) {
            let context = resp.get("context").and_then(|v| v.as_str());
            if let Some(combined) = combine_contexts(activity.as_deref(), context) {
                println!("{}", combined);
            }
            return Ok(());
        }
    }

    // 2) 인프로세스 폴백
    let outcome = match run_pipeline(&input.prompt) {
        Ok(o) => o,
        Err(_) => {
            if let Some(activity) = activity {
                println!("{}", activity);
            }
            return Ok(());
        }
    };
    log_outcome(&input.prompt, &input.session_id, &outcome);

    if let Some(combined) = combine_contexts(activity.as_deref(), outcome.context.as_deref()) {
        println!("{}", combined);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_rejects_structural_noise() {
        assert_eq!(
            gate("<task-notification>\n<task-id>abc</task-id> <summary>monitor done</summary>"),
            Err("noise")
        );
        assert_eq!(
            gate("<command-message>clear</command-message> more text here"),
            Err("noise")
        );
        assert_eq!(
            gate("<system-reminder> some long injected reminder text here"),
            Err("noise")
        );
        assert_eq!(
            gate("Caveat: The messages below were generated by the user"),
            Err("noise")
        );
        assert_eq!(
            gate("https://github.com/sendbird/ops-k8s/pull/7357"),
            Err("noise")
        );
        assert_eq!(
            gate("❯ ccproxy: warning could not persist refreshed kiro token keychain"),
            Err("noise")
        );
        assert_eq!(
            gate("━━ session cf3f0edd ━━ [KO] 그리고 지금 한시적으로 안정성을"),
            Err("noise")
        );
        assert_eq!(
            gate("[KMD] Session: 8e79893a some pasted context block here"),
            Err("noise")
        );
        // 옛 표기도 계속 걸러야 한다 — 과거 세션에서 복사해 온 텍스트.
        assert_eq!(
            gate("[QMD] Session: 8e79893a some pasted context block here"),
            Err("noise")
        );
        assert_eq!(
            gate("<qmd-context>\nRelevant knowledge from your QMD index:"),
            Err("noise")
        );
        assert_eq!(
            gate("<kmd-context>\nRelevant knowledge from your kmd index:"),
            Err("noise")
        );
    }

    #[test]
    fn gate_accepts_real_prompts() {
        assert!(
            gate("soda-cell-a 내 vs가 같은 네임스페이스로만 expose하고 있는데 뭔가 문제").is_ok()
        );
        assert!(gate("왜 envoy/istio 기반으로 안만들고 rust로 따로 만들었는지 알아보자").is_ok());
        // URL이 포함돼도 뒤에 실제 질의가 있으면 통과
        assert!(
            gate("https://github.com/sendbird/ops-k8s/pull/7357 이 코멘트가 왜 반복되는지 봐줘")
                .is_ok()
        );
    }

    fn hit(file: &str, score: f32) -> SearchHit {
        SearchHit {
            docid: "#0".into(),
            score,
            file: file.into(),
            abspath: None,
            title: String::new(),
            context: None,
            snippet: Some("x".into()),
        }
    }

    #[test]
    fn filter_hits_boosts_curated_over_learnings() {
        // learnings가 원 BM25 점수는 약간 높아도, 정제 지식 부스트로 앞서야 한다.
        // learnings 40*0.75=30 vs skills 34*1.3=44.2 → skills 우선.
        let raw = vec![
            hit("kmd://learnings/20260101 0-aaa.md", 40.0),
            hit("kmd://claude-skills/sb:jira-ticket/SKILL.md", 34.0),
            hit("kmd://learnings/20260102 0-bbb.md", 38.0),
        ];
        let out = filter_hits(raw);
        assert_eq!(collection_of(&out[0].file), "claude-skills");
    }

    #[test]
    fn filter_hits_keeps_learnings_when_dominant() {
        // 정제 지식 후보가 없으면 learnings가 그대로 남는다.
        let raw = vec![
            hit("kmd://learnings/a.md", 40.0),
            hit("kmd://learnings/b.md", 30.0),
        ];
        let out = filter_hits(raw);
        assert_eq!(out.len(), 2);
        assert_eq!(collection_of(&out[0].file), "learnings");
    }
}
