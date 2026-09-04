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

/// 프롬프트 최소 길이 — 영어 기준.
pub const MIN_PROMPT_LENGTH: usize = 20;

/// 한글 프롬프트의 최소 길이.
///
/// 같은 내용을 한국어는 영어보다 짧게 쓴다. 20자 기준은 실제 질문을 막았다:
/// "노드가 안 뜨는데 어디를 봐야 하지"(19자), "메모리 사용량 그래프를 어디서
/// 뽑지"(19자), "테라폼 레포가 어떻게 나뉘어 있지"(18자) — 완결된 질문인데
/// too_short로 차단된다. 대응하는 영어 질문은 36~50자다.
///
/// 실측: `too_short`로 차단된 한국어 프롬프트 1,599건 중 303건(고유)이
/// 15~19자 구간이고, 그중 88건이 질문형이다. 나머지는 짧은 작업 지시와
/// 이미지 참조로 검색 가치가 낮지만, 키워드 추출 후 5자 미만을 거르는
/// `query_too_short` 2차 게이트가 그것들을 다시 막는다 — 1차를 낮춰도
/// 무의미한 검색이 그대로 통과하지는 않는다.
pub const MIN_PROMPT_LENGTH_HANGUL: usize = 15;

/// 훅이 주입하는 히트 수의 기본값.
///
/// `KMD_MAX_RESULTS`로 덮을 수 있다. 이 값이 정답률의 상한을 정한다 —
/// 컬렉션 가중은 이 창 안에서 순서만 바꾸므로, 정답이 창 밖이면 어떤 가중도
/// 끌어올 수 없다. 그래서 창 크기가 가중보다 먼저 재야 할 파라미터다.
///
/// **3을 유지한다.** 5로 늘리면 gold 정답률이 61%→67%(도달 가능한 49건 기준,
/// +3건)로 오르지만 주입량이 중위 1,226B→1,872B(+53%)로 늘어난다. 주입은
/// 게이트를 통과한 **모든** 프롬프트에 붙으므로 프롬프트당 약 +430토큰이고,
/// `kmd util` 실측 채택률이 21%라 그 토큰의 대부분은 읽히지 않는다. 정답률
/// 6%p를 사기 위해 매 프롬프트에 그만큼을 상시로 내는 거래는 성립하지 않는다.
///
/// 천장은 창이 아니다: 창을 30까지 열어도 73%(도달 가능 기준)에서 멈춘다.
/// 남은 미스는 검색이 정답을 후보에조차 못 올린 경우다.
const DEFAULT_MAX_RESULTS: usize = 3;

pub fn max_results() -> usize {
    std::env::var("KMD_MAX_RESULTS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n >= 1)
        .unwrap_or(DEFAULT_MAX_RESULTS)
}
const SKIP_PREFIXES: &[&str] = &["/", "yes", "no", "ok", "sure", "thanks", "thank"];

/// 주입 대상 컬렉션. memory/rules는 CLAUDE.md로 상시 로드되므로 제외 대상 후보지만
/// 지금은 qmd_rag.py와 동일하게 유지 — stats로 실측 후 조정한다.
///
/// `KMD_INJECT_COLLECTIONS`(쉼표 구분)로 덮을 수 있다. 이 목록은 qmd에서
/// 물려받은 것이고 위 주석이 조정을 예고해 두었는데, 조정하려면 후보를
/// 넣고 빼며 재보는 수밖에 없다.
///
/// `wiki`(916건)를 넣어 봤고 **넣지 않기로 했다.** gold 전체로는 42%→47%로
/// 올라가지만, 그 이득이 전부 wiki 앵커 케이스 6건(TCP·oncall)에서 나온다.
/// 그 6건을 뺀 49건에서는 47%→45%로 **내려간다** — wiki가 노이즈로 작동한다.
/// gold의 wiki 앵커 비중(11%)이 실제 프롬프트 분포를 대표한다는 근거가 없어,
/// 순이득 3건을 근거로 채택하지 않는다. 다시 판단하려면 실사용 프롬프트에서
/// wiki 문서의 채택률을 재야 한다.
pub const CLAUDE_COLLECTIONS: &[&str] = &[
    "claude-memory",
    "claude-rules",
    "claude-agents",
    "claude-skills",
    "claude-contexts",
    "learnings",
];

/// 실제로 주입에 쓸 컬렉션 목록.
pub fn inject_collections() -> Vec<String> {
    match std::env::var("KMD_INJECT_COLLECTIONS") {
        Ok(v) if !v.trim().is_empty() => v
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect(),
        _ => CLAUDE_COLLECTIONS.iter().map(|s| s.to_string()).collect(),
    }
}

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
        .filter(|w| !STOP_WORDS.contains(w) && is_meaningful_token(w))
        .take(12)
        .collect::<Vec<_>>()
        .join(" ")
}

/// 토큰이 검색어로 쓸 만한 길이인가.
///
/// 영어는 3글자 미만을 버린다(관사·전치사 제거가 목적이다). 한글에는 그 기준을
/// 쓸 수 없다 — 한국어 내용어는 대개 2글자다. "버그 찾기 전에 먼저 확인해야 할
/// 게 있었는데"가 `> 2` 필터를 지나면 **"확인해야 있었는데"**만 남고 "버그"와
/// "찾기"가 사라진다. 정작 문서를 특정하는 단어가 버려지는 것이다.
///
/// 한글 토큰은 2글자부터 살린다. 1글자는 조사·의존명사("안", "할", "게")여서
/// 버리는 편이 맞고, tantivy 쪽 lindera 형태소 분석이 조사를 어차피 분리한다.
fn is_meaningful_token(w: &str) -> bool {
    let n = w.chars().count();
    if has_hangul(w) { n >= 2 } else { n > 2 }
}

/// 게이팅. 통과하면 검색 쿼리를 반환.
pub fn gate(prompt: &str) -> std::result::Result<String, &'static str> {
    let p = prompt.trim();
    let min_len = if has_hangul(p) {
        MIN_PROMPT_LENGTH_HANGUL
    } else {
        MIN_PROMPT_LENGTH
    };
    if p.chars().count() < min_len {
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
/// 컬렉션 가중 — 검색 점수에 곱해 재정렬한다.
///
/// `KMD_CURATED_WEIGHT`로 덮을 수 있다. 스윕 없이 값을 고르면 근거가 없고,
/// 재빌드 없이 스윕하려면 런타임 값이어야 한다.
fn collection_weight(coll: &str) -> f32 {
    match coll {
        "claude-memory" | "claude-skills" | "claude-agents" | "claude-rules"
        | "claude-contexts" => curated_weight(),
        "learnings" => 0.75,
        _ => 1.0,
    }
}

/// 정제 지식 가중의 기본값.
///
/// **측정 결과 이 값은 실효가 없다.** 1.0에서 5.0까지 스윕해도 gold 55케이스의
/// 정답률이 31%로 한 자리도 움직이지 않는다. 주입 결과 자체는 7건에서 바뀌지만
/// (learnings 하나가 skills로 교체되는 식) 그 교체가 정답 판정을 넘나들지
/// 않는다. 즉 상위 3건 안에서 순서만 섞이고, 정답이 3건 밖에 있으면 가중으로는
/// 끌어올 수 없다.
///
/// 그래도 값을 남기는 이유는 `kmd util`이 정제 지식의 값어치를 보여주기
/// 때문이다: 주입 파일의 10%인데 활용된 프롬프트의 31%에 기여한다
/// (skills 6%→15%, memory 3%→12%, rules 1%→4%). 규모 차이가 원인이고
/// (learnings 1,431건 대 정제 지식 98건), 배분을 고치려면 가중이 아니라
/// 검색 자체가 정답을 3건 안에 넣어야 한다.
///
/// `KMD_CURATED_WEIGHT`로 덮을 수 있으니, 코퍼스 균형이 달라지면 다시 스윕할
/// 것 — 지금 무효라는 것이 앞으로도 그렇다는 뜻은 아니다.
const DEFAULT_CURATED_WEIGHT: f32 = 1.3;

fn curated_weight() -> f32 {
    std::env::var("KMD_CURATED_WEIGHT")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|w| *w > 0.0)
        .unwrap_or(DEFAULT_CURATED_WEIGHT)
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
    let scope = inject_collections();
    let scope_refs: Vec<&str> = scope.iter().map(String::as_str).collect();
    #[cfg(feature = "embed")]
    if let Some(embedder) = warm {
        let store = crate::store::Store::open(&config::store_path())?;
        let hybrid = crate::embed::hybrid_with(
            &store,
            &config::tantivy_dir(),
            cfg,
            query,
            limit,
            Some(&scope_refs),
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
    bm25::search_in(&config::tantivy_dir(), cfg, query, limit, Some(&scope_refs))
}

pub fn filter_hits(hits: Vec<SearchHit>) -> Vec<SearchHit> {
    // L0 page index가 커버하는 learnings hit은 주입에서 제외 — 중복 주입 비용을
    // 줄인다. pageindex::load_sessions가 파일을 읽어야 하므로 매 호출마다 약간의
    // I/O 비용이 있지만, 훅 1회당 한 번이고 181세션 메타 파싱은 ~10ms 수준이다.
    let id_axes = crate::pageindex::load_session_axes().unwrap_or_default();
    let allowed = inject_collections();
    let mut seen = std::collections::HashSet::new();
    let mut kept: Vec<SearchHit> = hits
        .into_iter()
        .filter(|h| {
            let coll = collection_of(&h.file);
            if !allowed.iter().any(|c| c == coll) {
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
    kept.truncate(max_results());
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
///
/// 상주 임베더가 없으면 이 자리에서 로드한다. 데몬은 `run_pipeline_with`로
/// 자기 임베더를 넘기므로 이 경로를 타지 않는다.
///
/// 예전에는 `None`을 넘겨 무조건 BM25로 내려갔다. 그래서 CLI `kmd rag`와
/// `kmd eval --engine pipeline`이 hybrid를 **한 번도 실행하지 않았고**, 그
/// 상태로 잰 수치를 파이프라인 성능으로 보고했다. 실측으로 드러난 증거:
/// `kmd query`는 "터미널 화면을 설계할 때 참고할 것"에 `tui-design`을 1위로
/// 내는데 `kmd rag`는 세션 파일만 냈다.
pub fn run_pipeline(prompt: &str) -> Result<RagOutcome> {
    #[cfg(feature = "embed")]
    {
        // 게이트에 걸릴 프롬프트에 모델(220~420ms)을 로드하지 않는다.
        if gate(prompt).is_err() {
            return run_pipeline_with(prompt, None);
        }
        match crate::embed::Embedder::load() {
            Ok(e) => run_pipeline_with(prompt, Some(&e)),
            // 모델이 없는 설치는 정상 케이스다 — BM25로 내려간다.
            Err(_) => run_pipeline_with(prompt, None),
        }
    }
    #[cfg(not(feature = "embed"))]
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

    /// 한글 프롬프트는 영어보다 짧은 임계값을 쓴다.
    ///
    /// 20자 기준이 실제 질문을 막았다. 회귀하면 이 케이스들이 다시 차단되고,
    /// 훅은 조용히 아무것도 주입하지 않는다 — 에러가 아니라 침묵이라 눈에
    /// 띄지 않는다.
    #[test]
    fn the_hangul_gate_admits_short_but_complete_questions() {
        for q in [
            "노드가 안 뜨는데 어디를 봐야 하지",     // 19자
            "메모리 사용량 그래프를 어디서 뽑지",     // 19자
            "테라폼 레포가 어떻게 나뉘어 있지",       // 18자
        ] {
            assert!(
                gate(q).is_ok(),
                "{} ({}자) should pass the Hangul gate",
                q,
                q.chars().count()
            );
        }
        // 영어는 그대로 20자 기준 — 짧은 영어는 대개 응답이지 질문이 아니다.
        assert_eq!(gate("what about it").unwrap_err(), "too_short");
        // 한글이라도 15자 미만은 막힌다.
        assert_eq!(gate("이거 왜 안돼").unwrap_err(), "too_short");
    }

    /// 키워드 추출이 한국어 2글자 내용어를 버리지 않는다.
    ///
    /// `chars().count() > 2`는 영어 관사·전치사를 겨냥한 조건인데, 한국어
    /// 내용어는 대개 2글자여서 정작 문서를 특정하는 단어가 사라졌다.
    #[test]
    fn keyword_extraction_keeps_two_character_hangul_words() {
        let q = extract_keywords("버그 찾기 전에 먼저 확인해야 할 게 있었는데");
        assert!(q.contains("버그"), "핵심 명사가 살아야 한다: {}", q);
        assert!(q.contains("찾기"), "핵심 명사가 살아야 한다: {}", q);
        // 1글자는 조사·의존명사이므로 계속 버린다.
        assert!(!q.split_whitespace().any(|w| w == "할"), "{}", q);
        assert!(!q.split_whitespace().any(|w| w == "게"), "{}", q);
        // 영어 2글자는 여전히 버린다.
        let en = extract_keywords("go to the node and check it");
        assert!(!en.split_whitespace().any(|w| w == "go"), "{}", en);
        assert!(en.contains("node"), "{}", en);
    }

    /// `run_pipeline`은 임베딩을 쓸 수 있으면 반드시 쓴다.
    ///
    /// 이전 구현은 `run_pipeline_with(prompt, None)` 한 줄이어서 CLI와 eval이
    /// hybrid를 한 번도 실행하지 않았다. 컴파일은 되고 검색도 되며 에러도
    /// 없다 — BM25 결과가 그냥 나온다. 그래서 그 상태로 잰 수치를 파이프라인
    /// 성능으로 보고했다.
    ///
    /// 실제 인덱스 없이 검색 결과를 비교할 수는 없으니, 게이트에 걸리는
    /// 프롬프트에는 모델을 로드하지 않는다는 쪽을 고정한다. 그 분기가 사라지면
    /// 게이트로 버릴 프롬프트마다 220~420ms를 낸다.
    #[test]
    fn a_gated_prompt_does_not_load_the_model() {
        // 존재하지 않는 모델을 가리켜 로드가 실패하도록 만든다. 게이트에
        // 걸리는 프롬프트라면 로드를 시도하지 않으므로 결과가 같아야 한다.
        let too_short = "ok";
        let before = run_pipeline(too_short).expect("gated prompt still returns");
        assert_eq!(before.gate_reason, Some("too_short"));
        assert!(before.hits.is_empty());
        assert!(
            before.latency_ms < 200,
            "a gated prompt must not pay the model load ({}ms)",
            before.latency_ms
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
