//! Page index (L0/L1) — RAG push 주입의 대안.
//!
//! 현재 RAG는 프롬프트마다 BM25 상위 3건을 강제 주입하는데, 실측 채택률이 10%
//! (`kmd util`)다. 10건 중 9건은 컨텍스트만 오염시키고 버려진다. 게다가 매
//! 프롬프트에 검색 비용을 지불한다(p95 653ms).
//!
//! page index는 판단 주체를 뒤집는다. "이게 관련 있을 것"이라고 BM25가 추측해서
//! 밀어넣는 대신, "여기 이런 지형이 있다"만 한 번 깔아두고 에이전트가 필요할 때
//! 직접 진입하게 한다.
//!
//!   L0 (상시, ~300 tokens)  repo 축별 세션 수 + 기간. 진입 방법 한 줄.
//!   L1 (요청 시)            해당 축의 세션 목록 — 날짜, id, 한 줄 요약.
//!   L2                      기존 Read/kmd search — 본문.
//!
//! 입력은 kmd-learnings의 이미 구조화된 메타데이터(`Date:`, `**Files**:`,
//! `**Tools**:`)라서 별도 파싱이 거의 필요 없다. 278개 파일 중 `Date:` 277,
//! `**Files**:` 271로 사실상 전량 구조화돼 있다.

use anyhow::Result;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::OnceLock;
use walkdir::WalkDir;

/// 한 세션의 L0/L1 렌더링에 필요한 최소 정보.
#[derive(Debug, Clone, Serialize)]
pub struct SessionEntry {
    /// 세션 id 앞 8자 — 파일명에서 추출, L1에서 진입 키로 쓴다.
    pub id8: String,
    /// YYYYMMDD
    pub date: String,
    /// HH:MM when available.
    pub time: String,
    pub file: String,
    /// 작업 축(repo/설정파일). 한 세션이 여러 축에 속할 수 있다.
    pub repos: BTreeSet<String>,
    /// Files recorded in the learning metadata.
    pub files: Vec<String>,
    /// Working directory recorded at extraction time, or inferred from old files.
    pub cwd: String,
    /// 세션 의도 — 보일러플레이트 아닌 첫 user 프롬프트.
    pub summary: String,
    /// 가장 최근의 구체적인 user 프롬프트 — journal의 현재 작업 표시용.
    pub latest_summary: String,
    /// dedup 시 "더 진행된 스냅샷" 판정용.
    pub bytes: usize,
}

/// 하나의 L0 축.
#[derive(Debug, Serialize)]
pub struct Axis {
    pub name: String,
    pub sessions: usize,
    pub first: String,
    pub last: String,
}

#[derive(Debug, Serialize)]
pub struct Index {
    pub axes: Vec<Axis>,
    /// 세션이 1건뿐인 축 — L0에 나열하지 않고 이름만 모아둔다.
    pub singles: Vec<String>,
    pub total_sessions: usize,
}

fn learnings_dir() -> PathBuf {
    let h = std::env::var("HOME").expect("HOME");
    PathBuf::from(h).join(".claude/kmd-learnings")
}

/// 부수적으로 touch되는 경로 — 작업 축이 아니다. 이걸 안 걸러내면
/// `.claude/projects`(전 세션이 transcript를 읽음)가 최상위 축으로 올라와
/// L0이 전부 노이즈가 된다.
const NOISE_SEGMENTS: &[&str] = &[
    "projects",
    "image-cache",
    ".credentials.json",
    ".claude.json",
    "sessions",
    "tasks",
    ".ccx-cache.gob",
    "plans",
    "todos",
    "statsig",
    "shell-snapshots",
    "snapshots",
    "kmd-learnings",
    ".kmd-processed-snapshots",
    "history.jsonl",
];

/// vendored/빌드 캐시 — 내가 작업한 게 아니다.
const NOISE_PREFIXES: &[&str] = &[
    "index.crates.io",
    "registry",
    "node_modules",
    ".cargo",
    "target",
    ".worktrees",
];

/// 세션 의도가 아닌 프롬프트. 훅 주입물·재개 신호·이미지 첨부 등은
/// 요약으로 쓰면 L1이 읽을 수 없게 된다.
const BOILERPLATE: &[&str] = &[
    "[Request interrupted",
    "Implement the following plan",
    "<local-command-caveat>",
    "<local-command-stdout>",
    "<task-notification>",
    "<command-name>",
    "<command-message>",
    "<system-reminder>",
    "<teammate-message",
    "Caveat:",
    "[Image",
    "Base directory for this skill",
    "Tool loaded.",
    "This session is being continued",
    "A session-scoped Stop hook is now active",
    "Stop hook feedback:",
    "[Your previous response had no visible output",
    "(Re-invocation of",
    "/compact",
    "continue",
    "계속",
];

fn is_noise_repo(seg: &str) -> bool {
    NOISE_SEGMENTS.contains(&seg) || NOISE_PREFIXES.iter().any(|p| seg.starts_with(p))
}

/// 파일 경로 → 작업 축. `~/src/[sendbird/]<repo>` 와 `~/.claude/<file>` 두 형태를
/// 인식한다. 이 환경의 실제 레이아웃에 맞춘 휴리스틱.
pub fn repo_of(path: &str) -> Option<String> {
    if let Some(rest) = path.split("/src/").nth(1) {
        let mut segs = rest.split('/').filter(|s| !s.is_empty());
        let first = segs.next()?;
        // sendbird org 하위는 한 단계 더 들어간다
        let repo = if first == "sendbird" {
            segs.next()?
        } else {
            first
        };
        if is_noise_repo(repo) {
            return None;
        }
        return Some(repo.to_string());
    }
    if let Some(rest) = path.split(".claude/").nth(1) {
        let seg = rest.split('/').next()?;
        if seg.is_empty() || is_noise_repo(seg) {
            return None;
        }
        return Some(format!("~/.claude/{}", seg));
    }
    None
}

fn is_boilerplate(s: &str) -> bool {
    if s.chars().count() < 8 {
        return true;
    }
    BOILERPLATE.iter().any(|b| s.starts_with(b))
}

/// URL만 있는 프롬프트는 그대로 두면 L1에서 아무 정보가 없다. 의미 있는
/// 식별자로 축약한다.
fn condense(s: &str) -> String {
    let s = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if s.starts_with("http") && !s.contains(' ') {
        if let Some(rest) = s.split("github.com/").nth(1) {
            let segs: Vec<&str> = rest.split('/').collect();
            if segs.len() >= 4 && segs[2] == "pull" {
                return format!("PR {}#{}", segs[1], segs[3]);
            }
        }
        if let Some(rest) = s.split("slack.com/archives/").nth(1) {
            if let Some(ch) = rest.split('/').next() {
                return format!("Slack thread {}", ch);
            }
        }
    }
    s
}

fn line_after<'a>(raw: &'a str, prefix: &str) -> Option<&'a str> {
    raw.lines()
        .find(|l| l.starts_with(prefix))
        .map(|l| l[prefix.len()..].trim())
}

/// 백틱으로 감싼 경로들을 추출 — `**Files**:` 라인의 포맷.
fn backticked(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = s;
    while let Some(start) = rest.find('`') {
        rest = &rest[start + 1..];
        if let Some(end) = rest.find('`') {
            out.push(&rest[..end]);
            rest = &rest[end + 1..];
        } else {
            break;
        }
    }
    out
}

fn transcript_cwds() -> &'static HashMap<String, String> {
    static CWDS: OnceLock<HashMap<String, String>> = OnceLock::new();
    CWDS.get_or_init(|| {
        let Ok(home) = std::env::var("HOME") else {
            return HashMap::new();
        };
        let root = PathBuf::from(home).join(".claude/projects");
        let mut by_session = HashMap::new();
        for entry in WalkDir::new(root)
            .min_depth(2)
            .max_depth(2)
            .into_iter()
            .filter_map(Result::ok)
        {
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) != Some("jsonl") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|value| value.to_str()) else {
                continue;
            };
            let id8: String = stem.chars().take(8).collect();
            if id8.len() != 8 {
                continue;
            }
            let Ok(file) = fs::File::open(path) else {
                continue;
            };
            let cwd = BufReader::new(file)
                .lines()
                .map_while(Result::ok)
                .take(100)
                .filter_map(|line| serde_json::from_str::<serde_json::Value>(&line).ok())
                .filter_map(|entry| {
                    entry
                        .get("cwd")
                        .and_then(serde_json::Value::as_str)
                        .filter(|cwd| !cwd.is_empty())
                        .map(str::to_string)
                })
                .next();
            if let Some(cwd) = cwd {
                by_session.insert(id8, cwd);
            }
        }
        by_session
    })
}

fn transcript_cwd(id8: &str) -> Option<String> {
    transcript_cwds().get(id8).cloned()
}

fn parse_one(path: &PathBuf) -> Option<SessionEntry> {
    let raw = fs::read_to_string(path).ok()?;
    let file = path.file_name()?.to_string_lossy().to_string();

    // 파일명 규약: <YYYYMMDD>[ N]-<id8>.md
    let id8 = file
        .rsplit_once('-')
        .and_then(|(_, tail)| tail.strip_suffix(".md"))
        .unwrap_or(&file)
        .to_string();

    let timestamp = line_after(&raw, "Date: ").unwrap_or_default();
    let digits: String = timestamp.chars().filter(char::is_ascii_digit).collect();
    let date = digits.chars().take(8).collect::<String>();
    let time = if digits.len() >= 12 {
        format!("{}:{}", &digits[8..10], &digits[10..12])
    } else {
        String::new()
    };

    let mut repos = BTreeSet::new();
    let mut files = Vec::new();
    if let Some(files_line) = line_after(&raw, "**Files**: ") {
        for path in backticked(files_line) {
            files.push(path.to_string());
            if let Some(repo) = repo_of(path) {
                repos.insert(repo);
            }
        }
    }

    let cwd = line_after(&raw, "Cwd: ")
        .map(|value| value.trim_matches('`').to_string())
        .filter(|value| !value.is_empty())
        .or_else(|| transcript_cwd(&id8))
        .or_else(|| crate::locality::infer_cwd(&files))
        .unwrap_or_default();
    if let Some(repo) = crate::locality::resolve(&cwd).repo {
        repos.insert(repo);
    }

    let summaries: Vec<String> = raw
        .lines()
        .filter_map(|line| line.strip_prefix("**User**: "))
        .map(str::trim)
        .filter(|content| !is_boilerplate(content))
        .map(condense)
        .collect();
    let summary = summaries.first().cloned().unwrap_or_default();
    let latest_summary = summaries.last().cloned().unwrap_or_default();

    Some(SessionEntry {
        id8,
        date,
        time,
        file,
        repos,
        files,
        cwd,
        summary,
        latest_summary,
        bytes: raw.len(),
    })
}

fn load_learning_files() -> Result<Vec<SessionEntry>> {
    let dir = learnings_dir();
    let Ok(entries) = fs::read_dir(&dir) else {
        return Ok(Vec::new());
    };
    Ok(entries
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            (path.extension().and_then(|value| value.to_str()) == Some("md"))
                .then(|| parse_one(&path))
                .flatten()
        })
        .collect())
}

/// 모든 learning 파일을 날짜별 이력을 보존해 읽는다. 같은 세션·같은 날짜의
/// 중간 snapshot만 가장 큰 파일로 dedup한다.
pub fn load_learning_history() -> Result<Vec<SessionEntry>> {
    let mut by_session_date: BTreeMap<(String, String), SessionEntry> = BTreeMap::new();
    for entry in load_learning_files()? {
        let key = (entry.id8.clone(), entry.date.clone());
        by_session_date
            .entry(key)
            .and_modify(|current| {
                if entry.bytes > current.bytes {
                    *current = entry.clone();
                }
            })
            .or_insert(entry);
    }
    Ok(by_session_date.into_values().collect())
}

/// learnings 전체를 세션 단위로 dedup한다.
///
/// 한 세션이 여러 날짜와 snapshot으로 추출된 경우 가장 큰 파일 = 가장 진행된
/// snapshot을 남긴다. 날짜별 이력이 필요하면 `load_learning_history`를 사용한다.
pub fn load_sessions() -> Result<Vec<SessionEntry>> {
    let mut by_id: BTreeMap<String, SessionEntry> = BTreeMap::new();
    for entry in load_learning_history()? {
        by_id
            .entry(entry.id8.clone())
            .and_modify(|current| {
                if entry.bytes > current.bytes {
                    *current = entry.clone();
                }
            })
            .or_insert(entry);
    }
    Ok(by_id.into_values().collect())
}

/// id8 → 축 집합. `rag::filter_hits`가 L0 커버 hit을 skip할 때 쓴다.
/// load_sessions와 동일 소스에서 파생하므로 주입과 측정이 같은 기준을 공유한다.
pub fn load_session_axes() -> Result<HashMap<String, HashSet<String>>> {
    let sessions = load_sessions()?;
    let mut map = HashMap::with_capacity(sessions.len());
    for s in sessions {
        if !s.repos.is_empty() {
            map.insert(s.id8, s.repos.into_iter().collect());
        }
    }
    Ok(map)
}

fn fmt_month(d: &str) -> String {
    if d.len() >= 6 {
        format!("{}-{}", &d[..4], &d[4..6])
    } else {
        d.to_string()
    }
}

pub fn build() -> Result<Index> {
    let sessions = load_sessions()?;
    let total = sessions.len();

    let mut by_axis: BTreeMap<String, Vec<&SessionEntry>> = BTreeMap::new();
    for s in &sessions {
        for r in &s.repos {
            by_axis.entry(r.clone()).or_default().push(s);
        }
    }

    let mut axes = Vec::new();
    let mut singles = Vec::new();
    for (name, ss) in by_axis {
        if ss.len() < 2 {
            singles.push(name);
            continue;
        }
        let mut dates: Vec<&str> = ss
            .iter()
            .map(|s| s.date.as_str())
            .filter(|d| !d.is_empty())
            .collect();
        dates.sort_unstable();
        axes.push(Axis {
            name,
            sessions: ss.len(),
            first: dates.first().map(|d| fmt_month(d)).unwrap_or_default(),
            last: dates.last().map(|d| fmt_month(d)).unwrap_or_default(),
        });
    }
    // 세션 많은 축 우선 — L0에서 위쪽이 눈에 먼저 들어온다.
    axes.sort_by(|a, b| b.sessions.cmp(&a.sessions).then(a.name.cmp(&b.name)));

    Ok(Index {
        axes,
        singles,
        total_sessions: total,
    })
}

/// L0가 RAG 주입의 얼마를 커버하는가 — `kmd page --cover`.
///
/// 핵심 질문: RAG가 주입했던 learnings hit 중 L0가 나열하는 축 안에 있는 건
/// 몇 %인가. snippet 보유 hit(= 유효 후보)를 따로 세어 "RAG가 잡은 유효
/// 케이스를 L0가 놓치는 비율"을 검증한다.
///
/// 검증 단위는 "hit"이다. RAG가 주입 1회에 3 hit을 밀어넣을 수 있고, 그
/// 각각이 별개의 의미를 가지므로 건수가 채택률보다 직접적이다.
///
/// `kmd page` 와 동일한 `repo_of` 로직으로 id8 → 축을 뽑아 rag.jsonl의
/// `hits[].file` (`kmd://learnings/<date>-<id8>.md`)과 조인한다. pageindex
/// 자체의 로직을 쓰기 때문에 "측정과 주입이 같은 분해 규칙을 공유"한다 —
/// 이것이 이 측정을 검증 가능하게 만드는 조건이다.
pub fn cover() -> Result<()> {
    use crate::rag::{RagLogEntry, rag_log_path};

    // id8 → 축 집합. pageindex가 L0를 만들 때 쓰는 것과 같은 load_sessions.
    let sessions = load_sessions()?;
    let mut id_axes: HashMap<String, HashSet<String>> = HashMap::new();
    let mut all_axes: HashSet<String> = HashSet::new();
    for s in &sessions {
        if !s.repos.is_empty() {
            id_axes.insert(s.id8.clone(), s.repos.iter().cloned().collect());
            for r in &s.repos {
                all_axes.insert(r.clone());
            }
        }
    }

    let raw = match fs::read_to_string(rag_log_path()) {
        Ok(r) => r,
        Err(_) => {
            println!("no rag.jsonl — 커버리지를 측정하려면 RAG 훅 기록이 필요합니다");
            return Ok(());
        }
    };

    let mut total = 0usize;
    let mut with_snip = 0usize;
    let mut covered = 0usize;
    let mut covered_snip = 0usize;
    let mut no_axis = 0usize;
    let mut no_axis_snip = 0usize;

    for line in raw.lines() {
        let Ok(e) = serde_json::from_str::<RagLogEntry>(line) else {
            continue;
        };
        if e.injected == 0 {
            continue;
        }
        for h in &e.hits {
            // kmd://learnings/<file>.md 또는 qmd://learnings/<file>.md
            if !h.file.contains("learnings") {
                continue;
            }
            total += 1;
            let has_snip = !h.snippet.is_empty();
            if has_snip {
                with_snip += 1;
            }
            let stem = h
                .file
                .rsplit('/')
                .next()
                .unwrap_or(&h.file)
                .strip_suffix(".md")
                .unwrap_or(&h.file);
            let id8 = stem.rsplit('-').next().unwrap_or(stem).to_string();
            let axes = id_axes.get(&id8);
            match axes {
                Some(ax) if !ax.is_empty() && ax.iter().any(|a| all_axes.contains(a)) => {
                    covered += 1;
                    if has_snip {
                        covered_snip += 1;
                    }
                }
                _ => {
                    no_axis += 1;
                    if has_snip {
                        no_axis_snip += 1;
                    }
                }
            }
        }
    }

    let pct = |n: usize, d: usize| {
        if d == 0 {
            0.0
        } else {
            100.0 * n as f64 / d as f64
        }
    };
    println!("kmd page --cover — RAG 주입의 L0 커버리지\n");
    println!(
        "learnings 주입 hit: {} (snippet 보유 {})\n",
        total, with_snip
    );
    println!("=== 전체 learnings hit ===");
    println!(
        "  L0 축 커버:    {:>5}  ({:.0}%)",
        covered,
        pct(covered, total)
    );
    println!(
        "  축 없음(meta): {:>5}  ({:.0}%)",
        no_axis,
        pct(no_axis, total)
    );
    println!("\n=== snippet 보유 (유효 후보) ===");
    println!(
        "  L0 축 커버:    {:>5}  ({:.0}%)",
        covered_snip,
        pct(covered_snip, with_snip)
    );
    println!(
        "  축 없음:        {:>5}  ({:.0}%)",
        no_axis_snip,
        pct(no_axis_snip, with_snip)
    );
    println!(
        "\n주: '축 없음'은 learnings 메타의 **Files** 에서 repo를 추출할 수 없는 세션 —\n\
           L0로 커버 불가능한 RAG 영역이며, page index의 근본적 한계 경계다."
    );
    Ok(())
}

/// L0 렌더 — 세션 시작 시 주입할 상시 컨텍스트.
///
/// 목표는 "무엇이 있는지"와 "어떻게 들어가는지"만 알려주는 것. 본문은 한 줄도
/// 넣지 않는다. 그게 RAG와의 차이다.
pub fn render_l0(idx: &Index) -> String {
    let mut out = String::new();
    out.push_str("<session-index>\n");
    out.push_str(&format!(
        "과거 작업 세션 {} 건이 주제별로 인덱싱되어 있습니다.\n\
         특정 주제의 과거 맥락이 필요하면 `kmd page <axis>` 로 세션 목록을 확인하세요.\n\n",
        idx.total_sessions
    ));
    for a in &idx.axes {
        out.push_str(&format!(
            "  {} — {} sessions ({}~{})\n",
            a.name, a.sessions, a.first, a.last
        ));
    }
    if !idx.singles.is_empty() {
        let names: Vec<&str> = idx.singles.iter().map(String::as_str).collect();
        out.push_str(&format!("  1회성: {}\n", names.join(", ")));
    }
    out.push_str("</session-index>");
    out
}

/// L1 렌더 — 특정 축의 세션 목록. 여기까지도 본문은 없다.
pub fn render_l1(axis: &str, limit: usize) -> Result<String> {
    let sessions = load_sessions()?;
    let mut matched: Vec<&SessionEntry> = sessions
        .iter()
        .filter(|s| s.repos.iter().any(|r| r == axis || r.contains(axis)))
        .collect();
    if matched.is_empty() {
        return Ok(format!(
            "no sessions for axis '{}' — `kmd page` 로 축 목록을 확인하세요",
            axis
        ));
    }
    matched.sort_by(|a, b| b.date.cmp(&a.date));

    let shown = matched.len().min(limit);
    let mut out = format!("[{}] {} sessions\n", axis, matched.len());
    for s in matched.iter().take(limit) {
        let summary: String = s.summary.chars().take(90).collect();
        let summary = if summary.is_empty() {
            "(요약 없음)".to_string()
        } else {
            summary
        };
        out.push_str(&format!("  {} {} — {}\n", s.date, s.id8, summary));
    }
    if matched.len() > shown {
        out.push_str(&format!(
            "  … {} more (-n 으로 확장, 본문은 kmd search 또는 Read)\n",
            matched.len() - shown
        ));
    }
    out.push_str(&format!(
        "\n본문: ~/.claude/kmd-learnings/<date>*-<id8>.md\n"
    ));
    Ok(out)
}

/// `kmd page [axis]` CLI.
pub fn run(axis: Option<&str>, limit: usize, json: bool, l0: bool, cover: bool) -> Result<()> {
    if cover {
        return self::cover();
    }
    match axis {
        Some(a) => {
            if json {
                let sessions = load_sessions()?;
                let matched: Vec<&SessionEntry> = sessions
                    .iter()
                    .filter(|s| s.repos.iter().any(|r| r == a || r.contains(a)))
                    .collect();
                println!("{}", serde_json::to_string_pretty(&matched)?);
            } else {
                print!("{}", render_l1(a, limit)?);
            }
        }
        None => {
            let idx = build()?;
            if json {
                println!("{}", serde_json::to_string_pretty(&idx)?);
            } else {
                let rendered = render_l0(&idx);
                println!("{}", rendered);
                if !l0 {
                    eprintln!(
                        "\n— L0: {} chars ≈ {} tokens, {} axes, {} sessions",
                        rendered.len(),
                        rendered.len() / 3,
                        idx.axes.len(),
                        idx.total_sessions
                    );
                }
            }
        }
    }
    Ok(())
}

/// SessionStart 훅용 — `<session-index>` 블록을 `additionalContext` JSON으로.
///
/// 훅 규약: stdout에 `{"hookSpecificOutput":{"additionalContext":"..."}}`을 내면
/// Claude Code가 세션 시작 컨텍스트에 주입한다. `kmd page --l0`와 동일한
/// 렌더를 쓰되 훅 JSON 래핑만 더한다. 실패해도 조용히 빈 JSON을 내놓아
/// 세션 시작을 차단하지 않는다.
pub fn run_hook() -> Result<()> {
    let out = match build() {
        Ok(idx) => render_l0(&idx),
        Err(_) => String::new(),
    };
    let payload = serde_json::json!({
        "hookSpecificOutput": { "additionalContext": out }
    });
    println!("{}", serde_json::to_string(&payload)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_of_extracts_sendbird_org_repos() {
        assert_eq!(
            repo_of("/Users/x/src/sendbird/platform-tools/service-catalog/main.go").as_deref(),
            Some("platform-tools")
        );
    }

    #[test]
    fn repo_of_extracts_plain_src_repos() {
        assert_eq!(
            repo_of("/Users/x/src/keyolk/ccx/internal/tui/app.go").as_deref(),
            Some("keyolk")
        );
    }

    #[test]
    fn repo_of_rejects_incidental_claude_paths() {
        // 전 세션이 transcript를 읽으므로 축으로 잡으면 L0이 무의미해진다
        assert_eq!(repo_of("/Users/x/.claude/projects/foo/bar.jsonl"), None);
        assert_eq!(repo_of("/Users/x/.claude/image-cache/a.png"), None);
    }

    #[test]
    fn repo_of_keeps_real_claude_config_axes() {
        assert_eq!(
            repo_of("/Users/x/.claude/hooks/qmd_rag.py").as_deref(),
            Some("~/.claude/hooks")
        );
    }

    #[test]
    fn repo_of_rejects_vendored_repo_segments() {
        // repo 세그먼트 자체가 캐시인 경우만 거른다. src/<repo>/target/... 은
        // 여전히 <repo> 작업으로 본다 — 빌드 산출물도 그 repo의 것이다.
        assert_eq!(
            repo_of("/Users/x/src/index.crates.io-1949cf8c/tantivy-0.25/lib.rs"),
            None
        );
        assert_eq!(
            repo_of("/Users/x/src/keyolk/target/debug/build/x.rs").as_deref(),
            Some("keyolk")
        );
    }

    #[test]
    fn condense_shortens_bare_pr_urls() {
        assert_eq!(
            condense("https://github.com/sendbird/platform-tools/pull/4327"),
            "PR platform-tools#4327"
        );
    }

    #[test]
    fn condense_keeps_prose_prompts() {
        let p = "ccproxy 개선점이 없을지 생각해보자";
        assert_eq!(condense(p), p);
    }

    #[test]
    fn boilerplate_rejects_hook_injected_prompts() {
        assert!(is_boilerplate("<local-command-caveat>Caveat: ..."));
        assert!(is_boilerplate("[Request interrupted by user]"));
        assert!(is_boilerplate("계속"));
        assert!(!is_boilerplate(
            "현재 포탈 구조를 보고 리팩토링을 하고자 하는데"
        ));
    }

    #[test]
    fn boilerplate_rejects_session_resume_artifacts() {
        // 실측에서 L1 요약 자리를 차지했던 것들 — 세션 의도가 아니다
        assert!(is_boilerplate("Tool loaded."));
        assert!(is_boilerplate(
            "This session is being continued from a previous conversation"
        ));
        assert!(is_boilerplate(
            "(Re-invocation of /sb:pr-followup — the skill instructions"
        ));
        assert!(is_boilerplate(
            "A session-scoped Stop hook is now active with condition: deploy"
        ));
        assert!(is_boilerplate("/compact"));
    }

    #[test]
    fn backticked_extracts_file_paths() {
        let line = "`/a/b.go`, `/c/d.ts`";
        assert_eq!(backticked(line), vec!["/a/b.go", "/c/d.ts"]);
    }
}
