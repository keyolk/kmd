//! Operations dashboard for kmd runtime state, activity, journal, RAG, and checks.

use crate::activity::SessionActivity;
use anyhow::{Context, Result};
use chrono::{Local, Utc};
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Tabs, Wrap};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const REFRESH_INTERVAL: Duration = Duration::from_secs(5);
const CHECK_HISTORY_LIMIT: usize = 500;
const TABS: &[&str] = &["Sessions", "Operations", "Simulator"];
const SESSIONS_TAB: usize = 0;
const OPERATIONS_TAB: usize = 1;
const SIMULATOR_TAB: usize = 2;

struct TabGuide {
    purpose: &'static str,
    source: &'static str,
    action: &'static str,
}

const TAB_GUIDES: &[TabGuide] = &[
    TabGuide {
        purpose: "See which Claude sessions are querying kmd right now and what each one got.",
        source: "rag.jsonl query recency (active/recent/idle) plus live activity cards",
        action: "j/k selects a session · PgUp/PgDn scrolls its activity and query history",
    },
    TabGuide {
        purpose: "Monitor runtime, retrieval quality, evaluations, and self-checks together.",
        source: "daemon, hooks, store, rag.jsonl, evaluations, and check history",
        action: "r refreshes · t runs the complete self-check suite",
    },
    TabGuide {
        purpose: "Run real queries against the local index without writing logs.",
        source: "RAG pipeline · raw BM25 · Global across session/knowledge/project",
        action: "i types · Enter runs · m switches mode · ? lists every key",
    },
];

#[derive(Debug, Clone, Serialize)]
pub struct CollectionStatus {
    pub name: String,
    pub documents: i64,
    pub path: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeStatus {
    pub daemon_online: bool,
    pub daemon_socket: String,
    pub hooks_installed: usize,
    pub hooks_total: usize,
    pub settings_path: String,
    pub documents: i64,
    pub dirty_documents: i64,
    pub collections: Vec<CollectionStatus>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RagQueryRecord {
    pub ts: String,
    pub query: String,
    pub injected: usize,
    pub hits: Vec<crate::rag::RagHitLog>,
    pub latency_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct RagSession {
    pub session_id: String,
    pub latest_ts: String,
    pub queries: Vec<RagQueryRecord>,
}

#[derive(Debug, Serialize)]
pub struct RagSummary {
    pub total: usize,
    pub searched: usize,
    pub injected: usize,
    pub gated: usize,
    pub injection_rate_pct: f64,
    pub median_latency_ms: u64,
    pub p95_latency_ms: u64,
    pub sessions: Vec<RagSession>,
    pub recent: Vec<crate::rag::RagLogEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckItem {
    pub name: String,
    pub ok: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckRun {
    pub timestamp: String,
    pub duration_ms: u64,
    pub passed: usize,
    pub total: usize,
    pub checks: Vec<CheckItem>,
}

#[derive(Debug, Serialize)]
pub struct DashboardSnapshot {
    pub generated_at: String,
    pub cwd: String,
    pub runtime: RuntimeStatus,
    /// Live Claude sessions observed in the last 24h. Kept beside the RAG log so the
    /// Sessions view can also surface sessions that ran without any kmd retrieval.
    pub activity: Vec<crate::activity::ActivityCard>,
    pub journal: crate::journal::JournalView,
    pub rag: RagSummary,
    pub evaluations: Vec<crate::evaluation_log::EvaluationRun>,
    pub checks: Vec<CheckRun>,
    pub errors: Vec<String>,
}

fn checks_path() -> PathBuf {
    crate::rag::state_dir().join("checks.jsonl")
}

fn pct(n: usize, d: usize) -> f64 {
    if d == 0 {
        0.0
    } else {
        100.0 * n as f64 / d as f64
    }
}

fn runtime_status(errors: &mut Vec<String>) -> RuntimeStatus {
    let daemon_socket = crate::daemon::socket_path();
    let daemon_online = crate::daemon::try_request(&serde_json::json!({"cmd":"ping"}))
        .and_then(|value| value.get("ok").and_then(serde_json::Value::as_bool))
        .unwrap_or(false);

    let (hooks_installed, hooks_total, settings_path) = match crate::hook_config::status_summary() {
        Ok(summary) => summary,
        Err(error) => {
            errors.push(format!("hooks: {error}"));
            (0, 5, PathBuf::new())
        }
    };

    let mut documents = 0;
    let mut dirty_documents = 0;
    let mut collections = Vec::new();
    match crate::config::load() {
        Ok(config) => match crate::store::Store::open(&crate::config::store_path()) {
            Ok(store) => {
                match store.counts() {
                    Ok((total, dirty)) => {
                        documents = total;
                        dirty_documents = dirty;
                    }
                    Err(error) => errors.push(format!("store counts: {error}")),
                }
                let counts = store.collection_counts().unwrap_or_else(|error| {
                    errors.push(format!("collection counts: {error}"));
                    Vec::new()
                });
                for (name, collection) in config.collections {
                    let count = counts
                        .iter()
                        .find(|(candidate, _)| candidate == &name)
                        .map(|(_, count)| *count)
                        .unwrap_or(0);
                    collections.push(CollectionStatus {
                        name,
                        documents: count,
                        path: collection.path.display().to_string(),
                    });
                }
            }
            Err(error) => errors.push(format!("store: {error}")),
        },
        Err(error) => errors.push(format!("config: {error}")),
    }

    RuntimeStatus {
        daemon_online,
        daemon_socket: daemon_socket.display().to_string(),
        hooks_installed,
        hooks_total,
        settings_path: settings_path.display().to_string(),
        documents,
        dirty_documents,
        collections,
    }
}

fn rag_summary(errors: &mut Vec<String>) -> RagSummary {
    let entries = crate::stats::load_entries(Some(7 * 24 * 3600)).unwrap_or_else(|error| {
        errors.push(format!("rag log: {error}"));
        Vec::new()
    });
    let total = entries.len();
    let searched = entries
        .iter()
        .filter(|entry| entry.stage == "searched")
        .count();
    let injected = entries.iter().filter(|entry| entry.injected > 0).count();
    let gated = total.saturating_sub(searched);
    let mut latencies: Vec<u64> = entries
        .iter()
        .filter(|entry| entry.stage == "searched")
        .map(|entry| entry.latency_ms)
        .collect();
    latencies.sort_unstable();
    let median_latency_ms = latencies.get(latencies.len() / 2).copied().unwrap_or(0);
    let p95_latency_ms = if latencies.is_empty() {
        0
    } else {
        latencies[((latencies.len() - 1) * 95) / 100]
    };
    let sessions = group_rag_sessions(&entries, 30);
    let recent = entries.into_iter().rev().take(30).collect();
    RagSummary {
        total,
        searched,
        injected,
        gated,
        injection_rate_pct: pct(injected, searched),
        median_latency_ms,
        p95_latency_ms,
        sessions,
        recent,
    }
}

fn group_rag_sessions(entries: &[crate::rag::RagLogEntry], limit: usize) -> Vec<RagSession> {
    let mut grouped = std::collections::BTreeMap::<String, Vec<RagQueryRecord>>::new();
    for entry in entries.iter().filter(|entry| {
        entry.stage == "searched"
            && !entry.session_id.is_empty()
            && entry.query.as_ref().is_some_and(|query| !query.is_empty())
    }) {
        grouped
            .entry(entry.session_id.clone())
            .or_default()
            .push(RagQueryRecord {
                ts: entry.ts.clone(),
                query: entry.query.clone().unwrap_or_default(),
                injected: entry.injected,
                hits: entry.hits.clone(),
                latency_ms: entry.latency_ms,
            });
    }

    let mut sessions = grouped
        .into_iter()
        .map(|(session_id, mut queries)| {
            queries.sort_by_key(|entry| entry.ts.parse::<u64>().unwrap_or(0));
            RagSession {
                latest_ts: queries
                    .last()
                    .map(|entry| entry.ts.clone())
                    .unwrap_or_default(),
                session_id,
                queries,
            }
        })
        .collect::<Vec<_>>();
    sessions.sort_by(|a, b| {
        let epoch = |value: &str| value.parse::<u64>().unwrap_or(0);
        epoch(&b.latest_ts).cmp(&epoch(&a.latest_ts))
    });
    sessions.truncate(limit);
    sessions
}

/// One row in the Sessions view. A session can appear because it ran kmd queries,
/// because it is a live Claude session observed in the last 24h, or both.
struct SessionEntry<'a> {
    session_id: String,
    sort_epoch: i64,
    rag: Option<&'a RagSession>,
    live: Option<&'a crate::activity::ActivityCard>,
}

impl SessionEntry<'_> {
    fn origin(&self) -> &'static str {
        match (self.rag.is_some(), self.live.is_some()) {
            (true, true) => "live+rag",
            (true, false) => "rag",
            _ => "live",
        }
    }

    fn query_count(&self) -> usize {
        self.rag.map_or(0, |session| session.queries.len())
    }

    /// How recently this session talked to kmd — the liveness signal kmd already owns.
    fn activity(&self, now: i64) -> SessionActivity {
        let last_query = self
            .rag
            .and_then(|session| session.latest_ts.parse::<i64>().ok())
            .unwrap_or(0);
        SessionActivity::from_last_query(now, last_query)
    }
}

fn merge_session_entries<'a>(
    rag_sessions: &'a [RagSession],
    activity: &'a [crate::activity::ActivityCard],
) -> Vec<SessionEntry<'a>> {
    let mut entries: Vec<SessionEntry<'a>> = Vec::new();

    for session in rag_sessions {
        entries.push(SessionEntry {
            sort_epoch: session.latest_ts.parse::<i64>().unwrap_or(0),
            session_id: session.session_id.clone(),
            rag: Some(session),
            live: None,
        });
    }

    for card in activity {
        // rag.jsonl stores the full session id while activity cards may already be
        // truncated, so match on the shared 8-character prefix.
        let matched = entries.iter_mut().find(|entry| {
            let existing: String = entry.session_id.chars().take(8).collect();
            let candidate: String = card.session_id.chars().take(8).collect();
            existing == candidate
        });
        match matched {
            Some(entry) => {
                entry.live = Some(card);
                entry.sort_epoch = entry.sort_epoch.max(card.updated_epoch);
            }
            None => entries.push(SessionEntry {
                session_id: card.session_id.clone(),
                sort_epoch: card.updated_epoch,
                rag: None,
                live: Some(card),
            }),
        }
    }

    entries.sort_by(|a, b| {
        b.sort_epoch
            .cmp(&a.sort_epoch)
            .then(a.session_id.cmp(&b.session_id))
    });
    entries
}

fn parse_check_history(raw: &str) -> Result<Vec<CheckRun>> {
    raw.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| {
            serde_json::from_str(line)
                .with_context(|| format!("invalid check history at line {}", index + 1))
        })
        .collect()
}

fn load_check_history(limit: usize) -> Result<Vec<CheckRun>> {
    let raw = match fs::read_to_string(checks_path()) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error.into()),
    };
    let mut runs = parse_check_history(&raw)?;
    runs.reverse();
    runs.truncate(limit);
    Ok(runs)
}

fn bounded_check_history(raw: &str, run: &CheckRun) -> Result<String> {
    let mut runs = parse_check_history(raw)?;
    let keep = CHECK_HISTORY_LIMIT.saturating_sub(1);
    if runs.len() > keep {
        runs.drain(..runs.len() - keep);
    }
    runs.push(run.clone());

    let mut output = String::new();
    for run in runs {
        output.push_str(&serde_json::to_string(&run)?);
        output.push('\n');
    }
    Ok(output)
}

fn persist_check_run(path: &Path, run: &CheckRun) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let lock_path = path.with_extension("jsonl.lock");
    let lock = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(lock_path)?;
    lock.lock()?;
    let raw = match fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(error) => return Err(error.into()),
    };
    if let Ok(body) = bounded_check_history(&raw, run) {
        let pending = path.with_extension(format!("jsonl.{}.pending", std::process::id()));
        fs::write(&pending, body)?;
        if let Ok(metadata) = fs::metadata(path) {
            fs::set_permissions(&pending, metadata.permissions())?;
        }
        fs::rename(pending, path)?;
    } else {
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        if !raw.is_empty() && !raw.ends_with('\n') {
            writeln!(file)?;
        }
        writeln!(file, "{}", serde_json::to_string(run)?)?;
    }
    Ok(())
}

pub fn snapshot() -> Result<DashboardSnapshot> {
    let cwd = std::env::current_dir()?.display().to_string();
    let mut errors = Vec::new();
    let runtime = runtime_status(&mut errors);
    let activity = crate::activity::recent("", &cwd, 30).unwrap_or_else(|error| {
        errors.push(format!("activity: {error}"));
        Vec::new()
    });
    let journal = crate::journal::build(7, None, None, &cwd).unwrap_or_else(|error| {
        errors.push(format!("journal: {error}"));
        crate::journal::JournalView {
            from: String::new(),
            to: String::new(),
            anchor: crate::locality::resolve(&cwd),
            days: Vec::new(),
            session_count: 0,
            project_count: 0,
        }
    });
    let rag = rag_summary(&mut errors);
    let evaluations = crate::evaluation_log::load(50).unwrap_or_else(|error| {
        errors.push(format!("evaluation history: {error}"));
        Vec::new()
    });
    let checks = load_check_history(20).unwrap_or_else(|error| {
        errors.push(format!("check history: {error}"));
        Vec::new()
    });
    Ok(DashboardSnapshot {
        generated_at: Local::now().to_rfc3339(),
        cwd,
        runtime,
        activity,
        journal,
        rag,
        evaluations,
        checks,
        errors,
    })
}

fn check(name: &str, result: Result<String>) -> CheckItem {
    match result {
        Ok(detail) => CheckItem {
            name: name.to_string(),
            ok: true,
            detail,
        },
        Err(error) => CheckItem {
            name: name.to_string(),
            ok: false,
            detail: error.to_string(),
        },
    }
}

pub fn run_checks() -> Result<CheckRun> {
    let started = Instant::now();
    let mut checks = Vec::new();
    checks.push(check(
        "config",
        (|| {
            let config = crate::config::load()?;
            Ok(format!("{} collections", config.collections.len()))
        })(),
    ));
    checks.push(check(
        "store",
        (|| {
            let store = crate::store::Store::open(&crate::config::store_path())?;
            let (total, dirty) = store.counts()?;
            Ok(format!("{total} active, {dirty} dirty"))
        })(),
    ));
    checks.push(check(
        "tantivy",
        (|| {
            let index = tantivy::Index::open_in_dir(crate::config::tantivy_dir())?;
            let segments = index.searchable_segment_ids()?.len();
            Ok(format!("{segments} searchable segments"))
        })(),
    ));
    checks.push(check(
        "daemon",
        (|| {
            let response = crate::daemon::try_request(&serde_json::json!({"cmd":"ping"}))
                .ok_or_else(|| {
                    anyhow::anyhow!("unreachable at {}", crate::daemon::socket_path().display())
                })?;
            if response.get("ok").and_then(serde_json::Value::as_bool) == Some(true) {
                Ok("ping ok".to_string())
            } else {
                anyhow::bail!("unexpected response: {response}")
            }
        })(),
    ));
    checks.push(check(
        "claude-hooks",
        (|| {
            let (installed, total, path) = crate::hook_config::status_summary()?;
            if installed == total {
                Ok(format!("{installed}/{total} in {}", path.display()))
            } else {
                anyhow::bail!("{installed}/{total} installed in {}", path.display())
            }
        })(),
    ));
    checks.push(check(
        "rag-log",
        (|| {
            let entries = crate::stats::load_entries(None)?;
            Ok(format!("{} parsed entries", entries.len()))
        })(),
    ));
    checks.push(check(
        "check-history",
        (|| {
            let runs = load_check_history(500)?;
            Ok(format!("{} parsed runs", runs.len()))
        })(),
    ));
    checks.push(check(
        "evaluations",
        (|| {
            let runs = crate::evaluation_log::load(500)?;
            Ok(format!("{} parsed runs", runs.len()))
        })(),
    ));
    checks.push(check(
        "search",
        (|| {
            let config = crate::config::load()?;
            let hits =
                crate::bm25::search(&crate::config::tantivy_dir(), &config, "kmd hook", 3, None)?;
            Ok(format!("query completed with {} hits", hits.len()))
        })(),
    ));
    checks.push(check(
        "activity",
        (|| {
            let cwd = std::env::current_dir()?.display().to_string();
            let cards = crate::activity::recent("", &cwd, 30)?;
            Ok(format!("{} recent cards", cards.len()))
        })(),
    ));
    checks.push(check(
        "journal",
        (|| {
            let cwd = std::env::current_dir()?.display().to_string();
            let view = crate::journal::build(7, None, None, &cwd)?;
            Ok(format!(
                "{} sessions across {} projects",
                view.session_count, view.project_count
            ))
        })(),
    ));

    let passed = checks.iter().filter(|item| item.ok).count();
    let run = CheckRun {
        timestamp: Utc::now().to_rfc3339(),
        duration_ms: started.elapsed().as_millis() as u64,
        passed,
        total: checks.len(),
        checks,
    };
    persist_check_run(&checks_path(), &run)?;
    Ok(run)
}

struct App {
    tab: usize,
    scroll: u16,
    selected_session: usize,
    snapshot: DashboardSnapshot,
    simulator: crate::dashboard_simulator::SimulatorState,
    theme: crate::dashboard_theme::Theme,
    last_refresh: Instant,
    message: String,
    show_help: bool,
}

impl App {
    fn session_ids(&self) -> Vec<String> {
        merge_session_entries(&self.snapshot.rag.sessions, &self.snapshot.activity)
            .into_iter()
            .map(|entry| entry.session_id)
            .collect()
    }

    fn refresh(&mut self) {
        let selected_id = self.session_ids().get(self.selected_session).cloned();
        match snapshot() {
            Ok(snapshot) => {
                self.snapshot = snapshot;
                self.selected_session = selected_id
                    .and_then(|id| {
                        self.session_ids()
                            .iter()
                            .position(|candidate| candidate == &id)
                    })
                    .unwrap_or(0);
                self.last_refresh = Instant::now();
                self.message = "refreshed".to_string();
            }
            Err(error) => self.message = format!("refresh failed: {error}"),
        }
    }

    fn select_tab(&mut self, tab: usize) {
        self.tab = tab % TABS.len();
        self.scroll = 0;
    }

    fn select_session(&mut self, next: usize) {
        let count = self.session_ids().len();
        if count > 0 {
            self.selected_session = next.min(count - 1);
            self.scroll = 0;
        }
    }
}

/// Sessions that queried kmd inside the active window — the ones still running right now.
fn active_sessions(snapshot: &DashboardSnapshot) -> usize {
    let now = Utc::now().timestamp();
    merge_session_entries(&snapshot.rag.sessions, &snapshot.activity)
        .iter()
        .filter(|entry| entry.activity(now) == SessionActivity::Active)
        .count()
}

fn state_badge(ok: bool) -> &'static str {
    if ok {
        crate::dashboard_theme::BADGE_OK
    } else {
        crate::dashboard_theme::BADGE_FAIL
    }
}

fn overview_text(snapshot: &DashboardSnapshot) -> String {
    let runtime = &snapshot.runtime;
    let passing_streak = snapshot
        .checks
        .iter()
        .take_while(|run| run.passed == run.total)
        .count();
    let check_status = snapshot
        .checks
        .first()
        .map(|run| {
            format!(
                "{} {}/{} · {} · streak {}",
                state_badge(run.passed == run.total),
                run.passed,
                run.total,
                run.timestamp,
                passing_streak
            )
        })
        .unwrap_or_else(|| "not run (press t)".to_string());
    let evaluation_status = snapshot
        .evaluations
        .first()
        .map(|run| format!("{} · {} · {}", run.kind, run.summary, run.timestamp))
        .unwrap_or_else(|| "not run".to_string());
    let hooks_warning =
        crate::dashboard_theme::warning_suffix(runtime.hooks_installed < runtime.hooks_total);
    let documents_warning = crate::dashboard_theme::warning_suffix(runtime.dirty_documents > 0);
    let mut output = format!(
        "Runtime\n  {:<12} {}\n  {:<12} {}/{} installed{}\n  {:<12} {} active / {} dirty{}\n  {:<12} {}\n  {:<12} {}\n\nKnowledge\n  {:<12} {} sessions / {} projects (7d)\n  {:<12} {} active now / {} with queries (7d)\n  {:<12} {} prompts / {:.0}% injection (7d)\n  {:<12} median {}ms / p95 {}ms\n  {:<12} {}\n",
        "daemon",
        state_badge(runtime.daemon_online),
        "hooks",
        runtime.hooks_installed,
        runtime.hooks_total,
        hooks_warning,
        "documents",
        runtime.documents,
        runtime.dirty_documents,
        documents_warning,
        "socket",
        runtime.daemon_socket,
        "self-check",
        check_status,
        "journal",
        snapshot.journal.session_count,
        snapshot.journal.project_count,
        "sessions",
        active_sessions(snapshot),
        snapshot.rag.sessions.len(),
        "RAG",
        snapshot.rag.total,
        snapshot.rag.injection_rate_pct,
        "latency",
        snapshot.rag.median_latency_ms,
        snapshot.rag.p95_latency_ms,
        "evaluation",
        evaluation_status,
    );
    output.push_str("\nCollections\n");
    for collection in &runtime.collections {
        output.push_str(&format!(
            "  {:<20} {:>6}  {}\n",
            collection.name, collection.documents, collection.path
        ));
    }
    if !snapshot.errors.is_empty() {
        output.push_str("\nErrors\n");
        for error in &snapshot.errors {
            output.push_str(&format!("  • {error}\n"));
        }
    }
    output
}

fn rag_time(timestamp: &str) -> String {
    timestamp
        .parse::<i64>()
        .ok()
        .and_then(|seconds| chrono::DateTime::<Utc>::from_timestamp(seconds, 0))
        .map(|value| {
            value
                .with_timezone(&Local)
                .format("%m-%d %H:%M:%S")
                .to_string()
        })
        .unwrap_or_else(|| timestamp.to_string())
}

fn local_time(timestamp: &str) -> String {
    chrono::DateTime::parse_from_rfc3339(timestamp)
        .map(|value| {
            value
                .with_timezone(&Local)
                .format("%m-%d %H:%M:%S")
                .to_string()
        })
        .unwrap_or_else(|_| timestamp.to_string())
}

fn session_details(entry: &SessionEntry<'_>, now: i64) -> String {
    let id: String = entry.session_id.chars().take(8).collect();
    let mut output = format!(
        "Session {id}\nactivity: {}\norigin: {}\n",
        entry.activity(now).label(),
        entry.origin()
    );

    if let Some(card) = entry.live {
        output.push_str(&format!(
            "repo: {}\ncwd: {}\nupdated: {}\n",
            card.repo.as_deref().unwrap_or("(no repo)"),
            card.cwd,
            local_time(&card.updated_at)
        ));
    }

    match entry.rag {
        Some(session) => output.push_str(&format!(
            "queries: {}\nlatest query: {}\n",
            session.queries.len(),
            rag_time(&session.latest_ts)
        )),
        None => output.push_str("queries: 0\n"),
    }
    output.push('\n');

    if let Some(card) = entry.live {
        output.push_str("LIVE ACTIVITY\n");
        output.push_str(&format!("  summary: {}\n", card.summary));
        if card.latest_user != card.summary {
            output.push_str(&format!("  latest request: {}\n", card.latest_user));
        }
        if !card.latest_assistant.is_empty() {
            output.push_str(&format!("  latest result: {}\n", card.latest_assistant));
        }
        if !card.files.is_empty() {
            output.push_str(&format!("  files: {}\n", card.files.join(", ")));
        }
        output.push('\n');
    }

    let Some(session) = entry.rag else {
        output.push_str("(no kmd retrieval ran in this session)\n");
        return output;
    };

    for (index, entry) in session.queries.iter().enumerate() {
        output.push_str(&format!(
            "── Query {} · {} ──\nQUERY\n{}\n\nRESULTS · {} hits · {} injected · {}ms\n",
            index + 1,
            rag_time(&entry.ts),
            entry.query,
            entry.hits.len(),
            entry.injected,
            entry.latency_ms
        ));
        if entry.hits.is_empty() {
            output.push_str("  (no hits)\n");
        } else {
            for (hit_index, hit) in entry.hits.iter().enumerate() {
                output.push_str(&format!(
                    "  {}. {:.3} {}\n",
                    hit_index + 1,
                    hit.score,
                    hit.file
                ));
                if !hit.snippet.is_empty() {
                    for line in hit.snippet.lines() {
                        output.push_str("     ");
                        output.push_str(line);
                        output.push('\n');
                    }
                }
            }
        }
        output.push('\n');
    }
    output
}

fn operations_text(snapshot: &DashboardSnapshot) -> String {
    let mut output = overview_text(snapshot);
    output.push_str("\nRetrieval\n");
    output.push_str(&rag_text(snapshot));
    output.push_str("\nEvaluations\n");
    output.push_str(&evaluations_text(snapshot));
    output.push_str("\nSelf-checks\n");
    output.push_str(&checks_text(snapshot));
    output
}

fn rag_text(snapshot: &DashboardSnapshot) -> String {
    let rag = &snapshot.rag;
    let mut output = format!(
        "Last 7 days\n  total {} · searched {} · gated {} · injected {} ({:.0}%)\n  latency median {}ms · p95 {}ms\n\nRecent prompts\n",
        rag.total,
        rag.searched,
        rag.gated,
        rag.injected,
        rag.injection_rate_pct,
        rag.median_latency_ms,
        rag.p95_latency_ms
    );
    for entry in &rag.recent {
        let status = if entry.stage == "gated" {
            format!("GATED:{}", entry.gate_reason.as_deref().unwrap_or("?"))
        } else if entry.injected > 0 {
            format!("INJ:{}", entry.injected)
        } else {
            "MISS".to_string()
        };
        // Keep every entry on its two-line layout; embedded newlines would also mimic status rows.
        let prompt: String = entry
            .prompt
            .replace(['\r', '\n'], " ")
            .chars()
            .take(100)
            .collect();
        // dashboard_theme recognizes status rows by this two-space indent; prompt rows use four.
        output.push_str(&format!(
            "  {:<16} {:>5}ms  {}\n    {}\n",
            status, entry.latency_ms, entry.ts, prompt
        ));
    }
    output
}

fn evaluations_text(snapshot: &DashboardSnapshot) -> String {
    if snapshot.evaluations.is_empty() {
        return "아직 평가 기록이 없습니다. kmd eval, kmd util, 또는 kmd ab를 실행하세요."
            .to_string();
    }
    let mut output =
        String::from("Successful L1 retrieval, L2 utilization, and L3 A/B runs · latest first\n\n");
    for run in &snapshot.evaluations {
        output.push_str(&format!(
            "● {}  {:<18} {}\n",
            run.timestamp, run.kind, run.summary
        ));
        if let Some(object) = run.metrics.as_object() {
            let details = object
                .iter()
                .filter(|(_, value)| value.is_number() || value.is_string())
                .take(8)
                .map(|(key, value)| format!("{key}={value}"))
                .collect::<Vec<_>>();
            if !details.is_empty() {
                output.push_str(&format!("  {}\n", details.join(" · ")));
            }
        }
        output.push('\n');
    }
    output
}

fn checks_text(snapshot: &DashboardSnapshot) -> String {
    if snapshot.checks.is_empty() {
        return "아직 self-check 기록이 없습니다. t를 눌러 실행하세요.".to_string();
    }
    let mut output = String::new();
    for run in &snapshot.checks {
        output.push_str(&format!(
            "{}  {}  {}/{} passed  {}ms\n",
            run.timestamp,
            state_badge(run.passed == run.total),
            run.passed,
            run.total,
            run.duration_ms
        ));
        for item in &run.checks {
            output.push_str(&format!(
                "  {} {:<16} {}\n",
                if item.ok { "✓" } else { "✗" },
                item.name,
                item.detail
            ));
        }
        output.push('\n');
    }
    output
}

fn body_content(app: &App) -> Option<(crate::dashboard_theme::BodyKind, String)> {
    match app.tab {
        OPERATIONS_TAB => Some((
            crate::dashboard_theme::BodyKind::Operations,
            operations_text(&app.snapshot),
        )),
        _ => None,
    }
}

fn short_summary(value: &str, limit: usize) -> String {
    if value.chars().count() <= limit {
        value.to_string()
    } else {
        format!("{}…", value.chars().take(limit).collect::<String>())
    }
}

fn draw_sessions(frame: &mut Frame, area: Rect, app: &App) {
    let theme = app.theme;
    let entries = merge_session_entries(&app.snapshot.rag.sessions, &app.snapshot.activity);
    if entries.is_empty() {
        frame.render_widget(
            Paragraph::new(
                "표시할 session이 없습니다.\n최근 7일간 kmd 검색이 실행되었거나 최근 24시간 내 live Claude session이 있으면 여기에 나타납니다.",
            )
            .wrap(Wrap { trim: false })
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(theme.border())
                    .title_style(theme.heading())
                    .title(" Sessions "),
            ),
            area,
        );
        return;
    }

    let now = Utc::now().timestamp();
    let active_count = entries
        .iter()
        .filter(|entry| entry.activity(now) == SessionActivity::Active)
        .count();
    let columns =
        Layout::horizontal([Constraint::Percentage(34), Constraint::Percentage(66)]).split(area);
    let items = entries
        .iter()
        .map(|entry| {
            let id: String = entry.session_id.chars().take(8).collect();
            let activity = entry.activity(now);
            let activity_style = match activity {
                SessionActivity::Active => theme.success(),
                SessionActivity::Recent => theme.warning(),
                SessionActivity::Idle => theme.muted(),
            };
            let summary = entry
                .rag
                .and_then(|session| session.queries.last())
                .map(|query| query.query.as_str())
                .or_else(|| entry.live.map(|card| card.latest_user.as_str()))
                .unwrap_or("(no activity)");
            ListItem::new(vec![
                Line::from(vec![
                    Span::styled(format!("{id} "), theme.accent()),
                    Span::styled(format!("{:<7}", activity.label()), activity_style),
                    Span::styled(format!("{:<9}", entry.origin()), theme.muted()),
                    Span::styled(format!("{} queries", entry.query_count()), theme.muted()),
                ]),
                Line::raw(format!("  {}", short_summary(summary, 72))),
            ])
        })
        .collect::<Vec<_>>();
    let selected = app.selected_session.min(entries.len() - 1);
    let mut state = ListState::default().with_selected(Some(selected));
    let sessions = List::new(items)
        .highlight_symbol("> ")
        .highlight_style(theme.selected())
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title_style(theme.heading())
                .title(format!(
                    " Sessions · {} active now · {} total ",
                    active_count,
                    entries.len()
                )),
        );
    frame.render_stateful_widget(sessions, columns[0], &mut state);

    let detail = crate::dashboard_theme::style_body(
        crate::dashboard_theme::BodyKind::Sessions,
        session_details(&entries[selected], now),
        theme,
    );
    frame.render_widget(
        Paragraph::new(detail)
            .scroll((app.scroll, 0))
            .wrap(Wrap { trim: false })
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(theme.border())
                    .title_style(theme.heading())
                    .title(format!(" kmd query → results · scroll {} ", app.scroll)),
            ),
        columns[1],
    );
}

const HELP_ROWS: &[(&str, &str)] = &[
    ("1-3 / Tab", "switch view"),
    ("h/l / ←/→", "previous / next view"),
    ("j/k / ↑/↓", "Sessions: select session · Operations: scroll"),
    ("PgUp/PgDn", "scroll the timeline or Operations"),
    ("g / G", "top / bottom"),
    ("r", "refresh the snapshot"),
    ("t", "run the self-check suite"),
    ("i or /", "Simulator: start typing"),
    ("Esc", "Simulator: stop typing · otherwise quit"),
    ("Enter", "Simulator: run the query"),
    ("m", "Simulator: RAG ⇄ BM25 search ⇄ Global"),
    ("j/k", "Simulator: select a hit"),
    ("J/K", "Simulator: scroll result detail"),
    ("n/p", "Simulator: next / previous prompt history"),
    ("x", "Simulator: clear the query"),
    ("?", "toggle this help"),
    ("q", "quit"),
    (
        "Ctrl-C",
        "quit from anywhere, including while typing a query",
    ),
];

fn draw_help(frame: &mut Frame, area: Rect, theme: crate::dashboard_theme::Theme) {
    let width = area.width.saturating_sub(4).clamp(20, 72);
    let height = (HELP_ROWS.len() as u16 + 2).min(area.height.saturating_sub(2).max(3));
    let popup = Rect {
        x: area.x + (area.width.saturating_sub(width)) / 2,
        y: area.y + (area.height.saturating_sub(height)) / 2,
        width,
        height,
    };
    let rows = HELP_ROWS
        .iter()
        .map(|(keys, description)| {
            Line::from(vec![
                Span::raw(" "),
                Span::styled(format!("{keys:<18}"), theme.heading()),
                Span::raw((*description).to_string()),
            ])
        })
        .collect::<Vec<_>>();
    frame.render_widget(Clear, popup);
    frame.render_widget(
        Paragraph::new(rows).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.selected())
                .title_style(theme.heading())
                .title(" keys · ? or Esc closes "),
        ),
        popup,
    );
}

fn draw(frame: &mut Frame, app: &App) {
    let theme = app.theme;
    let areas = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(4),
        Constraint::Min(5),
        Constraint::Length(1),
    ])
    .split(frame.area());
    let titles = TABS
        .iter()
        .enumerate()
        .map(|(index, title)| Line::from(format!(" {}:{} ", index + 1, title)))
        .collect::<Vec<_>>();
    let tabs = Tabs::new(titles)
        .select(app.tab)
        .highlight_style(theme.selected())
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title_style(theme.heading())
                .title(format!(" kmd dashboard · {} ", app.snapshot.generated_at)),
        );
    frame.render_widget(tabs, areas[0]);

    let guide = &TAB_GUIDES[app.tab];
    let guide_text = Text::from(vec![
        Line::from(vec![
            Span::styled("Purpose  ", theme.heading()),
            Span::raw(guide.purpose),
        ]),
        Line::from(vec![
            Span::styled("Data     ", theme.muted()),
            Span::raw(guide.source),
        ]),
        Line::from(vec![
            Span::styled("Action   ", theme.muted()),
            Span::raw(guide.action),
        ]),
    ]);
    frame.render_widget(
        Paragraph::new(guide_text).wrap(Wrap { trim: false }),
        areas[1],
    );

    if app.tab == SESSIONS_TAB {
        draw_sessions(frame, areas[2], app);
    } else if app.tab == SIMULATOR_TAB {
        crate::dashboard_simulator::draw(frame, areas[2], &app.simulator);
    } else if let Some((kind, content)) = body_content(app) {
        let body = Paragraph::new(crate::dashboard_theme::style_body(kind, content, theme))
            .scroll((app.scroll, 0))
            .wrap(Wrap { trim: false })
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(theme.border())
                    .title_style(theme.heading())
                    .title(format!(" {} · scroll {} ", TABS[app.tab], app.scroll)),
            );
        frame.render_widget(body, areas[2]);
    }

    let hints = if app.tab == SIMULATOR_TAB {
        if app.simulator.typing() {
            "typing · Esc: command  Enter: run  ↑/↓: history  ?: keys"
        } else {
            "i: type  Enter: run  m: mode  j/k: hit  J/K: detail  x: clear  ?: keys"
        }
    } else if app.tab == SESSIONS_TAB {
        "1-3/Tab: view  j/k: session  PgUp/PgDn: timeline  r: refresh  t: checks  ?: keys"
    } else {
        "1-3/Tab: view  j/k: scroll  g/G: top/bottom  r: refresh  t: checks  ?: keys"
    };
    let footer = Line::from(vec![
        Span::styled(hints.to_string(), theme.muted()),
        Span::styled("  ·  ".to_string(), theme.muted()),
        Span::styled(app.message.clone(), theme.muted()),
    ]);
    frame.render_widget(Paragraph::new(footer), areas[3]);

    if app.show_help {
        draw_help(frame, frame.area(), theme);
    }
}

fn handle_key(app: &mut App, key: crossterm::event::KeyEvent) -> Result<bool> {
    // Ctrl-C quits from anywhere, ahead of the help overlay and the simulator's
    // typing mode. It used to quit from nowhere: is_reserved_chord dropped it on
    // the theory that the terminal would handle it, but ratatui::init puts the
    // terminal in raw mode with no SIGINT handler, so nothing did.
    if key
        .modifiers
        .contains(crossterm::event::KeyModifiers::CONTROL)
        && matches!(key.code, KeyCode::Char('c' | 'C'))
    {
        return Ok(true);
    }

    // Ctrl/Alt/Super chords belong to the terminal or multiplexer and must pass through.
    if crate::dashboard_simulator::is_reserved_chord(&key) {
        return Ok(false);
    }

    // Under a Korean input source the shortcut keys arrive as jamo (`q` -> `ㅂ`)
    // and do nothing until the input source is switched back. Rewrite them to
    // the Latin key at the same physical position -- but not while the simulator
    // is typing a query, where the jamo IS the input.
    let key = if app.tab == SIMULATOR_TAB && app.simulator.typing() {
        key
    } else {
        crate::keymap::normalize(key)
    };
    if app.show_help {
        app.show_help = false;
        return Ok(false);
    }
    if app.tab == SIMULATOR_TAB && app.simulator.typing() {
        app.simulator.handle_typing_key(key);
        return Ok(false);
    }

    match key.code {
        KeyCode::Char('q') | KeyCode::Esc => return Ok(true),
        KeyCode::Char('?') => app.show_help = true,
        KeyCode::Right | KeyCode::Char('l') | KeyCode::Tab => app.select_tab(app.tab + 1),
        KeyCode::Left | KeyCode::Char('h') | KeyCode::BackTab => {
            app.select_tab((app.tab + TABS.len() - 1) % TABS.len())
        }
        KeyCode::Char(value @ '1'..='3') => app.select_tab((value as usize) - ('1' as usize)),
        KeyCode::Char('r') => app.refresh(),
        KeyCode::Char('t') => match run_checks() {
            Ok(run) => {
                app.refresh();
                app.message = format!("self-check: {}/{} passed", run.passed, run.total);
                app.select_tab(OPERATIONS_TAB);
            }
            Err(error) => app.message = format!("self-check failed: {error}"),
        },
        _ if app.tab == SIMULATOR_TAB => app.simulator.handle_command_key(key),
        KeyCode::Down | KeyCode::Char('j') if app.tab == SESSIONS_TAB => {
            app.select_session(app.selected_session.saturating_add(1));
        }
        KeyCode::Up | KeyCode::Char('k') if app.tab == SESSIONS_TAB => {
            app.select_session(app.selected_session.saturating_sub(1));
        }
        KeyCode::Down | KeyCode::Char('j') => app.scroll = app.scroll.saturating_add(1),
        KeyCode::Up | KeyCode::Char('k') => app.scroll = app.scroll.saturating_sub(1),
        KeyCode::PageDown => app.scroll = app.scroll.saturating_add(10),
        KeyCode::PageUp => app.scroll = app.scroll.saturating_sub(10),
        KeyCode::Home | KeyCode::Char('g') => app.scroll = 0,
        KeyCode::End | KeyCode::Char('G') => app.scroll = u16::MAX,
        _ => {}
    }
    Ok(false)
}

fn run_tui(mut app: App) -> Result<()> {
    let mut terminal = ratatui::init();
    let result = (|| -> Result<()> {
        let mut dirty = true;
        loop {
            if dirty {
                terminal.draw(|frame| draw(frame, &app))?;
                dirty = false;
            }

            let wait = if app.simulator.running {
                Duration::from_millis(40)
            } else {
                Duration::from_millis(250)
            };
            if event::poll(wait)? {
                match event::read()? {
                    Event::Key(key) if key.kind == KeyEventKind::Press => {
                        if handle_key(&mut app, key)? {
                            break;
                        }
                        dirty = true;
                    }
                    Event::Resize(_, _) => dirty = true,
                    _ => {}
                }
            }
            if app.simulator.poll() {
                app.message = "simulator result ready".to_string();
                dirty = true;
            }
            if app.last_refresh.elapsed() >= REFRESH_INTERVAL {
                app.refresh();
                dirty = true;
            }
        }
        Ok(())
    })();
    ratatui::restore();
    result
}

pub fn run(json: bool, check_only: bool) -> Result<()> {
    if check_only {
        let run = run_checks()?;
        if json {
            println!("{}", serde_json::to_string_pretty(&run)?);
        } else {
            println!(
                "kmd self-check — {}/{} passed ({}ms)",
                run.passed, run.total, run.duration_ms
            );
            for item in &run.checks {
                println!(
                    "  {} {:<16} {}",
                    if item.ok { "ok" } else { "FAIL" },
                    item.name,
                    item.detail
                );
            }
        }
        if run.passed != run.total {
            anyhow::bail!("{} self-checks failed", run.total - run.passed);
        }
        return Ok(());
    }

    let snapshot = snapshot()?;
    if json {
        println!("{}", serde_json::to_string_pretty(&snapshot)?);
        return Ok(());
    }
    run_tui(App {
        tab: SESSIONS_TAB,
        scroll: 0,
        selected_session: 0,
        snapshot,
        simulator: crate::dashboard_simulator::SimulatorState::new(),
        theme: crate::dashboard_theme::Theme::from_env(),
        last_refresh: Instant::now(),
        message: "auto-refresh 5s".to_string(),
        show_help: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rag_summary_handles_empty_log() {
        assert_eq!(pct(0, 0), 0.0);
        assert_eq!(pct(1, 4), 25.0);
    }

    fn check_run() -> CheckRun {
        CheckRun {
            timestamp: "2026-08-01T00:00:00Z".into(),
            duration_ms: 10,
            passed: 1,
            total: 1,
            checks: vec![CheckItem {
                name: "store".into(),
                ok: true,
                detail: "42 active, 0 dirty".into(),
            }],
        }
    }

    #[test]
    fn check_history_round_trips() {
        let encoded = serde_json::to_string(&check_run()).unwrap();
        let decoded: CheckRun = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.passed, 1);
        assert!(decoded.checks[0].ok);
    }

    #[test]
    fn bounds_check_history() {
        let mut raw = String::new();
        for index in 0..505 {
            let mut run = check_run();
            run.timestamp = format!("run-{index:03}");
            raw.push_str(&serde_json::to_string(&run).unwrap());
            raw.push('\n');
        }
        let mut latest = check_run();
        latest.timestamp = "run-latest".into();

        let bounded = bounded_check_history(&raw, &latest).unwrap();
        let runs: Vec<CheckRun> = bounded
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(runs.len(), CHECK_HISTORY_LIMIT);
        assert_eq!(runs.first().unwrap().timestamp, "run-006");
        assert_eq!(runs.last().unwrap().timestamp, "run-latest");
    }

    #[test]
    fn rejects_invalid_check_history() {
        let error = parse_check_history("not-json\n").unwrap_err();
        assert!(error.to_string().contains("line 1"));
    }

    fn rag_entry(
        session_id: &str,
        ts: &str,
        stage: &str,
        query: Option<&str>,
    ) -> crate::rag::RagLogEntry {
        crate::rag::RagLogEntry {
            ts: ts.into(),
            session_id: session_id.into(),
            prompt: "not exposed by query sessions".into(),
            prompt_len: 29,
            hangul: false,
            stage: stage.into(),
            gate_reason: (stage == "gated").then(|| "too_short".into()),
            query: query.map(str::to_string),
            injected: usize::from(stage == "searched"),
            hits: Vec::new(),
            latency_ms: 10,
        }
    }

    #[test]
    fn groups_only_searched_queries_by_session() {
        let entries = vec![
            rag_entry("older-session", "100", "searched", Some("older second")),
            rag_entry("newer-session", "300", "searched", Some("newer query")),
            rag_entry("older-session", "50", "searched", Some("older first")),
            rag_entry("gated-session", "400", "gated", None),
            rag_entry("empty-query", "500", "searched", Some("")),
        ];

        let sessions = group_rag_sessions(&entries, 30);

        assert_eq!(sessions.len(), 2);
        assert_eq!(sessions[0].session_id, "newer-session");
        assert_eq!(sessions[1].session_id, "older-session");
        assert_eq!(sessions[1].latest_ts, "100");
        assert_eq!(sessions[1].queries[0].query, "older first");
        assert_eq!(sessions[1].queries[1].query, "older second");
        let encoded = serde_json::to_value(&sessions).unwrap();
        assert!(encoded.to_string().contains("older first"));
        assert!(
            !encoded
                .to_string()
                .contains("not exposed by query sessions")
        );
    }

    #[test]
    fn merges_live_activity_with_matching_rag_session() {
        let rag_sessions = vec![RagSession {
            session_id: "12345678-9abc-def0".into(),
            latest_ts: "1754010120".into(),
            queries: vec![RagQueryRecord {
                ts: "1754010120".into(),
                query: "merged query".into(),
                injected: 1,
                hits: Vec::new(),
                latency_ms: 12,
            }],
        }];
        let activity = vec![crate::activity::ActivityCard {
            // Activity cards may carry a truncated id, so merging keys on the 8-char prefix.
            session_id: "12345678".into(),
            updated_at: "2026-08-01T01:10:00Z".into(),
            updated_epoch: 1_754_010_600,
            cwd: "/repo".into(),
            repo: Some("kmd".into()),
            summary: "same session live card".into(),
            latest_user: "same session request".into(),
            latest_assistant: String::new(),
            files: Vec::new(),
            links: Vec::new(),
        }];

        let merged = merge_session_entries(&rag_sessions, &activity);

        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].origin(), "live+rag");
        assert_eq!(merged[0].query_count(), 1);
        let details = session_details(&merged[0], 1_754_010_600);
        assert!(details.contains("origin: live+rag"));
        assert!(details.contains("same session request"));
        assert!(details.contains("merged query"));
    }

    #[test]
    fn last_kmd_query_decides_session_activity() {
        let now = 1_754_010_000_i64;
        let session_at = |ts: i64| RagSession {
            session_id: "session".into(),
            latest_ts: ts.to_string(),
            queries: vec![RagQueryRecord {
                ts: ts.to_string(),
                query: "q".into(),
                injected: 0,
                hits: Vec::new(),
                latency_ms: 1,
            }],
        };

        let just_now = session_at(now - 60);
        let ten_minutes = session_at(now - 600);
        let half_hour = session_at(now - 1_800);
        let two_hours = session_at(now - 7_200);

        let activity_of = |session: &RagSession| {
            merge_session_entries(std::slice::from_ref(session), &[])[0].activity(now)
        };

        assert_eq!(activity_of(&just_now), SessionActivity::Active);
        assert_eq!(activity_of(&ten_minutes), SessionActivity::Active);
        assert_eq!(activity_of(&half_hour), SessionActivity::Recent);
        assert_eq!(activity_of(&two_hours), SessionActivity::Idle);

        // A live card alone is not proof of liveness — kmd only knows a session is
        // running when that session actually sent it a query.
        let card = crate::activity::ActivityCard {
            session_id: "livecard".into(),
            updated_at: "2026-08-01T00:00:00Z".into(),
            updated_epoch: now,
            cwd: "/repo".into(),
            repo: None,
            summary: "card".into(),
            latest_user: "card".into(),
            latest_assistant: String::new(),
            files: Vec::new(),
            links: Vec::new(),
        };
        let merged = merge_session_entries(&[], std::slice::from_ref(&card));
        assert_eq!(merged[0].activity(now), SessionActivity::Idle);
    }

    #[test]
    fn renders_dashboard_tabs_and_checks() {
        let snapshot = DashboardSnapshot {
            generated_at: "2026-08-01T00:00:00Z".into(),
            cwd: "/repo".into(),
            runtime: RuntimeStatus {
                daemon_online: true,
                daemon_socket: "/state/kmd.sock".into(),
                hooks_installed: 5,
                hooks_total: 5,
                settings_path: "/home/.claude/settings.json".into(),
                documents: 42,
                dirty_documents: 0,
                collections: vec![CollectionStatus {
                    name: "learnings".into(),
                    documents: 42,
                    path: "/knowledge".into(),
                }],
            },
            activity: vec![crate::activity::ActivityCard {
                session_id: "abcdef12-live".into(),
                updated_at: "2026-08-01T02:00:00Z".into(),
                updated_epoch: 1_754_013_600,
                cwd: "/repo".into(),
                repo: Some("kmd".into()),
                summary: "live session without kmd retrieval".into(),
                latest_user: "live session latest request".into(),
                latest_assistant: "live session latest result".into(),
                files: vec!["/repo/src/dashboard.rs".into()],
                links: Vec::new(),
            }],
            journal: crate::journal::JournalView {
                from: "20260801".into(),
                to: "20260801".into(),
                anchor: crate::locality::Space {
                    cwd: "/repo".into(),
                    ..Default::default()
                },
                days: Vec::new(),
                session_count: 0,
                project_count: 0,
            },
            rag: RagSummary {
                total: 10,
                searched: 8,
                injected: 4,
                gated: 2,
                injection_rate_pct: 50.0,
                median_latency_ms: 80,
                p95_latency_ms: 120,
                sessions: vec![RagSession {
                    session_id: "12345678-session".into(),
                    latest_ts: "1754010120".into(),
                    queries: vec![RagQueryRecord {
                        ts: "1754010120".into(),
                        query: "dashboard session query results".into(),
                        injected: 1,
                        hits: vec![crate::rag::RagHitLog {
                            file: "kmd://learnings/dashboard.md".into(),
                            score: 12.5,
                            snippet: "session별 retrieval 결과".into(),
                        }],
                        latency_ms: 80,
                    }],
                }],
                recent: vec![crate::rag::RagLogEntry {
                    ts: "2026-08-01T00:00:00Z".into(),
                    session_id: "session".into(),
                    prompt: "MISS should remain prompt text\nafter normalization".into(),
                    prompt_len: "MISS should remain prompt text\nafter normalization"
                        .chars()
                        .count(),
                    hangul: false,
                    stage: "gated".into(),
                    gate_reason: Some("too_short".into()),
                    query: None,
                    injected: 0,
                    hits: Vec::new(),
                    latency_ms: 0,
                }],
            },
            evaluations: vec![crate::evaluation_log::EvaluationRun {
                timestamp: "2026-08-01T00:00:00Z".into(),
                kind: "eval-known-item".into(),
                summary: "10 queries · R@5 90% · MRR 0.800".into(),
                metrics: serde_json::json!({"queries": 10, "recall_at_5_pct": 90}),
            }],
            checks: vec![check_run()],
            errors: Vec::new(),
        };
        let mut app = App {
            tab: SESSIONS_TAB,
            scroll: 0,
            selected_session: 0,
            snapshot,
            simulator: crate::dashboard_simulator::SimulatorState::new(),
            theme: crate::dashboard_theme::Theme::colored(),
            last_refresh: Instant::now(),
            message: "ready".into(),
            show_help: false,
        };
        let backend = ratatui::backend::TestBackend::new(120, 40);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let sessions = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        for label in TABS {
            assert!(sessions.contains(label), "missing view {label}");
        }
        assert!(sessions.contains("abcdef12"));
        assert!(sessions.contains("12345678"));
        assert!(sessions.contains("live"));
        assert!(sessions.contains("rag"));
        assert!(!sessions.contains("대화 원문은 화면에 표시하지 않는다"));

        let merged = merge_session_entries(&app.snapshot.rag.sessions, &app.snapshot.activity);
        assert_eq!(merged.len(), 2);
        // The live-only session is newer, so it sorts ahead of the retrieval session.
        assert_eq!(merged[0].origin(), "live");
        assert_eq!(merged[1].origin(), "rag");

        let live_only = session_details(&merged[0], 1_754_010_200);
        assert!(live_only.contains("LIVE ACTIVITY"));
        assert!(live_only.contains("live session latest request"));
        assert!(live_only.contains("(no kmd retrieval ran in this session)"));
        // No kmd query ever ran here, so kmd cannot claim the session is alive.
        assert!(live_only.contains("activity: idle"));

        let results = session_details(&merged[1], 1_754_010_200);
        assert!(results.contains("activity: active"));
        assert!(results.contains("dashboard session query results"));
        assert!(results.contains("12.500 kmd://learnings/dashboard.md"));
        assert!(results.contains("session별 retrieval 결과"));
        assert!(!results.contains("대화 원문은 화면에 표시하지 않는다"));

        app.theme = crate::dashboard_theme::Theme::plain();
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        assert!(
            terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .all(|cell| cell.fg == Color::Reset)
        );
        app.theme = crate::dashboard_theme::Theme::colored();

        let mut second = app.snapshot.rag.sessions[0].clone();
        second.session_id = "87654321-session".into();
        app.snapshot.rag.sessions.push(second);
        handle_key(
            &mut app,
            crossterm::event::KeyEvent::new(
                KeyCode::Char('j'),
                crossterm::event::KeyModifiers::NONE,
            ),
        )
        .unwrap();
        assert_eq!(app.selected_session, 1);
        assert_eq!(app.scroll, 0);
        handle_key(
            &mut app,
            crossterm::event::KeyEvent::new(
                KeyCode::PageDown,
                crossterm::event::KeyModifiers::NONE,
            ),
        )
        .unwrap();
        assert_eq!(app.selected_session, 1);
        assert_eq!(app.scroll, 10);

        handle_key(
            &mut app,
            crossterm::event::KeyEvent::new(
                KeyCode::Char('k'),
                crossterm::event::KeyModifiers::ALT,
            ),
        )
        .unwrap();
        assert_eq!(app.selected_session, 1);
        assert_eq!(app.scroll, 10);

        handle_key(
            &mut app,
            crossterm::event::KeyEvent::new(
                KeyCode::Char('?'),
                crossterm::event::KeyModifiers::NONE,
            ),
        )
        .unwrap();
        assert!(app.show_help);
        handle_key(
            &mut app,
            crossterm::event::KeyEvent::new(KeyCode::Esc, crossterm::event::KeyModifiers::NONE),
        )
        .unwrap();
        assert!(!app.show_help);

        app.select_tab(SIMULATOR_TAB);
        handle_key(
            &mut app,
            crossterm::event::KeyEvent::new(
                KeyCode::Char('i'),
                crossterm::event::KeyModifiers::NONE,
            ),
        )
        .unwrap();
        assert!(app.simulator.typing());
        handle_key(
            &mut app,
            crossterm::event::KeyEvent::new(
                KeyCode::Char('m'),
                crossterm::event::KeyModifiers::NONE,
            ),
        )
        .unwrap();
        assert_eq!(app.simulator.input, "m");
        assert_eq!(app.simulator.mode, crate::sim::SimulatorMode::Rag);
        handle_key(
            &mut app,
            crossterm::event::KeyEvent::new(KeyCode::Esc, crossterm::event::KeyModifiers::NONE),
        )
        .unwrap();
        handle_key(
            &mut app,
            crossterm::event::KeyEvent::new(
                KeyCode::Char('m'),
                crossterm::event::KeyModifiers::NONE,
            ),
        )
        .unwrap();
        assert_eq!(app.simulator.mode, crate::sim::SimulatorMode::Search);

        let operations = operations_text(&app.snapshot);
        assert!(operations.contains("42 active / 0 dirty"));
        assert!(operations.contains("1/1 passed"));
        assert!(operations.contains("MISS should remain prompt text after normalization"));
        assert!(operations.contains("eval-known-item"));
        assert!(operations.contains("R@5 90%"));

        app.select_tab(OPERATIONS_TAB);
        app.snapshot.runtime.hooks_installed = 4;
        app.snapshot.runtime.dirty_documents = 1;
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        assert!(
            terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .any(|cell| cell.symbol() == "!" && cell.fg == Color::Yellow)
        );
        assert!(
            terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .any(|cell| cell.symbol() == "●" && cell.fg == Color::Green)
        );
    }
}

#[cfg(test)]
mod key_tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    /// A dashboard with no data. The key handler only reads `tab`, `simulator`
    /// and `show_help`, so an empty snapshot is enough to drive it — and it
    /// keeps these tests off the filesystem the real loader reads.
    fn app() -> App {
        App {
            tab: SESSIONS_TAB,
            scroll: 0,
            selected_session: 0,
            snapshot: DashboardSnapshot {
                generated_at: String::new(),
                cwd: String::new(),
                runtime: RuntimeStatus {
                    daemon_online: false,
                    daemon_socket: String::new(),
                    hooks_installed: 0,
                    hooks_total: 0,
                    settings_path: String::new(),
                    documents: 0,
                    dirty_documents: 0,
                    collections: Vec::new(),
                },
                activity: Vec::new(),
                journal: crate::journal::JournalView {
                    from: String::new(),
                    to: String::new(),
                    anchor: crate::locality::Space {
                        cwd: String::new(),
                        repo: None,
                        repo_root: None,
                        worktree: None,
                        worktree_root: None,
                    },
                    days: Vec::new(),
                    session_count: 0,
                    project_count: 0,
                },
                rag: RagSummary {
                    total: 0,
                    searched: 0,
                    injected: 0,
                    gated: 0,
                    injection_rate_pct: 0.0,
                    median_latency_ms: 0,
                    p95_latency_ms: 0,
                    sessions: Vec::new(),
                    recent: Vec::new(),
                },
                evaluations: Vec::new(),
                checks: Vec::new(),
                errors: Vec::new(),
            },
            simulator: crate::dashboard_simulator::SimulatorState::new(),
            theme: crate::dashboard_theme::Theme::colored(),
            last_refresh: Instant::now(),
            message: String::new(),
            show_help: false,
        }
    }

    fn jamo(ch: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE)
    }

    fn ctrl_c() -> KeyEvent {
        KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)
    }

    // --- ctrl-c -------------------------------------------------------------

    // Ctrl-C used to quit from nowhere: is_reserved_chord dropped it on the
    // theory that the terminal would handle it, but ratatui::init puts the
    // terminal in raw mode with no SIGINT handler, so nothing did.
    #[test]
    fn ctrl_c_quits_from_the_sessions_tab() {
        assert!(handle_key(&mut app(), ctrl_c()).unwrap());
    }

    #[test]
    fn ctrl_c_quits_from_the_help_overlay() {
        let mut app = app();
        app.show_help = true;
        assert!(handle_key(&mut app, ctrl_c()).unwrap());
    }

    // The simulator swallows every character while typing, so ctrl-c is the
    // only exit that does not depend on its own bindings.
    #[test]
    fn ctrl_c_quits_out_of_the_simulator_query() {
        let mut app = app();
        app.select_tab(SIMULATOR_TAB);
        handle_key(&mut app, jamo('i')).unwrap();
        assert!(app.simulator.typing(), "`i` should start typing");
        assert!(handle_key(&mut app, ctrl_c()).unwrap());
    }

    // Other reserved chords still pass through to the terminal / multiplexer.
    #[test]
    fn other_ctrl_chords_are_still_reserved() {
        let mut app = app();
        assert!(
            !handle_key(
                &mut app,
                KeyEvent::new(KeyCode::Char('z'), KeyModifiers::CONTROL)
            )
            .unwrap()
        );
    }

    // --- CJK input source ---------------------------------------------------

    // Under a Korean input source every shortcut arrives as a jamo, so without
    // this mapping the dashboard goes dead until the user switches back.
    #[test]
    fn hangul_quits_like_latin_q() {
        // `ㅂ` sits on the physical `q` key under the 2-set layout.
        assert!(handle_key(&mut app(), jamo('ㅂ')).unwrap());
    }

    #[test]
    fn hangul_advances_the_tab_like_latin_l() {
        let mut app = app();
        // `ㅣ` is the physical `l`, which advances the tab.
        let before = app.tab;
        handle_key(&mut app, jamo('ㅣ')).unwrap();
        assert_ne!(app.tab, before, "ㅣ (physical l) must change tab");
    }

    #[test]
    fn hangul_starts_the_simulator_query_like_latin_i() {
        let mut app = app();
        app.select_tab(SIMULATOR_TAB);
        // `ㅑ` is the physical `i`, which starts typing a query.
        handle_key(&mut app, jamo('ㅑ')).unwrap();
        assert!(app.simulator.typing());
    }

    // …and once typing, the jamo IS the query: a Korean search term must reach
    // the input verbatim rather than being rewritten to Latin.
    #[test]
    fn the_simulator_query_keeps_hangul_verbatim() {
        let mut app = app();
        app.select_tab(SIMULATOR_TAB);
        handle_key(&mut app, jamo('i')).unwrap();
        handle_key(&mut app, jamo('ㅂ')).unwrap();
        assert_eq!(app.simulator.input, "ㅂ");
    }
}
