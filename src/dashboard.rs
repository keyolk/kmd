//! Operations dashboard for kmd runtime state, activity, journal, RAG, and checks.

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
        purpose: "Inspect each Claude session as a transparent request → response timeline.",
        source: "live transcripts observed during the last 24 hours",
        action: "j/k selects a session · PgUp/PgDn scrolls its complete timeline",
    },
    TabGuide {
        purpose: "Monitor runtime, retrieval quality, evaluations, and self-checks together.",
        source: "daemon, hooks, store, rag.jsonl, evaluations, and check history",
        action: "r refreshes · t runs the complete self-check suite",
    },
    TabGuide {
        purpose: "Run real queries against the local index without writing logs.",
        source: "RAG gate/filter/context pipeline or raw BM25 retrieval",
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

#[derive(Debug, Serialize)]
pub struct RagSummary {
    pub total: usize,
    pub searched: usize,
    pub injected: usize,
    pub gated: usize,
    pub injection_rate_pct: f64,
    pub median_latency_ms: u64,
    pub p95_latency_ms: u64,
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
    let recent = entries.into_iter().rev().take(30).collect();
    RagSummary {
        total,
        searched,
        injected,
        gated,
        injection_rate_pct: pct(injected, searched),
        median_latency_ms,
        p95_latency_ms,
        recent,
    }
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
    let mut activity = crate::activity::recent("", &cwd, 30).unwrap_or_else(|error| {
        errors.push(format!("activity: {error}"));
        Vec::new()
    });
    if let Err(error) = crate::activity::backfill_turns(&mut activity) {
        errors.push(format!("activity turn backfill: {error}"));
    }
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
    fn refresh(&mut self) {
        let selected_id = self
            .snapshot
            .activity
            .get(self.selected_session)
            .map(|card| card.session_id.clone());
        match snapshot() {
            Ok(snapshot) => {
                self.snapshot = snapshot;
                self.selected_session = selected_id
                    .and_then(|id| {
                        self.snapshot
                            .activity
                            .iter()
                            .position(|card| card.session_id == id)
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
        if !self.snapshot.activity.is_empty() {
            self.selected_session = next.min(self.snapshot.activity.len() - 1);
            self.scroll = 0;
        }
    }
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
        "Runtime\n  {:<12} {}\n  {:<12} {}/{} installed{}\n  {:<12} {} active / {} dirty{}\n  {:<12} {}\n  {:<12} {}\n\nKnowledge\n  {:<12} {} sessions / {} projects (7d)\n  {:<12} {} live cards (24h)\n  {:<12} {} prompts / {:.0}% injection (7d)\n  {:<12} median {}ms / p95 {}ms\n  {:<12} {}\n",
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
        "activity",
        snapshot.activity.len(),
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

fn session_time(timestamp: &str) -> String {
    chrono::DateTime::parse_from_rfc3339(timestamp)
        .map(|value| value.with_timezone(&Local).format("%H:%M:%S").to_string())
        .unwrap_or_else(|_| "--:--:--".to_string())
}

fn session_timeline(card: &crate::activity::ActivityCard) -> String {
    let id: String = card.session_id.chars().take(8).collect();
    let mut output = format!(
        "Session {id}\nrepo: {}\ncwd: {}\nupdated: {}\nturns: {}\n",
        card.repo.as_deref().unwrap_or("(no repo)"),
        card.cwd,
        card.updated_at,
        card.turns.len()
    );
    if !card.files.is_empty() {
        output.push_str(&format!("files: {}\n", card.files.join(", ")));
    }
    output.push('\n');

    if card.turns.is_empty() {
        output
            .push_str("이 카드는 이전 저장 형식으로 생성되어 마지막 요청과 응답만 표시합니다.\n\n");
        output.push_str("USER\n");
        output.push_str(&card.latest_user);
        output.push_str("\n\nCLAUDE\n");
        if card.latest_assistant.is_empty() {
            output.push_str("(텍스트 응답 없음)\n");
        } else {
            output.push_str(&card.latest_assistant);
            output.push('\n');
        }
        return output;
    }

    for (index, turn) in card.turns.iter().enumerate() {
        output.push_str(&format!(
            "── Turn {} · {} ──\nUSER\n{}\n\nCLAUDE\n",
            index + 1,
            session_time(&turn.timestamp),
            turn.user
        ));
        if turn.assistant.is_empty() {
            output.push_str("(텍스트 응답 없음 또는 응답 대기 중)");
        } else {
            output.push_str(&turn.assistant);
        }
        output.push_str("\n\n");
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
    if app.snapshot.activity.is_empty() {
        frame.render_widget(
            Paragraph::new(
                "최근 24시간의 session activity가 없습니다.\nClaude 세션에서 Stop hook이 실행되면 요청과 응답 timeline이 여기에 나타납니다.",
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

    let columns =
        Layout::horizontal([Constraint::Percentage(34), Constraint::Percentage(66)]).split(area);
    let items = app
        .snapshot
        .activity
        .iter()
        .map(|card| {
            let id: String = card.session_id.chars().take(8).collect();
            let updated = chrono::DateTime::parse_from_rfc3339(&card.updated_at)
                .map(|value| {
                    value
                        .with_timezone(&Local)
                        .format("%m-%d %H:%M")
                        .to_string()
                })
                .unwrap_or_else(|_| card.updated_at.clone());
            ListItem::new(vec![
                Line::from(vec![
                    Span::styled(format!("{id} "), theme.accent()),
                    Span::raw(card.repo.as_deref().unwrap_or("(no repo)").to_string()),
                    Span::styled(format!("  {updated}"), theme.muted()),
                ]),
                Line::raw(format!("  {}", short_summary(&card.latest_user, 72))),
            ])
        })
        .collect::<Vec<_>>();
    let mut state = ListState::default().with_selected(Some(app.selected_session));
    let sessions = List::new(items)
        .highlight_symbol("> ")
        .highlight_style(theme.selected())
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title_style(theme.heading())
                .title(format!(" Sessions · {} live ", app.snapshot.activity.len())),
        );
    frame.render_stateful_widget(sessions, columns[0], &mut state);

    let card = &app.snapshot.activity[app.selected_session];
    let timeline = crate::dashboard_theme::style_body(
        crate::dashboard_theme::BodyKind::Sessions,
        session_timeline(card),
        theme,
    );
    frame.render_widget(
        Paragraph::new(timeline)
            .scroll((app.scroll, 0))
            .wrap(Wrap { trim: false })
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(theme.border())
                    .title_style(theme.heading())
                    .title(format!(
                        " Request → Claude response · scroll {} ",
                        app.scroll
                    )),
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
    ("m", "Simulator: RAG pipeline ⇄ BM25 search"),
    ("j/k", "Simulator: select a hit"),
    ("J/K", "Simulator: scroll result detail"),
    ("n/p", "Simulator: next / previous prompt history"),
    ("x", "Simulator: clear the query"),
    ("?", "toggle this help"),
    ("q", "quit"),
];

fn draw_help(frame: &mut Frame, area: Rect, theme: crate::dashboard_theme::Theme) {
    let width = area.width.saturating_sub(4).min(72).max(20);
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
    // Ctrl/Alt/Super chords belong to the terminal or multiplexer and must pass through.
    if crate::dashboard_simulator::is_reserved_chord(&key) {
        return Ok(false);
    }
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
                session_id: "12345678-session".into(),
                updated_at: "2026-08-01T01:02:00Z".into(),
                updated_epoch: 1_754_011_320,
                cwd: "/repo".into(),
                repo: Some("kmd".into()),
                summary: "대시보드 세션 흐름을 개선한다".into(),
                latest_user: "질의와 응답을 세션별로 보여줘".into(),
                latest_assistant: "세션 중심 화면으로 변경했습니다.".into(),
                turns: vec![crate::activity::ActivityTurn {
                    timestamp: "2026-08-01T01:01:00Z".into(),
                    user: "질의와 응답을 세션별로 보여줘".into(),
                    assistant: "세션 중심 화면으로 변경했습니다.".into(),
                }],
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
        assert!(sessions.contains("12345678"));
        assert!(sessions.contains("USER"));
        assert!(sessions.contains("CLAUDE"));
        let timeline = session_timeline(&app.snapshot.activity[0]);
        assert!(timeline.contains("질의와 응답을 세션별로 보여줘"));
        assert!(timeline.contains("세션 중심 화면으로 변경했습니다."));

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

        let mut second = app.snapshot.activity[0].clone();
        second.session_id = "87654321-session".into();
        app.snapshot.activity.push(second);
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
