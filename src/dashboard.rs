//! Operations dashboard for kmd runtime state, activity, journal, RAG, and checks.

use anyhow::{Context, Result};
use chrono::{Local, Utc};
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Tabs, Wrap};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const REFRESH_INTERVAL: Duration = Duration::from_secs(5);
const CHECK_HISTORY_LIMIT: usize = 500;
const TABS: &[&str] = &[
    "Overview",
    "Activity",
    "Journal",
    "RAG",
    "Evaluations",
    "Checks",
    "Simulator",
];
const SIMULATOR_TAB: usize = 6;

struct TabGuide {
    purpose: &'static str,
    source: &'static str,
    action: &'static str,
}

const TAB_GUIDES: &[TabGuide] = &[
    TabGuide {
        purpose: "Runtime and knowledge health at a glance.",
        source: "daemon, hooks, store, collections, recent checks/evaluations",
        action: "r refreshes · t runs the complete self-check suite",
    },
    TabGuide {
        purpose: "What active Claude sessions are currently doing.",
        source: "live transcripts observed during the last 24 hours",
        action: "use cwd and session prefix to locate the source session",
    },
    TabGuide {
        purpose: "Recent work grouped by date and ranked by repository locality.",
        source: "persisted learnings plus live session activity",
        action: "compare current cwd with nearby project/session entries",
    },
    TabGuide {
        purpose: "How UserPromptSubmit retrieval behaves in real usage.",
        source: "7-day rag.jsonl gate, injection, miss, and latency history",
        action: "inspect recent prompts, then reproduce one in Simulator",
    },
    TabGuide {
        purpose: "Whether retrieval improves finding, use, and final answers.",
        source: "persisted L1 eval, L2 utilization, and L3 A/B runs",
        action: "run kmd eval/util/ab to append comparable measurements",
    },
    TabGuide {
        purpose: "Operational verification of every local kmd dependency.",
        source: "config, store, index, daemon, hooks, logs, search, activity, journal",
        action: "t executes checks and keeps the latest 500 runs",
    },
    TabGuide {
        purpose: "Run real queries against the local index without writing logs.",
        source: "RAG gate/filter/context pipeline or raw BM25 retrieval",
        action: "Enter runs · Alt-m changes mode · ↑/↓ recalls prompt history",
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
    snapshot: DashboardSnapshot,
    simulator: crate::dashboard_simulator::SimulatorState,
    last_refresh: Instant,
    message: String,
}

impl App {
    fn refresh(&mut self) {
        match snapshot() {
            Ok(snapshot) => {
                self.snapshot = snapshot;
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
}

fn state_badge(ok: bool) -> &'static str {
    if ok { "● OK" } else { "● FAIL" }
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
    let mut output = format!(
        "Runtime\n  {:<12} {}\n  {:<12} {}/{} installed\n  {:<12} {} active / {} dirty\n  {:<12} {}\n  {:<12} {}\n\nKnowledge\n  {:<12} {} sessions / {} projects (7d)\n  {:<12} {} live cards (24h)\n  {:<12} {} prompts / {:.0}% injection (7d)\n  {:<12} median {}ms / p95 {}ms\n  {:<12} {}\n",
        "daemon",
        state_badge(runtime.daemon_online),
        "hooks",
        runtime.hooks_installed,
        runtime.hooks_total,
        "documents",
        runtime.documents,
        runtime.dirty_documents,
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

fn activity_text(snapshot: &DashboardSnapshot) -> String {
    if snapshot.activity.is_empty() {
        return "최근 24시간 activity가 없습니다.".to_string();
    }
    let mut output = String::new();
    for card in &snapshot.activity {
        let time = chrono::DateTime::parse_from_rfc3339(&card.updated_at)
            .map(|value| {
                value
                    .with_timezone(&Local)
                    .format("%m-%d %H:%M")
                    .to_string()
            })
            .unwrap_or_else(|_| card.updated_at.clone());
        let id: String = card.session_id.chars().take(8).collect();
        output.push_str(&format!(
            "● {}  {:<16} {}\n  {}\n",
            time,
            card.repo.as_deref().unwrap_or("(no repo)"),
            id,
            card.summary
        ));
        if card.latest_user != card.summary {
            output.push_str(&format!("  현재: {}\n", card.latest_user));
        }
        if !card.latest_assistant.is_empty() {
            output.push_str(&format!("  결과: {}\n", card.latest_assistant));
        }
        output.push_str(&format!("  cwd: {}\n\n", card.cwd));
    }
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
        let prompt: String = entry.prompt.chars().take(100).collect();
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
            "{}  {}/{} passed  {}ms\n",
            run.timestamp, run.passed, run.total, run.duration_ms
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

fn body_text(app: &App) -> String {
    match app.tab {
        0 => overview_text(&app.snapshot),
        1 => activity_text(&app.snapshot),
        2 => crate::journal::render(&app.snapshot.journal),
        3 => rag_text(&app.snapshot),
        4 => evaluations_text(&app.snapshot),
        5 => checks_text(&app.snapshot),
        _ => String::new(),
    }
}

fn color_style(color: Color) -> Style {
    if std::env::var_os("NO_COLOR").is_some() {
        Style::default()
    } else {
        Style::default().fg(color)
    }
}

fn draw(frame: &mut Frame, app: &App) {
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
        .highlight_style(color_style(Color::Cyan).bold())
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(" kmd dashboard · {} ", app.snapshot.generated_at)),
        );
    frame.render_widget(tabs, areas[0]);

    let guide = &TAB_GUIDES[app.tab];
    let guide_text = Text::from(vec![
        Line::from(vec![
            Span::styled("Purpose  ", color_style(Color::Cyan).bold()),
            Span::raw(guide.purpose),
        ]),
        Line::from(vec![
            Span::styled("Data     ", color_style(Color::DarkGray)),
            Span::raw(guide.source),
        ]),
        Line::from(vec![
            Span::styled("Action   ", color_style(Color::DarkGray)),
            Span::raw(guide.action),
        ]),
    ]);
    frame.render_widget(
        Paragraph::new(guide_text).wrap(Wrap { trim: false }),
        areas[1],
    );

    if app.tab == SIMULATOR_TAB {
        crate::dashboard_simulator::draw(
            frame,
            areas[2],
            &app.simulator,
            color_style(Color::DarkGray),
        );
    } else {
        let body = Paragraph::new(body_text(app))
            .scroll((app.scroll, 0))
            .wrap(Wrap { trim: false })
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(format!(" {} · scroll {} ", TABS[app.tab], app.scroll)),
            );
        frame.render_widget(body, areas[2]);
    }

    let help_text = if app.tab == SIMULATOR_TAB {
        format!(
            "Tab: next  Shift-Tab/Esc: leave  Enter: run  Alt-m: mode  ↑/↓: history  Alt-j/k: hit  PgUp/PgDn: detail  {}",
            app.message
        )
    } else {
        format!(
            "←/→ or 1-7: tab  j/k ↑/↓ PgUp/PgDn: scroll  r: refresh  t: self-check  q/Esc: quit  {}",
            app.message
        )
    };
    frame.render_widget(
        Paragraph::new(help_text).style(color_style(Color::DarkGray)),
        areas[3],
    );
}

fn handle_key(app: &mut App, key: crossterm::event::KeyEvent) -> Result<bool> {
    if app.tab == SIMULATOR_TAB {
        match (key.code, key.modifiers) {
            (KeyCode::Tab, _) => app.select_tab(app.tab + 1),
            (KeyCode::BackTab, _) | (KeyCode::Esc, _) => app.select_tab(0),
            _ => {
                app.simulator.handle_key(key);
            }
        }
        return Ok(false);
    }

    match (key.code, key.modifiers) {
        (KeyCode::Char('q'), _) | (KeyCode::Esc, _) => return Ok(true),
        (KeyCode::Right, _) | (KeyCode::Char('l'), _) | (KeyCode::Tab, _) => {
            app.select_tab(app.tab + 1)
        }
        (KeyCode::Left, _) | (KeyCode::Char('h'), _) | (KeyCode::BackTab, _) => {
            app.select_tab((app.tab + TABS.len() - 1) % TABS.len())
        }
        (KeyCode::Char(value @ '1'..='7'), _) => app.select_tab((value as usize) - ('1' as usize)),
        (KeyCode::Down, _) | (KeyCode::Char('j'), _) => app.scroll = app.scroll.saturating_add(1),
        (KeyCode::Up, _) | (KeyCode::Char('k'), _) => app.scroll = app.scroll.saturating_sub(1),
        (KeyCode::PageDown, _) => app.scroll = app.scroll.saturating_add(10),
        (KeyCode::PageUp, _) => app.scroll = app.scroll.saturating_sub(10),
        (KeyCode::Home, _) => app.scroll = 0,
        (KeyCode::Char('r'), _) => app.refresh(),
        (KeyCode::Char('t'), _) => match run_checks() {
            Ok(run) => {
                app.refresh();
                app.message = format!("self-check: {}/{} passed", run.passed, run.total);
                app.select_tab(5);
            }
            Err(error) => app.message = format!("self-check failed: {error}"),
        },
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
        tab: 0,
        scroll: 0,
        snapshot,
        simulator: crate::dashboard_simulator::SimulatorState::new(),
        last_refresh: Instant::now(),
        message: "auto-refresh 5s".to_string(),
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
            activity: Vec::new(),
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
                recent: Vec::new(),
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
            tab: 0,
            scroll: 0,
            snapshot,
            simulator: crate::dashboard_simulator::SimulatorState::new(),
            last_refresh: Instant::now(),
            message: "ready".into(),
        };
        let backend = ratatui::backend::TestBackend::new(120, 40);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let overview = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        for label in TABS {
            assert!(overview.contains(label), "missing tab {label}");
        }
        assert!(overview.contains("42 active / 0 dirty"));
        assert!(overview.contains("1/1"));
        assert!(overview.contains("streak 1"));

        app.select_tab(4);
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let evaluations = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(evaluations.contains("eval-known-item"));
        assert!(evaluations.contains("R@5 90%"));

        app.select_tab(5);
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let checks = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(checks.contains("1/1 passed"));
        assert!(checks.contains("42 active, 0 dirty"));
    }
}
