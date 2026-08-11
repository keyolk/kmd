//! Operations dashboard for kmd runtime state, activity, journal, RAG, and checks.

use crate::palette;
use anyhow::{Context, Result};
use chrono::{Local, Utc};
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Tabs, Wrap};
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
    message_style: Style,
    show_help: bool,
}

impl App {
    fn refresh(&mut self) {
        match snapshot() {
            Ok(snapshot) => {
                let errors = snapshot.errors.len();
                self.snapshot = snapshot;
                self.last_refresh = Instant::now();
                if errors == 0 {
                    self.set_message("refreshed".to_string(), palette::muted());
                } else {
                    self.set_message(format!("refreshed with {errors} error(s)"), palette::warn());
                }
            }
            Err(error) => self.set_message(format!("refresh failed: {error}"), palette::failure()),
        }
    }

    fn set_message(&mut self, message: String, style: Style) {
        self.message = message;
        self.message_style = style;
    }

    fn select_tab(&mut self, tab: usize) {
        self.tab = tab % TABS.len();
        self.scroll = 0;
    }
}

/// `● OK` / `● FAIL` — the word carries the state so monochrome stays readable.
fn state_badge(ok: bool) -> Span<'static> {
    Span::styled(
        if ok { "● OK" } else { "● FAIL" }.to_string(),
        palette::state(ok),
    )
}

fn section(title: &str) -> Line<'static> {
    Line::from(Span::styled(title.to_string(), palette::heading()))
}

fn field(name: &str, mut spans: Vec<Span<'static>>) -> Line<'static> {
    let mut line = vec![
        Span::raw("  "),
        Span::styled(format!("{name:<12}"), palette::label()),
    ];
    line.append(&mut spans);
    Line::from(line)
}

/// Injection rate thresholds mirror how the operator reads the RAG tab: above
/// half the searched prompts is healthy, a quarter is worth a look, below that
/// means retrieval is mostly wasted work.
fn rate_style(pct: f64) -> Style {
    if pct >= 50.0 {
        palette::success()
    } else if pct >= 25.0 {
        palette::warn()
    } else {
        palette::failure()
    }
}

fn latency_style(ms: u64) -> Style {
    if ms <= 300 {
        palette::success()
    } else if ms <= 1000 {
        palette::warn()
    } else {
        palette::failure()
    }
}

fn overview_lines(snapshot: &DashboardSnapshot) -> Vec<Line<'static>> {
    let runtime = &snapshot.runtime;
    let hooks_ok = runtime.hooks_installed == runtime.hooks_total;
    let passing_streak = snapshot
        .checks
        .iter()
        .take_while(|run| run.passed == run.total)
        .count();

    let mut lines = vec![section("Runtime")];
    lines.push(field("daemon", vec![state_badge(runtime.daemon_online)]));
    lines.push(field(
        "hooks",
        vec![
            Span::styled(
                format!("{}/{}", runtime.hooks_installed, runtime.hooks_total),
                palette::state(hooks_ok),
            ),
            Span::styled(" installed".to_string(), palette::muted()),
        ],
    ));
    lines.push(field(
        "documents",
        vec![
            Span::styled(runtime.documents.to_string(), palette::strong()),
            Span::styled(" active / ".to_string(), palette::muted()),
            Span::styled(
                runtime.dirty_documents.to_string(),
                palette::state(runtime.dirty_documents == 0),
            ),
            Span::styled(" dirty".to_string(), palette::muted()),
        ],
    ));
    lines.push(field(
        "socket",
        vec![Span::styled(
            runtime.daemon_socket.clone(),
            palette::muted(),
        )],
    ));
    match snapshot.checks.first() {
        Some(run) => lines.push(field(
            "self-check",
            vec![
                state_badge(run.passed == run.total),
                Span::raw(" "),
                Span::styled(
                    format!("{}/{}", run.passed, run.total),
                    palette::state(run.passed == run.total),
                ),
                Span::styled(format!(" · {} · ", run.timestamp), palette::muted()),
                Span::styled(format!("streak {passing_streak}"), palette::value()),
            ],
        )),
        None => lines.push(field(
            "self-check",
            vec![Span::styled(
                "not run (press t)".to_string(),
                palette::warn(),
            )],
        )),
    }

    lines.push(Line::default());
    lines.push(section("Knowledge"));
    lines.push(field(
        "journal",
        vec![
            Span::styled(
                snapshot.journal.session_count.to_string(),
                palette::strong(),
            ),
            Span::styled(" sessions / ".to_string(), palette::muted()),
            Span::styled(
                snapshot.journal.project_count.to_string(),
                palette::strong(),
            ),
            Span::styled(" projects (7d)".to_string(), palette::muted()),
        ],
    ));
    lines.push(field(
        "activity",
        vec![
            Span::styled(snapshot.activity.len().to_string(), palette::strong()),
            Span::styled(" live cards (24h)".to_string(), palette::muted()),
        ],
    ));
    lines.push(field(
        "RAG",
        vec![
            Span::styled(snapshot.rag.total.to_string(), palette::strong()),
            Span::styled(" prompts / ".to_string(), palette::muted()),
            Span::styled(
                format!("{:.0}%", snapshot.rag.injection_rate_pct),
                rate_style(snapshot.rag.injection_rate_pct),
            ),
            Span::styled(" injection (7d)".to_string(), palette::muted()),
        ],
    ));
    lines.push(field(
        "latency",
        vec![
            Span::styled("median ".to_string(), palette::muted()),
            Span::styled(
                format!("{}ms", snapshot.rag.median_latency_ms),
                latency_style(snapshot.rag.median_latency_ms),
            ),
            Span::styled(" · p95 ".to_string(), palette::muted()),
            Span::styled(
                format!("{}ms", snapshot.rag.p95_latency_ms),
                latency_style(snapshot.rag.p95_latency_ms),
            ),
        ],
    ));
    match snapshot.evaluations.first() {
        Some(run) => lines.push(field(
            "evaluation",
            vec![
                Span::styled(run.kind.clone(), palette::accent()),
                Span::styled(" · ".to_string(), palette::muted()),
                Span::styled(run.summary.clone(), palette::value()),
                Span::styled(format!(" · {}", run.timestamp), palette::muted()),
            ],
        )),
        None => lines.push(field(
            "evaluation",
            vec![Span::styled("not run".to_string(), palette::warn())],
        )),
    }

    lines.push(Line::default());
    lines.push(section("Collections"));
    for collection in &runtime.collections {
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(format!("{:<20}", collection.name), palette::label()),
            Span::styled(format!("{:>6}", collection.documents), palette::strong()),
            Span::raw("  "),
            Span::styled(collection.path.clone(), palette::muted()),
        ]));
    }

    if !snapshot.errors.is_empty() {
        lines.push(Line::default());
        lines.push(Line::from(Span::styled(
            "Errors".to_string(),
            palette::failure().add_modifier(Modifier::BOLD),
        )));
        for error in &snapshot.errors {
            lines.push(Line::from(vec![
                Span::styled("  • ".to_string(), palette::failure()),
                Span::styled(error.clone(), palette::failure()),
            ]));
        }
    }
    lines
}

fn activity_lines(snapshot: &DashboardSnapshot) -> Vec<Line<'static>> {
    if snapshot.activity.is_empty() {
        return vec![Line::from(Span::styled(
            "최근 24시간 activity가 없습니다.".to_string(),
            palette::muted(),
        ))];
    }
    let mut lines = Vec::new();
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
        lines.push(Line::from(vec![
            Span::styled("● ".to_string(), palette::success()),
            Span::styled(time, palette::label()),
            Span::raw("  "),
            Span::styled(
                format!("{:<16}", card.repo.as_deref().unwrap_or("(no repo)")),
                palette::accent(),
            ),
            Span::styled(id, palette::muted()),
        ]));
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(card.summary.clone(), palette::value()),
        ]));
        if card.latest_user != card.summary {
            lines.push(Line::from(vec![
                Span::styled("  현재: ".to_string(), palette::label()),
                Span::styled(card.latest_user.clone(), palette::value()),
            ]));
        }
        if !card.latest_assistant.is_empty() {
            lines.push(Line::from(vec![
                Span::styled("  결과: ".to_string(), palette::label()),
                Span::styled(card.latest_assistant.clone(), palette::value()),
            ]));
        }
        lines.push(Line::from(vec![
            Span::styled("  cwd: ".to_string(), palette::muted()),
            Span::styled(card.cwd.clone(), palette::muted()),
        ]));
        lines.push(Line::default());
    }
    lines
}

fn rag_lines(snapshot: &DashboardSnapshot) -> Vec<Line<'static>> {
    let rag = &snapshot.rag;
    let mut lines = vec![
        section("Last 7 days"),
        Line::from(vec![
            Span::styled("  total ".to_string(), palette::muted()),
            Span::styled(rag.total.to_string(), palette::strong()),
            Span::styled(" · searched ".to_string(), palette::muted()),
            Span::styled(rag.searched.to_string(), palette::value()),
            Span::styled(" · gated ".to_string(), palette::muted()),
            Span::styled(rag.gated.to_string(), palette::warn()),
            Span::styled(" · injected ".to_string(), palette::muted()),
            Span::styled(rag.injected.to_string(), palette::success()),
            Span::raw(" "),
            Span::styled(
                format!("({:.0}%)", rag.injection_rate_pct),
                rate_style(rag.injection_rate_pct),
            ),
        ]),
        Line::from(vec![
            Span::styled("  latency median ".to_string(), palette::muted()),
            Span::styled(
                format!("{}ms", rag.median_latency_ms),
                latency_style(rag.median_latency_ms),
            ),
            Span::styled(" · p95 ".to_string(), palette::muted()),
            Span::styled(
                format!("{}ms", rag.p95_latency_ms),
                latency_style(rag.p95_latency_ms),
            ),
        ]),
        Line::default(),
        section("Recent prompts"),
    ];
    for entry in &rag.recent {
        let (status, status_style) = if entry.stage == "gated" {
            (
                format!("GATED:{}", entry.gate_reason.as_deref().unwrap_or("?")),
                palette::warn(),
            )
        } else if entry.injected > 0 {
            (format!("INJ:{}", entry.injected), palette::success())
        } else {
            ("MISS".to_string(), palette::muted())
        };
        let prompt: String = entry.prompt.chars().take(100).collect();
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(format!("{status:<16}"), status_style),
            Span::styled(
                format!("{:>5}ms", entry.latency_ms),
                latency_style(entry.latency_ms),
            ),
            Span::styled(format!("  {}", entry.ts), palette::muted()),
        ]));
        lines.push(Line::from(vec![
            Span::raw("    "),
            Span::styled(prompt, palette::value()),
        ]));
    }
    lines
}

fn evaluations_lines(snapshot: &DashboardSnapshot) -> Vec<Line<'static>> {
    if snapshot.evaluations.is_empty() {
        return vec![Line::from(Span::styled(
            "아직 평가 기록이 없습니다. kmd eval, kmd util, 또는 kmd ab를 실행하세요.".to_string(),
            palette::warn(),
        ))];
    }
    let mut lines = vec![
        section("Successful L1 retrieval, L2 utilization, and L3 A/B runs · latest first"),
        Line::default(),
    ];
    for run in &snapshot.evaluations {
        lines.push(Line::from(vec![
            Span::styled("● ".to_string(), palette::accent()),
            Span::styled(run.timestamp.clone(), palette::muted()),
            Span::raw("  "),
            Span::styled(format!("{:<18}", run.kind), palette::accent()),
            Span::styled(run.summary.clone(), palette::value()),
        ]));
        if let Some(object) = run.metrics.as_object() {
            let details = object
                .iter()
                .filter(|(_, value)| value.is_number() || value.is_string())
                .take(8)
                .map(|(key, value)| format!("{key}={value}"))
                .collect::<Vec<_>>();
            if !details.is_empty() {
                lines.push(Line::from(vec![
                    Span::raw("  "),
                    Span::styled(details.join(" · "), palette::muted()),
                ]));
            }
        }
        lines.push(Line::default());
    }
    lines
}

fn checks_lines(snapshot: &DashboardSnapshot) -> Vec<Line<'static>> {
    if snapshot.checks.is_empty() {
        return vec![Line::from(Span::styled(
            "아직 self-check 기록이 없습니다. t를 눌러 실행하세요.".to_string(),
            palette::warn(),
        ))];
    }
    let mut lines = Vec::new();
    for run in &snapshot.checks {
        let ok = run.passed == run.total;
        lines.push(Line::from(vec![
            Span::styled(run.timestamp.clone(), palette::muted()),
            Span::raw("  "),
            Span::styled(
                format!("{}/{} passed", run.passed, run.total),
                palette::state(ok),
            ),
            Span::styled(format!("  {}ms", run.duration_ms), palette::muted()),
        ]));
        for item in &run.checks {
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(
                    if item.ok { "✓" } else { "✗" }.to_string(),
                    palette::state(item.ok),
                ),
                Span::raw(" "),
                Span::styled(format!("{:<16}", item.name), palette::label()),
                Span::styled(
                    item.detail.clone(),
                    if item.ok {
                        palette::value()
                    } else {
                        palette::failure()
                    },
                ),
            ]));
        }
        lines.push(Line::default());
    }
    lines
}

/// Colorize `journal::render` output by structural prefix so the journal tab
/// gains hierarchy without duplicating the renderer.
fn journal_lines(rendered: &str) -> Vec<Line<'static>> {
    rendered
        .lines()
        .map(|raw| {
            let trimmed = raw.trim_start();
            let indent = raw.len() - trimmed.len();
            let style = if raw.starts_with("kmd journal") {
                palette::heading()
            } else if raw.starts_with("anchor:") {
                palette::muted()
            } else if indent == 0 && !trimmed.is_empty() {
                palette::heading()
            } else if indent == 2 {
                palette::accent()
            } else if trimmed.starts_with('●') {
                palette::success()
            } else if trimmed.starts_with('○') {
                palette::value()
            } else if trimmed.starts_with("cwd:")
                || trimmed.starts_with("worktree:")
                || trimmed.starts_with("workspace:")
            {
                palette::muted()
            } else if trimmed.starts_with("현재:") {
                palette::label()
            } else {
                palette::value()
            };
            Line::from(Span::styled(raw.to_string(), style))
        })
        .collect()
}

fn body_lines(app: &App) -> Vec<Line<'static>> {
    match app.tab {
        0 => overview_lines(&app.snapshot),
        1 => activity_lines(&app.snapshot),
        2 => journal_lines(&crate::journal::render(&app.snapshot.journal)),
        3 => rag_lines(&app.snapshot),
        4 => evaluations_lines(&app.snapshot),
        5 => checks_lines(&app.snapshot),
        _ => Vec::new(),
    }
}

const HELP_ROWS: &[(&str, &str)] = &[
    ("1-7", "jump to a tab"),
    ("Tab / Shift-Tab", "next / previous tab"),
    ("h l  ← →", "previous / next tab"),
    ("j k  ↑ ↓", "scroll (Simulator: select hit)"),
    ("J K", "Simulator: scroll result detail"),
    ("PgUp PgDn", "scroll a page"),
    ("g G", "jump to top / bottom"),
    ("r", "refresh the snapshot"),
    ("t", "run the self-check suite"),
    ("i or /", "Simulator: start typing a query"),
    ("Esc", "Simulator: stop typing · otherwise quit"),
    ("Enter", "Simulator: run the query"),
    ("m", "Simulator: RAG pipeline ⇄ BM25 search"),
    ("x", "Simulator: clear the query"),
    ("n p", "Simulator: next / previous prompt history"),
    ("?", "toggle this help"),
    ("q", "quit"),
];

fn draw_help(frame: &mut Frame, area: Rect) {
    let width = area.width.saturating_sub(4).min(64).max(20);
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
                Span::styled(format!("{keys:<16}"), palette::label()),
                Span::styled((*description).to_string(), palette::value()),
            ])
        })
        .collect::<Vec<_>>();
    frame.render_widget(Clear, popup);
    frame.render_widget(
        Paragraph::new(rows).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(palette::border_focus())
                .title(Span::styled(
                    " keys · ? or Esc closes ".to_string(),
                    palette::heading(),
                )),
        ),
        popup,
    );
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
        .map(|(index, title)| {
            Line::from(vec![
                Span::styled(format!(" {}:", index + 1), palette::muted()),
                Span::raw(format!("{title} ")),
            ])
        })
        .collect::<Vec<_>>();
    let tabs = Tabs::new(titles)
        .select(app.tab)
        .style(palette::value())
        .highlight_style(palette::heading().add_modifier(Modifier::REVERSED))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(palette::border())
                .title(Line::from(vec![
                    Span::styled(" kmd dashboard ".to_string(), palette::heading()),
                    Span::styled(
                        format!("· {} ", app.snapshot.generated_at),
                        palette::muted(),
                    ),
                ])),
        );
    frame.render_widget(tabs, areas[0]);

    let guide = &TAB_GUIDES[app.tab];
    let guide_text = Text::from(vec![
        Line::from(vec![
            Span::styled("Purpose  ", palette::heading()),
            Span::styled(guide.purpose.to_string(), palette::value()),
        ]),
        Line::from(vec![
            Span::styled("Data     ", palette::label()),
            Span::styled(guide.source.to_string(), palette::muted()),
        ]),
        Line::from(vec![
            Span::styled("Action   ", palette::label()),
            Span::styled(guide.action.to_string(), palette::muted()),
        ]),
    ]);
    frame.render_widget(
        Paragraph::new(guide_text).wrap(Wrap { trim: false }),
        areas[1],
    );

    if app.tab == SIMULATOR_TAB {
        crate::dashboard_simulator::draw(frame, areas[2], &app.simulator);
    } else {
        let body = Paragraph::new(body_lines(app))
            .scroll((app.scroll, 0))
            .wrap(Wrap { trim: false })
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(palette::border_focus())
                    .title(Line::from(vec![
                        Span::styled(format!(" {} ", TABS[app.tab]), palette::heading()),
                        Span::styled(format!("· scroll {} ", app.scroll), palette::muted()),
                    ])),
            );
        frame.render_widget(body, areas[2]);
    }

    let hints = if app.tab == SIMULATOR_TAB {
        if app.simulator.typing() {
            "typing · Esc: stop  Enter: run  ↑/↓: history"
        } else {
            "i: type  Enter: run  m: mode  j/k: hit  J/K: detail  x: clear  ?: keys"
        }
    } else {
        "1-7/Tab: tab  j/k: scroll  g/G: top/bottom  r: refresh  t: self-check  ?: keys  q: quit"
    };
    let mut footer = vec![Span::styled(hints.to_string(), palette::muted())];
    if !app.message.is_empty() {
        footer.push(Span::styled("  ·  ".to_string(), palette::muted()));
        footer.push(Span::styled(app.message.clone(), app.message_style));
    }
    frame.render_widget(Paragraph::new(Line::from(footer)), areas[3]);

    if app.show_help {
        draw_help(frame, frame.area());
    }
}

fn handle_key(app: &mut App, key: crossterm::event::KeyEvent) -> Result<bool> {
    // Terminal-reserved chords (Ctrl/Alt) are deliberately unbound: the
    // Simulator uses a command/typing mode split instead, so plain letters stay
    // available as commands without stealing keys the terminal owns.
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
        KeyCode::Char(value @ '1'..='7') => app.select_tab((value as usize) - ('1' as usize)),
        KeyCode::Char('r') => app.refresh(),
        KeyCode::Char('t') => match run_checks() {
            Ok(run) => {
                app.refresh();
                app.set_message(
                    format!("self-check: {}/{} passed", run.passed, run.total),
                    palette::state(run.passed == run.total),
                );
                app.select_tab(5);
            }
            Err(error) => {
                app.set_message(format!("self-check failed: {error}"), palette::failure())
            }
        },
        _ if app.tab == SIMULATOR_TAB => {
            app.simulator.handle_command_key(key);
        }
        KeyCode::Down | KeyCode::Char('j') => app.scroll = app.scroll.saturating_add(1),
        KeyCode::Up | KeyCode::Char('k') => app.scroll = app.scroll.saturating_sub(1),
        KeyCode::PageDown => app.scroll = app.scroll.saturating_add(10),
        KeyCode::PageUp => app.scroll = app.scroll.saturating_sub(10),
        KeyCode::Char('g') | KeyCode::Home => app.scroll = 0,
        KeyCode::Char('G') | KeyCode::End => app.scroll = u16::MAX / 2,
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
        message_style: palette::muted(),
        show_help: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyEvent, KeyModifiers};

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

    fn test_app() -> App {
        App {
            tab: 0,
            scroll: 0,
            snapshot: sample_snapshot(),
            simulator: crate::dashboard_simulator::SimulatorState::new(),
            last_refresh: Instant::now(),
            message: "ready".into(),
            message_style: palette::muted(),
            show_help: false,
        }
    }

    fn sample_snapshot() -> DashboardSnapshot {
        DashboardSnapshot {
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
        }
    }

    #[test]
    fn renders_dashboard_tabs_and_checks() {
        let mut app = test_app();
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

    #[test]
    fn overview_colors_state_and_thresholds() {
        let lines = overview_lines(&sample_snapshot());
        let styles: Vec<Style> = lines
            .iter()
            .flat_map(|line| line.spans.iter().map(|span| span.style))
            .collect();

        assert!(
            styles.contains(&palette::success()),
            "healthy state should use the success token"
        );
        assert!(
            styles.contains(&palette::heading()),
            "sections should use the heading token"
        );
        assert!(
            styles.contains(&palette::muted()),
            "metadata should use the muted token"
        );
    }

    #[test]
    fn rate_and_latency_thresholds_map_to_tokens() {
        assert_eq!(rate_style(80.0), palette::success());
        assert_eq!(rate_style(30.0), palette::warn());
        assert_eq!(rate_style(5.0), palette::failure());

        assert_eq!(latency_style(120), palette::success());
        assert_eq!(latency_style(700), palette::warn());
        assert_eq!(latency_style(2500), palette::failure());
    }

    #[test]
    fn failing_check_uses_failure_token() {
        let mut snapshot = sample_snapshot();
        snapshot.checks[0].passed = 0;
        snapshot.checks[0].checks[0].ok = false;

        let styles: Vec<Style> = checks_lines(&snapshot)
            .iter()
            .flat_map(|line| line.spans.iter().map(|span| span.style))
            .collect();
        assert!(styles.contains(&palette::failure()));
    }

    /// Ctrl/Alt chords collide with the terminal and the tmux prefix, so the
    /// dashboard must never consume them.
    #[test]
    fn reserved_chords_are_not_bound() {
        for (code, modifiers) in [
            (KeyCode::Char('c'), KeyModifiers::CONTROL),
            (KeyCode::Char('u'), KeyModifiers::CONTROL),
            (KeyCode::Char('m'), KeyModifiers::ALT),
            (KeyCode::Char('j'), KeyModifiers::ALT),
        ] {
            let mut app = test_app();
            app.select_tab(SIMULATOR_TAB);
            app.simulator
                .handle_command_key(KeyEvent::new(code, modifiers));

            assert_eq!(
                app.simulator.mode,
                crate::sim::SimulatorMode::Rag,
                "{code:?}+{modifiers:?} must not change the retrieval mode"
            );
            assert!(
                !app.simulator.typing(),
                "{code:?}+{modifiers:?} must not start typing"
            );
        }
    }

    #[test]
    fn help_overlay_toggles_and_absorbs_the_next_key() {
        let mut app = test_app();
        assert!(!handle_key(&mut app, KeyEvent::from(KeyCode::Char('?'))).unwrap());
        assert!(app.show_help);

        // The next key closes the overlay instead of acting, so `q` cannot quit
        // while the operator is still reading the keymap.
        assert!(!handle_key(&mut app, KeyEvent::from(KeyCode::Char('q'))).unwrap());
        assert!(!app.show_help);
    }

    #[test]
    fn typing_mode_keeps_q_out_of_the_quit_path() {
        let mut app = test_app();
        app.select_tab(SIMULATOR_TAB);
        handle_key(&mut app, KeyEvent::from(KeyCode::Char('i'))).unwrap();
        assert!(app.simulator.typing());

        assert!(!handle_key(&mut app, KeyEvent::from(KeyCode::Char('q'))).unwrap());
        assert_eq!(app.simulator.input, "q");

        handle_key(&mut app, KeyEvent::from(KeyCode::Esc)).unwrap();
        assert!(!app.simulator.typing());
        assert!(handle_key(&mut app, KeyEvent::from(KeyCode::Char('q'))).unwrap());
    }
}
