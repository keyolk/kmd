//! RAG 시뮬레이션 — 인터랙티브 TUI(`kmd sim`)와 히스토리 재생(`kmd sim --replay`).
//!
//! TUI: 프롬프트를 타이핑하면 실제 훅과 동일한 파이프라인을 실행해
//! 게이팅/추출 쿼리/히트/주입 컨텍스트를 실시간으로 보여준다.
//! 재생: ~/.claude/history.jsonl의 실제 프롬프트를 일괄 시뮬레이션해 주입률 리포트.

use crate::rag::{self, RagOutcome};
use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Wrap};

// ---------------------------------------------------------------- replay ----

pub fn replay(count: usize, hangul_only: bool) -> Result<()> {
    let path = std::path::PathBuf::from(std::env::var("HOME")?).join(".claude/history.jsonl");
    let raw = std::fs::read_to_string(&path)?;
    let mut prompts: Vec<String> = raw
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter_map(|v| v.get("display").and_then(|d| d.as_str()).map(String::from))
        .filter(|p| !p.is_empty())
        .collect();
    if hangul_only {
        prompts.retain(|p| rag::has_hangul(p));
    }
    // 최근 프롬프트부터 재생
    prompts.reverse();
    prompts.truncate(count);

    let total = prompts.len();
    let mut gated = 0usize;
    let mut injected = 0usize;
    let mut searched = 0usize;
    let mut chunk_sum = 0usize;

    println!("replaying {} prompts from history.jsonl ...\n", total);
    for p in &prompts {
        let o = rag::run_pipeline(p)?;
        let short: String = p.chars().take(52).collect();
        match o.gate_reason {
            Some(reason) => {
                gated += 1;
                println!("  GATE({:12}) {:?}", reason, short);
            }
            None => {
                searched += 1;
                if o.context.is_some() {
                    injected += 1;
                    chunk_sum += o.hits.len();
                    let top = o.hits.first().map(|h| h.file.as_str()).unwrap_or("");
                    println!("  INJ {} {:>4}ms {:?}\n      -> {}", o.hits.len(), o.latency_ms, short, top);
                } else {
                    println!("  MISS  {:>4}ms {:?}", o.latency_ms, short);
                }
            }
        }
    }

    println!();
    println!("== replay report ==");
    println!("  total:    {}", total);
    println!("  gated:    {} ({:.0}%)", gated, pct(gated, total));
    println!("  searched: {}", searched);
    println!(
        "  injected: {}/{} ({:.0}%), avg {:.2} chunks",
        injected,
        searched,
        pct(injected, searched),
        if searched > 0 { chunk_sum as f64 / searched as f64 } else { 0.0 }
    );
    Ok(())
}

fn pct(n: usize, d: usize) -> f64 {
    if d == 0 { 0.0 } else { 100.0 * n as f64 / d as f64 }
}

// ------------------------------------------------------------------- tui ----

struct App {
    input: String,
    /// 마지막 실행 결과 (입력 변경 후 Enter로 갱신)
    outcome: Option<RagOutcome>,
    error: Option<String>,
    history: Vec<String>,
    history_idx: Option<usize>,
}

pub fn tui() -> Result<()> {
    // 과거 프롬프트를 위/아래 키로 불러올 수 있게 로드
    let history: Vec<String> = std::env::var("HOME")
        .ok()
        .map(|h| std::path::PathBuf::from(h).join(".claude/history.jsonl"))
        .and_then(|p| std::fs::read_to_string(p).ok())
        .map(|raw| {
            raw.lines()
                .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
                .filter_map(|v| v.get("display").and_then(|d| d.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default();

    let mut terminal = ratatui::init();
    let result = run_app(
        &mut terminal,
        App {
            input: String::new(),
            outcome: None,
            error: None,
            history,
            history_idx: None,
        },
    );
    ratatui::restore();
    result
}

fn run_app(terminal: &mut Terminal<impl Backend>, mut app: App) -> Result<()> {
    loop {
        terminal.draw(|f| draw(f, &app))?;
        if let Event::Key(key) = event::read()? {
            if key.kind != KeyEventKind::Press {
                continue;
            }
            match (key.code, key.modifiers) {
                (KeyCode::Esc, _) | (KeyCode::Char('c'), KeyModifiers::CONTROL) => return Ok(()),
                (KeyCode::Enter, _) => {
                    if !app.input.trim().is_empty() {
                        match rag::run_pipeline(&app.input) {
                            Ok(o) => {
                                app.outcome = Some(o);
                                app.error = None;
                            }
                            Err(e) => app.error = Some(e.to_string()),
                        }
                    }
                }
                (KeyCode::Backspace, _) => {
                    app.input.pop();
                }
                (KeyCode::Up, _) => {
                    if !app.history.is_empty() {
                        let idx = match app.history_idx {
                            None => app.history.len() - 1,
                            Some(0) => 0,
                            Some(i) => i - 1,
                        };
                        app.history_idx = Some(idx);
                        app.input = app.history[idx].clone();
                    }
                }
                (KeyCode::Down, _) => {
                    if let Some(i) = app.history_idx {
                        if i + 1 < app.history.len() {
                            app.history_idx = Some(i + 1);
                            app.input = app.history[i + 1].clone();
                        } else {
                            app.history_idx = None;
                            app.input.clear();
                        }
                    }
                }
                (KeyCode::Char('u'), KeyModifiers::CONTROL) => {
                    app.input.clear();
                    app.history_idx = None;
                }
                (KeyCode::Char(c), _) => {
                    app.input.push(c);
                    app.history_idx = None;
                }
                _ => {}
            }
        }
    }
}

fn draw(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // input
            Constraint::Length(4), // pipeline status
            Constraint::Min(8),    // hits + context
            Constraint::Length(1), // help
        ])
        .split(f.area());

    // 입력창
    let input = Paragraph::new(app.input.as_str())
        .block(Block::default().borders(Borders::ALL).title(" prompt (Enter=run) "));
    f.render_widget(input, chunks[0]);

    // 파이프라인 상태
    let status_text = match (&app.error, &app.outcome) {
        (Some(e), _) => format!("error: {}", e),
        (None, None) => "프롬프트를 입력하고 Enter — 훅과 동일한 파이프라인이 실행됩니다".into(),
        (None, Some(o)) => match &o.gate_reason {
            Some(r) => format!("GATED: {} — 이 프롬프트는 훅에서 검색 없이 스킵됩니다", r),
            None => format!(
                "query: {}\nhits: {}  |  injected chunks: {}  |  latency: {}ms",
                o.query.as_deref().unwrap_or(""),
                o.hits.len(),
                if o.context.is_some() { o.hits.len() } else { 0 },
                o.latency_ms
            ),
        },
    };
    let status = Paragraph::new(status_text)
        .wrap(Wrap { trim: false })
        .block(Block::default().borders(Borders::ALL).title(" pipeline "));
    f.render_widget(status, chunks[1]);

    // 히트 + 컨텍스트
    let body_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(chunks[2]);

    let hit_items: Vec<ListItem> = app
        .outcome
        .as_ref()
        .map(|o| {
            o.hits
                .iter()
                .map(|h| {
                    ListItem::new(format!(
                        "{:.1} {}",
                        h.score,
                        h.file.strip_prefix("kmd://").unwrap_or(&h.file)
                    ))
                })
                .collect()
        })
        .unwrap_or_default();
    let hits = List::new(hit_items)
        .block(Block::default().borders(Borders::ALL).title(" hits (claude collections) "));
    f.render_widget(hits, body_chunks[0]);

    let ctx_text = app
        .outcome
        .as_ref()
        .and_then(|o| o.context.clone())
        .unwrap_or_else(|| "(주입될 컨텍스트 없음)".into());
    let ctx = Paragraph::new(ctx_text)
        .wrap(Wrap { trim: false })
        .block(Block::default().borders(Borders::ALL).title(" injected <qmd-context> "));
    f.render_widget(ctx, body_chunks[1]);

    // 도움말
    let help = Paragraph::new("Enter: run  ↑/↓: history  Ctrl-U: clear  Esc: quit")
        .style(Style::default().fg(Color::DarkGray));
    f.render_widget(help, chunks[3]);
}
