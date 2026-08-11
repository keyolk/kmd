//! Interactive query simulator embedded in the operations dashboard.

use crate::sim::{self, SimulatorMode, SimulatorResult};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};
use std::sync::mpsc::{self, Receiver, Sender};

const SEARCH_LIMIT: usize = 10;
const MIN_WIDTH: u16 = 60;
const MIN_HEIGHT: u16 = 15;
const WIDE_WIDTH: u16 = 96;

type WorkerResult = (u64, Result<SimulatorResult, String>);

pub struct SimulatorState {
    pub input: String,
    pub mode: SimulatorMode,
    pub result: Option<SimulatorResult>,
    pub error: Option<String>,
    pub running: bool,
    pub selected_hit: usize,
    pub detail_scroll: u16,
    history: Vec<String>,
    history_idx: Option<usize>,
    generation: u64,
    tx: Sender<WorkerResult>,
    rx: Receiver<WorkerResult>,
}

impl SimulatorState {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            input: String::new(),
            mode: SimulatorMode::Rag,
            result: None,
            error: None,
            running: false,
            selected_hit: 0,
            detail_scroll: 0,
            history: sim::prompt_history(),
            history_idx: None,
            generation: 0,
            tx,
            rx,
        }
    }

    pub fn run(&mut self) {
        let input = self.input.trim().to_string();
        if input.is_empty() || self.running {
            return;
        }
        self.generation += 1;
        let generation = self.generation;
        let mode = self.mode;
        let tx = self.tx.clone();
        self.running = true;
        self.error = None;
        self.selected_hit = 0;
        self.detail_scroll = 0;
        std::thread::spawn(move || {
            let result =
                sim::execute(mode, &input, SEARCH_LIMIT).map_err(|error| error.to_string());
            let _ = tx.send((generation, result));
        });
    }

    pub fn poll(&mut self) -> bool {
        let mut changed = false;
        while let Ok((generation, result)) = self.rx.try_recv() {
            if generation != self.generation {
                continue;
            }
            self.running = false;
            match result {
                Ok(result) => {
                    self.result = Some(result);
                    self.error = None;
                }
                Err(error) => {
                    self.result = None;
                    self.error = Some(error);
                }
            }
            changed = true;
        }
        changed
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> bool {
        match (key.code, key.modifiers) {
            (KeyCode::Enter, _) => self.run(),
            (KeyCode::Char('m'), KeyModifiers::ALT) => {
                self.mode = self.mode.toggle();
                self.result = None;
                self.error = None;
            }
            (KeyCode::Char('u'), KeyModifiers::CONTROL) => {
                self.input.clear();
                self.history_idx = None;
            }
            (KeyCode::Backspace, _) => {
                self.input.pop();
                self.history_idx = None;
            }
            (KeyCode::Up, _) => self.previous_history(),
            (KeyCode::Down, _) => self.next_history(),
            (KeyCode::Char('j'), KeyModifiers::ALT) => self.next_hit(),
            (KeyCode::Char('k'), KeyModifiers::ALT) => self.previous_hit(),
            (KeyCode::PageDown, _) => self.detail_scroll = self.detail_scroll.saturating_add(8),
            (KeyCode::PageUp, _) => self.detail_scroll = self.detail_scroll.saturating_sub(8),
            (KeyCode::Home, _) => self.detail_scroll = 0,
            (KeyCode::Char(value), modifiers)
                if modifiers.is_empty() || modifiers == KeyModifiers::SHIFT =>
            {
                self.input.push(value);
                self.history_idx = None;
            }
            _ => return false,
        }
        true
    }

    fn previous_history(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let index = match self.history_idx {
            None => self.history.len() - 1,
            Some(0) => 0,
            Some(index) => index - 1,
        };
        self.history_idx = Some(index);
        self.input = self.history[index].clone();
    }

    fn next_history(&mut self) {
        let Some(index) = self.history_idx else {
            return;
        };
        if index + 1 < self.history.len() {
            self.history_idx = Some(index + 1);
            self.input = self.history[index + 1].clone();
        } else {
            self.history_idx = None;
            self.input.clear();
        }
    }

    fn next_hit(&mut self) {
        let count = self.result.as_ref().map_or(0, |result| result.hits.len());
        if count > 0 {
            self.selected_hit = (self.selected_hit + 1).min(count - 1);
            self.detail_scroll = 0;
        }
    }

    fn previous_hit(&mut self) {
        self.selected_hit = self.selected_hit.saturating_sub(1);
        self.detail_scroll = 0;
    }
}

pub fn draw(
    frame: &mut Frame,
    area: Rect,
    state: &SimulatorState,
    theme: crate::dashboard_theme::Theme,
) {
    if area.width < MIN_WIDTH || area.height < MIN_HEIGHT {
        let message = Paragraph::new(format!(
            "Simulator needs at least {MIN_WIDTH}x{MIN_HEIGHT}. Current area: {}x{}",
            area.width, area.height
        ))
        .wrap(Wrap { trim: false })
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title_style(theme.warning())
                .title(" terminal too small "),
        );
        frame.render_widget(message, area);
        return;
    }

    let rows = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(4),
        Constraint::Min(6),
    ])
    .split(area);

    let input_title = format!(
        " query · {} · Enter runs{} ",
        state.mode.label(),
        if state.running { " · RUNNING" } else { "" }
    );
    let input_title_style = if state.running {
        theme.warning().add_modifier(Modifier::BOLD)
    } else {
        theme.heading()
    };
    let input = Paragraph::new(state.input.as_str()).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(theme.border())
            .title_style(input_title_style)
            .title(input_title),
    );
    frame.render_widget(input, rows[0]);

    let (status_text, status_style) = match (&state.error, &state.result, state.running) {
        (_, _, true) => (
            "Running against the local kmd index…".to_string(),
            theme.warning(),
        ),
        (Some(error), _, _) => (format!("ERROR · {error}"), theme.error()),
        (_, Some(result), _) if result.gate_reason.is_some() => (
            format!(
                "GATED · {} · no search or context injection",
                result.gate_reason.as_deref().unwrap_or("unknown")
            ),
            theme.warning(),
        ),
        (_, Some(result), _) => (
            format!(
                "{} · query: {}\nhits {} · context {} · {}ms",
                result.mode.label(),
                result.query,
                result.hits.len(),
                if result.context.is_some() { "yes" } else { "no" },
                result.latency_ms
            ),
            Style::default(),
        ),
        _ => (
            "Type a real prompt or keyword query. RAG shows gate → extraction → filtered hits → injected context; BM25 shows raw retrieval.".to_string(),
            Style::default(),
        ),
    };
    frame.render_widget(
        Paragraph::new(status_text)
            .style(status_style)
            .wrap(Wrap { trim: false })
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(theme.border())
                    .title_style(theme.heading())
                    .title(" execution "),
            ),
        rows[1],
    );

    if area.width >= WIDE_WIDTH {
        let columns = Layout::horizontal([Constraint::Percentage(42), Constraint::Percentage(58)])
            .split(rows[2]);
        draw_hits(frame, columns[0], state, theme);
        draw_detail(frame, columns[1], state, theme);
    } else {
        let body = Layout::vertical([Constraint::Percentage(45), Constraint::Percentage(55)])
            .split(rows[2]);
        draw_hits(frame, body[0], state, theme);
        draw_detail(frame, body[1], state, theme);
    }
}

fn draw_hits(
    frame: &mut Frame,
    area: Rect,
    state: &SimulatorState,
    theme: crate::dashboard_theme::Theme,
) {
    let items = state
        .result
        .as_ref()
        .map(|result| {
            result
                .hits
                .iter()
                .enumerate()
                .map(|(index, hit)| {
                    let path = hit.file.strip_prefix("kmd://").unwrap_or(&hit.file);
                    ListItem::new(Line::from(vec![
                        Span::styled(format!("{:>2}. ", index + 1), theme.muted()),
                        Span::raw(format!("{:.2} {path}", hit.score)),
                    ]))
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let mut list_state = ListState::default();
    if !items.is_empty() {
        list_state.select(Some(state.selected_hit.min(items.len() - 1)));
    }
    let list = List::new(items)
        .highlight_symbol("> ")
        .highlight_style(theme.selected())
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title_style(theme.heading())
                .title(" hits · Alt-j/k selects "),
        );
    frame.render_stateful_widget(list, area, &mut list_state);
}

fn draw_detail(
    frame: &mut Frame,
    area: Rect,
    state: &SimulatorState,
    theme: crate::dashboard_theme::Theme,
) {
    let (title, content) = match state.result.as_ref() {
        Some(result) if result.mode == SimulatorMode::Rag => (
            " injected context · PgUp/PgDn scroll ",
            result
                .context
                .clone()
                .unwrap_or_else(|| "(no context would be injected)".to_string()),
        ),
        Some(result) => {
            let detail = result
                .hits
                .get(state.selected_hit)
                .map(|hit| {
                    format!(
                        "{}\nscore: {:.3}\ntitle: {}\ncontext: {}\n\n{}",
                        hit.file,
                        hit.score,
                        hit.title,
                        hit.context.as_deref().unwrap_or("-"),
                        hit.snippet.as_deref().unwrap_or("(no snippet)")
                    )
                })
                .unwrap_or_else(|| "(no search hits)".to_string());
            (" selected hit · PgUp/PgDn scroll ", detail)
        }
        None => (
            " result detail ",
            "Run a query to inspect results.".to_string(),
        ),
    };
    frame.render_widget(
        Paragraph::new(content)
            .scroll((state.detail_scroll, 0))
            .wrap(Wrap { trim: false })
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(theme.border())
                    .title_style(theme.heading())
                    .title(title),
            ),
        area,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_toggle_round_trips() {
        assert_eq!(SimulatorMode::Rag.toggle(), SimulatorMode::Search);
        assert_eq!(SimulatorMode::Search.toggle(), SimulatorMode::Rag);
    }

    #[test]
    fn narrow_layout_renders_floor_message() {
        let backend = ratatui::backend::TestBackend::new(50, 12);
        let mut terminal = Terminal::new(backend).unwrap();
        let state = SimulatorState::new();
        terminal
            .draw(|frame| {
                draw(
                    frame,
                    frame.area(),
                    &state,
                    crate::dashboard_theme::Theme::plain(),
                )
            })
            .unwrap();
        let output = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(output.contains("needs at least"));
    }

    #[test]
    fn error_status_renders_in_red() {
        let backend = ratatui::backend::TestBackend::new(100, 30);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut state = SimulatorState::new();
        state.error = Some("search failed".to_string());

        terminal
            .draw(|frame| {
                draw(
                    frame,
                    frame.area(),
                    &state,
                    crate::dashboard_theme::Theme::colored(),
                )
            })
            .unwrap();

        assert!(
            terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .any(|cell| cell.symbol() == "E" && cell.fg == Color::Red)
        );
    }
}
