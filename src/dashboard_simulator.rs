//! Interactive query simulator embedded in the operations dashboard.
//!
//! Input is split into a command mode and a typing mode so every action stays on
//! a plain key. Terminal-reserved chords (Ctrl/Alt) are never bound — inside a
//! multiplexer those collide with the terminal or the tmux prefix.

use crate::palette;
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

/// Ctrl/Alt chords belong to the terminal and the multiplexer prefix. Leaving
/// them unhandled keeps `Ctrl-C`, `Ctrl-Z`, and tmux bindings working while the
/// dashboard is focused.
pub fn is_reserved_chord(key: &KeyEvent) -> bool {
    key.modifiers
        .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT | KeyModifiers::SUPER)
}

pub struct SimulatorState {
    pub input: String,
    pub mode: SimulatorMode,
    pub result: Option<SimulatorResult>,
    pub error: Option<String>,
    pub running: bool,
    pub selected_hit: usize,
    pub detail_scroll: u16,
    typing: bool,
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
            typing: false,
            history: sim::prompt_history(),
            history_idx: None,
            generation: 0,
            tx,
            rx,
        }
    }

    pub fn typing(&self) -> bool {
        self.typing
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

    /// Keys while composing a query. Esc returns to command mode; every
    /// printable key is text, so no letter is stolen from the query.
    pub fn handle_typing_key(&mut self, key: KeyEvent) {
        if is_reserved_chord(&key) {
            return;
        }
        match key.code {
            KeyCode::Esc => self.typing = false,
            KeyCode::Enter => self.run(),
            KeyCode::Backspace => {
                self.input.pop();
                self.history_idx = None;
            }
            KeyCode::Up => self.previous_history(),
            KeyCode::Down => self.next_history(),
            KeyCode::Char(value) => {
                self.input.push(value);
                self.history_idx = None;
            }
            _ => {}
        }
    }

    /// Keys while not composing a query.
    pub fn handle_command_key(&mut self, key: KeyEvent) {
        if is_reserved_chord(&key) {
            return;
        }
        match key.code {
            KeyCode::Char('i') | KeyCode::Char('/') => self.typing = true,
            KeyCode::Enter => self.run(),
            KeyCode::Char('m') => {
                self.mode = self.mode.toggle();
                self.result = None;
                self.error = None;
            }
            KeyCode::Char('x') => {
                self.input.clear();
                self.result = None;
                self.error = None;
                self.history_idx = None;
            }
            KeyCode::Char('p') | KeyCode::Up => self.previous_history(),
            KeyCode::Char('n') | KeyCode::Down => self.next_history(),
            KeyCode::Char('j') => self.next_hit(),
            KeyCode::Char('k') => self.previous_hit(),
            KeyCode::Char('J') | KeyCode::PageDown => {
                self.detail_scroll = self.detail_scroll.saturating_add(8)
            }
            KeyCode::Char('K') | KeyCode::PageUp => {
                self.detail_scroll = self.detail_scroll.saturating_sub(8)
            }
            KeyCode::Home => self.detail_scroll = 0,
            _ => {}
        }
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

pub fn draw(frame: &mut Frame, area: Rect, state: &SimulatorState) {
    if area.width < MIN_WIDTH || area.height < MIN_HEIGHT {
        let message = Paragraph::new(format!(
            "Simulator needs at least {MIN_WIDTH}x{MIN_HEIGHT}. Current area: {}x{}",
            area.width, area.height
        ))
        .style(palette::warn())
        .wrap(Wrap { trim: false })
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(palette::warn())
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

    draw_query(frame, rows[0], state);
    draw_status(frame, rows[1], state);

    if area.width >= WIDE_WIDTH {
        let columns = Layout::horizontal([Constraint::Percentage(42), Constraint::Percentage(58)])
            .split(rows[2]);
        draw_hits(frame, columns[0], state);
        draw_detail(frame, columns[1], state);
    } else {
        let body = Layout::vertical([Constraint::Percentage(45), Constraint::Percentage(55)])
            .split(rows[2]);
        draw_hits(frame, body[0], state);
        draw_detail(frame, body[1], state);
    }
}

fn draw_query(frame: &mut Frame, area: Rect, state: &SimulatorState) {
    let mut title = vec![
        Span::styled(" query ".to_string(), palette::heading()),
        Span::styled("· ".to_string(), palette::muted()),
        Span::styled(state.mode.label().to_string(), palette::accent()),
        Span::styled(" · ".to_string(), palette::muted()),
    ];
    if state.typing() {
        title.push(Span::styled("TYPING".to_string(), palette::success()));
        title.push(Span::styled(
            " Esc stops · Enter runs ".to_string(),
            palette::muted(),
        ));
    } else {
        title.push(Span::styled("COMMAND".to_string(), palette::label()));
        title.push(Span::styled(
            " i types · Enter runs ".to_string(),
            palette::muted(),
        ));
    }
    if state.running {
        title.push(Span::styled("· RUNNING ".to_string(), palette::warn()));
    }

    let body = if state.input.is_empty() {
        Line::from(Span::styled(
            "press i to type a prompt or keyword query".to_string(),
            palette::muted(),
        ))
    } else {
        Line::from(Span::styled(state.input.clone(), palette::value()))
    };
    frame.render_widget(
        Paragraph::new(body).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(if state.typing() {
                    palette::border_focus()
                } else {
                    palette::border()
                })
                .title(Line::from(title)),
        ),
        area,
    );
}

fn draw_status(frame: &mut Frame, area: Rect, state: &SimulatorState) {
    let lines = match (&state.error, &state.result, state.running) {
        (_, _, true) => vec![Line::from(Span::styled(
            "Running against the local kmd index…".to_string(),
            palette::warn(),
        ))],
        (Some(error), _, _) => vec![Line::from(vec![
            Span::styled("ERROR ".to_string(), palette::failure()),
            Span::styled(error.clone(), palette::failure()),
        ])],
        (_, Some(result), _) if result.gate_reason.is_some() => vec![
            Line::from(vec![
                Span::styled("GATED ".to_string(), palette::warn()),
                Span::styled(
                    result.gate_reason.clone().unwrap_or_default(),
                    palette::warn(),
                ),
            ]),
            Line::from(Span::styled(
                "the hook would skip search and inject nothing".to_string(),
                palette::muted(),
            )),
        ],
        (_, Some(result), _) => {
            let injected = result.context.is_some();
            let mut lines = vec![
                Line::from(vec![
                    Span::styled(result.mode.label().to_string(), palette::accent()),
                    Span::styled(" · query: ".to_string(), palette::muted()),
                    Span::styled(result.query.clone(), palette::value()),
                ]),
                Line::from(vec![
                    Span::styled("hits ".to_string(), palette::muted()),
                    Span::styled(
                        result.hits.len().to_string(),
                        palette::state(!result.hits.is_empty()),
                    ),
                    Span::styled(" · context ".to_string(), palette::muted()),
                    Span::styled(
                        if injected { "yes" } else { "no" }.to_string(),
                        if injected {
                            palette::success()
                        } else {
                            palette::muted()
                        },
                    ),
                    Span::styled(" · ".to_string(), palette::muted()),
                    Span::styled(format!("{}ms", result.latency_ms), palette::value()),
                ]),
            ];
            if !result.axes.is_empty() {
                let mut spans = vec![Span::styled("axes ".to_string(), palette::muted())];
                for (index, (axis, count)) in result.axes.iter().enumerate() {
                    if index > 0 {
                        spans.push(Span::styled(" · ".to_string(), palette::muted()));
                    }
                    spans.push(Span::styled(format!("{axis} "), palette::accent()));
                    spans.push(Span::styled(count.to_string(), palette::state(*count > 0)));
                }
                lines.push(Line::from(spans));
            }
            lines
        }
        _ => vec![
            Line::from(Span::styled(
                "RAG pipeline: gate → extracted query → filtered hits → injected context"
                    .to_string(),
                palette::muted(),
            )),
            Line::from(Span::styled(
                "BM25 search: raw retrieval across every collection".to_string(),
                palette::muted(),
            )),
            Line::from(Span::styled(
                "Global: session / knowledge / project, quota'd per axis · m switches mode"
                    .to_string(),
                palette::muted(),
            )),
        ],
    };
    frame.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: false }).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(palette::border())
                .title(Span::styled(" execution ".to_string(), palette::heading())),
        ),
        area,
    );
}

/// hits 리스트에 축 헤더를 섞은 결과. 선택은 hit 인덱스로 유지되므로 화면 행과
/// hit 인덱스를 잇는 매핑을 함께 돌려준다.
struct HitRows<'a> {
    items: Vec<ListItem<'a>>,
    /// 화면 행 → hit 인덱스. 헤더 행은 None.
    row_to_hit: Vec<Option<usize>>,
}

fn hit_rows(state: &SimulatorState) -> HitRows<'static> {
    let Some(result) = state.result.as_ref() else {
        return HitRows {
            items: Vec::new(),
            row_to_hit: Vec::new(),
        };
    };

    // 축 경계를 hit 인덱스로 바꿔 둔다: 그 인덱스 앞에 헤더 행이 들어간다.
    let mut header_at: Vec<(usize, String, usize)> = Vec::new();
    let mut cursor = 0usize;
    for (axis, count) in &result.axes {
        header_at.push((cursor, axis.clone(), *count));
        cursor += count;
    }

    let mut items = Vec::new();
    let mut row_to_hit = Vec::new();
    for (index, hit) in result.hits.iter().enumerate() {
        if let Some((_, axis, count)) = header_at.iter().find(|(at, _, _)| *at == index) {
            items.push(ListItem::new(Line::from(vec![
                Span::styled(format!("── {axis} "), palette::heading()),
                Span::styled(format!("({count})"), palette::muted()),
            ])));
            row_to_hit.push(None);
        }
        let path = hit.file.strip_prefix("kmd://").unwrap_or(&hit.file);
        let (collection, rest) = path.split_once('/').unwrap_or(("", path));
        items.push(ListItem::new(Line::from(vec![
            Span::styled(format!("{:>2}. ", index + 1), palette::muted()),
            Span::styled(format!("{:>6.2} ", hit.score), palette::success()),
            Span::styled(format!("{collection}/"), palette::accent()),
            Span::styled(rest.to_string(), palette::value()),
        ])));
        row_to_hit.push(Some(index));
    }

    // 결과가 0건인 축도 보여준다 — "검색은 됐는데 없음"과 "그 축이 인덱싱되지
    // 않음"은 다른 상태다.
    for (at, axis, count) in &header_at {
        if *count == 0 && *at >= result.hits.len() {
            items.push(ListItem::new(Line::from(vec![
                Span::styled(format!("── {axis} "), palette::heading()),
                Span::styled("(0)".to_string(), palette::muted()),
            ])));
            row_to_hit.push(None);
        }
    }

    HitRows { items, row_to_hit }
}

fn draw_hits(frame: &mut Frame, area: Rect, state: &SimulatorState) {
    let HitRows { items, row_to_hit } = hit_rows(state);
    let mut list_state = ListState::default();
    if !items.is_empty() {
        // 선택은 hit 인덱스 기준이므로 해당 hit이 놓인 화면 행을 찾는다.
        let row = row_to_hit
            .iter()
            .position(|slot| *slot == Some(state.selected_hit))
            .unwrap_or(0);
        list_state.select(Some(row));
    }
    let title = if state.result.as_ref().is_some_and(|r| !r.axes.is_empty()) {
        " hits · j/k selects · axes quota'd separately "
    } else {
        " hits · j/k selects "
    };
    let list = List::new(items)
        .highlight_symbol("> ")
        .highlight_style(palette::selection())
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(palette::border())
                .title(Span::styled(title.to_string(), palette::heading())),
        );
    frame.render_stateful_widget(list, area, &mut list_state);
}

fn draw_detail(frame: &mut Frame, area: Rect, state: &SimulatorState) {
    let (title, lines) = match state.result.as_ref() {
        Some(result) if result.mode == SimulatorMode::Rag => (
            " injected context ",
            match result.context.as_ref() {
                Some(context) => context
                    .lines()
                    .map(|line| {
                        let style = if line.starts_with("[KMD]") || line.starts_with("[QMD]") {
                            palette::accent()
                        } else if line.starts_with('<') || line == "---" {
                            palette::muted()
                        } else {
                            palette::value()
                        };
                        Line::from(Span::styled(line.to_string(), style))
                    })
                    .collect::<Vec<_>>(),
                None => vec![Line::from(Span::styled(
                    "(no context would be injected)".to_string(),
                    palette::muted(),
                ))],
            },
        ),
        Some(result) => (
            " selected hit ",
            match result.hits.get(state.selected_hit) {
                Some(hit) => {
                    let mut lines = vec![
                        Line::from(Span::styled(hit.file.clone(), palette::accent())),
                        Line::from(vec![
                            Span::styled("score:   ".to_string(), palette::label()),
                            Span::styled(format!("{:.3}", hit.score), palette::success()),
                        ]),
                        Line::from(vec![
                            Span::styled("title:   ".to_string(), palette::label()),
                            Span::styled(hit.title.clone(), palette::value()),
                        ]),
                        Line::from(vec![
                            Span::styled("context: ".to_string(), palette::label()),
                            Span::styled(
                                hit.context.clone().unwrap_or_else(|| "-".to_string()),
                                palette::muted(),
                            ),
                        ]),
                        Line::default(),
                    ];
                    lines.extend(
                        hit.snippet
                            .as_deref()
                            .unwrap_or("(no snippet)")
                            .lines()
                            .map(|line| {
                                Line::from(Span::styled(line.to_string(), palette::value()))
                            })
                            .collect::<Vec<_>>(),
                    );
                    lines
                }
                None => vec![Line::from(Span::styled(
                    "(no search hits)".to_string(),
                    palette::muted(),
                ))],
            },
        ),
        None => (
            " result detail ",
            vec![Line::from(Span::styled(
                "Run a query to inspect results.".to_string(),
                palette::muted(),
            ))],
        ),
    };
    frame.render_widget(
        Paragraph::new(lines)
            .scroll((state.detail_scroll, 0))
            .wrap(Wrap { trim: false })
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(palette::border())
                    .title(Line::from(vec![
                        Span::styled(title.to_string(), palette::heading()),
                        Span::styled("· J/K scrolls ".to_string(), palette::muted()),
                    ])),
            ),
        area,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyModifiers;

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn mode_toggle_round_trips() {
        assert_eq!(SimulatorMode::Rag.toggle(), SimulatorMode::Search);
        assert_eq!(SimulatorMode::Search.toggle(), SimulatorMode::Global);
        assert_eq!(SimulatorMode::Global.toggle(), SimulatorMode::Rag);
    }

    fn hit(file: &str) -> crate::bm25::SearchHit {
        crate::bm25::SearchHit {
            docid: "#000001".into(),
            score: 1.0,
            file: file.into(),
            title: "t".into(),
            context: None,
            snippet: None,
        }
    }

    fn global_result(axes: Vec<(&str, usize)>) -> SimulatorResult {
        let mut hits = Vec::new();
        for (axis, count) in &axes {
            for i in 0..*count {
                hits.push(hit(&format!("kmd://{axis}/doc{i}.md")));
            }
        }
        SimulatorResult {
            mode: SimulatorMode::Global,
            query: "q".into(),
            gate_reason: None,
            hits,
            context: None,
            latency_ms: 1,
            axes: axes
                .into_iter()
                .map(|(a, c)| (a.to_string(), c))
                .collect(),
        }
    }

    /// 축 헤더가 들어가도 j/k 선택은 hit 인덱스를 가리켜야 한다.
    #[test]
    fn axis_headers_do_not_shift_hit_selection() {
        let mut state = SimulatorState::new();
        state.result = Some(global_result(vec![("session", 2), ("project", 2)]));

        let rows = hit_rows(&state);
        // 헤더 2 + hit 4
        assert_eq!(rows.items.len(), 6);
        assert_eq!(
            rows.row_to_hit,
            vec![None, Some(0), Some(1), None, Some(2), Some(3)],
        );
    }

    /// 결과가 0건인 축도 화면에 남아야 한다 — "없음"과 "인덱싱 안 됨"은 다르다.
    #[test]
    fn empty_axis_still_renders_a_header() {
        let mut state = SimulatorState::new();
        state.result = Some(global_result(vec![("session", 1), ("project", 0)]));

        let rows = hit_rows(&state);
        // session 헤더 + hit 1개 + 빈 project 헤더
        assert_eq!(rows.items.len(), 3);
        assert_eq!(rows.row_to_hit, vec![None, Some(0), None]);
    }

    /// 축이 없는 모드(RAG/BM25)는 헤더 없이 평평하게 그린다.
    #[test]
    fn non_global_modes_render_without_headers() {
        let mut state = SimulatorState::new();
        let mut result = global_result(vec![("session", 3)]);
        result.mode = SimulatorMode::Search;
        result.axes.clear();
        state.result = Some(result);

        let rows = hit_rows(&state);
        assert_eq!(rows.items.len(), 3);
        assert_eq!(rows.row_to_hit, vec![Some(0), Some(1), Some(2)]);
    }

    #[test]
    fn typing_mode_captures_command_letters_as_text() {
        let mut state = SimulatorState::new();
        state.handle_command_key(press(KeyCode::Char('i')));
        assert!(state.typing());

        for value in "mxjq".chars() {
            state.handle_typing_key(press(KeyCode::Char(value)));
        }
        assert_eq!(state.input, "mxjq");
        assert_eq!(state.mode, SimulatorMode::Rag);

        state.handle_typing_key(press(KeyCode::Esc));
        assert!(!state.typing());
    }

    #[test]
    fn command_mode_toggles_and_clears_without_modifiers() {
        let mut state = SimulatorState::new();
        state.input = "keep".to_string();

        state.handle_command_key(press(KeyCode::Char('m')));
        assert_eq!(state.mode, SimulatorMode::Search);

        state.handle_command_key(press(KeyCode::Char('x')));
        assert!(state.input.is_empty());
    }

    #[test]
    fn command_mode_scrolls_detail_with_shifted_letters() {
        let mut state = SimulatorState::new();
        state.handle_command_key(press(KeyCode::Char('J')));
        assert_eq!(state.detail_scroll, 8);

        state.handle_command_key(press(KeyCode::Char('K')));
        assert_eq!(state.detail_scroll, 0);
    }

    #[test]
    fn narrow_layout_renders_floor_message() {
        let backend = ratatui::backend::TestBackend::new(50, 12);
        let mut terminal = Terminal::new(backend).unwrap();
        let state = SimulatorState::new();
        terminal
            .draw(|frame| draw(frame, frame.area(), &state))
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
}
