//! Interactive query simulator embedded in the operations dashboard.
//!
//! Input is split into a command mode and a typing mode so every action stays on
//! a plain key. Which key does what lives in `dashboard_keys`; this module
//! exposes one method per action and the text-input fallback.

use crate::dashboard_keys::{Action, Keymap};
use crate::dashboard_worker::Background;
use crate::sim::{self, SimulatorMode, SimulatorResult};
use crate::theme::Theme;
use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};

const SEARCH_LIMIT: usize = 10;
const MIN_WIDTH: u16 = 60;
const MIN_HEIGHT: u16 = 15;
const WIDE_WIDTH: u16 = 96;

pub struct SimulatorState {
    pub input: String,
    pub mode: SimulatorMode,
    pub result: Option<SimulatorResult>,
    pub error: Option<String>,
    pub selected_hit: usize,
    pub detail_scroll: u16,
    /// Largest useful `detail_scroll` for the detail pane drawn last (layout
    /// feedback from draw, so `K` can scroll back up after overshooting).
    detail_max: std::cell::Cell<u16>,
    /// Hit regions recorded by `draw`, read by the dashboard's mouse handler.
    pub layout: std::cell::RefCell<crate::dashboard_layout::SimLayout>,
    typing: bool,
    history: Vec<String>,
    history_idx: Option<usize>,
    /// Generation-guarded query worker shared with the dashboard refresh and
    /// the standalone `kmd sim`.
    job: Background<SimulatorResult>,
}

impl SimulatorState {
    pub fn new() -> Self {
        Self::new_with_history(sim::prompt_history())
    }

    fn new_with_history(history: Vec<String>) -> Self {
        Self {
            input: String::new(),
            mode: SimulatorMode::Rag,
            result: None,
            error: None,
            selected_hit: 0,
            detail_scroll: 0,
            detail_max: std::cell::Cell::new(u16::MAX),
            layout: Default::default(),
            typing: false,
            history,
            history_idx: None,
            job: Background::new(),
        }
    }

    /// A simulator that does not read ~/.claude/history.jsonl, for hermetic tests.
    #[cfg(test)]
    pub fn without_history() -> Self {
        Self::new_with_history(Vec::new())
    }

    pub fn typing(&self) -> bool {
        self.typing
    }

    pub fn running(&self) -> bool {
        self.job.in_flight()
    }

    /// Runs the current query. A query still running is superseded, not waited
    /// on: Enter after editing used to be ignored while the old query ran, so
    /// the result shown belonged to text no longer in the box.
    pub fn run(&mut self) {
        let input = self.input.trim().to_string();
        if input.is_empty() {
            return;
        }
        let mode = self.mode;
        self.error = None;
        self.selected_hit = 0;
        self.detail_scroll = 0;
        self.job
            .replace(move || sim::execute(mode, &input, SEARCH_LIMIT));
    }

    /// Drops the in-flight query's result when it arrives. Used when the mode
    /// or the query changes under it, so a result for the old mode or a cleared
    /// query is never shown under the new state.
    fn invalidate(&mut self) {
        self.job.cancel();
    }

    pub fn poll(&mut self) -> bool {
        let Some(result) = self.job.poll() else {
            return false;
        };
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
        true
    }

    /// Text input while typing, for keys the keymap does not bind. Every
    /// printable key is text, so no letter is stolen from the query; Ctrl/Alt
    /// chords never reach here.
    pub fn type_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Backspace => {
                self.input.pop();
                self.history_idx = None;
            }
            KeyCode::Char(value) => {
                self.input.push(value);
                self.history_idx = None;
            }
            _ => {}
        }
    }

    pub fn start_typing(&mut self) {
        self.typing = true;
    }

    pub fn stop_typing(&mut self) {
        self.typing = false;
    }

    pub fn toggle_mode(&mut self) {
        self.invalidate();
        self.mode = self.mode.toggle();
        self.result = None;
        self.error = None;
    }

    /// Restores a mode saved by a previous dashboard run.
    pub fn set_mode(&mut self, mode: SimulatorMode) {
        if self.mode != mode {
            self.invalidate();
            self.mode = mode;
            self.result = None;
            self.error = None;
        }
    }

    pub fn clear(&mut self) {
        self.invalidate();
        self.input.clear();
        self.result = None;
        self.error = None;
        self.history_idx = None;
    }

    pub fn detail_down(&mut self) {
        self.detail_scroll = self
            .detail_scroll
            .min(self.detail_max.get())
            .saturating_add(8);
    }

    pub fn detail_up(&mut self) {
        self.detail_scroll = self
            .detail_scroll
            .min(self.detail_max.get())
            .saturating_sub(8);
    }

    pub fn detail_top(&mut self) {
        self.detail_scroll = 0;
    }

    pub fn previous_history(&mut self) {
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

    pub fn next_history(&mut self) {
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

    /// Selects hit `index` (a mouse click), as `next_hit` would land on it.
    pub fn select_hit(&mut self, index: usize) {
        let count = self.result.as_ref().map_or(0, |result| result.hits.len());
        if index < count {
            self.selected_hit = index;
            self.detail_scroll = 0;
        }
    }

    pub fn next_hit(&mut self) {
        let count = self.result.as_ref().map_or(0, |result| result.hits.len());
        if count > 0 {
            self.selected_hit = (self.selected_hit + 1).min(count - 1);
            self.detail_scroll = 0;
        }
    }

    pub fn previous_hit(&mut self) {
        self.selected_hit = self.selected_hit.saturating_sub(1);
        self.detail_scroll = 0;
    }
}

pub fn draw(frame: &mut Frame, area: Rect, state: &SimulatorState, theme: Theme, keys: &Keymap) {
    if area.width < MIN_WIDTH || area.height < MIN_HEIGHT {
        let message = Paragraph::new(format!(
            "Simulator needs at least {MIN_WIDTH}x{MIN_HEIGHT}. Current area: {}x{}",
            area.width, area.height
        ))
        .style(theme.warning())
        .wrap(Wrap { trim: false })
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.warning())
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
    *state.layout.borrow_mut() = crate::dashboard_layout::SimLayout {
        query: Some(rows[0]),
        ..Default::default()
    };

    draw_query(frame, rows[0], state, theme, keys);
    draw_status(frame, rows[1], state, theme, keys);

    if area.width >= WIDE_WIDTH {
        let columns = Layout::horizontal([Constraint::Percentage(42), Constraint::Percentage(58)])
            .split(rows[2]);
        draw_hits(frame, columns[0], state, theme);
        draw_detail(frame, columns[1], state, theme, keys);
    } else {
        let body = Layout::vertical([Constraint::Percentage(45), Constraint::Percentage(55)])
            .split(rows[2]);
        draw_hits(frame, body[0], state, theme);
        draw_detail(frame, body[1], state, theme, keys);
    }
}

fn draw_query(frame: &mut Frame, area: Rect, state: &SimulatorState, theme: Theme, keys: &Keymap) {
    let mut title = vec![
        Span::styled(" query ".to_string(), theme.heading()),
        Span::styled("· ".to_string(), theme.muted()),
        Span::styled(state.mode.label().to_string(), theme.kind()),
        Span::styled(" · ".to_string(), theme.muted()),
    ];
    if state.typing() {
        title.push(Span::styled("TYPING".to_string(), theme.success()));
        title.push(Span::styled(
            format!(
                " {} stops · {} runs ",
                keys.key_label(Action::TypingStop),
                keys.key_label(Action::TypingRun)
            ),
            theme.muted(),
        ));
    } else {
        title.push(Span::styled("COMMAND".to_string(), theme.label()));
        title.push(Span::styled(
            format!(
                " {} types · {} runs ",
                keys.key_label(Action::SimType),
                keys.key_label(Action::SimRun)
            ),
            theme.muted(),
        ));
    }
    if state.running() {
        title.push(Span::styled("· RUNNING ".to_string(), theme.warning()));
    }

    let body = if state.input.is_empty() {
        Line::from(Span::styled(
            format!(
                "press {} to type a prompt or keyword query",
                keys.key_label(Action::SimType)
            ),
            theme.muted(),
        ))
    } else {
        Line::from(Span::styled(state.input.clone(), theme.value()))
    };
    frame.render_widget(
        Paragraph::new(body).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(if state.typing() {
                    theme.border_focus()
                } else {
                    theme.border()
                })
                .title(Line::from(title)),
        ),
        area,
    );
}

fn draw_status(frame: &mut Frame, area: Rect, state: &SimulatorState, theme: Theme, keys: &Keymap) {
    let lines = match (&state.error, &state.result, state.running()) {
        (_, _, true) => vec![Line::from(Span::styled(
            "Running against the local kmd index…".to_string(),
            theme.warning(),
        ))],
        (Some(error), _, _) => vec![Line::from(vec![
            Span::styled("ERROR ".to_string(), theme.error()),
            Span::styled(error.clone(), theme.error()),
        ])],
        (_, Some(result), _) if result.gate_reason.is_some() => vec![
            Line::from(vec![
                Span::styled("GATED ".to_string(), theme.warning()),
                Span::styled(
                    result.gate_reason.clone().unwrap_or_default(),
                    theme.warning(),
                ),
            ]),
            Line::from(Span::styled(
                "the hook would skip search and inject nothing".to_string(),
                theme.muted(),
            )),
        ],
        (_, Some(result), _) => {
            let injected = result.context.is_some();
            let mut lines = vec![
                Line::from(vec![
                    Span::styled(result.mode.label().to_string(), theme.kind()),
                    Span::styled(" · query: ".to_string(), theme.muted()),
                    Span::styled(result.query.clone(), theme.value()),
                ]),
                Line::from(vec![
                    Span::styled("hits ".to_string(), theme.muted()),
                    Span::styled(
                        result.hits.len().to_string(),
                        theme.state(!result.hits.is_empty()),
                    ),
                    Span::styled(" · context ".to_string(), theme.muted()),
                    Span::styled(
                        if injected { "yes" } else { "no" }.to_string(),
                        if injected {
                            theme.success()
                        } else {
                            theme.muted()
                        },
                    ),
                    Span::styled(" · ".to_string(), theme.muted()),
                    Span::styled(format!("{}ms", result.latency_ms), theme.value()),
                ]),
            ];
            if !result.axes.is_empty() {
                let mut spans = vec![Span::styled("axes ".to_string(), theme.muted())];
                for (index, (axis, count)) in result.axes.iter().enumerate() {
                    if index > 0 {
                        spans.push(Span::styled(" · ".to_string(), theme.muted()));
                    }
                    spans.push(Span::styled(format!("{axis} "), theme.kind()));
                    spans.push(Span::styled(count.to_string(), theme.state(*count > 0)));
                }
                lines.push(Line::from(spans));
            }
            lines
        }
        _ => vec![
            Line::from(Span::styled(
                "RAG pipeline: gate → extracted query → filtered hits → injected context"
                    .to_string(),
                theme.muted(),
            )),
            Line::from(Span::styled(
                "BM25 search: raw retrieval across every collection".to_string(),
                theme.muted(),
            )),
            Line::from(Span::styled(
                format!(
                    "Global: session / knowledge / project, quota'd per axis · {} switches mode",
                    keys.key_label(Action::SimMode)
                ),
                theme.muted(),
            )),
        ],
    };
    frame.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: false }).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(Span::styled(" execution ".to_string(), theme.heading())),
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

fn hit_rows(state: &SimulatorState, theme: Theme) -> HitRows<'static> {
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
                Span::styled(format!("── {axis} "), theme.heading()),
                Span::styled(format!("({count})"), theme.muted()),
            ])));
            row_to_hit.push(None);
        }
        let path = hit.file.strip_prefix("kmd://").unwrap_or(&hit.file);
        let (collection, rest) = path.split_once('/').unwrap_or(("", path));
        items.push(ListItem::new(Line::from(vec![
            Span::styled(format!("{:>2}. ", index + 1), theme.muted()),
            Span::styled(format!("{:>6.2} ", hit.score), theme.success()),
            Span::styled(format!("{collection}/"), theme.kind()),
            Span::styled(rest.to_string(), theme.value()),
        ])));
        row_to_hit.push(Some(index));
    }

    // 결과가 0건인 축도 보여준다 — "검색은 됐는데 없음"과 "그 축이 인덱싱되지
    // 않음"은 다른 상태다.
    for (at, axis, count) in &header_at {
        if *count == 0 && *at >= result.hits.len() {
            items.push(ListItem::new(Line::from(vec![
                Span::styled(format!("── {axis} "), theme.heading()),
                Span::styled("(0)".to_string(), theme.muted()),
            ])));
            row_to_hit.push(None);
        }
    }

    HitRows { items, row_to_hit }
}

fn draw_hits(frame: &mut Frame, area: Rect, state: &SimulatorState, theme: Theme) {
    let HitRows { items, row_to_hit } = hit_rows(state, theme);
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
        .highlight_style(theme.selected())
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border())
                .title(Span::styled(title.to_string(), theme.heading())),
        );
    let inner = Block::default().borders(Borders::ALL).inner(area);
    frame.render_stateful_widget(list, area, &mut list_state);
    state.layout.borrow_mut().hits = Some(crate::dashboard_layout::ListHit {
        area: inner,
        first: list_state.offset(),
        row_height: 1,
        rows: row_to_hit,
    });
}

fn draw_detail(frame: &mut Frame, area: Rect, state: &SimulatorState, theme: Theme, keys: &Keymap) {
    state.layout.borrow_mut().detail = Some(area);
    let (title, lines) = match state.result.as_ref() {
        Some(result) if result.mode == SimulatorMode::Rag => (
            " injected context ",
            match result.context.as_ref() {
                Some(context) => context
                    .lines()
                    .map(|line| {
                        let style = if line.starts_with("[KMD]") || line.starts_with("[QMD]") {
                            theme.kind()
                        } else if line.starts_with('<') || line == "---" {
                            theme.muted()
                        } else {
                            theme.value()
                        };
                        Line::from(Span::styled(line.to_string(), style))
                    })
                    .collect::<Vec<_>>(),
                None => vec![Line::from(Span::styled(
                    "(no context would be injected)".to_string(),
                    theme.muted(),
                ))],
            },
        ),
        Some(result) => (
            " selected hit ",
            match result.hits.get(state.selected_hit) {
                Some(hit) => {
                    let mut lines = vec![
                        Line::from(Span::styled(hit.file.clone(), theme.kind())),
                        Line::from(vec![
                            Span::styled("score:   ".to_string(), theme.label()),
                            Span::styled(format!("{:.3}", hit.score), theme.success()),
                        ]),
                        Line::from(vec![
                            Span::styled("title:   ".to_string(), theme.label()),
                            Span::styled(hit.title.clone(), theme.value()),
                        ]),
                        Line::from(vec![
                            Span::styled("context: ".to_string(), theme.label()),
                            Span::styled(
                                hit.context.clone().unwrap_or_else(|| "-".to_string()),
                                theme.muted(),
                            ),
                        ]),
                        Line::default(),
                    ];
                    lines.extend(
                        hit.snippet
                            .as_deref()
                            .unwrap_or("(no snippet)")
                            .lines()
                            .map(|line| Line::from(Span::styled(line.to_string(), theme.value())))
                            .collect::<Vec<_>>(),
                    );
                    lines
                }
                None => vec![Line::from(Span::styled(
                    "(no search hits)".to_string(),
                    theme.muted(),
                ))],
            },
        ),
        None => (
            " result detail ",
            vec![Line::from(Span::styled(
                "Run a query to inspect results.".to_string(),
                theme.muted(),
            ))],
        ),
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme.border())
        .title(Line::from(vec![
            Span::styled(title.to_string(), theme.heading()),
            Span::styled(
                format!(
                    "· {}/{} scrolls ",
                    keys.key_label(Action::SimDetailDown),
                    keys.key_label(Action::SimDetailUp)
                ),
                theme.muted(),
            ),
        ]));
    let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
    let (scroll, max) =
        crate::dashboard_layout::clamp_scroll(&paragraph, block.inner(area), state.detail_scroll);
    state.detail_max.set(max);
    frame.render_widget(paragraph.scroll((scroll, 0)).block(block), area);
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
            abspath: None,
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
            axes: axes.into_iter().map(|(a, c)| (a.to_string(), c)).collect(),
        }
    }

    /// 축 헤더가 들어가도 j/k 선택은 hit 인덱스를 가리켜야 한다.
    #[test]
    fn axis_headers_do_not_shift_hit_selection() {
        let mut state = SimulatorState::new();
        state.result = Some(global_result(vec![("session", 2), ("project", 2)]));

        let rows = hit_rows(&state, Theme::colored());
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

        let rows = hit_rows(&state, Theme::colored());
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

        let rows = hit_rows(&state, Theme::colored());
        assert_eq!(rows.items.len(), 3);
        assert_eq!(rows.row_to_hit, vec![Some(0), Some(1), Some(2)]);
    }

    #[test]
    fn typing_mode_captures_command_letters_as_text() {
        let mut state = SimulatorState::new();
        state.start_typing();
        assert!(state.typing());

        for value in "mxjq".chars() {
            state.type_key(press(KeyCode::Char(value)));
        }
        assert_eq!(state.input, "mxjq");
        assert_eq!(state.mode, SimulatorMode::Rag);

        state.stop_typing();
        assert!(!state.typing());
    }

    #[test]
    fn command_mode_toggles_and_clears_without_modifiers() {
        let mut state = SimulatorState::new();
        state.input = "keep".to_string();

        state.toggle_mode();
        assert_eq!(state.mode, SimulatorMode::Search);

        state.clear();
        assert!(state.input.is_empty());
    }

    #[test]
    fn command_mode_scrolls_detail_with_shifted_letters() {
        let mut state = SimulatorState::new();
        state.detail_down();
        assert_eq!(state.detail_scroll, 8);

        state.detail_up();
        assert_eq!(state.detail_scroll, 0);
    }

    /// Starts a query that blocks until the returned sender fires, so a test
    /// can change state while it is in flight.
    fn start_blocked_query(state: &mut SimulatorState) -> std::sync::mpsc::Sender<()> {
        let (release, gate) = std::sync::mpsc::channel::<()>();
        state.job.replace(move || {
            gate.recv().ok();
            Ok(global_result(vec![("session", 1)]))
        });
        release
    }

    /// A result for the previous mode used to land after `m` and show under
    /// the new mode's title.
    #[test]
    fn mode_switch_drops_the_in_flight_result() {
        let mut state = SimulatorState::without_history();
        let release = start_blocked_query(&mut state);
        assert!(state.running());
        state.toggle_mode();
        assert!(!state.running());

        release.send(()).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(!state.poll());
        assert!(state.result.is_none());
    }

    #[test]
    fn clearing_the_query_drops_the_in_flight_result() {
        let mut state = SimulatorState::without_history();
        state.input = "old query".into();
        let release = start_blocked_query(&mut state);
        state.clear();

        release.send(()).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(!state.poll());
        assert!(state.result.is_none());
        assert!(!state.running());
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
                    Theme::plain(),
                    &Keymap::defaults(),
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
}
