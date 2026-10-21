//! Drag selection tests. Every press and drag lands on a cell found in the
//! drawn buffer, never on a coordinate from the code's own layout math.

use super::key_tests::empty_snapshot;
use super::*;
use crossterm::event::{KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use ratatui::buffer::Buffer;
use ratatui::style::Modifier;

const W: u16 = 120;
const H: u16 = 24;

fn render(app: &App) -> Buffer {
    let mut terminal = Terminal::new(ratatui::backend::TestBackend::new(W, H)).unwrap();
    terminal.draw(|frame| draw(frame, app)).unwrap();
    terminal.backend().buffer().clone()
}

fn row_text(b: &Buffer, y: u16) -> String {
    (0..b.area.width).map(|x| b[(x, y)].symbol()).collect()
}

/// The cell where `text` starts on screen (ASCII needles only).
fn find(b: &Buffer, text: &str) -> (u16, u16) {
    for y in 0..b.area.height {
        let cells: Vec<&str> = (0..b.area.width).map(|x| b[(x, y)].symbol()).collect();
        let want: Vec<String> = text.chars().map(String::from).collect();
        for x in 0..cells.len().saturating_sub(want.len() - 1) {
            if want.iter().enumerate().all(|(i, c)| cells[x + i] == c) {
                return (x as u16, y);
            }
        }
    }
    panic!("{text:?} is not on screen");
}

fn ev(kind: MouseEventKind, (x, y): (u16, u16)) -> MouseEvent {
    MouseEvent {
        kind,
        column: x,
        row: y,
        modifiers: KeyModifiers::NONE,
    }
}
fn down(at: (u16, u16)) -> MouseEvent {
    ev(MouseEventKind::Down(MouseButton::Left), at)
}
fn drag(at: (u16, u16)) -> MouseEvent {
    ev(MouseEventKind::Drag(MouseButton::Left), at)
}
fn up(at: (u16, u16)) -> MouseEvent {
    ev(MouseEventKind::Up(MouseButton::Left), at)
}

/// Feed events as the event loop does, redrawing before each one so the
/// recorded panes are the last frame's. Returns the last text to copy.
fn gesture(app: &mut App, events: &[MouseEvent], now: Instant) -> Option<String> {
    let mut copy = None;
    for e in events {
        render(app);
        copy = app.on_mouse(*e, now).or(copy);
    }
    copy
}

fn session(index: usize, queries: usize) -> RagSession {
    RagSession {
        session_id: format!("s{index:07}-x"),
        latest_ts: format!("{}", 1000 - index),
        queries: (0..queries)
            .map(|q| RagQueryRecord {
                ts: format!("{}", 1000 - index),
                query: format!("query {index}.{q:02}"),
                injected: 0,
                hits: Vec::new(),
                latency_ms: 1,
            })
            .collect(),
    }
}

fn app_with(sessions: usize, queries: usize) -> App {
    let mut snapshot = empty_snapshot();
    snapshot.rag.sessions = (0..sessions).map(|i| session(i, queries)).collect();
    App::new(
        snapshot,
        crate::dashboard_simulator::SimulatorState::without_history(),
    )
}

/// The detail pane's rows holding a query line, top first. Only cells right
/// of the detail's "Session" title count: the Sessions list repeats the
/// latest query as each row's summary.
fn query_rows(b: &Buffer) -> Vec<(u16, String)> {
    // The detail starts right of the two borders where the panes meet.
    let detail_x = find(b, "││").0 as usize + 2;
    (0..b.area.height)
        .filter_map(|y| {
            let row: String = (detail_x as u16..b.area.width)
                .map(|x| b[(x, y)].symbol())
                .collect();
            let at = row.find("query 0.")?;
            Some((y, row[at..at + 10].to_string()))
        })
        .collect()
}

#[test]
fn a_drag_from_the_sessions_list_toward_the_detail_stays_in_the_list() {
    let mut app = app_with(4, 3);
    let b = render(&app);
    let from = find(&b, "s0000001");
    // In the detail pane, on the row of the next session.
    let into = find(&b, "latest query:");
    let copy = gesture(
        &mut app,
        &[down(from), drag(into), up(into)],
        Instant::now(),
    )
    .unwrap();
    assert!(copy.contains("s0000001"), "{copy:?}");
    assert!(copy.contains("s0000002"), "{copy:?}");
    assert!(!copy.contains("latest"), "detail text leaked: {copy:?}");
    assert!(!copy.contains('│'), "border leaked: {copy:?}");

    let painted = render(&app);
    let (_, y) = find(&painted, "s0000002");
    let border_x = (0..from.0)
        .rev()
        .find(|&x| painted[(x, y)].symbol() == "│")
        .expect("the list's left border");
    let right_border = (from.0..W)
        .find(|&x| painted[(x, y)].symbol() == "│")
        .expect("the list's right border");
    assert!(painted[(from.0, y)].modifier.contains(Modifier::REVERSED));
    for x in (0..=border_x).chain(right_border..W) {
        assert!(
            !painted[(x, y)].modifier.contains(Modifier::REVERSED),
            "highlight at {x},{y} outside the list: {:?}",
            row_text(&painted, y)
        );
    }
}

#[test]
fn a_drag_past_the_detail_bottom_scrolls_and_keeps_scrolled_out_lines() {
    let mut app = app_with(1, 40);
    let b = render(&app);
    let from = find(&b, "Session s0000000");
    let below = (from.0 + 2, H + 3);
    let now = Instant::now();
    gesture(
        &mut app,
        &[down(from), drag(below), drag(below), drag(below)],
        now,
    );
    assert_eq!(app.scroll, 3, "three drag steps below scroll three lines");
    let b = render(&app);
    let on_screen = (0..H).any(|y| row_text(&b, y).contains("Session s0000000"));
    assert!(!on_screen, "the first line scrolled out");
    let (last_y, last) = query_rows(&b).last().cloned().expect("a query line");
    // Back inside the pane: the release keeps the focus of the last drag.
    let end = (from.0 + 9, last_y);
    let copy = gesture(&mut app, &[drag(end), up(end)], now).unwrap();
    assert!(copy.starts_with("Session s0000000\nactivity"), "{copy:?}");
    assert!(copy.ends_with(&last), "{copy:?}");
    assert!(
        copy.lines().count() as u16 > last_y - from.1 + 1,
        "the copy holds the scrolled-out lines too: {copy:?}"
    );
}

#[test]
fn a_press_without_motion_still_selects_the_session() {
    let mut app = app_with(4, 1);
    let b = render(&app);
    let at = find(&b, "s0000002");
    let copy = gesture(&mut app, &[down(at), up(at)], Instant::now());
    assert_eq!(copy, None);
    assert_eq!(app.selected_session, 2);
    assert!(!app.selection.active());
}

#[test]
fn a_second_click_on_a_session_stays_a_click() {
    let mut app = app_with(4, 1);
    let b = render(&app);
    let at = find(&b, "s0000002");
    let t = Instant::now();
    gesture(&mut app, &[down(at), up(at)], t);
    app.selected_session = 0;
    gesture(
        &mut app,
        &[down(at)],
        t + std::time::Duration::from_millis(100),
    );
    assert_eq!(
        app.selected_session, 2,
        "the list's second click reaches the app"
    );
    assert!(!app.selection.active());
}

#[test]
fn a_double_click_in_the_detail_copies_a_word() {
    let mut app = app_with(1, 3);
    let b = render(&app);
    let at = find(&b, "Session s0000000");
    let word = (at.0 + 9, at.1);
    let t = Instant::now();
    gesture(&mut app, &[down(word), up(word)], t);
    let copy = gesture(
        &mut app,
        &[down(word)],
        t + std::time::Duration::from_millis(100),
    );
    assert_eq!(copy.as_deref(), Some("s0000000"));
}

#[test]
fn a_key_clears_the_selection() {
    let mut app = app_with(1, 3);
    let b = render(&app);
    let from = find(&b, "Session s0000000");
    let to = find(&b, "origin:");
    gesture(&mut app, &[down(from), drag(to), up(to)], Instant::now());
    assert!(app.selection.active());
    handle_key(
        &mut app,
        crossterm::event::KeyEvent::new(crossterm::event::KeyCode::Char('j'), KeyModifiers::NONE),
    )
    .unwrap();
    assert!(!app.selection.active());
}

#[test]
fn with_help_open_the_sessions_list_is_not_selectable() {
    let mut app = app_with(4, 1);
    app.show_help = true;
    let b = render(&app);
    let at = find(&b, "s0000000"); // left of the help popup
    let copy = gesture(
        &mut app,
        &[down(at), drag((at.0 + 4, at.1 + 1))],
        Instant::now(),
    );
    assert_eq!(copy, None);
    assert!(!app.selection.active(), "the base under help was selected");
}

#[test]
fn a_drag_inside_help_copies_help_text() {
    let mut app = app_with(1, 1);
    app.show_help = true;
    let b = render(&app);
    let (x, y) = find(&b, " keys ");
    let from = (x + 2, y + 2);
    let to = (W - 1, y + 3);
    let copy = gesture(&mut app, &[down(from), drag(to), up(to)], Instant::now()).unwrap();
    assert!(!copy.is_empty());
    assert!(!copy.contains('│'), "border leaked: {copy:?}");
    assert!(!copy.contains("s0000000"), "base leaked: {copy:?}");
}
