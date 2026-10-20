//! Mouse tests. Every click lands on a cell found in the drawn buffer, not on
//! a coordinate recomputed with the code's own layout arithmetic, so a hit
//! test that drifts from what is on screen fails here.

use super::key_tests::empty_snapshot;
use super::*;
use crossterm::event::{KeyModifiers, MouseButton, MouseEvent, MouseEventKind};

const W: u16 = 120;
const H: u16 = 24;

fn render(app: &App) -> ratatui::buffer::Buffer {
    let mut terminal = Terminal::new(ratatui::backend::TestBackend::new(W, H)).unwrap();
    terminal.draw(|frame| draw(frame, app)).unwrap();
    terminal.backend().buffer().clone()
}

/// The cell where `text` starts on screen.
fn find(app: &App, text: &str) -> (u16, u16) {
    let buffer = render(app);
    let want: Vec<String> = text.chars().map(String::from).collect();
    for y in 0..buffer.area.height {
        for x in 0..buffer.area.width {
            let fits = want.iter().enumerate().all(|(i, ch)| {
                x + (i as u16) < buffer.area.width && buffer[(x + i as u16, y)].symbol() == ch
            });
            if fits {
                return (x, y);
            }
        }
    }
    panic!("{text:?} is not on screen");
}

fn event(kind: MouseEventKind, x: u16, y: u16) -> MouseEvent {
    MouseEvent {
        kind,
        column: x,
        row: y,
        modifiers: KeyModifiers::NONE,
    }
}

fn click(app: &mut App, x: u16, y: u16) {
    render(app); // the map is whatever the last frame drew
    app.handle_mouse(event(MouseEventKind::Down(MouseButton::Left), x, y));
}

fn wheel(app: &mut App, kind: MouseEventKind, x: u16, y: u16) {
    render(app);
    app.handle_mouse(event(kind, x, y));
}

fn session(index: usize, queries: usize) -> RagSession {
    RagSession {
        session_id: format!("s{index:07}-x"),
        latest_ts: format!("{}", 1000 - index),
        queries: (0..queries)
            .map(|q| RagQueryRecord {
                ts: format!("{}", 1000 - index),
                query: format!("query {index}.{q}"),
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

fn selected_id(app: &App) -> String {
    app.session_ids()[app.selected_session].clone()
}

#[test]
fn clicking_a_tab_switches_the_view() {
    let mut app = app_with(1, 1);
    let (x, y) = find(&app, "2:Operations");
    click(&mut app, x, y);
    assert_eq!(app.tab, OPERATIONS_TAB);
    let (x, y) = find(&app, "3:Simulator");
    click(&mut app, x, y);
    assert_eq!(app.tab, SIMULATOR_TAB);
}

#[test]
fn clicking_a_session_selects_it_even_when_the_list_is_scrolled() {
    let mut app = app_with(30, 1);
    // Push the selection down so ratatui scrolls the list; the click must
    // follow the offset it drew with.
    for _ in 0..20 {
        app.apply(Action::SessionNext);
    }
    let (x, y) = find(&app, "s0000018");
    click(&mut app, x, y);
    assert_eq!(selected_id(&app), "s0000018-x");
}

#[test]
fn clicking_a_sessions_summary_line_selects_that_session() {
    // Each row is two lines; a one-line count would select the next one.
    let mut app = app_with(5, 1);
    let (x, y) = find(&app, "s0000002");
    click(&mut app, x, y + 1);
    assert_eq!(selected_id(&app), "s0000002-x");
}

#[test]
fn the_wheel_over_sessions_moves_three() {
    let mut app = app_with(10, 1);
    let (x, y) = find(&app, "s0000000");
    wheel(&mut app, MouseEventKind::ScrollDown, x, y);
    assert_eq!(app.selected_session, 3);
    wheel(&mut app, MouseEventKind::ScrollUp, x, y);
    assert_eq!(app.selected_session, 0);
}

#[test]
fn the_wheel_over_the_detail_scrolls_it_without_moving_the_selection() {
    let mut app = app_with(3, 60);
    let (x, y) = find(&app, "kmd query");
    wheel(&mut app, MouseEventKind::ScrollDown, x, y + 2);
    assert_eq!(app.scroll, 3);
    assert_eq!(app.selected_session, 0);
}

#[test]
fn help_owns_the_mouse() {
    let mut app = app_with(5, 1);
    app.apply(Action::Help);
    let (x, y) = find(&app, " keys · ");
    click(&mut app, x, y + 1);
    assert!(app.show_help, "a click inside help keeps it open");
    assert_eq!(
        app.selected_session, 0,
        "and does not reach the list beneath"
    );
    click(&mut app, 0, H - 1);
    assert!(!app.show_help, "a click outside closes it");
}

#[test]
fn release_and_motion_do_nothing() {
    let mut app = app_with(5, 1);
    let (x, y) = find(&app, "s0000003");
    for kind in [
        MouseEventKind::Up(MouseButton::Left),
        MouseEventKind::Moved,
        MouseEventKind::Drag(MouseButton::Left),
    ] {
        render(&app);
        app.handle_mouse(event(kind, x, y));
        assert_eq!(app.selected_session, 0, "{kind:?}");
    }
}

#[test]
fn clicking_the_query_box_starts_typing_and_typing_keeps_the_keyboard() {
    let mut app = app_with(1, 1);
    app.apply(Action::ViewSimulator);
    let (x, y) = find(&app, "query");
    click(&mut app, x, y);
    assert!(app.simulator.typing());
    // While typing, a click on a tab must not pull focus out of the query.
    let (x, y) = find(&app, "1:Sessions");
    click(&mut app, x, y);
    assert_eq!(app.tab, SIMULATOR_TAB);
}
