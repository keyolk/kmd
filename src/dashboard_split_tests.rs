//! The resizable splits wired into the dashboard: Sessions list | detail and
//! the Simulator's hits | detail (side by side, or stacked when narrow).
//! Every press lands on a cell found in the drawn buffer; the divider is the
//! pair of adjacent borders (`││`, or a `└` row over a `┌` row), found there
//! too, never computed from the layout.

use super::key_tests::empty_snapshot;
use super::*;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use ratatui::buffer::Buffer;

const W: u16 = 120;
const H: u16 = 30;

fn render_at(app: &App, w: u16, h: u16) -> Buffer {
    let mut terminal = Terminal::new(ratatui::backend::TestBackend::new(w, h)).unwrap();
    terminal.draw(|frame| draw(frame, app)).unwrap();
    terminal.backend().buffer().clone()
}

fn render(app: &App) -> Buffer {
    render_at(app, W, H)
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

/// The side-by-side divider: the first column of two adjacent vertical
/// borders on row `y` (light, or heavy while dragged).
fn divider_x(b: &Buffer, y: u16) -> Option<u16> {
    let v = |x: u16| matches!(b[(x, y)].symbol(), "│" | "┃");
    (0..b.area.width.saturating_sub(1)).find(|&x| v(x) && v(x + 1))
}

/// The stacked divider: below the hits block's top, a bottom border row
/// directly over a top border row.
fn divider_y(b: &Buffer) -> Option<u16> {
    let starts = |y: u16, c: &[&str]| c.contains(&b[(0, y)].symbol());
    let top = find(b, " hits · ").1;
    (top..b.area.height.saturating_sub(1))
        .find(|&y| starts(y, &["└", "┗"]) && starts(y + 1, &["┌", "┏"]))
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

/// Each event against the frame drawn before it, as the event loop does.
fn gesture(app: &mut App, events: &[MouseEvent], now: Instant) {
    for e in events {
        render(app);
        app.on_mouse(*e, now);
    }
}

fn key(app: &mut App, c: char) {
    handle_key(app, KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)).unwrap();
}

fn session(index: usize) -> RagSession {
    RagSession {
        session_id: format!("s{index:07}-x"),
        latest_ts: format!("{}", 1000 - index),
        queries: (0..3)
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

fn app_with_sessions() -> App {
    let mut snapshot = empty_snapshot();
    snapshot.rag.sessions = (0..4).map(session).collect();
    App::new(
        snapshot,
        crate::dashboard_simulator::SimulatorState::without_history(),
    )
}

/// The divider on the row of the first session.
fn sessions_divider(app: &App) -> (u16, u16) {
    let b = render(app);
    let y = find(&b, "s0000000").1;
    (divider_x(&b, y).expect("a divider beside the sessions"), y)
}

#[test]
fn dragging_the_sessions_divider_moves_it_and_the_click_targets_follow() {
    let mut app = app_with_sessions();
    let (x0, y) = sessions_divider(&app);
    let to = x0 + 12;
    gesture(
        &mut app,
        &[down((x0, y)), drag((to, y)), up((to, y))],
        Instant::now(),
    );
    assert_eq!(sessions_divider(&app).0, to);
    assert!(!app.selection.active(), "a divider drag selected text");

    // Where the detail was is now the list: a press there selects a session.
    let b = render(&app);
    let at = (x0 + 3, find(&b, "s0000002").1);
    gesture(
        &mut app,
        &[down(at), up(at)],
        Instant::now() + Duration::from_secs(1),
    );
    assert_eq!(app.selected_session, 2);
}

#[test]
fn a_detail_drag_still_selects_text_and_leaves_the_divider() {
    let mut app = app_with_sessions();
    let (x0, _) = sessions_divider(&app);
    let b = render(&app);
    let from = find(&b, "Session s0000000");
    let to = find(&b, "origin:");
    gesture(&mut app, &[down(from), drag(to)], Instant::now());
    assert!(
        app.selection.active(),
        "a press in the detail reaches selection"
    );
    assert_eq!(sessions_divider(&app).0, x0);
}

#[test]
fn a_divider_press_drops_the_old_highlight() {
    let mut app = app_with_sessions();
    let b = render(&app);
    let from = find(&b, "Session s0000000");
    let to = find(&b, "origin:");
    gesture(&mut app, &[down(from), drag(to), up(to)], Instant::now());
    assert!(app.selection.active());
    let (x0, y) = sessions_divider(&app);
    gesture(&mut app, &[down((x0, y))], Instant::now());
    assert!(
        !app.selection.active(),
        "stale highlight after a divider press"
    );
}

#[test]
fn a_drag_draws_the_divider_heavy() {
    let mut app = app_with_sessions();
    let (x0, y) = sessions_divider(&app);
    gesture(&mut app, &[down((x0, y))], Instant::now());
    let b = render(&app);
    assert_eq!(b[(x0, y)].symbol(), "┃", "{}", row_text(&b, y));
    gesture(&mut app, &[up((x0, y))], Instant::now());
    assert_eq!(render(&app)[(x0, y)].symbol(), "│");
}

#[test]
fn a_double_click_on_the_divider_resets_it() {
    let mut app = app_with_sessions();
    let (x0, y) = sessions_divider(&app);
    key(&mut app, '>');
    key(&mut app, '>');
    let (x1, _) = sessions_divider(&app);
    assert!(x1 > x0);
    let t = Instant::now();
    gesture(&mut app, &[down((x1, y)), up((x1, y))], t);
    gesture(&mut app, &[down((x1, y))], t + Duration::from_millis(100));
    assert_eq!(sessions_divider(&app).0, x0);
}

#[test]
fn with_help_open_the_divider_is_not_the_splits() {
    let mut app = app_with_sessions();
    let (x0, y) = sessions_divider(&app);
    app.show_help = true;
    render(&app);
    app.on_mouse(down((x0, y)), Instant::now());
    assert!(!app.sessions_split.dragging(), "help owns the mouse");
    app.show_help = false;
    gesture(&mut app, &[drag((x0 + 10, y))], Instant::now());
    assert_eq!(sessions_divider(&app).0, x0);
}

#[test]
fn the_keys_move_the_sessions_divider_and_the_footer_names_them() {
    let mut app = app_with_sessions();
    let (x0, _) = sessions_divider(&app);
    key(&mut app, '>');
    assert!(sessions_divider(&app).0 > x0);
    key(&mut app, '<');
    key(&mut app, '<');
    assert!(sessions_divider(&app).0 < x0);
    key(&mut app, '=');
    assert_eq!(sessions_divider(&app).0, x0);

    let wide = render_at(&app, 240, H);
    let footer = row_text(&wide, H - 1);
    assert!(
        footer.contains("</>: split") && footer.contains("=: reset split"),
        "{footer}"
    );
    // Lowest priority: dropped first when the footer is short.
    let footer = row_text(&render_at(&app, 100, H), H - 1);
    assert!(!footer.contains("split"), "{footer}");
}

#[test]
fn one_shrink_after_growing_past_the_limit_moves_at_once() {
    // 60 columns: the detail's minimum stops the divider well before the
    // ratio bound, the gap where an unclamped ratio made presses dead.
    let mut app = app_with_sessions();
    let at_60 = |app: &App| {
        let b = render_at(app, 60, H);
        divider_x(&b, find(&b, "s0000000").1).expect("divider at 60 columns")
    };
    at_60(&app);
    for _ in 0..60 {
        key(&mut app, '>');
    }
    let limit = at_60(&app);
    key(&mut app, '<');
    assert!(at_60(&app) < limit, "dead zone at {limit}");
}

#[test]
fn the_split_keys_are_inert_in_operations_and_text_while_typing() {
    let mut app = app_with_sessions();
    app.select_tab(OPERATIONS_TAB);
    render(&app);
    key(&mut app, '>');
    assert!(!app.sessions_split.is_set());

    app.select_tab(SIMULATOR_TAB);
    render(&app);
    key(&mut app, 'i');
    key(&mut app, '>');
    assert_eq!(app.simulator.input, ">");
    assert!(!app.simulator.wide.is_set());
}

#[test]
fn the_simulator_moves_whichever_split_is_on_screen() {
    let mut app = app_with_sessions();
    app.select_tab(SIMULATOR_TAB);

    // Wide: hits | detail side by side.
    let b = render(&app);
    let y = find(&b, " hits · ").1 + 2;
    let x0 = divider_x(&b, y).expect("side-by-side divider");
    key(&mut app, '>');
    assert!(divider_x(&render(&app), y).unwrap() > x0);

    // Narrow: stacked, with its own ratio; the wide one stays as it was.
    let narrow = render_at(&app, 80, H);
    let y0 = divider_y(&narrow).expect("stacked divider");
    let wide_ratio = app.simulator.wide.ratio();
    key(&mut app, '>');
    let y1 = divider_y(&render_at(&app, 80, H)).unwrap();
    assert!(y1 > y0, "stacked {y0} -> {y1}");
    assert_eq!(app.simulator.wide.ratio(), wide_ratio);

    // Drag the stacked divider: the event loop's frame is the narrow one.
    let at = (10, y1);
    for e in [down(at), drag((10, y1 - 2)), up((10, y1 - 2))] {
        render_at(&app, 80, H);
        app.on_mouse(e, Instant::now());
    }
    assert_eq!(divider_y(&render_at(&app, 80, H)), Some(y1 - 2));
}

#[test]
fn small_terminals_never_overflow_and_collapse_to_the_list() {
    for (w, h) in [(80, 24), (60, 24), (40, 20)] {
        let app = app_with_sessions();
        let b = render_at(&app, w, h);
        let y = find(&b, "s0000000").1;
        assert!(divider_x(&b, y).is_some(), "{w} cols: both panes fit");
        let (list, detail) = app.sessions_split.bordered(Rect::new(0, 0, w, h));
        assert!(list.right() <= w && detail.right() <= w);
    }
    // Under both minimums and the divider (16 + 20 + 2) only the list shows.
    let app = app_with_sessions();
    let b = render_at(&app, 36, 20);
    let y = find(&b, "s0000000").1;
    assert_eq!(divider_x(&b, y), None, "{}", row_text(&b, y));
}

#[test]
fn a_moved_split_survives_a_restart_and_an_unmoved_one_is_not_saved() {
    let dir = std::env::temp_dir().join(format!("kmd-split-{}", std::process::id()));
    fs::remove_dir_all(&dir).ok();
    let path = dir.join("kmd/ui.json");

    let mut first = app_with_sessions();
    first.ui_path = Some(path.clone());
    first.persist_ui_on_exit().unwrap();
    assert!(
        !fs::read_to_string(&path)
            .unwrap_or_default()
            .contains("split"),
        "an unmoved split was saved"
    );

    render(&first);
    key(&mut first, '>');
    key(&mut first, '>');
    let (moved, _) = sessions_divider(&first);
    first.persist_ui_on_exit().unwrap();

    let mut second = app_with_sessions();
    second.restore_ui(crate::ui_state::load(&path).unwrap());
    assert_eq!(sessions_divider(&second).0, moved);

    // A percentage where a fraction belongs is rejected, not read as 34x.
    fs::write(&path, r#"{"split_sessions":{"ratio":34}}"#).unwrap();
    let mut third = app_with_sessions();
    let default = sessions_divider(&third).0;
    third.restore_ui(crate::ui_state::load(&path).unwrap());
    assert_eq!(sessions_divider(&third).0, default);
    fs::remove_dir_all(&dir).ok();
}
