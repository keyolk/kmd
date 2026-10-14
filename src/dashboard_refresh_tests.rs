//! Regression tests for background refresh, status, selection, scroll, and Esc.

use super::key_tests::{app, empty_snapshot};
use super::*;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use std::time::Duration;

fn press(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, KeyModifiers::NONE)
}

fn rag_session(id: &str, ts: &str) -> RagSession {
    RagSession {
        session_id: id.into(),
        latest_ts: ts.into(),
        queries: vec![RagQueryRecord {
            ts: ts.into(),
            query: format!("query from {id}"),
            injected: 0,
            hits: Vec::new(),
            latency_ms: 1,
        }],
    }
}

fn snapshot_with(sessions: Vec<RagSession>) -> DashboardSnapshot {
    let mut snapshot = empty_snapshot();
    snapshot.rag.sessions = sessions;
    snapshot
}

/// Drives the event loop's poll until the background job lands.
fn pump(app: &mut App) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if app.poll_background() {
            return;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    panic!("background job did not land");
}

#[test]
fn refresh_keeps_the_selection_on_the_same_session_after_a_reorder() {
    let mut app = app();
    app.apply_snapshot(snapshot_with(vec![
        rag_session("aaaaaaaa-1", "300"),
        rag_session("bbbbbbbb-2", "200"),
    ]));
    app.select_session(1); // bbbbbbbb
    // bbbbbbbb queried again and now sorts first.
    app.apply_snapshot(snapshot_with(vec![
        rag_session("aaaaaaaa-1", "300"),
        rag_session("bbbbbbbb-2", "400"),
    ]));
    assert_eq!(app.session_ids()[app.selected_session], "bbbbbbbb-2");
}

#[test]
fn a_vanished_session_keeps_the_slot_and_drops_its_scroll() {
    let mut app = app();
    app.apply_snapshot(snapshot_with(vec![
        rag_session("aaaaaaaa-1", "300"),
        rag_session("bbbbbbbb-2", "200"),
        rag_session("cccccccc-3", "100"),
    ]));
    app.select_session(1);
    app.scroll = 7;
    app.apply_snapshot(snapshot_with(vec![
        rag_session("aaaaaaaa-1", "300"),
        rag_session("cccccccc-3", "100"),
    ]));
    assert_eq!(
        app.selected_session, 1,
        "stay in the slot, not jump to the top"
    );
    assert_eq!(
        app.scroll, 0,
        "the old session's scroll must not apply to another"
    );
}

#[test]
fn a_refresh_runs_off_the_input_thread_and_lands_through_poll() {
    let mut app = app();
    assert!(
        app.refresher
            .start(|| { Ok(snapshot_with(vec![rag_session("dddddddd-4", "500")])) })
    );
    assert!(app.refresher.in_flight());
    // A manual refresh while one runs is queued, not started on top of it.
    app.request_refresh(true);
    assert!(app.refresh_again);
    pump(&mut app);
    assert_eq!(app.session_ids(), vec!["dddddddd-4".to_string()]);
    assert!(
        app.refresher.in_flight(),
        "the queued manual refresh starts once the first one lands"
    );
}

#[test]
fn a_failed_refresh_keeps_the_last_snapshot_and_shows_a_sticky_error() {
    let mut app = app();
    app.apply_snapshot(snapshot_with(vec![rag_session("aaaaaaaa-1", "300")]));
    assert!(app.refresher.start(|| anyhow::bail!("store locked")));
    pump(&mut app);
    assert_eq!(app.session_ids(), vec!["aaaaaaaa-1".to_string()]);
    let (text, level) = app.status.current().expect("error is shown");
    assert!(text.contains("store locked"));
    assert_eq!(level, Level::Error);
    assert!(!app.status.expire(Instant::now() + Duration::from_secs(60)));
}

#[test]
fn an_auto_refresh_does_not_overwrite_a_check_result() {
    let mut app = app();
    app.status
        .set("checks", Level::Warning, "self-check: 9/11 passed");
    assert!(app.refresher.start(|| Ok(empty_snapshot())));
    pump(&mut app);
    assert_eq!(
        app.status.current(),
        Some(("self-check: 9/11 passed", Level::Warning))
    );
}

#[test]
fn esc_does_not_quit_the_dashboard() {
    let mut app = app();
    assert!(!handle_key(&mut app, press(KeyCode::Esc)).unwrap());
    app.select_tab(SIMULATOR_TAB);
    handle_key(&mut app, press(KeyCode::Char('i'))).unwrap();
    // Esc leaves typing; a second Esc used to close the whole dashboard.
    assert!(!handle_key(&mut app, press(KeyCode::Esc)).unwrap());
    assert!(!handle_key(&mut app, press(KeyCode::Esc)).unwrap());
    assert!(handle_key(&mut app, press(KeyCode::Char('q'))).unwrap());
}

#[test]
fn bottom_then_up_scrolls_from_the_last_page_not_from_65535() {
    let mut app = app();
    app.select_tab(OPERATIONS_TAB);
    let backend = ratatui::backend::TestBackend::new(100, 20);
    let mut terminal = Terminal::new(backend).unwrap();

    handle_key(&mut app, press(KeyCode::Char('G'))).unwrap();
    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let max = app.scroll_max.get();
    assert!(max > 0, "operations text should overflow a 20-row terminal");
    assert!(max < 200);

    handle_key(&mut app, press(KeyCode::Char('k'))).unwrap();
    assert_eq!(app.scroll, max - 1);
}

#[test]
fn the_footer_shows_the_status_before_the_hints() {
    let mut app = app();
    app.status
        .set("checks", Level::Error, "self-check failed: boom");
    // Narrower than the hints alone: the status must still be visible.
    let backend = ratatui::backend::TestBackend::new(40, 20);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|frame| draw(frame, &app)).unwrap();
    let buffer = terminal.backend().buffer();
    let last_row = (0..40)
        .map(|x| buffer[(x, 19)].symbol().to_string())
        .collect::<String>();
    assert!(last_row.contains("self-check failed"), "{last_row}");
}
