//! Tests for the derived footer/help (one key table) and persisted UI state.

use super::key_tests::{app, empty_snapshot};
use super::*;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

fn press(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, KeyModifiers::NONE)
}

fn screen(app: &App, width: u16, height: u16) -> String {
    let mut terminal = Terminal::new(ratatui::backend::TestBackend::new(width, height)).unwrap();
    terminal.draw(|frame| draw(frame, app)).unwrap();
    let buffer = terminal.backend().buffer().clone();
    (0..buffer.area.height)
        .map(|y| {
            (0..buffer.area.width)
                .map(|x| buffer[(x, y)].symbol())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn last_line(text: &str) -> &str {
    text.lines().last().unwrap_or_default()
}

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("kmd-dash-{name}-{}", std::process::id()));
    fs::remove_dir_all(&dir).ok();
    dir
}

// --- one key table -----------------------------------------------------------

#[test]
fn the_footer_is_derived_from_the_active_contexts() {
    let mut app = app();
    let sessions = screen(&app, 160, 30);
    let footer = last_line(&sessions);
    assert!(footer.contains("j/k: session"), "{footer}");
    assert!(footer.contains("q: quit"), "{footer}");

    app.select_tab(SIMULATOR_TAB);
    let command = screen(&app, 160, 30);
    assert!(
        last_line(&command).contains("i: type"),
        "{}",
        last_line(&command)
    );

    handle_key(&mut app, press(KeyCode::Char('i'))).unwrap();
    let typing = screen(&app, 160, 30);
    let footer = last_line(&typing);
    assert!(footer.contains("typing · "), "{footer}");
    assert!(footer.contains("Esc: command"), "{footer}");
    // `?` while typing is text, so the footer must not offer it.
    assert!(!footer.contains("?: keys"), "{footer}");
    handle_key(&mut app, press(KeyCode::Char('?'))).unwrap();
    assert_eq!(app.simulator.input, "?");
    assert!(!app.show_help);
}

#[test]
fn a_rebound_key_changes_dispatch_footer_guide_and_help_together() {
    let mut app = app();
    app.keys = Keymap::defaults_with_test_overrides("sim.mode: M\n");
    app.select_tab(SIMULATOR_TAB);
    let text = screen(&app, 160, 30);
    assert!(last_line(&text).contains("M: mode"));
    assert!(text.contains("M mode"), "guide line follows the keymap");

    handle_key(&mut app, press(KeyCode::Char('m'))).unwrap();
    assert_eq!(app.simulator.mode, crate::sim::SimulatorMode::Rag);
    handle_key(&mut app, press(KeyCode::Char('M'))).unwrap();
    assert_eq!(app.simulator.mode, crate::sim::SimulatorMode::Search);

    app.show_help = true;
    let help = screen(&app, 120, 60);
    assert!(help.contains(&format!("{:<18}RAG ⇄ BM25", "M")), "{help}");
}

#[test]
fn the_help_overlay_lists_every_binding() {
    let mut app = app();
    handle_key(&mut app, press(KeyCode::Char('?'))).unwrap();
    assert!(app.show_help);
    let help = screen(&app, 120, 60);
    for binding in app.keys.bindings() {
        assert!(
            help.contains(binding.help),
            "help misses {:?}",
            binding.help
        );
    }
}

#[test]
fn the_help_overlay_scrolls_when_short_and_any_other_key_closes_it() {
    let mut app = app();
    handle_key(&mut app, press(KeyCode::Char('?'))).unwrap();
    let short = screen(&app, 100, 20);
    assert!(short.contains("scrolls"), "{short}");
    handle_key(&mut app, press(KeyCode::Char('j'))).unwrap();
    assert!(app.show_help, "j scrolls the overlay instead of closing it");
    assert_eq!(app.help_scroll, 1);
    handle_key(&mut app, press(KeyCode::Char('z'))).unwrap();
    assert!(!app.show_help);
}

// --- persisted UI state ------------------------------------------------------

fn session(id: &str, ts: &str) -> RagSession {
    RagSession {
        session_id: id.into(),
        latest_ts: ts.into(),
        queries: vec![RagQueryRecord {
            ts: ts.into(),
            query: "q".into(),
            injected: 0,
            hits: Vec::new(),
            latency_ms: 1,
        }],
    }
}

fn app_with_sessions() -> App {
    let mut snapshot = empty_snapshot();
    snapshot.rag.sessions = vec![session("aaaa1111", "300"), session("bbbb2222", "200")];
    App::new(
        snapshot,
        crate::dashboard_simulator::SimulatorState::without_history(),
    )
}

#[test]
fn ui_state_survives_a_restart() {
    let dir = temp_dir("restart");
    let path = dir.join("kmd/ui.json");

    let mut first = app_with_sessions();
    first.ui_path = Some(path.clone());
    first.select_session(1);
    first.select_tab(SIMULATOR_TAB);
    first.simulator.toggle_mode();
    first.simulator.input = "last query".into();
    first.persist_ui_on_exit().unwrap();

    let mut second = app_with_sessions();
    // The list reordered between runs: restore by id, not by row.
    second.snapshot.rag.sessions.reverse();
    second.restore_ui(crate::ui_state::load(&path).unwrap());
    assert_eq!(second.tab, SIMULATOR_TAB);
    assert_eq!(second.simulator.mode, crate::sim::SimulatorMode::Search);
    assert_eq!(second.simulator.input, "last query");
    assert_eq!(second.session_ids()[second.selected_session], "bbbb2222");
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn ui_state_is_written_off_the_key_path_and_only_when_changed() {
    let dir = temp_dir("worker");
    let path = dir.join("kmd/ui.json");
    let mut app = app_with_sessions();
    app.ui_path = Some(path.clone());
    app.ui_saved = app.ui_state();

    // Unchanged state is never written.
    app.ui_checked = Instant::now() - Duration::from_secs(5);
    app.persist_ui();
    assert!(!app.ui_saver.in_flight());
    assert!(!path.exists());

    // A change is handed to the worker; handle_key itself never writes.
    handle_key(&mut app, press(KeyCode::Char('2'))).unwrap();
    assert!(!path.exists());
    app.ui_checked = Instant::now() - Duration::from_secs(5);
    app.persist_ui();
    assert!(app.ui_saver.in_flight());
    app.ui_saver.wait(Duration::from_secs(5)).unwrap().unwrap();
    let saved = crate::ui_state::load(&path).unwrap();
    assert_eq!(saved.view.as_deref(), Some("operations"));
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn unknown_saved_values_are_ignored() {
    let mut app = app_with_sessions();
    app.restore_ui(crate::ui_state::UiState {
        view: Some("gone".into()),
        session_id: Some("no-such-session".into()),
        simulator_mode: None,
        query: String::new(),
    });
    assert_eq!(app.tab, SESSIONS_TAB);
    assert_eq!(app.selected_session, 0);
}
