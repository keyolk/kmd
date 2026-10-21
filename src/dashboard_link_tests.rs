//! `o` opens the selected item's link: the session's text in Sessions, the
//! result detail in the Simulator.

use super::key_tests::empty_snapshot;
use super::*;
use crate::dashboard_link::stub;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

fn press(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, KeyModifiers::NONE)
}

/// A live card; `age` seconds older than the newest, so the list order is
/// the order given.
fn card(id: &str, summary: &str, age: i64) -> crate::activity::ActivityCard {
    crate::activity::ActivityCard {
        session_id: id.into(),
        updated_at: "2026-08-01T01:10:00Z".into(),
        updated_epoch: 1_754_010_600 - age,
        cwd: "/repo".into(),
        repo: Some("kmd".into()),
        summary: summary.into(),
        latest_user: summary.into(),
        latest_assistant: String::new(),
        files: Vec::new(),
        links: Vec::new(),
    }
}

/// A Sessions view over live cards, in order, with the recorder installed.
fn sessions(summaries: &[&str]) -> App {
    let mut snapshot = empty_snapshot();
    snapshot.activity = summaries
        .iter()
        .enumerate()
        .map(|(i, s)| card(&format!("{i:08}"), s, i as i64 * 60))
        .collect();
    let mut app = App::new(
        snapshot,
        crate::dashboard_simulator::SimulatorState::without_history(),
    );
    app.launcher = stub::record;
    app
}

fn status(app: &App) -> &str {
    app.status.current().map(|(text, _)| text).unwrap_or("")
}

fn screen(app: &App, width: u16, height: u16) -> ratatui::buffer::Buffer {
    let mut terminal = Terminal::new(ratatui::backend::TestBackend::new(width, height)).unwrap();
    terminal.draw(|frame| draw(frame, app)).unwrap();
    terminal.backend().buffer().clone()
}

#[test]
fn o_opens_exactly_the_selected_sessions_url() {
    let mut app = sessions(&[
        "review https://github.com/keyolk/kmd/pull/21 before merging",
        "look at https://x.io/other",
    ]);
    let url = "https://github.com/keyolk/kmd/pull/21";
    handle_key(&mut app, press(KeyCode::Char('o'))).unwrap();
    assert_eq!(stub::take(), [url]);
    assert_eq!(status(&app), format!("opened {url}"));

    // The other session opens its own link.
    handle_key(&mut app, press(KeyCode::Char('j'))).unwrap();
    handle_key(&mut app, press(KeyCode::Char('o'))).unwrap();
    assert_eq!(stub::take(), ["https://x.io/other"]);
}

#[test]
fn o_with_several_links_opens_the_first_and_says_how_many() {
    let mut app = sessions(&["compare https://a.io/1 with https://b.io/2 please"]);
    handle_key(&mut app, press(KeyCode::Char('o'))).unwrap();
    assert_eq!(stub::take(), ["https://a.io/1"]);
    // The summary is also the latest request, which the detail does not
    // repeat, so each URL is counted once.
    assert_eq!(status(&app), "opened https://a.io/1 (the first of 2 links)");
}

#[test]
fn o_without_a_link_launches_nothing_and_says_so() {
    let mut app = sessions(&["refactor the tokenizer without links"]);
    app.launcher = stub::unexpected;
    handle_key(&mut app, press(KeyCode::Char('o'))).unwrap();
    assert_eq!(status(&app), "no link in this session");

    let mut empty = super::key_tests::app();
    empty.launcher = stub::unexpected;
    handle_key(&mut empty, press(KeyCode::Char('o'))).unwrap();
    assert_eq!(status(&empty), "no link in this session");
}

#[test]
fn a_javascript_or_file_url_is_never_opened() {
    let mut app = sessions(&["run javascript:alert(1) or open file:///etc/passwd now"]);
    app.launcher = stub::unexpected;
    handle_key(&mut app, press(KeyCode::Char('o'))).unwrap();
    assert_eq!(status(&app), "no link in this session");
}

#[test]
fn the_frame_carries_no_link_escapes() {
    // The dashboard prints URLs in full and links none: an OSC 8 escape in a
    // ratatui cell would be measured as text and shift the row.
    let app = sessions(&["review https://github.com/keyolk/kmd/pull/21 before merging"]);
    let buffer = screen(&app, 160, 30);
    assert!(
        buffer.content.iter().all(|c| !c.symbol().contains('\x1b')),
        "a cell holds an escape"
    );
    let text: String = buffer.content.iter().map(|c| c.symbol()).collect();
    assert!(text.contains("https://github.com/keyolk/kmd/pull/21"));
}

#[test]
fn footer_and_help_show_o() {
    let mut app = sessions(&["x"]);
    let footer: String = {
        let buffer = screen(&app, 200, 30);
        (0..200).map(|x| buffer[(x, 29)].symbol()).collect()
    };
    assert!(footer.contains("o: open link"), "{footer}");
    let help = app.keys.help_sections();
    for (title, action) in [
        ("Sessions", "session's text"),
        ("Simulator", "result detail"),
    ] {
        let rows = &help.iter().find(|(t, _)| *t == title).unwrap().1;
        assert!(
            rows.iter().any(|(k, h)| k == "o" && h.contains(action)),
            "{title}: {rows:?}"
        );
    }
    app.select_tab(SIMULATOR_TAB);
    let buffer = screen(&app, 200, 30);
    let footer: String = (0..200).map(|x| buffer[(x, 29)].symbol()).collect();
    assert!(footer.contains("o: open link"), "{footer}");
}

fn hit(file: &str, snippet: &str) -> crate::bm25::SearchHit {
    crate::bm25::SearchHit {
        docid: "#000001".into(),
        score: 1.0,
        file: file.into(),
        abspath: None,
        title: "t".into(),
        context: None,
        snippet: Some(snippet.into()),
    }
}

fn simulator(hits: Vec<crate::bm25::SearchHit>) -> App {
    let mut app = sessions(&[]);
    app.select_tab(SIMULATOR_TAB);
    app.simulator.result = Some(crate::sim::SimulatorResult {
        mode: crate::sim::SimulatorMode::Search,
        query: "q".into(),
        gate_reason: None,
        hits,
        context: None,
        latency_ms: 1,
        axes: Vec::new(),
    });
    app
}

#[test]
fn o_in_the_simulator_opens_the_selected_hits_link() {
    let mut app = simulator(vec![
        hit("kmd://notes/a.md", "see https://docs.rs/ratatui (0.30)."),
        hit(
            "kmd://notes/b.md",
            "PR: https://github.com/keyolk/tmc/pull/15",
        ),
    ]);
    handle_key(&mut app, press(KeyCode::Char('o'))).unwrap();
    assert_eq!(stub::take(), ["https://docs.rs/ratatui"]);
    handle_key(&mut app, press(KeyCode::Char('j'))).unwrap();
    handle_key(&mut app, press(KeyCode::Char('o'))).unwrap();
    assert_eq!(stub::take(), ["https://github.com/keyolk/tmc/pull/15"]);
}

#[test]
fn o_in_the_simulator_ignores_kmd_uris_and_says_so() {
    let mut app = simulator(vec![hit("kmd://notes/a.md", "no web link here")]);
    app.launcher = stub::unexpected;
    handle_key(&mut app, press(KeyCode::Char('o'))).unwrap();
    assert_eq!(status(&app), "no link in the result detail");
}

#[test]
fn o_types_while_the_simulator_is_typing() {
    let mut app = simulator(Vec::new());
    app.launcher = stub::unexpected;
    handle_key(&mut app, press(KeyCode::Char('i'))).unwrap();
    handle_key(&mut app, press(KeyCode::Char('o'))).unwrap();
    assert_eq!(app.simulator.input, "o");
}
