use super::*;
use crossterm::event::KeyModifiers;

// A list on the left (cells 0-9), a border at 10, a text pane at 11-30,
// as in the Go kit's tests.
const LIST_W: u16 = 10;
const TEXT_X: u16 = 11;
const TEXT_W: u16 = 20;
const TEXT_H: u16 = 3;

fn doc() -> Vec<String> {
    [
        "line0 alpha",
        "line1 ~/src/keyolk/tuikit",
        "line2 gamma",
        "line3 delta",
        "line4 epsilon",
        "line5 한글 단어",
    ]
    .map(String::from)
    .to_vec()
}

fn frame(offset: usize) -> Buffer {
    let mut b = Buffer::empty(Rect::new(0, 0, TEXT_X + TEXT_W, TEXT_H));
    let left = ["item-one", "item-two", "item-three"];
    let lines = doc();
    for y in 0..TEXT_H {
        b.set_string(0, y, left[y as usize], ratatui::style::Style::default());
        b.set_string(LIST_W, y, "│", ratatui::style::Style::default());
        if let Some(l) = lines.get(offset + y as usize) {
            b.set_stringn(
                TEXT_X,
                y,
                l,
                TEXT_W as usize,
                ratatui::style::Style::default(),
            );
        }
    }
    b
}

fn panes(offset: usize) -> Vec<Pane> {
    vec![
        Pane::new("list", Rect::new(0, 0, LIST_W, TEXT_H), Kind::List),
        Pane::new("text", Rect::new(TEXT_X, 0, TEXT_W, TEXT_H), Kind::Text)
            .with_lines(doc(), offset),
    ]
}

fn ev(kind: MouseEventKind, x: u16, y: u16) -> MouseEvent {
    MouseEvent {
        kind,
        column: x,
        row: y,
        modifiers: KeyModifiers::NONE,
    }
}
fn press(x: u16, y: u16) -> MouseEvent {
    ev(MouseEventKind::Down(MouseButton::Left), x, y)
}
fn drag(x: u16, y: u16) -> MouseEvent {
    ev(MouseEventKind::Drag(MouseButton::Left), x, y)
}
fn up(x: u16, y: u16) -> MouseEvent {
    ev(MouseEventKind::Up(MouseButton::Left), x, y)
}

#[test]
fn a_drag_is_clamped_to_the_pane_under_the_press() {
    let (mut s, f, p, t) = (Selection::default(), frame(0), panes(0), Instant::now());
    assert!(!s.mouse(&press(0, 0), &p, &f, t).consumed);
    assert!(s.mouse(&drag(25, 1), &p, &f, t).consumed);
    let r = s.mouse(&up(25, 1), &p, &f, t);
    assert_eq!(r.copy.as_deref(), Some("item-one\nitem-two"));
    let mut b = f.clone();
    s.paint(&mut b, &p);
    for y in 0..2 {
        for x in LIST_W..TEXT_X + TEXT_W {
            assert!(
                !b[(x, y)].modifier.contains(Modifier::REVERSED),
                "cell {x},{y} outside the list is highlighted"
            );
        }
        assert!(b[(0, y)].modifier.contains(Modifier::REVERSED));
    }
}

#[test]
fn a_drag_past_the_bottom_scrolls_and_keeps_scrolled_out_lines() {
    let (mut s, t) = (Selection::default(), Instant::now());
    let mut off = 0;
    s.mouse(&press(TEXT_X, 0), &panes(off), &frame(off), t);
    for _ in 0..3 {
        let r = s.mouse(&drag(TEXT_X + 4, TEXT_H + 1), &panes(off), &frame(off), t);
        assert_eq!((r.scroll, r.scroll_pane), (1, "text"));
        off += 1;
    }
    let r = s.mouse(&up(TEXT_X + 4, TEXT_H - 1), &panes(off), &frame(off), t);
    let copy = r.copy.unwrap();
    assert!(copy.starts_with("line0 alpha\nline1"), "{copy:?}");
    assert!(copy.ends_with("\nline5"), "{copy:?}");
}

#[test]
fn double_and_triple_click_pick_a_word_and_a_line_in_text_only() {
    let (mut s, f, p, t) = (Selection::default(), frame(0), panes(0), Instant::now());
    let x = TEXT_X + 8; // inside "~/src/keyolk/tuikit"
    s.mouse(&press(x, 1), &p, &f, t);
    s.mouse(&up(x, 1), &p, &f, t);
    let r = s.mouse(&press(x, 1), &p, &f, t + Duration::from_millis(100));
    assert_eq!(
        r.copy.as_deref(),
        Some("~/src/keyolk/t"),
        "the word ends at the pane edge"
    );
    let r = s.mouse(&press(x, 1), &p, &f, t + Duration::from_millis(200));
    assert_eq!(r.copy.as_deref(), Some("line1 ~/src/keyolk/t"));

    let mut s = Selection::default();
    s.mouse(&press(1, 0), &p, &f, t);
    let r = s.mouse(&press(1, 0), &p, &f, t + Duration::from_millis(100));
    assert!(
        !r.consumed && r.copy.is_none(),
        "a list keeps its second click"
    );
}

#[test]
fn wide_characters_are_read_from_the_buffer_without_padding() {
    let (mut s, t) = (Selection::default(), Instant::now());
    let p = vec![Pane::new(
        "text",
        Rect::new(TEXT_X, 0, TEXT_W, TEXT_H),
        Kind::Text,
    )];
    let f = frame(3); // rows: line3, line4, line5 한글 단어
    s.mouse(&press(TEXT_X, 2), &p, &f, t);
    s.mouse(&drag(TEXT_X + TEXT_W, 2), &p, &f, t);
    let r = s.mouse(&up(TEXT_X + TEXT_W, 2), &p, &f, t);
    assert_eq!(r.copy.as_deref(), Some("line5 한글 단어"));
}

#[test]
fn clear_drops_the_range() {
    let (mut s, f, p, t) = (Selection::default(), frame(0), panes(0), Instant::now());
    s.mouse(&press(TEXT_X, 0), &p, &f, t);
    s.mouse(&drag(TEXT_X + 3, 1), &p, &f, t);
    assert!(s.active());
    s.clear();
    assert!(!s.active());
}
