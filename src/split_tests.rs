use super::*;
use crossterm::event::KeyModifiers;

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
fn motion(x: u16, y: u16) -> MouseEvent {
    ev(MouseEventKind::Drag(MouseButton::Left), x, y)
}
fn hover(x: u16, y: u16) -> MouseEvent {
    ev(MouseEventKind::Moved, x, y)
}
fn release(x: u16, y: u16) -> MouseEvent {
    ev(MouseEventKind::Up(MouseButton::Left), x, y)
}

/// 101 columns: the panes share 100, so A's width is the percent.
const AREA: Rect = Rect::new(0, 0, 101, 10);

fn one() -> Config {
    Config {
        divider: 1,
        ..Default::default()
    }
}
fn split(c: Config) -> Split {
    Split::new(c)
}
fn width_a(s: &Split, r: Rect) -> u16 {
    s.layout(r).0.width
}

#[test]
fn layout_tiles_the_area_and_keeps_the_divider_out_of_both_panes() {
    let cases = [
        (split(one()), Rect::new(0, 0, 80, 24), 40, 1, 39),
        (split(one()), Rect::new(2, 3, 80, 20), 40, 1, 39),
        (
            split(Config {
                divider: 2,
                ..one()
            }),
            Rect::new(0, 0, 80, 24),
            39,
            2,
            39,
        ),
        (
            split(Config {
                divider: 3,
                ..one()
            }),
            Rect::new(0, 0, 80, 24),
            39,
            3,
            38,
        ),
        (
            split(Config {
                min_a: 30,
                default: 0.2,
                ..one()
            }),
            Rect::new(0, 0, 60, 10),
            30,
            1,
            29,
        ),
        (
            split(Config {
                min_b: 20,
                default: 0.9,
                ..one()
            }),
            Rect::new(0, 0, 60, 10),
            39,
            1,
            20,
        ),
    ];
    for (s, r, a, d, b) in cases {
        let (ra, rd, rb) = s.layout(r);
        assert_eq!((ra.width, rd.width, rb.width), (a, d, b), "{r:?}");
        assert_eq!((ra.x, rd.x, rb.x), (r.x, r.x + a, r.x + a + d));
        for x in r.left()..r.right() {
            let n = [ra, rd, rb]
                .iter()
                .filter(|p| p.contains(Position::new(x, r.y)))
                .count();
            assert_eq!(n, 1, "cell {x} in {n} rects");
        }
    }
}

#[test]
fn a_split_that_does_not_fit_shows_the_primary_pane_only() {
    let s = split(Config {
        min_a: 30,
        min_b: 20,
        ..one()
    });
    let r = Rect::new(0, 0, 50, 10);
    assert_eq!(s.layout(r), (r, Rect::default(), Rect::default()));
    let s = split(Config {
        primary: Side::B,
        ..s.config
    });
    assert_eq!(s.layout(r), (Rect::default(), Rect::default(), r));
    let s = split(Config {
        collapse: 70,
        ..one()
    });
    assert!(s.collapsed(Rect::new(0, 0, 69, 10)));
    assert!(!s.collapsed(Rect::new(0, 0, 70, 10)));
}

#[test]
fn vertical_layout_cuts_rows() {
    let s = split(Config {
        orientation: Orientation::Vertical,
        ..one()
    });
    let (a, d, b) = s.layout(Rect::new(0, 0, 80, 24));
    assert_eq!(
        (a.height, d.y, d.height, b.y, b.height),
        (12, 12, 1, 13, 11)
    );
}

#[test]
fn the_ratio_survives_a_resize() {
    let mut s = split(one());
    s.set(0.3, AREA);
    assert_eq!(width_a(&s, Rect::new(0, 0, 201, 10)), 60);
}

#[test]
fn writes_clamp_to_the_minimums_and_bounds() {
    let area = Rect::new(0, 0, 61, 10); // avail 60
    let mut s = split(Config {
        min_a: 30,
        min_b: 20,
        ..one()
    });
    for v in [-1.0, 0.0, 0.01, 0.2] {
        s.set(v, area);
        assert_eq!(s.ratio(), 0.5, "set({v})");
    }
    s.set(0.95, area);
    assert!((s.ratio() - 40.0 / 60.0).abs() < 1e-9);
    let before = s.ratio();
    s.set(f64::NAN, area);
    assert_eq!(s.ratio(), before);
    let mut s = split(Config {
        lo: 0.25,
        hi: 0.75,
        ..one()
    });
    s.set(0.9, Rect::default());
    assert_eq!(s.ratio(), 0.75);
}

/// narwhal's unclamped delta: grow past the limit stored a value the
/// layout could not show, so the presses back did nothing for a while.
#[test]
fn no_dead_zone_past_the_limit() {
    let area = Rect::new(0, 0, 81, 10);
    let mut s = split(Config { min_b: 30, ..one() });
    for _ in 0..20 {
        s.grow(area);
    }
    let a1 = width_a(&s, area);
    s.shrink(area);
    assert!(
        width_a(&s, area) < a1,
        "first shrink after the limit moved nothing"
    );
}

#[test]
fn a_step_moves_at_least_one_cell() {
    let area = Rect::new(0, 0, 11, 3); // a 0.01 step is a tenth of a cell
    let s = split(Config {
        step: 0.01,
        ..one()
    });
    let a0 = width_a(&s, area);
    let mut g = s.clone();
    g.grow(area);
    assert_eq!(width_a(&g, area), a0 + 1);
    let mut k = s.clone();
    k.shrink(area);
    assert_eq!(width_a(&k, area), a0 - 1);
}

#[test]
fn handle_reports_whether_the_screen_changed() {
    let mut s = split(Config { hi: 0.55, ..one() });
    let o = s.handle(Action::Grow, AREA);
    assert!(o.consumed && o.changed && s.is_set());
    let o = s.handle(Action::Grow, AREA);
    assert!(o.consumed && !o.changed, "grow at hi: {o:?}");
    let o = s.handle(Action::Reset, AREA);
    assert!(o.changed && o.reset && !s.is_set());
    assert_eq!(s.ratio(), 0.5);
}

#[test]
fn drag_moves_the_divider_and_release_ends_it() {
    let mut s = split(one());
    let t = Instant::now();
    let o = s.mouse(&press(50, 3), AREA, t);
    assert!(o.consumed && !o.changed);
    assert!(s.mouse(&motion(60, 3), AREA, t).changed);
    assert!(s.mouse(&motion(70, 4), AREA, t).changed);
    assert!(s.mouse(&release(70, 4), AREA, t).consumed);
    assert!(!s.dragging());
    assert_eq!(width_a(&s, AREA), 70);
    assert!(
        !s.mouse(&motion(30, 3), AREA, t).consumed,
        "motion after release"
    );
}

#[test]
fn a_drag_keeps_the_grab_offset_on_a_wide_divider() {
    let mut s = split(Config {
        divider: 3,
        ..one()
    });
    let (_, div, _) = s.layout(AREA);
    let t = Instant::now();
    s.mouse(&press(div.x + 2, 0), AREA, t);
    s.mouse(&motion(div.x + 3, 0), AREA, t);
    assert_eq!(width_a(&s, AREA), div.x + 1);
}

#[test]
fn a_drag_past_the_edges_clamps() {
    let mut s = split(Config {
        min_a: 20,
        min_b: 30,
        ..one()
    });
    let t = Instant::now();
    s.mouse(&press(50, 0), AREA, t);
    s.mouse(&motion(0, 0), AREA, t);
    assert_eq!(width_a(&s, AREA), 20);
    s.mouse(&motion(500, 0), AREA, t);
    assert_eq!(width_a(&s, AREA), 70);
}

#[test]
fn buttonless_motion_ends_a_drag_whose_release_was_lost() {
    let mut s = split(one());
    let t = Instant::now();
    s.mouse(&press(50, 0), AREA, t);
    s.mouse(&motion(55, 0), AREA, t);
    assert!(s.mouse(&hover(80, 0), AREA, t).consumed);
    assert!(!s.dragging());
    assert!(!s.mouse(&motion(90, 0), AREA, t).consumed);
    assert_eq!(width_a(&s, AREA), 55);
}

#[test]
fn a_pane_press_is_never_the_splits() {
    let mut s = split(one());
    for x in [0, 49, 51, 100] {
        assert!(
            !s.mouse(&press(x, 2), AREA, Instant::now()).consumed,
            "x={x}"
        );
    }
    let mut g = split(Config { grab: 1, ..one() });
    for (x, want) in [(48, false), (49, true), (50, true), (51, true), (52, false)] {
        let got = g.mouse(&press(x, 2), AREA, Instant::now()).consumed;
        g.mouse(&release(x, 2), AREA, Instant::now());
        assert_eq!(got, want, "grab 1 at x={x}");
    }
}

#[test]
fn a_vertical_drag_moves_rows() {
    let mut s = split(Config {
        orientation: Orientation::Vertical,
        ..one()
    });
    let r = Rect::new(0, 0, 40, 21);
    let t = Instant::now();
    s.mouse(&press(5, 10), r, t);
    assert!(s.mouse(&motion(5, 14), r, t).changed);
    assert_eq!(s.layout(r).0.height, 14);
}

#[test]
fn a_double_click_on_the_divider_resets() {
    let mut s = split(one());
    s.set(0.7, AREA);
    let t = Instant::now();
    s.mouse(&press(70, 0), AREA, t);
    s.mouse(&release(70, 0), AREA, t + Duration::from_millis(50));
    let o = s.mouse(&press(70, 0), AREA, t + Duration::from_millis(100));
    assert!(o.consumed && o.changed && o.reset, "{o:?}");
    assert_eq!(s.ratio(), 0.5);
    assert!(!s.dragging());
}

#[test]
fn a_regrab_right_after_a_drag_does_not_reset() {
    let mut s = split(one());
    let t = Instant::now();
    s.mouse(&press(50, 0), AREA, t);
    s.mouse(&motion(60, 0), AREA, t + Duration::from_millis(10));
    s.mouse(&release(60, 0), AREA, t + Duration::from_millis(20));
    let o = s.mouse(&press(60, 0), AREA, t + Duration::from_millis(30));
    assert!(!o.reset);
    assert_eq!(width_a(&s, AREA), 60);
    assert!(s.dragging());
}

#[test]
fn restore_rejects_garbage_and_clamps() {
    let mut s = split(Config {
        lo: 0.25,
        hi: 0.75,
        ..one()
    });
    for bad in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0,
        -0.3,
        1.0,
        50.0,
    ] {
        assert!(!s.restore(State { ratio: bad }), "{bad}");
        assert!(!s.is_set());
    }
    assert!(s.restore(State { ratio: 0.9 }));
    assert_eq!(s.ratio(), 0.75);
    assert!(s.restore(State { ratio: 0.3 }));
    assert_eq!(s.state(), Some(State { ratio: 0.3 }));
}

#[test]
fn an_unmoved_split_has_nothing_to_save() {
    let mut s = split(one());
    assert_eq!(s.state(), None);
    s.grow(AREA);
    assert!(s.state().is_some());
    s.reset();
    assert_eq!(s.state(), None);
}

/// A restored ratio is clamped to the bounds only (no size is known yet);
/// the first key press must still move the divider.
#[test]
fn a_step_after_restore_moves_at_once() {
    let area = Rect::new(0, 0, 81, 10); // min_b 30 caps A at 50
    let mut s = split(Config { min_b: 30, ..one() });
    s.restore(State { ratio: 0.85 });
    let a0 = width_a(&s, area);
    s.shrink(area);
    assert!(width_a(&s, area) < a0);
}

#[test]
fn the_divider_draws_light_idle_heavy_dragging_and_blank_when_asked() {
    let mut s = split(one());
    let area = Rect::new(0, 0, 11, 2);
    let (_, div, _) = s.layout(area);
    let mut buf = Buffer::empty(area);
    s.draw_divider(&mut buf, div, "│", Style::default());
    assert_eq!(buf[(div.x, 1)].symbol(), "│");
    s.mouse(&press(div.x, 0), area, Instant::now());
    s.draw_divider(&mut buf, div, "│", Style::default());
    assert_eq!(buf[(div.x, 1)].symbol(), "┃");
    let s = split(Config {
        look: Look::Blank,
        ..one()
    });
    s.draw_divider(&mut buf, div, "│", Style::default());
    assert_eq!(buf[(div.x, 1)].symbol(), " ");
}

#[test]
fn lenient_state_drops_garbage_without_failing_the_file() {
    #[derive(Deserialize)]
    struct Ui {
        #[serde(default, deserialize_with = "lenient")]
        split: Option<State>,
        n: u8,
    }
    for (raw, want) in [
        (r#"{"split":{"ratio":0.4},"n":1}"#, Some(0.4)),
        (r#"{"split":{"ratio":"x"},"n":1}"#, None),
        (r#"{"split":7,"n":1}"#, None),
        (r#"{"split":null,"n":1}"#, None),
        (r#"{"n":1}"#, None),
    ] {
        let ui: Ui = serde_json::from_str(raw).expect(raw);
        assert_eq!(ui.split.map(|s| s.ratio), want, "{raw}");
        assert_eq!(ui.n, 1);
    }
}

#[test]
fn bordered_blocks_share_the_divider_and_their_inners_stay_off_it() {
    use ratatui::widgets::{Block, Borders};
    let r = Rect::new(0, 0, 40, 10);
    for orientation in [Orientation::Horizontal, Orientation::Vertical] {
        let s = split(Config {
            divider: 2,
            orientation,
            ..one()
        });
        let (a, div, b) = s.layout(r);
        let (ba, bb) = s.bordered(r);
        assert_eq!(ba.union(bb), r);
        assert_eq!(ba.intersection(bb).area(), 0, "the blocks overlap");
        let (ia, ib) = (
            Block::default().borders(Borders::ALL).inner(ba),
            Block::default().borders(Borders::ALL).inner(bb),
        );
        assert!(a.union(ia) == a && b.union(ib) == b, "inner off its pane");
        assert!(ia.intersection(div).is_empty() && ib.intersection(div).is_empty());
    }
}

#[test]
fn a_drag_marks_the_border_divider_heavy() {
    let mut s = split(Config {
        divider: 2,
        ..one()
    });
    let area = Rect::new(0, 0, 21, 3);
    let (_, div, _) = s.layout(area);
    let mut buf = Buffer::empty(area);
    buf[(div.x, 1)].set_symbol("│");
    s.mark_drag(&mut buf, div);
    assert_eq!(buf[(div.x, 1)].symbol(), "│", "idle stays light");
    s.mouse(&press(div.x, 1), area, Instant::now());
    s.mark_drag(&mut buf, div);
    assert_eq!(buf[(div.x, 1)].symbol(), "┃");
}
