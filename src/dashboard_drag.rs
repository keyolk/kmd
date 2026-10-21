//! Drag selection wiring for the dashboard: which panes a drag may select in,
//! recorded while `draw` lays them out (like `MouseMap`), and what a
//! selection event does to the app. The rules live in `dashboard_selection`.
//!
//! - tabs, Sessions list, Simulator hits, footer: `List` (clicks unchanged)
//! - guide, scrolled pane (session detail / Operations), Simulator query and
//!   detail, help: `Text` (double click = word, triple = line)
//! - while help is open only help is selectable.

use std::sync::mpsc::{Receiver, Sender};

use crossterm::event::MouseEvent;

use super::*;
pub(super) use crate::dashboard_selection::Kind;
use crate::dashboard_selection::{Copied, Pane};

pub(super) type CopiedChannel = (Sender<Copied>, Receiver<Copied>);

/// Record a pane read from the drawn frame.
pub(super) fn record(app: &App, id: &'static str, rect: Rect, kind: Kind) {
    app.mouse_map
        .borrow_mut()
        .panes
        .push(Pane::new(id, rect, kind));
}

/// Record a scrolled paragraph pane with its whole wrapped content, so a drag
/// past its edge can scroll it. `paragraph` is the one drawn into `inner`
/// (without its block) and `scroll` the clamped offset it was drawn at.
pub(super) fn record_text(
    app: &App,
    id: &'static str,
    inner: Rect,
    paragraph: &Paragraph,
    scroll: u16,
) {
    let lines = crate::dashboard_selection::paragraph_lines(paragraph, inner.width);
    app.mouse_map
        .borrow_mut()
        .panes
        .push(Pane::new(id, inner, Kind::Text).with_lines(lines, usize::from(scroll)));
}

/// End of `draw`: complete the pane list, keep the frame for reading, and
/// paint the highlight over everything.
pub(super) fn finish(frame: &mut Frame, app: &App, tabs: Rect, guide: Rect, footer: Rect) {
    let mut map = app.mouse_map.borrow_mut();
    let mut panes = vec![
        Pane::new("tabs", tabs, Kind::List),
        Pane::new("guide", guide, Kind::Text),
    ];
    if app.tab == SIMULATOR_TAB {
        let sim = app.simulator.layout.borrow();
        let inner = |r: Rect| Block::default().borders(Borders::ALL).inner(r);
        if let Some(query) = sim.query {
            // Selecting in the input never moves its cursor or edits it: the
            // selection only reads cells.
            panes.push(Pane::new("sim_query", inner(query), Kind::Text));
        }
        if let Some(hits) = &sim.hits {
            panes.push(Pane::new("sim_hits", hits.area, Kind::List));
        }
        if let Some(detail) = sim.detail {
            panes.push(
                Pane::new("sim_detail", inner(detail), Kind::Text)
                    .with_lines(sim.detail_lines.clone(), sim.detail_offset),
            );
        }
    }
    panes.append(&mut map.panes);
    panes.push(Pane::new("footer", footer, Kind::List));
    if app.show_help {
        // The base under the overlay must not be selectable.
        panes.retain(|p| p.id == "help");
    }
    map.frame = frame.buffer_mut().clone();
    app.selection.paint(frame.buffer_mut(), &panes);
    map.panes = panes;
}

impl App {
    /// One mouse event: the split first, then the selection, then, unless
    /// one of them consumed the event, the click handling it had before.
    /// Returns text to copy.
    pub(super) fn on_mouse(&mut self, event: MouseEvent, now: Instant) -> Option<String> {
        // Help owns the mouse while open; the split under it is not there.
        let split = self.mouse_map.borrow().split.filter(|_| !self.show_help);
        if let Some((id, area)) = split
            && self.split_mut(id).mouse(&event, area, now).consumed
        {
            // The press never reaches the selection, which would otherwise
            // have dropped its old highlight: drop it here, or a stale
            // highlight is painted over panes that just moved.
            if matches!(
                event.kind,
                crossterm::event::MouseEventKind::Down(crossterm::event::MouseButton::Left)
            ) {
                self.selection.clear();
            }
            return None;
        }
        let map = self.mouse_map.borrow();
        let out = self.selection.mouse(&event, &map.panes, &map.frame, now);
        let offset = map
            .panes
            .iter()
            .find(|p| p.id == out.scroll_pane)
            .map_or(0, |p| p.offset);
        drop(map);
        if out.scroll != 0 {
            self.scroll_pane(out.scroll_pane, offset, out.scroll);
        }
        if !out.consumed {
            self.handle_mouse(event);
        }
        out.copy.filter(|text| !text.is_empty())
    }

    /// Scroll a pane by one drag step from the offset it was drawn at, through
    /// the same field its keys and wheel move.
    fn scroll_pane(&mut self, id: &str, offset: usize, delta: i32) {
        let next = u16::try_from((offset as i64 + i64::from(delta)).max(0)).unwrap_or(u16::MAX);
        match id {
            "scrolled" => self.scroll = next,
            "help" => self.help_scroll = next,
            "sim_detail" => self.simulator.detail_scroll = next,
            _ => {}
        }
    }

    /// Show finished clipboard writes on the status line. True when one landed.
    pub(super) fn take_copied(&mut self) -> bool {
        let mut any = false;
        while let Ok(done) = self.copied.1.try_recv() {
            let level = if done.error.is_some() {
                Level::Error
            } else {
                Level::Success
            };
            self.status.set("copy", level, done.message());
            any = true;
        }
        any
    }
}
