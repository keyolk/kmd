//! Mouse support for the dashboard: keyboard-first, mouse as augmentation.
//!
//! Every gesture resolves to an existing keymap `Action` and runs through
//! `App::apply`, so a click does exactly what the key it stands for does.
//! Hit regions come from `MouseMap`, which `draw` fills with the rects and
//! list offsets it actually used — ratatui's `List` picks its own scroll
//! offset while rendering, so recomputing it here would drift from what is on
//! screen.
//!
//! - tab title: switch to that view (`ViewSessions`/…).
//! - Sessions list: click selects; wheel moves three sessions.
//! - a scrolled pane (session detail, Operations): wheel scrolls three lines.
//! - Simulator: click on the query box starts typing (`SimType`); click on a
//!   hit selects it; wheel over hits moves the hit, over detail scrolls it.
//! - help overlay: owns the mouse while open — wheel scrolls it, a click
//!   outside closes it, nothing reaches the view beneath.

use std::cell::RefCell;

use crossterm::event::{MouseButton, MouseEvent, MouseEventKind};
use ratatui::layout::{Position, Rect};

use super::*;
use crate::dashboard_layout::ListHit;

const WHEEL_ROWS: usize = 3;

/// Hit regions as drawn in the last frame.
#[derive(Clone, Debug, Default)]
pub(super) struct MouseMap {
    /// Each tab title's cells, in tab order.
    pub tabs: Vec<Rect>,
    /// Sessions list inner area, the first visible session and lines per row.
    pub sessions: Option<ListHit>,
    /// The scrollable paragraph pane (session detail or Operations body).
    pub scrolled: Option<Rect>,
    pub help: Option<Rect>,
    /// The list | detail split drawn, and the area it was laid out in.
    pub split: Option<(crate::dashboard_layout::SplitId, Rect)>,
    /// Panes a drag can select in, in drawing order (`dashboard_drag`).
    pub panes: Vec<crate::dashboard_selection::Pane>,
    /// The frame as drawn, before the highlight: text of panes without
    /// their own lines is read from it.
    pub frame: ratatui::buffer::Buffer,
}

pub(super) type MouseCell = RefCell<MouseMap>;

fn wheel(kind: MouseEventKind) -> Option<bool> {
    match kind {
        MouseEventKind::ScrollDown => Some(true),
        MouseEventKind::ScrollUp => Some(false),
        _ => None,
    }
}

fn contains(rect: Option<Rect>, x: u16, y: u16) -> bool {
    rect.is_some_and(|r| r.contains(Position { x, y }))
}

impl App {
    /// Only presses and wheel steps act; release, drag and motion do not.
    pub(super) fn handle_mouse(&mut self, event: MouseEvent) -> bool {
        let map = self.mouse_map.borrow().clone();
        let (x, y) = (event.column, event.row);
        let down = wheel(event.kind);
        let left = matches!(event.kind, MouseEventKind::Down(MouseButton::Left));
        if down.is_none() && !left {
            return false;
        }

        if self.show_help {
            match down {
                Some(true) => self.repeat(Action::HelpDown),
                Some(false) => self.repeat(Action::HelpUp),
                None if !contains(map.help, x, y) => self.apply(Action::HelpClose),
                None => false,
            };
            return true;
        }

        if self.tab == SIMULATOR_TAB && self.simulator.typing() {
            // Typing owns input until Esc, as it does for keys: a tab key is
            // text there, so a tab click must not switch views either.
            return false;
        }

        if left && let Some(tab) = map.tabs.iter().position(|r| r.contains(Position { x, y })) {
            self.apply(match tab {
                SESSIONS_TAB => Action::ViewSessions,
                SIMULATOR_TAB => Action::ViewSimulator,
                _ => Action::ViewOperations,
            });
            return true;
        }

        if self.tab == SIMULATOR_TAB {
            return self.simulator_mouse(&map, x, y, down, left);
        }

        if let Some(list) = map
            .sessions
            .as_ref()
            .filter(|l| l.area.contains(Position { x, y }))
        {
            match down {
                Some(true) => self.repeat(Action::SessionNext),
                Some(false) => self.repeat(Action::SessionPrev),
                None => {
                    if let Some(index) = list.item_at(x, y) {
                        self.select_session(index);
                    }
                    false
                }
            };
            return true;
        }
        if contains(map.scrolled, x, y) {
            match down {
                Some(true) => self.repeat(Action::ScrollDown),
                Some(false) => self.repeat(Action::ScrollUp),
                None => false,
            };
            return true;
        }
        false
    }

    fn simulator_mouse(
        &mut self,
        _map: &MouseMap,
        x: u16,
        y: u16,
        down: Option<bool>,
        left: bool,
    ) -> bool {
        let map = self.simulator.layout.borrow().clone();
        if left && contains(map.query, x, y) {
            self.apply(Action::SimType);
            return true;
        }
        if let Some(hits) = map
            .hits
            .as_ref()
            .filter(|l| l.area.contains(Position { x, y }))
        {
            match down {
                Some(true) => self.repeat(Action::SimHitNext),
                Some(false) => self.repeat(Action::SimHitPrev),
                None => {
                    if let Some(hit) = hits.item_at(x, y) {
                        self.simulator.select_hit(hit);
                    }
                    false
                }
            };
            return true;
        }
        if contains(map.detail, x, y) {
            match down {
                Some(true) => self.repeat(Action::SimDetailDown),
                Some(false) => self.repeat(Action::SimDetailUp),
                None => false,
            };
            return true;
        }
        false
    }

    /// One wheel notch is three key presses of the same action.
    fn repeat(&mut self, action: Action) -> bool {
        for _ in 0..WHEEL_ROWS {
            self.apply(action);
        }
        false
    }
}

/// Where ratatui's `Tabs` puts each title inside `inner`: one cell of padding
/// either side and a one-cell divider between titles (ratatui's defaults).
pub(super) fn tab_rects(inner: Rect, titles: &[String]) -> Vec<Rect> {
    use unicode_width::UnicodeWidthStr;
    let mut x = inner.x;
    let mut out = Vec::with_capacity(titles.len());
    for title in titles {
        let width = title.width() as u16 + 2; // padding left + right
        out.push(Rect {
            x,
            y: inner.y,
            width: width.min(inner.right().saturating_sub(x)),
            height: inner.height,
        });
        x = x.saturating_add(width + 1); // divider
    }
    out
}
