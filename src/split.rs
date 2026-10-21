//! A resizable two-pane split: the ratio, its clamps, the layout and the
//! gestures that change it. Pane content stays the app's to draw.
//!
//! A port of tuikit's Go `split` package (README "Split panes"), so every
//! fleet TUI resizes panes the same way:
//!
//! - The ratio is the fraction of the space left after the divider that pane
//!   A gets, never a cell count, so a terminal resize keeps the proportion.
//! - Every write clamps it: to `[lo, hi]`, and, when the area is known, to
//!   the ratios that leave each pane its minimum at that size. A key press
//!   steps from the divider position on screen and moves it at least one
//!   cell, so there is no dead zone of presses that change nothing.
//! - `layout` gives the divider's cells to neither pane, so hit tests and
//!   text selection on the panes never see a divider press.
//! - Mouse events go to the split before the selection: a press on the
//!   divider is the split's, a press in a pane never is.
//! - `<` shrinks pane A, `>` grows it, `=` resets; a double click on the
//!   divider resets too.
//!
//! Two lessons from the Go rollout are built in rather than left to each app:
//! the split remembers whether the user has moved it (`is_set`), so an
//! unmoved split keeps following the app's default rule and is not saved;
//! and `Outcome::reset` says when a double click reset it.

use std::time::{Duration, Instant};

use crossterm::event::{MouseButton, MouseEvent, MouseEventKind};
use ratatui::buffer::Buffer;
use ratatui::layout::{Position, Rect};
use ratatui::style::{Modifier, Style};
use serde::{Deserialize, Deserializer, Serialize};

/// How close two presses on the divider must be to count as a double click
/// (tuikit `mouse.DoubleClickWindow`).
pub const DOUBLE_CLICK: Duration = Duration::from_millis(400);

const DEFAULT_RATIO: f64 = 0.5;
const DEFAULT_STEP: f64 = 0.05;
const DEFAULT_LO: f64 = 0.1;
const DEFAULT_HI: f64 = 0.9;

/// The direction the two panes sit in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Orientation {
    /// A left of B; sizes are columns.
    #[default]
    Horizontal,
    /// A above B; sizes are rows.
    Vertical,
}

/// One pane.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Side {
    /// Left or top.
    #[default]
    A,
    /// Right or bottom.
    // Part of the shared contract; kmd keeps the list when it collapses.
    #[allow(dead_code)]
    B,
}

/// What the divider cells show when idle. A drag always draws a heavy line,
/// so the gesture is visible even where the idle divider is blank.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Look {
    #[default]
    Line,
    // Part of the shared contract (ccproxy-style blank dividers); kmd's
    // dividers are the panes' own borders.
    #[allow(dead_code)]
    Blank,
}

/// The split's key actions, bound in the app's one binding table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Shrink,
    Grow,
    Reset,
}

/// What the split did with an event or an action.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Outcome {
    /// The event was the split's: skip the selection and click handling.
    pub consumed: bool,
    /// The saved state changed (the ratio, or whether one is set): save it.
    pub changed: bool,
    /// A double click on the divider reset the ratio.
    pub reset: bool,
}

/// How a split is laid out and how far it may move: set in a literal and
/// passed to `Split::new`.
#[derive(Clone, Debug, Default)]
pub struct Config {
    pub orientation: Orientation,
    /// Smallest sizes, in cells, of each pane. A pane is never laid out
    /// narrower than 1 cell whatever these say.
    pub min_a: u16,
    pub min_b: u16,
    /// The divider's size in cells. Its cells belong to neither pane.
    pub divider: u16,
    /// Below this size only `primary` is shown, full size. 0 never
    /// collapses by size; a split whose minimums do not fit collapses anyway.
    pub collapse: u16,
    pub primary: Side,
    /// The fraction one key press moves the divider (0.05 when 0).
    pub step: f64,
    /// Widens the divider's press target by this many cells on each side.
    pub grab: u16,
    /// The ratio reset returns to (0.5 when 0).
    pub default: f64,
    /// Bounds of the ratio (0.1 and 0.9 when 0).
    pub lo: f64,
    pub hi: f64,
    // Read by `draw_divider` only, which kmd does not call (see `Look`).
    #[allow(dead_code)]
    pub look: Look,
}

/// A split's configuration and state, kept on the app's model. The ratio
/// and the drag change only through methods, which clamp.
#[derive(Clone, Debug, Default)]
pub struct Split {
    pub config: Config,
    /// `None` until the user moves the divider: the default then applies.
    ratio: Option<f64>,
    dragging: bool,
    /// Pointer offset inside the divider when the drag began, so grabbing a
    /// wide divider by its edge does not jump it.
    grab_off: i32,
    /// The last divider press that may be the first half of a double click.
    last_press: Option<Instant>,
}

/// What an app persists for the split, inside its UI state file.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct State {
    pub ratio: f64,
}

/// Reads a saved `State` without failing the file around it: a missing,
/// mistyped or garbage value is `None`, so a broken ratio costs the ratio
/// and not the rest of the UI state. Range checks happen in `restore`.
pub fn lenient<'de, D: Deserializer<'de>>(d: D) -> Result<Option<State>, D::Error> {
    let v = serde_json::Value::deserialize(d)?;
    Ok(v.get("ratio")
        .and_then(serde_json::Value::as_f64)
        .map(|ratio| State { ratio }))
}

fn valid_fraction(v: f64) -> bool {
    v.is_finite() && v > 0.0 && v < 1.0
}

impl Split {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            ..Self::default()
        }
    }

    fn bounds(&self) -> (f64, f64) {
        let lo = if valid_fraction(self.config.lo) {
            self.config.lo
        } else {
            DEFAULT_LO
        };
        let hi = if valid_fraction(self.config.hi) {
            self.config.hi
        } else {
            DEFAULT_HI
        };
        (lo, hi.max(lo))
    }

    fn clamp_bounds(&self, v: f64) -> f64 {
        let (lo, hi) = self.bounds();
        v.clamp(lo, hi)
    }

    fn default_ratio(&self) -> f64 {
        let d = if valid_fraction(self.config.default) {
            self.config.default
        } else {
            DEFAULT_RATIO
        };
        self.clamp_bounds(d)
    }

    /// The fraction of the space after the divider that A gets.
    pub fn ratio(&self) -> f64 {
        self.clamp_bounds(self.ratio.unwrap_or_else(|| self.default_ratio()))
    }

    /// Whether the user has moved the divider since the default (a reset
    /// returns to unset). An unset split follows the app's default rule and
    /// has nothing worth saving.
    // Read by the persistence tests; `state()` is what the app saves.
    #[allow(dead_code)]
    pub fn is_set(&self) -> bool {
        self.ratio.is_some()
    }

    /// Whether a divider drag is in progress.
    // kmd marks the drag through `mark_drag`; tests read this.
    #[allow(dead_code)]
    pub fn dragging(&self) -> bool {
        self.dragging
    }

    fn size(&self, r: Rect) -> i32 {
        i32::from(match self.config.orientation {
            Orientation::Horizontal => r.width,
            Orientation::Vertical => r.height,
        })
    }

    /// The range of A's size that leaves both panes their minimum, and the
    /// space they share; `ok` is false when they do not both fit.
    fn cells(&self, r: Rect) -> (i32, i32, i32, bool) {
        let avail = self.size(r) - i32::from(self.config.divider);
        let lo = i32::from(self.config.min_a.max(1));
        let hi = avail - i32::from(self.config.min_b.max(1));
        (lo, hi, avail, r.width > 0 && r.height > 0 && lo <= hi)
    }

    /// Whether `r` shows only the primary pane: it is under `collapse`, or
    /// too small for both minimums and the divider.
    pub fn collapsed(&self, r: Rect) -> bool {
        let (_, _, _, ok) = self.cells(r);
        !ok || (self.config.collapse > 0 && self.size(r) < i32::from(self.config.collapse))
    }

    /// Clamps to the bounds and, when `r` fits both panes, to the ratios
    /// whose layout respects the minimums at `r`: a stored ratio is then
    /// always one the screen shows, so the next key press moves the divider.
    fn clamp_for(&self, v: f64, r: Rect) -> f64 {
        let v = self.clamp_bounds(v);
        let (lo, hi, avail, ok) = self.cells(r);
        if ok && !self.collapsed(r) {
            return v.clamp(
                f64::from(lo) / f64::from(avail),
                f64::from(hi) / f64::from(avail),
            );
        }
        v
    }

    fn size_a(&self, r: Rect) -> i32 {
        let (lo, hi, avail, _) = self.cells(r);
        ((self.ratio() * f64::from(avail)).round() as i32).clamp(lo, hi)
    }

    /// Cuts `r` into pane A, the divider and pane B. The divider's cells are
    /// in neither pane. A hidden pane (collapsed split) is the zero Rect, and
    /// so is the divider then.
    pub fn layout(&self, r: Rect) -> (Rect, Rect, Rect) {
        if r.width == 0 || r.height == 0 {
            return Default::default();
        }
        if self.collapsed(r) {
            return match self.config.primary {
                Side::A => (r, Rect::default(), Rect::default()),
                Side::B => (Rect::default(), Rect::default(), r),
            };
        }
        let (_, _, avail, _) = self.cells(r);
        // All three fit in r (cells checked it), so these never overflow u16.
        let na = self.size_a(r) as u16;
        let nb = (avail as u16) - na;
        let d = self.config.divider;
        match self.config.orientation {
            Orientation::Horizontal => (
                Rect { width: na, ..r },
                Rect {
                    x: r.x + na,
                    width: d,
                    ..r
                },
                Rect {
                    x: r.x + na + d,
                    width: nb,
                    ..r
                },
            ),
            Orientation::Vertical => (
                Rect { height: na, ..r },
                Rect {
                    y: r.y + na,
                    height: d,
                    ..r
                },
                Rect {
                    y: r.y + na + d,
                    height: nb,
                    ..r
                },
            ),
        }
    }

    /// Stores ratio `v`, clamped for `area` (pass `Rect::default()` when the
    /// size is unknown: the bounds still apply). NaN and infinities are
    /// ignored.
    pub fn set(&mut self, v: f64, area: Rect) {
        if v.is_finite() {
            self.ratio = Some(self.clamp_for(v, area));
        }
    }

    /// Back to the default, and to following it: an unset split is not saved.
    pub fn reset(&mut self) {
        self.ratio = None;
    }

    /// Moves the divider one step toward B, at least one cell.
    pub fn grow(&mut self, area: Rect) {
        self.nudge(1, area);
    }

    /// Moves the divider one step toward A, at least one cell.
    pub fn shrink(&mut self, area: Rect) {
        self.nudge(-1, area);
    }

    fn nudge(&mut self, dir: i32, area: Rect) {
        let step = if self.config.step > 0.0 && self.config.step.is_finite() {
            self.config.step
        } else {
            DEFAULT_STEP
        };
        let (_, _, avail, ok) = self.cells(area);
        if !ok || self.collapsed(area) {
            self.set(self.ratio() + f64::from(dir) * step, Rect::default());
            return;
        }
        // Step from what the screen shows, not from the stored ratio, and by
        // at least one cell: a step smaller than a cell on a narrow terminal
        // would otherwise be a press that does nothing.
        let avail_f = f64::from(avail);
        let cur = self.size_a(area);
        let mut next =
            ((f64::from(cur) / avail_f + f64::from(dir) * step) * avail_f).round() as i32;
        if next == cur {
            next = cur + dir;
        }
        self.set(f64::from(next) / avail_f, area);
    }

    /// Runs a key action against the area the split is laid out in.
    pub fn handle(&mut self, action: Action, area: Rect) -> Outcome {
        let before = self.ratio;
        match action {
            Action::Grow => self.grow(area),
            Action::Shrink => self.shrink(area),
            Action::Reset => self.reset(),
        }
        Outcome {
            consumed: true,
            changed: self.ratio != before,
            reset: action == Action::Reset,
        }
    }

    fn along(&self, x: u16, y: u16) -> i32 {
        i32::from(match self.config.orientation {
            Orientation::Horizontal => x,
            Orientation::Vertical => y,
        })
    }

    fn grab_contains(&self, div: Rect, x: u16, y: u16) -> bool {
        let g = i32::from(self.config.grab);
        if div.width == 0 && div.height == 0 {
            return false;
        }
        let (x, y) = (i32::from(x), i32::from(y));
        let (dx, dy, dw, dh) = (
            i32::from(div.x),
            i32::from(div.y),
            i32::from(div.width),
            i32::from(div.height),
        );
        let (x0, y0, x1, y1) = match self.config.orientation {
            Orientation::Horizontal => (dx - g, dy, dx + dw + g, dy + dh),
            Orientation::Vertical => (dx, dy - g, dx + dw, dy + dh + g),
        };
        x >= x0 && x < x1 && y >= y0 && y < y1
    }

    /// Feeds one mouse event, with `area` the rect the split is laid out in
    /// (the same one drawing passes to `layout`). Call it before the
    /// selection and skip both selection and clicks when `consumed`.
    ///
    /// A left press on the divider starts a drag; motion with the left
    /// button moves the divider, clamped; a release, or motion with no button
    /// (the release was lost outside the window), ends it. A second press on
    /// the divider within `DOUBLE_CLICK`, with no drag between, resets.
    pub fn mouse(&mut self, ev: &MouseEvent, area: Rect, now: Instant) -> Outcome {
        let consumed = Outcome {
            consumed: true,
            ..Outcome::default()
        };
        match ev.kind {
            MouseEventKind::Down(MouseButton::Left) => {
                let (_, div, _) = self.layout(area);
                if !self.grab_contains(div, ev.column, ev.row) {
                    self.dragging = false;
                    self.last_press = None;
                    return Outcome::default();
                }
                let double = self
                    .last_press
                    .is_some_and(|t| now.saturating_duration_since(t) <= DOUBLE_CLICK);
                if double {
                    let before = self.ratio;
                    self.reset();
                    self.dragging = false;
                    // A third press starts over rather than resetting again.
                    self.last_press = None;
                    return Outcome {
                        consumed: true,
                        changed: self.ratio != before,
                        reset: true,
                    };
                }
                self.dragging = true;
                self.last_press = Some(now);
                self.grab_off = self.along(ev.column, ev.row) - self.along(div.x, div.y);
                consumed
            }
            MouseEventKind::Drag(MouseButton::Left) if self.dragging => {
                let (_, _, avail, ok) = self.cells(area);
                if !ok || self.collapsed(area) {
                    return consumed;
                }
                let before = self.ratio;
                let na = self.along(ev.column, ev.row) - self.grab_off - self.along(area.x, area.y);
                self.set(f64::from(na) / f64::from(avail), area);
                let changed = self.ratio != before;
                if changed {
                    // A press that went on to drag is not the first half of a
                    // double click: re-grabbing right after a drag must not
                    // reset.
                    self.last_press = None;
                }
                Outcome {
                    changed,
                    ..consumed
                }
            }
            MouseEventKind::Up(_) | MouseEventKind::Moved if self.dragging => {
                self.dragging = false;
                consumed
            }
            _ => Outcome::default(),
        }
    }

    /// The rects for two bordered blocks that share a 2-cell divider: each
    /// pane grown by one cell into it, so A's far border and B's near border
    /// are the divider's cells. This keeps the look of two adjacent bordered
    /// blocks while the pane rects (and their inners) stay off the divider.
    /// With the divider hidden (collapsed) the panes are returned as laid out.
    pub fn bordered(&self, r: Rect) -> (Rect, Rect) {
        let (a, div, b) = self.layout(r);
        if div.is_empty() {
            return (a, b);
        }
        match self.config.orientation {
            Orientation::Horizontal => (
                Rect {
                    width: a.width + 1,
                    ..a
                },
                Rect {
                    x: b.x - 1,
                    width: b.width + 1,
                    ..b
                },
            ),
            Orientation::Vertical => (
                Rect {
                    height: a.height + 1,
                    ..a
                },
                Rect {
                    y: b.y - 1,
                    height: b.height + 1,
                    ..b
                },
            ),
        }
    }

    /// While a drag is in progress, redraws the borders already in `div`
    /// heavy and bold, so the divider being dragged stands out even under
    /// `NO_COLOR`. For dividers made of the panes' own borders.
    pub fn mark_drag(&self, buf: &mut Buffer, div: Rect) {
        if !self.dragging {
            return;
        }
        let div = div.intersection(buf.area);
        for y in div.top()..div.bottom() {
            for x in div.left()..div.right() {
                let cell = &mut buf[Position::new(x, y)];
                let heavy = match cell.symbol() {
                    "│" => "┃",
                    "─" => "━",
                    "┌" | "╭" => "┏",
                    "┐" | "╮" => "┓",
                    "└" | "╰" => "┗",
                    "┘" | "╯" => "┛",
                    _ => "",
                };
                if !heavy.is_empty() {
                    cell.set_symbol(heavy);
                }
                cell.modifier.insert(Modifier::BOLD);
            }
        }
    }

    /// Draws the divider into `div`: `glyph` (the theme's vertical or
    /// horizontal line) centred across it, or blanks for `Look::Blank`.
    /// While dragging the line is heavy and bold, so the highlight survives
    /// `NO_COLOR`.
    // Shared contract for a divider of its own; kmd's is two block borders.
    #[allow(dead_code)]
    pub fn draw_divider(&self, buf: &mut Buffer, div: Rect, glyph: &str, style: Style) {
        let div = div.intersection(buf.area);
        if div.is_empty() {
            return;
        }
        let (glyph, style) = if self.dragging {
            let heavy = match glyph {
                "│" | "|" => "┃",
                "─" | "-" => "━",
                g => g,
            };
            (heavy, style.add_modifier(Modifier::BOLD))
        } else {
            (glyph, style)
        };
        let blank = self.config.look == Look::Blank && !self.dragging;
        for y in div.top()..div.bottom() {
            for x in div.left()..div.right() {
                let on_line = match self.config.orientation {
                    Orientation::Horizontal => x == div.x + (div.width - 1) / 2,
                    Orientation::Vertical => y == div.y + (div.height - 1) / 2,
                };
                let cell = &mut buf[Position::new(x, y)];
                if on_line && !blank {
                    cell.set_symbol(glyph).set_style(style);
                } else {
                    cell.set_symbol(" ");
                }
            }
        }
    }

    /// The persistable state, `None` while the user has not moved the
    /// divider (so a later change of the default reaches them).
    pub fn state(&self) -> Option<State> {
        self.ratio.map(|_| State {
            ratio: self.ratio(),
        })
    }

    /// Applies a saved state, clamped to the bounds. A ratio that is not a
    /// fraction strictly between 0 and 1 (NaN, a percentage like 50) is
    /// rejected: nothing changes and the result is false. The minimum sizes
    /// are enforced again at the next layout and key press.
    pub fn restore(&mut self, s: State) -> bool {
        if !valid_fraction(s.ratio) {
            return false;
        }
        self.ratio = Some(self.clamp_bounds(s.ratio));
        true
    }
}

#[cfg(test)]
#[path = "split_tests.rs"]
mod tests;
