//! Mouse text selection that stays inside one pane, the way Claude Code's mod
//! engine selects text. A port of the rules in the Go package
//! `github.com/keyolk/tuikit/selection`; there is no Rust kit, so the module
//! depends only on ratatui and crossterm and is meant to be copied verbatim.
//!
//! With mouse capture on, the terminal's own drag-to-select is gone, and
//! Shift-drag crosses every pane, border and the footer. Instead:
//!
//! - A press picks the pane under it; columns are clamped to that pane, so a
//!   drag never reaches a neighbor or a border. Rows run across lines inside
//!   the pane.
//! - Dragging past the top or bottom of a text pane asks the app to scroll it
//!   (`Outcome::scroll`), and the selection keeps the lines that scrolled
//!   out, because such a pane hands over its whole content (`Pane::lines`).
//! - In a `Text` pane a double click selects a word and a triple click a line.
//!   `List` panes keep the fleet convention (second click = Enter), so
//!   multi-click selection is off there.
//! - Release copies the text; the highlight stays until the next press or key.
//!
//! A press that does not move is still the app's click: `Outcome::consumed`
//! stays false so click-to-select and the wheel keep working.

use std::time::{Duration, Instant};

use crossterm::event::{MouseButton, MouseEvent, MouseEventKind};
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Modifier;
use unicode_width::UnicodeWidthStr;

#[path = "dashboard_selection_copy.rs"]
mod copy;
pub use copy::{Copied, copy_in_background};

/// Same window as a double click on a list row, so a word pick and a row
/// activation feel the same.
pub const MULTI_CLICK: Duration = Duration::from_millis(400);

/// What a double click means in a pane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The second click on the selected row is the Enter action.
    List,
    /// A double click selects a word, a triple click a line.
    Text,
}

/// One region a selection can live in, with the rect it was drawn into.
/// Panes are passed in drawing order; the last one containing a point wins, so
/// an overlay listed after the base owns its area.
#[derive(Clone, Debug)]
pub struct Pane {
    pub id: &'static str,
    pub rect: Rect,
    pub kind: Kind,
    /// The pane's full content as plain text, one entry per drawn line, and
    /// `offset` the index of the line drawn at `rect.y`. A pane with lines
    /// can be scrolled by a drag and keeps the lines that scroll out; a pane
    /// without them is read from the drawn buffer and cannot scroll.
    pub lines: Option<Vec<String>>,
    pub offset: usize,
}

impl Pane {
    pub fn new(id: &'static str, rect: Rect, kind: Kind) -> Self {
        Self {
            id,
            rect,
            kind,
            lines: None,
            offset: 0,
        }
    }

    pub fn with_lines(mut self, lines: Vec<String>, offset: usize) -> Self {
        self.lines = Some(lines);
        self.offset = offset;
        self
    }
}

/// A cell of a pane: `row` is a line of `Pane::lines` (or a row of the rect
/// when the pane has none), `col` a cell column inside the rect.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Point {
    pub row: usize,
    pub col: u16,
}

/// What `Selection::mouse` did with an event.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Outcome {
    /// The event belongs to the selection (a drag, the release ending one, a
    /// word or line pick); the app must not also treat it as a click.
    pub consumed: bool,
    /// The selected text, set once when a selection is finished.
    pub copy: Option<String>,
    /// Scroll pane `scroll_pane` by this many lines (negative is up), because
    /// a drag went past its edge.
    pub scroll: i32,
    pub scroll_pane: &'static str,
}

/// The selection state. The default has no selection.
#[derive(Clone, Debug, Default)]
pub struct Selection {
    pane: &'static str,
    anchor: Point,
    focus: Point,
    active: bool,
    dragging: bool,
    moved: bool,

    last_at: Option<Instant>,
    last_pane: &'static str,
    last_cell: Point,
    press_runs: u8,
}

impl Selection {
    /// Whether a selection is drawn.
    // Only tests read it here; kept so the module stays a copy of the other
    // fleet ports.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn active(&self) -> bool {
        self.active
    }

    /// Drop the selection. Called on every key: a key is the "next action"
    /// that ends the engine's selection. The multi-click memory survives, as
    /// in the Go kit, so only the range goes.
    pub fn clear(&mut self) {
        *self = Selection {
            last_at: self.last_at,
            last_pane: self.last_pane,
            last_cell: self.last_cell,
            press_runs: self.press_runs,
            ..Selection::default()
        };
    }

    /// The selection's pane and its start and end, start first.
    pub fn range(&self) -> Option<(&'static str, Point, Point)> {
        if !self.active {
            return None;
        }
        let (a, b) = (self.anchor, self.focus);
        if (b.row, b.col) < (a.row, a.col) {
            Some((self.pane, b, a))
        } else {
            Some((self.pane, a, b))
        }
    }

    /// Feed one mouse event. `frame` is the last drawn buffer, read for panes
    /// without lines; `now` is injectable for tests.
    pub fn mouse(
        &mut self,
        event: &MouseEvent,
        panes: &[Pane],
        frame: &Buffer,
        now: Instant,
    ) -> Outcome {
        match event.kind {
            MouseEventKind::Down(MouseButton::Left) => self.press(event, panes, frame, now),
            MouseEventKind::Drag(MouseButton::Left) if self.dragging => self.drag(event, panes),
            MouseEventKind::Up(_) if self.dragging => self.release(panes, frame),
            _ => Outcome::default(),
        }
    }

    fn press(&mut self, ev: &MouseEvent, panes: &[Pane], frame: &Buffer, now: Instant) -> Outcome {
        self.active = false;
        let Some(p) = pane_at(panes, ev.column, ev.row) else {
            self.dragging = false;
            self.press_runs = 0;
            return Outcome::default();
        };
        let cell = point_in(p, ev.column as i32, ev.row as i32);
        let again = self
            .last_at
            .is_some_and(|at| now.saturating_duration_since(at) <= MULTI_CLICK);
        if p.id == self.last_pane && cell == self.last_cell && again {
            self.press_runs = self.press_runs.saturating_add(1);
        } else {
            self.press_runs = 1;
        }
        self.last_at = Some(now);
        self.last_pane = p.id;
        self.last_cell = cell;
        self.pane = p.id;
        self.anchor = cell;
        self.focus = cell;
        self.dragging = true;
        self.moved = false;

        if p.kind != Kind::Text || self.press_runs < 2 {
            return Outcome::default();
        }
        let line = plain_line(p, cell.row, frame);
        let (lo, hi) = if self.press_runs == 2 {
            match word_at(&line, cell.col) {
                Some(span) => span,
                None => {
                    return Outcome {
                        consumed: true,
                        ..Outcome::default()
                    };
                }
            }
        } else {
            (0, (line.width() as u16).saturating_sub(1))
        };
        self.anchor = Point {
            row: cell.row,
            col: lo,
        };
        self.focus = Point {
            row: cell.row,
            col: hi,
        };
        self.active = true;
        self.dragging = false;
        if self.press_runs >= 3 {
            self.press_runs = 0; // a fourth click starts over instead of repeating
        }
        Outcome {
            consumed: true,
            copy: Some(self.text(panes, frame)),
            ..Outcome::default()
        }
    }

    fn drag(&mut self, ev: &MouseEvent, panes: &[Pane]) -> Outcome {
        let Some(p) = pane_by_id(panes, self.pane) else {
            return Outcome::default();
        };
        let mut out = Outcome {
            consumed: true,
            ..Outcome::default()
        };
        let (top, bottom) = (p.rect.y as i32, p.rect.y as i32 + p.rect.height as i32);
        let mut y = ev.row as i32;
        if let Some(lines) = &p.lines {
            if y < top && p.offset > 0 {
                out.scroll = -1;
                out.scroll_pane = p.id;
                y = top - 1;
            } else if y >= bottom && p.offset + (p.rect.height as usize) < lines.len() {
                out.scroll = 1;
                out.scroll_pane = p.id;
                y = bottom;
            }
        }
        let focus = point_in(p, ev.column as i32, y);
        if !self.moved && focus == self.anchor {
            return Outcome::default();
        }
        self.focus = focus;
        self.moved = true;
        self.active = true;
        out
    }

    fn release(&mut self, panes: &[Pane], frame: &Buffer) -> Outcome {
        self.dragging = false;
        if !self.moved {
            return Outcome::default();
        }
        Outcome {
            consumed: true,
            copy: Some(self.text(panes, frame)),
            ..Outcome::default()
        }
    }

    /// The selected text: lines joined with "\n", trailing blanks trimmed per
    /// line, only the selection's pane's columns.
    pub fn text(&self, panes: &[Pane], frame: &Buffer) -> String {
        let Some((id, a, b)) = self.range() else {
            return String::new();
        };
        let Some(p) = pane_by_id(panes, id) else {
            return String::new();
        };
        let mut out = Vec::new();
        for row in a.row..=b.row {
            let line = plain_line(p, row, frame);
            let lo = if row == a.row { a.col } else { 0 };
            let hi = if row == b.row {
                b.col
            } else {
                p.rect.width.saturating_sub(1)
            };
            out.push(cut(&line, lo, hi).trim_end().to_string());
        }
        out.join("\n")
    }

    /// Reverse the selected cells of the drawn buffer. Called last in draw so
    /// the highlight sits over whatever the panes drew; only cells inside the
    /// pane are touched, and rows scrolled out of view are skipped. Reverse
    /// rather than a colour so it survives NO_COLOR.
    pub fn paint(&self, buf: &mut Buffer, panes: &[Pane]) {
        let Some((id, a, b)) = self.range() else {
            return;
        };
        let Some(p) = pane_by_id(panes, id) else {
            return;
        };
        let r = p.rect.intersection(buf.area);
        for y in r.y..r.bottom() {
            let mut row = (y - p.rect.y) as usize;
            if p.lines.is_some() {
                row += p.offset;
            }
            if row < a.row || row > b.row {
                continue;
            }
            let lo = if row == a.row { a.col } else { 0 };
            let hi = if row == b.row {
                b.col
            } else {
                p.rect.width.saturating_sub(1)
            };
            let from = p.rect.x.saturating_add(lo).max(r.x);
            let to = p.rect.x.saturating_add(hi).min(r.right().saturating_sub(1));
            for x in from..=to {
                if from > to {
                    break;
                }
                let cell = &mut buf[(x, y)];
                cell.modifier.insert(Modifier::REVERSED);
            }
        }
    }
}

/// The rows `paragraph` draws at `width` with no scroll, as plain text: a
/// wrapped pane's full content in the same line breaks the screen shows, so
/// a drag can scroll it and keep the lines that scrolled out. Rendered into
/// a scratch buffer because ratatui does not expose its wrapped lines.
pub fn paragraph_lines(paragraph: &ratatui::widgets::Paragraph, width: u16) -> Vec<String> {
    use ratatui::widgets::Widget;
    let height = paragraph.line_count(width).min(u16::MAX as usize) as u16;
    let area = Rect::new(0, 0, width, height);
    let mut buf = Buffer::empty(area);
    paragraph.clone().render(area, &mut buf);
    let whole = Pane::new("scratch", area, Kind::Text);
    (0..height as usize)
        .map(|row| plain_line(&whole, row, &buf))
        .collect()
}

/// The engine's word class: letters, digits and the characters paths and
/// flags are made of, so a double click takes a whole path.
fn word_char(c: char) -> bool {
    c.is_alphanumeric() || "_/.-+~\\".contains(c)
}

/// The cell columns of the word covering column `col`.
fn word_at(line: &str, col: u16) -> Option<(u16, u16)> {
    let mut cells = Vec::new(); // (char, first column, last column)
    let mut x = 0u16;
    for c in line.chars() {
        let w = (unicode_width::UnicodeWidthChar::width(c).unwrap_or(0) as u16).max(1);
        cells.push((c, x, x + w - 1));
        x += w;
    }
    let at = cells
        .iter()
        .position(|&(_, lo, hi)| col >= lo && col <= hi)?;
    if !word_char(cells[at].0) {
        return None;
    }
    let mut first = at;
    while first > 0 && word_char(cells[first - 1].0) {
        first -= 1;
    }
    let mut last = at;
    while last + 1 < cells.len() && word_char(cells[last + 1].0) {
        last += 1;
    }
    Some((cells[first].1, cells[last].2))
}

/// The cells of `line` whose columns fall in `lo..=hi`. A wide character
/// counts when it starts inside the range.
fn cut(line: &str, lo: u16, hi: u16) -> String {
    let mut out = String::new();
    let mut x = 0u16;
    for c in line.chars() {
        let w = unicode_width::UnicodeWidthChar::width(c).unwrap_or(0) as u16;
        if x >= lo && x <= hi {
            out.push(c);
        }
        x += w;
        if x > hi {
            break;
        }
    }
    out
}

/// Row `row` of `p` as plain text, the pane's columns only.
fn plain_line(p: &Pane, row: usize, frame: &Buffer) -> String {
    if let Some(lines) = &p.lines {
        return lines
            .get(row)
            .map(|l| cut(l, 0, p.rect.width.saturating_sub(1)))
            .unwrap_or_default();
    }
    let y = p.rect.y as usize + row;
    let area = frame.area;
    if y < area.y as usize || y >= area.bottom() as usize {
        return String::new();
    }
    let y = y as u16;
    let mut out = String::new();
    let mut skip = 0;
    for x in p.rect.x..p.rect.right().min(area.right()) {
        if x < area.x {
            continue;
        }
        // ratatui resets the cells a wide character covers to " "; reading
        // them would put a space inside every CJK word.
        if skip > 0 {
            skip -= 1;
            continue;
        }
        let sym = frame[(x, y)].symbol();
        skip = sym.width().saturating_sub(1);
        out.push_str(sym);
    }
    out
}

/// Map a screen cell to a point of `p`, clamped to the pane: columns never
/// leave it (the engine's scope), rows stay within its lines.
fn point_in(p: &Pane, x: i32, y: i32) -> Point {
    let max_col = (p.rect.width as i32 - 1).max(0);
    let col = (x - p.rect.x as i32).clamp(0, max_col) as u16;
    let mut row = y - p.rect.y as i32;
    let rows = match &p.lines {
        Some(lines) => {
            row += p.offset as i32;
            lines.len() as i32
        }
        None => p.rect.height as i32,
    };
    let row = row.clamp(0, (rows - 1).max(0)) as usize;
    Point { row, col }
}

fn pane_at(panes: &[Pane], x: u16, y: u16) -> Option<&Pane> {
    panes
        .iter()
        .rev()
        .find(|p| p.rect.contains(ratatui::layout::Position { x, y }))
}

fn pane_by_id<'a>(panes: &'a [Pane], id: &str) -> Option<&'a Pane> {
    panes.iter().find(|p| p.id == id)
}

#[cfg(test)]
#[path = "dashboard_selection_tests.rs"]
mod tests;
