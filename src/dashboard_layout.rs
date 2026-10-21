//! Cell-aware text cuts and scroll bounds for the dashboard panes.

use crate::split::{Config, Orientation, Split};
use ratatui::layout::{Position, Rect};
use ratatui::widgets::Paragraph;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

/// Cuts `value` to at most `max` terminal cells, ending with `…` when cut.
/// Counting chars made a Korean summary twice as wide as its budget, because
/// each Hangul syllable takes two cells.
pub fn truncate_to_width(value: &str, max: usize) -> String {
    if value.width() <= max {
        return value.to_string();
    }
    if max == 0 {
        return String::new();
    }
    let budget = max - 1; // one cell for the ellipsis
    let mut used = 0;
    let mut output = String::new();
    for ch in value.chars() {
        let width = ch.width().unwrap_or(0);
        if used + width > budget {
            break;
        }
        used += width;
        output.push(ch);
    }
    output.push('…');
    output
}

/// Largest useful scroll offset for a wrapped paragraph drawn in `inner`, and
/// `requested` clamped to it. `G` and repeated PgDn used to scroll past the end
/// into an empty pane, and `k` then had to climb back from 65535 one row at a time.
pub fn clamp_scroll(paragraph: &Paragraph, inner: Rect, requested: u16) -> (u16, u16) {
    let lines = paragraph.line_count(inner.width);
    let max = u16::try_from(lines.saturating_sub(inner.height as usize)).unwrap_or(u16::MAX);
    (requested.min(max), max)
}

/// Which resizable split a view drew, so the mouse and the split keys move
/// the one on screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitId {
    /// Sessions: list | detail.
    Sessions,
    /// Simulator from `WIDE_WIDTH` up: hits | detail side by side.
    SimWide,
    /// Simulator below it: hits above detail.
    SimStacked,
}

/// The Sessions list | detail split, 34/66 as it was fixed before. The
/// divider is the two adjacent block borders (see `Split::bordered`).
pub fn sessions_split() -> Split {
    Split::new(Config {
        divider: 2,
        default: 0.34,
        min_a: 16,
        min_b: 20,
        lo: 0.15,
        hi: 0.85,
        ..Config::default()
    })
}

/// The Simulator's hits | detail split on a wide terminal, 42/58.
pub fn sim_wide_split() -> Split {
    Split::new(Config {
        divider: 2,
        default: 0.42,
        min_a: 20,
        min_b: 30,
        lo: 0.15,
        hi: 0.85,
        ..Config::default()
    })
}

/// The Simulator's hits above detail on a narrow terminal, 45/55. Its own
/// ratio: rows and columns are not the same proportion.
pub fn sim_stacked_split() -> Split {
    Split::new(Config {
        orientation: Orientation::Vertical,
        divider: 2,
        default: 0.45,
        min_a: 2,
        min_b: 2,
        lo: 0.15,
        hi: 0.85,
        ..Config::default()
    })
}

/// Hit regions the Simulator tab drew last, for the dashboard's mouse handler.
#[derive(Clone, Debug, Default)]
pub struct SimLayout {
    /// The hits | detail split drawn, and the area it was laid out in.
    pub split: Option<(SplitId, Rect)>,
    pub query: Option<Rect>,
    pub hits: Option<ListHit>,
    pub detail: Option<Rect>,
    /// The detail pane's wrapped lines and the first one drawn, for drag
    /// selection (it scrolls the pane and keeps scrolled-out lines).
    pub detail_lines: Vec<String>,
    pub detail_offset: usize,
}

/// A list as ratatui drew it: `rows[i]` is what screen row `first + i` of
/// `area` holds (`None` for a header row that is not an item).
#[derive(Clone, Debug, Default)]
pub struct ListHit {
    pub area: Rect,
    pub first: usize,
    pub row_height: u16,
    pub rows: Vec<Option<usize>>,
}

impl ListHit {
    /// The item under (x, y), if any.
    pub fn item_at(&self, x: u16, y: u16) -> Option<usize> {
        if !self.area.contains(Position { x, y }) || self.row_height == 0 {
            return None;
        }
        let row = self.first + usize::from((y - self.area.y) / self.row_height);
        self.rows.get(row).copied().flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::widgets::Wrap;

    #[test]
    fn truncates_by_cells_not_chars() {
        assert_eq!(truncate_to_width("가나다라", 8), "가나다라");
        let cut = truncate_to_width("가나다라", 5);
        assert_eq!(cut, "가나…");
        assert!(cut.width() <= 5);
        assert_eq!(truncate_to_width("abcdef", 4), "abc…");
        assert_eq!(truncate_to_width("abc", 0), "");
    }

    #[test]
    fn scroll_stops_at_the_last_full_page() {
        let text = (0..30).map(|i| format!("line {i}\n")).collect::<String>();
        let paragraph = Paragraph::new(text).wrap(Wrap { trim: false });
        let inner = Rect::new(0, 0, 40, 10);
        assert_eq!(clamp_scroll(&paragraph, inner, u16::MAX), (20, 20));
        assert_eq!(clamp_scroll(&paragraph, inner, 3), (3, 20));
    }
}
