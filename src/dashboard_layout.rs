//! Cell-aware text cuts and scroll bounds for the dashboard panes.

use ratatui::layout::Rect;
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
