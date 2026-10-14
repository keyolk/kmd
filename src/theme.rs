//! Semantic style roles shared by every kmd TUI: the dashboard, its Simulator
//! view, and the standalone `kmd sim`.
//!
//! Render code names a role (`heading`, `warning`, `selected`), never a raw
//! color, so hues are chosen in one place. This replaces two modules that had
//! drifted apart: `palette.rs` (free functions, `LightGreen`/`DarkGray`) for the
//! Simulator, and `dashboard_theme.rs` (`Theme`, `Green`/`Gray`) for the rest
//! of the dashboard, so the same "success" or "muted" text rendered in two
//! different colors depending on the view.
//!
//! `NO_COLOR` drops hues only. Bold and reverse are kept, so headings and the
//! selected row stay distinguishable on a monochrome terminal. Every colored
//! state is also carried by a word or symbol (`● OK`, `GATED`, `✗`).

use ratatui::style::{Color, Modifier, Style};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Theme {
    colors: bool,
}

impl Theme {
    pub fn from_env() -> Self {
        Self {
            colors: std::env::var_os("NO_COLOR").is_none(),
        }
    }

    #[cfg(test)]
    pub fn colored() -> Self {
        Self { colors: true }
    }

    #[cfg(test)]
    pub fn plain() -> Self {
        Self { colors: false }
    }

    fn fg(self, color: Color) -> Style {
        if self.colors {
            Style::default().fg(color)
        } else {
            Style::default()
        }
    }

    /// Section titles, block titles, and key names in help.
    pub fn heading(self) -> Style {
        self.fg(Color::Cyan).add_modifier(Modifier::BOLD)
    }

    /// Field labels inside a section (`score:`, `title:`).
    pub fn label(self) -> Style {
        self.fg(Color::Cyan)
    }

    /// Primary values. Left unstyled so the terminal's own foreground applies.
    pub fn value(self) -> Style {
        Style::default()
    }

    /// Secondary metadata: paths, timestamps, hints.
    pub fn muted(self) -> Style {
        // Gray, not DarkGray: DarkGray disappears on many dark terminal themes.
        self.fg(Color::Gray)
    }

    /// Identifiers the user selects by (session ids).
    pub fn accent(self) -> Style {
        self.fg(Color::Cyan)
    }

    /// Categories and kinds: retrieval mode, collection, axis, evaluation kind.
    pub fn kind(self) -> Style {
        self.fg(Color::Magenta)
    }

    /// Neutral status and informational markers.
    pub fn info(self) -> Style {
        self.fg(Color::LightBlue)
    }

    pub fn success(self) -> Style {
        self.fg(Color::Green)
    }

    pub fn warning(self) -> Style {
        self.fg(Color::Yellow)
    }

    pub fn error(self) -> Style {
        self.fg(Color::Red)
    }

    /// ok/fail styling, always paired with a word or symbol by the caller.
    pub fn state(self, ok: bool) -> Style {
        if ok { self.success() } else { self.error() }
    }

    /// Unfocused panel borders.
    pub fn border(self) -> Style {
        self.muted()
    }

    /// Focused panel borders: focus changes the border color, never its presence.
    pub fn border_focus(self) -> Style {
        self.accent()
    }

    /// Selected list row and active tab. Reverse carries the selection when
    /// colors are off.
    pub fn selected(self) -> Style {
        self.accent()
            .add_modifier(Modifier::BOLD | Modifier::REVERSED)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_maps_to_success_and_failure() {
        let theme = Theme::colored();
        assert_eq!(theme.state(true), theme.success());
        assert_eq!(theme.state(false), theme.error());
    }

    #[test]
    fn value_keeps_the_terminal_default() {
        assert_eq!(Theme::colored().value(), Style::default());
    }

    #[test]
    fn no_color_drops_hues_but_keeps_bold_and_reverse() {
        let theme = Theme::plain();
        for style in [
            theme.heading(),
            theme.muted(),
            theme.kind(),
            theme.success(),
            theme.error(),
            theme.selected(),
        ] {
            assert_eq!(style.fg, None);
        }
        assert!(theme.heading().add_modifier.contains(Modifier::BOLD));
        assert!(
            theme
                .selected()
                .add_modifier
                .contains(Modifier::BOLD | Modifier::REVERSED)
        );
    }
}
