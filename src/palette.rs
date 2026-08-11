//! Semantic color tokens for the dashboard TUI.
//!
//! Render paths reference these tokens instead of raw colors so theming and
//! `NO_COLOR` are handled in one place. Every colored state is also carried by a
//! word or symbol, so the dashboard stays readable in monochrome.

use ratatui::style::{Color, Modifier, Style};
use std::sync::OnceLock;

const ACCENT: Color = Color::Magenta;
const INFO: Color = Color::Cyan;
const SUCCESS: Color = Color::LightGreen;
const WARN: Color = Color::Yellow;
const FAILURE: Color = Color::LightRed;
const MUTED: Color = Color::DarkGray;

fn colors_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NO_COLOR").is_none())
}

fn fg(color: Color) -> Style {
    if colors_enabled() {
        Style::default().fg(color)
    } else {
        Style::default()
    }
}

/// Section titles and the active tab.
pub fn heading() -> Style {
    fg(INFO).add_modifier(Modifier::BOLD)
}

/// Field labels inside a section.
pub fn label() -> Style {
    fg(INFO)
}

/// Primary values — left unstyled so terminal defaults stay readable.
pub fn value() -> Style {
    Style::default()
}

/// Secondary metadata: paths, timestamps, hints.
pub fn muted() -> Style {
    fg(MUTED)
}

pub fn success() -> Style {
    fg(SUCCESS)
}

pub fn warn() -> Style {
    fg(WARN)
}

pub fn failure() -> Style {
    fg(FAILURE)
}

/// Categories and kinds (evaluation kind, retrieval mode).
pub fn accent() -> Style {
    fg(ACCENT)
}

/// Unfocused panel borders.
pub fn border() -> Style {
    fg(MUTED)
}

/// Focused panel borders — focus changes border color, never border presence.
pub fn border_focus() -> Style {
    fg(INFO)
}

/// Selected row in a list.
pub fn selection() -> Style {
    Style::default().add_modifier(Modifier::REVERSED)
}

/// ok/fail styling paired with a caller-supplied word or symbol.
pub fn state(ok: bool) -> Style {
    if ok { success() } else { failure() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_maps_to_success_and_failure() {
        assert_eq!(state(true), success());
        assert_eq!(state(false), failure());
    }

    #[test]
    fn value_style_keeps_terminal_default() {
        assert_eq!(value(), Style::default());
    }
}
