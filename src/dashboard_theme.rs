//! Semantic, terminal-native colors for the operations dashboard.

use ratatui::prelude::*;

pub const BADGE_OK: &str = "● OK";
pub const BADGE_FAIL: &str = "● FAIL";
pub const WARNING_SUFFIX: &str = " !";

pub fn warning_suffix(warn: bool) -> &'static str {
    if warn { WARNING_SUFFIX } else { "" }
}

#[derive(Clone, Copy, Debug)]
pub struct Theme {
    colors: bool,
}

#[derive(Clone, Copy, Debug)]
pub enum BodyKind {
    Sessions,
    Operations,
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
        // NO_COLOR removes hues while preserving non-color emphasis such as bold and reverse.
        if self.colors {
            Style::default().fg(color)
        } else {
            Style::default()
        }
    }

    pub fn accent(self) -> Style {
        self.fg(Color::Cyan)
    }

    pub fn muted(self) -> Style {
        // Gray stays legible on dark themes where DarkGray often disappears.
        self.fg(Color::Gray)
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

    pub fn info(self) -> Style {
        self.fg(Color::LightBlue)
    }

    pub fn heading(self) -> Style {
        self.accent().add_modifier(Modifier::BOLD)
    }

    pub fn selected(self) -> Style {
        self.accent()
            .add_modifier(Modifier::BOLD | Modifier::REVERSED)
    }

    pub fn border(self) -> Style {
        self.muted()
    }
}

pub fn style_body(kind: BodyKind, body: String, theme: Theme) -> Text<'static> {
    Text::from(
        body.lines()
            .map(|line| style_line(kind, line, theme))
            .collect::<Vec<_>>(),
    )
}

fn style_line(kind: BodyKind, line: &str, theme: Theme) -> Line<'static> {
    if is_heading(kind, line) {
        return Line::styled(line.to_string(), theme.heading());
    }

    match kind {
        BodyKind::Sessions => style_session_line(line, theme),
        BodyKind::Operations => style_operations_line(line, theme),
    }
}

fn is_heading(kind: BodyKind, line: &str) -> bool {
    match kind {
        BodyKind::Sessions => {
            line.starts_with("Session ")
                || line.starts_with("── Turn ")
                || matches!(line, "USER" | "CLAUDE")
        }
        BodyKind::Operations => {
            matches!(
                line,
                "Runtime"
                    | "Knowledge"
                    | "Collections"
                    | "Retrieval"
                    | "Last 7 days"
                    | "Recent prompts"
                    | "Evaluations"
                    | "Self-checks"
            ) || line.starts_with("Successful L1 retrieval")
        }
    }
}

fn style_overview_line(line: &str, theme: Theme) -> Line<'static> {
    if line == "Errors" {
        return Line::styled(line.to_string(), theme.error().add_modifier(Modifier::BOLD));
    }
    if line.starts_with("  • ") {
        return Line::styled(line.to_string(), theme.error());
    }
    if let Some(index) = line.find(BADGE_OK) {
        return style_range(line, index, BADGE_OK.len(), theme.success());
    }
    if let Some(index) = line.find(BADGE_FAIL) {
        return style_range(line, index, BADGE_FAIL.len(), theme.error());
    }
    if (line.starts_with("  hooks") || line.starts_with("  documents"))
        && let Some(prefix) = line.strip_suffix(WARNING_SUFFIX)
    {
        return style_range(line, prefix.len(), WARNING_SUFFIX.len(), theme.warning());
    }
    Line::raw(line.to_string())
}

fn style_session_line(line: &str, theme: Theme) -> Line<'static> {
    if line.starts_with("repo:")
        || line.starts_with("cwd:")
        || line.starts_with("updated:")
        || line.starts_with("turns:")
        || line.starts_with("files:")
    {
        return Line::styled(line.to_string(), theme.muted());
    }
    if line.starts_with("(텍스트 응답 없음") {
        return Line::styled(line.to_string(), theme.warning());
    }
    Line::raw(line.to_string())
}

fn style_operations_line(line: &str, theme: Theme) -> Line<'static> {
    let trimmed = line.trim_start();
    if trimmed.starts_with("GATED:") || trimmed.starts_with("INJ:") || trimmed.starts_with("MISS") {
        return style_rag_line(line, theme);
    }
    if line.starts_with("  ✓ ") || line.starts_with("  ✗ ") {
        return style_check_line(line, theme);
    }
    if line.starts_with('●') {
        return style_leading_marker(line, "●", theme.info());
    }
    style_overview_line(line, theme)
}

fn style_rag_line(line: &str, theme: Theme) -> Line<'static> {
    let trimmed = line.trim_start();
    let indent = line.len() - trimmed.len();
    // rag_text emits status rows at two spaces and prompt rows at four spaces.
    // Prompt text may legitimately start with a status word, so keep this distinction strict.
    if indent != 2 {
        return Line::raw(line.to_string());
    }
    let (token_len, style) = if trimmed.starts_with("GATED:") {
        (
            trimmed.find(char::is_whitespace).unwrap_or(trimmed.len()),
            theme.warning(),
        )
    } else if trimmed.starts_with("INJ:") {
        (
            trimmed.find(char::is_whitespace).unwrap_or(trimmed.len()),
            theme.success(),
        )
    } else if trimmed.starts_with("MISS") {
        ("MISS".len(), theme.error())
    } else {
        return Line::raw(line.to_string());
    };
    style_range(line, indent, token_len, style)
}

fn style_check_line(line: &str, theme: Theme) -> Line<'static> {
    if !line.starts_with(' ')
        && let Some(index) = line.find(BADGE_OK)
    {
        return style_range(line, index, BADGE_OK.len(), theme.success());
    }
    if !line.starts_with(' ')
        && let Some(index) = line.find(BADGE_FAIL)
    {
        return style_range(line, index, BADGE_FAIL.len(), theme.error());
    }
    if line.starts_with("  ✓ ") {
        return style_range(line, 2, '✓'.len_utf8(), theme.success());
    }
    if line.starts_with("  ✗ ") {
        return style_range(line, 2, '✗'.len_utf8(), theme.error());
    }
    Line::raw(line.to_string())
}

fn style_leading_marker(line: &str, marker: &str, style: Style) -> Line<'static> {
    style_range(line, 0, marker.len(), style)
}

fn style_range(line: &str, start: usize, len: usize, style: Style) -> Line<'static> {
    let Some(end) = start.checked_add(len) else {
        return Line::raw(line.to_string());
    };
    if end > line.len() || !line.is_char_boundary(start) || !line.is_char_boundary(end) {
        return Line::raw(line.to_string());
    }
    Line::from(vec![
        Span::raw(line[..start].to_string()),
        Span::styled(line[start..end].to_string(), style),
        Span::raw(line[end..].to_string()),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colors_semantic_status_tokens() {
        let theme = Theme::colored();
        let ok = style_line(BodyKind::Operations, "  daemon       ● OK", theme);
        let warning = style_line(
            BodyKind::Operations,
            "  hooks        4/5 installed !",
            theme,
        );
        let gated = style_line(BodyKind::Operations, "  GATED:short      0ms", theme);
        let failed = style_line(BodyKind::Operations, "  ✗ daemon unreachable", theme);

        assert_eq!(ok.spans[1].style.fg, Some(Color::Green));
        assert_eq!(warning.spans[1].style.fg, Some(Color::Yellow));
        assert_eq!(gated.spans[1].style.fg, Some(Color::Yellow));
        assert_eq!(failed.spans[1].style.fg, Some(Color::Red));
    }

    #[test]
    fn does_not_color_prompt_or_check_detail_status_words() {
        let prompt = style_line(
            BodyKind::Operations,
            "    MISS should remain ordinary prompt text",
            Theme::colored(),
        );
        let check = style_line(
            BodyKind::Operations,
            "  ✓ store            detail contains ✗ and ● FAIL",
            Theme::colored(),
        );

        assert_eq!(prompt.spans[0].style.fg, None);
        assert_eq!(check.spans[1].content, "✓");
        assert_eq!(check.spans[1].style.fg, Some(Color::Green));
        assert_eq!(check.spans[2].style.fg, None);
    }

    #[test]
    fn invalid_style_range_falls_back_to_plain_text() {
        let line = style_range("● OK", 1, 1, Theme::colored().success());

        assert_eq!(line.spans.len(), 1);
        assert_eq!(line.spans[0].content, "● OK");
        assert_eq!(line.spans[0].style.fg, None);
    }

    #[test]
    fn plain_theme_keeps_emphasis_without_color() {
        let theme = Theme::plain();
        let heading = style_line(BodyKind::Operations, "Runtime", theme);
        let ok = style_line(BodyKind::Operations, "  daemon       ● OK", theme);

        assert_eq!(heading.style.fg, None);
        assert!(heading.style.add_modifier.contains(Modifier::BOLD));
        assert_eq!(ok.spans[1].style.fg, None);
    }
}
