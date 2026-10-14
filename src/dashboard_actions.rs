//! What the dashboard does for each keymap action, the derived `?` overlay,
//! and saving/restoring UI state. Split from dashboard.rs, which already holds
//! the snapshot loaders and renderers; a child module so it can reach `App`'s
//! private fields.

use super::*;

impl App {
    /// Active key contexts, innermost first. Must be one of
    /// `dashboard_keys::STACKS`, which is what the uniqueness test checks.
    pub(super) fn ctx_stack(&self) -> &'static [Ctx] {
        if self.show_help {
            STACKS[0]
        } else if self.tab == SIMULATOR_TAB && self.simulator.typing() {
            STACKS[1]
        } else if self.tab == SESSIONS_TAB {
            STACKS[2]
        } else if self.tab == OPERATIONS_TAB {
            STACKS[3]
        } else {
            STACKS[4]
        }
    }

    /// Runs one action. Returns true when the dashboard should quit.
    pub(super) fn apply(&mut self, action: Action) -> bool {
        match action {
            Action::ForceQuit | Action::Quit => return true,
            Action::Help => {
                self.show_help = true;
                self.help_scroll = 0;
            }
            Action::HelpClose => self.show_help = false,
            Action::HelpDown => {
                self.help_scroll = self.help_scroll.min(self.help_max.get()).saturating_add(1)
            }
            Action::HelpUp => {
                self.help_scroll = self.help_scroll.min(self.help_max.get()).saturating_sub(1)
            }
            Action::ViewNext => self.select_tab(self.tab + 1),
            Action::ViewPrev => self.select_tab((self.tab + TABS.len() - 1) % TABS.len()),
            Action::ViewSessions => self.select_tab(SESSIONS_TAB),
            Action::ViewOperations => self.select_tab(OPERATIONS_TAB),
            Action::ViewSimulator => self.select_tab(SIMULATOR_TAB),
            Action::Refresh => self.request_refresh(true),
            Action::Checks => self.request_checks(),
            Action::SessionNext => self.select_session(self.selected_session.saturating_add(1)),
            Action::SessionPrev => self.select_session(self.selected_session.saturating_sub(1)),
            Action::ScrollDown => self.scroll_down(1),
            Action::ScrollUp => self.scroll_up(1),
            Action::PageDown => self.scroll_down(10),
            Action::PageUp => self.scroll_up(10),
            Action::Top => self.scroll = 0,
            Action::Bottom => self.scroll = u16::MAX,
            Action::SimType => self.simulator.start_typing(),
            Action::SimRun | Action::TypingRun => self.simulator.run(),
            Action::SimMode => self.simulator.toggle_mode(),
            Action::SimClear => self.simulator.clear(),
            Action::SimHistoryPrev | Action::TypingHistoryPrev => self.simulator.previous_history(),
            Action::SimHistoryNext | Action::TypingHistoryNext => self.simulator.next_history(),
            Action::SimHitNext => self.simulator.next_hit(),
            Action::SimHitPrev => self.simulator.previous_hit(),
            Action::SimDetailDown => self.simulator.detail_down(),
            Action::SimDetailUp => self.simulator.detail_up(),
            Action::SimDetailTop => self.simulator.detail_top(),
            Action::TypingStop => self.simulator.stop_typing(),
        }
        false
    }

    pub(super) fn ui_state(&self) -> crate::ui_state::UiState {
        crate::ui_state::UiState {
            view: Some(TABS[self.tab].to_ascii_lowercase()),
            session_id: self.session_ids().get(self.selected_session).cloned(),
            simulator_mode: Some(self.simulator.mode),
            query: String::new(),
        }
        .with_query(&self.simulator.input)
    }

    /// Applies state saved by a previous run. Unknown values are ignored.
    pub(super) fn restore_ui(&mut self, saved: crate::ui_state::UiState) {
        if let Some(tab) = saved
            .view
            .as_deref()
            .and_then(|view| TABS.iter().position(|tab| tab.eq_ignore_ascii_case(view)))
        {
            self.select_tab(tab);
        }
        if let Some(index) = saved.session_id.as_ref().and_then(|id| {
            self.session_ids()
                .iter()
                .position(|candidate| candidate == id)
        }) {
            self.selected_session = index;
        }
        if let Some(mode) = saved.simulator_mode {
            self.simulator.set_mode(mode);
        }
        self.simulator.input = saved.query.clone();
        // What was just restored is what is on disk; do not rewrite it.
        self.ui_saved = self.ui_state();
    }

    /// Hands changed UI state to a worker to write. At most once a second and
    /// never while a write is in flight, so typing a query does not spawn a
    /// writer per key; the exit path catches whatever is left.
    pub(super) fn persist_ui(&mut self) {
        if let Some(result) = self.ui_saver.poll() {
            match result {
                Ok(()) => self.status.resolve("ui-state"),
                Err(error) => self.status.set(
                    "ui-state",
                    Level::Warning,
                    format!("could not save UI state: {error}"),
                ),
            }
        }
        let Some(path) = self.ui_path.clone() else {
            return;
        };
        if self.ui_checked.elapsed() < Duration::from_secs(1) || self.ui_saver.in_flight() {
            return;
        }
        self.ui_checked = Instant::now();
        let current = self.ui_state();
        if current == self.ui_saved {
            return;
        }
        self.ui_saved = current.clone();
        self.ui_saver
            .start(move || crate::ui_state::save(&path, &current));
    }

    /// Final write on exit, after the terminal is restored. Waits briefly for
    /// a write in flight so it cannot land after this one with older state.
    pub(super) fn persist_ui_on_exit(&mut self) -> Result<()> {
        let Some(path) = self.ui_path.clone() else {
            return Ok(());
        };
        if self.ui_saver.in_flight() {
            self.ui_saver.wait(Duration::from_secs(2));
        }
        let current = self.ui_state();
        if current != self.ui_saved {
            crate::ui_state::save(&path, &current)?;
        }
        Ok(())
    }
}

/// The `?` overlay, derived from the keymap so it lists exactly what dispatch
/// does. Scrolls when the terminal is shorter than the keymap.
pub(super) fn draw_help(frame: &mut Frame, area: Rect, app: &App) {
    let theme = app.theme;
    let mut rows = Vec::new();
    for (title, bindings) in app.keys.help_sections() {
        if bindings.is_empty() {
            continue;
        }
        if !rows.is_empty() {
            rows.push(Line::raw(""));
        }
        rows.push(Line::styled(format!(" {title}"), theme.muted()));
        for (keys, description) in bindings {
            rows.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(format!("{keys:<18}"), theme.heading()),
                Span::raw(description.to_string()),
            ]));
        }
    }
    let width = area.width.saturating_sub(4).clamp(20, 76);
    let height = (rows.len() as u16 + 2).min(area.height.saturating_sub(2).max(3));
    let popup = Rect {
        x: area.x + (area.width.saturating_sub(width)) / 2,
        y: area.y + (area.height.saturating_sub(height)) / 2,
        width,
        height,
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme.selected())
        .title_style(theme.heading());
    let paragraph = Paragraph::new(rows);
    let (scroll, max) =
        crate::dashboard_layout::clamp_scroll(&paragraph, block.inner(popup), app.help_scroll);
    app.help_max.set(max);
    let close = app.keys.key_label(Action::HelpClose);
    let title = if max == 0 {
        format!(" keys · {close} closes ")
    } else {
        format!(
            " keys · {close} closes · {}/{} scrolls · {scroll}/{max} ",
            app.keys.key_label(Action::HelpDown),
            app.keys.key_label(Action::HelpUp)
        )
    };
    frame.render_widget(Clear, popup);
    frame.render_widget(
        paragraph.scroll((scroll, 0)).block(block.title(title)),
        popup,
    );
}
