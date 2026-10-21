//! The dashboard's one key table.
//!
//! Dispatch, the footer hints, the per-view "Action" guide line and the `?`
//! overlay are all derived from `DEFAULTS`. Before this, the footer, the help
//! rows and the match arms were three hand-written copies that drifted: the
//! typing-mode footer offered `?: keys` while `?` was typed into the query.
//!
//! Each binding belongs to a context. The dashboard's state picks a stack of
//! active contexts (`Keymap::lookup`); a key may appear at most once across
//! any stack, which `conflicts()` checks and a test enforces. Users rebind by
//! action name in `~/.config/kmd/keys.yaml`:
//!
//! ```yaml
//! sim.mode: M
//! view.next: [tab, "]"]
//! ```

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use std::collections::HashMap;
use std::path::Path;

/// Ctrl/Alt/Super chords belong to the terminal and the multiplexer prefix
/// (PR #7 dropped every such binding). The one exception is `ctrl+c`, which
/// `ForceQuit` claims because raw mode leaves no SIGINT handler to do it.
pub fn is_reserved_chord(key: &KeyEvent) -> bool {
    key.modifiers
        .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT | KeyModifiers::SUPER)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ctx {
    /// Active in every state, including text input and the help overlay.
    Always,
    /// The `?` overlay. Keys it does not bind close it.
    Help,
    /// Simulator query input. Keys it does not bind are text.
    Typing,
    /// Every view outside text input and help.
    Global,
    Sessions,
    Operations,
    /// Detail scrolling shared by Sessions and Operations.
    Scroll,
    /// Simulator outside text input.
    Simulator,
    /// Views with a resizable split: Sessions and the Simulator.
    Split,
}

impl Ctx {
    fn title(self) -> &'static str {
        match self {
            Self::Always => "Anywhere",
            Self::Help => "This overlay",
            Self::Typing => "Simulator · typing a query",
            Self::Global => "All views",
            Self::Sessions => "Sessions",
            Self::Operations => "Operations",
            Self::Scroll => "Sessions & Operations",
            Self::Simulator => "Simulator",
            Self::Split => "Sessions & Simulator · split",
        }
    }

    /// Order of sections in the help overlay.
    const HELP_ORDER: [Ctx; 9] = [
        Ctx::Global,
        Ctx::Always,
        Ctx::Sessions,
        Ctx::Operations,
        Ctx::Scroll,
        Ctx::Simulator,
        Ctx::Split,
        Ctx::Typing,
        Ctx::Help,
    ];
}

/// Every context stack the dashboard can be in. `conflicts()` checks each.
pub const STACKS: &[&[Ctx]] = &[
    &[Ctx::Always, Ctx::Help],
    &[Ctx::Always, Ctx::Typing],
    &[
        Ctx::Always,
        Ctx::Global,
        Ctx::Sessions,
        Ctx::Scroll,
        Ctx::Split,
    ],
    &[Ctx::Always, Ctx::Global, Ctx::Operations, Ctx::Scroll],
    &[Ctx::Always, Ctx::Global, Ctx::Simulator, Ctx::Split],
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    ForceQuit,
    Quit,
    Help,
    HelpClose,
    HelpDown,
    HelpUp,
    ViewNext,
    ViewPrev,
    ViewSessions,
    ViewOperations,
    ViewSimulator,
    Refresh,
    Checks,
    SessionNext,
    SessionPrev,
    SessionOpenLink,
    ScrollDown,
    ScrollUp,
    PageDown,
    PageUp,
    Top,
    Bottom,
    SimType,
    SimRun,
    SimMode,
    SimClear,
    SimHistoryPrev,
    SimHistoryNext,
    SimHitNext,
    SimHitPrev,
    SimDetailDown,
    SimDetailUp,
    SimDetailTop,
    SimOpenLink,
    TypingStop,
    TypingRun,
    TypingHistoryPrev,
    TypingHistoryNext,
    SplitShrink,
    SplitGrow,
    SplitReset,
}

/// Footer entry: lower priority numbers survive a narrow terminal longer;
/// priority 1 is never dropped. `pair` shows another action's first key next
/// to this one's (`j/k: session`) and hides that action's own entry.
#[derive(Clone, Copy, Debug)]
struct Footer {
    priority: u8,
    label: &'static str,
    pair: Option<Action>,
}

struct DefaultBinding {
    action: Action,
    name: &'static str,
    ctx: Ctx,
    keys: &'static [&'static str],
    help: &'static str,
    footer: Option<Footer>,
}

const fn hint(priority: u8, label: &'static str) -> Option<Footer> {
    Some(Footer {
        priority,
        label,
        pair: None,
    })
}

const fn hint_pair(priority: u8, label: &'static str, pair: Action) -> Option<Footer> {
    Some(Footer {
        priority,
        label,
        pair: Some(pair),
    })
}

macro_rules! bind {
    ($action:ident, $name:literal, $ctx:ident, [$($key:literal),*], $help:literal, $footer:expr) => {
        DefaultBinding {
            action: Action::$action,
            name: $name,
            ctx: Ctx::$ctx,
            keys: &[$($key),*],
            help: $help,
            footer: $footer,
        }
    };
}

// One row per binding, kept on one line each so the table reads as a table.
#[rustfmt::skip]
const DEFAULTS: &[DefaultBinding] = &[
    bind!(ForceQuit, "force_quit", Always, ["ctrl+c"], "quit, even while typing a query", None),
    bind!(Quit, "quit", Global, ["q"], "quit", hint(1, "quit")),
    bind!(Help, "help", Global, ["?"], "show this keymap", hint(1, "keys")),
    bind!(ViewNext, "view.next", Global, ["tab", "l", "right"], "next view", hint_pair(2, "view", Action::ViewPrev)),
    bind!(ViewPrev, "view.prev", Global, ["backtab", "h", "left"], "previous view", None),
    bind!(ViewSessions, "view.sessions", Global, ["1"], "Sessions view", None),
    bind!(ViewOperations, "view.operations", Global, ["2"], "Operations view", None),
    bind!(ViewSimulator, "view.simulator", Global, ["3"], "Simulator view", None),
    bind!(Refresh, "refresh", Global, ["r"], "refresh the snapshot", hint(3, "refresh")),
    bind!(Checks, "checks", Global, ["t"], "run the self-check suite", hint(3, "checks")),
    bind!(SessionNext, "session.next", Sessions, ["j", "down"], "select the next session", hint_pair(2, "session", Action::SessionPrev)),
    bind!(SessionPrev, "session.prev", Sessions, ["k", "up"], "select the previous session", None),
    // `o` is the fleet's open-link key (tui-design E11).
    bind!(SessionOpenLink, "session.open_link", Sessions, ["o"], "open the first link in the session's text", hint(3, "open link")),
    bind!(ScrollDown, "scroll.down", Operations, ["j", "down"], "scroll down a line", hint_pair(2, "scroll", Action::ScrollUp)),
    bind!(ScrollUp, "scroll.up", Operations, ["k", "up"], "scroll up a line", None),
    bind!(PageDown, "scroll.page_down", Scroll, ["pgdn"], "scroll down a page", hint_pair(3, "page", Action::PageUp)),
    bind!(PageUp, "scroll.page_up", Scroll, ["pgup"], "scroll up a page", None),
    bind!(Top, "scroll.top", Scroll, ["g", "home"], "jump to the top", hint_pair(4, "top/bottom", Action::Bottom)),
    bind!(Bottom, "scroll.bottom", Scroll, ["G", "end"], "jump to the bottom", None),
    bind!(SimType, "sim.type", Simulator, ["i", "/"], "start typing a query", hint(2, "type")),
    bind!(SimRun, "sim.run", Simulator, ["enter"], "run the query", hint(2, "run")),
    bind!(SimMode, "sim.mode", Simulator, ["m"], "RAG ⇄ BM25 search ⇄ Global", hint(2, "mode")),
    bind!(SimHitNext, "sim.hit_next", Simulator, ["j"], "select the next hit", hint_pair(3, "hit", Action::SimHitPrev)),
    bind!(SimHitPrev, "sim.hit_prev", Simulator, ["k"], "select the previous hit", None),
    bind!(SimDetailDown, "sim.detail_down", Simulator, ["J", "pgdn"], "scroll result detail down", hint_pair(3, "detail", Action::SimDetailUp)),
    bind!(SimDetailUp, "sim.detail_up", Simulator, ["K", "pgup"], "scroll result detail up", None),
    bind!(SimDetailTop, "sim.detail_top", Simulator, ["home"], "result detail to the top", None),
    bind!(SimOpenLink, "sim.open_link", Simulator, ["o"], "open the first link in the result detail", hint(3, "open link")),
    bind!(SimHistoryPrev, "sim.history_prev", Simulator, ["p", "up"], "previous prompt from history", hint_pair(4, "history", Action::SimHistoryNext)),
    bind!(SimHistoryNext, "sim.history_next", Simulator, ["n", "down"], "next prompt from history", None),
    bind!(SimClear, "sim.clear", Simulator, ["x"], "clear the query", hint(4, "clear")),
    bind!(TypingStop, "typing.stop", Typing, ["esc"], "stop typing (command mode)", hint(1, "command")),
    bind!(TypingRun, "typing.run", Typing, ["enter"], "run the query", hint(1, "run")),
    bind!(TypingHistoryPrev, "typing.history_prev", Typing, ["up"], "previous prompt from history", hint_pair(2, "history", Action::TypingHistoryNext)),
    bind!(TypingHistoryNext, "typing.history_next", Typing, ["down"], "next prompt from history", None),
    // Last in the footer: resizing is rare next to the view's own keys.
    // Shrink first, so the hint reads `</>` in keyboard order.
    bind!(SplitShrink, "split.shrink", Split, ["<"], "narrow the list (first pane)", hint_pair(5, "split", Action::SplitGrow)),
    bind!(SplitGrow, "split.grow", Split, [">"], "widen the list (first pane)", None),
    bind!(SplitReset, "split.reset", Split, ["="], "reset the split (so does a double click on it)", hint(5, "reset split")),
    bind!(HelpClose, "help.close", Help, ["?", "esc", "q"], "close this overlay (so does any unbound key)", hint(1, "close")),
    bind!(HelpDown, "help.down", Help, ["j", "down"], "scroll this overlay down", hint_pair(2, "scroll", Action::HelpUp)),
    bind!(HelpUp, "help.up", Help, ["k", "up"], "scroll this overlay up", None),
];

/// One key as the table stores it. Shift is not part of it: `G` arrives as
/// `Char('G')` with or without the SHIFT flag depending on the terminal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Key {
    code: KeyCode,
    ctrl: bool,
}

impl Key {
    /// The table key for an event, or None for Alt/Super chords, which never
    /// match a binding and pass through to the terminal.
    fn from_event(event: &KeyEvent) -> Option<Self> {
        if event
            .modifiers
            .intersects(KeyModifiers::ALT | KeyModifiers::SUPER)
        {
            return None;
        }
        let ctrl = event.modifiers.contains(KeyModifiers::CONTROL);
        let code = match event.code {
            KeyCode::Char(ch) if ctrl => KeyCode::Char(ch.to_ascii_lowercase()),
            code => code,
        };
        Some(Self { code, ctrl })
    }

    pub fn parse(text: &str) -> Result<Self, String> {
        let (ctrl, rest) = match text.strip_prefix("ctrl+") {
            Some(rest) => (true, rest),
            None => (false, text),
        };
        let code = match rest.to_ascii_lowercase().as_str() {
            "enter" => KeyCode::Enter,
            "esc" => KeyCode::Esc,
            "tab" => KeyCode::Tab,
            "backtab" | "shift+tab" => KeyCode::BackTab,
            "up" => KeyCode::Up,
            "down" => KeyCode::Down,
            "left" => KeyCode::Left,
            "right" => KeyCode::Right,
            "pgup" => KeyCode::PageUp,
            "pgdn" => KeyCode::PageDown,
            "home" => KeyCode::Home,
            "end" => KeyCode::End,
            "space" => KeyCode::Char(' '),
            "backspace" => KeyCode::Backspace,
            _ => {
                let mut chars = rest.chars();
                match (chars.next(), chars.next()) {
                    (Some(ch), None) => {
                        KeyCode::Char(if ctrl { ch.to_ascii_lowercase() } else { ch })
                    }
                    _ => return Err(format!("unknown key {text:?}")),
                }
            }
        };
        Ok(Self { code, ctrl })
    }

    pub fn label(self) -> String {
        let name = match self.code {
            KeyCode::Enter => "Enter".to_string(),
            KeyCode::Esc => "Esc".to_string(),
            KeyCode::Tab => "Tab".to_string(),
            KeyCode::BackTab => "S-Tab".to_string(),
            KeyCode::Up => "↑".to_string(),
            KeyCode::Down => "↓".to_string(),
            KeyCode::Left => "←".to_string(),
            KeyCode::Right => "→".to_string(),
            KeyCode::PageUp => "PgUp".to_string(),
            KeyCode::PageDown => "PgDn".to_string(),
            KeyCode::Home => "Home".to_string(),
            KeyCode::End => "End".to_string(),
            KeyCode::Backspace => "Backspace".to_string(),
            KeyCode::Char(' ') => "Space".to_string(),
            KeyCode::Char(ch) => ch.to_string(),
            other => format!("{other:?}"),
        };
        if self.ctrl {
            format!("Ctrl-{name}")
        } else {
            name
        }
    }
}

#[derive(Clone, Debug)]
pub struct Binding {
    pub action: Action,
    pub name: &'static str,
    pub ctx: Ctx,
    pub keys: Vec<Key>,
    pub help: &'static str,
    footer: Option<Footer>,
}

#[derive(Clone, Debug)]
pub struct Keymap {
    bindings: Vec<Binding>,
}

/// One help overlay section: its title and (keys, description) rows.
pub type HelpSection = (&'static str, Vec<(String, &'static str)>);

impl Keymap {
    pub fn defaults() -> Self {
        let bindings = DEFAULTS
            .iter()
            .map(|default| Binding {
                action: default.action,
                name: default.name,
                ctx: default.ctx,
                keys: default
                    .keys
                    .iter()
                    .map(|key| Key::parse(key).expect("default key parses"))
                    .collect(),
                help: default.help,
                footer: default.footer,
            })
            .collect();
        Self { bindings }
    }

    /// Defaults with the user's `keys.yaml` applied. A missing file is not an
    /// error. An unreadable or invalid file, or one whose keys would collide,
    /// is ignored as a whole and the reason returned for the status line: a
    /// half-applied keymap is harder to reason about than the defaults.
    pub fn load(path: &Path) -> (Self, Option<String>) {
        let defaults = Self::defaults();
        let raw = match std::fs::read_to_string(path) {
            Ok(raw) => raw,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return (defaults, None),
            Err(error) => {
                return (
                    defaults,
                    Some(format!("{}: {error} · using default keys", path.display())),
                );
            }
        };
        match defaults.clone().with_overrides(&raw) {
            Ok(keymap) => (keymap, None),
            Err(error) => (
                defaults,
                Some(format!("{}: {error} · using default keys", path.display())),
            ),
        }
    }

    fn with_overrides(mut self, raw: &str) -> Result<Self, String> {
        let overrides: HashMap<String, serde_yaml::Value> = if raw.trim().is_empty() {
            HashMap::new()
        } else {
            serde_yaml::from_str(raw).map_err(|error| error.to_string())?
        };
        for (name, value) in overrides {
            let binding = self
                .bindings
                .iter_mut()
                .find(|binding| binding.name == name)
                .ok_or_else(|| format!("unknown action {name:?}"))?;
            if binding.action == Action::ForceQuit {
                return Err("force_quit stays on ctrl+c".to_string());
            }
            let texts: Vec<String> = match value {
                serde_yaml::Value::String(text) => vec![text],
                serde_yaml::Value::Sequence(items) => items
                    .into_iter()
                    .map(|item| match item {
                        serde_yaml::Value::String(text) => Ok(text),
                        other => Err(format!("{name}: key must be a string, got {other:?}")),
                    })
                    .collect::<Result<_, _>>()?,
                other => return Err(format!("{name}: expected a key or a list, got {other:?}")),
            };
            if texts.is_empty() {
                return Err(format!("{name}: needs at least one key"));
            }
            let keys = texts
                .iter()
                .map(|text| Key::parse(text))
                .collect::<Result<Vec<_>, _>>()?;
            if keys.iter().any(|key| key.ctrl) {
                return Err(format!(
                    "{name}: Ctrl chords stay with the terminal and tmux"
                ));
            }
            binding.keys = keys;
        }
        let conflicts = self.conflicts();
        if !conflicts.is_empty() {
            return Err(conflicts.join("; "));
        }
        Ok(self)
    }

    /// Every key bound twice within one context stack, as readable messages.
    pub fn conflicts(&self) -> Vec<String> {
        let mut found = Vec::new();
        for stack in STACKS {
            let mut seen: HashMap<Key, &Binding> = HashMap::new();
            for binding in self.bindings.iter().filter(|b| stack.contains(&b.ctx)) {
                for key in &binding.keys {
                    if let Some(previous) = seen.insert(*key, binding) {
                        let message = format!(
                            "{} is bound to both {} and {}",
                            key.label(),
                            previous.name,
                            binding.name
                        );
                        if !found.contains(&message) {
                            found.push(message);
                        }
                    }
                }
            }
        }
        found
    }

    pub fn lookup(&self, stack: &[Ctx], event: &KeyEvent) -> Option<Action> {
        let key = Key::from_event(event)?;
        self.bindings
            .iter()
            .find(|binding| stack.contains(&binding.ctx) && binding.keys.contains(&key))
            .map(|binding| binding.action)
    }

    fn binding(&self, action: Action) -> Option<&Binding> {
        self.bindings
            .iter()
            .find(|binding| binding.action == action)
    }

    /// The first key bound to `action`, for inline hints such as "i types".
    pub fn key_label(&self, action: Action) -> String {
        self.binding(action)
            .and_then(|binding| binding.keys.first())
            .map(|key| key.label())
            .unwrap_or_default()
    }

    /// Footer hints for `ctxs`, most important first, as ("j/k", "session").
    pub fn hints(&self, ctxs: &[Ctx]) -> Vec<(u8, String, &'static str)> {
        let mut hints: Vec<(u8, String, &'static str)> = self
            .bindings
            .iter()
            .filter(|binding| ctxs.contains(&binding.ctx))
            .filter_map(|binding| {
                let priority = binding.footer?.priority;
                let (keys, label) = self.hint_for(binding.action)?;
                Some((priority, keys, label))
            })
            .collect();
        hints.sort_by_key(|(priority, _, _)| *priority);
        hints
    }

    /// One action's footer hint, e.g. ("r", "refresh"), for the view guide.
    pub fn hint_for(&self, action: Action) -> Option<(String, &'static str)> {
        let footer = self.binding(action)?.footer?;
        let mut keys = self.key_label(action);
        if let Some(pair) = footer.pair {
            keys = format!("{keys}/{}", self.key_label(pair));
        }
        Some((keys, footer.label))
    }

    /// Help overlay rows grouped by context, every binding included.
    pub fn help_sections(&self) -> Vec<HelpSection> {
        Ctx::HELP_ORDER
            .iter()
            .map(|ctx| {
                let rows = self
                    .bindings
                    .iter()
                    .filter(|binding| binding.ctx == *ctx)
                    .map(|binding| {
                        let keys = binding
                            .keys
                            .iter()
                            .map(|key| key.label())
                            .collect::<Vec<_>>()
                            .join(" / ");
                        (keys, binding.help)
                    })
                    .collect();
                (ctx.title(), rows)
            })
            .collect()
    }

    #[cfg(test)]
    pub fn defaults_with_test_overrides(raw: &str) -> Self {
        Self::defaults()
            .with_overrides(raw)
            .expect("valid test overrides")
    }

    #[cfg(test)]
    pub fn bindings(&self) -> &[Binding] {
        &self.bindings
    }
}

/// Fits footer hints into `width` cells: drops the least important hints
/// (highest priority number) first and never drops priority 1.
pub fn fit_hints(hints: &[(u8, String, &'static str)], width: usize) -> String {
    let render = |kept: &[&(u8, String, &'static str)]| {
        kept.iter()
            .map(|(_, keys, label)| format!("{keys}: {label}"))
            .collect::<Vec<_>>()
            .join("  ")
    };
    let mut kept: Vec<&(u8, String, &'static str)> = hints.iter().collect();
    loop {
        let text = render(&kept);
        if unicode_width::UnicodeWidthStr::width(text.as_str()) <= width {
            return text;
        }
        let Some(drop) = kept
            .iter()
            .enumerate()
            .filter(|(_, (priority, _, _))| *priority > 1)
            .max_by_key(|(index, (priority, _, _))| (*priority, *index))
            .map(|(index, _)| index)
        else {
            return text;
        };
        kept.remove(drop);
    }
}

#[cfg(test)]
#[path = "dashboard_keys_tests.rs"]
mod tests;
