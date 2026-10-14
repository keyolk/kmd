//! Uniqueness, reachability, overrides and footer fitting for the key table.

use super::*;

fn press(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, KeyModifiers::NONE)
}

#[test]
fn default_keys_are_unique_within_every_context_stack() {
    assert_eq!(Keymap::defaults().conflicts(), Vec::<String>::new());
}

/// Every binding listed in help must actually fire from some state: no
/// key may be shadowed by an earlier binding in a stack that holds its
/// context, and every context must appear in some stack.
#[test]
fn every_action_in_help_is_reachable() {
    let keymap = Keymap::defaults();
    for binding in keymap.bindings() {
        let stack = STACKS
            .iter()
            .find(|stack| stack.contains(&binding.ctx))
            .unwrap_or_else(|| panic!("{} is in no context stack", binding.name));
        for key in &binding.keys {
            let modifiers = if key.ctrl {
                KeyModifiers::CONTROL
            } else {
                KeyModifiers::NONE
            };
            let event = KeyEvent::new(key.code, modifiers);
            assert_eq!(
                keymap.lookup(stack, &event),
                Some(binding.action),
                "{} on {} does not reach {}",
                key.label(),
                binding.ctx.title(),
                binding.name
            );
        }
    }
    let listed: usize = keymap
        .help_sections()
        .iter()
        .map(|(_, rows)| rows.len())
        .sum();
    assert_eq!(listed, keymap.bindings().len(), "help omits a binding");
}

#[test]
fn action_names_are_unique() {
    let mut names: Vec<_> = DEFAULTS.iter().map(|default| default.name).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), DEFAULTS.len());
}

#[test]
fn shifted_letters_match_with_or_without_the_shift_flag() {
    let keymap = Keymap::defaults();
    let stack = STACKS[2];
    let plain = press(KeyCode::Char('G'));
    let shifted = KeyEvent::new(KeyCode::Char('G'), KeyModifiers::SHIFT);
    assert_eq!(keymap.lookup(stack, &plain), Some(Action::Bottom));
    assert_eq!(keymap.lookup(stack, &shifted), Some(Action::Bottom));
}

#[test]
fn alt_chords_never_match_a_plain_binding() {
    let keymap = Keymap::defaults();
    let alt_k = KeyEvent::new(KeyCode::Char('k'), KeyModifiers::ALT);
    assert_eq!(keymap.lookup(STACKS[2], &alt_k), None);
}

#[test]
fn overrides_rebind_by_action_name() {
    let keymap = Keymap::defaults()
        .with_overrides("sim.mode: M\nview.next: [tab, \"]\"]\n")
        .unwrap();
    let sim = STACKS[4];
    assert_eq!(
        keymap.lookup(sim, &press(KeyCode::Char('M'))),
        Some(Action::SimMode)
    );
    assert_eq!(keymap.lookup(sim, &press(KeyCode::Char('m'))), None);
    assert_eq!(
        keymap.lookup(sim, &press(KeyCode::Char(']'))),
        Some(Action::ViewNext)
    );
    assert_eq!(keymap.key_label(Action::SimMode), "M");
}

#[test]
fn conflicting_or_invalid_overrides_are_rejected() {
    let conflict = Keymap::defaults()
        .with_overrides("sim.mode: j\n")
        .unwrap_err();
    assert!(conflict.contains("sim.mode"), "{conflict}");
    assert!(
        Keymap::defaults()
            .with_overrides("no.such: x\n")
            .unwrap_err()
            .contains("unknown action")
    );
    assert!(
        Keymap::defaults()
            .with_overrides("quit: ctrl+q\n")
            .unwrap_err()
            .contains("Ctrl")
    );
    assert!(
        Keymap::defaults()
            .with_overrides("force_quit: Q\n")
            .is_err()
    );
    assert!(Keymap::defaults().with_overrides("quit: []\n").is_err());
}

#[test]
fn a_bad_keys_file_falls_back_to_defaults_with_a_reason() {
    let dir = std::env::temp_dir().join(format!("kmd-keys-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("keys.yaml");
    std::fs::write(&path, "quit: j\nscroll.down: j\n").unwrap();
    let (keymap, warning) = Keymap::load(&path);
    assert!(warning.unwrap().contains("using default keys"));
    assert_eq!(keymap.key_label(Action::Quit), "q");

    let (_, warning) = Keymap::load(&dir.join("missing.yaml"));
    assert!(warning.is_none());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn narrow_footers_drop_low_priority_hints_but_keep_quit_and_help() {
    let keymap = Keymap::defaults();
    let hints = keymap.hints(&[Ctx::Global, Ctx::Sessions, Ctx::Scroll]);
    let wide = fit_hints(&hints, 200);
    assert!(wide.contains("g/G: top/bottom"));
    assert!(wide.contains("j/k: session"));
    let narrow = fit_hints(&hints, 30);
    assert!(narrow.contains("q: quit"));
    assert!(narrow.contains("?: keys"));
    assert!(!narrow.contains("top/bottom"));
}

#[test]
fn typing_hints_do_not_offer_keys_that_would_be_typed() {
    let hints = Keymap::defaults().hints(&[Ctx::Typing]);
    let text = fit_hints(&hints, 200);
    assert!(!text.contains('?'), "{text}");
    assert!(text.contains("Esc: command"));
}
