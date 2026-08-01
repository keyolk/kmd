//! Install and remove kmd-owned Claude Code hooks without disturbing other settings.

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value, json};
use std::fs;
use std::path::{Path, PathBuf};

const HOOKS: &[(&str, &str, &str)] = &[
    ("SessionStart", "", "page --hook"),
    ("UserPromptSubmit", "", "rag --hook"),
    ("Stop", "", "hook stop"),
    ("SessionEnd", "", "hook session-end"),
    ("PostToolUse", "Write|Edit|MultiEdit", "hook mark-dirty"),
];

fn settings_path() -> Result<PathBuf> {
    let home = std::env::var("HOME").context("HOME is not set")?;
    Ok(PathBuf::from(home).join(".claude/settings.json"))
}

fn shell_quote(path: &Path) -> String {
    let value = path.to_string_lossy();
    if value
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || "/._-~".contains(ch))
    {
        value.into_owned()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

fn default_binary() -> Result<PathBuf> {
    std::env::current_exe().context("failed to resolve the current kmd executable")
}

fn load(path: &Path) -> Result<Value> {
    if !path.exists() {
        return Ok(json!({}));
    }
    let raw =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let value: Value = serde_json::from_str(&raw)
        .with_context(|| format!("{} contains invalid JSON", path.display()))?;
    if !value.is_object() {
        bail!("{} must contain a JSON object", path.display());
    }
    Ok(value)
}

fn hooks_object(settings: &mut Value) -> Result<&mut Map<String, Value>> {
    let root = settings
        .as_object_mut()
        .context("Claude settings must be a JSON object")?;
    let hooks = root.entry("hooks").or_insert_with(|| json!({}));
    hooks
        .as_object_mut()
        .context("Claude settings 'hooks' must be a JSON object")
}

fn event_groups<'a>(hooks: &'a mut Map<String, Value>, event: &str) -> Result<&'a mut Vec<Value>> {
    let groups = hooks.entry(event).or_insert_with(|| json!([]));
    groups
        .as_array_mut()
        .with_context(|| format!("Claude settings hooks.{} must be an array", event))
}

fn action_of(command: &str) -> Option<&str> {
    for (_, _, action) in HOOKS {
        let Some(executable) = command.strip_suffix(action) else {
            continue;
        };
        let executable = executable.trim_end().trim_matches(['\'', '"']);
        if Path::new(executable)
            .file_name()
            .and_then(|name| name.to_str())
            == Some("kmd")
        {
            return Some(*action);
        }
    }
    None
}

fn is_managed_hook(hook: &Value) -> bool {
    hook.get("type").and_then(Value::as_str) == Some("command")
        && hook
            .get("command")
            .and_then(Value::as_str)
            .and_then(action_of)
            .is_some()
}

fn remove_managed(settings: &mut Value) -> Result<usize> {
    let hooks = hooks_object(settings)?;
    let mut removed = 0;
    for value in hooks.values_mut() {
        let Some(groups) = value.as_array_mut() else {
            continue;
        };
        groups.retain_mut(|group| {
            let Some(object) = group.as_object_mut() else {
                return true;
            };
            let Some(items) = object.get_mut("hooks").and_then(Value::as_array_mut) else {
                return true;
            };
            let before = items.len();
            items.retain(|hook| !is_managed_hook(hook));
            let removed_here = before - items.len();
            removed += removed_here;
            // Remove only groups that became empty because kmd entries were removed.
            !(removed_here > 0 && items.is_empty())
        });
    }
    Ok(removed)
}

fn add_hook(settings: &mut Value, event: &str, matcher: &str, command: String) -> Result<()> {
    let groups = event_groups(hooks_object(settings)?, event)?;
    if let Some(group) = groups.iter_mut().find(|group| {
        group.get("matcher").and_then(Value::as_str).unwrap_or("") == matcher
            && group.get("hooks").is_some_and(Value::is_array)
    }) {
        group
            .get_mut("hooks")
            .and_then(Value::as_array_mut)
            .expect("checked above")
            .push(json!({"type": "command", "command": command}));
    } else {
        groups.push(json!({
            "matcher": matcher,
            "hooks": [{"type": "command", "command": command}]
        }));
    }
    Ok(())
}

fn install_into(settings: &mut Value, binary: &Path) -> Result<usize> {
    remove_managed(settings)?;
    let executable = shell_quote(binary);
    for (event, matcher, action) in HOOKS {
        add_hook(
            settings,
            event,
            matcher,
            format!("{} {}", executable, action),
        )?;
    }
    Ok(HOOKS.len())
}

fn installed(settings: &Value) -> Vec<(&'static str, &'static str, &'static str, String)> {
    let mut found = Vec::new();
    let Some(hooks) = settings.get("hooks").and_then(Value::as_object) else {
        return found;
    };
    for (event, matcher, action) in HOOKS {
        let Some(groups) = hooks.get(*event).and_then(Value::as_array) else {
            continue;
        };
        for group in groups {
            if group.get("matcher").and_then(Value::as_str).unwrap_or("") != *matcher {
                continue;
            }
            let Some(items) = group.get("hooks").and_then(Value::as_array) else {
                continue;
            };
            if let Some(command) = items.iter().find_map(|hook| {
                let command = hook.get("command").and_then(Value::as_str)?;
                (action_of(command) == Some(*action)).then(|| command.to_string())
            }) {
                found.push((*event, *matcher, *action, command));
                break;
            }
        }
    }
    found
}

fn write_atomic(path: &Path, settings: &Value) -> Result<()> {
    let parent = path
        .parent()
        .with_context(|| format!("{} has no parent directory", path.display()))?;
    fs::create_dir_all(parent).with_context(|| format!("failed to create {}", parent.display()))?;
    let temp = parent.join(format!(
        ".{}.kmd-{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("settings"),
        std::process::id()
    ));
    let body = serde_json::to_string_pretty(settings)? + "\n";
    fs::write(&temp, body).with_context(|| format!("failed to write {}", temp.display()))?;
    if let Ok(metadata) = fs::metadata(path) {
        fs::set_permissions(&temp, metadata.permissions())?;
    }
    fs::rename(&temp, path).with_context(|| format!("failed to replace {}", path.display()))?;
    Ok(())
}

pub fn install(binary: Option<&Path>) -> Result<()> {
    let path = settings_path()?;
    let binary = binary
        .map(Path::to_path_buf)
        .map(Ok)
        .unwrap_or_else(default_binary)?;
    if !binary.is_file() {
        bail!("kmd binary does not exist: {}", binary.display());
    }
    let mut settings = load(&path)?;
    let count = install_into(&mut settings, &binary)?;
    write_atomic(&path, &settings)?;
    println!(
        "installed {} Claude Code hooks in {}",
        count,
        path.display()
    );
    println!("binary: {}", binary.display());
    Ok(())
}

pub fn status() -> Result<()> {
    let path = settings_path()?;
    let settings = load(&path)?;
    let found = installed(&settings);
    println!(
        "kmd Claude Code hooks: {}/{} installed",
        found.len(),
        HOOKS.len()
    );
    println!("settings: {}", path.display());
    for (event, matcher, _, command) in &found {
        let suffix = if matcher.is_empty() {
            String::new()
        } else {
            format!(" [{}]", matcher)
        };
        println!("  ok  {}{} -> {}", event, suffix, command);
    }
    for (event, matcher, action) in HOOKS {
        if !found
            .iter()
            .any(|(found_event, found_matcher, found_action, _)| {
                found_event == event && found_matcher == matcher && found_action == action
            })
        {
            let suffix = if matcher.is_empty() {
                String::new()
            } else {
                format!(" [{}]", matcher)
            };
            println!("  missing  {}{} -> kmd {}", event, suffix, action);
        }
    }
    if found.len() != HOOKS.len() {
        bail!("kmd Claude Code hooks are not fully installed");
    }
    Ok(())
}

pub fn uninstall() -> Result<()> {
    let path = settings_path()?;
    if !path.exists() {
        println!("no Claude Code settings found at {}", path.display());
        return Ok(());
    }
    let mut settings = load(&path)?;
    let removed = remove_managed(&mut settings)?;
    if removed > 0 {
        write_atomic(&path, &settings)?;
    }
    println!(
        "removed {} kmd Claude Code hooks from {}",
        removed,
        path.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn existing_settings() -> Value {
        json!({
            "model": "opus",
            "hooks": {
                "Stop": [{
                    "matcher": "",
                    "hooks": [
                        {"type": "command", "command": "other-stop"},
                        {"type": "command", "command": "~/.local/bin/kmd hook stop"}
                    ]
                }],
                "Notification": [{"matcher": "", "hooks": []}]
            }
        })
    }

    #[test]
    fn install_is_idempotent_and_preserves_other_settings() {
        let mut settings = existing_settings();
        let binary = Path::new("/opt/kmd/bin/kmd");
        install_into(&mut settings, binary).unwrap();
        install_into(&mut settings, binary).unwrap();

        assert_eq!(settings["model"], "opus");
        assert_eq!(installed(&settings).len(), HOOKS.len());
        let serialized = serde_json::to_string(&settings).unwrap();
        assert_eq!(serialized.matches("/opt/kmd/bin/kmd hook stop").count(), 1);
        assert!(serialized.contains("other-stop"));
        assert_eq!(settings["hooks"]["Notification"][0]["hooks"], json!([]));
    }

    #[test]
    fn uninstall_removes_only_managed_commands() {
        let mut settings = existing_settings();
        install_into(&mut settings, Path::new("/opt/kmd/bin/kmd")).unwrap();
        assert_eq!(remove_managed(&mut settings).unwrap(), HOOKS.len());
        assert!(installed(&settings).is_empty());
        let serialized = serde_json::to_string(&settings).unwrap();
        assert!(serialized.contains("other-stop"));
        assert!(serialized.contains("Notification"));
    }

    #[test]
    fn quotes_binary_paths_with_spaces() {
        assert_eq!(
            shell_quote(Path::new("/Applications/Kmd Tools/kmd")),
            "'/Applications/Kmd Tools/kmd'"
        );
    }
}
