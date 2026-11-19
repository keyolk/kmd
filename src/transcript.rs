//! Shared rules for reading Claude Code transcripts as a conversation.
//!
//! A transcript line is not always a turn of the conversation. Claude Code also writes
//! bookkeeping records next to the turns, and some of them carry a full `message`:
//! `api-request-blob` (2.1.3xx) stores the exact request body, so it repeats the human
//! prompt *and* every injected block (`<system-reminder>` with CLAUDE.md, hook context,
//! plugin notes). Readers that took `entry.message` from any line exported those blocks
//! as if the user had typed them — on 2026-11-20, 781 of 2,730 learnings files carried
//! CLAUDE.md as a `**User**` turn. Every transcript reader goes through this module so
//! the activity cards, the learnings export and the eval readers agree on what a turn is.

use serde_json::Value;

/// Whether a transcript line is a conversation turn. Lines without a `type` come from
/// old snapshots and the tests, and are taken as turns.
pub fn is_turn(entry: &Value) -> bool {
    if entry.get("isMeta").and_then(Value::as_bool) == Some(true) {
        return false;
    }
    match entry.get("type").and_then(Value::as_str) {
        None => true,
        Some(kind) => kind == "user" || kind == "assistant",
    }
}

/// Who a piece of `user`-role text came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UserText {
    /// Typed by the person.
    Human(String),
    /// Relayed from another session or a plugin (a peer message, a cc-mods inbox
    /// prompt). The text is the payload without Claude Code's English lead-in.
    Relay(String),
    /// Injected by the harness (reminders, hook output, notifications, skill bodies).
    Injected,
}

const INJECTED_PREFIXES: &[&str] = &[
    "<system-reminder",
    "<local-command",
    "<command-",
    "<task-notification",
    "<tool-use",
    "<bash-",
    "<teammate-",
    "<kmd-context",
    "<daily-activity",
    "<session-index",
    "caveat:",
    "tool loaded.",
    "skill /",
    "base directory for this skill:",
    "stop hook feedback:",
    "a session-scoped stop hook is now active",
    "this session is being continued",
    "the user sent a new message while you were working",
    "[request interrupted",
    "[your previous response had no visible output",
    "[image",
];

/// Text the cc-mods plugin submits as a prompt; it marks its own prompts with these heads.
const PLUGIN_PROMPT_PREFIXES: &[&str] = &["[cc-mods]", "[coord]", "[office→coord]", "[cc-mods context:"];

/// Claude Code relays plugin and peer messages behind a one-line English lead-in and
/// follows plugin payloads with an explanation paragraph. Peel both off.
fn unwrap_relay(text: &str) -> Option<&str> {
    const MARK: &str = " sent a message:";
    let head_end = text.char_indices().nth(80).map(|(i, _)| i).unwrap_or(text.len());
    let i = text[..head_end].find(MARK)?;
    let lead = &text[..i];
    if !((lead.starts_with("The ") && lead.ends_with(" plugin")) || lead == "Another Claude session") {
        return None;
    }
    let body = text[i + MARK.len()..].trim_start();
    let body = body
        .find("\n\nThis is how Claude Code surfaces")
        .map(|cut| &body[..cut])
        .unwrap_or(body);
    Some(body.trim_end())
}

/// Strip a `<cross-session-message …>` wrapper down to the message it carries.
fn unwrap_peer(text: &str) -> &str {
    let Some(rest) = text.strip_prefix("<cross-session-message") else {
        return text;
    };
    let Some(open_end) = rest.find('>') else {
        return text;
    };
    let body = &rest[open_end + 1..];
    body.rfind("</cross-session-message>")
        .map(|close| &body[..close])
        .unwrap_or(body)
        .trim()
}

fn injected(lower: &str) -> bool {
    INJECTED_PREFIXES.iter().any(|prefix| lower.starts_with(prefix))
}

/// Classify one `user`-role text block.
pub fn classify_user_text(text: &str) -> UserText {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return UserText::Injected;
    }
    if let Some(payload) = unwrap_relay(trimmed) {
        let payload = unwrap_peer(payload.trim());
        let lower = payload.to_lowercase();
        // A plugin relaying a task notification or a reminder is still harness output.
        if payload.is_empty() || injected(&lower) {
            return UserText::Injected;
        }
        return UserText::Relay(payload.to_string());
    }
    if trimmed.starts_with("<cross-session-message") {
        let payload = unwrap_peer(trimmed);
        return if payload.is_empty() {
            UserText::Injected
        } else {
            UserText::Relay(payload.to_string())
        };
    }
    if PLUGIN_PROMPT_PREFIXES.iter().any(|prefix| trimmed.starts_with(prefix))
        || trimmed.starts_with("Note from the cc-mods")
    {
        return UserText::Relay(trimmed.to_string());
    }
    if injected(&trimmed.to_lowercase()) {
        return UserText::Injected;
    }
    UserText::Human(trimmed.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn bookkeeping_records_are_not_turns() {
        assert!(is_turn(&json!({"type":"user","message":{"role":"user","content":"hi"}})));
        assert!(is_turn(&json!({"type":"assistant","message":{"role":"assistant","content":"hi"}})));
        assert!(is_turn(&json!({"message":{"role":"user","content":"old snapshot"}})));
        // The request copy repeats the prompt with every injected block.
        assert!(!is_turn(&json!({"type":"api-request-blob","message":{"role":"user","content":"x"}})));
        assert!(!is_turn(&json!({"type":"attachment"})));
        assert!(!is_turn(&json!({"type":"user","isMeta":true,"message":{"role":"user","content":"x"}})));
    }

    #[test]
    fn harness_text_is_injected() {
        for text in [
            "<system-reminder>\nContents of /Users/x/.claude/CLAUDE.md …</system-reminder>",
            "<task-notification> <task-id>b1</task-id></task-notification>",
            "Stop hook feedback: CLICKABLE LINKS",
            "<daily-activity>\n- 03:41 …</daily-activity>",
            "The cc-mods plugin sent a message: <task-notification> <summary>Monitor</summary></task-notification>\n\nThis is how Claude Code surfaces a prompt a plugin submits",
        ] {
            assert_eq!(classify_user_text(text), UserText::Injected, "{text}");
        }
    }

    #[test]
    fn relays_keep_their_payload_and_lose_the_lead_in() {
        assert_eq!(
            classify_user_text(
                "Another Claude session sent a message: <cross-session-message from=\"uds:/tmp/cc-socks/1.sock\" from-mode=\"bypass\">\n[tweb-6f] DONE install\n</cross-session-message>"
            ),
            UserText::Relay("[tweb-6f] DONE install".into())
        );
        assert_eq!(
            classify_user_text(
                "The cc-mods plugin sent a message: [worker x] READY sha=1\n\nThis is how Claude Code surfaces a prompt a plugin submits between turns."
            ),
            UserText::Relay("[worker x] READY sha=1".into())
        );
        assert!(matches!(
            classify_user_text("[cc-mods] Your turn ended without a `status` call."),
            UserText::Relay(_)
        ));
    }

    #[test]
    fn typed_prompts_are_human() {
        assert_eq!(
            classify_user_text("  rag/kmd 도  "),
            UserText::Human("rag/kmd 도".into())
        );
        // Mentioning a tag is not the tag.
        assert!(matches!(
            classify_user_text("why does <system-reminder> show up in learnings?"),
            UserText::Human(_)
        ));
    }
}
