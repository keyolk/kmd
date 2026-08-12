//! Cross-session activity cards built from live Claude Code transcripts.
//!
//! Stop hooks refresh one card per session. UserPromptSubmit hooks read recent cards
//! directly, so daily awareness does not depend on snapshot creation or BM25 indexing.

use anyhow::Result;
use chrono::{DateTime, Local, Utc};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};

const WINDOW_HOURS: i64 = 24;
const DEFAULT_LIMIT: usize = 6;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityCard {
    pub session_id: String,
    pub updated_at: String,
    pub updated_epoch: i64,
    pub cwd: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repo: Option<String>,
    pub summary: String,
    pub latest_user: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub latest_assistant: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub links: Vec<String>,
}

fn activity_dir() -> PathBuf {
    crate::rag::state_dir().join("activity")
}

fn normalize(text: &str, limit: usize) -> String {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.chars().count() <= limit {
        return compact;
    }
    format!("{}…", compact.chars().take(limit).collect::<String>())
}

fn meaningful_user_text(text: &str) -> bool {
    let text = text.trim();
    if text.chars().count() < 4 {
        return false;
    }
    let lower = text.to_lowercase();
    const PREFIXES: &[&str] = &[
        "<local-command",
        "<command-",
        "<system-reminder",
        "<task-notification",
        "<tool-use",
        "<bash-",
        "<teammate-",
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
    if PREFIXES.iter().any(|prefix| lower.starts_with(prefix)) {
        return false;
    }
    !matches!(lower.as_str(), "continue" | "계속" | "진행해" | "해줘")
}

fn push_unique(values: &mut Vec<String>, value: String) {
    if !value.is_empty() && !values.contains(&value) {
        values.push(value);
    }
}

fn collect_text_and_tools(content: &Value, files: &mut Vec<String>) -> Vec<String> {
    match content {
        Value::String(text) => vec![text.to_string()],
        Value::Array(blocks) => {
            let mut texts = Vec::new();
            for block in blocks {
                match block.get("type").and_then(Value::as_str).unwrap_or("") {
                    "text" => {
                        if let Some(text) = block.get("text").and_then(Value::as_str) {
                            texts.push(text.to_string());
                        }
                    }
                    "tool_use" => {
                        let input = block.get("input").and_then(Value::as_object);
                        if let Some(file) = input
                            .and_then(|value| value.get("file_path"))
                            .and_then(Value::as_str)
                        {
                            push_unique(files, file.to_string());
                        }
                    }
                    _ => {}
                }
            }
            texts
        }
        _ => Vec::new(),
    }
}

fn parse_transcript(raw: &str, session_id: &str) -> Option<ActivityCard> {
    let url_re = Regex::new(r#"https?://[^\s<>\)\]\}"']+"#).ok()?;
    let mut users = Vec::new();
    let mut assistants = Vec::new();
    let mut files = Vec::new();
    let mut cwd = String::new();
    let mut latest: Option<DateTime<chrono::FixedOffset>> = None;

    for line in raw.lines() {
        let Ok(entry) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if let Some(value) = entry.get("cwd").and_then(Value::as_str)
            && !value.is_empty()
        {
            cwd = value.to_string();
        }
        let timestamp = entry
            .get("timestamp")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if let Ok(parsed) = DateTime::parse_from_rfc3339(timestamp)
            && latest.as_ref().is_none_or(|current| parsed > *current)
        {
            latest = Some(parsed);
        }

        let message = entry.get("message").unwrap_or(&entry);
        let role = message.get("role").and_then(Value::as_str).unwrap_or("");
        let Some(content) = message.get("content") else {
            continue;
        };
        let texts = collect_text_and_tools(content, &mut files);
        match role {
            "user" => {
                let text = texts
                    .into_iter()
                    .filter(|text| meaningful_user_text(text))
                    .collect::<Vec<_>>()
                    .join("\n\n");
                if !text.is_empty() {
                    users.push(normalize(&text, 280));
                }
            }
            "assistant" => {
                let text = texts
                    .into_iter()
                    .filter(|text| !text.trim().is_empty())
                    .collect::<Vec<_>>()
                    .join("\n\n");
                if !text.is_empty() {
                    assistants.push(normalize(&text, 240));
                }
            }
            _ => {}
        }
    }

    let latest_user = users.last()?.clone();
    let summary = users
        .iter()
        .rev()
        .find(|text| text.chars().count() >= 20)
        .cloned()
        .unwrap_or_else(|| latest_user.clone());
    let mut links = Vec::new();
    for text in [&summary, &latest_user] {
        for found in url_re.find_iter(text) {
            push_unique(&mut links, found.as_str().to_string());
        }
    }
    let updated = latest
        .map(|value| value.with_timezone(&Utc))
        .unwrap_or_else(Utc::now);
    let repo = crate::pageindex::repo_of(&cwd).or_else(|| {
        files
            .iter()
            .rev()
            .find_map(|path| crate::pageindex::repo_of(path))
    });

    Some(ActivityCard {
        session_id: session_id.to_string(),
        updated_at: updated.to_rfc3339(),
        updated_epoch: updated.timestamp(),
        cwd,
        repo,
        summary,
        latest_user,
        latest_assistant: assistants.last().cloned().unwrap_or_default(),
        files: files
            .into_iter()
            .rev()
            .take(5)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect(),
        links: links
            .into_iter()
            .rev()
            .take(3)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect(),
    })
}

fn persist_card(card: &ActivityCard) -> Result<bool> {
    let dir = activity_dir();
    fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{}.json", card.session_id));
    let body = serde_json::to_string_pretty(card)? + "\n";
    if fs::read_to_string(&path).ok().as_deref() == Some(body.as_str()) {
        return Ok(false);
    }

    // Stop hooks and dashboard migration can overlap, so each process needs its own atomic temp file.
    let pending = dir.join(format!(
        ".{}.{}.pending",
        card.session_id,
        std::process::id()
    ));
    fs::write(&pending, body)?;
    fs::rename(pending, path)?;
    Ok(true)
}

pub fn update_from_transcript(session_id: &str, transcript_path: &Path) -> Result<bool> {
    let raw = fs::read_to_string(transcript_path)?;
    let Some(card) = parse_transcript(&raw, session_id) else {
        return Ok(false);
    };
    persist_card(&card)
}

pub fn recent(current_session: &str, current_cwd: &str, limit: usize) -> Result<Vec<ActivityCard>> {
    let now = Utc::now().timestamp();
    let current_repo = crate::pageindex::repo_of(current_cwd);
    let mut cards = Vec::new();
    let Ok(entries) = fs::read_dir(activity_dir()) else {
        return Ok(cards);
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let Ok(raw) = fs::read_to_string(path) else {
            continue;
        };
        let Ok(card) = serde_json::from_str::<ActivityCard>(&raw) else {
            continue;
        };
        let age = now - card.updated_epoch;
        if card.session_id != current_session && (0..=WINDOW_HOURS * 3600).contains(&age) {
            cards.push(card);
        }
    }

    cards.sort_by(|a, b| {
        let a_same = current_repo.is_some() && a.repo == current_repo;
        let b_same = current_repo.is_some() && b.repo == current_repo;
        b_same
            .cmp(&a_same)
            .then(b.updated_epoch.cmp(&a.updated_epoch))
    });
    cards.truncate(if limit == 0 { DEFAULT_LIMIT } else { limit });
    Ok(cards)
}

fn short_file(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// A session is `active` while it is still sending kmd queries, `recent` when it went
/// quiet within the last hour, and `idle` beyond that.
pub const ACTIVE_WINDOW_SECS: i64 = 15 * 60;
pub const RECENT_WINDOW_SECS: i64 = 60 * 60;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionActivity {
    Active,
    Recent,
    Idle,
}

impl SessionActivity {
    pub fn label(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Recent => "recent",
            Self::Idle => "idle",
        }
    }

    /// Classify by how long ago the session last talked to kmd. The UserPromptSubmit hook
    /// only runs inside a running session, so that timestamp is proof the session was
    /// alive at that moment — kmd needs no separate pid or heartbeat probe.
    pub fn from_last_query(now: i64, last_query_epoch: i64) -> Self {
        if last_query_epoch <= 0 {
            return Self::Idle;
        }
        match now.saturating_sub(last_query_epoch) {
            age if age <= ACTIVE_WINDOW_SECS => Self::Active,
            age if age <= RECENT_WINDOW_SECS => Self::Recent,
            _ => Self::Idle,
        }
    }
}

/// session id prefix → last kmd query epoch, from the RAG log.
/// Keyed on the shared 8-character prefix because activity cards may store a truncated id.
pub fn last_query_epochs() -> std::collections::HashMap<String, i64> {
    let mut latest = std::collections::HashMap::new();
    let Ok(entries) = crate::stats::load_entries(Some(RECENT_WINDOW_SECS as u64)) else {
        return latest;
    };
    for entry in entries {
        let Ok(ts) = entry.ts.parse::<i64>() else {
            continue;
        };
        let key: String = entry.session_id.chars().take(8).collect();
        if key.is_empty() {
            continue;
        }
        latest
            .entry(key)
            .and_modify(|current| *current = (*current).max(ts))
            .or_insert(ts);
    }
    latest
}

pub fn render_recent(
    current_session: &str,
    current_cwd: &str,
    limit: usize,
) -> Result<Option<String>> {
    let cards = recent(current_session, current_cwd, limit)?;
    if cards.is_empty() {
        return Ok(None);
    }

    let now = Utc::now().timestamp();
    let last_queries = last_query_epochs();
    let mut out = String::from(
        "<daily-activity>\n다른 Claude Code 세션의 최근 24시간 활동입니다. `active`는 지금 kmd에 질의 중인 세션, `recent`는 최근 1시간 내 질의한 세션입니다. 현재 요청과 관련 있을 때만 참고하고, 중요한 상태는 다시 검증하세요.\n",
    );
    for card in cards {
        let id8: String = card.session_id.chars().take(8).collect();
        let time = DateTime::parse_from_rfc3339(&card.updated_at)
            .map(|value| value.with_timezone(&Local).format("%H:%M").to_string())
            .unwrap_or_else(|_| "--:--".to_string());
        let axis = card.repo.as_deref().unwrap_or_else(|| {
            card.cwd
                .trim_end_matches('/')
                .rsplit('/')
                .next()
                .unwrap_or("unknown")
        });
        let activity =
            SessionActivity::from_last_query(now, last_queries.get(&id8).copied().unwrap_or(0));
        out.push_str(&format!(
            "- {} [{}:{}] ({}) {}\n",
            time,
            axis,
            id8,
            activity.label(),
            card.summary
        ));
        if card.latest_user != card.summary {
            out.push_str(&format!("  현재 요청: {}\n", card.latest_user));
        }
        if !card.latest_assistant.is_empty() {
            out.push_str(&format!("  결과: {}\n", card.latest_assistant));
        }
        if !card.files.is_empty() {
            let files = card
                .files
                .iter()
                .rev()
                .take(3)
                .map(|path| short_file(path))
                .collect::<Vec<_>>();
            out.push_str(&format!("  파일: {}\n", files.join(", ")));
        }
        if !card.links.is_empty() {
            out.push_str(&format!("  링크: {}\n", card.links.join(", ")));
        }
    }
    out.push_str("</daily-activity>");
    Ok(Some(out))
}

pub fn run(limit: usize, json: bool) -> Result<()> {
    let cards = recent("", "", limit)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&cards)?);
    } else if let Some(rendered) = render_recent("", "", limit)? {
        println!("{}", rendered);
    } else {
        println!("최근 24시간의 다른 세션 활동이 없습니다.");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_session_activity_by_last_kmd_query() {
        let now = 1_754_010_000_i64;

        assert_eq!(
            SessionActivity::from_last_query(now, now - 60),
            SessionActivity::Active
        );
        assert_eq!(
            SessionActivity::from_last_query(now, now - ACTIVE_WINDOW_SECS),
            SessionActivity::Active
        );
        assert_eq!(
            SessionActivity::from_last_query(now, now - ACTIVE_WINDOW_SECS - 1),
            SessionActivity::Recent
        );
        assert_eq!(
            SessionActivity::from_last_query(now, now - RECENT_WINDOW_SECS - 1),
            SessionActivity::Idle
        );
        // No query at all means kmd has no evidence the session is running.
        assert_eq!(
            SessionActivity::from_last_query(now, 0),
            SessionActivity::Idle
        );
    }

    #[test]
    fn parses_live_transcript_into_activity_card() {
        let raw = r#"
{"type":"user","sessionId":"abc","cwd":"/Users/x/src/sendbird/platform-tools","timestamp":"2026-08-01T01:00:00Z","message":{"role":"user","content":"<command-name>/clear</command-name>"}}
{"type":"user","sessionId":"abc","cwd":"/Users/x/src/sendbird/platform-tools","timestamp":"2026-08-01T01:01:00Z","message":{"role":"user","content":"Delight region 지원을 마저 구현하자"}}
{"type":"assistant","sessionId":"abc","cwd":"/Users/x/src/sendbird/platform-tools","timestamp":"2026-08-01T01:02:00Z","message":{"role":"assistant","content":[{"type":"tool_use","name":"Edit","input":{"file_path":"/Users/x/src/sendbird/platform-tools/main.go"}},{"type":"text","text":"region identity에 product 차원을 추가했고 테스트도 통과했습니다."}]}}
"#;
        let card = parse_transcript(raw, "abc").expect("card");
        assert_eq!(card.repo.as_deref(), Some("platform-tools"));
        assert_eq!(card.summary, "Delight region 지원을 마저 구현하자");
        assert_eq!(
            card.files,
            vec!["/Users/x/src/sendbird/platform-tools/main.go"]
        );
        assert!(card.latest_assistant.contains("테스트도 통과"));
    }

    #[test]
    fn keeps_recent_specific_topic_over_short_follow_up() {
        let raw = r#"
{"type":"user","sessionId":"abc","cwd":"/Users/x","timestamp":"2026-08-01T01:00:00Z","message":{"role":"user","content":"현재 여러 세션의 당일 작업 공유 구조를 개선해 보자 https://example.com/current"}}
{"type":"user","sessionId":"abc","cwd":"/Users/x","timestamp":"2026-08-01T01:01:00Z","message":{"role":"user","content":"개선해보자"}}
"#;
        let card = parse_transcript(raw, "abc").expect("card");
        assert!(card.summary.starts_with("현재 여러 세션의 당일 작업 공유"));
        assert_eq!(card.latest_user, "개선해보자");
        assert_eq!(card.links, vec!["https://example.com/current"]);
    }

    #[test]
    fn skips_resume_and_short_control_prompts() {
        assert!(!meaningful_user_text("Tool loaded."));
        assert!(!meaningful_user_text("계속"));
        assert!(!meaningful_user_text(
            "Skill /sb:pr-followup is already loaded above; instructions unchanged."
        ));
        assert!(!meaningful_user_text(
            "<bash-stdout>Attempting to automatically open the SSO authorization page"
        ));
        assert!(!meaningful_user_text(
            "Stop hook feedback: STOP HOOK VIOLATION: NOTHING IS PRE-EXISTING."
        ));
        assert!(!meaningful_user_text(
            "A session-scoped Stop hook is now active with condition: deploy"
        ));
        assert!(!meaningful_user_text(
            "[Your previous response had no visible output. Please continue.]"
        ));
        assert!(!meaningful_user_text(
            "[Image: source: /Users/x/.claude/image-cache/session/1.png]"
        ));
        assert!(meaningful_user_text("현재 구현 상태를 다시 확인해 보자"));
    }
}
