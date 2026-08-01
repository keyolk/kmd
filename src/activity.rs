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

fn collect_text_and_tools(
    role: &str,
    content: &Value,
    users: &mut Vec<String>,
    assistants: &mut Vec<String>,
    files: &mut Vec<String>,
) {
    match content {
        Value::String(text) => {
            if role == "user" && meaningful_user_text(text) {
                users.push(normalize(text, 280));
            } else if role == "assistant" && !text.trim().is_empty() {
                assistants.push(normalize(text, 240));
            }
        }
        Value::Array(blocks) => {
            for block in blocks {
                let kind = block.get("type").and_then(Value::as_str).unwrap_or("");
                if kind == "text" {
                    let text = block.get("text").and_then(Value::as_str).unwrap_or("");
                    if role == "user" && meaningful_user_text(text) {
                        users.push(normalize(text, 280));
                    } else if role == "assistant" && !text.trim().is_empty() {
                        assistants.push(normalize(text, 240));
                    }
                } else if kind == "tool_use" {
                    let input = block.get("input").and_then(Value::as_object);
                    if let Some(file) = input
                        .and_then(|i| i.get("file_path"))
                        .and_then(Value::as_str)
                    {
                        push_unique(files, file.to_string());
                    }
                }
            }
        }
        _ => {}
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
        if let Some(value) = entry.get("cwd").and_then(Value::as_str) {
            if !value.is_empty() {
                cwd = value.to_string();
            }
        }
        if let Some(value) = entry.get("timestamp").and_then(Value::as_str) {
            if let Ok(parsed) = DateTime::parse_from_rfc3339(value) {
                if latest.as_ref().is_none_or(|current| parsed > *current) {
                    latest = Some(parsed);
                }
            }
        }

        let message = entry.get("message").unwrap_or(&entry);
        let role = message.get("role").and_then(Value::as_str).unwrap_or("");
        let Some(content) = message.get("content") else {
            continue;
        };
        collect_text_and_tools(role, content, &mut users, &mut assistants, &mut files);
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

pub fn update_from_transcript(session_id: &str, transcript_path: &Path) -> Result<bool> {
    let raw = fs::read_to_string(transcript_path)?;
    let Some(card) = parse_transcript(&raw, session_id) else {
        return Ok(false);
    };

    let dir = activity_dir();
    fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{}.json", session_id));
    let body = serde_json::to_string_pretty(&card)? + "\n";
    if fs::read_to_string(&path).ok().as_deref() == Some(body.as_str()) {
        return Ok(false);
    }

    let pending = dir.join(format!(".{}.pending", session_id));
    fs::write(&pending, body)?;
    fs::rename(pending, path)?;
    Ok(true)
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

pub fn render_recent(
    current_session: &str,
    current_cwd: &str,
    limit: usize,
) -> Result<Option<String>> {
    let cards = recent(current_session, current_cwd, limit)?;
    if cards.is_empty() {
        return Ok(None);
    }

    let mut out = String::from(
        "<daily-activity>\n다른 Claude Code 세션의 최근 24시간 활동입니다. 현재 요청과 관련 있을 때만 참고하고, 중요한 상태는 다시 검증하세요.\n",
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
        out.push_str(&format!("- {} [{}:{}] {}\n", time, axis, id8, card.summary));
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
