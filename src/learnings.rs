//! Extract learnings from Claude Code conversation snapshots.
//!
//! `~/.claude/snapshots/<session_id>/*.jsonl` 의 mtime 최신 스냅샷 하나를 읽어
//! `~/.claude/kmd-learnings/<YYYYMMDD>-<session[:8]>.md` 로 출력. 짧은 세션
//! (message_count < 4)은 건너뛴다. 처리된 session_id는
//! `~/.claude/.kmd-processed-snapshots` 에 append하여 dedup.
//!
//! 이전 Python 구현(`extract_learnings.py`)과 동일한 추출 정책:
///   - content가 string이면 user/assistant 텍스트로; array면 text 블록만 대화로,
///     tool_use 블록은 tools_used/files_touched 메타로 추출.
///   - assistant 텍스트는 50자 초과만, 1500자 캡. user 프롬프트는 전문.
///   - tool_result(명령 출력)은 버린다.

use anyhow::Result;
use regex::Regex;
use serde_json::Value;
use std::collections::{BTreeSet, HashSet};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

fn home(p: &str) -> PathBuf {
    let h = std::env::var("HOME").expect("HOME");
    PathBuf::from(h).join(p)
}

fn snapshots_dir() -> PathBuf {
    home(".claude/snapshots")
}

fn learnings_dir() -> PathBuf {
    home(".claude/kmd-learnings")
}

fn processed_log() -> PathBuf {
    home(".claude/.kmd-processed-snapshots")
}

struct Extracted {
    conversation: Vec<(String, String)>, // (role, text)
    tools_used: BTreeSet<String>,
    files_touched: BTreeSet<String>,
    message_count: usize,
}

fn extract_from_snapshot(path: &Path) -> Option<Extracted> {
    let file = fs::File::open(path).ok()?;
    let reader = BufReader::new(file);
    let path_re = Regex::new(r"[\w/.-]+\.\w+").ok()?;
    let mut conversation = Vec::new();
    let mut tools_used = BTreeSet::new();
    let mut files_touched = BTreeSet::new();
    let mut message_count = 0;

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let entry: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        message_count += 1;

        let msg = entry.get("message").unwrap_or(&entry);
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
        let content = msg.get("content");
        match content {
            Some(Value::Array(blocks)) => {
                for block in blocks {
                    let Some(b) = block.as_object() else { continue };
                    let btype = b.get("type").and_then(|t| t.as_str()).unwrap_or("");
                    if btype == "text" {
                        let text = b.get("text").and_then(|t| t.as_str()).unwrap_or("");
                        if !text.is_empty() {
                            conversation.push((role.to_string(), text.to_string()));
                        }
                    } else if btype == "tool_use" {
                        let name = b.get("name").and_then(|n| n.as_str()).unwrap_or("unknown");
                        tools_used.insert(name.to_string());
                        if let Some(input) = b.get("input").and_then(|i| i.as_object()) {
                            if let Some(fp) = input.get("file_path").and_then(|f| f.as_str()) {
                                files_touched.insert(fp.to_string());
                            }
                            if let Some(cmd) = input.get("command").and_then(|c| c.as_str()) {
                                for m in path_re.find_iter(cmd) {
                                    let p = m.as_str();
                                    if p.contains('/') {
                                        files_touched.insert(p.to_string());
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Some(Value::String(s)) => {
                if !s.is_empty() {
                    conversation.push((role.to_string(), s.clone()));
                }
            }
            _ => {}
        }
    }

    if message_count == 0 {
        return None;
    }
    Some(Extracted {
        conversation,
        tools_used,
        files_touched,
        message_count,
    })
}

fn timestamp_from_stem(path: &Path) -> String {
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let taken: String = stem.chars().take(19).collect();
    taken.replace('T', " ")
}

fn generate_md(session_id: &str, data: &Extracted, timestamp: &str) -> String {
    let id8: String = session_id.chars().take(8).collect();
    let mut lines: Vec<String> = Vec::new();
    lines.push(format!("# Session: {}", id8));
    lines.push(format!("Date: {}", timestamp));
    lines.push(String::new());

    let tools: Vec<String> = data.tools_used.iter().take(15).cloned().collect();
    let files: Vec<String> = data.files_touched.iter().take(15).cloned().collect();
    if !tools.is_empty() {
        lines.push(format!("**Tools**: {}", tools.join(", ")));
    }
    if !files.is_empty() {
        let quoted: Vec<String> = files.iter().map(|f| format!("`{}`", f)).collect();
        lines.push(format!("**Files**: {}", quoted.join(", ")));
    }
    if !tools.is_empty() || !files.is_empty() {
        lines.push(String::new());
    }

    lines.push("## Conversation".to_string());
    lines.push(String::new());

    for (role, text) in &data.conversation {
        let clean = text.trim();
        if clean.is_empty() {
            continue;
        }
        if role == "user" {
            lines.push(format!("**User**: {}", clean));
            lines.push(String::new());
        } else if role == "assistant" {
            let len = clean.chars().count();
            if len > 50 {
                let body = if len > 1500 {
                    let cut: String = clean.chars().take(1500).collect();
                    format!("{}...", cut)
                } else {
                    clean.to_string()
                };
                lines.push(format!("**Assistant**: {}", body));
                lines.push(String::new());
            }
        }
    }

    lines.push(format!("_Messages: {}_", data.message_count));
    lines.join("\n")
}

fn latest_snapshot(dir: &Path) -> Option<PathBuf> {
    let mut snaps: Vec<(SystemTime, PathBuf)> = fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) == Some("jsonl") {
                let mtime = e.metadata().ok()?.modified().ok()?;
                Some((mtime, p))
            } else {
                None
            }
        })
        .collect();
    snaps.sort_by_key(|(m, _)| *m);
    snaps.last().map(|(_, p)| p.clone())
}

fn load_processed() -> HashSet<String> {
    fs::read_to_string(processed_log())
        .ok()
        .map(|s| {
            s.lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

fn mark_processed(session_id: &str) -> Result<()> {
    if let Some(parent) = processed_log().parent() {
        let _ = fs::create_dir_all(parent);
    }
    let mut f = fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(processed_log())?;
    writeln!(f, "{}", session_id)?;
    Ok(())
}

fn process_session_dir(dir: &Path, session_id: &str, dry_run: bool) -> Result<bool> {
    let latest = match latest_snapshot(dir) {
        Some(p) => p,
        None => return Ok(false),
    };
    let data = match extract_from_snapshot(&latest) {
        Some(d) => d,
        None => return Ok(false),
    };
    if data.message_count < 4 {
        return Ok(false);
    }
    let timestamp = timestamp_from_stem(&latest);
    let md = generate_md(session_id, &data, &timestamp);

    if dry_run {
        println!("\n{}", "=".repeat(60));
        println!("Session: {}", session_id);
        println!("{}", "=".repeat(60));
        println!("{}", md);
        return Ok(true);
    }

    let date_prefix = timestamp.get(..10).unwrap_or(&timestamp).replace('-', "");
    let id8: String = session_id.chars().take(8).collect();
    fs::create_dir_all(learnings_dir())?;
    let out = learnings_dir().join(format!("{}-{}.md", date_prefix, id8));
    fs::write(&out, md)?;
    mark_processed(session_id)?;
    println!(
        "  Extracted: {} ({} messages)",
        out.file_name().unwrap().to_string_lossy(),
        data.message_count
    );
    Ok(true)
}

/// Process a single session by id. Returns whether a learning file was produced.
pub fn process_session(session_id: &str, dry_run: bool) -> Result<bool> {
    let dir = snapshots_dir().join(session_id);
    if !dir.is_dir() {
        return Ok(false);
    }
    process_session_dir(&dir, session_id, dry_run)
}

/// CLI entrypoint mirroring the previous Python `extract_learnings.py` flags.
pub fn run(session: Option<&str>, recent: u32, dry_run: bool, force: bool) -> Result<()> {
    if !snapshots_dir().is_dir() {
        println!("No snapshots directory found");
        return Ok(());
    }
    fs::create_dir_all(learnings_dir())?;

    let processed: HashSet<String> = if force {
        HashSet::new()
    } else {
        load_processed()
    };

    let mut count = 0;
    if let Some(sid) = session {
        let dir = snapshots_dir().join(sid);
        if !dir.is_dir() {
            println!("Session not found: {}", sid);
            return Ok(());
        }
        if process_session_dir(&dir, sid, dry_run)? {
            count += 1;
        }
    } else {
        let cutoff = if recent > 0 {
            SystemTime::now()
                .checked_sub(std::time::Duration::from_secs(u64::from(recent) * 86400))
        } else {
            None
        };

        let mut dirs: Vec<(PathBuf, String)> = fs::read_dir(snapshots_dir())?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                if !force && processed.contains(&name) {
                    return None;
                }
                if let Some(c) = cutoff {
                    let m = e.metadata().ok()?.modified().ok()?;
                    if m < c {
                        return None;
                    }
                }
                Some((e.path(), name))
            })
            .collect();
        dirs.sort_by(|a, b| a.1.cmp(&b.1));

        for (dir, name) in dirs {
            if process_session_dir(&dir, &name, dry_run)? {
                count += 1;
            }
        }
    }

    let action = if dry_run { "Would extract" } else { "Extracted" };
    println!("\n{} learnings from {} sessions", action, count);
    Ok(())
}
