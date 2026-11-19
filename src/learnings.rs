//! Extract historical learnings from Claude Code transcripts and snapshots.
//!
//! SessionEnd reads the live `transcript_path` directly, while the manual command can
//! still process the newest `~/.claude/snapshots/<session_id>/*.jsonl` fallback. Output
//! goes to `~/.claude/kmd-learnings/<YYYYMMDD>-<session[:8]>.md`; sessions with fewer
//! than four records are skipped.
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
    cwd: String,
    message_count: usize,
}

fn extract_from_snapshot(path: &Path) -> Option<Extracted> {
    let file = fs::File::open(path).ok()?;
    let reader = BufReader::new(file);
    let path_re = Regex::new(r"[\w/.-]+\.\w+").ok()?;
    let mut conversation = Vec::new();
    let mut tools_used = BTreeSet::new();
    let mut files_touched = BTreeSet::new();
    let mut cwd = String::new();
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
        if let Some(value) = entry.get("cwd").and_then(Value::as_str) {
            if !value.is_empty() {
                cwd = value.to_string();
            }
        }

        if !crate::transcript::is_turn(&entry) {
            continue;
        }
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
                        push_turn(&mut conversation, role, text);
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
            Some(Value::String(s)) => push_turn(&mut conversation, role, s),
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
        cwd,
        message_count,
    })
}

/// Keep what the person typed as `user`, relayed messages as `relay`, and drop text the
/// harness injected, so a learnings file never presents CLAUDE.md or a reminder as
/// something the user said.
fn push_turn(conversation: &mut Vec<(String, String)>, role: &str, text: &str) {
    if text.is_empty() {
        return;
    }
    if role != "user" {
        conversation.push((role.to_string(), text.to_string()));
        return;
    }
    match crate::transcript::classify_user_text(text) {
        crate::transcript::UserText::Human(text) => conversation.push(("user".into(), text)),
        crate::transcript::UserText::Relay(text) => conversation.push(("relay".into(), text)),
        crate::transcript::UserText::Injected => {}
    }
}

fn timestamp_from_stem(path: &Path) -> String {
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let taken: String = stem.chars().take(19).collect();
    taken.replace('T', " ")
}

fn timestamp_from_transcript(path: &Path) -> String {
    let latest = fs::File::open(path)
        .ok()
        .map(BufReader::new)
        .into_iter()
        .flat_map(|reader| reader.lines().map_while(Result::ok))
        .filter_map(|line| serde_json::from_str::<Value>(&line).ok())
        .filter_map(|entry| {
            entry
                .get("timestamp")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .max();

    latest
        .and_then(|value| chrono::DateTime::parse_from_rfc3339(&value).ok())
        .map(|value| {
            value
                .with_timezone(&chrono::Local)
                .format("%Y%m%d %H%M%S")
                .to_string()
        })
        .unwrap_or_else(|| chrono::Local::now().format("%Y%m%d %H%M%S").to_string())
}

fn generate_md(session_id: &str, data: &Extracted, timestamp: &str) -> String {
    let id8: String = session_id.chars().take(8).collect();
    let mut lines: Vec<String> = Vec::new();
    lines.push(format!("# Session: {}", id8));
    lines.push(format!("Date: {}", timestamp));
    if !data.cwd.is_empty() {
        lines.push(format!("Cwd: `{}`", data.cwd));
    }
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
        } else if role == "relay" {
            lines.push(format!("**Relay**: {}", clean));
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
    f.write_all(format!("{}\n", session_id).as_bytes())?;
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

    // Snapshot stems look like `20260223T1…`; keep the eight date digits only, or the
    // hour leaks into the name (`20260223 1-<id8>.md`, 286 such files by 2026-11-20).
    let date_prefix: String = timestamp.chars().filter(char::is_ascii_digit).take(8).collect();
    let out = write_learning(&learnings_dir(), session_id, &date_prefix, &md)?;
    mark_processed(session_id)?;
    println!(
        "  Extracted: {} ({} messages)",
        out.file_name().unwrap().to_string_lossy(),
        data.message_count
    );
    Ok(true)
}

/// Write `<date>-<id8>.md` and remove the session's exports under other dates.
///
/// The file name carries the date of the last turn, so a session resumed on a later day
/// used to get a second file holding the whole transcript again; search then returned
/// the same passage once per file (133 sessions had 2+ files on 2026-11-20). Each export
/// is the full transcript, so the newest one supersedes the rest.
fn write_learning(dir: &Path, session_id: &str, date_prefix: &str, md: &str) -> Result<PathBuf> {
    let id8: String = session_id.chars().take(8).collect();
    fs::create_dir_all(dir)?;
    let name = format!("{}-{}.md", date_prefix, id8);
    let out = dir.join(&name);
    fs::write(&out, md)?;
    for entry in fs::read_dir(dir)?.flatten() {
        let other = entry.file_name().to_string_lossy().to_string();
        if other != name && export_id8(&other).as_deref() == Some(id8.as_str()) {
            fs::remove_file(entry.path())?;
        }
    }
    Ok(out)
}

/// The session an export file belongs to: `<8 date digits>[ <hour>]-<id8>.md`.
fn export_id8(name: &str) -> Option<String> {
    let stem = name.strip_suffix(".md")?;
    let (date, id8) = stem.rsplit_once('-')?;
    let dated = date.len() >= 8 && date.chars().take(8).all(|c| c.is_ascii_digit());
    let hex = id8.len() == 8 && id8.chars().all(|c| c.is_ascii_hexdigit());
    (dated && hex).then(|| id8.to_string())
}

/// Process a live Claude Code transcript without waiting for compaction.
pub fn process_transcript(session_id: &str, path: &Path, dry_run: bool) -> Result<bool> {
    let data = match extract_from_snapshot(path) {
        Some(data) => data,
        None => return Ok(false),
    };
    if data.message_count < 4 {
        return Ok(false);
    }

    let timestamp = timestamp_from_transcript(path);
    let md = generate_md(session_id, &data, &timestamp);
    if dry_run {
        println!("\n{}", "=".repeat(60));
        println!("Session: {}", session_id);
        println!("{}", "=".repeat(60));
        println!("{}", md);
        return Ok(true);
    }

    let date_prefix = timestamp.get(..8).unwrap_or(&timestamp);
    let out = write_learning(&learnings_dir(), session_id, date_prefix, &md)?;
    println!(
        "  Extracted live transcript: {} ({} messages)",
        out.file_name().unwrap().to_string_lossy(),
        data.message_count
    );
    Ok(true)
}

/// Process a single compacted session snapshot by id.
pub fn process_session(session_id: &str, dry_run: bool) -> Result<bool> {
    let dir = snapshots_dir().join(session_id);
    if !dir.is_dir() {
        return Ok(false);
    }
    process_session_dir(&dir, session_id, dry_run)
}

/// Turn heads `generate_md` writes. A turn runs until the next head line.
const TURN_HEADS: &[&str] = &["**User**: ", "**Relay**: ", "**Assistant**: ", "_Messages: "];

/// Re-apply the user/relay/injected split to an existing export whose transcript is gone.
fn scrub_markdown(md: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut turn: Vec<&str> = Vec::new();
    let flush = |turn: &mut Vec<&str>, out: &mut Vec<String>| {
        if turn.is_empty() {
            return;
        }
        let block = turn.join("\n");
        turn.clear();
        if let Some(text) = block.strip_prefix("**User**: ") {
            match crate::transcript::classify_user_text(text) {
                crate::transcript::UserText::Human(_) => out.push(block),
                crate::transcript::UserText::Relay(relay) => out.push(format!("**Relay**: {}\n", relay)),
                crate::transcript::UserText::Injected => {}
            }
        } else {
            out.push(block);
        }
    };
    for line in md.lines() {
        if TURN_HEADS.iter().any(|head| line.starts_with(head)) {
            flush(&mut turn, &mut out);
        }
        turn.push(line);
    }
    flush(&mut turn, &mut out);
    out.join("\n")
}

/// The transcript a learnings export came from, if Claude Code still keeps it.
fn transcript_for(id8: &str) -> Option<(String, PathBuf)> {
    let projects = home(".claude/projects");
    let mut found: Vec<(SystemTime, String, PathBuf)> = Vec::new();
    for dir in fs::read_dir(projects).ok()?.flatten() {
        let Ok(entries) = fs::read_dir(dir.path()) else { continue };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if let Some(sid) = name.strip_suffix(".jsonl").filter(|sid| sid.starts_with(id8)) {
                let mtime = entry.metadata().and_then(|m| m.modified()).unwrap_or(SystemTime::UNIX_EPOCH);
                found.push((mtime, sid.to_string(), entry.path()));
            }
        }
    }
    found.sort_by_key(|(mtime, _, _)| *mtime);
    found.pop().map(|(_, sid, path)| (sid, path))
}

/// `kmd learnings-extract --rebuild`: bring every existing export up to the current rules.
pub fn rebuild(dry_run: bool) -> Result<()> {
    let dir = learnings_dir();
    let mut by_session: std::collections::BTreeMap<String, Vec<String>> = Default::default();
    for entry in fs::read_dir(&dir)?.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if let Some(id8) = export_id8(&name) {
            by_session.entry(id8).or_default().push(name);
        }
    }
    let (mut dropped, mut reexported, mut scrubbed, mut unchanged) = (0, 0, 0, 0);
    for (id8, mut names) in by_session {
        names.sort();
        let keep = names.pop().expect("one file per group");
        for old in names {
            dropped += 1;
            if !dry_run {
                fs::remove_file(dir.join(&old))?;
            }
        }
        if let Some((sid, path)) = transcript_for(&id8) {
            if !dry_run && process_transcript_quiet(&sid, &path)? {
                reexported += 1;
                continue;
            }
            if dry_run {
                reexported += 1;
                continue;
            }
        }
        let path = dir.join(&keep);
        let md = fs::read_to_string(&path)?;
        let clean = scrub_markdown(&md);
        if clean == md {
            unchanged += 1;
        } else {
            scrubbed += 1;
            if !dry_run {
                fs::write(&path, clean)?;
            }
        }
    }
    println!(
        "learnings rebuild{}: {} re-exported from transcripts, {} scrubbed in place, {} unchanged, {} superseded files removed",
        if dry_run { " (dry run)" } else { "" },
        reexported,
        scrubbed,
        unchanged,
        dropped
    );
    Ok(())
}

fn process_transcript_quiet(session_id: &str, path: &Path) -> Result<bool> {
    let Some(data) = extract_from_snapshot(path) else { return Ok(false) };
    if data.message_count < 4 {
        return Ok(false);
    }
    let timestamp = timestamp_from_transcript(path);
    let md = generate_md(session_id, &data, &timestamp);
    write_learning(&learnings_dir(), session_id, timestamp.get(..8).unwrap_or(&timestamp), &md)?;
    Ok(true)
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
            SystemTime::now().checked_sub(std::time::Duration::from_secs(u64::from(recent) * 86400))
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

    let action = if dry_run {
        "Would extract"
    } else {
        "Extracted"
    };
    println!("\n{} learnings from {} sessions", action, count);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("kmd-learnings-{}-{}", name, std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn export_keeps_typed_prompts_and_drops_injected_copies() {
        let dir = scratch("extract");
        let path = dir.join("t.jsonl");
        fs::write(
            &path,
            r#"{"type":"user","cwd":"/Users/x","message":{"role":"user","content":"kmd 결함을 고치자"}}
{"type":"api-request-blob","message":{"role":"user","content":[{"type":"text","text":"<system-reminder>\nContents of /Users/x/.claude/CLAUDE.md\n</system-reminder>"},{"type":"text","text":"kmd 결함을 고치자"}]}}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"transcript 분류 모듈을 만들고 learnings와 activity가 같은 규칙을 쓰게 했습니다. 테스트도 넣었습니다."}]}}
{"type":"user","message":{"role":"user","content":"Another Claude session sent a message: <cross-session-message from-mode=\"bypass\">\n[worker w] DONE\n</cross-session-message>"}}
{"type":"user","message":{"role":"user","content":[{"type":"text","text":"<task-notification> <task-id>b1</task-id></task-notification>"}]}}
"#,
        )
        .unwrap();
        let data = extract_from_snapshot(&path).expect("extracted");
        let md = generate_md("abcdef12-0000", &data, "20261120 000000");
        assert_eq!(md.matches("**User**: kmd 결함을 고치자").count(), 1);
        assert!(md.contains("**Relay**: [worker w] DONE"));
        assert!(!md.contains("CLAUDE.md"));
        assert!(!md.contains("task-notification"));
        assert_eq!(data.message_count, 4);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn scrub_drops_injected_user_turns_and_relabels_relays() {
        let md = "# Session: abcdef12\n\n## Conversation\n\n**User**: 진짜 질문\n\n**User**: <system-reminder>\nContents of CLAUDE.md\n\n- rule\n</system-reminder>\n\n**Assistant**: 답변입니다. 충분히 긴 답변이어야 내보낸다고 가정합니다.\n\n**User**: Another Claude session sent a message: <cross-session-message from-mode=\"bypass\">\n[w] DONE\n</cross-session-message>\n\n_Messages: 4_";
        let out = scrub_markdown(md);
        assert!(out.contains("**User**: 진짜 질문"));
        assert!(!out.contains("CLAUDE.md"));
        assert!(out.contains("**Relay**: [w] DONE"));
        assert!(out.contains("**Assistant**: 답변입니다"));
        assert!(out.ends_with("_Messages: 4_"));
    }

    #[test]
    fn a_resumed_session_keeps_one_export() {
        let dir = scratch("dedupe");
        fs::write(dir.join("20260908-4c769ec1.md"), "old").unwrap();
        fs::write(dir.join("20260912-4c769ec1.md"), "older").unwrap();
        fs::write(dir.join("20260912-99999999.md"), "other session").unwrap();
        fs::write(dir.join("20260224 0-4c769ec1.md"), "snapshot-era name").unwrap();
        fs::write(dir.join("README.md"), "not an export").unwrap();
        let out = write_learning(&dir, "4c769ec1-aaaa", "20260915", "new").unwrap();
        let mut names: Vec<String> = fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        names.sort();
        assert_eq!(names, vec!["20260912-99999999.md", "20260915-4c769ec1.md", "README.md"]);
        assert_eq!(fs::read_to_string(out).unwrap(), "new");
        let _ = fs::remove_dir_all(dir);
    }
}
