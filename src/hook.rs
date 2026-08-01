//! Claude Code hook entrypoints.
//!
//! `kmd hook stop` refreshes the live activity card after each response.
//! `kmd hook session-end` persists the final transcript and queues reindexing once.
//! `kmd hook mark-dirty` records writes under configured collections.
//!
//! Hooks consume JSON on stdin and intentionally return success when optional inputs are
//! absent, because hook failures can block prompts or session shutdown.

use anyhow::Result;
use chrono::Local;
use serde_json::Value;
use std::fs;
use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

fn dirty_flag() -> PathBuf {
    crate::rag::state_dir().join(".kmd-dirty")
}

fn sync_log() -> PathBuf {
    crate::rag::state_dir().join("sync.log")
}

fn append_sync(line: &str) {
    let dir = crate::rag::state_dir();
    let _ = fs::create_dir_all(&dir);
    if let Ok(mut f) = fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(sync_log())
    {
        use std::io::Write;
        let _ = writeln!(f, "{}", line);
    }
}

fn read_hook_input() -> Value {
    let mut raw = String::new();
    let _ = std::io::stdin().read_to_string(&mut raw);
    serde_json::from_str(&raw).unwrap_or_default()
}

fn session_input(input: &Value) -> (String, Option<PathBuf>) {
    let session_id = input
        .get("session_id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let transcript_path = input
        .get("transcript_path")
        .and_then(Value::as_str)
        .filter(|path| !path.is_empty())
        .map(PathBuf::from);
    (session_id, transcript_path)
}

fn persist_dirty(reason: &str) {
    let dir = crate::rag::state_dir();
    let _ = fs::create_dir_all(&dir);
    if let Ok(mut file) = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(dirty_flag())
    {
        use std::io::Write;
        let _ = writeln!(file, "{}", reason);
    }
}

fn update_accepted(response: &Value) -> bool {
    let ok = response.get("ok").and_then(Value::as_bool).unwrap_or(false);
    let accepted = response
        .get("queued")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    ok && accepted
}

fn queue_update(reason: &str) -> bool {
    let t0 = Instant::now();
    let queued = crate::daemon::try_request(&serde_json::json!({"cmd":"update"}))
        .as_ref()
        .is_some_and(update_accepted);
    let elapsed = t0.elapsed().as_secs_f32();
    let ts = Local::now().format("%Y-%m-%dT%H:%M:%S");
    append_sync(&format!(
        "{} {} kmd queued in {:.1}s (ok={})",
        ts, reason, elapsed, queued
    ));
    queued
}

/// Stop hook: refresh the live activity card and flush explicitly dirty collections.
pub fn stop() -> Result<()> {
    let input = read_hook_input();
    let (session_id, transcript_path) = session_input(&input);

    if !session_id.is_empty() {
        if let Some(path) = transcript_path.as_deref().filter(|path| path.is_file()) {
            let _ = crate::activity::update_from_transcript(&session_id, path);
        }
    }

    let flag = dirty_flag();
    if flag.exists() {
        // Remove the observed signal before queueing so a concurrent writer can create
        // a fresh flag. Restore it when the daemon cannot accept this update.
        let _ = fs::remove_file(&flag);
        if !queue_update("dirty-files") {
            persist_dirty("dirty-files");
        }
    }
    Ok(())
}

/// SessionEnd hook: persist the final transcript as historical learning once.
pub fn session_end() -> Result<()> {
    let input = read_hook_input();
    let (session_id, transcript_path) = session_input(&input);
    if session_id.is_empty() {
        return Ok(());
    }

    let id8: String = session_id.chars().take(8).collect();
    let exported = if let Some(path) = transcript_path.as_deref().filter(|path| path.is_file()) {
        let _ = crate::activity::update_from_transcript(&session_id, path);
        crate::learnings::process_transcript(&session_id, path, false).unwrap_or(false)
    } else {
        crate::learnings::process_session(&session_id, false).unwrap_or(false)
    };
    if exported {
        let reason = format!("session:{}", id8);
        if !queue_update(&reason) {
            persist_dirty(&reason);
        }
    }
    Ok(())
}

fn expand_path(p: &str) -> PathBuf {
    if p.starts_with('~') {
        if let Ok(h) = std::env::var("HOME") {
            let rest = p.strip_prefix("~/").unwrap_or(&p[1..]);
            return PathBuf::from(h).join(rest);
        }
    }
    let pb = PathBuf::from(p);
    if pb.is_absolute() {
        pb
    } else {
        std::env::current_dir().unwrap_or_default().join(pb)
    }
}

/// PostToolUse hook. stdin: `{"tool_input": {"file_path": "..."}}`.
pub fn mark_dirty() -> Result<()> {
    let mut raw = String::new();
    let _ = std::io::stdin().read_to_string(&mut raw);
    let file_path = serde_json::from_str::<Value>(&raw)
        .ok()
        .and_then(|v| {
            v.get("tool_input")
                .and_then(|t| t.get("file_path"))
                .and_then(|f| f.as_str())
                .map(str::to_string)
        })
        .unwrap_or_default();
    if file_path.is_empty() {
        return Ok(());
    }

    let abs = expand_path(&file_path);

    // watched paths = kmd config 의 collection paths (single source of truth).
    // config 로드 실패 시 조용히 종료 — 훅은 실패하면 안 된다.
    let cfg = match crate::config::load() {
        Ok(c) => c,
        Err(_) => return Ok(()),
    };

    for c in cfg.collections.values() {
        if abs.starts_with(&c.path) {
            let dir = crate::rag::state_dir();
            let _ = fs::create_dir_all(&dir);
            if let Ok(mut f) = fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(dirty_flag())
            {
                use std::io::Write;
                let _ = writeln!(f, "{}", abs.display());
            }
            break;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_acceptance_requires_an_accepted_queue() {
        assert!(update_accepted(&serde_json::json!({"ok": true})));
        assert!(update_accepted(
            &serde_json::json!({"ok": true, "queued": true})
        ));
        assert!(!update_accepted(
            &serde_json::json!({"ok": true, "queued": false})
        ));
        assert!(!update_accepted(
            &serde_json::json!({"ok": false, "queued": true})
        ));
        assert!(!update_accepted(&serde_json::json!({})));
    }
}
