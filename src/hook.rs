//! Claude Code hook entrypoints — kmd 바이너리가 Python 스크립트를 대체.
//!
//! `kmd hook stop`     — Stop hook: 세션 learnings 추출 + 데몬에 reindex 큐잉.
//! `kmd hook mark-dirty` — PostToolUse hook: watched collection 파일 변경 시 dirty 플래그.
//!
//! rag::run_hook와 동일한 규약: stdin JSON 읽고, 절대 실패하지 않는다(Ok(())).
//! Claude Code 훅이 비정상 종료하면 사용자 프롬프트/세션이 차단되기 때문.

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

/// Stop hook. stdin: `{"session_id": "..."}`.
pub fn stop() -> Result<()> {
    let mut raw = String::new();
    let _ = std::io::stdin().read_to_string(&mut raw);
    let session_id = serde_json::from_str::<Value>(&raw)
        .ok()
        .and_then(|v| v.get("session_id").and_then(|s| s.as_str()).map(str::to_string))
        .unwrap_or_default();

    let mut reasons: Vec<String> = Vec::new();
    let mut exported = false;

    if !session_id.is_empty() {
        let id8: String = session_id.chars().take(8).collect();
        match crate::learnings::process_session(&session_id, false) {
            Ok(true) => {
                exported = true;
                reasons.push(format!("session:{}", id8));
            }
            _ => {}
        }
    }

    let flag = dirty_flag();
    let dirty = flag.exists();
    if dirty {
        let _ = fs::remove_file(&flag);
        reasons.push("dirty-files".to_string());
    }

    if exported || dirty {
        let t0 = Instant::now();
        let queued_ok = crate::daemon::try_request(&serde_json::json!({"cmd":"update"}))
            .map(|r| r.get("ok").and_then(|o| o.as_bool()).unwrap_or(false))
            .unwrap_or(false);
        // 데몬이 응답하지 않으면 로컬 동기 인덱싱을 스킵 — Stop 훅이 블로킹(최대 86s)하지 않도록.
        // 다음 Stop에서 dirty/exported 신호가 남아있지 않으므로, 데몬이 살아나면 자연 복구.
        let elapsed = t0.elapsed().as_secs_f32();
        let ts = Local::now().format("%Y-%m-%dT%H:%M:%S");
        let reason = reasons.join("+");
        append_sync(&format!(
            "{} {} kmd queued in {:.1}s (ok={})",
            ts, reason, elapsed, queued_ok
        ));
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
