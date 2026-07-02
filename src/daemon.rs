//! 데몬 모드 — unix socket 서버에 warm 상태(tantivy reader + 임베딩 모델)를 상주.
//!
//! 프로토콜: JSON Lines. 요청 한 줄 → 응답 한 줄.
//!   {"cmd":"rag","prompt":"...","session_id":"..."}   → {"context":"...","hits":[...],...}
//!   {"cmd":"search","query":"...","limit":10}          → {"hits":[...]}
//!   {"cmd":"update"}                                   → {"ok":true,...}  (백그라운드 큐)
//!   {"cmd":"ping"}                                     → {"ok":true}
//!
//! CLI는 소켓이 살아 있으면 데몬 경유, 아니면 인프로세스 폴백한다.

use crate::rag;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

pub fn socket_path() -> PathBuf {
    if let Ok(p) = std::env::var("KMD_SOCKET") {
        return PathBuf::from(p);
    }
    rag::state_dir().join("kmd.sock")
}

#[derive(Deserialize)]
struct Request {
    cmd: String,
    #[serde(default)]
    prompt: String,
    #[serde(default)]
    session_id: String,
    #[serde(default)]
    query: String,
    #[serde(default = "default_limit")]
    limit: usize,
}

fn default_limit() -> usize {
    10
}

#[derive(Serialize)]
struct RagResponse<'a> {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    context: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gate_reason: Option<&'a str>,
    latency_ms: u64,
}

/// 데몬 서버 본체. foreground로 실행 (launchd가 관리).
pub fn serve() -> Result<()> {
    let sock = socket_path();
    if let Some(parent) = sock.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // 이전 소켓 정리 (연결 안 되면 stale)
    if sock.exists() {
        if UnixStream::connect(&sock).is_ok() {
            anyhow::bail!("daemon already running on {}", sock.display());
        }
        std::fs::remove_file(&sock)?;
    }

    let listener = UnixListener::bind(&sock)?;
    eprintln!("kmd daemon listening on {}", sock.display());

    // update 직렬화 (동시에 하나만)
    let updating = AtomicBool::new(false);
    let update_lock = Mutex::new(());

    // 시그널 정리: SIGTERM/SIGINT 시 소켓 제거는 launchd 재시작에 맡기고 단순화
    std::thread::scope(|scope| {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            scope.spawn(|| {
                let _ = handle_conn(stream, &updating, &update_lock);
            });
        }
    });
    Ok(())
}

fn handle_conn(
    stream: UnixStream,
    updating: &AtomicBool,
    update_lock: &Mutex<()>,
) -> Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut writer = stream;
    let mut line = String::new();

    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Ok(()); // EOF
        }
        let req: Request = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                let _ = writeln!(writer, r#"{{"ok":false,"error":"bad request: {}"}}"#, e);
                continue;
            }
        };

        match req.cmd.as_str() {
            "ping" => {
                writeln!(writer, r#"{{"ok":true}}"#)?;
            }
            "rag" => {
                // run_pipeline은 인덱스를 mmap reader로 매번 열지만 페이지 캐시로 충분히 빠름.
                // (임베딩 모델 warm은 vsearch 붙일 때 이 프로세스에 상주시킨다)
                match rag::run_pipeline(&req.prompt) {
                    Ok(o) => {
                        rag::log_outcome(&req.prompt, &req.session_id, &o);
                        let resp = RagResponse {
                            ok: true,
                            context: o.context.as_deref(),
                            gate_reason: o.gate_reason,
                            latency_ms: o.latency_ms,
                        };
                        writeln!(writer, "{}", serde_json::to_string(&resp)?)?;
                    }
                    Err(e) => {
                        writeln!(writer, r#"{{"ok":false,"error":{:?}}}"#, e.to_string())?;
                    }
                }
            }
            "search" => {
                let result = (|| -> Result<String> {
                    let cfg = crate::config::load()?;
                    let hits = crate::bm25::search(
                        &crate::config::tantivy_dir(),
                        &cfg,
                        &req.query,
                        req.limit,
                        None,
                    )?;
                    Ok(serde_json::json!({"ok": true, "hits": hits}).to_string())
                })();
                match result {
                    Ok(json) => writeln!(writer, "{}", json)?,
                    Err(e) => writeln!(writer, r#"{{"ok":false,"error":{:?}}}"#, e.to_string())?,
                }
            }
            "update" => {
                if updating.swap(true, Ordering::SeqCst) {
                    writeln!(writer, r#"{{"ok":true,"queued":false,"reason":"update already running"}}"#)?;
                } else {
                    // 응답 먼저 (훅이 기다리지 않게), 작업은 이 스레드에서 계속
                    writeln!(writer, r#"{{"ok":true,"queued":true}}"#)?;
                    let _guard = update_lock.lock().unwrap();
                    let r = (|| -> Result<()> {
                        let cfg = crate::config::load()?;
                        let mut store = crate::store::Store::open(&crate::config::store_path())?;
                        crate::scan::update(&cfg, &mut store, false)?;
                        crate::bm25::index_dirty(&crate::config::tantivy_dir(), &mut store)?;
                        Ok(())
                    })();
                    if let Err(e) = r {
                        eprintln!("daemon update failed: {}", e);
                    }
                    updating.store(false, Ordering::SeqCst);
                }
            }
            other => {
                writeln!(writer, r#"{{"ok":false,"error":"unknown cmd: {}"}}"#, other)?;
            }
        }
    }
}

// ---------------------------------------------------------------- client ----

/// 데몬에 요청 한 번 보내고 응답 받기. 데몬 없으면 None.
pub fn try_request(payload: &serde_json::Value) -> Option<serde_json::Value> {
    let sock = socket_path();
    let mut stream = UnixStream::connect(&sock).ok()?;
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(10)))
        .ok()?;
    writeln!(stream, "{}", payload).ok()?;
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).ok()?;
    serde_json::from_str(&line).ok()
}

/// launchd plist 생성 + 로드 안내.
pub fn install_launchd() -> Result<()> {
    let home = std::env::var("HOME")?;
    let bin = std::env::current_exe()?;
    let plist_path = PathBuf::from(&home).join("Library/LaunchAgents/ai.kmd.daemon.plist");
    let log_dir = rag::state_dir();
    std::fs::create_dir_all(&log_dir)?;
    let plist = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>ai.kmd.daemon</string>
    <key>ProgramArguments</key>
    <array>
        <string>{bin}</string>
        <string>daemon</string>
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>{log}/daemon.log</string>
    <key>StandardErrorPath</key><string>{log}/daemon.log</string>
</dict>
</plist>
"#,
        bin = bin.display(),
        log = log_dir.display(),
    );
    std::fs::write(&plist_path, plist)?;
    println!("wrote {}", plist_path.display());
    println!("load with:   launchctl load -w {}", plist_path.display());
    println!("unload with: launchctl unload {}", plist_path.display());
    Ok(())
}
