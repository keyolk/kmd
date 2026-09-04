//! 데몬 모드 — unix socket 서버에 warm 상태(tantivy reader + 임베딩 모델)를 상주.
//!
//! 프로토콜: JSON Lines. 요청 한 줄 → 응답 한 줄.
//!   {"cmd":"rag","prompt":"...","session_id":"..."}   → {"context":"...","hits":[...],...}
//!   {"cmd":"search","query":"...","limit":10}          → {"hits":[...]}
//!   {"cmd":"global","query":"...","limit":5,"axis":"…"} → {"axes":[{axis,hits}]}
//!   {"cmd":"vsearch"|"hybrid","query":"...","limit":5}  → {"hits":[...]}  (embed 피처)
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
    /// `global` 커맨드에서 한 축만 질의할 때. 비면 세 축 전부.
    #[serde(default)]
    axis: Option<String>,
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

    // 임베딩 모델을 상주시킨다.
    //
    // 벡터 검색 지연은 모델 로드(웜 220ms, 콜드 420ms)와 브루트포스 스캔으로
    // 갈리는데, 프로세스가 살아 있는 동안 모델 로드는 한 번만 내면 되는
    // 비용이다. 스캔은 이렇게 없앨 수 없다 — 청크 수에 선형이므로 인덱스
    // 구조를 바꿔야 한다.
    //
    // 로드 실패는 치명적이지 않다: 벡터 커맨드만 못 쓰고 rag/search는 그대로
    // 동작해야 한다. 모델 파일이 없는 설치가 정상 케이스다.
    #[cfg(feature = "embed")]
    let embedder = match crate::embed::Embedder::load() {
        Ok(e) => {
            eprintln!("kmd daemon: embedding model resident");
            Some(e)
        }
        Err(e) => {
            eprintln!("kmd daemon: no embedding model ({e}) — vector commands unavailable");
            None
        }
    };
    #[cfg(not(feature = "embed"))]
    let embedder: Option<()> = None;

    // 시그널 정리: SIGTERM/SIGINT 시 소켓 제거는 launchd 재시작에 맡기고 단순화
    // 세 참조를 루프 밖에서 만들어 클로저가 Copy로 가져가게 한다. `move`가
    // 값을 끌어가면 두 번째 연결에서 컴파일이 깨진다.
    let updating = &updating;
    let update_lock = &update_lock;
    let embedder = &embedder;
    std::thread::scope(|scope| {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            scope.spawn(move || {
                let _ = handle_conn(stream, updating, update_lock, embedder);
            });
        }
    });
    Ok(())
}

#[cfg(feature = "embed")]
type WarmEmbedder = Option<crate::embed::Embedder>;
#[cfg(not(feature = "embed"))]
type WarmEmbedder = Option<()>;

fn handle_conn(
    stream: UnixStream,
    updating: &AtomicBool,
    update_lock: &Mutex<()>,
    embedder: &WarmEmbedder,
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
                // 상주 임베더를 넘긴다 — 있으면 hybrid, 없으면 BM25로 내려간다.
                match rag::run_pipeline_with(&req.prompt, embedder.as_ref()) {
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
            "global" => {
                // MCP 서버가 쓰는 경로. tantivy reader가 이 프로세스에 warm하게
                // 있으므로, 클라이언트가 매번 새로 띄우는 MCP 프로세스보다
                // 여기서 검색하는 편이 빠르다.
                let result = (|| -> Result<String> {
                    let cfg = crate::config::load()?;
                    let axes = crate::global::search(
                        &crate::config::tantivy_dir(),
                        &cfg,
                        &req.query,
                        req.limit,
                        req.axis.as_deref(),
                    )?;
                    Ok(serde_json::json!({"ok": true, "axes": axes}).to_string())
                })();
                match result {
                    Ok(json) => writeln!(writer, "{}", json)?,
                    Err(e) => writeln!(writer, r#"{{"ok":false,"error":{:?}}}"#, e.to_string())?,
                }
            }
            "vsearch" | "hybrid" => {
                let result = daemon_vector_search(&req, embedder);
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
                        #[cfg(feature = "embed")]
                        {
                            let dirty_ids: Vec<i64> =
                                store.dirty_docs()?.iter().map(|d| d.id).collect();
                            crate::embed::purge_stale(&store, &dirty_ids)?;
                        }
                        crate::bm25::index_dirty(&crate::config::tantivy_dir(), &mut store)?;
                        // 신규/변경 문서 임베딩 — 한 사이클에 200개 캡 (백그라운드 시간 바운드)
                        #[cfg(feature = "embed")]
                        {
                            if let Err(e) = crate::embed::embed_pending(&mut store, Some(200), &[])
                            {
                                eprintln!("daemon embed failed (non-fatal): {}", e);
                            }
                        }
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

/// 상주 임베더로 벡터/hybrid 검색. `axis`가 주어지면 그 축의 컬렉션으로 좁힌다.
#[cfg(feature = "embed")]
fn daemon_vector_search(req: &Request, embedder: &WarmEmbedder) -> Result<String> {
    let Some(embedder) = embedder.as_ref() else {
        anyhow::bail!("no embedding model loaded — run `kmd embed` and restart the daemon")
    };
    let cfg = crate::config::load()?;
    let store = crate::store::Store::open(&crate::config::store_path())?;
    // 축 필터는 global과 같은 규약을 쓴다. 축을 안 주면 전 임베딩을 훑는다.
    let colls: Vec<&str> = match req.axis.as_deref() {
        Some(a) => crate::global::collections_for(a, &cfg),
        None => Vec::new(),
    };
    let filter = (!colls.is_empty()).then_some(colls.as_slice());
    let hits = if req.cmd == "hybrid" {
        crate::embed::hybrid_with(
            &store,
            &crate::config::tantivy_dir(),
            &cfg,
            &req.query,
            req.limit,
            filter,
            Some(embedder),
        )?
    } else {
        crate::embed::vsearch_with(&store, &req.query, req.limit, filter, Some(embedder))?
    };
    Ok(serde_json::json!({"ok": true, "hits": hits}).to_string())
}

#[cfg(not(feature = "embed"))]
fn daemon_vector_search(_req: &Request, _embedder: &WarmEmbedder) -> Result<String> {
    anyhow::bail!("daemon built without the 'embed' feature")
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
