//! MCP 서버 (stdio) — kmd 검색을 Claude Code 외의 클라이언트에도 노출한다.
//!
//! Claude Code에서는 이미 `UserPromptSubmit` 훅(자동 주입)과 `kmd` 스킬
//! (`Bash(kmd:*)`)로 kmd에 닿는다. 이 서버가 존재하는 이유는 훅도 스킬도 쓸 수
//! 없는 클라이언트 — Claude Desktop, Cursor, Codex — 때문이다.
//!
//! Claude Code가 필요한 프로토콜 표면은 작다(initialize, tools/list,
//! tools/call). narwhal이 같은 이유로 손으로 쓴 JSON-RPC를 쓰고 있고, 여기도
//! 의존성을 추가하지 않는다.
//!
//! 검색은 가능하면 데몬을 경유한다. 데몬이 tantivy reader를 warm하게 들고 있어
//! 콜드 경로보다 빠르고, 이 프로세스는 클라이언트가 아무 때나 재시작한다.
//! 데몬이 없으면 인프로세스로 폴백한다 — CLI가 이미 쓰는 규약이다.

use crate::{bm25, config, global};
use anyhow::{Result, anyhow};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};

const PROTOCOL_VERSION: &str = "2025-06-18";
const SERVER_NAME: &str = "kmd";
const SERVER_VERSION: &str = "0.1.0";

/// 기본 히트 수. 축당 5건 × 3축이면 컨텍스트에 부담 없이 전체 그림이 나온다.
const DEFAULT_LIMIT: usize = 5;
/// 요청이 올릴 수 있는 상한. MCP 응답은 그대로 모델 컨텍스트에 들어가므로,
/// 클라이언트가 limit=1000을 보내 컨텍스트를 태우는 것을 막는다.
const MAX_LIMIT: usize = 50;

pub fn serve() -> Result<()> {
    let stdin = std::io::stdin();
    let mut out = std::io::stdout();
    let mut reader = BufReader::new(stdin.lock());
    let mut line = String::new();

    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Ok(()); // EOF — 클라이언트가 닫았다.
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let req: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                write_error(&mut out, Value::Null, -32700, &format!("parse error: {}", e))?;
                continue;
            }
        };
        dispatch(&mut out, &req)?;
    }
}

fn dispatch(out: &mut impl Write, req: &Value) -> Result<()> {
    let method = req.get("method").and_then(Value::as_str).unwrap_or("");
    let id = req.get("id").cloned().unwrap_or(Value::Null);

    match method {
        "initialize" => write_result(
            out,
            id,
            json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            }),
        ),
        "notifications/initialized" => Ok(()), // 알림 — 응답하지 않는다.
        "tools/list" => write_result(out, id, json!({"tools": tool_definitions()})),
        "tools/call" => handle_tool_call(out, req, id),
        "ping" => write_result(out, id, json!({})),
        _ => {
            // id 없는 요청은 알림이므로 응답하면 프로토콜 위반이다.
            if req.get("id").is_none() {
                return Ok(());
            }
            write_error(out, id, -32601, &format!("method not found: {}", method))
        }
    }
}

fn handle_tool_call(out: &mut impl Write, req: &Value, id: Value) -> Result<()> {
    let params = req.get("params").cloned().unwrap_or(Value::Null);
    let name = params.get("name").and_then(Value::as_str).unwrap_or("");
    let args = params.get("arguments").cloned().unwrap_or(json!({}));

    match call_tool(name, &args) {
        Ok(text) => write_result(out, id, json!({"content": [{"type": "text", "text": text}]})),
        // 도구 오류는 in-band로 돌려준다. 요청 자체를 실패시키면 모델이 이유를
        // 못 보고 같은 호출을 반복한다.
        Err(e) => write_result(
            out,
            id,
            json!({
                "content": [{"type": "text", "text": format!("error: {}", e)}],
                "isError": true,
            }),
        ),
    }
}

fn call_tool(name: &str, args: &Value) -> Result<String> {
    match name {
        "kmd_search" => tool_search(args),
        "kmd_status" => tool_status(),
        _ => Err(anyhow!("unknown tool: {}", name)),
    }
}

// ------------------------------------------------------------------ tools ----

fn tool_definitions() -> Value {
    json!([
        {
            "name": "kmd_search",
            "description": concat!(
                "Search local knowledge: curated notes, past Claude Code sessions, ",
                "and source code across all local git repositories. Korean queries ",
                "work — the index uses ko-dic morphological analysis, not character ",
                "n-grams.\n\n",
                "Returns hits grouped by axis, each with a readable absolute path. ",
                "Read that path to get the full document; the snippet is only enough ",
                "to judge relevance.\n\n",
                "Search all three axes when you don't know where the answer lives. ",
                "Narrow with `axis` when you do: 'project' for code, 'session' for ",
                "what was done before, 'knowledge' for curated notes."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search terms. Korean or English."
                    },
                    "axis": {
                        "type": "string",
                        "enum": [global::SESSION, global::KNOWLEDGE, global::PROJECT],
                        "description": concat!(
                            "Restrict to one axis. Omit to search all three, which ",
                            "returns a quota from each so a large axis cannot crowd ",
                            "out the others."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Hits per axis. Default 5, max 50.",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "kmd_status",
            "description": concat!(
                "What the index actually contains: each collection's document count, ",
                "axis, and source path. Use it when a search returns nothing to tell ",
                "an empty index from a query that simply missed."
            ),
            "inputSchema": {"type": "object", "properties": {}},
        },
    ])
}

fn tool_search(args: &Value) -> Result<String> {
    let query = args
        .get("query")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|q| !q.is_empty())
        .ok_or_else(|| anyhow!("query is required"))?;
    let axis = args.get("axis").and_then(Value::as_str);
    let limit = args
        .get("limit")
        .and_then(Value::as_u64)
        .map(|n| (n as usize).clamp(1, MAX_LIMIT))
        .unwrap_or(DEFAULT_LIMIT);

    if let Some(a) = axis
        && !global::AXES.contains(&a)
    {
        return Err(anyhow!(
            "unknown axis {:?} — expected one of {}",
            a,
            global::AXES.join(", ")
        ));
    }

    let results = search_axes(query, limit, axis)?;
    Ok(render_hits(&results))
}

/// 축별 검색 — 데몬 우선, 없으면 인프로세스.
fn search_axes(query: &str, limit: usize, axis: Option<&str>) -> Result<Vec<global::AxisResults>> {
    if let Some(results) = daemon_global(query, limit, axis) {
        return Ok(results);
    }
    let cfg = config::load()?;
    global::search(&config::tantivy_dir(), &cfg, query, limit, axis)
}

/// 데몬의 `global` 커맨드를 시도한다. 데몬이 없거나 그 커맨드를 모르는 구버전이면
/// None을 돌려 호출자가 인프로세스로 폴백하게 한다.
fn daemon_global(query: &str, limit: usize, axis: Option<&str>) -> Option<Vec<global::AxisResults>> {
    let mut payload = json!({"cmd": "global", "query": query, "limit": limit});
    if let Some(a) = axis {
        payload["axis"] = json!(a);
    }
    let resp = crate::daemon::try_request(&payload)?;
    if resp.get("ok").and_then(Value::as_bool) != Some(true) {
        return None;
    }
    serde_json::from_value(resp.get("axes")?.clone()).ok()
}

fn tool_status() -> Result<String> {
    let cfg = config::load()?;
    let store = crate::store::Store::open(&config::store_path())?;
    let (total, dirty) = store.counts()?;
    let counts = store.collection_counts()?;

    let mut out = format!("{} documents indexed ({} pending reindex)\n", total, dirty);
    for (name, coll) in &cfg.collections {
        let n = counts
            .iter()
            .find(|(c, _)| c == name)
            .map(|(_, n)| *n)
            .unwrap_or(0);
        out.push_str(&format!(
            "  {:<16} {:>7} docs  [{}]  {}\n",
            name,
            n,
            global::axis_of(name, &cfg),
            coll.path.display()
        ));
    }
    Ok(out)
}

/// 히트를 텍스트로 렌더한다.
///
/// JSON을 그대로 돌려주지 않는 이유: MCP 응답은 모델 컨텍스트에 그대로 들어간다.
/// JSON 구두점은 토큰을 쓰면서 모델이 읽는 데 보태는 게 없다. 대신 각 히트가
/// 읽을 수 있는 절대경로를 갖게 해서, 후속 동작(Read)에 필요한 것을 준다.
fn render_hits(results: &[global::AxisResults]) -> String {
    let empty = results.iter().all(|r| r.hits.is_empty());
    if empty {
        return "no hits — try fewer or different terms, or kmd_status to see what \
                is indexed"
            .to_string();
    }

    let mut out = String::new();
    for axis in results {
        if axis.hits.is_empty() {
            continue;
        }
        out.push_str(&format!("## {}\n", axis.axis));
        for hit in &axis.hits {
            out.push_str(&render_hit(hit));
        }
        out.push('\n');
    }
    out.trim_end().to_string()
}

fn render_hit(hit: &bm25::SearchHit) -> String {
    // 경로는 abspath를 쓴다 — 클라이언트가 바로 열 수 있는 형태다. 없으면
    // (컬렉션이 설정에서 빠진 경우) kmd URI로 폴백하되 열 수 없음을 명시한다.
    let mut s = match &hit.abspath {
        Some(p) => format!("{}\n", p),
        None => format!("{} (path unresolved — collection not in index.yml)\n", hit.file),
    };
    if !hit.title.is_empty() {
        s.push_str(&format!("  {}\n", one_line(&hit.title, 100)));
    }
    if let Some(snippet) = &hit.snippet {
        s.push_str(&format!("  {}\n", one_line(snippet, 200)));
    }
    s
}

/// 개행을 접고 길이를 자른다. 원문 개행이 그대로 나가면 히트 경계가 사라진다.
fn one_line(s: &str, max: usize) -> String {
    let folded: String = s
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if folded.chars().count() <= max {
        return folded;
    }
    let cut: String = folded.chars().take(max).collect();
    format!("{}…", cut.trim_end())
}

// ------------------------------------------------------------------- wire ----

fn write_result(out: &mut impl Write, id: Value, result: Value) -> Result<()> {
    write_message(out, json!({"jsonrpc": "2.0", "id": id, "result": result}))
}

fn write_error(out: &mut impl Write, id: Value, code: i32, message: &str) -> Result<()> {
    write_message(
        out,
        json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}),
    )
}

fn write_message(out: &mut impl Write, msg: Value) -> Result<()> {
    writeln!(out, "{}", msg)?;
    // stdout은 파이프이므로 라인 버퍼가 아니다. flush하지 않으면 클라이언트가
    // 응답을 기다리며 멈춘다.
    out.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(method: &str, params: Value) -> Value {
        let req = json!({"jsonrpc": "2.0", "id": 1, "method": method, "params": params});
        let mut buf: Vec<u8> = Vec::new();
        dispatch(&mut buf, &req).expect("dispatch");
        let text = String::from_utf8(buf).expect("utf8");
        if text.trim().is_empty() {
            return Value::Null;
        }
        serde_json::from_str(text.trim()).expect("json response")
    }

    /// 도구 오류는 JSON-RPC 오류가 아니라 isError 결과로 나가야 한다.
    ///
    /// 이 구분이 중요한 이유: JSON-RPC 오류는 클라이언트가 삼키고 모델에게는
    /// 도구가 조용히 실패한 것으로 보인다. 그러면 모델이 이유를 모른 채 같은
    /// 호출을 반복한다. in-band 오류는 모델이 읽고 고칠 수 있다.
    #[test]
    fn tool_errors_are_reported_in_band() {
        let resp = call("tools/call", json!({"name": "kmd_search", "arguments": {}}));
        assert!(
            resp.get("error").is_none(),
            "a missing argument is not a protocol error: {}",
            resp
        );
        let result = &resp["result"];
        assert_eq!(result["isError"], json!(true));
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(
            text.contains("query is required"),
            "the message says what was wrong: {}",
            text
        );
    }

    #[test]
    fn unknown_tool_is_an_in_band_error() {
        let resp = call("tools/call", json!({"name": "kmd_nope", "arguments": {}}));
        assert_eq!(resp["result"]["isError"], json!(true));
    }

    /// 잘못된 축은 유효한 값을 알려준다. 열거형을 스키마에만 두면 클라이언트가
    /// 검증하지 않을 때 조용히 빈 결과가 나온다.
    #[test]
    fn unknown_axis_names_the_valid_ones() {
        let resp = call(
            "tools/call",
            json!({"name": "kmd_search", "arguments": {"query": "x", "axis": "sessions"}}),
        );
        let text = resp["result"]["content"][0]["text"].as_str().unwrap();
        assert_eq!(resp["result"]["isError"], json!(true));
        assert!(text.contains("session"), "lists the real axes: {}", text);
    }

    /// 알림(id 없음)에는 응답하지 않는다 — 응답하면 프로토콜 위반이다.
    #[test]
    fn notifications_get_no_response() {
        let req = json!({"jsonrpc": "2.0", "method": "notifications/initialized"});
        let mut buf: Vec<u8> = Vec::new();
        dispatch(&mut buf, &req).expect("dispatch");
        assert!(buf.is_empty(), "wrote {:?}", String::from_utf8_lossy(&buf));

        let req = json!({"jsonrpc": "2.0", "method": "some/unknown/notification"});
        let mut buf: Vec<u8> = Vec::new();
        dispatch(&mut buf, &req).expect("dispatch");
        assert!(buf.is_empty(), "wrote {:?}", String::from_utf8_lossy(&buf));
    }

    #[test]
    fn initialize_advertises_tools() {
        let resp = call("initialize", json!({}));
        assert_eq!(resp["result"]["protocolVersion"], json!(PROTOCOL_VERSION));
        assert!(resp["result"]["capabilities"]["tools"].is_object());
    }

    #[test]
    fn tools_list_declares_every_dispatchable_tool() {
        let resp = call("tools/list", json!({}));
        let tools = resp["result"]["tools"].as_array().expect("array");
        let names: Vec<&str> = tools
            .iter()
            .map(|t| t["name"].as_str().expect("name"))
            .collect();
        // 선언과 디스패치가 갈라지면 모델이 있다고 들은 도구를 호출했을 때
        // "unknown tool"이 돌아온다. 양방향으로 확인한다.
        assert!(names.contains(&"kmd_search"));
        assert!(names.contains(&"kmd_status"));
        for name in &names {
            let resp = call("tools/call", json!({"name": name, "arguments": {}}));
            let text = resp["result"]["content"][0]["text"]
                .as_str()
                .unwrap_or_default();
            assert!(
                !text.contains("unknown tool"),
                "{} is listed but not dispatchable",
                name
            );
        }
    }

    /// limit은 상한이 걸려야 한다 — MCP 응답은 모델 컨텍스트에 그대로 들어간다.
    #[test]
    fn limit_is_clamped() {
        let args = json!({"query": "x", "limit": 100_000});
        let limit = args
            .get("limit")
            .and_then(Value::as_u64)
            .map(|n| (n as usize).clamp(1, MAX_LIMIT))
            .unwrap_or(DEFAULT_LIMIT);
        assert_eq!(limit, MAX_LIMIT);
    }

    #[test]
    fn one_line_folds_newlines_and_truncates() {
        assert_eq!(one_line("a\nb  c", 100), "a b c");
        let long = "x".repeat(300);
        let cut = one_line(&long, 10);
        assert_eq!(cut.chars().count(), 11, "10 chars plus the ellipsis");
        assert!(cut.ends_with('…'));
    }

    /// 히트가 경로를 못 만들면 그 사실을 말한다. 열 수 없는 kmd:// URI를
    /// 경로처럼 내보내면 클라이언트가 Read에 실패하는 이유를 알 수 없다.
    #[test]
    fn unresolved_path_says_so() {
        let hit = bm25::SearchHit {
            docid: "#0".into(),
            score: 1.0,
            file: "kmd://gone/x.md".into(),
            abspath: None,
            title: String::new(),
            context: None,
            snippet: None,
        };
        let rendered = render_hit(&hit);
        assert!(rendered.contains("path unresolved"), "{}", rendered);
    }

    #[test]
    fn empty_results_suggest_a_next_step() {
        let out = render_hits(&[global::AxisResults {
            axis: "session".into(),
            hits: vec![],
        }]);
        assert!(out.contains("no hits"), "{}", out);
        assert!(out.contains("kmd_status"), "points at the diagnostic: {}", out);
    }
}
