mod bm25;
mod config;
mod daemon;
#[cfg(feature = "embed")]
mod embed;
mod output;
mod rag;
mod scan;
mod sim;
mod stats;
mod store;
mod tokenize;

use anyhow::Result;
use clap::{Parser, Subcommand};

/// Korean-aware markdown search — qmd-compatible CLI.
#[derive(Parser)]
#[command(name = "kmd", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Scan collections from index.yml and (re)index changed files
    Update {
        /// Force reindex of all files
        #[arg(long)]
        force: bool,
        /// Queue update on the daemon and return immediately (falls back to local)
        #[arg(long = "async")]
        r#async: bool,
    },
    /// Keyword search (Korean morphological BM25)
    Search {
        query: String,
        /// Max results
        #[arg(short = 'n', long, default_value_t = 10)]
        limit: usize,
        /// Restrict to a collection
        #[arg(short, long)]
        collection: Option<String>,
        /// JSON output (qmd-compatible schema)
        #[arg(long)]
        json: bool,
    },
    /// RAG pipeline: read UserPromptSubmit JSON on stdin, print injected context
    Rag {
        /// Hook mode (stdin JSON, logs to rag.jsonl, never fails)
        #[arg(long)]
        hook: bool,
        /// Run pipeline for a prompt given as argument (debugging)
        prompt: Option<String>,
    },
    /// Aggregated stats from rag.jsonl (injection rate, gates, latency)
    Stats {
        /// Only include entries from the last N hours
        #[arg(long)]
        hours: Option<u64>,
    },
    /// Recent rag.jsonl entries
    Log {
        /// Number of entries
        #[arg(short = 'n', long, default_value_t = 20)]
        count: usize,
        /// Follow (tail -f)
        #[arg(short, long)]
        follow: bool,
    },
    /// Interactive RAG simulator (TUI), or batch replay of prompt history
    Sim {
        /// Replay prompts from ~/.claude/history.jsonl instead of TUI
        #[arg(long)]
        replay: bool,
        /// Number of history prompts to replay (most recent first)
        #[arg(short = 'n', long, default_value_t = 100)]
        count: usize,
        /// Replay only prompts containing Hangul
        #[arg(long)]
        hangul: bool,
    },
    /// Generate embeddings for documents without them (requires 'embed' feature)
    Embed {
        /// Max documents to embed this run
        #[arg(short = 'n', long)]
        limit: Option<usize>,
    },
    /// Semantic vector search (requires 'embed' feature)
    Vsearch {
        query: String,
        #[arg(short = 'n', long, default_value_t = 10)]
        limit: usize,
        #[arg(long)]
        json: bool,
    },
    /// Hybrid BM25 + vector search with RRF fusion (requires 'embed' feature)
    Query {
        query: String,
        #[arg(short = 'n', long, default_value_t = 10)]
        limit: usize,
        #[arg(long)]
        json: bool,
    },
    /// Run the warm daemon (unix socket server)
    Daemon {
        /// Write launchd plist and print load instructions
        #[arg(long)]
        install: bool,
    },
    /// Index/collection status
    Status,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Update { force, r#async } => cmd_update(force, r#async),
        Command::Search {
            query,
            limit,
            collection,
            json,
        } => cmd_search(&query, limit, collection.as_deref(), json),
        Command::Rag { hook, prompt } => cmd_rag(hook, prompt.as_deref()),
        Command::Stats { hours } => stats::print_stats(hours.map(|h| h * 3600)),
        Command::Log { count, follow } => stats::print_log(count, follow),
        Command::Sim { replay, count, hangul } => {
            if replay {
                sim::replay(count, hangul)
            } else {
                sim::tui()
            }
        }
        Command::Embed { limit } => cmd_embed(limit),
        Command::Vsearch { query, limit, json } => cmd_vsearch(&query, limit, json),
        Command::Query { query, limit, json } => cmd_query(&query, limit, json),
        Command::Daemon { install } => {
            if install {
                daemon::install_launchd()
            } else {
                daemon::serve()
            }
        }
        Command::Status => cmd_status(),
    }
}

fn cmd_update(force: bool, r#async: bool) -> Result<()> {
    if r#async && !force {
        // 데몬에 위임 — 즉시 반환 (Stop 훅용 논블로킹 경로)
        if let Some(resp) = daemon::try_request(&serde_json::json!({"cmd": "update"})) {
            if resp.get("ok").and_then(|v| v.as_bool()) == Some(true) {
                eprintln!("update queued on daemon");
                return Ok(());
            }
        }
        eprintln!("daemon unavailable — running update locally");
    }
    let cfg = config::load()?;
    let mut store = store::Store::open(&config::store_path())?;
    let stats = scan::update(&cfg, &mut store, force)?;
    // 내용이 바뀐 문서는 임베딩도 무효 — tantivy 반영 전에 dirty 목록으로 제거
    #[cfg(feature = "embed")]
    {
        let dirty_ids: Vec<i64> = store.dirty_docs()?.iter().map(|d| d.id).collect();
        embed::purge_stale(&store, &dirty_ids)?;
    }
    let indexed = bm25::index_dirty(&config::tantivy_dir(), &mut store)?;
    eprintln!(
        "scanned {} files ({} added, {} updated, {} removed), bm25 indexed {}",
        stats.seen, stats.added, stats.updated, stats.removed, indexed
    );
    Ok(())
}

fn cmd_search(query: &str, limit: usize, collection: Option<&str>, json: bool) -> Result<()> {
    let cfg = config::load()?;
    let results = bm25::search(&config::tantivy_dir(), &cfg, query, limit, collection)?;
    if json {
        println!("{}", output::to_json(&results)?);
    } else {
        output::print_cli(&results);
    }
    Ok(())
}

fn cmd_rag(hook: bool, prompt: Option<&str>) -> Result<()> {
    if hook {
        return rag::run_hook();
    }
    let Some(p) = prompt else {
        eprintln!("usage: kmd rag --hook   (stdin JSON)  or  kmd rag \"<prompt>\"");
        std::process::exit(2);
    };
    let outcome = rag::run_pipeline(p)?;
    match outcome.gate_reason {
        Some(reason) => println!("GATED: {}", reason),
        None => {
            println!("query: {}", outcome.query.as_deref().unwrap_or(""));
            println!("latency: {}ms", outcome.latency_ms);
            println!("hits: {}", outcome.hits.len());
            for h in &outcome.hits {
                println!("  {:.1} {}", h.score, h.file);
            }
            if let Some(ctx) = &outcome.context {
                println!("\n{}", ctx);
            }
        }
    }
    Ok(())
}

fn cmd_status() -> Result<()> {
    let cfg = config::load()?;
    let store = store::Store::open(&config::store_path())?;
    output::print_status(&cfg, &store)?;
    Ok(())
}

#[cfg(feature = "embed")]
fn cmd_embed(limit: Option<usize>) -> Result<()> {
    let mut store = store::Store::open(&config::store_path())?;
    let n = embed::embed_pending(&mut store, limit)?;
    let (vectors, pending) = embed::embedding_counts(&store)?;
    eprintln!("embedded {} docs — {} chunks total, {} docs pending", n, vectors, pending);
    Ok(())
}

#[cfg(feature = "embed")]
fn cmd_vsearch(query: &str, limit: usize, json: bool) -> Result<()> {
    let store = store::Store::open(&config::store_path())?;
    let results = embed::vsearch(&store, query, limit)?;
    if json {
        println!("{}", output::to_json(&results)?);
    } else {
        output::print_cli(&results);
    }
    Ok(())
}

#[cfg(feature = "embed")]
fn cmd_query(query: &str, limit: usize, json: bool) -> Result<()> {
    let cfg = config::load()?;
    let store = store::Store::open(&config::store_path())?;
    let results = embed::hybrid(&store, &config::tantivy_dir(), &cfg, query, limit)?;
    if json {
        println!("{}", output::to_json(&results)?);
    } else {
        output::print_cli(&results);
    }
    Ok(())
}

#[cfg(not(feature = "embed"))]
fn cmd_embed(_limit: Option<usize>) -> Result<()> {
    anyhow::bail!("built without 'embed' feature — rebuild with: cargo build --release --features embed")
}

#[cfg(not(feature = "embed"))]
fn cmd_vsearch(_query: &str, _limit: usize, _json: bool) -> Result<()> {
    anyhow::bail!("built without 'embed' feature — rebuild with: cargo build --release --features embed")
}

#[cfg(not(feature = "embed"))]
fn cmd_query(_query: &str, _limit: usize, _json: bool) -> Result<()> {
    anyhow::bail!("built without 'embed' feature — rebuild with: cargo build --release --features embed")
}
