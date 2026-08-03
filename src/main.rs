mod ab;
mod activity;
mod bm25;
mod config;
mod daemon;
mod dashboard;
mod dashboard_simulator;
#[cfg(feature = "embed")]
mod embed;
mod eval;
mod evaluation_log;
mod hook;
mod hook_config;
mod journal;
mod learnings;
mod locality;
mod output;
mod pageindex;
mod rag;
mod scan;
mod sim;
mod stats;
mod store;
mod tokenize;
mod util;

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
    /// Operations dashboard for runtime, sessions, RAG, evaluations, and checks
    Dashboard {
        /// Print the current dashboard snapshot as JSON instead of opening the TUI
        #[arg(long)]
        json: bool,
        /// Run self-checks, persist the result, and exit
        #[arg(long)]
        check: bool,
    },
    /// Retrieval quality eval (L1): known-item self-supervised or gold-labeled
    Eval {
        /// Path to a gold YAML (prompt/expect_any). Omit for known-item mode.
        #[arg(long)]
        gold: Option<std::path::PathBuf>,
        /// known-item: collections to sample from (repeatable). Default: knowledge collections.
        #[arg(long)]
        collection: Vec<String>,
        /// known-item: number of docs to sample (0 = all). Deterministic.
        #[arg(long, default_value_t = 200)]
        sample: usize,
        /// Cutoff k for Recall@k / MRR
        #[arg(short = 'k', long, default_value_t = 5)]
        k: usize,
        /// Also run each query against qmd for side-by-side comparison
        #[arg(long)]
        compare_qmd: bool,
        /// known-item: only evaluate Korean-derived queries
        #[arg(long)]
        hangul: bool,
        /// JSON output
        #[arg(long)]
        json: bool,
    },
    /// Injection utilization (L2): did answers actually use injected context?
    Util {
        /// Only include entries from the last N hours
        #[arg(long)]
        hours: Option<u64>,
        /// JSON output
        #[arg(long)]
        json: bool,
    },
    /// Show what real sessions actually got augmented (prompt → files → usage)
    Show {
        /// Filter to a session id prefix
        session: Option<String>,
        /// Max prompts to show
        #[arg(short = 'n', long, default_value_t = 20)]
        limit: usize,
        /// Include synthetic sessions and auto prompts (task-notification, etc.)
        #[arg(long)]
        all: bool,
        /// JSON output
        #[arg(long)]
        json: bool,
    },
    /// A/B blind comparison (L3): kmd vs qmd context, proxy or external judge
    Ab {
        /// Prompts file (one per line) for building A/B pairs
        #[arg(long)]
        prompts: Option<std::path::PathBuf>,
        /// Emit blind pairs to this file (+ .key.jsonl) for external judging
        #[arg(long)]
        emit: Option<std::path::PathBuf>,
        /// Tally external verdicts.jsonl ({id,winner}); needs --prompts <pairs.jsonl>
        #[arg(long)]
        judge: Option<std::path::PathBuf>,
        /// JSON output
        #[arg(long)]
        json: bool,
    },
    /// Extract historical learnings from Claude Code transcripts or snapshots
    LearningsExtract {
        /// Process specific session ID
        #[arg(long)]
        session: Option<String>,
        /// Process last N days
        #[arg(long, default_value_t = 0)]
        recent: u32,
        /// Preview without writing
        #[arg(long)]
        dry_run: bool,
        /// Re-process already processed sessions
        #[arg(long)]
        force: bool,
    },
    /// Recent cross-session activity collected from live transcripts
    Activity {
        /// Max activity cards to show
        #[arg(short = 'n', long, default_value_t = 10)]
        limit: usize,
        /// JSON output
        #[arg(long)]
        json: bool,
    },
    /// Date-oriented learnings and sessions, ranked by cwd/repo/worktree locality
    Journal {
        /// Number of calendar days to show
        #[arg(long, default_value_t = 7)]
        days: u32,
        /// Show one date (YYYY-MM-DD or YYYYMMDD)
        #[arg(long)]
        date: Option<String>,
        /// Filter by canonical repo, project, or worktree name
        #[arg(short, long)]
        project: Option<String>,
        /// Spatial anchor; defaults to the current directory
        #[arg(long)]
        cwd: Option<String>,
        /// JSON output
        #[arg(long)]
        json: bool,
    },
    /// Session page index (L0/L1) — browsable axes instead of pushed RAG context
    Page {
        /// Axis (repo name) to list sessions for. Omit for the L0 index.
        axis: Option<String>,
        /// Max sessions to list in L1
        #[arg(short = 'n', long, default_value_t = 15)]
        limit: usize,
        /// Print the L0 block exactly as a hook would inject it (no stderr note)
        #[arg(long)]
        l0: bool,
        /// Measure L0 coverage of past RAG injections (reads rag.jsonl)
        #[arg(long)]
        cover: bool,
        /// SessionStart hook mode — emit additionalContext JSON
        #[arg(long)]
        hook: bool,
        /// JSON output
        #[arg(long)]
        json: bool,
    },
    /// Install, inspect, and run Claude Code hooks
    Hook {
        #[command(subcommand)]
        sub: HookSub,
    },
}

#[derive(Subcommand)]
enum HookSub {
    /// Install kmd hooks into ~/.claude/settings.json
    Install {
        /// kmd executable path; defaults to the currently running binary
        #[arg(long)]
        binary: Option<std::path::PathBuf>,
    },
    /// Check whether all kmd Claude Code hooks are installed
    Status,
    /// Remove only kmd-owned hooks from ~/.claude/settings.json
    Uninstall,
    /// Stop hook: refresh live activity and flush dirty collections
    Stop,
    /// SessionEnd hook: persist the final transcript and queue reindex
    SessionEnd,
    /// PostToolUse hook: mark index dirty on watched file edits
    MarkDirty,
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
        Command::Sim {
            replay,
            count,
            hangul,
        } => {
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
        Command::Dashboard { json, check } => dashboard::run(json, check),
        Command::Eval {
            gold,
            collection,
            sample,
            k,
            compare_qmd,
            hangul,
            json,
        } => {
            if let Some(path) = gold {
                eval::gold(&path, k, compare_qmd, json)
            } else {
                let colls = if collection.is_empty() {
                    rag::CLAUDE_COLLECTIONS
                        .iter()
                        .map(|s| s.to_string())
                        .collect()
                } else {
                    collection
                };
                eval::known_item(colls, sample, k, compare_qmd, hangul, json)
            }
        }
        Command::Util { hours, json } => util::print_utilization(hours.map(|h| h * 3600), json),
        Command::Show {
            session,
            limit,
            all,
            json,
        } => util::show(session.as_deref(), limit, all, json),
        Command::Ab {
            prompts,
            emit,
            judge,
            json,
        } => ab::run(prompts.as_deref(), emit.as_deref(), judge.as_deref(), json),
        Command::LearningsExtract {
            session,
            recent,
            dry_run,
            force,
        } => learnings::run(session.as_deref(), recent, dry_run, force),
        Command::Activity { limit, json } => activity::run(limit, json),
        Command::Journal {
            days,
            date,
            project,
            cwd,
            json,
        } => journal::run(
            days,
            date.as_deref(),
            project.as_deref(),
            cwd.as_deref(),
            json,
        ),
        Command::Page {
            axis,
            limit,
            l0,
            cover,
            hook,
            json,
        } => {
            if hook {
                pageindex::run_hook()
            } else {
                pageindex::run(axis.as_deref(), limit, json, l0, cover)
            }
        }
        Command::Hook { sub } => match sub {
            HookSub::Install { binary } => hook_config::install(binary.as_deref()),
            HookSub::Status => hook_config::status(),
            HookSub::Uninstall => hook_config::uninstall(),
            HookSub::Stop => hook::stop(),
            HookSub::SessionEnd => hook::session_end(),
            HookSub::MarkDirty => hook::mark_dirty(),
        },
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
    eprintln!(
        "embedded {} docs — {} chunks total, {} docs pending",
        n, vectors, pending
    );
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
    anyhow::bail!(
        "built without 'embed' feature — rebuild with: cargo build --release --features embed"
    )
}

#[cfg(not(feature = "embed"))]
fn cmd_vsearch(_query: &str, _limit: usize, _json: bool) -> Result<()> {
    anyhow::bail!(
        "built without 'embed' feature — rebuild with: cargo build --release --features embed"
    )
}

#[cfg(not(feature = "embed"))]
fn cmd_query(_query: &str, _limit: usize, _json: bool) -> Result<()> {
    anyhow::bail!(
        "built without 'embed' feature — rebuild with: cargo build --release --features embed"
    )
}
