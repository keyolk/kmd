mod bm25;
mod config;
mod output;
mod rag;
mod scan;
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
    /// Index/collection status
    Status,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Update { force } => cmd_update(force),
        Command::Search {
            query,
            limit,
            collection,
            json,
        } => cmd_search(&query, limit, collection.as_deref(), json),
        Command::Rag { hook, prompt } => cmd_rag(hook, prompt.as_deref()),
        Command::Stats { hours } => stats::print_stats(hours.map(|h| h * 3600)),
        Command::Log { count, follow } => stats::print_log(count, follow),
        Command::Status => cmd_status(),
    }
}

fn cmd_update(force: bool) -> Result<()> {
    let cfg = config::load()?;
    let mut store = store::Store::open(&config::store_path())?;
    let stats = scan::update(&cfg, &mut store, force)?;
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
