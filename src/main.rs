mod bm25;
mod config;
mod output;
mod scan;
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

fn cmd_status() -> Result<()> {
    let cfg = config::load()?;
    let store = store::Store::open(&config::store_path())?;
    output::print_status(&cfg, &store)?;
    Ok(())
}
