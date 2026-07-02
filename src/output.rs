//! 출력 포맷 — qmd --json 호환 스키마(docid/score/file/title/context/snippet).

use crate::bm25::SearchHit;
use crate::config::IndexConfig;
use crate::store::Store;
use anyhow::Result;

pub fn to_json(hits: &[SearchHit]) -> Result<String> {
    Ok(serde_json::to_string_pretty(hits)?)
}

pub fn print_cli(hits: &[SearchHit]) {
    if hits.is_empty() {
        println!("No results found.");
        return;
    }
    for h in hits {
        println!("{:.2}  {}  {}", h.score, h.file, h.title);
        if let Some(s) = &h.snippet {
            for line in s.lines().take(3) {
                println!("      {}", line);
            }
        }
        println!();
    }
}

pub fn print_status(cfg: &IndexConfig, store: &Store) -> Result<()> {
    let (total, dirty) = store.counts()?;
    println!("kmd status");
    println!("  documents: {} active, {} dirty", total, dirty);
    println!("  collections ({}):", cfg.collections.len());
    let counts = store.collection_counts()?;
    for (name, coll) in &cfg.collections {
        let n = counts
            .iter()
            .find(|(c, _)| c == name)
            .map(|(_, n)| *n)
            .unwrap_or(0);
        println!("    {:<16} {:>6} files  ({})", name, n, coll.path.display());
    }
    Ok(())
}
