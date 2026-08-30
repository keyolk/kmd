//! 출력 포맷 — docid/score/file/title/context/snippet.

use crate::bm25::SearchHit;
use crate::config::IndexConfig;
use crate::global::AxisResults;
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

pub fn axes_to_json(axes: &[AxisResults]) -> Result<String> {
    Ok(serde_json::to_string_pretty(axes)?)
}

/// 축별 섹션으로 나눠 출력. 결과 없는 축도 표시해서 "검색은 됐는데 없다"와
/// "그 축이 인덱싱되지 않았다"를 구분할 수 있게 한다.
pub fn print_axes(axes: &[AxisResults]) {
    if axes.is_empty() {
        println!("No collections configured.");
        return;
    }
    for a in axes {
        println!("── {} ({})", a.axis, a.hits.len());
        if a.hits.is_empty() {
            println!("   no results");
            println!();
            continue;
        }
        for h in &a.hits {
            println!("{:.2}  {}  {}", h.score, h.file, h.title);
            if let Some(s) = &h.snippet {
                for line in s.lines().take(2) {
                    println!("      {}", line);
                }
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
        println!(
            "    {:<16} {:>7} files  [{}]  ({})",
            name,
            n,
            crate::global::axis_of(name, cfg),
            coll.path.display()
        );
    }
    Ok(())
}
