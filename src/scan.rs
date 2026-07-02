//! 컬렉션 스캔 — index.yml의 path/pattern 기반 파일 발견 + 증분 upsert.

use crate::config::IndexConfig;
use crate::store::{Store, UpsertOutcome};
use anyhow::Result;
use globset::{Glob, GlobSetBuilder};
use sha2::{Digest, Sha256};
use std::time::UNIX_EPOCH;
use walkdir::WalkDir;

#[derive(Debug, Default)]
pub struct ScanStats {
    pub seen: usize,
    pub added: usize,
    pub updated: usize,
    pub removed: usize,
}

pub fn update(cfg: &IndexConfig, store: &mut Store, force: bool) -> Result<ScanStats> {
    let mut stats = ScanStats::default();

    for (name, coll) in &cfg.collections {
        if !coll.path.is_dir() {
            eprintln!("warn: collection {} path missing: {}", name, coll.path.display());
            continue;
        }
        // "**/*.{md,txt}" 같은 brace 패턴 지원
        let mut gs = GlobSetBuilder::new();
        gs.add(Glob::new(&coll.pattern)?);
        let glob = gs.build()?;

        let context = coll.root_context().map(str::to_string);
        let mut seen_paths: Vec<String> = Vec::new();

        for entry in WalkDir::new(&coll.path)
            .follow_links(false)
            .into_iter()
            .filter_entry(|e| !is_hidden(e))
            .filter_map(|e| e.ok())
        {
            if !entry.file_type().is_file() {
                continue;
            }
            let rel = match entry.path().strip_prefix(&coll.path) {
                Ok(r) => r,
                Err(_) => continue,
            };
            if !glob.is_match(rel) {
                continue;
            }
            let relpath = rel.to_string_lossy().to_string();
            seen_paths.push(relpath.clone());
            stats.seen += 1;

            let meta = entry.metadata()?;
            let mtime_ns = meta
                .modified()?
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as i64)
                .unwrap_or(0);
            let size = meta.len() as i64;

            let body = match std::fs::read_to_string(entry.path()) {
                Ok(b) => b,
                Err(_) => continue, // 바이너리/읽기 불가 스킵
            };
            let hash = if force {
                // force 시 hash를 다르게 만들어 무조건 재인덱싱
                format!("force-{:x}", Sha256::digest(body.as_bytes()))
            } else {
                format!("{:x}", Sha256::digest(body.as_bytes()))
            };
            let title = extract_title(&relpath, &body);

            match store.upsert_doc(
                name,
                &relpath,
                &title,
                &body,
                context.as_deref(),
                mtime_ns,
                size,
                &hash,
            )? {
                UpsertOutcome::Added => stats.added += 1,
                UpsertOutcome::Updated => stats.updated += 1,
                UpsertOutcome::Unchanged => {}
            }
        }

        stats.removed += store.deactivate_missing(name, &seen_paths)?;
    }

    Ok(stats)
}

fn is_hidden(entry: &walkdir::DirEntry) -> bool {
    entry.depth() > 0
        && entry
            .file_name()
            .to_str()
            .map(|s| s.starts_with('.'))
            .unwrap_or(false)
}

/// 첫 `# 헤더` 또는 frontmatter title, 없으면 파일명.
fn extract_title(relpath: &str, body: &str) -> String {
    let mut in_frontmatter = false;
    for (i, line) in body.lines().enumerate() {
        let trimmed = line.trim();
        if i == 0 && trimmed == "---" {
            in_frontmatter = true;
            continue;
        }
        if in_frontmatter {
            if trimmed == "---" {
                in_frontmatter = false;
            } else if let Some(t) = trimmed.strip_prefix("title:") {
                return t.trim().trim_matches('"').to_string();
            }
            continue;
        }
        if let Some(h) = trimmed.strip_prefix("# ") {
            return h.trim().to_string();
        }
    }
    relpath
        .rsplit('/')
        .next()
        .unwrap_or(relpath)
        .trim_end_matches(".md")
        .to_string()
}
