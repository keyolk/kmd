//! rag.jsonl 관측 명령 — stats(집계) / log(최근 상세).

use crate::rag::{RagLogEntry, rag_log_path};
use anyhow::Result;
use std::collections::BTreeMap;

fn load_entries(since_secs: Option<u64>) -> Result<Vec<RagLogEntry>> {
    let path = rag_log_path();
    let raw = match std::fs::read_to_string(&path) {
        Ok(r) => r,
        Err(_) => return Ok(vec![]),
    };
    let cutoff = since_secs.map(|s| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            .saturating_sub(s)
    });
    Ok(raw
        .lines()
        .filter_map(|l| serde_json::from_str::<RagLogEntry>(l).ok())
        .filter(|e| match cutoff {
            Some(c) => e.ts.parse::<u64>().map(|t| t >= c).unwrap_or(true),
            None => true,
        })
        .collect())
}

fn pct(n: usize, d: usize) -> f64 {
    if d == 0 { 0.0 } else { 100.0 * n as f64 / d as f64 }
}

pub fn print_stats(since_secs: Option<u64>) -> Result<()> {
    let entries = load_entries(since_secs)?;
    if entries.is_empty() {
        println!("no rag log entries yet ({})", rag_log_path().display());
        return Ok(());
    }

    let total = entries.len();
    let searched: Vec<_> = entries.iter().filter(|e| e.stage == "searched").collect();
    let injected: Vec<_> = searched.iter().filter(|e| e.injected > 0).collect();

    println!("kmd rag stats — {} prompts", total);
    println!();

    // 게이팅 분포
    let mut gates: BTreeMap<&str, usize> = BTreeMap::new();
    for e in &entries {
        let key = e.gate_reason.as_deref().unwrap_or("searched");
        *gates.entry(key).or_default() += 1;
    }
    println!("  gate:");
    for (k, v) in &gates {
        println!("    {:16} {:5} ({:.0}%)", k, v, pct(*v, total));
    }
    println!();

    // 주입률 (한글/영어 분리)
    for (label, filt) in [
        ("all", None::<bool>),
        ("hangul", Some(true)),
        ("non-hangul", Some(false)),
    ] {
        let s: Vec<_> = searched
            .iter()
            .filter(|e| filt.map(|h| e.hangul == h).unwrap_or(true))
            .collect();
        let inj = s.iter().filter(|e| e.injected > 0).count();
        let chunks: usize = s.iter().map(|e| e.injected).sum();
        if s.is_empty() {
            continue;
        }
        println!(
            "  inject[{:10}] {:4}/{:4} searched ({:.0}%), avg {:.2} chunks",
            label,
            inj,
            s.len(),
            pct(inj, s.len()),
            chunks as f64 / s.len() as f64
        );
    }
    println!();

    // 컬렉션 히트
    let mut colls: BTreeMap<String, usize> = BTreeMap::new();
    for e in &searched {
        for h in &e.hits {
            let coll = h
                .file
                .strip_prefix("qmd://")
                .or_else(|| h.file.strip_prefix("kmd://"))
                .and_then(|r| r.split('/').next())
                .unwrap_or("?")
                .to_string();
            *colls.entry(coll).or_default() += 1;
        }
    }
    if !colls.is_empty() {
        println!("  collection hits:");
        let mut sorted: Vec<_> = colls.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        for (k, v) in sorted {
            println!("    {:20} {}", k, v);
        }
        println!();
    }

    // 레이턴시
    let mut lats: Vec<u64> = searched.iter().map(|e| e.latency_ms).collect();
    lats.sort_unstable();
    if !lats.is_empty() {
        println!(
            "  latency: median {}ms / p95 {}ms / max {}ms",
            lats[lats.len() / 2],
            lats[(lats.len() as f64 * 0.95) as usize % lats.len()],
            lats.last().unwrap()
        );
    }

    let _ = injected;
    Ok(())
}

pub fn print_log(count: usize, follow: bool) -> Result<()> {
    let path = rag_log_path();
    let entries = load_entries(None)?;
    for e in entries.iter().rev().take(count).rev() {
        print_entry(e);
    }
    if follow {
        // 간단한 tail -f: 파일 길이 폴링
        use std::io::{BufRead, BufReader, Seek, SeekFrom};
        let mut f = std::fs::File::open(&path)?;
        f.seek(SeekFrom::End(0))?;
        let mut reader = BufReader::new(f);
        loop {
            let mut line = String::new();
            let n = reader.read_line(&mut line)?;
            if n == 0 {
                std::thread::sleep(std::time::Duration::from_millis(500));
                continue;
            }
            if let Ok(e) = serde_json::from_str::<RagLogEntry>(&line) {
                print_entry(&e);
            }
        }
    }
    Ok(())
}

fn print_entry(e: &RagLogEntry) {
    let ts = e
        .ts
        .parse::<u64>()
        .ok()
        .map(fmt_epoch)
        .unwrap_or_else(|| e.ts.clone());
    let prompt_short: String = e.prompt.chars().take(60).collect();
    let lang = if e.hangul { "ko" } else { "en" };

    match e.stage.as_str() {
        "gated" => {
            println!(
                "{} [{}] GATED({}) {:?}",
                ts,
                lang,
                e.gate_reason.as_deref().unwrap_or("?"),
                prompt_short
            );
        }
        _ => {
            println!(
                "{} [{}] inj={} {}ms q={:?} prompt={:?}",
                ts,
                lang,
                e.injected,
                e.latency_ms,
                e.query.as_deref().unwrap_or(""),
                prompt_short
            );
            for h in &e.hits {
                println!("      {:.1} {}", h.score, h.file);
            }
        }
    }
}

fn fmt_epoch(secs: u64) -> String {
    // 로컬타임 변환 없이 UTC 단순 변환 (관측용으로 충분)
    let days = secs / 86400;
    let rem = secs % 86400;
    let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    // 1970-01-01 기준 날짜 계산 (그레고리안)
    let mut year = 1970u64;
    let mut d = days;
    loop {
        let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
        let ydays = if leap { 366 } else { 365 };
        if d < ydays {
            break;
        }
        d -= ydays;
        year += 1;
    }
    let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    let months = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut month = 1;
    for len in months {
        if d < len {
            break;
        }
        d -= len;
        month += 1;
    }
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year,
        month,
        d + 1,
        h,
        m,
        s
    )
}
