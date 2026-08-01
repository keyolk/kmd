//! 검색 품질 평가(L1) — `kmd eval`.
//!
//! 두 모드:
//! 1. `known-item` (자기지도): 문서에서 한글 구절을 뽑아 rag 파이프라인과 동일하게
//!    키워드화한 뒤, 그 쿼리로 원본 문서가 top-k에 돌아오는지 측정한다. 라벨이
//!    필요 없고 수백 쿼리로 재현 가능하며, 한글 형태소 매칭을 격리해서 잰다.
//!    `--compare-qmd`로 같은 쿼리를 qmd에도 던져 나란히 대조한다.
//! 2. `gold <path>`: 사람이 라벨한 gold YAML(prompt→기대 파일 substring)로 회귀 측정.
//!
//! 지표: Recall@1/3/5, MRR@k. 한글/영어 쿼리를 분리 집계한다.

use crate::config;
use crate::rag;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Instant;

// ------------------------------------------------------------- 공통 지표 ----

#[derive(Default, Clone, Serialize)]
pub struct Metrics {
    pub queries: usize,
    pub recall_at_1: usize,
    pub recall_at_3: usize,
    pub recall_at_5: usize,
    /// MRR 누적합 (평균은 mrr()로)
    mrr_sum: f64,
    /// 검색 지연 누적(ms)
    latency_sum_ms: u128,
}

impl Metrics {
    /// rank는 1-based, 매칭 못 하면 None.
    fn record(&mut self, rank: Option<usize>, latency_ms: u128) {
        self.queries += 1;
        self.latency_sum_ms += latency_ms;
        if let Some(r) = rank {
            if r <= 1 {
                self.recall_at_1 += 1;
            }
            if r <= 3 {
                self.recall_at_3 += 1;
            }
            if r <= 5 {
                self.recall_at_5 += 1;
            }
            self.mrr_sum += 1.0 / r as f64;
        }
    }
    pub fn mrr(&self) -> f64 {
        if self.queries == 0 {
            0.0
        } else {
            self.mrr_sum / self.queries as f64
        }
    }
    pub fn avg_latency_ms(&self) -> f64 {
        if self.queries == 0 {
            0.0
        } else {
            self.latency_sum_ms as f64 / self.queries as f64
        }
    }
    fn pct(n: usize, d: usize) -> f64 {
        if d == 0 {
            0.0
        } else {
            100.0 * n as f64 / d as f64
        }
    }
}

/// 한 엔진의 전체/한글/영어 분할 지표.
#[derive(Default, Serialize)]
pub struct EngineReport {
    pub engine: String,
    pub all: Metrics,
    pub korean: Metrics,
    pub english: Metrics,
}

// ---------------------------------------------------------- 검색 어댑터 ----

/// kmd BM25 검색 → expected 파일의 rank.
/// qmd는 파일명의 공백을 하이픈으로 정규화하므로, 엔진 간 공정 비교를 위해
/// 양쪽 경로를 동일 정규화(공백류→'-', 소문자)한 full-path로 매칭한다.
fn rank_kmd(expected: &str, hits: &[String]) -> Option<usize> {
    let want = norm_path(expected);
    hits.iter().position(|f| norm_path(f) == want).map(|i| i + 1)
}

/// 경로 정규화 — 공백/연속 공백을 '-'로, 소문자화. qmd/kmd 파일명 표기 차이 흡수.
fn norm_path(uri: &str) -> String {
    let mut out = String::with_capacity(uri.len());
    let mut prev_dash = false;
    for c in uri.to_lowercase().chars() {
        if c.is_whitespace() {
            if !prev_dash {
                out.push('-');
                prev_dash = true;
            }
        } else {
            out.push(c);
            prev_dash = false;
        }
    }
    out
}

// ------------------------------------------------------- known-item 모드 ----

struct DocSample {
    file: String, // qmd://coll/relpath
    query: String,
    hangul: bool,
}

/// 문서 본문에서 검색 쿼리를 만든다.
/// 한글 문서: 한글이 가장 많은 라인을 골라 rag와 동일 키워드 추출.
/// 그 외: 제목/본문 앞부분에서 영어 키워드 추출.
fn make_query(title: &str, body: &str) -> Option<(String, bool)> {
    // 후보 라인: 코드/헤더 기호 걷어내고 어느 정도 긴 라인만.
    let best_ko = body
        .lines()
        .map(|l| l.trim())
        .filter(|l| l.chars().count() >= 12)
        .max_by_key(|l| l.chars().filter(|c| ('\u{AC00}'..='\u{D7A3}').contains(c)).count());

    if let Some(line) = best_ko {
        if rag::has_hangul(line) {
            let q = rag::extract_keywords(line);
            if q.chars().count() >= 5 {
                return Some((q, true));
            }
        }
    }
    // 영어 폴백 — 제목 + 첫 긴 라인
    let en_line = body
        .lines()
        .map(|l| l.trim())
        .find(|l| l.chars().count() >= 20 && l.chars().any(|c| c.is_ascii_alphabetic()));
    let src = format!("{} {}", title, en_line.unwrap_or(""));
    let q = rag::extract_keywords(&src);
    if q.chars().count() >= 5 {
        Some((q, false))
    } else {
        None
    }
}

/// 재현 가능한 결정적 샘플링 — relpath FNV-1a 해시로 정렬 후 앞 N개.
fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn load_samples(collections: &[String], sample: usize, hangul_only: bool) -> Result<Vec<DocSample>> {
    let store = crate::store::Store::open(&config::store_path())?;
    let placeholders = collections.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let sql = format!(
        "SELECT collection, relpath, title, body FROM documents \
         WHERE active = 1 AND collection IN ({placeholders})"
    );
    let mut stmt = store.conn.prepare(&sql)?;
    let params: Vec<&dyn rusqlite::ToSql> =
        collections.iter().map(|c| c as &dyn rusqlite::ToSql).collect();
    let rows = stmt.query_map(params.as_slice(), |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, String>(2)?,
            r.get::<_, String>(3)?,
        ))
    })?;

    let mut all: Vec<DocSample> = Vec::new();
    for row in rows {
        let (coll, relpath, title, body) = row?;
        let Some((query, hangul)) = make_query(&title, &body) else {
            continue;
        };
        if hangul_only && !hangul {
            continue;
        }
        all.push(DocSample {
            file: format!("qmd://{}/{}", coll, relpath),
            query,
            hangul,
        });
    }
    // 결정적 셔플 후 절단
    all.sort_by_key(|d| fnv1a(&d.file));
    if sample > 0 && all.len() > sample {
        all.truncate(sample);
    }
    Ok(all)
}

/// qmd에 쿼리를 던져 top-k file 목록을 파싱.
fn qmd_search(query: &str, k: usize) -> Option<Vec<String>> {
    let out = std::process::Command::new("qmd")
        .args(["search", query, "-n", &k.to_string(), "--json"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).ok()?;
    Some(
        v.as_array()?
            .iter()
            .filter_map(|h| h.get("file").and_then(|f| f.as_str()).map(String::from))
            .collect(),
    )
}

pub fn known_item(
    collections: Vec<String>,
    sample: usize,
    k: usize,
    compare_qmd: bool,
    hangul_only: bool,
    json: bool,
) -> Result<()> {
    let cfg = config::load()?;
    let samples = load_samples(&collections, sample, hangul_only)?;
    if samples.is_empty() {
        anyhow::bail!("no eval samples — check collections / index");
    }

    let mut kmd = EngineReport {
        engine: "kmd".into(),
        ..Default::default()
    };
    let mut qmd = EngineReport {
        engine: "qmd".into(),
        ..Default::default()
    };

    for s in &samples {
        // kmd
        let t = Instant::now();
        let hits = crate::bm25::search(&config::tantivy_dir(), &cfg, &s.query, k, None)?;
        let lat = t.elapsed().as_millis();
        let files: Vec<String> = hits.iter().map(|h| h.file.clone()).collect();
        let rank = rank_kmd(&s.file, &files);
        kmd.all.record(rank, lat);
        if s.hangul {
            kmd.korean.record(rank, lat);
        } else {
            kmd.english.record(rank, lat);
        }

        if compare_qmd {
            let t = Instant::now();
            let qfiles = qmd_search(&s.query, k).unwrap_or_default();
            let lat = t.elapsed().as_millis();
            let rank = rank_kmd(&s.file, &qfiles);
            qmd.all.record(rank, lat);
            if s.hangul {
                qmd.korean.record(rank, lat);
            } else {
                qmd.english.record(rank, lat);
            }
        }
    }

    let reports: Vec<&EngineReport> = if compare_qmd {
        vec![&kmd, &qmd]
    } else {
        vec![&kmd]
    };

    if json {
        #[derive(Serialize)]
        struct Out<'a> {
            mode: &'a str,
            collections: &'a [String],
            k: usize,
            reports: &'a [&'a EngineReport],
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&Out {
                mode: "known-item",
                collections: &collections,
                k,
                reports: &reports,
            })?
        );
    } else {
        print_report("known-item", &collections, k, &reports);
    }
    Ok(())
}

// ------------------------------------------------------------- gold 모드 ----

#[derive(Debug, Deserialize)]
struct GoldCase {
    prompt: String,
    /// top-k 히트의 file 또는 title에 이 substring 중 하나라도 있으면 정답.
    #[serde(default)]
    expect_any: Vec<String>,
}

pub fn gold(path: &Path, k: usize, compare_qmd: bool, json: bool) -> Result<()> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("cannot read gold file {}", path.display()))?;
    let cases: Vec<GoldCase> =
        serde_yaml::from_str(&raw).with_context(|| format!("invalid gold yaml {}", path.display()))?;
    if cases.is_empty() {
        anyhow::bail!("gold file has no cases");
    }
    let cfg = config::load()?;

    let mut kmd = EngineReport {
        engine: "kmd".into(),
        ..Default::default()
    };
    let mut qmd = EngineReport {
        engine: "qmd".into(),
        ..Default::default()
    };

    for c in &cases {
        // 실제 훅과 동일하게 프롬프트에서 키워드 추출.
        let query = match rag::gate(&c.prompt) {
            Ok(q) => q,
            Err(_) => c.prompt.clone(),
        };
        let hangul = rag::has_hangul(&c.prompt);

        let t = Instant::now();
        let hits = crate::bm25::search(&config::tantivy_dir(), &cfg, &query, k, None)?;
        let lat = t.elapsed().as_millis();
        let rank = gold_rank(&c.expect_any, hits.iter().map(|h| (h.file.as_str(), h.title.as_str())));
        kmd.all.record(rank, lat);
        if hangul {
            kmd.korean.record(rank, lat);
        } else {
            kmd.english.record(rank, lat);
        }

        if compare_qmd {
            let t = Instant::now();
            let qhits = qmd_search(&query, k).unwrap_or_default();
            let lat = t.elapsed().as_millis();
            let rank = gold_rank(&c.expect_any, qhits.iter().map(|f| (f.as_str(), "")));
            qmd.all.record(rank, lat);
            if hangul {
                qmd.korean.record(rank, lat);
            } else {
                qmd.english.record(rank, lat);
            }
        }
    }

    let reports: Vec<&EngineReport> = if compare_qmd {
        vec![&kmd, &qmd]
    } else {
        vec![&kmd]
    };
    if json {
        println!("{}", serde_json::to_string_pretty(&reports)?);
    } else {
        print_report("gold", &[format!("{} cases", cases.len())], k, &reports);
    }
    Ok(())
}

/// expect_any substring 중 하나라도 매칭되는 첫 히트의 rank(1-based).
fn gold_rank<'a>(
    expect: &[String],
    hits: impl Iterator<Item = (&'a str, &'a str)>,
) -> Option<usize> {
    for (i, (file, title)) in hits.enumerate() {
        let hay = format!("{} {}", file, title).to_lowercase();
        if expect.iter().any(|e| hay.contains(&e.to_lowercase())) {
            return Some(i + 1);
        }
    }
    None
}

// ----------------------------------------------------------- 리포트 출력 ----

fn print_report(mode: &str, scope: &[String], k: usize, reports: &[&EngineReport]) {
    println!("kmd eval — mode: {}   k: {}", mode, k);
    println!("scope: {}", scope.join(", "));
    println!();
    let q = reports[0].all.queries;
    println!("queries: {} total  ({} korean, {} english)", q, reports[0].korean.queries, reports[0].english.queries);
    println!();

    for label in ["all", "korean", "english"] {
        println!("── {} ──", label);
        println!(
            "  {:<6} {:>8} {:>8} {:>8} {:>8} {:>10}",
            "engine", "R@1", "R@3", "R@5", "MRR", "lat(ms)"
        );
        for r in reports {
            let m = match label {
                "korean" => &r.korean,
                "english" => &r.english,
                _ => &r.all,
            };
            if m.queries == 0 {
                continue;
            }
            println!(
                "  {:<6} {:>7.0}% {:>7.0}% {:>7.0}% {:>8.3} {:>10.1}",
                r.engine,
                Metrics::pct(m.recall_at_1, m.queries),
                Metrics::pct(m.recall_at_3, m.queries),
                Metrics::pct(m.recall_at_5, m.queries),
                m.mrr(),
                m.avg_latency_ms(),
            );
        }
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_path_bridges_space_and_dash() {
        // qmd(하이픈)와 kmd(공백) 파일명이 동일 정규화로 매칭돼야 공정 비교가 성립.
        assert_eq!(
            norm_path("qmd://learnings/20260429 0-85eaf018.md"),
            norm_path("qmd://learnings/20260429-0-85eaf018.md"),
        );
    }

    #[test]
    fn norm_path_collapses_repeated_whitespace() {
        assert_eq!(norm_path("a  b\tc"), "a-b-c");
    }

    #[test]
    fn rank_matches_after_normalization() {
        let hits = vec![
            "qmd://x/other.md".to_string(),
            "qmd://learnings/20260429-0-abc.md".to_string(),
        ];
        // expected는 공백형, 히트는 하이픈형이어도 rank 2로 매칭.
        assert_eq!(rank_kmd("qmd://learnings/20260429 0-abc.md", &hits), Some(2));
    }

    #[test]
    fn rank_none_when_absent() {
        let hits = vec!["qmd://x/a.md".to_string()];
        assert_eq!(rank_kmd("qmd://y/z.md", &hits), None);
    }

    #[test]
    fn gold_rank_substring_and_ordering() {
        let hits = [
            ("qmd://wiki/tcp/keepalive.md", "TCP Keepalive"),
            ("qmd://claude-memory/tls-certs.md", "TLS Certs"),
        ];
        // 첫 매칭의 1-based rank.
        assert_eq!(gold_rank(&["tls-certs".into()], hits.iter().copied()), Some(2));
        assert_eq!(gold_rank(&["keepalive".into()], hits.iter().copied()), Some(1));
        assert_eq!(gold_rank(&["nonexistent".into()], hits.iter().copied()), None);
    }

    #[test]
    fn metrics_recall_and_mrr() {
        let mut m = Metrics::default();
        m.record(Some(1), 10); // R@1,3,5 + mrr 1.0
        m.record(Some(3), 10); // R@3,5 + mrr 1/3
        m.record(None, 10); // miss
        assert_eq!(m.queries, 3);
        assert_eq!(m.recall_at_1, 1);
        assert_eq!(m.recall_at_3, 2);
        assert_eq!(m.recall_at_5, 2);
        assert!((m.mrr() - (1.0 + 1.0 / 3.0) / 3.0).abs() < 1e-9);
    }
}
