//! 검색 품질 평가(L1) — `kmd eval`.
//!
//! 두 모드:
//! 1. `known-item` (자기지도): 문서에서 한글 구절을 뽑아 rag 파이프라인과 동일하게
//!    키워드화한 뒤, 그 쿼리로 원본 문서가 top-k에 돌아오는지 측정한다. 라벨이
//!    필요 없고 수백 쿼리로 재현 가능하며, 한글 형태소 매칭을 격리해서 잰다.
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

// ------------------------------------------------------------ 검색 엔진 ----

/// 평가할 검색 엔진. `--engine`으로 고르거나 `--compare`로 전부 돌린다.
#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Engine {
    /// lindera ko-dic BM25 (embed 피처 없이도 동작)
    Bm25,
    /// embeddinggemma 코사인 유사도
    Vector,
    /// BM25 + 벡터 RRF 융합
    Hybrid,
    /// RAG 훅이 실제로 주입하는 것 — 검색 + `rag::filter_hits`.
    ///
    /// 다른 세 엔진은 **원시 검색 결과**를 잰다. 훅은 그 위에 컬렉션 가중
    /// 재정렬, L0 커버 제외, 3건 절단을 얹는데 그것들이 `filter_hits`에 있어서
    /// 원시 측정으로는 보이지 않는다. 실측: `KMD_CURATED_WEIGHT`를 1.0에서
    /// 5.0까지 바꿔도 gold 점수가 한 자리도 안 움직였다 — 그 가중을 쓰는
    /// 코드를 eval이 아예 지나가지 않았기 때문이다.
    Pipeline,
}

impl Engine {
    fn label(self) -> &'static str {
        match self {
            Engine::Bm25 => "bm25",
            Engine::Vector => "vector",
            Engine::Hybrid => "hybrid",
            Engine::Pipeline => "pipeline",
        }
    }

    /// 이 엔진이 임베딩을 읽는가.
    ///
    /// 검색 범위를 좁힐지 정하는 기준이다. 엔진 **개수**가 아니라 어느 엔진을
    /// 쓰는지가 기준이어야 한다: `--engine hybrid` 단독은 엔진이 하나여도
    /// 벡터를 읽으므로, 범위를 안 좁히면 임베딩이 없는 project 축 36만 건이
    /// BM25 쪽에만 후보로 들어와 융합이 왜곡된다. 실측으로 같은 factor에서
    /// `--compare`는 R@5 52%, `--engine hybrid`는 44%가 나왔다 — 엔진이 아니라
    /// 모집단이 달랐던 것이다.
    fn reads_embeddings(self) -> bool {
        // pipeline은 embed 피처가 있으면 hybrid로 검색하므로 벡터를 읽는다.
        matches!(self, Engine::Vector | Engine::Hybrid | Engine::Pipeline)
    }
}

/// 검색 범위를 좁혀야 하는가 — 임베딩을 읽는 엔진이 하나라도 있으면 그렇다.
fn needs_embedded_scope(engines: &[Engine]) -> bool {
    engines.iter().any(|e| e.reads_embeddings())
}

/// 한 쿼리를 한 엔진으로 검색한다.
///
/// 세 엔진이 같은 컬렉션 집합을 보게 하는 것이 이 함수의 존재 이유다. BM25는
/// tantivy 쿼리에서, 벡터는 SQL WHERE에서 각각 필터하므로 호출부가 같은
/// `collections`를 넘겨야 모집단이 일치한다.
fn run_engine(
    engine: Engine,
    cfg: &config::IndexConfig,
    store: &crate::store::Store,
    query: &str,
    prompt: &str,
    k: usize,
    collections: Option<&[&str]>,
) -> Result<Vec<crate::bm25::SearchHit>> {
    match engine {
        // 훅과 같은 경로. **원본 프롬프트**를 넘긴다 — 다른 엔진은 호출부가
        // 미리 키워드를 뽑아 `query`로 주지만, `run_pipeline`은 게이트와 키워드
        // 추출을 스스로 한다. 추출된 쿼리를 넘기면 그것이 다시 게이트에 들어가
        // `too_short`로 대부분 걸린다(실측: gold 55케이스 중 통과가 16%뿐이었고,
        // 그래서 어떤 가중을 줘도 점수가 한 자리도 움직이지 않았다).
        //
        // `filter_hits`가 컬렉션 범위와 결과 수(3건)를 스스로 정하므로 k와
        // collections도 쓰지 않는다. 훅이 실제 하는 일을 재는 것이 이 엔진의
        // 목적이고, 파라미터를 끼워넣으면 그게 아니게 된다.
        Engine::Pipeline => {
            let outcome = crate::rag::run_pipeline(prompt)?;
            Ok(outcome.hits)
        }
        Engine::Bm25 => crate::bm25::search_in(&config::tantivy_dir(), cfg, query, k, collections),
        #[cfg(feature = "embed")]
        Engine::Vector => crate::embed::vsearch_in(store, query, k, collections),
        #[cfg(feature = "embed")]
        Engine::Hybrid => crate::embed::hybrid_in(
            store,
            &config::tantivy_dir(),
            cfg,
            query,
            k,
            collections,
        ),
        #[cfg(not(feature = "embed"))]
        Engine::Vector | Engine::Hybrid => {
            let _ = store;
            anyhow::bail!(
                "engine {} needs the 'embed' feature — rebuild with: cargo build --release --features embed",
                engine.label()
            )
        }
    }
}

/// `--engine` / `--compare` 조합을 실제로 돌릴 엔진 목록으로 바꾼다.
fn engines_for(engine: Option<Engine>, compare: bool) -> Vec<Engine> {
    if compare {
        // pipeline은 `--compare`에 넣지 않는다. 나머지 셋은 같은 조건에서 겨루는
        // 검색 엔진이지만 pipeline은 그 위의 필터·절단까지 포함한 다른 층위여서,
        // 한 표에 놓으면 엔진 비교로 오독된다. `--engine pipeline`으로 따로 잰다.
        vec![Engine::Bm25, Engine::Vector, Engine::Hybrid]
    } else {
        vec![engine.unwrap_or(Engine::Bm25)]
    }
}

/// 임베딩이 존재할 수 있는 컬렉션 — 본문을 store에 복사하는 것들.
///
/// `embed_pending`이 `abspath IS NULL`로 거르는 것과 같은 집합을 설정 쪽에서
/// 표현한다. 두 조건이 갈라지면 비교가 조용히 편향되므로 근거를 여기 남긴다.
fn embedded_collections(cfg: &config::IndexConfig) -> Vec<String> {
    cfg.collections
        .iter()
        .filter(|(_, c)| c.stores_body())
        .map(|(name, _)| name.clone())
        .collect()
}

/// 대조 범위의 임베딩 커버리지를 재고, 완전하지 않으면 경고한다.
///
/// 부분 커버리지는 **양방향으로** 결과를 왜곡하는데, 어느 쪽으로 왜곡되는지는
/// 어느 문서가 빠졌는지에 달려 있어 수치만 보고는 구분되지 않는다.
///
/// - 정답 문서에 임베딩이 없으면 벡터는 그 쿼리를 절대 못 맞힌다 (과소평가).
/// - 경쟁 문서에 임베딩이 없으면 벡터만 상대가 적은 모집단에서 검색한다 (과대평가).
///
/// 그래서 커버리지를 리포트에 같이 싣는다. 100%가 아닌 대조 결과는 엔진 우열의
/// 근거로 쓸 수 없다 — 숫자는 나오지만 무엇을 재는지가 정해지지 않는다.
fn embedding_coverage(
    store: &crate::store::Store,
    collections: &[&str],
) -> Result<(i64, i64)> {
    if collections.is_empty() {
        return Ok((0, 0));
    }
    let placeholders = vec!["?"; collections.len()].join(",");
    let sql = format!(
        "SELECT COUNT(*),
                SUM(CASE WHEN EXISTS (SELECT 1 FROM embeddings e WHERE e.doc_id = d.id)
                         THEN 1 ELSE 0 END)
         FROM documents d
         WHERE d.active = 1 AND d.abspath IS NULL AND d.collection IN ({})",
        placeholders
    );
    let params: Vec<&dyn rusqlite::ToSql> = collections
        .iter()
        .map(|c| c as &dyn rusqlite::ToSql)
        .collect();
    let (total, embedded): (i64, Option<i64>) = store
        .conn
        .query_row(&sql, params.as_slice(), |r| Ok((r.get(0)?, r.get(1)?)))?;
    Ok((total, embedded.unwrap_or(0)))
}

/// 커버리지가 완전하지 않을 때 stderr에 경고를 낸다. 반환값은 리포트에 실을 수치.
fn warn_partial_coverage(
    store: &crate::store::Store,
    scope: Option<&[&str]>,
) -> Result<Option<(i64, i64)>> {
    let Some(cols) = scope else {
        return Ok(None);
    };
    let (total, embedded) = embedding_coverage(store, cols)?;
    if total > 0 && embedded < total {
        eprintln!(
            "warning: embedding coverage {}/{} ({:.0}%) over the compared scope — \
             vector/hybrid numbers are not a fair comparison yet. \
             Run `kmd embed` to completion first.",
            embedded,
            total,
            100.0 * embedded as f64 / total as f64
        );
    }
    Ok(Some((total, embedded)))
}

// ---------------------------------------------------------- 검색 어댑터 ----

/// kmd BM25 검색 → expected 파일의 rank.
/// 과거 인덱스는 파일명의 공백을 하이픈으로 정규화했으므로, 표기 차이를 흡수하려고
/// 양쪽 경로를 동일 정규화(공백류→'-', 소문자)한 full-path로 매칭한다.
fn rank_kmd(expected: &str, hits: &[String]) -> Option<usize> {
    let want = norm_path(expected);
    hits.iter().position(|f| norm_path(f) == want).map(|i| i + 1)
}

/// 경로 정규화 — 스킴 접두어를 벗기고, 공백/연속 공백을 '-'로, 소문자화.
/// 과거 gold 파일이 옛 스킴(qmd://)으로 적혀 있어도 매칭되게 한다.
fn norm_path(uri: &str) -> String {
    let stripped = uri
        .strip_prefix("qmd://")
        .or_else(|| uri.strip_prefix("kmd://"))
        .unwrap_or(uri);
    let mut out = String::with_capacity(stripped.len());
    let mut prev_dash = false;
    for c in stripped.to_lowercase().chars() {
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
    file: String, // kmd://coll/relpath
    query: String,
    hangul: bool,
}

/// 문서 본문에서 검색 쿼리를 만든다.
/// 한글 문서: 한글이 가장 많은 라인을 골라 rag와 동일 키워드 추출.
/// 그 외: 제목/본문 앞부분에서 영어 키워드 추출.
///
/// **이 쿼리는 문서와 어휘가 완전히 겹친다.** 실측하면 상당수가 문서의 제목
/// 줄 그대로다(`# AWS Accounts & Kubernetes Contexts`로 그 문서를 찾는 꼴).
/// 그래서 known-item 모드는 BM25가 설계상 가장 강한 조건에서 측정하며, 의미
/// 검색이 기여할 여지가 거의 없다 — 실측 BM25 R@5 90% 대 vector 69%.
///
/// 그 수치를 엔진 우열로 읽으면 안 된다. known-item이 재는 것은 "인덱스가
/// 이 문서를 되찾을 수 있는가"(라벨 없이 수백 쿼리로 재현 가능한 회귀 지표)이고,
/// "사용자의 말로 이 문서에 닿을 수 있는가"는 재지 않는다. 후자는
/// `eval/gold.yaml`의 패러프레이즈 구간이 재며, 거기서는 순서가 뒤집힌다
/// (한국어 BM25 6% 대 vector 39%). 두 모드는 서로 다른 질문에 답한다.
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
            file: format!("kmd://{}/{}", coll, relpath),
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

pub fn known_item(
    collections: Vec<String>,
    sample: usize,
    k: usize,
    hangul_only: bool,
    engine: Option<Engine>,
    compare: bool,
    json: bool,
) -> Result<()> {
    let cfg = config::load()?;
    let samples = load_samples(&collections, sample, hangul_only)?;
    if samples.is_empty() {
        anyhow::bail!("no eval samples — check collections / index");
    }
    let store = crate::store::Store::open(&config::store_path())?;
    let engines = engines_for(engine, compare);

    // 임베딩을 읽는 엔진이 끼면 검색 범위를 샘플 컬렉션으로 좁힌다.
    //
    // BM25 단독 평가는 전 인덱스를 대상으로 한다(retrievability ceiling). 그 범위를
    // 벡터에 그대로 쓰면 비교가 성립하지 않는다: 벡터 인덱스에는 project 축 36만 건이
    // 아예 없어서, 벡터 엔진만 경쟁 문서가 적은 모집단에서 검색하게 된다. 그건
    // 벡터가 더 잘 찾은 게 아니라 상대가 없었던 것이다.
    let scope: Option<Vec<&str>> = if needs_embedded_scope(&engines) {
        Some(collections.iter().map(String::as_str).collect())
    } else {
        None
    };
    let scope_ref = scope.as_deref();
    let coverage = warn_partial_coverage(&store, scope_ref)?;

    let mut reports_owned: Vec<EngineReport> = Vec::new();
    for eng in &engines {
        let mut rep = EngineReport {
            engine: eng.label().into(),
            ..Default::default()
        };
        for s in &samples {
            let t = Instant::now();
            // known-item 샘플은 문서 본문에서 만든 쿼리뿐이고 원본 프롬프트가
            // 없다. pipeline 엔진에는 그 쿼리를 프롬프트로 준다 — 게이트를
            // 통과하지 못하면 히트가 비고, 그건 이 모드에서 정직한 결과다.
            let hits = run_engine(*eng, &cfg, &store, &s.query, &s.query, k, scope_ref)?;
            let lat = t.elapsed().as_millis();
            let files: Vec<String> = hits.iter().map(|h| h.file.clone()).collect();
            let rank = rank_kmd(&s.file, &files);
            rep.all.record(rank, lat);
            if s.hangul {
                rep.korean.record(rank, lat);
            } else {
                rep.english.record(rank, lat);
            }
        }
        reports_owned.push(rep);
    }

    let reports: Vec<&EngineReport> = reports_owned.iter().collect();
    let head = &reports_owned[0];
    crate::evaluation_log::record(
        "eval-known-item",
        format!(
            "{} queries · {} R@5 {:.0}% · MRR {:.3}",
            head.all.queries,
            head.engine,
            Metrics::pct(head.all.recall_at_5, head.all.queries),
            head.all.mrr()
        ),
        serde_json::json!({
            "mode": "known-item",
            "collections": &collections,
            "k": k,
            "scoped": scope.is_some(),
            "embedding_coverage": coverage.map(|(t, e)| serde_json::json!({"total": t, "embedded": e})),
            "reports": &reports,
        }),
    );

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

pub fn gold(
    path: &Path,
    k: usize,
    engine: Option<Engine>,
    compare: bool,
    json: bool,
) -> Result<()> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("cannot read gold file {}", path.display()))?;
    let cases: Vec<GoldCase> =
        serde_yaml::from_str(&raw).with_context(|| format!("invalid gold yaml {}", path.display()))?;
    if cases.is_empty() {
        anyhow::bail!("gold file has no cases");
    }
    let cfg = config::load()?;
    let store = crate::store::Store::open(&config::store_path())?;
    let engines = engines_for(engine, compare);

    // 임베딩을 읽는 엔진이 끼면 임베딩이 존재하는 컬렉션으로 범위를 좁힌다.
    // BM25 단독 gold가 전 인덱스를 보는 것은 의도(retrievability ceiling)지만,
    // project 축은 임베딩 대상이 아니므로 그 범위에서 벡터를 재면 안 된다.
    let embedded = embedded_collections(&cfg);
    let scope: Option<Vec<&str>> = if needs_embedded_scope(&engines) {
        Some(embedded.iter().map(String::as_str).collect())
    } else {
        None
    };
    let scope_ref = scope.as_deref();
    let coverage = warn_partial_coverage(&store, scope_ref)?;

    let mut reports_owned: Vec<EngineReport> = Vec::new();
    for eng in &engines {
        let mut rep = EngineReport {
            engine: eng.label().into(),
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
            let hits = run_engine(*eng, &cfg, &store, &query, &c.prompt, k, scope_ref)?;
            let lat = t.elapsed().as_millis();
            let rank =
                gold_rank(&c.expect_any, hits.iter().map(|h| (h.file.as_str(), h.title.as_str())));
            rep.all.record(rank, lat);
            if hangul {
                rep.korean.record(rank, lat);
            } else {
                rep.english.record(rank, lat);
            }
        }
        reports_owned.push(rep);
    }

    let reports: Vec<&EngineReport> = reports_owned.iter().collect();
    let head = &reports_owned[0];
    crate::evaluation_log::record(
        "eval-gold",
        format!(
            "{} cases · {} R@5 {:.0}% · MRR {:.3}",
            head.all.queries,
            head.engine,
            Metrics::pct(head.all.recall_at_5, head.all.queries),
            head.all.mrr()
        ),
        serde_json::json!({
            "mode": "gold",
            "path": path,
            "k": k,
            "scoped": scope.is_some(),
            "embedding_coverage": coverage.map(|(t, e)| serde_json::json!({"total": t, "embedded": e})),
            "reports": &reports,
        }),
    );
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
        // 옛 인덱스는 파일명 공백을 하이픈으로 바꿨다. gold 파일이 그 표기로
        // 남아 있어도 매칭돼야 한다.
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
