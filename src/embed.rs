//! 임베딩 + 벡터 검색 — embeddinggemma-300M GGUF.
//!
//! - `kmd embed`: dirty가 아닌 활성 문서 중 임베딩 없는 것을 청크 단위로 임베딩
//! - 저장: SQLite `embeddings` 테이블 (BLOB f32-le), 브루트포스 코사인 검색
//!   (25k 문서 × 수 청크 × 768dim은 브루트포스로 수십 ms 수준)
//!
//! embeddinggemma 프롬프트 규약:
//!   문서: "title: {title} | text: {chunk}"
//!   쿼리: "task: search result | query: {q}"

use crate::bm25::SearchHit;
use crate::config;
use crate::store::Store;
use anyhow::{Context, Result, anyhow};
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use rusqlite::params;
use std::num::NonZeroU32;
use std::path::PathBuf;

const CHUNK_CHARS: usize = 2000; // ~500 tokens
const CHUNK_OVERLAP: usize = 200;
const CTX_TOKENS: u32 = 2048;

/// 한 문서에서 임베딩할 청크 수 상한.
///
/// 임베딩은 청크당 수십 ms의 GPU 연산이므로 비용이 문서 길이에 선형이고, 긴
/// 문서일수록 그 비용을 정당화하기 어렵다 — `exp` 컬렉션의 55MB짜리
/// `argocd-apps.json`은 약 27,500 청크로 몇 시간을 쓰는데, 덤프 JSON에서
/// 의미 검색이 건질 것은 거의 없다. 실측: 이 파일 하나가 전체 코퍼스 임베딩
/// 작업을 438건 남은 지점에서 멈춰 세웠다.
///
/// 상한을 넘는 문서는 **앞부분만** 임베딩한다. 건너뛰지 않는 이유는 문서
/// 앞머리가 대개 그 문서가 무엇인지 말해 주고(제목, 헤더, 서론), 그거라도
/// 있으면 의미 검색으로 도달할 수 있기 때문이다. BM25는 어차피 전문을
/// 인덱싱하므로 뒷부분도 키워드로는 찾힌다.
///
/// 64청크 ≈ 128KB. 이 코퍼스에서 이 선을 넘는 문서는 407건(전체의 0.7%)이고
/// 대부분 상태 덤프와 데이터셋이다.
const MAX_CHUNKS_PER_DOC: usize = 64;

pub fn model_path() -> PathBuf {
    if let Ok(p) = std::env::var("KMD_EMBED_MODEL") {
        return PathBuf::from(p);
    }
    crate::config::models_dir().join("hf_ggml-org_embeddinggemma-300M-Q8_0.gguf")
}

pub struct Embedder {
    backend: LlamaBackend,
    model: LlamaModel,
}

impl Embedder {
    pub fn load() -> Result<Self> {
        let path = model_path();
        if !path.exists() {
            return Err(anyhow!("embedding model not found: {}", path.display()));
        }
        let backend = LlamaBackend::init()?;
        let params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, &path, &params)
            .context("load embedding model")?;
        Ok(Embedder { backend, model })
    }

    /// 재사용 가능한 컨텍스트 생성 (모델당 1회 — 매 embed마다 만들면 매우 느림).
    pub fn make_context(&self) -> Result<llama_cpp_2::context::LlamaContext<'_>> {
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(CTX_TOKENS))
            .with_n_batch(CTX_TOKENS)
            .with_n_ubatch(CTX_TOKENS) // encoder는 n_ubatch >= n_tokens 필요
            .with_embeddings(true)
            .with_pooling_type(LlamaPoolingType::Mean);
        Ok(self.model.new_context(&self.backend, ctx_params)?)
    }

    /// 텍스트 하나를 임베딩. 반환은 L2-정규화된 f32 벡터.
    pub fn embed_with(
        &self,
        ctx: &mut llama_cpp_2::context::LlamaContext<'_>,
        text: &str,
    ) -> Result<Vec<f32>> {
        let mut tokens = self.model.str_to_token(text, AddBos::Always)?;
        tokens.truncate(CTX_TOKENS as usize - 8);

        ctx.clear_kv_cache();
        let mut batch = LlamaBatch::new(tokens.len(), 1);
        batch.add_sequence(&tokens, 0, false)?;
        ctx.encode(&mut batch)?;

        let emb = ctx.embeddings_seq_ith(0)?;
        Ok(l2_normalize(emb))
    }

    /// 단발 호출 편의 함수 (쿼리 임베딩용).
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut ctx = self.make_context()?;
        self.embed_with(&mut ctx, text)
    }
}

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

fn doc_prompt(title: &str, chunk: &str) -> String {
    format!("title: {} | text: {}", title, chunk)
}

fn query_prompt(q: &str) -> String {
    format!("task: search result | query: {}", q)
}

/// 문자 기반 청킹 (마크다운 빈 줄 경계 우선).
pub fn chunk_body(body: &str) -> Vec<(usize, String)> {
    if body.chars().count() <= CHUNK_CHARS {
        return vec![(0, body.to_string())];
    }
    let chars: Vec<char> = body.chars().collect();
    let mut chunks = Vec::new();
    let mut start = 0usize;
    let mut idx = 0usize;
    while start < chars.len() {
        let end = (start + CHUNK_CHARS).min(chars.len());
        // 빈 줄 경계로 스냅 (뒤쪽 400자 내에서)
        let mut cut = end;
        if end < chars.len() {
            let window_start = end.saturating_sub(400);
            let segment: String = chars[window_start..end].iter().collect();
            if let Some(pos) = segment.rfind("\n\n") {
                let candidate = window_start + segment[..pos].chars().count();
                if candidate > start + CHUNK_CHARS / 2 {
                    cut = candidate;
                }
            }
        }
        let text: String = chars[start..cut].iter().collect();
        chunks.push((idx, text));
        idx += 1;
        if cut >= chars.len() {
            break;
        }
        start = cut.saturating_sub(CHUNK_OVERLAP);
        // overlap으로 인한 무한루프 방지
        if start + CHUNK_CHARS / 2 < cut {
            start = cut;
        }
    }
    chunks
}

fn ensure_schema(store: &Store) -> Result<()> {
    store.conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS embeddings (
            doc_id    INTEGER NOT NULL,
            chunk_idx INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            vector    BLOB NOT NULL,
            PRIMARY KEY (doc_id, chunk_idx)
        );
        CREATE INDEX IF NOT EXISTS idx_embeddings_doc ON embeddings(doc_id);
        "#,
    )?;
    Ok(())
}

fn vec_to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn blob_to_vec(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// 활성 문서 중 임베딩이 없거나 오래된 것을 임베딩. 처리한 문서 수 반환.
///
/// `collections`가 비어 있지 않으면 그 컬렉션만 임베딩한다. 대상은 id 순으로
/// 처리되는데 `exp`가 대상의 95%를 차지하면서 id도 가장 작아, 필터 없이는
/// 훅이 실제로 읽는 컬렉션(learnings/claude-*)이 맨 뒤로 밀린다.
pub fn embed_pending(
    store: &mut Store,
    limit: Option<usize>,
    collections: &[String],
) -> Result<usize> {
    ensure_schema(store)?;

    // 임베딩 없는 활성 문서 목록.
    //
    // 본문 미저장 문서(abspath IS NOT NULL — project 축)는 제외한다. 코드 38만 건
    // 임베딩은 비용이 비현실적이고, 코드 검색은 식별자 정확 매칭이 지배적이라
    // BM25로 충분하다. 벡터 검색은 knowledge/session 축에만 적용된다.
    let mut sql = String::from(
        "SELECT d.id, d.title, d.body FROM documents d
         WHERE d.active = 1
           AND d.abspath IS NULL
           AND NOT EXISTS (SELECT 1 FROM embeddings e WHERE e.doc_id = d.id)",
    );
    if !collections.is_empty() {
        let placeholders = vec!["?"; collections.len()].join(",");
        sql.push_str(&format!(" AND d.collection IN ({})", placeholders));
    }
    sql.push_str(" ORDER BY d.id");
    let mut stmt = store.conn.prepare(&sql)?;
    let params: Vec<&dyn rusqlite::ToSql> =
        collections.iter().map(|c| c as &dyn rusqlite::ToSql).collect();
    let docs: Vec<(i64, String, String)> = stmt
        .query_map(params.as_slice(), |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect::<std::result::Result<_, _>>()?;
    drop(stmt);

    // 비활성 문서의 임베딩 정리
    store.conn.execute(
        "DELETE FROM embeddings WHERE doc_id IN (SELECT id FROM documents WHERE active = 0)",
        [],
    )?;
    // 내용이 바뀐(dirty였다가 재인덱싱된) 문서는 update 시점에 임베딩도 지워야 하지만,
    // 단순화: hash 변경 시 upsert가 dirty=1로 만들고, 여기서는 재임베딩 대상으로 잡히도록
    // scan 쪽에서 embeddings를 지운다 (아래 purge_stale 참고).

    let total = docs.len();
    let todo = limit.map(|l| l.min(total)).unwrap_or(total);
    if todo == 0 {
        return Ok(0);
    }

    eprintln!("embedding {} documents (of {} pending) ...", todo, total);
    let embedder = Embedder::load()?;
    let mut ctx = embedder.make_context()?;

    let mut done = 0usize;
    let mut truncated = 0usize;
    for (id, title, body) in docs.into_iter().take(todo) {
        let mut chunks = chunk_body(&body);
        if chunks.len() > MAX_CHUNKS_PER_DOC {
            truncated += 1;
            chunks.truncate(MAX_CHUNKS_PER_DOC);
        }

        // 임베딩은 트랜잭션 **밖에서** 계산한다. 문서당 수백 ms가 걸리는 GPU
        // 연산이라, 트랜잭션 안에서 돌리면 그 시간 내내 쓰기 락을 물고 있게 되고
        // 데몬의 주기적 embed 사이클이나 다른 kmd 프로세스가 그 락에 걸린다.
        let vectors: Vec<(i64, &str, Vec<f32>)> = chunks
            .iter()
            .map(|(idx, text)| {
                embedder
                    .embed_with(&mut ctx, &doc_prompt(&title, text))
                    .map(|v| (*idx as i64, text.as_str(), v))
            })
            .collect::<Result<_>>()?;

        // IMMEDIATE로 시작한다. DEFERRED(=`unchecked_transaction`의 기본)는 읽기로
        // 시작해 쓰기로 승격하는데, SQLite는 그 승격에서만 `busy_timeout`을
        // 무시하고 즉시 SQLITE_BUSY를 반환한다(데드락 회피). 30초 타임아웃을
        // 걸어 두고도 동시 실행이 "database is locked"로 즉사한 실제 원인이다.
        let tx = store
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        for (idx, text, v) in &vectors {
            tx.execute(
                "INSERT OR REPLACE INTO embeddings (doc_id, chunk_idx, chunk_text, vector) VALUES (?1,?2,?3,?4)",
                params![id, idx, text, vec_to_blob(v)],
            )?;
        }
        tx.commit()?;
        done += 1;
        if done % 50 == 0 {
            eprintln!("  {}/{}", done, todo);
        }
    }
    // 절단은 조용히 넘기지 않는다. 그 문서들은 뒷부분이 벡터 검색에 없으므로,
    // "임베딩 완료"가 곧 "전문이 의미 검색 대상"을 뜻하지 않는다.
    if truncated > 0 {
        eprintln!(
            "  {} document(s) exceeded {} chunks and were embedded head-only",
            truncated, MAX_CHUNKS_PER_DOC
        );
    }
    Ok(done)
}

/// 내용이 바뀐 문서의 구식 임베딩 제거 (scan 후 호출).
pub fn purge_stale(store: &Store, dirty_ids: &[i64]) -> Result<()> {
    ensure_schema(store)?;
    let mut stmt = store
        .conn
        .prepare("DELETE FROM embeddings WHERE doc_id = ?1")?;
    for id in dirty_ids {
        stmt.execute(params![id])?;
    }
    Ok(())
}

/// 벡터 검색 — 쿼리 임베딩 후 브루트포스 코사인 top-k.
pub fn vsearch(store: &Store, query: &str, limit: usize) -> Result<Vec<SearchHit>> {
    vsearch_in(store, query, limit, None)
}

/// 컬렉션 집합으로 좁힌 벡터 검색. `collections`가 None이면 전 인덱스.
///
/// BM25 쪽 `search_in`과 짝을 이룬다. 훅과 eval은 특정 컬렉션만 보는데, 필터가
/// 없으면 두 엔진이 서로 다른 모집단을 검색해 비교가 성립하지 않는다.
pub fn vsearch_in(
    store: &Store,
    query: &str,
    limit: usize,
    collections: Option<&[&str]>,
) -> Result<Vec<SearchHit>> {
    ensure_schema(store)?;
    // KMD_VSEARCH_TIMING=1이면 단계별 시간을 stderr에 낸다. 훅 지연을 줄이려면
    // 모델 로드와 브루트포스 스캔 중 어느 쪽이 지배적인지 알아야 하고, 둘은
    // 대응이 완전히 다르다(데몬 상주 대 인덱스 구조 변경).
    let timing = std::env::var("KMD_VSEARCH_TIMING").is_ok();
    let t0 = std::time::Instant::now();
    let embedder = Embedder::load()?;
    let qv = embedder.embed(&query_prompt(query))?;
    let t_model = t0.elapsed();

    // 스코어링 패스는 doc_id, chunk_idx, vector만 읽는다.
    //
    // 코사인 점수에 필요한 것은 벡터뿐이고, chunk_text와 문서 메타는 top-k가
    // 정해진 뒤 그 몇 건에만 필요하다. 그런데 이 쿼리는 매번 임베딩 테이블
    // 전체를 훑으므로(브루트포스), 함께 select하면 그 바이트를 전부 읽는다 —
    // 실측 이 코퍼스에서 vector 866MB 대 chunk_text 488MB로, 텍스트를 빼면
    // 읽는 양이 36% 줄어든다. 페이지 캐시가 식은 콜드 경로에서 이 차이가 크다.
    let mut sql = String::from(
        "SELECT e.doc_id, e.chunk_idx, e.vector
         FROM embeddings e JOIN documents d ON d.id = e.doc_id
         WHERE d.active = 1",
    );
    let filter: Vec<String> = collections
        .map(|c| c.iter().map(|s| s.to_string()).collect())
        .unwrap_or_default();
    if !filter.is_empty() {
        let placeholders = vec!["?"; filter.len()].join(",");
        sql.push_str(&format!(" AND d.collection IN ({})", placeholders));
    }
    let mut stmt = store.conn.prepare(&sql)?;
    let params: Vec<&dyn rusqlite::ToSql> =
        filter.iter().map(|c| c as &dyn rusqlite::ToSql).collect();
    let rows = stmt.query_map(params.as_slice(), |r| {
        Ok((
            r.get::<_, i64>(0)?,
            r.get::<_, i64>(1)?,
            r.get::<_, Vec<u8>>(2)?,
        ))
    })?;

    // 문서별 최고 청크 점수와 그 청크 번호만 유지. 청크 번호를 들고 가는 이유는
    // 스니펫을 뒤에서 그 청크로 되찾아야 하기 때문이다.
    use std::collections::HashMap;
    let mut best: HashMap<i64, (f32, i64)> = HashMap::new();
    for row in rows {
        let (doc_id, idx, blob) = row?;
        let v = blob_to_vec(&blob);
        let score: f32 = qv.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
        let entry = best.entry(doc_id).or_insert((f32::MIN, 0));
        if score > entry.0 {
            *entry = (score, idx);
        }
    }

    let t_scan = t0.elapsed() - t_model;
    let mut scored: Vec<_> = best.into_iter().collect();
    // 코사인 점수 내림차순, 동점은 doc_id로 결정적으로 깬다. `best`가 HashMap이고
    // `sort_by`가 안정 정렬이라, 타이브레이크가 없으면 같은 점수 문서의 순서가
    // 프로세스별 해시 시드를 따라 실행마다 바뀐다 — 실측: 같은 쿼리의 3위가
    // 실행마다 다른 파일로 나왔다. f32 코사인은 동점이 흔하다.
    scored.sort_by(|a, b| {
        b.1.0
            .partial_cmp(&a.1.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scored.truncate(limit);

    // top-k가 정해진 뒤에야 그 몇 건의 텍스트와 메타를 읽는다.
    let cfg = config::load()?;
    let mut detail = store.conn.prepare(
        "SELECT e.chunk_text, d.collection, d.relpath, d.title, d.context, d.abspath
         FROM embeddings e JOIN documents d ON d.id = e.doc_id
         WHERE e.doc_id = ?1 AND e.chunk_idx = ?2",
    )?;
    let mut out = Vec::with_capacity(scored.len());
    for (doc_id, (score, chunk_idx)) in scored {
        let row = detail.query_row(params![doc_id, chunk_idx], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, Option<String>>(4)?,
                r.get::<_, Option<String>>(5)?,
            ))
        });
        // 스코어링과 이 조회 사이에 문서가 지워졌으면(동시 update) 그 히트는 버린다.
        let Ok((text, coll, relpath, title, context, abspath)) = row else {
            continue;
        };
        out.push(SearchHit {
            docid: format!("#{:06x}", doc_id),
            score,
            file: format!("kmd://{}/{}", coll, relpath),
            abspath: crate::bm25::resolve_abspath(
                &cfg,
                abspath.as_deref().unwrap_or(""),
                &coll,
                &relpath,
            ),
            title,
            context,
            snippet: Some(text.chars().take(300).collect()),
        });
    }
    if timing {
        eprintln!(
            "vsearch timing: model+query-embed {:?}, brute-force scan {:?}, detail {:?}",
            t_model,
            t_scan,
            t0.elapsed() - t_model - t_scan
        );
    }
    Ok(out)
}

/// hybrid: BM25 + 벡터를 RRF(reciprocal rank fusion)로 융합.
pub fn hybrid(
    store: &Store,
    tantivy_dir: &std::path::Path,
    cfg: &config::IndexConfig,
    query: &str,
    limit: usize,
) -> Result<Vec<SearchHit>> {
    hybrid_in(store, tantivy_dir, cfg, query, limit, None)
}

/// 컬렉션 집합으로 좁힌 hybrid.
pub fn hybrid_in(
    store: &Store,
    tantivy_dir: &std::path::Path,
    cfg: &config::IndexConfig,
    query: &str,
    limit: usize,
    collections: Option<&[&str]>,
) -> Result<Vec<SearchHit>> {
    const K: f32 = 60.0;
    // 후보를 최종 limit보다 훨씬 넓게 가져온다. RRF는 두 리스트를 대칭으로
    // 합산하므로, 후보창이 좁으면 양쪽에 흔하게 걸친 무관 문서가 한쪽에만
    // 있는 정답을 이긴다 — limit*2(=10)에서는 양쪽 10위 문서(2/(60+10)=0.029)가
    // 벡터 1위 정답(1/61=0.016)보다 높다. 창을 넓히면 정답의 순위가 상대적으로
    // 앞서므로 그 역전이 줄어든다.
    // 스윕 가능하게 환경변수로 뺀다. 기본값은 측정으로 정한다 — 후보창을
    // 넓히면 gold(패러프레이즈)가 좋아지고 known-item(어휘 완전 일치)이
    // 나빠지는 트레이드오프가 있어서, 한쪽만 보고 정하면 다른 쪽이 회귀한다.
    const CANDIDATE_FACTOR: usize = 8;
    let factor = std::env::var("KMD_RRF_FACTOR")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n >= 1)
        .unwrap_or(CANDIDATE_FACTOR);
    let pool = limit * factor;
    let bm = crate::bm25::search_in(tantivy_dir, cfg, query, pool, collections)
        .unwrap_or_default();
    let vs = vsearch_in(store, query, pool, collections).unwrap_or_default();

    Ok(fuse_rrf(bm, vs, limit, K))
}

/// 두 랭킹을 RRF로 융합한다. 순수 함수 — 모델도 인덱스도 필요 없으므로
/// 테스트할 수 있다.
///
/// RRF는 두 리스트를 **대칭으로** 합산한다. 그래서 한쪽에만 있는 정답은
/// 기여를 한 번 받고, 양쪽에 흔하게 걸친 무관 문서는 두 번 받는다. 후보창이
/// 좁으면 이 산술이 뒤집힌다 — `pool=10`에서 양쪽 10위 문서(2/(60+10)=0.029)가
/// 벡터 1위 정답(1/61=0.016)을 이긴다. 창을 넓히는 것이 호출부의 대응이다.
fn fuse_rrf(
    bm: Vec<SearchHit>,
    vs: Vec<SearchHit>,
    limit: usize,
    k: f32,
) -> Vec<SearchHit> {
    use std::collections::HashMap;
    let mut fused: HashMap<String, (f32, SearchHit)> = HashMap::new();
    for list in [bm, vs] {
        for (rank, hit) in list.into_iter().enumerate() {
            let rr = 1.0 / (k + rank as f32 + 1.0);
            fused
                .entry(hit.file.clone())
                .and_modify(|e| e.0 += rr)
                .or_insert((rr, hit));
        }
    }
    let mut out: Vec<_> = fused.into_values().collect();
    // RRF 점수 내림차순, 동점은 파일 경로로 결정적으로 깬다.
    //
    // 동점 타이브레이크가 없으면 순위가 실행마다 바뀐다: `fused`는 HashMap이고
    // Rust의 해시는 프로세스마다 시드가 다르므로 순회 순서가 매번 다르며,
    // `sort_by`는 안정 정렬이라 동점 문서의 상대 순서가 그 순회 순서를 그대로
    // 물려받는다. 후보창을 넓히면 동점(양쪽 리스트에서 대칭 위치에 있는 문서)이
    // 많아져 이 비결정성이 지표에까지 드러난다 — 실측: 같은 입력에 대해 gold
    // R@5가 52%와 56% 사이에서, MRR이 0.369~0.380 사이에서 흔들렸다.
    //
    // 순위가 흔들리는 검색은 평가할 수 없다. 어느 순서가 "옳은지"는 정의되지
    // 않으므로 안정성만 확보한다.
    out.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.file.cmp(&b.1.file))
    });
    out.into_iter()
        .take(limit)
        .map(|(rrf, mut hit)| {
            hit.score = rrf;
            hit
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(file: &str) -> SearchHit {
        SearchHit {
            file: file.into(),
            ..Default::default()
        }
    }

    /// 동점 순위가 실행마다 흔들리면 안 된다.
    ///
    /// `fused`는 HashMap이고 Rust의 해시는 프로세스마다 시드가 다르다.
    /// `sort_by`는 안정 정렬이라, 타이브레이크가 없으면 동점 문서의 상대 순서가
    /// 그 순회 순서를 그대로 물려받는다. 이 테스트는 한 프로세스 안에서 도는
    /// 탓에 시드가 고정돼 있어 그 자체로 재현하지 못하므로, 대신 **동점 집합의
    /// 순서가 정해진 규칙(파일 경로)을 따르는지**를 고정한다.
    #[test]
    fn ties_break_deterministically_by_path() {
        // b와 c는 (1위, 3위)와 (3위, 1위)로 정확히 같은 점수를 받는다.
        // a는 (2위, 2위)라 근소하게 낮다 — 1/61+1/63 > 2/62. 볼록성 때문이고,
        // 여기서 확인하려는 것은 그 서열이 아니라 **동점 b/c의 순서**다.
        let bm = vec![hit("kmd://c/b.md"), hit("kmd://c/a.md"), hit("kmd://c/c.md")];
        let vs = vec![hit("kmd://c/c.md"), hit("kmd://c/a.md"), hit("kmd://c/b.md")];
        let out = fuse_rrf(bm, vs, 3, 60.0);
        let files: Vec<&str> = out.iter().map(|h| h.file.as_str()).collect();
        assert_eq!(
            &files[..2],
            &["kmd://c/b.md", "kmd://c/c.md"],
            "tied documents are ordered by path, not by hash iteration order"
        );
        assert_eq!(files[2], "kmd://c/a.md", "a scores lowest by RRF convexity");
    }

    /// 한쪽에만 있는 1위 정답이 양쪽에 걸친 하위 문서에게 밀리는 조건을 고정한다.
    ///
    /// 이것이 RRF의 대칭 합산이 만드는 실제 실패이며, 후보창을 넓히는 이유다.
    /// 좁은 창에서 역전이 일어남을 명시적으로 남겨 두면, 나중에 창을 다시
    /// 좁히려는 변경이 무엇을 되돌리는지 알 수 있다.
    #[test]
    fn a_single_list_winner_loses_to_a_document_in_both_lists() {
        let answer = "kmd://c/answer.md";
        let filler: Vec<SearchHit> = (0..10)
            .map(|i| hit(&format!("kmd://c/f{:02}.md", i)))
            .collect();
        // 벡터는 정답을 1위로 올린다. BM25는 정답을 아예 못 찾는다.
        let mut vs = vec![hit(answer)];
        vs.extend(filler.iter().map(|h| hit(&h.file)));
        let bm: Vec<SearchHit> = filler.iter().map(|h| hit(&h.file)).collect();

        let out = fuse_rrf(bm, vs, 5, 60.0);
        let files: Vec<&str> = out.iter().map(|h| h.file.as_str()).collect();
        assert_ne!(
            files[0], answer,
            "symmetric RRF puts documents present in both lists above a              single-list winner — this is the failure the candidate window              width mitigates"
        );
        assert!(
            files.contains(&answer) == false,
            "with 10 both-list fillers the answer is pushed out of the top 5              entirely: {:?}",
            files
        );
    }

    /// 한쪽 리스트가 비어도 나머지가 그대로 나온다.
    #[test]
    fn an_empty_list_does_not_erase_the_other() {
        let out = fuse_rrf(vec![], vec![hit("kmd://c/only.md")], 5, 60.0);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].file, "kmd://c/only.md");
    }

    /// 같은 문서가 양쪽에 있으면 점수가 합산되고 히트는 하나만 남는다.
    #[test]
    fn a_document_in_both_lists_appears_once_with_summed_score() {
        let out = fuse_rrf(
            vec![hit("kmd://c/x.md")],
            vec![hit("kmd://c/x.md")],
            5,
            60.0,
        );
        assert_eq!(out.len(), 1, "no duplicate entry for the same file");
        let expected = 2.0 / 61.0;
        assert!(
            (out[0].score - expected).abs() < 1e-6,
            "score {} is not the sum of both contributions ({})",
            out[0].score,
            expected
        );
    }
}

pub fn embedding_counts(store: &Store) -> Result<(i64, i64)> {
    ensure_schema(store)?;
    let vectors: i64 = store
        .conn
        .query_row("SELECT COUNT(*) FROM embeddings", [], |r| r.get(0))?;
    // `abspath IS NULL`은 embed_pending의 대상 조건과 같아야 한다. 없으면 임베딩
    // 대상이 아닌 project 축 코드 36만 건이 "pending"으로 보고된다.
    let pending: i64 = store.conn.query_row(
        "SELECT COUNT(*) FROM documents d WHERE d.active = 1
           AND d.abspath IS NULL
           AND NOT EXISTS (SELECT 1 FROM embeddings e WHERE e.doc_id = d.id)",
        [],
        |r| r.get(0),
    )?;
    Ok((vectors, pending))
}
