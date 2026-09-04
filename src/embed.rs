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
    for (id, title, body) in docs.into_iter().take(todo) {
        let chunks = chunk_body(&body);
        let tx = store.conn.unchecked_transaction()?;
        for (idx, text) in &chunks {
            let v = embedder.embed_with(&mut ctx, &doc_prompt(&title, text))?;
            tx.execute(
                "INSERT OR REPLACE INTO embeddings (doc_id, chunk_idx, chunk_text, vector) VALUES (?1,?2,?3,?4)",
                params![id, *idx as i64, text, vec_to_blob(&v)],
            )?;
        }
        tx.commit()?;
        done += 1;
        if done % 50 == 0 {
            eprintln!("  {}/{}", done, todo);
        }
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
    let embedder = Embedder::load()?;
    let qv = embedder.embed(&query_prompt(query))?;

    let mut sql = String::from(
        "SELECT e.doc_id, e.chunk_idx, e.chunk_text, e.vector,
                d.collection, d.relpath, d.title, d.context
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
            r.get::<_, String>(2)?,
            r.get::<_, Vec<u8>>(3)?,
            r.get::<_, String>(4)?,
            r.get::<_, String>(5)?,
            r.get::<_, String>(6)?,
            r.get::<_, Option<String>>(7)?,
        ))
    })?;

    // 문서별 최고 청크 점수만 유지
    use std::collections::HashMap;
    let mut best: HashMap<i64, (f32, String, String, String, String, Option<String>)> =
        HashMap::new();
    for row in rows {
        let (doc_id, _idx, text, blob, coll, relpath, title, context) = row?;
        let v = blob_to_vec(&blob);
        let score: f32 = qv.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
        let entry = best.entry(doc_id).or_insert_with(|| {
            (f32::MIN, String::new(), coll.clone(), relpath.clone(), title.clone(), context.clone())
        });
        if score > entry.0 {
            *entry = (score, text, coll, relpath, title, context);
        }
    }

    let mut scored: Vec<_> = best.into_iter().collect();
    scored.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(limit);

    Ok(scored
        .into_iter()
        .map(|(doc_id, (score, text, coll, relpath, title, context))| SearchHit {
            docid: format!("#{:06x}", doc_id),
            score,
            file: format!("kmd://{}/{}", coll, relpath),
            title,
            context,
            snippet: Some(text.chars().take(300).collect()),
        })
        .collect())
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
    let bm = crate::bm25::search_in(tantivy_dir, cfg, query, limit * 2, collections)
        .unwrap_or_default();
    let vs = vsearch_in(store, query, limit * 2, collections).unwrap_or_default();

    use std::collections::HashMap;
    let mut fused: HashMap<String, (f32, SearchHit)> = HashMap::new();
    for (rank, hit) in bm.into_iter().enumerate() {
        let rr = 1.0 / (K + rank as f32 + 1.0);
        fused
            .entry(hit.file.clone())
            .and_modify(|e| e.0 += rr)
            .or_insert((rr, hit));
    }
    for (rank, hit) in vs.into_iter().enumerate() {
        let rr = 1.0 / (K + rank as f32 + 1.0);
        fused
            .entry(hit.file.clone())
            .and_modify(|e| e.0 += rr)
            .or_insert((rr, hit));
    }
    let mut out: Vec<_> = fused.into_values().collect();
    out.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(out
        .into_iter()
        .take(limit)
        .map(|(rrf, mut hit)| {
            hit.score = rrf;
            hit
        })
        .collect())
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
