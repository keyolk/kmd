//! tantivy BM25 인덱스 — 한글(lindera)/영어(stemmer) 이중 필드.

use crate::config::IndexConfig;
use crate::store::Store;
use crate::tokenize;
use anyhow::Result;
use serde::Serialize;
use std::path::Path;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{
    FAST, Field, INDEXED, IndexRecordOption, STORED, STRING, Schema, TextFieldIndexing,
    TextOptions, Value,
};
use tantivy::{Index, IndexWriter, TantivyDocument, Term, doc};

#[derive(Debug, Serialize)]
pub struct SearchHit {
    pub docid: String,
    pub score: f32,
    /// kmd URI: kmd://collection/relpath
    pub file: String,
    pub title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
}

struct Fields {
    doc_id: Field,
    collection: Field,
    relpath: Field,
    title: Field,
    context: Field,
    body_stored: Field,
    title_ko: Field,
    title_en: Field,
    body_ko: Field,
    body_en: Field,
}

fn build_schema() -> (Schema, Fields) {
    let mut b = Schema::builder();

    let ko_indexing = TextFieldIndexing::default()
        .set_tokenizer(tokenize::KO_TOKENIZER)
        .set_index_option(IndexRecordOption::WithFreqsAndPositions);
    let en_indexing = TextFieldIndexing::default()
        .set_tokenizer(tokenize::EN_TOKENIZER)
        .set_index_option(IndexRecordOption::WithFreqsAndPositions);
    let ko_opts = TextOptions::default().set_indexing_options(ko_indexing);
    let en_opts = TextOptions::default().set_indexing_options(en_indexing);

    let fields = Fields {
        doc_id: b.add_i64_field("doc_id", INDEXED | STORED | FAST),
        collection: b.add_text_field("collection", STRING | STORED),
        relpath: b.add_text_field("relpath", STORED),
        title: b.add_text_field("title", STORED),
        context: b.add_text_field("context", STORED),
        body_stored: b.add_text_field("body_stored", STORED),
        title_ko: b.add_text_field("title_ko", ko_opts.clone()),
        title_en: b.add_text_field("title_en", en_opts.clone()),
        body_ko: b.add_text_field("body_ko", ko_opts),
        body_en: b.add_text_field("body_en", en_opts),
    };
    (b.build(), fields)
}

fn open_or_create(dir: &Path) -> Result<(Index, Fields)> {
    let (schema, fields) = build_schema();
    std::fs::create_dir_all(dir)?;
    let index = match Index::open_in_dir(dir) {
        Ok(idx) => idx,
        Err(_) => Index::create_in_dir(dir, schema.clone())?,
    };
    tokenize::register(&index)?;
    Ok((index, fields))
}

/// dirty 문서를 tantivy에 반영(삭제 후 재삽입). 인덱싱된 문서 수 반환.
pub fn index_dirty(dir: &Path, store: &mut Store) -> Result<usize> {
    let dirty = store.dirty_docs()?;
    if dirty.is_empty() {
        return Ok(0);
    }
    let (index, f) = open_or_create(dir)?;
    let mut writer: IndexWriter = index.writer(256_000_000)?;

    let mut ids = Vec::with_capacity(dirty.len());
    let mut indexed = 0usize;
    for d in &dirty {
        writer.delete_term(Term::from_field_i64(f.doc_id, d.id));
        if store.is_active(d.id)? {
            writer.add_document(doc!(
                f.doc_id => d.id,
                f.collection => d.collection.clone(),
                f.relpath => d.relpath.clone(),
                f.title => d.title.clone(),
                f.context => d.context.clone().unwrap_or_default(),
                f.body_stored => d.body.clone(),
                f.title_ko => d.title.clone(),
                f.title_en => d.title.clone(),
                f.body_ko => d.body.clone(),
                f.body_en => d.body.clone(),
            ))?;
            indexed += 1;
        }
        ids.push(d.id);
    }
    writer.commit()?;
    store.clear_dirty(&ids)?;
    Ok(indexed)
}

pub fn search(
    dir: &Path,
    _cfg: &IndexConfig,
    query: &str,
    limit: usize,
    collection: Option<&str>,
) -> Result<Vec<SearchHit>> {
    let (index, f) = open_or_create(dir)?;
    let reader = index.reader()?;
    let searcher = reader.searcher();

    let mut parser = QueryParser::for_index(
        &index,
        vec![f.title_ko, f.title_en, f.body_ko, f.body_en],
    );
    parser.set_field_boost(f.title_ko, 2.0);
    parser.set_field_boost(f.title_en, 2.0);
    // 사용자 쿼리는 구문이 아니라 키워드 집합 — lenient 파싱
    let (parsed, _errors) = parser.parse_query_lenient(query);

    // 컬렉션 필터가 있으면 과잉 수집 후 필터링 (규모상 충분)
    let fetch = if collection.is_some() { limit * 20 } else { limit };
    let top = searcher.search(&parsed, &TopDocs::with_limit(fetch.max(limit)))?;

    let mut hits = Vec::new();
    for (score, addr) in top {
        let retrieved: TantivyDocument = searcher.doc(addr)?;
        let get_str = |field: Field| -> String {
            retrieved
                .get_first(field)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string()
        };
        let coll = get_str(f.collection);
        if let Some(want) = collection {
            if coll != want {
                continue;
            }
        }
        let doc_id = retrieved
            .get_first(f.doc_id)
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let body = get_str(f.body_stored);
        let context = {
            let c = get_str(f.context);
            if c.is_empty() { None } else { Some(c) }
        };
        hits.push(SearchHit {
            docid: format!("#{:06x}", doc_id),
            score,
            file: format!("kmd://{}/{}", coll, get_str(f.relpath)),
            title: get_str(f.title),
            context,
            snippet: Some(make_snippet(&body, query, 300)),
        });
        if hits.len() >= limit {
            break;
        }
    }
    Ok(hits)
}

/// 쿼리 토큰이 처음 등장하는 주변 ±window 문자를 스니펫으로.
/// 형태소 변형으로 못 찾으면 문서 앞부분을 사용.
fn make_snippet(body: &str, query: &str, max_len: usize) -> String {
    let lower_body = body.to_lowercase();
    let pos = query
        .split_whitespace()
        .filter(|t| t.chars().count() >= 2)
        .filter_map(|t| lower_body.find(&t.to_lowercase()))
        .min();

    let center = pos.unwrap_or(0);
    // char 경계로 스니펫 범위 계산
    let chars: Vec<(usize, char)> = body.char_indices().collect();
    if chars.is_empty() {
        return String::new();
    }
    let center_idx = chars
        .binary_search_by_key(&center, |(i, _)| *i)
        .unwrap_or_else(|i| i.min(chars.len() - 1));
    let half = max_len / 2;
    let start = center_idx.saturating_sub(half);
    let end = (center_idx + half).min(chars.len());
    let s: String = chars[start..end].iter().map(|(_, c)| c).collect();
    let mut out = s.trim().to_string();
    if start > 0 {
        out = format!("…{}", out);
    }
    if end < chars.len() {
        out.push('…');
    }
    out
}
