//! tantivy BM25 인덱스 — 한글(lindera)/영어(stemmer) 이중 필드.

use crate::config::IndexConfig;
use crate::store::Store;
use crate::tokenize;
use anyhow::Result;
use serde::Serialize;
use std::path::Path;
use tantivy::collector::TopDocs;
use tantivy::query::{BooleanQuery, Occur, Query, QueryParser, TermQuery};
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
    /// 읽을 수 있는 절대경로. `file`은 컬렉션 상대 URI라, 그것만으로 파일을
    /// 열려면 호출자가 index.yml에서 컬렉션 `path`를 찾아 조인해야 한다.
    /// 검색은 이미 그 경로를 알고 있으므로(본문 미저장 문서는 스니펫을 거기서
    /// 읽는다) 그냥 실어 보낸다 — 에이전트가 히트를 바로 Read할 수 있다.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub abspath: Option<String>,
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
    abspath: Field,
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
        abspath: b.add_text_field("abspath", STORED),
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
            // 본문 미저장 문서는 원본에서 읽는다. 파일이 사라졌으면 색인만 건너뛰고
            // dirty는 해제한다 — 다음 스캔이 deactivate_missing으로 정리한다.
            if let Some(body) = crate::store::body_of(d) {
                // abspath가 있으면 body_stored를 비워 인덱스 크기를 아낀다.
                // 스니펫은 검색 시점에 원본에서 읽는다.
                let stored = if d.abspath.is_some() { "" } else { body.as_str() };
                writer.add_document(doc!(
                    f.doc_id => d.id,
                    f.collection => d.collection.clone(),
                    f.relpath => d.relpath.clone(),
                    f.title => d.title.clone(),
                    f.context => d.context.clone().unwrap_or_default(),
                    f.body_stored => stored,
                    f.abspath => d.abspath.clone().unwrap_or_default(),
                    f.title_ko => d.title.clone(),
                    f.title_en => d.title.clone(),
                    f.body_ko => body.clone(),
                    f.body_en => body.clone(),
                ))?;
                indexed += 1;
            }
        }
        ids.push(d.id);
    }
    writer.commit()?;
    store.clear_dirty(&ids)?;
    Ok(indexed)
}

pub fn search(
    dir: &Path,
    cfg: &IndexConfig,
    query: &str,
    limit: usize,
    collection: Option<&str>,
) -> Result<Vec<SearchHit>> {
    let filter = collection.map(|c| std::slice::from_ref(&c).to_vec());
    search_in(dir, cfg, query, limit, filter.as_deref())
}

/// 컬렉션 집합으로 좁힌 검색. `collections`가 None이면 전 인덱스.
///
/// `kmd global`이 한 축(여러 컬렉션)을 한 번에 질의하려고 쓴다.
pub fn search_in(
    dir: &Path,
    cfg: &IndexConfig,
    query: &str,
    limit: usize,
    collections: Option<&[&str]>,
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

    // 컬렉션 필터는 질의에 넣는다. 과잉 수집 후 후처리로 거르면, 인덱스가 커질수록
    // 큰 컬렉션이 상위를 모두 차지해 작은 컬렉션이 한 건도 살아남지 못한다
    // (project 36만 건 추가 후 실제로 learnings가 통째로 사라졌다).
    let final_query: Box<dyn Query> = match collections {
        Some(want) if !want.is_empty() => {
            let mut clauses: Vec<(Occur, Box<dyn Query>)> =
                vec![(Occur::Must, Box::new(BooleanQuery::from(vec![(
                    Occur::Must,
                    parsed,
                )])))];
            let coll_clauses: Vec<(Occur, Box<dyn Query>)> = want
                .iter()
                .map(|c| {
                    let term = Term::from_field_text(f.collection, c);
                    let q: Box<dyn Query> =
                        Box::new(TermQuery::new(term, IndexRecordOption::Basic));
                    (Occur::Should, q)
                })
                .collect();
            clauses.push((Occur::Must, Box::new(BooleanQuery::new(coll_clauses))));
            Box::new(BooleanQuery::new(clauses))
        }
        _ => parsed,
    };

    let top = searcher.search(&final_query, &TopDocs::with_limit(limit))?;

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
        if let Some(want) = collections
            && !want.contains(&coll.as_str())
        {
            continue;
        }
        let doc_id = retrieved
            .get_first(f.doc_id)
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let abspath = get_str(f.abspath);
        // 본문 미저장 문서는 스니펫을 원본에서 읽는다. 파일이 사라졌으면(브랜치
        // 전환, 삭제) 인덱스가 워킹트리보다 최신인 것이므로 그 hit은 버린다.
        let body = if abspath.is_empty() {
            get_str(f.body_stored)
        } else {
            match std::fs::read_to_string(&abspath) {
                Ok(b) => b,
                Err(_) => continue,
            }
        };
        let context = {
            let c = get_str(f.context);
            if c.is_empty() { None } else { Some(c) }
        };
        let relpath = get_str(f.relpath);
        hits.push(SearchHit {
            docid: format!("#{:06x}", doc_id),
            score,
            file: format!("kmd://{}/{}", coll, relpath),
            // 본문 미저장 문서는 인덱스가 절대경로를 들고 있다. 본문 저장
            // 문서는 컬렉션 루트 + relpath로 만든다.
            abspath: resolve_abspath(cfg, &abspath, &coll, &relpath),
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

/// 히트의 읽을 수 있는 절대경로.
///
/// 인덱스에 abspath가 있으면(본문 미저장 컬렉션) 그것이 진실이다. 없으면
/// 컬렉션 루트에 relpath를 붙인다. 설정에 없는 컬렉션(이름이 바뀐 뒤 남은
/// 인덱스)이면 None — 추측한 경로를 주는 것보다 없다고 말하는 편이 낫다.
pub(crate) fn resolve_abspath(
    cfg: &IndexConfig,
    indexed: &str,
    collection: &str,
    relpath: &str,
) -> Option<String> {
    if !indexed.is_empty() {
        return Some(indexed.to_string());
    }
    let root = &cfg.collections.get(collection)?.path;
    Some(root.join(relpath).to_string_lossy().into_owned())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Store;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static SEQ: AtomicUsize = AtomicUsize::new(0);

    struct TmpDir(std::path::PathBuf);

    impl TmpDir {
        fn new(tag: &str) -> Self {
            let n = SEQ.fetch_add(1, Ordering::SeqCst);
            let p = std::env::temp_dir().join(format!(
                "kmd-bm25-test-{}-{}-{}",
                std::process::id(),
                tag,
                n
            ));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).expect("mkdir temp");
            TmpDir(p)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TmpDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn empty_cfg() -> IndexConfig {
        serde_yaml::from_str("collections: {}").unwrap()
    }

    /// 한 컬렉션에 `count`개의 문서를 넣는다. 본문은 모두 같은 단어를 담아
    /// 어느 컬렉션이든 질의에 매칭되게 한다.
    fn seed(store: &mut Store, collection: &str, count: usize, word: &str) {
        for i in 0..count {
            store
                .upsert_doc(
                    collection,
                    &format!("doc{}.md", i),
                    &format!("{} {}", collection, i),
                    &format!("{} body {}", word, i),
                    None,
                    0,
                    0,
                    &format!("hash-{}-{}", collection, i),
                    None,
                )
                .unwrap();
        }
    }

    /// 큰 컬렉션이 상위를 독식해도 작은 컬렉션이 자기 몫을 받아야 한다.
    ///
    /// 회귀 가드: 예전에는 `limit * 20`만 가져와 후처리로 걸렀다. project 36만 건을
    /// 인덱싱한 뒤 상위 40건이 전부 project가 되면서 `--axis session -n 2`가
    /// 0건을 반환했다.
    #[test]
    fn small_collection_survives_a_much_larger_one() {
        let store_dir = TmpDir::new("store");
        let index_dir = TmpDir::new("index");
        let mut store = Store::open(&store_dir.path().join("s.sqlite")).unwrap();

        seed(&mut store, "big", 500, "shared");
        seed(&mut store, "small", 3, "shared");
        index_dirty(index_dir.path(), &mut store).unwrap();

        let cfg = empty_cfg();
        let hits = search_in(index_dir.path(), &cfg, "shared", 2, Some(&["small"])).unwrap();

        assert_eq!(hits.len(), 2, "small collection must still yield hits");
        assert!(
            hits.iter().all(|h| h.file.starts_with("kmd://small/")),
            "filter leaked other collections: {:?}",
            hits.iter().map(|h| &h.file).collect::<Vec<_>>()
        );
    }

    #[test]
    fn multiple_collections_are_unioned() {
        let store_dir = TmpDir::new("store-union");
        let index_dir = TmpDir::new("index-union");
        let mut store = Store::open(&store_dir.path().join("s.sqlite")).unwrap();

        seed(&mut store, "a", 5, "shared");
        seed(&mut store, "b", 5, "shared");
        seed(&mut store, "c", 5, "shared");
        index_dirty(index_dir.path(), &mut store).unwrap();

        let cfg = empty_cfg();
        let hits = search_in(index_dir.path(), &cfg, "shared", 20, Some(&["a", "b"])).unwrap();

        assert_eq!(hits.len(), 10, "both requested collections contribute");
        assert!(
            !hits.iter().any(|h| h.file.starts_with("kmd://c/")),
            "unrequested collection leaked"
        );
    }

    #[test]
    fn no_filter_searches_everything() {
        let store_dir = TmpDir::new("store-all");
        let index_dir = TmpDir::new("index-all");
        let mut store = Store::open(&store_dir.path().join("s.sqlite")).unwrap();

        seed(&mut store, "a", 3, "shared");
        seed(&mut store, "b", 3, "shared");
        index_dirty(index_dir.path(), &mut store).unwrap();

        let cfg = empty_cfg();
        let hits = search_in(index_dir.path(), &cfg, "shared", 20, None).unwrap();
        assert_eq!(hits.len(), 6);
    }

    /// 본문 미저장 문서는 색인도 스니펫도 원본 파일에서 읽는다.
    #[test]
    fn body_less_doc_indexes_and_snippets_from_disk() {
        let src = TmpDir::new("src");
        let store_dir = TmpDir::new("store-abspath");
        let index_dir = TmpDir::new("index-abspath");

        let file = src.path().join("code.go");
        std::fs::write(&file, "package main\n// uniquetoken lives here\n").unwrap();

        let mut store = Store::open(&store_dir.path().join("s.sqlite")).unwrap();
        store
            .upsert_doc(
                "project",
                "code.go",
                "code.go",
                "", // 본문은 store에 없다
                None,
                0,
                0,
                "h1",
                Some(file.to_str().unwrap()),
            )
            .unwrap();
        let n = index_dirty(index_dir.path(), &mut store).unwrap();
        assert_eq!(n, 1, "body is read from disk for indexing");

        let cfg = empty_cfg();
        let hits = search_in(index_dir.path(), &cfg, "uniquetoken", 5, None).unwrap();
        assert_eq!(hits.len(), 1, "term from the on-disk body is searchable");
        assert!(
            hits[0].snippet.as_deref().unwrap().contains("uniquetoken"),
            "snippet is read back from the original file"
        );
    }

    /// 히트는 읽을 수 있는 절대경로를 들고 온다.
    ///
    /// 두 경로가 서로 다르게 유도되므로 둘 다 검증한다: 본문 미저장 문서는
    /// 인덱스에 박힌 abspath를 그대로 쓰고, 본문 저장 문서는 설정의 컬렉션
    /// 루트에 relpath를 붙인다. 한쪽만 테스트하면 다른 쪽이 조용히 None을
    /// 반환해도 통과한다 — abspath는 Option이라 빠져도 타입이 막아주지 않는다.
    #[test]
    fn hits_carry_a_readable_abspath() {
        let src = TmpDir::new("src-abs");
        let store_dir = TmpDir::new("store-abs2");
        let index_dir = TmpDir::new("index-abs2");

        let on_disk = src.path().join("code.go");
        std::fs::write(&on_disk, "package main\n// findme lives here\n").unwrap();

        let mut store = Store::open(&store_dir.path().join("s.sqlite")).unwrap();
        // 본문 미저장 — abspath가 인덱스에 있다.
        store
            .upsert_doc(
                "project",
                "code.go",
                "code.go",
                "",
                None,
                0,
                0,
                "h1",
                Some(on_disk.to_str().unwrap()),
            )
            .unwrap();
        // 본문 저장 — abspath는 컬렉션 루트에서 유도해야 한다.
        store
            .upsert_doc(
                "notes",
                "n.md",
                "n.md",
                "findme in a note",
                None,
                0,
                0,
                "h2",
                None,
            )
            .unwrap();
        index_dirty(index_dir.path(), &mut store).unwrap();

        let cfg: IndexConfig = serde_yaml::from_str(&format!(
            "collections:\n  notes:\n    path: {}\n",
            src.path().display()
        ))
        .unwrap();
        let hits = search_in(index_dir.path(), &cfg, "findme", 5, None).unwrap();
        assert_eq!(hits.len(), 2, "both documents match");

        let project = hits.iter().find(|h| h.file.contains("project")).unwrap();
        assert_eq!(
            project.abspath.as_deref(),
            on_disk.to_str(),
            "a body-less doc reports the abspath the index already holds"
        );

        let note = hits.iter().find(|h| h.file.contains("notes")).unwrap();
        assert_eq!(
            note.abspath.as_deref(),
            src.path().join("n.md").to_str(),
            "a stored-body doc reports collection root + relpath"
        );
    }

    /// 설정에 없는 컬렉션은 경로를 추측하지 않는다.
    ///
    /// 컬렉션 이름이 바뀌었거나 index.yml에서 빠진 뒤 남은 인덱스가 이 경우다.
    /// 존재하지 않는 경로를 그럴듯하게 돌려주면 호출자가 Read에 실패하는 이유를
    /// 알 수 없으므로, 모른다고 말한다.
    #[test]
    fn unknown_collection_yields_no_abspath() {
        let store_dir = TmpDir::new("store-abs3");
        let index_dir = TmpDir::new("index-abs3");
        let mut store = Store::open(&store_dir.path().join("s.sqlite")).unwrap();
        store
            .upsert_doc("orphan", "n.md", "n.md", "findme", None, 0, 0, "h", None)
            .unwrap();
        index_dirty(index_dir.path(), &mut store).unwrap();

        let hits = search_in(index_dir.path(), &empty_cfg(), "findme", 5, None).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(
            hits[0].abspath.is_none(),
            "a collection absent from the config reports no path, not a guessed one"
        );
    }

    /// 원본이 사라진 문서는 색인에서 조용히 빠진다 — 에러가 아니다.
    #[test]
    fn missing_original_is_skipped_not_fatal() {
        let store_dir = TmpDir::new("store-missing");
        let index_dir = TmpDir::new("index-missing");
        let mut store = Store::open(&store_dir.path().join("s.sqlite")).unwrap();

        store
            .upsert_doc(
                "project",
                "gone.go",
                "gone.go",
                "",
                None,
                0,
                0,
                "h1",
                Some("/nonexistent/gone.go"),
            )
            .unwrap();

        let n = index_dirty(index_dir.path(), &mut store).unwrap();
        assert_eq!(n, 0, "vanished file contributes no document");
        // dirty는 해제돼 다음 스캔이 계속 진행될 수 있어야 한다.
        assert!(store.dirty_docs().unwrap().is_empty());
    }
}
