//! `kmd get` — 검색 히트 하나의 본문을 출력한다.
//!
//! 에이전트는 `kmd search`가 찍은 `kmd://<collection>/<path>`나 `--json`의
//! `docid`를 그대로 넘긴다. 둘 다 검색과 같은 인덱스로 푼다: docid는 store
//! 문서 id의 16진 표기이고, URI의 (collection, relpath)는 store의 UNIQUE 키다.
//! 본문은 검색이 스니펫을 만드는 tantivy 문서에서 꺼낸다(`bm25::fetch`).
//! 검색 CLI처럼 데몬 없이 인덱스를 직접 읽는다.

use crate::bm25::{self, Document};
use crate::config::{self, IndexConfig};
use crate::store::Store;
use anyhow::{Result, bail};
use std::path::{Path, PathBuf};

#[derive(Debug, PartialEq)]
enum Target {
    Docid(i64),
    Uri { collection: String, relpath: String },
    Path(String),
}

pub fn run(target: &str, from: Option<usize>, lines: Option<usize>, json: bool) -> Result<()> {
    let cfg = config::load()?;
    let doc = fetch_target(
        &cfg,
        &config::store_path(),
        &config::tantivy_dir(),
        target,
        from,
        lines,
    )?;
    if json {
        println!("{}", serde_json::to_string_pretty(&doc)?);
    } else {
        print!("{}", doc.content);
        if !doc.content.is_empty() && !doc.content.ends_with('\n') {
            println!();
        }
    }
    Ok(())
}

/// 경로를 인자로 받는 본체 — 테스트가 실제 설정·인덱스를 읽지 않게 한다.
fn fetch_target(
    cfg: &IndexConfig,
    store_path: &Path,
    index_dir: &Path,
    raw: &str,
    from: Option<usize>,
    lines: Option<usize>,
) -> Result<Document> {
    let (base, line) = split_line_suffix(raw);
    let target = parse_target(base);
    let Some(mut doc) = resolve(cfg, store_path, index_dir, &target)? else {
        bail!("{}", not_found(base, &target, cfg));
    };
    // 명시한 --from이 `:<line>` 접미사보다 우선한다.
    let start = from.or(line).unwrap_or(1);
    if start > 1 || lines.is_some() {
        doc.content = slice_lines(&doc.content, start, lines);
    }
    Ok(doc)
}

/// qmd 호환 `<target>:<line>` 접미사를 떼어 낸다. `kmd://`의 콜론은 뒤가
/// 숫자가 아니므로 걸리지 않는다.
fn split_line_suffix(raw: &str) -> (&str, Option<usize>) {
    if let Some((head, tail)) = raw.rsplit_once(':')
        && !head.is_empty()
        && !tail.is_empty()
        && tail.bytes().all(|b| b.is_ascii_digit())
        && let Ok(n) = tail.parse::<usize>()
    {
        return (head, Some(n));
    }
    (raw, None)
}

fn parse_target(raw: &str) -> Target {
    // qmd:// — 옛 습관으로 들어오는 스킴. eval의 norm_path도 같은 이유로 받는다.
    for scheme in ["kmd://", "qmd://"] {
        if let Some(rest) = raw.strip_prefix(scheme) {
            let (collection, relpath) = rest.split_once('/').unwrap_or((rest, ""));
            return Target::Uri {
                collection: collection.to_string(),
                relpath: relpath.to_string(),
            };
        }
    }
    // 같은 이름의 파일이 있으면 경로가 이긴다 — `#` 없는 16진 docid와 겹칠 수 있다.
    if !Path::new(raw).exists()
        && let Some(id) = parse_docid(raw)
    {
        return Target::Docid(id);
    }
    Target::Path(raw.to_string())
}

/// `#06ab80` 또는 `06ab80`. 셸에서 따옴표 없는 `#…`은 주석이라 `#`을 빼고
/// 넘기는 경우가 많다. `#` 없는 형태는 search가 찍는 최소 폭(6자리) 이상만
/// docid로 본다 — `abc` 같은 짧은 이름을 docid로 오인하지 않게.
fn parse_docid(raw: &str) -> Option<i64> {
    let (hex, hashed) = match raw.strip_prefix('#') {
        Some(h) => (h, true),
        None => (raw, false),
    };
    if hex.is_empty() || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    if !hashed && hex.len() < 6 {
        return None;
    }
    i64::from_str_radix(hex, 16).ok()
}

fn resolve(
    cfg: &IndexConfig,
    store_path: &Path,
    index_dir: &Path,
    target: &Target,
) -> Result<Option<Document>> {
    let candidates = match target {
        Target::Docid(id) => return bm25::fetch(index_dir, cfg, *id),
        Target::Uri {
            collection,
            relpath,
        } => vec![(collection.clone(), relpath.clone())],
        Target::Path(p) => {
            let mut c = path_candidates(cfg, p);
            // 스킴을 뺀 `collection/relpath`.
            if let Some((coll, rel)) = p.split_once('/')
                && cfg.collections.contains_key(coll)
            {
                c.push((coll.to_string(), rel.to_string()));
            }
            c
        }
    };
    let store = Store::open(store_path)?;
    for (coll, rel) in &candidates {
        if let Some(id) = store.id_of(coll, rel)?
            && let Some(doc) = bm25::fetch(index_dir, cfg, id)?
        {
            return Ok(Some(doc));
        }
    }
    // 인덱스 밖의 파일이라도 읽을 수 있으면 읽는다. docid가 없으므로 null.
    if let Target::Path(p) = target {
        let abs = absolutize(p);
        if let Ok(content) = std::fs::read_to_string(&abs) {
            let shown = abs.to_string_lossy().into_owned();
            return Ok(Some(Document {
                file: shown.clone(),
                abspath: Some(shown),
                docid: None,
                title: String::new(),
                content,
            }));
        }
    }
    Ok(None)
}

/// 파일 경로 → (collection, relpath) 후보. 스캔은 relpath를 컬렉션 `path`
/// 기준으로 만들므로(scan.rs) 그 역연산이다. 심볼릭 링크로 들어온 경로도
/// 잡도록 원래 경로와 canonical 경로를 모두 시도하고, 루트가 깊은(relpath가
/// 짧은) 컬렉션부터 본다 — 중첩된 컬렉션이면 더 구체적인 쪽이 맞다.
fn path_candidates(cfg: &IndexConfig, raw: &str) -> Vec<(String, String)> {
    let abs = absolutize(raw);
    let mut paths = vec![abs.clone()];
    if let Ok(c) = abs.canonicalize()
        && c != abs
    {
        paths.push(c);
    }
    let mut out: Vec<(String, String)> = Vec::new();
    for (name, coll) in &cfg.collections {
        let mut roots = vec![coll.path.clone()];
        if let Ok(c) = coll.path.canonicalize()
            && c != coll.path
        {
            roots.push(c);
        }
        for p in &paths {
            for root in &roots {
                if let Ok(rel) = p.strip_prefix(root)
                    && !rel.as_os_str().is_empty()
                {
                    let pair = (name.clone(), rel.to_string_lossy().into_owned());
                    if !out.contains(&pair) {
                        out.push(pair);
                    }
                }
            }
        }
    }
    out.sort_by_key(|(_, rel)| Path::new(rel).components().count());
    out
}

fn absolutize(raw: &str) -> PathBuf {
    let p = PathBuf::from(raw);
    if p.is_absolute() {
        return p;
    }
    std::env::current_dir().map(|cwd| cwd.join(&p)).unwrap_or(p)
}

/// `from`(1부터)에서 시작해 최대 `count`줄. 줄바꿈은 그대로 둔다.
fn slice_lines(content: &str, from: usize, count: Option<usize>) -> String {
    let rest = content.split_inclusive('\n').skip(from.saturating_sub(1));
    match count {
        Some(n) => rest.take(n).collect(),
        None => rest.collect(),
    }
}

/// 한 줄 에러: 무엇을 찾았는지와 다음에 할 일.
fn not_found(raw: &str, target: &Target, cfg: &IndexConfig) -> String {
    let looked_up = match target {
        Target::Docid(id) => format!("docid {}", bm25::docid_of(*id)),
        Target::Uri { collection, .. } if !cfg.collections.contains_key(collection) => format!(
            "collection {:?} (not in index.yml; known: {})",
            collection,
            cfg.collections
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Target::Uri {
            collection,
            relpath,
        } => format!("collection {:?}, path {:?}", collection, relpath),
        Target::Path(p) => format!(
            "path {:?} (not under an indexed collection and not a readable file)",
            p
        ),
    };
    format!(
        "no document for {} — looked up {} in the kmd index; find it with: kmd search <query>",
        raw, looked_up
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bm25::index_dirty;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static SEQ: AtomicUsize = AtomicUsize::new(0);

    /// 테스트 전용 임시 디렉터리 (bm25.rs/scan.rs와 같은 방식, tempfile 없이).
    struct TmpDir(PathBuf);

    impl TmpDir {
        fn new(tag: &str) -> Self {
            let n = SEQ.fetch_add(1, Ordering::SeqCst);
            let p = std::env::temp_dir().join(format!(
                "kmd-get-test-{}-{}-{}",
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

    /// 본문 저장 컬렉션(notes)과 본문 미저장 컬렉션(project) 하나씩 색인한
    /// 격리된 환경. 실제 ~/.config/kmd, ~/.cache/kmd는 건드리지 않는다.
    struct Fixture {
        notes: TmpDir,
        project: TmpDir,
        store: TmpDir,
        index: TmpDir,
        cfg: IndexConfig,
    }

    impl Fixture {
        fn new(tag: &str) -> Self {
            let notes = TmpDir::new(&format!("{}-notes", tag));
            let project = TmpDir::new(&format!("{}-project", tag));
            let store_dir = TmpDir::new(&format!("{}-store", tag));
            let index = TmpDir::new(&format!("{}-index", tag));

            std::fs::create_dir_all(project.path().join("repo")).unwrap();
            let code = project.path().join("repo/main.go");
            std::fs::write(&code, "package main\n\nfunc main() {}\n").unwrap();

            let mut store = Store::open(&store_dir.path().join("s.sqlite")).unwrap();
            store
                .upsert_doc(
                    "notes",
                    "tcp/keepalive.md",
                    "TCP Keepalive",
                    "# TCP Keepalive\nline2\nline3\nline4\n",
                    None,
                    0,
                    0,
                    "h1",
                    None,
                )
                .unwrap();
            store
                .upsert_doc(
                    "project",
                    "repo/main.go",
                    "main.go",
                    "",
                    None,
                    0,
                    0,
                    "h2",
                    Some(code.to_str().unwrap()),
                )
                .unwrap();
            index_dirty(index.path(), &mut store).unwrap();

            let cfg: IndexConfig = serde_yaml::from_str(&format!(
                "collections:\n  notes:\n    path: {}\n  project:\n    path: {}\n    source: git-repos\n",
                notes.path().display(),
                project.path().display()
            ))
            .unwrap();
            Fixture {
                notes,
                project,
                store: store_dir,
                index,
                cfg,
            }
        }

        fn get(&self, raw: &str, from: Option<usize>, lines: Option<usize>) -> Result<Document> {
            fetch_target(
                &self.cfg,
                &self.store.path().join("s.sqlite"),
                self.index.path(),
                raw,
                from,
                lines,
            )
        }

        /// 검색이 이 문서에 대해 찍는 히트 — get은 이것을 그대로 받아야 한다.
        fn hit(&self, query: &str) -> bm25::SearchHit {
            let mut hits = bm25::search_in(self.index.path(), &self.cfg, query, 1, None).unwrap();
            assert_eq!(hits.len(), 1, "fixture doc must be searchable: {}", query);
            hits.remove(0)
        }
    }

    /// 검색 출력의 `file`을 그대로 넘기면 그 문서가 열린다 — 이 커맨드의 존재 이유.
    #[test]
    fn search_hit_uri_opens_the_document() {
        let fx = Fixture::new("uri");
        let hit = fx.hit("keepalive");
        let doc = fx.get(&hit.file, None, None).unwrap();
        assert_eq!(doc.file, hit.file);
        assert_eq!(doc.docid.as_deref(), Some(hit.docid.as_str()));
        assert_eq!(doc.abspath, hit.abspath);
        assert_eq!(doc.title, "TCP Keepalive");
        assert_eq!(doc.content, "# TCP Keepalive\nline2\nline3\nline4\n");
    }

    /// `--json`의 docid는 `#` 유무와 관계없이 같은 문서로 풀린다.
    #[test]
    fn docid_with_or_without_hash_opens_the_document() {
        let fx = Fixture::new("docid");
        let hit = fx.hit("keepalive");
        let bare = hit.docid.trim_start_matches('#').to_string();
        for raw in [hit.docid.as_str(), bare.as_str()] {
            let doc = fx.get(raw, None, None).unwrap();
            assert_eq!(doc.file, hit.file, "docid {:?}", raw);
        }
    }

    /// 본문 미저장 문서는 원본 파일에서 읽고, 그 절대경로로도 찾을 수 있다.
    #[test]
    fn body_less_doc_reads_from_disk_and_resolves_by_path() {
        let fx = Fixture::new("path");
        let abs = fx.project.path().join("repo/main.go");
        let doc = fx.get(abs.to_str().unwrap(), None, None).unwrap();
        assert_eq!(doc.file, "kmd://project/repo/main.go");
        assert!(doc.docid.is_some(), "an indexed path reports its docid");
        assert!(doc.content.contains("func main"));

        let by_uri = fx.get("kmd://project/repo/main.go", None, None).unwrap();
        assert_eq!(by_uri.docid, doc.docid);
    }

    /// 본문 저장 문서도 컬렉션 루트 기준 경로로 찾는다(원본 파일이 없어도).
    #[test]
    fn stored_body_doc_resolves_by_collection_root_path() {
        let fx = Fixture::new("root");
        let p = fx.notes.path().join("tcp/keepalive.md");
        let doc = fx.get(p.to_str().unwrap(), None, None).unwrap();
        assert_eq!(doc.file, "kmd://notes/tcp/keepalive.md");
    }

    #[test]
    fn scheme_less_collection_path_resolves() {
        let fx = Fixture::new("noscheme");
        let doc = fx.get("notes/tcp/keepalive.md", None, None).unwrap();
        assert_eq!(doc.file, "kmd://notes/tcp/keepalive.md");
    }

    #[test]
    fn from_and_lines_slice_the_content() {
        let fx = Fixture::new("slice");
        let doc = fx
            .get("kmd://notes/tcp/keepalive.md", Some(2), Some(2))
            .unwrap();
        assert_eq!(doc.content, "line2\nline3\n");
    }

    /// qmd 호환 `:<line>` 접미사는 시작 줄이고, 명시한 --from이 이긴다.
    #[test]
    fn line_suffix_sets_the_start_line() {
        let fx = Fixture::new("suffix");
        let doc = fx
            .get("kmd://notes/tcp/keepalive.md:3", None, None)
            .unwrap();
        assert_eq!(doc.content, "line3\nline4\n");
        let doc = fx
            .get("kmd://notes/tcp/keepalive.md:3", Some(4), None)
            .unwrap();
        assert_eq!(doc.content, "line4\n");
    }

    /// 인덱스 밖이라도 읽을 수 있는 파일은 docid 없이 출력한다.
    #[test]
    fn unindexed_readable_file_is_printed_without_docid() {
        let fx = Fixture::new("unindexed");
        let p = fx.notes.path().join("loose.txt");
        std::fs::write(&p, "loose\n").unwrap();
        let doc = fx.get(p.to_str().unwrap(), None, None).unwrap();
        assert_eq!(doc.content, "loose\n");
        assert!(doc.docid.is_none());
    }

    /// 못 찾으면 한 줄 에러가 찾은 대상과 `kmd search`를 알려 준다.
    #[test]
    fn not_found_names_the_lookup_and_suggests_search() {
        let fx = Fixture::new("missing");
        for (raw, needle) in [
            ("kmd://notes/nope.md", "path \"nope.md\""),
            ("kmd://wiki/x.md", "not in index.yml; known: notes, project"),
            ("#ffffff", "docid #ffffff"),
            (
                "/definitely/not/here.md",
                "path \"/definitely/not/here.md\"",
            ),
        ] {
            let err = fx.get(raw, None, None).unwrap_err().to_string();
            assert!(!err.contains('\n'), "one line: {}", err);
            assert!(err.contains(raw), "names the target: {}", err);
            assert!(err.contains(needle), "{:?} missing from: {}", needle, err);
            assert!(
                err.contains("kmd search <query>"),
                "suggests search: {}",
                err
            );
        }
    }

    #[test]
    fn target_parsing() {
        assert_eq!(parse_docid("#06ab80"), Some(0x06ab80));
        assert_eq!(parse_docid("06ab80"), Some(0x06ab80));
        assert_eq!(parse_docid("#1"), Some(1));
        assert_eq!(
            parse_docid("abc"),
            None,
            "short bare hex is a name, not a docid"
        );
        assert_eq!(parse_docid("README.md"), None);
        assert_eq!(
            split_line_suffix("kmd://x/y.md:120"),
            ("kmd://x/y.md", Some(120))
        );
        assert_eq!(split_line_suffix("kmd://x/y.md"), ("kmd://x/y.md", None));
        assert_eq!(
            parse_target("kmd://project/keyolk/cc-mods/README.md"),
            Target::Uri {
                collection: "project".into(),
                relpath: "keyolk/cc-mods/README.md".into()
            }
        );
    }
}
