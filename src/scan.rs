//! 컬렉션 스캔 — index.yml의 path/pattern 기반 파일 발견 + 증분 upsert.
//!
//! 열거 방식은 두 가지다. `Source::Glob`은 walkdir로 전수 순회하고,
//! `Source::GitRepos`는 루트 아래 git repo들을 찾아 `git ls-files`로 추적 파일만
//! 얻는다. 후자는 gitignore된 산출물이 자동으로 빠지고 훨씬 빠르다.

use crate::config::{Collection, IndexConfig, Source};
use crate::store::{Store, UpsertOutcome};
use anyhow::Result;
use globset::{Glob, GlobSet, GlobSetBuilder};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::UNIX_EPOCH;
use walkdir::WalkDir;

#[derive(Debug, Default)]
pub struct ScanStats {
    pub seen: usize,
    pub added: usize,
    pub updated: usize,
    pub removed: usize,
    /// 바이너리·과대 파일 등으로 건너뛴 수.
    pub skipped: usize,
}

/// 설정과 무관하게 항상 제외하는 경로.
///
/// 평가 세트는 질문과 정답 앵커를 나란히 담고 있어서, 인덱싱되면 자기 질문을
/// 자기가 최고 점수로 찾는다 — 실측: gold 프롬프트 20건 중 18건에서 평가
/// 파일이 top-5에 들었고, 한 케이스는 144점으로 실제 정답 후보(57점)를
/// 압도했다. 그 상태로 잰 검색 품질 수치는 무엇을 재는지 정해지지 않는다.
///
/// 사용자 설정에 맡기지 않는 이유: 이건 kmd 자신의 성질이다. kmd 저장소를
/// (또는 kmd를 포크한 무엇이든) `source: git-repos`로 인덱싱하는 사람이면
/// 누구나 같은 오염을 겪고, 그 사실을 알아야 exclude를 쓸 수 있다.
const ALWAYS_EXCLUDE: &[&str] = &["**/eval/gold.yaml", "**/eval/prompts.txt"];


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

        let mut ex = GlobSetBuilder::new();
        for pat in &coll.exclude {
            ex.add(Glob::new(pat)?);
        }
        for pat in ALWAYS_EXCLUDE {
            ex.add(Glob::new(pat)?);
        }
        let exclude = ex.build()?;

        let context = coll.root_context().map(str::to_string);
        let mut seen_paths: Vec<String> = Vec::new();

        let files = match coll.source {
            Source::Glob => enumerate_glob(coll, &glob),
            Source::GitRepos => enumerate_git_repos(coll),
        };

        for rel in files {
            // exclude는 glob 패턴과 달리 상대경로 전체에 매칭
            if exclude.is_match(&rel) {
                continue;
            }
            let abs = coll.path.join(&rel);
            let relpath = rel.to_string_lossy().to_string();

            match index_file(store, name, coll, &abs, &relpath, context.as_deref(), force)? {
                Some(outcome) => {
                    seen_paths.push(relpath);
                    stats.seen += 1;
                    match outcome {
                        UpsertOutcome::Added => stats.added += 1,
                        UpsertOutcome::Updated => stats.updated += 1,
                        UpsertOutcome::Unchanged => {}
                    }
                }
                None => stats.skipped += 1,
            }
        }

        stats.removed += store.deactivate_missing(name, &seen_paths)?;
    }

    Ok(stats)
}

/// 한 파일을 읽어 upsert. 건너뛴 경우(과대·바이너리·읽기 실패) None.
fn index_file(
    store: &mut Store,
    name: &str,
    coll: &Collection,
    abs: &Path,
    relpath: &str,
    context: Option<&str>,
    force: bool,
) -> Result<Option<UpsertOutcome>> {
    let meta = match std::fs::metadata(abs) {
        Ok(m) => m,
        Err(_) => return Ok(None),
    };
    if !meta.is_file() {
        return Ok(None);
    }
    if let Some(max) = coll.max_file_size
        && meta.len() > max
    {
        return Ok(None);
    }

    let mtime_ns = meta
        .modified()?
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(0);
    let size = meta.len() as i64;

    // 변경 감지에는 본문이 필요하다. mtime+size만으로는 신뢰할 수 없다.
    let body = match std::fs::read_to_string(abs) {
        Ok(b) => b,
        Err(_) => return Ok(None), // 바이너리/읽기 불가 스킵
    };
    let hash = if force {
        // force 시 hash를 다르게 만들어 무조건 재인덱싱
        format!("force-{:x}", Sha256::digest(body.as_bytes()))
    } else {
        format!("{:x}", Sha256::digest(body.as_bytes()))
    };
    let title = extract_title(relpath, &body);

    // 본문 미저장 컬렉션은 절대경로만 남기고 본문은 검색 시점에 원본에서 읽는다.
    let (stored_body, abspath) = if coll.stores_body() {
        (body.as_str(), None)
    } else {
        ("", Some(abs.to_string_lossy().to_string()))
    };

    let outcome = store.upsert_doc(
        name,
        relpath,
        &title,
        stored_body,
        context,
        mtime_ns,
        size,
        &hash,
        abspath.as_deref(),
    )?;
    Ok(Some(outcome))
}

/// walkdir 전수 순회 — 컬렉션 루트 기준 상대경로를 반환.
fn enumerate_glob(coll: &Collection, glob: &GlobSet) -> Vec<PathBuf> {
    let mut out = Vec::new();
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
        out.push(rel.to_path_buf());
    }
    out
}

/// 루트 아래 git repo들을 찾아 각각 `git ls-files`. 반환값은 컬렉션 루트 기준
/// 상대경로 — repo 경로 자체가 접두어로 붙는다.
fn enumerate_git_repos(coll: &Collection) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for repo in find_git_repos(&coll.path, GIT_REPO_SCAN_DEPTH) {
        let repo_rel = match repo.strip_prefix(&coll.path) {
            Ok(r) => r.to_path_buf(),
            Err(_) => continue,
        };
        let output = match Command::new("git")
            .arg("-C")
            .arg(&repo)
            .arg("ls-files")
            .arg("-z")
            .output()
        {
            Ok(o) if o.status.success() => o,
            _ => continue,
        };
        for raw in output.stdout.split(|b| *b == 0) {
            if raw.is_empty() {
                continue;
            }
            let s = match std::str::from_utf8(raw) {
                Ok(s) => s,
                Err(_) => continue,
            };
            out.push(repo_rel.join(s));
        }
    }
    out
}

/// `~/src` 아래 repo는 대개 1~3단계에 있다. 더 깊이 들어가면 vendor된
/// 서브모듈까지 잡혀 비용만 늘어난다.
const GIT_REPO_SCAN_DEPTH: usize = 3;

/// `.git`을 가진 디렉터리를 찾는다. repo를 찾으면 그 아래로는 내려가지 않는다
/// (중첩 repo/서브모듈은 상위 repo의 ls-files가 다루거나 무시된다).
fn find_git_repos(root: &Path, max_depth: usize) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![(root.to_path_buf(), 0usize)];
    while let Some((dir, depth)) = stack.pop() {
        if dir.join(".git").exists() {
            out.push(dir);
            continue;
        }
        if depth >= max_depth {
            continue;
        }
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.filter_map(|e| e.ok()) {
            if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            let name = entry.file_name();
            if name.to_str().map(|s| s.starts_with('.')).unwrap_or(false) {
                continue;
            }
            stack.push((entry.path(), depth + 1));
        }
    }
    out.sort();
    out
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::IndexConfig;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static SEQ: AtomicUsize = AtomicUsize::new(0);

    /// 테스트 전용 임시 디렉터리. tempfile 의존성을 더하지 않으려고 pid+seq로
    /// 격리하고 Drop에서 지운다.
    struct TmpDir(PathBuf);

    impl TmpDir {
        fn new(tag: &str) -> Self {
            let n = SEQ.fetch_add(1, Ordering::SeqCst);
            let p = std::env::temp_dir().join(format!(
                "kmd-scan-test-{}-{}-{}",
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

    fn write(root: &Path, rel: &str, body: &str) {
        let p = root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, body).unwrap();
    }

    fn git(repo: &Path, args: &[&str]) {
        let out = Command::new("git")
            .arg("-C")
            .arg(repo)
            .args(args)
            .output()
            .expect("git runs");
        assert!(out.status.success(), "git {:?} failed", args);
    }

    /// 커밋까지 마친 최소 repo. ls-files는 인덱스를 보므로 add까지면 충분하지만,
    /// 실제 사용 형태에 맞춰 커밋한다.
    fn init_repo(root: &Path) {
        git(root, &["init", "-q"]);
        git(root, &["config", "user.email", "t@example.com"]);
        git(root, &["config", "user.name", "t"]);
    }

    fn cfg_from(yaml: &str) -> IndexConfig {
        serde_yaml::from_str(yaml).expect("valid yaml")
    }

    #[test]
    fn finds_repos_at_varying_depths() {
        let tmp = TmpDir::new("depth");
        let root = tmp.path();
        for rel in ["a", "org/b", "org/nested/c"] {
            std::fs::create_dir_all(root.join(rel)).unwrap();
            init_repo(&root.join(rel));
        }
        let found = find_git_repos(root, GIT_REPO_SCAN_DEPTH);
        assert_eq!(found.len(), 3, "found: {:?}", found);
    }

    #[test]
    fn does_not_descend_into_a_repo() {
        // repo 안의 서브모듈/vendor된 repo는 상위 repo의 ls-files가 다룬다.
        let tmp = TmpDir::new("nested");
        let root = tmp.path();
        std::fs::create_dir_all(root.join("outer/vendor/inner")).unwrap();
        init_repo(&root.join("outer"));
        init_repo(&root.join("outer/vendor/inner"));

        let found = find_git_repos(root, GIT_REPO_SCAN_DEPTH);
        assert_eq!(found, vec![root.join("outer")]);
    }

    #[test]
    fn git_repos_enumerates_tracked_files_only() {
        let tmp = TmpDir::new("tracked");
        let root = tmp.path();
        let repo = root.join("proj");
        std::fs::create_dir_all(&repo).unwrap();
        init_repo(&repo);
        write(&repo, "keep.go", "package main");
        write(&repo, ".gitignore", "ignored.go\n");
        write(&repo, "ignored.go", "package ignored");
        git(&repo, &["add", "keep.go", ".gitignore"]);

        let cfg = cfg_from(&format!(
            "collections:\n  project:\n    path: {}\n    source: git-repos\n",
            root.display()
        ));
        let coll = &cfg.collections["project"];
        let files: Vec<String> = enumerate_git_repos(coll)
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect();

        assert!(files.contains(&"proj/keep.go".to_string()), "{:?}", files);
        assert!(
            !files.iter().any(|f| f.contains("ignored.go")),
            "gitignored file leaked: {:?}",
            files
        );
    }

    #[test]
    /// 평가 세트는 설정과 무관하게 인덱싱되지 않는다.
    ///
    /// 이 파일이 인덱스에 들어가면 검색 품질 측정이 자기 자신을 찾는다.
    /// 회귀하면 수치가 조용히 부풀고, 그게 오염인지 개선인지 구분되지 않는다.
    #[test]
    fn eval_sets_are_never_indexed() {
        let tmp = TmpDir::new("always-exclude");
        let root = tmp.path();
        std::fs::create_dir_all(root.join("eval")).unwrap();
        std::fs::write(root.join("eval/gold.yaml"), "- prompt: \"x\"\n").unwrap();
        std::fs::write(root.join("eval/prompts.txt"), "x\n").unwrap();
        std::fs::write(root.join("keep.md"), "kept\n").unwrap();

        // exclude를 하나도 설정하지 않은 컬렉션에서도 빠져야 한다.
        let cfg: IndexConfig = serde_yaml::from_str(&format!(
            "collections:\n  c:\n    path: {}\n    pattern: \"**/*.{{md,yaml,txt}}\"\n",
            root.display()
        ))
        .unwrap();
        let store_dir = TmpDir::new("always-exclude-store");
        let mut store = Store::open(&store_dir.path().join("s.sqlite")).unwrap();
        update(&cfg, &mut store, false).unwrap();

        let paths: Vec<String> = store
            .conn
            .prepare("SELECT relpath FROM documents WHERE active = 1")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .collect::<std::result::Result<_, _>>()
            .unwrap();
        assert!(
            paths.iter().any(|p| p.contains("keep.md")),
            "an ordinary file is still indexed: {:?}",
            paths
        );
        assert!(
            !paths.iter().any(|p| p.contains("gold.yaml")),
            "gold.yaml must never be indexed: {:?}",
            paths
        );
        assert!(
            !paths.iter().any(|p| p.contains("prompts.txt")),
            "prompts.txt must never be indexed: {:?}",
            paths
        );
    }

    fn exclude_patterns_drop_matching_paths() {
        let tmp = TmpDir::new("exclude");
        let root = tmp.path();
        let repo = root.join("proj");
        std::fs::create_dir_all(&repo).unwrap();
        init_repo(&repo);
        write(&repo, "main.go", "package main");
        write(&repo, "vendor/dep/dep.go", "package dep");
        write(&repo, "api.pb.go", "// generated");
        git(&repo, &["add", "-A"]);

        let cfg = cfg_from(&format!(
            r#"
collections:
  project:
    path: {}
    source: git-repos
    store_body: false
    exclude:
      - "**/vendor/**"
      - "**/*.pb.go"
"#,
            root.display()
        ));
        let coll = &cfg.collections["project"];

        let mut ex = GlobSetBuilder::new();
        for pat in &coll.exclude {
            ex.add(Glob::new(pat).unwrap());
        }
        let exclude = ex.build().unwrap();

        let kept: Vec<String> = enumerate_git_repos(coll)
            .into_iter()
            .filter(|r| !exclude.is_match(r))
            .map(|p| p.to_string_lossy().to_string())
            .collect();

        assert_eq!(kept, vec!["proj/main.go".to_string()]);
    }

    #[test]
    fn store_body_defaults_follow_source() {
        let cfg = cfg_from(
            r#"
collections:
  wiki:
    path: /tmp/wiki
  project:
    path: /tmp/src
    source: git-repos
  forced:
    path: /tmp/f
    source: git-repos
    store_body: true
"#,
        );
        // glob 컬렉션은 기존대로 본문을 저장한다.
        assert!(cfg.collections["wiki"].stores_body());
        // git-repos는 기본이 미저장.
        assert!(!cfg.collections["project"].stores_body());
        // 명시값이 source 기본을 이긴다.
        assert!(cfg.collections["forced"].stores_body());
    }

    #[test]
    fn body_less_collection_stores_abspath_not_body() {
        let tmp = TmpDir::new("abspath");
        let root = tmp.path();
        let repo = root.join("proj");
        std::fs::create_dir_all(&repo).unwrap();
        init_repo(&repo);
        write(&repo, "main.go", "package main\n// findme\n");
        git(&repo, &["add", "-A"]);

        let store_dir = TmpDir::new("store");
        let mut store = Store::open(&store_dir.path().join("s.sqlite")).unwrap();
        let cfg = cfg_from(&format!(
            "collections:\n  project:\n    path: {}\n    source: git-repos\n",
            root.display()
        ));

        let stats = update(&cfg, &mut store, false).unwrap();
        assert_eq!(stats.added, 1, "stats: {:?}", stats);

        let docs = store.dirty_docs().unwrap();
        let doc = docs.iter().find(|d| d.relpath.ends_with("main.go")).unwrap();
        assert!(doc.body.is_empty(), "body should not be copied into store");
        assert!(doc.abspath.is_some(), "abspath must be recorded");
        // 본문은 원본에서 읽힌다.
        let body = crate::store::body_of(doc).expect("reads original file");
        assert!(body.contains("findme"));
    }

    #[test]
    fn body_of_returns_none_when_original_is_gone() {
        // 인덱스가 워킹트리보다 최신인 경우 — 에러가 아니라 None.
        let doc = crate::store::Doc {
            id: 1,
            collection: "project".into(),
            relpath: "gone.go".into(),
            title: "gone".into(),
            body: String::new(),
            context: None,
            abspath: Some("/nonexistent/path/gone.go".into()),
        };
        assert!(crate::store::body_of(&doc).is_none());
    }

    #[test]
    fn max_file_size_skips_large_files() {
        let tmp = TmpDir::new("size");
        let root = tmp.path();
        let repo = root.join("proj");
        std::fs::create_dir_all(&repo).unwrap();
        init_repo(&repo);
        write(&repo, "small.go", "package main");
        write(&repo, "big.go", &"x".repeat(5000));
        git(&repo, &["add", "-A"]);

        let store_dir = TmpDir::new("store-size");
        let mut store = Store::open(&store_dir.path().join("s.sqlite")).unwrap();
        let cfg = cfg_from(&format!(
            "collections:\n  project:\n    path: {}\n    source: git-repos\n    max_file_size: 1000\n",
            root.display()
        ));

        let stats = update(&cfg, &mut store, false).unwrap();
        assert_eq!(stats.added, 1, "only the small file indexes");
        assert_eq!(stats.skipped, 1, "the large file is counted as skipped");
    }

    #[test]
    fn glob_collection_still_stores_body() {
        // 기존 컬렉션의 동작이 바뀌지 않았는지 — 하위 호환 회귀 가드.
        let tmp = TmpDir::new("glob");
        let root = tmp.path();
        write(root, "note.md", "# Title\n\nbody text\n");

        let store_dir = TmpDir::new("store-glob");
        let mut store = Store::open(&store_dir.path().join("s.sqlite")).unwrap();
        let cfg = cfg_from(&format!(
            "collections:\n  wiki:\n    path: {}\n    pattern: \"**/*.md\"\n",
            root.display()
        ));

        update(&cfg, &mut store, false).unwrap();
        let docs = store.dirty_docs().unwrap();
        let doc = docs.iter().find(|d| d.relpath == "note.md").unwrap();
        assert!(doc.abspath.is_none(), "glob collections keep abspath NULL");
        assert!(doc.body.contains("body text"));
        assert_eq!(doc.title, "Title");
    }
}
