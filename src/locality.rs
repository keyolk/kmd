//! Spatial locality derived from cwd, Git roots, and linked worktrees.

use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Space {
    pub cwd: String,
    pub repo: Option<String>,
    pub repo_root: Option<String>,
    pub worktree: Option<String>,
    pub worktree_root: Option<String>,
}

fn normalize(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

fn start_dir(path: &Path) -> PathBuf {
    if path.is_file() {
        path.parent().unwrap_or(path).to_path_buf()
    } else {
        path.to_path_buf()
    }
}

fn git_common_root(git_marker: &Path) -> Option<PathBuf> {
    if git_marker.is_dir() {
        return git_marker.parent().map(normalize);
    }
    let raw = fs::read_to_string(git_marker).ok()?;
    let gitdir = raw.trim().strip_prefix("gitdir:")?.trim();
    let gitdir = if Path::new(gitdir).is_absolute() {
        PathBuf::from(gitdir)
    } else {
        git_marker.parent()?.join(gitdir)
    };
    let gitdir = normalize(&gitdir);
    let components: Vec<_> = gitdir.components().collect();
    let worktrees = components
        .iter()
        .rposition(|part| part.as_os_str() == "worktrees")?;
    let common_git = components[..worktrees].iter().collect::<PathBuf>();
    common_git.parent().map(normalize)
}

fn find_git_space(path: &Path) -> Option<(PathBuf, PathBuf)> {
    let start = start_dir(path);
    for ancestor in start.ancestors() {
        let marker = ancestor.join(".git");
        if marker.exists() {
            let worktree_root = normalize(ancestor);
            let repo_root = git_common_root(&marker).unwrap_or_else(|| worktree_root.clone());
            return Some((repo_root, worktree_root));
        }
    }
    None
}

fn lexical_space(path: &Path) -> (Option<PathBuf>, Option<PathBuf>) {
    let components: Vec<String> = path
        .components()
        .map(|part| part.as_os_str().to_string_lossy().to_string())
        .collect();

    for marker in [".worktree", ".worktrees", "worktrees"] {
        if let Some(index) = components.iter().position(|part| part == marker) {
            if index > 0 && index + 1 < components.len() {
                let repo_root = components[..index].iter().collect::<PathBuf>();
                let worktree_root = components[..=index + 1].iter().collect::<PathBuf>();
                return (Some(repo_root), Some(worktree_root));
            }
        }
    }

    if let Some(src) = components.iter().position(|part| part == "src") {
        let mut repo_index = src + 1;
        if components.get(repo_index).is_some_and(|part| {
            matches!(
                part.as_str(),
                "sendbird" | "keyolk" | "wordfactory" | "test"
            )
        }) {
            repo_index += 1;
        }
        if repo_index < components.len() {
            let root = components[..=repo_index].iter().collect::<PathBuf>();
            return (Some(root.clone()), Some(root));
        }
    }
    (None, None)
}

fn label(path: &Path) -> Option<String> {
    path.file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .map(str::to_string)
}

pub fn resolve(path: &str) -> Space {
    if path.trim().is_empty() {
        return Space::default();
    }
    let input = PathBuf::from(path);
    let cwd = normalize(&input);
    let (repo_root, worktree_root) = find_git_space(&cwd)
        .map(|(repo, worktree)| (Some(repo), Some(worktree)))
        .unwrap_or_else(|| lexical_space(&cwd));
    let repo = repo_root.as_deref().and_then(label);
    let worktree = match (&repo_root, &worktree_root) {
        (Some(repo_root), Some(worktree_root)) if repo_root != worktree_root => worktree_root
            .file_name()
            .and_then(|name| name.to_str())
            .map(str::to_string),
        _ => None,
    };

    Space {
        cwd: cwd.display().to_string(),
        repo,
        repo_root: repo_root.map(|path| path.display().to_string()),
        worktree,
        worktree_root: worktree_root.map(|path| path.display().to_string()),
    }
}

pub fn score(candidate: &Space, anchor: &Space, files: &[String]) -> u8 {
    if anchor.cwd.is_empty() {
        return 0;
    }
    if !candidate.cwd.is_empty() && candidate.cwd == anchor.cwd {
        return 4;
    }
    if candidate.worktree_root.is_some()
        && candidate.worktree_root == anchor.worktree_root
        && candidate.repo == anchor.repo
    {
        return 3;
    }
    if candidate.repo.is_some() && candidate.repo == anchor.repo {
        return 2;
    }
    if let Some(root) = anchor
        .worktree_root
        .as_deref()
        .or(anchor.repo_root.as_deref())
    {
        if files.iter().any(|file| Path::new(file).starts_with(root)) {
            return 1;
        }
    }
    0
}

pub fn label_space(space: &Space) -> String {
    match (&space.repo, &space.worktree) {
        (Some(repo), Some(worktree)) => format!("{}@{}", repo, worktree),
        (Some(repo), None) => repo.clone(),
        _ => "(프로젝트 없음)".to_string(),
    }
}

/// Infer the most likely working directory for historical learnings that predate
/// the `Cwd:` metadata. The most frequently touched Git/worktree root wins.
pub fn infer_cwd(files: &[String]) -> Option<String> {
    let mut roots = std::collections::BTreeMap::<String, usize>::new();
    for file in files {
        if !Path::new(file).is_absolute() {
            continue;
        }
        let space = resolve(file);
        let root = space
            .worktree_root
            .or(space.repo_root)
            .filter(|value| !value.is_empty());
        if let Some(root) = root {
            *roots.entry(root).or_default() += 1;
        }
    }
    roots
        .into_iter()
        .max_by(|(left_path, left_count), (right_path, right_count)| {
            left_count.cmp(right_count).then(
                left_path
                    .matches('/')
                    .count()
                    .cmp(&right_path.matches('/').count()),
            )
        })
        .map(|(path, _)| path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lexical_worktree_keeps_canonical_repo() {
        let space = resolve("/Users/x/src/sendbird/platform-tools/.worktree/build-civiz/web");
        assert_eq!(space.repo.as_deref(), Some("platform-tools"));
        assert_eq!(space.worktree.as_deref(), Some("build-civiz"));
        assert_eq!(label_space(&space), "platform-tools@build-civiz");
    }

    #[test]
    fn locality_score_prefers_exact_then_worktree_then_repo() {
        let anchor = resolve("/Users/x/src/sendbird/platform-tools/.worktree/build/web");
        let exact = resolve("/Users/x/src/sendbird/platform-tools/.worktree/build/web");
        let same_worktree = resolve("/Users/x/src/sendbird/platform-tools/.worktree/build/api");
        let same_repo = resolve("/Users/x/src/sendbird/platform-tools/service-catalog");
        assert_eq!(score(&exact, &anchor, &[]), 4);
        assert_eq!(score(&same_worktree, &anchor, &[]), 3);
        assert_eq!(score(&same_repo, &anchor, &[]), 2);
    }
}
