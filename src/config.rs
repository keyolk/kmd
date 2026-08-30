//! index.yml 로딩 — `~/.config/kmd/index.yml`.
//!
//! ```yaml
//! collections:
//!   wiki:
//!     path: /Users/me/wiki
//!     pattern: "**/*.md"
//!     context:
//!       "": "설명"
//!   project:
//!     path: /Users/me/src
//!     source: git-repos      # path 아래 repo들을 git ls-files로 열거
//!     axis: project
//!     store_body: false      # 본문을 store에 복사하지 않고 원본을 읽는다
//!     max_file_size: 1048576
//!     exclude: ["**/vendor/**"]
//! ```

use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
pub struct IndexConfig {
    pub collections: BTreeMap<String, Collection>,
}

/// 컬렉션의 파일 열거 방식.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Source {
    /// `path` 아래를 walkdir로 순회하며 `pattern`에 맞는 파일. 기존 동작.
    #[default]
    Glob,
    /// `path` 아래의 git repo들을 찾아 `git ls-files`로 추적 파일만 열거.
    /// gitignore된 빌드 산출물이 자동으로 빠지고, walkdir 전수 순회보다 빠르다.
    GitRepos,
}

#[derive(Debug, Deserialize)]
pub struct Collection {
    pub path: PathBuf,
    #[serde(default = "default_pattern")]
    pub pattern: String,
    /// prefix → 설명. ""(루트) 키가 컬렉션 전체 컨텍스트다.
    #[serde(default)]
    pub context: BTreeMap<String, String>,
    #[serde(default)]
    pub source: Source,
    /// 본문을 store에 복사할지. 생략 시 source 기본값(Glob=true, GitRepos=false).
    ///
    /// false면 store에는 절대경로만 남기고 인덱싱·스니펫이 원본 파일을 직접 읽는다.
    /// 코드처럼 원본이 제자리에 남아 있는 컬렉션에서 중복 저장을 피한다.
    #[serde(default)]
    pub store_body: Option<bool>,
    /// 제외할 경로 glob (컬렉션 루트 기준 상대경로에 매칭).
    #[serde(default)]
    pub exclude: Vec<String>,
    /// 이 크기를 넘는 파일은 건너뛴다.
    #[serde(default)]
    pub max_file_size: Option<u64>,
    /// `kmd global`의 축. 생략 시 컬렉션 이름으로 추론한다.
    #[serde(default)]
    pub axis: Option<String>,
}

impl Collection {
    pub fn root_context(&self) -> Option<&str> {
        self.context.get("").map(String::as_str)
    }

    /// 본문을 store에 복사할지 — 명시값이 없으면 source에서 정한다.
    pub fn stores_body(&self) -> bool {
        self.store_body
            .unwrap_or(self.source != Source::GitRepos)
    }
}

fn default_pattern() -> String {
    "**/*.md".to_string()
}

fn config_home() -> PathBuf {
    if let Ok(dir) = std::env::var("KMD_CONFIG_DIR") {
        return PathBuf::from(dir);
    }
    dirs_home().join(".config/kmd")
}

fn cache_home() -> PathBuf {
    if let Ok(dir) = std::env::var("KMD_CACHE_DIR") {
        return PathBuf::from(dir);
    }
    dirs_home().join(".cache/kmd")
}

fn dirs_home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME not set"))
}

pub fn config_path() -> PathBuf {
    config_home().join("index.yml")
}

pub fn store_path() -> PathBuf {
    cache_home().join("store.sqlite")
}

pub fn tantivy_dir() -> PathBuf {
    cache_home().join("tantivy")
}

/// GGUF 모델 보관 위치.
#[cfg(feature = "embed")]
pub fn models_dir() -> PathBuf {
    cache_home().join("models")
}

pub fn load() -> Result<IndexConfig> {
    let path = config_path();
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("cannot read {}", path.display()))?;
    let cfg: IndexConfig =
        serde_yaml::from_str(&raw).with_context(|| format!("invalid yaml: {}", path.display()))?;
    Ok(cfg)
}
