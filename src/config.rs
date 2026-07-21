//! index.yml 로딩 — `~/.config/kmd/index.yml`. (이전 qmd 공유 경로에서 이관.)
//!
//! ```yaml
//! collections:
//!   wiki:
//!     path: /Users/me/wiki
//!     pattern: "**/*.md"
//!     context:
//!       "": "설명"
//! ```

use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
pub struct IndexConfig {
    pub collections: BTreeMap<String, Collection>,
}

#[derive(Debug, Deserialize)]
pub struct Collection {
    pub path: PathBuf,
    #[serde(default = "default_pattern")]
    pub pattern: String,
    /// prefix → 설명. qmd는 ""(루트) 키를 컬렉션 전체 컨텍스트로 사용.
    #[serde(default)]
    pub context: BTreeMap<String, String>,
}

impl Collection {
    pub fn root_context(&self) -> Option<&str> {
        self.context.get("").map(String::as_str)
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

pub fn load() -> Result<IndexConfig> {
    let path = config_path();
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("cannot read {}", path.display()))?;
    let cfg: IndexConfig =
        serde_yaml::from_str(&raw).with_context(|| format!("invalid yaml: {}", path.display()))?;
    Ok(cfg)
}
