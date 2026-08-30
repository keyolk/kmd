//! `kmd global` — knowledge / session / project 세 축을 한 쿼리로 검색.
//!
//! 단일 랭킹으로 병합하지 않고 축별 쿼터로 나누는 이유: BM25 점수 스케일이 축마다
//! 다르다. 코드는 식별자가 반복돼 tf가 높고, 세션 요약은 짧다. 한 리스트로 섞으면
//! 한 축이 상위를 독식해 나머지가 보이지 않는다.
//!
//! rag 자동 주입 경로(`rag.rs`의 `CLAUDE_COLLECTIONS`)와는 무관하다. project 축은
//! 여기서 명시적으로 호출할 때만 나온다.

use crate::bm25::{self, SearchHit};
use crate::config::IndexConfig;
use anyhow::Result;
use serde::Serialize;
use std::path::Path;

pub const KNOWLEDGE: &str = "knowledge";
pub const SESSION: &str = "session";
pub const PROJECT: &str = "project";

/// 표시 순서 — 좁은 것부터 넓은 것으로.
pub const AXES: &[&str] = &[SESSION, KNOWLEDGE, PROJECT];

#[derive(Debug, Serialize)]
pub struct AxisResults {
    pub axis: String,
    pub hits: Vec<SearchHit>,
}

/// 컬렉션이 속한 축. `index.yml`의 `axis:`가 우선이고, 없으면 이름으로 추론한다.
///
/// 추론 규칙은 기존 컬렉션들이 axis 없이도 동작하게 하려는 것이다 — learnings는
/// 세션 기록이고 나머지 claude-*/wiki는 지식이다.
pub fn axis_of(name: &str, cfg: &IndexConfig) -> String {
    if let Some(coll) = cfg.collections.get(name)
        && let Some(a) = &coll.axis
    {
        return a.clone();
    }
    match name {
        "learnings" | "narwhal-runs" => SESSION.to_string(),
        _ => KNOWLEDGE.to_string(),
    }
}

/// 한 축에 속한 컬렉션 이름들.
pub fn collections_for<'a>(axis: &str, cfg: &'a IndexConfig) -> Vec<&'a str> {
    cfg.collections
        .keys()
        .filter(|name| axis_of(name, cfg) == axis)
        .map(String::as_str)
        .collect()
}

/// 축별로 상위 `per_axis`건씩. `only`가 주어지면 그 축만.
pub fn search(
    dir: &Path,
    cfg: &IndexConfig,
    query: &str,
    per_axis: usize,
    only: Option<&str>,
) -> Result<Vec<AxisResults>> {
    let axes: Vec<&str> = match only {
        Some(a) => vec![a],
        None => AXES.to_vec(),
    };

    let mut out = Vec::new();
    for axis in axes {
        let colls = collections_for(axis, cfg);
        if colls.is_empty() {
            continue;
        }
        let hits = bm25::search_in(dir, cfg, query, per_axis, Some(&colls))?;
        out.push(AxisResults {
            axis: axis.to_string(),
            hits,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_from(yaml: &str) -> IndexConfig {
        serde_yaml::from_str(yaml).expect("valid yaml")
    }

    #[test]
    fn axis_falls_back_to_name_when_unset() {
        let cfg = cfg_from(
            r#"
collections:
  wiki:
    path: /tmp/wiki
  learnings:
    path: /tmp/learnings
"#,
        );
        assert_eq!(axis_of("wiki", &cfg), KNOWLEDGE);
        assert_eq!(axis_of("learnings", &cfg), SESSION);
    }

    #[test]
    fn explicit_axis_wins_over_name() {
        let cfg = cfg_from(
            r#"
collections:
  learnings:
    path: /tmp/learnings
    axis: knowledge
"#,
        );
        assert_eq!(axis_of("learnings", &cfg), KNOWLEDGE);
    }

    #[test]
    fn unknown_collection_defaults_to_knowledge() {
        let cfg = cfg_from("collections: {}");
        assert_eq!(axis_of("nonexistent", &cfg), KNOWLEDGE);
    }

    #[test]
    fn collections_group_by_axis() {
        let cfg = cfg_from(
            r#"
collections:
  wiki:
    path: /tmp/wiki
  claude-memory:
    path: /tmp/mem
  learnings:
    path: /tmp/learnings
  project:
    path: /tmp/src
    axis: project
"#,
        );
        let mut knowledge = collections_for(KNOWLEDGE, &cfg);
        knowledge.sort();
        assert_eq!(knowledge, vec!["claude-memory", "wiki"]);
        assert_eq!(collections_for(SESSION, &cfg), vec!["learnings"]);
        assert_eq!(collections_for(PROJECT, &cfg), vec!["project"]);
    }

    #[test]
    fn axis_with_no_collections_is_absent() {
        // project 컬렉션을 아직 설정하지 않은 사용자 — 그 축은 건너뛴다.
        let cfg = cfg_from(
            r#"
collections:
  wiki:
    path: /tmp/wiki
"#,
        );
        assert!(collections_for(PROJECT, &cfg).is_empty());
    }
}
