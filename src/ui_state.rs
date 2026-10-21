//! Dashboard UI state that survives a restart: the open view, the selected
//! session, the Simulator mode, and the last query typed into it.
//!
//! Stored as `$XDG_STATE_HOME/kmd/ui.json` (`~/.local/state/kmd/ui.json` when
//! unset). Writes go to a unique temp file that is renamed over the target,
//! so a crash or two dashboards saving at once never leave a torn file. The
//! dashboard writes from a worker thread, never from key handling.

use crate::sim::SimulatorMode;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Longest query kept. A pasted transcript should not turn ui.json into a
/// second history file.
const QUERY_LIMIT: usize = 4096;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct UiState {
    /// View name, lowercase: "sessions", "operations", "simulator".
    pub view: Option<String>,
    /// Full id of the selected session; restored by id, not by row index,
    /// because the session list reorders between runs.
    pub session_id: Option<String>,
    pub simulator_mode: Option<SimulatorMode>,
    pub query: String,
    /// Each pane split, only once the user has moved it: an unmoved split
    /// follows the default. A garbage value drops that ratio, not the file.
    #[serde(
        deserialize_with = "crate::split::lenient",
        skip_serializing_if = "Option::is_none"
    )]
    pub split_sessions: Option<crate::split::State>,
    #[serde(
        deserialize_with = "crate::split::lenient",
        skip_serializing_if = "Option::is_none"
    )]
    pub split_sim_wide: Option<crate::split::State>,
    #[serde(
        deserialize_with = "crate::split::lenient",
        skip_serializing_if = "Option::is_none"
    )]
    pub split_sim_stacked: Option<crate::split::State>,
}

impl UiState {
    pub fn with_query(mut self, query: &str) -> Self {
        self.query = query.chars().take(QUERY_LIMIT).collect();
        self
    }
}

/// `None` when neither XDG_STATE_HOME nor HOME is set: persistence is then
/// off rather than written somewhere surprising.
pub fn path() -> Option<PathBuf> {
    path_from(std::env::var_os("XDG_STATE_HOME"), std::env::var_os("HOME"))
}

fn path_from(
    xdg_state_home: Option<std::ffi::OsString>,
    home: Option<std::ffi::OsString>,
) -> Option<PathBuf> {
    // The XDG spec says an empty value is treated as unset.
    let base = match xdg_state_home.filter(|dir| !dir.is_empty()) {
        Some(dir) => PathBuf::from(dir),
        None => PathBuf::from(home.filter(|dir| !dir.is_empty())?).join(".local/state"),
    };
    Some(base.join("kmd/ui.json"))
}

/// A missing file is the default state, not an error.
pub fn load(path: &Path) -> Result<UiState> {
    match fs::read_to_string(path) {
        Ok(raw) => serde_json::from_str(&raw).with_context(|| format!("parse {}", path.display())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(UiState::default()),
        Err(error) => Err(error).with_context(|| format!("read {}", path.display())),
    }
}

pub fn save(path: &Path, state: &UiState) -> Result<()> {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let parent = path
        .parent()
        .with_context(|| format!("{} has no parent directory", path.display()))?;
    fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    // Unique per process and per write: an earlier save still in flight must
    // not share (and truncate) this temp file.
    let temp = parent.join(format!(
        ".ui.json.{}.{}.tmp",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let body = serde_json::to_vec_pretty(state)?;
    let written = (|| -> Result<()> {
        let mut file = fs::File::create(&temp)?;
        file.write_all(&body)?;
        file.sync_all()?;
        fs::rename(&temp, path)?;
        Ok(())
    })();
    if written.is_err() {
        // Best effort: the temp file is ours and useless once the write failed.
        fs::remove_file(&temp).ok();
    }
    written.with_context(|| format!("write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("kmd-ui-{name}-{}", std::process::id()));
        fs::remove_dir_all(&dir).ok();
        dir
    }

    #[test]
    fn round_trips_through_an_atomic_write() {
        let dir = temp_dir("roundtrip");
        let path = dir.join("kmd/ui.json");
        let state = UiState {
            view: Some("simulator".into()),
            session_id: Some("12345678-session".into()),
            simulator_mode: Some(SimulatorMode::Global),
            query: String::new(),
            split_sessions: Some(crate::split::State { ratio: 0.4 }),
            split_sim_wide: None,
            split_sim_stacked: Some(crate::split::State { ratio: 0.6 }),
        }
        .with_query("벡터 검색");
        save(&path, &state).unwrap();
        assert_eq!(load(&path).unwrap(), state);
        let leftovers: Vec<_> = fs::read_dir(path.parent().unwrap())
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "temp file left behind");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_missing_file_is_the_default_state() {
        let dir = temp_dir("missing");
        assert_eq!(load(&dir.join("ui.json")).unwrap(), UiState::default());
    }

    #[test]
    fn a_corrupt_file_is_an_error_not_a_panic() {
        let dir = temp_dir("corrupt");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ui.json");
        fs::write(&path, "{not json").unwrap();
        assert!(load(&path).is_err());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn unknown_fields_and_missing_fields_are_tolerated() {
        let state: UiState = serde_json::from_str(r#"{"view":"operations","future":1}"#).unwrap();
        assert_eq!(state.view.as_deref(), Some("operations"));
        assert!(state.query.is_empty());
    }

    #[test]
    fn path_prefers_xdg_state_home_and_falls_back_to_home() {
        assert_eq!(
            path_from(Some("/xdg".into()), Some("/home/u".into())),
            Some(PathBuf::from("/xdg/kmd/ui.json"))
        );
        assert_eq!(
            path_from(Some("".into()), Some("/home/u".into())),
            Some(PathBuf::from("/home/u/.local/state/kmd/ui.json"))
        );
        assert_eq!(path_from(None, None), None);
    }

    #[test]
    fn a_garbage_split_drops_that_ratio_and_keeps_the_rest() {
        let state: UiState = serde_json::from_str(
            r#"{"view":"simulator","split_sessions":{"ratio":"wide"},"split_sim_wide":{"ratio":0.3},"split_sim_stacked":[]}"#,
        )
        .unwrap();
        assert_eq!(state.view.as_deref(), Some("simulator"));
        assert_eq!(state.split_sessions, None);
        assert_eq!(
            state.split_sim_wide,
            Some(crate::split::State { ratio: 0.3 })
        );
        assert_eq!(state.split_sim_stacked, None);
    }

    #[test]
    fn long_queries_are_capped() {
        let state = UiState::default().with_query(&"x".repeat(QUERY_LIMIT + 10));
        assert_eq!(state.query.chars().count(), QUERY_LIMIT);
    }
}
