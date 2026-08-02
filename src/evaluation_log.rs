//! Persisted history for retrieval, utilization, and A/B evaluation runs.

use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};

const HISTORY_LIMIT: usize = 500;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationRun {
    pub timestamp: String,
    pub kind: String,
    pub summary: String,
    pub metrics: Value,
}

pub fn path() -> PathBuf {
    crate::rag::state_dir().join("evaluations.jsonl")
}

fn parse_history(raw: &str) -> Result<Vec<EvaluationRun>> {
    raw.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| {
            serde_json::from_str(line)
                .with_context(|| format!("invalid evaluation history at line {}", index + 1))
        })
        .collect()
}

fn bounded_history(raw: &str, run: &EvaluationRun) -> Result<String> {
    let mut runs = parse_history(raw)?;
    let keep = HISTORY_LIMIT.saturating_sub(1);
    if runs.len() > keep {
        runs.drain(..runs.len() - keep);
    }
    runs.push(run.clone());

    let mut output = String::new();
    for run in runs {
        output.push_str(&serde_json::to_string(&run)?);
        output.push('\n');
    }
    Ok(output)
}

fn persist(path: &Path, run: &EvaluationRun) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let lock_path = path.with_extension("jsonl.lock");
    let lock = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(lock_path)?;
    lock.lock()?;
    let raw = match fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(error) => return Err(error.into()),
    };
    let body = bounded_history(&raw, run)?;
    let pending = path.with_extension(format!("jsonl.{}.pending", std::process::id()));
    fs::write(&pending, body)?;
    if let Ok(metadata) = fs::metadata(path) {
        fs::set_permissions(&pending, metadata.permissions())?;
    }
    fs::rename(pending, path)?;
    Ok(())
}

pub fn record(kind: &str, summary: impl Into<String>, metrics: Value) {
    let run = EvaluationRun {
        timestamp: Utc::now().to_rfc3339(),
        kind: kind.to_string(),
        summary: summary.into(),
        metrics,
    };
    if let Err(error) = persist(&path(), &run) {
        eprintln!("failed to persist evaluation history: {error}");
    }
}

pub fn load(limit: usize) -> Result<Vec<EvaluationRun>> {
    let raw = match fs::read_to_string(path()) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error.into()),
    };
    let mut runs = parse_history(&raw)?;
    runs.reverse();
    runs.truncate(limit);
    Ok(runs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(index: usize) -> EvaluationRun {
        EvaluationRun {
            timestamp: format!("run-{index:03}"),
            kind: "eval-known-item".into(),
            summary: "10 queries".into(),
            metrics: serde_json::json!({"queries": 10}),
        }
    }

    #[test]
    fn bounds_history() {
        let mut raw = String::new();
        for index in 0..505 {
            raw.push_str(&serde_json::to_string(&run(index)).unwrap());
            raw.push('\n');
        }
        let latest = EvaluationRun {
            timestamp: "latest".into(),
            ..run(0)
        };
        let bounded = bounded_history(&raw, &latest).unwrap();
        let runs: Vec<EvaluationRun> = bounded
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(runs.len(), HISTORY_LIMIT);
        assert_eq!(runs.first().unwrap().timestamp, "run-006");
        assert_eq!(runs.last().unwrap().timestamp, "latest");
    }

    #[test]
    fn rejects_invalid_history() {
        let error = parse_history("invalid\n").unwrap_err();
        assert!(error.to_string().contains("line 1"));
    }
}
