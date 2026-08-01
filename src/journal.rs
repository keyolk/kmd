//! Date-oriented view joining historical learnings with live session activity.
//! Entries are ranked by temporal recency and spatial proximity to an anchor cwd.

use anyhow::Result;
use chrono::{DateTime, Duration, Local, NaiveDate};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

const NO_PROJECT: &str = "(프로젝트 없음)";

#[derive(Debug, Clone, Serialize)]
pub struct JournalEntry {
    pub date: String,
    pub time: String,
    pub project: String,
    pub projects: Vec<String>,
    pub session_id: String,
    pub source: String,
    pub summary: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub latest: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub cwd: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub workspace: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worktree: Option<String>,
    pub locality: u8,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub learning_file: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct JournalProject {
    pub name: String,
    pub locality: u8,
    pub sessions: Vec<JournalEntry>,
}

#[derive(Debug, Serialize)]
pub struct JournalDay {
    pub date: String,
    pub session_count: usize,
    pub project_count: usize,
    pub projects: Vec<JournalProject>,
}

#[derive(Debug, Serialize)]
pub struct JournalView {
    pub from: String,
    pub to: String,
    pub anchor: crate::locality::Space,
    pub days: Vec<JournalDay>,
    pub session_count: usize,
    pub project_count: usize,
}

fn normalize_date(value: &str) -> Option<String> {
    let digits: String = value.chars().filter(char::is_ascii_digit).collect();
    if digits.len() != 8 {
        return None;
    }
    NaiveDate::parse_from_str(&digits, "%Y%m%d")
        .ok()
        .map(|date| date.format("%Y%m%d").to_string())
}

fn display_date(value: &str) -> String {
    NaiveDate::parse_from_str(value, "%Y%m%d")
        .map(|date| date.format("%Y-%m-%d").to_string())
        .unwrap_or_else(|_| value.to_string())
}

fn short_file(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

fn resolve_workspace(cwd: &str, files: &[String]) -> (crate::locality::Space, String) {
    let direct = crate::locality::resolve(cwd);
    if direct.repo.is_some() {
        return (direct, cwd.to_string());
    }
    if let Some(inferred) = crate::locality::infer_cwd(files) {
        return (crate::locality::resolve(&inferred), inferred);
    }
    (direct, cwd.to_string())
}

fn project_names(
    repos: impl IntoIterator<Item = String>,
    space: &crate::locality::Space,
    files: &[String],
) -> Vec<String> {
    let fallback: BTreeSet<String> = repos.into_iter().collect();
    let mut canonical = BTreeSet::new();
    if let Some(repo) = &space.repo {
        canonical.insert(repo.clone());
    }
    for file in files {
        if let Some(repo) = crate::locality::resolve(file).repo {
            canonical.insert(repo);
        }
    }

    if canonical.is_empty() {
        fallback.into_iter().collect()
    } else {
        canonical.into_iter().collect()
    }
}

fn matches_project(
    projects: &[String],
    space: &crate::locality::Space,
    filter: Option<&str>,
) -> bool {
    filter.is_none_or(|filter| {
        projects
            .iter()
            .any(|project| project == filter || project.contains(filter))
            || space
                .worktree
                .as_deref()
                .is_some_and(|worktree| worktree.contains(filter))
            || crate::locality::label_space(space).contains(filter)
    })
}

fn select_project(
    projects: &[String],
    space: &crate::locality::Space,
    filter: Option<&str>,
) -> String {
    if let Some(repo) = &space.repo {
        return repo.clone();
    }
    if let Some(filter) = filter {
        if let Some(project) = projects.iter().find(|project| project.contains(filter)) {
            return project.clone();
        }
    }
    projects
        .first()
        .cloned()
        .unwrap_or_else(|| NO_PROJECT.to_string())
}

fn historical_entries(
    project: Option<&str>,
    anchor: &crate::locality::Space,
) -> Result<Vec<JournalEntry>> {
    Ok(crate::pageindex::load_learning_history()?
        .into_iter()
        .filter_map(|session| {
            if session.date.len() != 8 {
                return None;
            }
            let (space, workspace) = resolve_workspace(&session.cwd, &session.files);
            let projects = project_names(session.repos, &space, &session.files);
            if !matches_project(&projects, &space, project) {
                return None;
            }
            let locality = crate::locality::score(&space, anchor, &session.files);
            let selected = select_project(&projects, &space, project);
            let latest = if session.latest_summary != session.summary {
                session.latest_summary
            } else {
                String::new()
            };
            Some(JournalEntry {
                date: session.date,
                time: session.time,
                project: selected,
                projects,
                session_id: session.id8,
                source: "learning".to_string(),
                summary: session.summary,
                latest,
                cwd: session.cwd,
                workspace,
                worktree: space.worktree,
                locality,
                files: session.files.into_iter().rev().take(3).collect(),
                learning_file: Some(session.file),
            })
        })
        .collect())
}

fn live_entries(
    project: Option<&str>,
    anchor: &crate::locality::Space,
) -> Result<Vec<JournalEntry>> {
    Ok(crate::activity::recent("", "", usize::MAX)?
        .into_iter()
        .filter_map(|card| {
            let updated = DateTime::parse_from_rfc3339(&card.updated_at).ok()?;
            let local = updated.with_timezone(&Local);
            let (space, workspace) = resolve_workspace(&card.cwd, &card.files);
            let projects = project_names(card.repo.into_iter(), &space, &card.files);
            if !matches_project(&projects, &space, project) {
                return None;
            }
            let locality = crate::locality::score(&space, anchor, &card.files);
            let selected = select_project(&projects, &space, project);
            let id8: String = card.session_id.chars().take(8).collect();
            let latest = if card.latest_user != card.summary {
                card.latest_user
            } else {
                String::new()
            };
            Some(JournalEntry {
                date: local.format("%Y%m%d").to_string(),
                time: local.format("%H:%M").to_string(),
                project: selected,
                projects,
                session_id: id8,
                source: "live".to_string(),
                summary: card.summary,
                latest,
                cwd: card.cwd,
                workspace,
                worktree: space.worktree,
                locality,
                files: card.files.into_iter().rev().take(3).collect(),
                learning_file: None,
            })
        })
        .collect())
}

fn merge_live(historical: Option<JournalEntry>, mut live: JournalEntry) -> JournalEntry {
    if let Some(historical) = historical {
        live.source = "live+learning".to_string();
        live.learning_file = historical.learning_file;
        if live.files.is_empty() {
            live.files = historical.files;
        }
        if live.projects.is_empty() {
            live.projects = historical.projects;
        }
    }
    live
}

pub fn build(
    days: u32,
    date: Option<&str>,
    project: Option<&str>,
    cwd: &str,
) -> Result<JournalView> {
    let today = Local::now().date_naive();
    let (from, to) = if let Some(value) = date {
        let normalized = normalize_date(value).ok_or_else(|| {
            anyhow::anyhow!("invalid date '{}': use YYYY-MM-DD or YYYYMMDD", value)
        })?;
        (normalized.clone(), normalized)
    } else {
        let span = days.max(1);
        let from = today - Duration::days(i64::from(span - 1));
        (
            from.format("%Y%m%d").to_string(),
            today.format("%Y%m%d").to_string(),
        )
    };
    let anchor = crate::locality::resolve(cwd);

    let mut entries: BTreeMap<(String, String), JournalEntry> = BTreeMap::new();
    for entry in historical_entries(project, &anchor)? {
        if entry.date >= from && entry.date <= to {
            entries.insert((entry.date.clone(), entry.session_id.clone()), entry);
        }
    }
    for live in live_entries(project, &anchor)? {
        if live.date >= from && live.date <= to {
            let key = (live.date.clone(), live.session_id.clone());
            let historical = entries.remove(&key);
            entries.insert(key, merge_live(historical, live));
        }
    }

    let mut grouped: BTreeMap<String, BTreeMap<String, Vec<JournalEntry>>> = BTreeMap::new();
    for entry in entries.into_values() {
        grouped
            .entry(entry.date.clone())
            .or_default()
            .entry(entry.project.clone())
            .or_default()
            .push(entry);
    }

    let mut all_projects = BTreeSet::new();
    let mut journal_days = Vec::new();
    let mut session_count = 0usize;
    for (date, projects) in grouped.into_iter().rev() {
        let mut rendered_projects = Vec::new();
        let mut day_sessions = 0usize;
        for (name, mut sessions) in projects {
            sessions.sort_by(|a, b| {
                b.locality
                    .cmp(&a.locality)
                    .then(b.time.cmp(&a.time))
                    .then(a.session_id.cmp(&b.session_id))
            });
            let locality = sessions
                .iter()
                .map(|session| session.locality)
                .max()
                .unwrap_or(0);
            day_sessions += sessions.len();
            all_projects.insert(name.clone());
            rendered_projects.push(JournalProject {
                name,
                locality,
                sessions,
            });
        }
        rendered_projects.sort_by(|a, b| {
            b.locality
                .cmp(&a.locality)
                .then(b.sessions.len().cmp(&a.sessions.len()))
                .then(a.name.cmp(&b.name))
        });
        session_count += day_sessions;
        journal_days.push(JournalDay {
            date,
            session_count: day_sessions,
            project_count: rendered_projects.len(),
            projects: rendered_projects,
        });
    }

    Ok(JournalView {
        from,
        to,
        anchor,
        days: journal_days,
        session_count,
        project_count: all_projects.len(),
    })
}

fn truncate(value: &str, limit: usize) -> String {
    if value.chars().count() <= limit {
        value.to_string()
    } else {
        format!("{}…", value.chars().take(limit).collect::<String>())
    }
}

fn locality_marker(score: u8) -> &'static str {
    match score {
        4 => "====",
        3 => "===",
        2 => "==",
        1 => "=",
        _ => "-",
    }
}

pub fn render(view: &JournalView) -> String {
    let anchor = crate::locality::label_space(&view.anchor);
    let mut output = format!(
        "kmd journal — {} ~ {} · {} sessions · {} projects\nanchor: {}  {}\n",
        display_date(&view.from),
        display_date(&view.to),
        view.session_count,
        view.project_count,
        anchor,
        view.anchor.cwd
    );
    if view.days.is_empty() {
        output.push_str("\n조건에 맞는 learning 또는 session activity가 없습니다.\n");
        return output;
    }

    for day in &view.days {
        output.push_str(&format!(
            "\n{}  {} sessions · {} projects\n",
            display_date(&day.date),
            day.session_count,
            day.project_count
        ));
        for project in &day.projects {
            output.push_str(&format!(
                "  {} {} ({})\n",
                locality_marker(project.locality),
                project.name,
                project.sessions.len()
            ));
            for session in &project.sessions {
                let time = if session.time.is_empty() {
                    "--:--"
                } else {
                    &session.time
                };
                let marker = if session.source.starts_with("live") {
                    "●"
                } else {
                    "○"
                };
                output.push_str(&format!(
                    "    {} {} {:<4} {} {}\n",
                    marker,
                    time,
                    locality_marker(session.locality),
                    session.session_id,
                    truncate(&session.summary, 100)
                ));
                if !session.latest.is_empty() {
                    output.push_str(&format!("      현재: {}\n", truncate(&session.latest, 110)));
                }
                if let Some(worktree) = &session.worktree {
                    output.push_str(&format!("      worktree: {}\n", worktree));
                }
                if !session.workspace.is_empty() && session.workspace != session.cwd {
                    output.push_str(&format!("      workspace: {}\n", session.workspace));
                }
                if !session.cwd.is_empty() {
                    output.push_str(&format!("      cwd: {}\n", session.cwd));
                }
                if session.projects.len() > 1 {
                    output.push_str(&format!("      관련: {}\n", session.projects.join(", ")));
                }
                if !session.files.is_empty() {
                    let files = session
                        .files
                        .iter()
                        .map(|path| short_file(path))
                        .collect::<Vec<_>>();
                    output.push_str(&format!("      파일: {}\n", files.join(", ")));
                }
                if let Some(file) = &session.learning_file {
                    output.push_str(&format!(
                        "      learning: ~/.claude/kmd-learnings/{}\n",
                        file
                    ));
                }
            }
        }
    }
    output.push_str(
        "\n● live  ○ learning  locality: ==== exact cwd · === worktree · == repo · = files\n",
    );
    output
}

pub fn run(
    days: u32,
    date: Option<&str>,
    project: Option<&str>,
    cwd: Option<&str>,
    json: bool,
) -> Result<()> {
    let anchor = match cwd {
        Some(cwd) => cwd.to_string(),
        None => std::env::current_dir()?.display().to_string(),
    };
    let view = build(days, date, project, &anchor)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&view)?);
    } else {
        print!("{}", render(&view));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_compact_and_dashed_dates() {
        assert_eq!(normalize_date("20260801").as_deref(), Some("20260801"));
        assert_eq!(normalize_date("2026-08-01").as_deref(), Some("20260801"));
        assert_eq!(normalize_date("2026-13-01"), None);
    }

    #[test]
    fn project_filter_matches_repo_and_worktree() {
        let space = crate::locality::resolve(
            "/Users/x/src/sendbird/platform-tools/.worktree/build-civiz/web",
        );
        let projects = vec!["platform-tools".to_string()];
        assert!(matches_project(&projects, &space, Some("platform-tools")));
        assert!(matches_project(&projects, &space, Some("build-civiz")));
        assert!(!matches_project(&projects, &space, Some("soda-k8s")));
    }
}
