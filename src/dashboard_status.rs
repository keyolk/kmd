//! The dashboard's one-line status.
//!
//! The previous `message: String` was overwritten with "refreshed" by the
//! 5-second auto-refresh, so a self-check result or a refresh failure stayed on
//! screen for at most five seconds, and every message was drawn in the same
//! muted color. Here each status carries a level and a source: results expire
//! after a few seconds, progress stays until its job lands, and an error stays
//! until the same source succeeds again or something newer replaces it.

use std::time::{Duration, Instant};

const RESULT_TTL: Duration = Duration::from_secs(5);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Info,
    Progress,
    Success,
    Warning,
    Error,
}

#[derive(Debug)]
struct Entry {
    source: &'static str,
    text: String,
    level: Level,
    until: Option<Instant>,
}

#[derive(Debug, Default)]
pub struct StatusLine {
    current: Option<Entry>,
}

impl StatusLine {
    pub fn set(&mut self, source: &'static str, level: Level, text: impl Into<String>) {
        self.set_at(source, level, text, Instant::now());
    }

    fn set_at(
        &mut self,
        source: &'static str,
        level: Level,
        text: impl Into<String>,
        now: Instant,
    ) {
        let until = match level {
            Level::Info | Level::Success => Some(now + RESULT_TTL),
            // Progress waits for its job; warnings and errors wait for the user
            // or a fix -- a failure that vanishes on a timer is easy to miss.
            Level::Progress | Level::Warning | Level::Error => None,
        };
        self.current = Some(Entry {
            source,
            text: text.into(),
            level,
            until,
        });
    }

    /// Clears the status if `source` owns it: its job finished or its error is fixed.
    pub fn resolve(&mut self, source: &'static str) {
        if self
            .current
            .as_ref()
            .is_some_and(|entry| entry.source == source)
        {
            self.current = None;
        }
    }

    /// Drops an expired status. Returns true when the screen needs a redraw.
    /// Called from the event loop, not from draw, so an idle dashboard still
    /// clears it.
    pub fn expire(&mut self, now: Instant) -> bool {
        let expired = self
            .current
            .as_ref()
            .and_then(|entry| entry.until)
            .is_some_and(|until| now >= until);
        if expired {
            self.current = None;
        }
        expired
    }

    pub fn current(&self) -> Option<(&str, Level)> {
        self.current
            .as_ref()
            .map(|entry| (entry.text.as_str(), entry.level))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn results_expire_but_errors_stay() {
        let now = Instant::now();
        let mut status = StatusLine::default();
        status.set_at("checks", Level::Success, "11/11 passed", now);
        assert!(!status.expire(now));
        assert!(status.expire(now + RESULT_TTL));
        assert!(status.current().is_none());

        status.set_at("refresh", Level::Error, "refresh failed", now);
        assert!(!status.expire(now + Duration::from_secs(3600)));
        assert_eq!(status.current(), Some(("refresh failed", Level::Error)));
    }

    #[test]
    fn resolve_only_clears_its_own_source() {
        let mut status = StatusLine::default();
        status.set("checks", Level::Error, "self-check failed");
        status.resolve("refresh");
        assert!(
            status.current().is_some(),
            "a refresh must not hide a check error"
        );
        status.resolve("checks");
        assert!(status.current().is_none());
    }
}
