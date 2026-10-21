//! URLs reachable from the dashboard, the fleet convention (tui-design E11):
//! `o` opens the selected item's URL.
//!
//! A port of tuikit `link` (Go), which Rust cannot import; tmc carries the
//! same module. Keep them in step: same validity rule, same URL finding, same
//! notes.
//!
//! The dashboard draws no OSC 8 hyperlinks, so `wrap` is not ported. Every URL
//! it shows is printed in full (session text, hit snippets); none hides behind
//! a label. Linking the long ones that wrap across rows would need ratatui
//! 0.30's `CellDiffOption::ForcedWidth`: under 0.29 a cell holding the escape
//! is measured as text and the diff skips the cell after it.
//!
//! Only absolute http and https URLs are opened. The URL comes from data (a
//! prompt, a retrieved note), and a `file:`, `javascript:` or custom-scheme
//! URL is not something a key in a TUI should launch.

use std::borrow::Cow;
use std::fmt;

/// The longest URL that is linked. Terminals cap OSC 8 URIs (VTE at 2083
/// bytes) and drop longer ones, sometimes with the text.
pub const MAX_URL: usize = 2048;

/// Whether `u` is an absolute http or https URL with a host and nothing a
/// terminal escape could be smuggled through (control characters, spaces).
pub fn valid(u: &str) -> bool {
    if u.is_empty() || u.len() > MAX_URL {
        return false;
    }
    if u.bytes().any(|c| c <= b' ' || c == 0x7f) {
        return false;
    }
    let Some((scheme, rest)) = u.split_once("://") else {
        return false;
    };
    if !scheme.eq_ignore_ascii_case("http") && !scheme.eq_ignore_ascii_case("https") {
        return false;
    }
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    let host_port = authority.rsplit('@').next().unwrap_or("");
    let host = match host_port.strip_prefix('[') {
        // An IPv6 literal: the port, if any, follows the bracket.
        Some(v6) => v6.split(']').next().unwrap_or(""),
        None => host_port.split(':').next().unwrap_or(""),
    };
    !host.is_empty()
}

/// `u` with every non-ASCII byte percent-encoded, as browsers do: the same
/// URL to a browser, and an argument no opener can misread.
fn ascii_url(u: &str) -> Cow<'_, str> {
    if u.is_ascii() {
        return Cow::Borrowed(u);
    }
    let mut out = String::with_capacity(u.len() * 3);
    for c in u.bytes() {
        if c.is_ascii() {
            out.push(c as char);
        } else {
            out.push_str(&format!("%{c:02X}"));
        }
    }
    Cow::Owned(out)
}

/// The http and https URLs in `text`, in order, escapes ignored. Trailing
/// sentence punctuation is not part of a URL ("see https://x.io/a."), and
/// neither is a closing bracket that has no opening one in the URL
/// ("(https://x.io/a)").
pub fn find(text: &str) -> Vec<String> {
    spans(&strip_escapes(text))
        .into_iter()
        .map(|(_, u)| u.to_string())
        .collect()
}

/// `find` on plain text, with each URL's byte offset.
fn spans(text: &str) -> Vec<(usize, &str)> {
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(at) = text[from..].find("http").map(|i| from + i) {
        let rest = &text[at..];
        if !rest.starts_with("http://") && !rest.starts_with("https://") {
            from = at + 4;
            continue;
        }
        let len = rest
            .find(|c: char| c.is_whitespace() || matches!(c, '<' | '>' | '"' | '\'' | '`'))
            .unwrap_or(rest.len());
        let u = trim_url(&rest[..len]);
        if valid(u) {
            out.push((at, u));
        }
        from = at + len.max(4);
    }
    out
}

fn trim_url(mut u: &str) -> &str {
    while let Some(last) = u.chars().last() {
        let count = |c: char| u.matches(c).count();
        let cut = ".,;:!?".contains(last)
            || (last == ')' && count('(') < count(')'))
            || (last == ']' && count('[') < count(']'));
        if !cut {
            break;
        }
        u = &u[..u.len() - 1];
    }
    u
}

/// `text` without CSI (`ESC [ … final`) and OSC (`ESC ] … BEL/ST`) escapes.
fn strip_escapes(text: &str) -> Cow<'_, str> {
    if !text.contains('\x1b') {
        return Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\x1b' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('[') => {
                for c in chars.by_ref() {
                    if ('\x40'..='\x7e').contains(&c) {
                        break;
                    }
                }
            }
            Some(']') => {
                while let Some(c) = chars.next() {
                    if c == '\x07' {
                        break;
                    }
                    if c == '\x1b' && chars.peek() == Some(&'\\') {
                        chars.next();
                        break;
                    }
                }
            }
            _ => {}
        }
    }
    Cow::Owned(out)
}

/// Why a URL was not opened.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OpenError {
    /// Not an http or https URL (see `valid`).
    Invalid,
    /// Running over SSH: the browser would open on the remote host, not in
    /// front of the user.
    Remote,
    /// The platform has no known URL opener.
    NoOpener,
    /// The opener could not be started.
    Spawn(String),
    /// There was no URL to open. Said rather than doing nothing, so the key
    /// always answers.
    NoLink,
}

impl fmt::Display for OpenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpenError::Invalid => f.write_str("not an http(s) URL"),
            OpenError::Remote => f.write_str("no local browser over SSH"),
            OpenError::NoOpener => f.write_str("no URL opener on this platform"),
            OpenError::Spawn(e) => write!(f, "start the opener: {e}"),
            OpenError::NoLink => f.write_str("no link"),
        }
    }
}

/// Opens a URL in the browser. The app holds one so tests can put a recorder
/// in its place; `DEFAULT` is the real one, except under `cfg(test)`.
pub type Launcher = fn(&str) -> Result<(), OpenError>;

/// What the app starts with. Under test a launch nobody stubbed fails the test
/// instead of opening a browser on the developer's screen.
#[cfg(not(test))]
pub const DEFAULT: Launcher = launch;
#[cfg(test)]
pub const DEFAULT: Launcher = stub::unexpected;

/// The result of `open_first`, and the status line it earns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Opened {
    pub url: String,
    pub err: Option<OpenError>,
    /// How many URLs it chose from; above 1 the note says "the first of N" so
    /// the user knows to look.
    pub of: usize,
}

impl Opened {
    /// "opened <url>", "opened <url> (the first of 3 links)", or why not.
    pub fn note(&self) -> String {
        match &self.err {
            Some(OpenError::NoLink) => "no link here".into(),
            Some(e) => format!("open failed: {e}"),
            None if self.of > 1 => format!("opened {} (the first of {} links)", self.url, self.of),
            None => format!("opened {}", self.url),
        }
    }
}

/// Open the first valid URL of `urls` with `launcher` and say how many there
/// were. Invalid URLs are skipped, never handed to the launcher; with none
/// left the result is `NoLink`.
pub fn open_first(launcher: Launcher, urls: &[String]) -> Opened {
    let valid: Vec<&String> = urls.iter().filter(|u| valid(u)).collect();
    let Some(first) = valid.first() else {
        return Opened {
            url: String::new(),
            err: Some(OpenError::NoLink),
            of: 0,
        };
    };
    Opened {
        url: first.to_string(),
        err: launcher(first).err(),
        of: valid.len(),
    }
}

/// The real launcher: the platform opener with the URL as one argument,
/// never through a shell, with no terminal attached so it cannot draw over
/// the TUI.
// Unreferenced under test, where `DEFAULT` is the stub that fails instead.
#[cfg_attr(test, allow(dead_code))]
pub fn launch(u: &str) -> Result<(), OpenError> {
    launch_in(u, &|key| {
        std::env::var_os(key).is_some_and(|v| !v.is_empty())
    })
}

fn launch_in(u: &str, env_set: &dyn Fn(&str) -> bool) -> Result<(), OpenError> {
    use std::process::{Command, Stdio};

    if !valid(u) {
        return Err(OpenError::Invalid);
    }
    if env_set("SSH_TTY") || env_set("SSH_CONNECTION") {
        return Err(OpenError::Remote);
    }
    let opener = match std::env::consts::OS {
        "macos" => "open",
        "linux" | "freebsd" | "openbsd" | "netbsd" => "xdg-open",
        _ => return Err(OpenError::NoOpener),
    };
    // `valid` holds, so the argument starts with "http" and cannot be read
    // as a flag.
    let mut child = Command::new(opener)
        .arg(ascii_url(u).as_ref())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| OpenError::Spawn(e.to_string()))?;
    // Reaped off the input thread so the key never waits on the browser. The
    // exit status is not reported: the opener has already been handed the
    // URL, and a status arriving seconds later would land on whatever the
    // user is doing by then.
    std::thread::spawn(move || child.wait().map(drop));
    Ok(())
}

/// Test launchers: one that records, one that fails the test.
#[cfg(test)]
pub mod stub {
    use std::cell::RefCell;

    use super::OpenError;

    thread_local! {
        static OPENED: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    }

    /// Records the URL instead of opening it. Launches happen on the thread
    /// that handled the key, which is the test's own.
    pub fn record(u: &str) -> Result<(), OpenError> {
        OPENED.with(|o| o.borrow_mut().push(u.to_string()));
        Ok(())
    }

    /// The URLs `record` saw, emptying the record.
    pub fn take() -> Vec<String> {
        OPENED.with(|o| std::mem::take(&mut *o.borrow_mut()))
    }

    pub fn unexpected(u: &str) -> Result<(), OpenError> {
        panic!("a test reached the real launcher with {u}; set the launcher to stub::record")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_takes_only_http_with_a_host() {
        let long = format!("https://x.io/{}", "a".repeat(MAX_URL));
        for (u, want) in [
            ("https://github.com/keyolk/tuikit/pull/9", true),
            ("http://localhost:8080/x?a=1&b=2", true),
            ("https://x.io/path;with;semis", true),
            ("HTTPS://x.io/a", true),
            ("https://user@x.io:8443/a", true),
            ("http://[::1]:8080/", true),
            ("", false),
            ("github.com/keyolk", false),
            ("https://", false),
            ("https:///path", false),
            ("file:///etc/passwd", false),
            ("javascript:alert(1)", false),
            ("ssh://host/x", false),
            ("https://x.io/\x1b]8;;evil\x1b\\", false),
            ("https://x.io/a b", false),
            (long.as_str(), false),
        ] {
            assert_eq!(valid(u), want, "valid({u:?})");
        }
    }

    #[test]
    fn a_non_ascii_url_is_percent_encoded_for_the_opener() {
        assert_eq!(
            ascii_url("https://x.io/한글-서비스?q=값"),
            "https://x.io/%ED%95%9C%EA%B8%80-%EC%84%9C%EB%B9%84%EC%8A%A4?q=%EA%B0%92"
        );
        assert!(matches!(ascii_url("https://x.io/a"), Cow::Borrowed(_)));
    }

    #[test]
    fn find_strips_punctuation_unmatched_brackets_and_escapes() {
        let text = "see https://x.io/a. and (https://en.wikipedia.org/wiki/Go_(game)) or \
                    \x1b[1mhttps://y.io/b?q=1\x1b[0m, not ftp://z.io or file:///x; [https://z.io/c]";
        assert_eq!(
            find(text),
            [
                "https://x.io/a",
                "https://en.wikipedia.org/wiki/Go_(game)",
                "https://y.io/b?q=1",
                "https://z.io/c",
            ]
        );
        assert!(find("httpx http:// nothing here").is_empty());
    }

    #[test]
    fn spans_give_byte_offsets() {
        let text = "한글 https://x.io/a 끝";
        assert_eq!(spans(text), [(7, "https://x.io/a")]);
    }

    #[test]
    fn open_first_says_how_many_and_skips_bad_urls() {
        let urls = ["javascript:x", "https://x.io/a", "https://x.io/b"].map(String::from);
        let m = open_first(stub::record, &urls);
        assert_eq!(m.url, "https://x.io/a");
        assert_eq!(m.of, 2);
        assert_eq!(m.err, None);
        assert_eq!(m.note(), "opened https://x.io/a (the first of 2 links)");
        assert_eq!(stub::take(), ["https://x.io/a"]);

        let one = open_first(stub::record, &["https://x.io/c".into()]);
        assert_eq!(one.note(), "opened https://x.io/c");
        stub::take();

        // Nothing valid: the launcher is never reached (it would panic).
        let none = open_first(stub::unexpected, &["file:///etc/passwd".into()]);
        assert_eq!(none.err, Some(OpenError::NoLink));
        assert_eq!(none.note(), "no link here");
    }

    #[test]
    fn launch_refuses_bad_urls_and_ssh_before_running_anything() {
        let never = |_: &str| false;
        for u in ["file:///etc/passwd", "-a Calculator", "x-man-page://ls", ""] {
            assert_eq!(launch_in(u, &never), Err(OpenError::Invalid), "{u:?}");
        }
        let ssh = |k: &str| k == "SSH_CONNECTION";
        assert_eq!(
            launch_in("https://x.io/a", &ssh),
            Err(OpenError::Remote),
            "the browser would open on the remote host"
        );
        assert_eq!(
            Opened {
                url: "https://x.io/a".into(),
                err: Some(OpenError::Remote),
                of: 1
            }
            .note(),
            "open failed: no local browser over SSH"
        );
    }
}
