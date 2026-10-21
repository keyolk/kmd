//! Clipboard writes for a finished selection, off the UI thread.

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::mpsc::Sender;

/// The result of a background copy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Copied {
    pub chars: usize,
    pub error: Option<String>,
}

impl Copied {
    /// The toast text: what was copied, or why it was not.
    pub fn message(&self) -> String {
        match &self.error {
            None => format!("copied {} chars", self.chars),
            Some(e) => format!("copy failed: {e}"),
        }
    }
}

/// Put `text` on the clipboard off the UI thread and report on `done`.
/// Locally the platform tool (pbcopy, wl-copy, xclip) is used, which never
/// writes to the terminal ratatui is drawing on. Over SSH, or with no tool,
/// it falls back to OSC 52 on /dev/tty, as the engine does; tmux forwards it
/// with set-clipboard on.
pub fn copy_in_background(text: String, done: Sender<Copied>) {
    std::thread::spawn(move || {
        let error = write_clipboard(&text).err().map(|e| e.to_string());
        // The receiver is gone only when the app is quitting; nobody is left
        // to show the result to.
        done.send(Copied {
            chars: text.chars().count(),
            error,
        })
        .ok();
    });
}

fn write_clipboard(text: &str) -> std::io::Result<()> {
    let ssh = std::env::var_os("SSH_TTY").is_some() || std::env::var_os("SSH_CONNECTION").is_some();
    if !ssh {
        let tools: &[&[&str]] = if cfg!(target_os = "macos") {
            &[&["pbcopy"]]
        } else {
            &[
                &["wl-copy"],
                &["xclip", "-selection", "clipboard"],
                &["xsel", "--clipboard", "--input"],
            ]
        };
        for argv in tools {
            let child = Command::new(argv[0])
                .args(&argv[1..])
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn();
            let mut child = match child {
                Ok(c) => c,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => return Err(e),
            };
            if let Some(mut stdin) = child.stdin.take() {
                stdin.write_all(text.as_bytes())?;
            }
            let status = child.wait()?;
            if status.success() {
                return Ok(());
            }
            return Err(std::io::Error::other(format!(
                "{} exited {status}",
                argv[0]
            )));
        }
    }
    let mut tty = std::fs::OpenOptions::new().write(true).open("/dev/tty")?;
    tty.write_all(osc52(text).as_bytes())?;
    tty.flush()
}

/// The clipboard-write escape for `text`.
pub fn osc52(text: &str) -> String {
    format!("\x1b]52;c;{}\x07", base64(text.as_bytes()))
}

fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            T[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            T[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn osc52_is_base64_of_the_text() {
        assert_eq!(osc52("hi!"), "\x1b]52;c;aGkh\x07");
        assert_eq!(base64(b"ab"), "YWI=");
        assert_eq!(base64(b"a"), "YQ==");
    }
}
