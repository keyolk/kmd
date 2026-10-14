//! Background jobs for the dashboard.
//!
//! A dashboard snapshot opens the store, parses a week of rag.jsonl, builds the
//! journal, and pings the daemon with a 10s read timeout; the self-check suite
//! also opens tantivy and runs a search. Running those on the input thread froze
//! every key for as long as they took, every five seconds. Jobs here run on a
//! worker thread and hand their result back through a channel the event loop
//! polls.

use std::panic::{self, AssertUnwindSafe};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

type Message<T> = (u64, Result<T, String>);

/// One kind of background job: at most one runs at a time, and a result only
/// lands if no newer request superseded it.
pub struct Background<T> {
    tx: Sender<Message<T>>,
    rx: Receiver<Message<T>>,
    generation: u64,
    in_flight: bool,
}

impl<T: Send + 'static> Background<T> {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            tx,
            rx,
            generation: 0,
            in_flight: false,
        }
    }

    pub fn in_flight(&self) -> bool {
        self.in_flight
    }

    /// Starts `job` on a worker thread. Returns false, and starts nothing, while
    /// the previous job is still running: a slow refresh must not pile up a new
    /// thread every tick.
    pub fn start<F>(&mut self, job: F) -> bool
    where
        F: FnOnce() -> anyhow::Result<T> + Send + 'static,
    {
        if self.in_flight {
            return false;
        }
        self.generation += 1;
        self.in_flight = true;
        let generation = self.generation;
        let tx = self.tx.clone();
        thread::spawn(move || {
            // The receiver lives as long as the dashboard; a failed send only
            // means the dashboard already exited and nobody is waiting.
            let _ = tx.send((generation, run_caught(job)));
        });
        true
    }

    /// Starts `job` even while an older one runs, and drops the older one's
    /// result when it lands. For interactive queries: the user edited the query
    /// and pressed Enter again, so the answer to the old text is no longer
    /// wanted. The old thread is not killed (std threads cannot be); it runs to
    /// completion and its send is ignored by `poll`.
    pub fn replace<F>(&mut self, job: F)
    where
        F: FnOnce() -> anyhow::Result<T> + Send + 'static,
    {
        self.in_flight = false;
        self.start(job);
    }

    /// Forgets the running job: its result will be discarded. Used when the
    /// state the job was answering for changes under it (mode switch, cleared
    /// query).
    pub fn cancel(&mut self) {
        self.generation += 1;
        self.in_flight = false;
    }

    /// Takes the finished job's result, if any. Results from superseded
    /// generations are discarded.
    pub fn poll(&mut self) -> Option<Result<T, String>> {
        let mut latest = None;
        while let Ok((generation, result)) = self.rx.try_recv() {
            if generation == self.generation {
                self.in_flight = false;
                latest = Some(result);
            }
        }
        latest
    }

    /// Blocks until the current job lands or `timeout` passes. Only for exit
    /// paths and tests; the event loop polls.
    pub fn wait(&mut self, timeout: std::time::Duration) -> Option<Result<T, String>> {
        let deadline = std::time::Instant::now() + timeout;
        while std::time::Instant::now() < deadline {
            if let Some(result) = self.poll() {
                return Some(result);
            }
            thread::sleep(std::time::Duration::from_millis(5));
        }
        None
    }
}

/// Runs `job`, turning both an error and a panic into an error string. Without
/// this a panicking worker never sends, and the caller's "running" flag sticks.
pub fn run_caught<T>(job: impl FnOnce() -> anyhow::Result<T>) -> Result<T, String> {
    match panic::catch_unwind(AssertUnwindSafe(job)) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(error)) => Err(error.to_string()),
        Err(payload) => Err(format!("panicked: {}", panic_message(payload.as_ref()))),
    }
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_string()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// `ratatui::init` installs a panic hook that restores the terminal on a panic
/// in *any* thread. A worker panic would then leave raw mode and the alternate
/// screen while the UI keeps drawing into the shell. Call this right after
/// `ratatui::init` so only the UI thread's panic restores the terminal; workers
/// report theirs as an error string through `run_caught`.
pub fn restore_terminal_only_on_ui_thread_panic() {
    let ui_thread = thread::current().id();
    let ratatui_hook = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        if thread::current().id() == ui_thread {
            ratatui_hook(info);
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn a_panicking_job_reports_an_error_and_clears_in_flight() {
        let mut job = Background::<u32>::new();
        assert!(job.start(|| panic!("boom")));
        let result = job.wait(Duration::from_secs(5)).expect("result arrives");
        assert!(result.unwrap_err().contains("boom"));
        assert!(!job.in_flight());
    }

    #[test]
    fn a_second_start_is_refused_while_the_first_runs() {
        let (release, gate) = mpsc::channel::<()>();
        let mut job = Background::<u32>::new();
        assert!(job.start(move || {
            gate.recv().ok();
            Ok(1)
        }));
        assert!(!job.start(|| Ok(2)), "overlapping job must not start");
        release.send(()).unwrap();
        assert_eq!(job.wait(Duration::from_secs(5)).unwrap().unwrap(), 1);
        assert!(job.start(|| Ok(3)), "next job starts once the first landed");
        assert_eq!(job.wait(Duration::from_secs(5)).unwrap().unwrap(), 3);
    }

    #[test]
    fn replace_supersedes_the_running_job() {
        let (release, gate) = mpsc::channel::<()>();
        let mut job = Background::<u32>::new();
        assert!(job.start(move || {
            gate.recv().ok();
            Ok(1)
        }));
        job.replace(|| Ok(2));
        assert_eq!(job.wait(Duration::from_secs(5)).unwrap().unwrap(), 2);
        release.send(()).unwrap();
        // The superseded job's late result must never surface.
        std::thread::sleep(Duration::from_millis(50));
        assert!(job.poll().is_none());
        assert!(!job.in_flight());
    }

    #[test]
    fn cancel_drops_the_running_jobs_result() {
        let (release, gate) = mpsc::channel::<()>();
        let mut job = Background::<u32>::new();
        assert!(job.start(move || {
            gate.recv().ok();
            Ok(1)
        }));
        job.cancel();
        assert!(!job.in_flight());
        release.send(()).unwrap();
        std::thread::sleep(Duration::from_millis(50));
        assert!(job.poll().is_none());
    }

    #[test]
    fn errors_keep_their_message() {
        let result = run_caught(|| -> anyhow::Result<()> { anyhow::bail!("store locked") });
        assert_eq!(result.unwrap_err(), "store locked");
    }
}
