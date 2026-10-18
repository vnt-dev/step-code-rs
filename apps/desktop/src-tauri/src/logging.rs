//! Rolling file logging for the desktop app.
//!
//! `tracing` events and the `log` records emitted by the workspace crates are
//! written to `<logs dir>/step-code.log`. The live file rolls over once it
//! reaches [`MAX_LOG_FILE_BYTES`], and the [`KEEP_LOG_FILES`] most recent
//! generations are kept beside it, so a long session cannot fill the disk.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::EnvFilter;

/// Size at which the live log file rolls over to the next generation.
pub const MAX_LOG_FILE_BYTES: u64 = 2 * 1024 * 1024;

/// Generations kept beside the live file; the oldest is dropped on rotation.
pub const KEEP_LOG_FILES: usize = 5;

/// Base name of the log files inside the logs directory.
const LOG_BASENAME: &str = "step-code";

/// Log filter used when `RUST_LOG` is unset or cannot be parsed.
const DEFAULT_FILTER: &str = "info";

fn live_path(directory: &Path, basename: &str) -> PathBuf {
    directory.join(format!("{basename}.log"))
}

/// A log file that rolls over once it reaches `max_bytes`, keeping `keep`
/// generations named `<basename>.log.1` (newest) up to `<basename>.log.<keep>`.
struct RollingLog {
    directory: PathBuf,
    basename: String,
    max_bytes: u64,
    keep: usize,
    file: Option<File>,
    size: u64,
}

impl RollingLog {
    fn open(directory: &Path, basename: &str, max_bytes: u64, keep: usize) -> io::Result<Self> {
        fs::create_dir_all(directory)?;
        let path = live_path(directory, basename);
        // A file left behind by an earlier run keeps counting towards the cap.
        let size = fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        Ok(Self { directory: directory.to_path_buf(), basename: basename.to_owned(), max_bytes, keep, file: Some(file), size })
    }

    fn generation_path(&self, generation: usize) -> PathBuf {
        self.directory.join(format!("{}.log.{generation}", self.basename))
    }

    /// Close the live file, shift the generations down and start a fresh one.
    fn rotate(&mut self) -> io::Result<()> {
        // Windows refuses to rename a file that is still open.
        self.file = None;
        let live = live_path(&self.directory, &self.basename);
        if self.keep == 0 {
            fs::remove_file(&live).ok();
        } else {
            fs::remove_file(self.generation_path(self.keep)).ok();
            for generation in (1..self.keep).rev() {
                let from = self.generation_path(generation);
                if from.exists() {
                    fs::remove_file(self.generation_path(generation + 1)).ok();
                    fs::rename(from, self.generation_path(generation + 1))?;
                }
            }
            if live.exists() {
                fs::rename(&live, self.generation_path(1))?;
            }
        }
        self.file = Some(OpenOptions::new().create(true).append(true).open(&live)?);
        self.size = 0;
        Ok(())
    }
}

impl Write for RollingLog {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        // An event larger than the cap is still written whole: rotating would
        // only move the oversized line into a file of its own.
        if self.size > 0 && self.size + buf.len() as u64 > self.max_bytes {
            self.rotate()?;
        }
        let file = self.file.as_mut().ok_or_else(|| io::Error::other("log file is closed"))?;
        file.write_all(buf)?;
        self.size += buf.len() as u64;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.as_mut().map_or(Ok(()), File::flush)
    }
}

/// Hands every event the same rolling file.
pub struct RollingLogWriter(Mutex<RollingLog>);

impl RollingLogWriter {
    fn open(directory: &Path, basename: &str, max_bytes: u64, keep: usize) -> io::Result<Self> {
        Ok(Self(Mutex::new(RollingLog::open(directory, basename, max_bytes, keep)?)))
    }
}

impl<'a> MakeWriter<'a> for RollingLogWriter {
    type Writer = EventWriter<'a>;

    fn make_writer(&'a self) -> Self::Writer {
        EventWriter { file: self.0.lock().unwrap_or_else(|poisoned| poisoned.into_inner()), buffer: Vec::new() }
    }
}

/// Buffers one event and appends it when the event is complete, so a rotation
/// never splits a line across two files.
pub struct EventWriter<'a> {
    file: MutexGuard<'a, RollingLog>,
    buffer: Vec<u8>,
}

impl io::Write for EventWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.buffer.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Drop for EventWriter<'_> {
    fn drop(&mut self) {
        let _ = self.file.write_all(&self.buffer);
        let _ = self.file.flush();
    }
}

/// Install the rolling file logger into `logs_dir`.
///
/// `tracing` events and the `log` records from the workspace crates share one
/// subscriber, filtered by `RUST_LOG` (defaulting to `info`). Set
/// `RUST_LOG=info,http=debug` for provider request methods, origins and statuses.
/// HTTP headers, URL paths/parameters and request bodies are omitted.
pub fn init(logs_dir: &Path) -> io::Result<()> {
    fs::create_dir_all(logs_dir)?;
    let writer = RollingLogWriter::open(logs_dir, LOG_BASENAME, MAX_LOG_FILE_BYTES, KEEP_LOG_FILES)?;
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(DEFAULT_FILTER));
    let subscriber = tracing_subscriber::fmt::Subscriber::builder()
        // The log is a file, and the previous logger never styled it either.
        .with_ansi(false)
        .with_env_filter(filter)
        .with_writer(writer)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .map_err(|_| io::Error::other("a global tracing subscriber is already installed"))?;
    // Records emitted through the `log` facade become tracing events, so the
    // crates that still use it reach the same file.
    tracing_log::LogTracer::builder()
        .ignore_crate("tracing")
        .init()
        .map_err(|error| io::Error::other(format!("install the log compatibility layer: {error}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write `count` events of `size` bytes each, numbered so a file's contents
    /// reveal which events it holds.
    fn write_events(writer: &RollingLogWriter, count: usize, size: usize) {
        for index in 1..=count {
            let line = format!("{index:0width$}\n", width = size - 1);
            assert_eq!(line.len(), size);
            let mut event = writer.make_writer();
            event.write_all(line.as_bytes()).expect("event written");
        }
    }

    fn generations(directory: &Path) -> Vec<PathBuf> {
        let mut found: Vec<PathBuf> = fs::read_dir(directory)
            .expect("logs directory readable")
            .map(|entry| entry.expect("directory entry readable").path())
            .collect();
        found.sort();
        found
    }

    #[test]
    fn rolls_over_at_the_cap_and_keeps_the_requested_generations() {
        let directory = tempfile::tempdir().expect("temporary directory");
        // Three 8-byte events fill the 24-byte cap, so every fourth rotates.
        let writer = RollingLogWriter::open(directory.path(), "step-code", 24, 2).expect("writer opens");
        write_events(&writer, 7, 8);
        let live = fs::read_to_string(live_path(directory.path(), "step-code")).expect("live file readable");
        let first = fs::read_to_string(directory.path().join("step-code.log.1")).expect("first generation readable");
        let second = fs::read_to_string(directory.path().join("step-code.log.2")).expect("second generation readable");
        assert_eq!(live, "0000007\n");
        assert_eq!(first, "0000004\n0000005\n0000006\n");
        assert_eq!(second, "0000001\n0000002\n0000003\n");
        // The oldest generation was dropped instead of becoming a third file.
        assert_eq!(generations(directory.path()).len(), 3);
        assert!(!directory.path().join("step-code.log.3").exists());
    }

    #[test]
    fn a_file_left_by_an_earlier_run_keeps_counting_towards_the_cap() {
        let directory = tempfile::tempdir().expect("temporary directory");
        fs::write(directory.path().join("step-code.log"), "0123456789abcdefghij").expect("leftover file written");
        let writer = RollingLogWriter::open(directory.path(), "step-code", 24, 2).expect("writer opens");
        write_events(&writer, 1, 8);
        // The 20 leftover bytes plus 8 more exceed 24, so the run rotated first.
        assert_eq!(fs::read_to_string(live_path(directory.path(), "step-code")).expect("live file readable"), "0000001\n");
        assert_eq!(fs::read_to_string(directory.path().join("step-code.log.1")).expect("first generation readable"), "0123456789abcdefghij");
    }

    #[test]
    fn an_event_larger_than_the_cap_is_written_whole() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let writer = RollingLogWriter::open(directory.path(), "step-code", 8, 5).expect("writer opens");
        let mut event = writer.make_writer();
        event.write_all(b"0123456789abcdef").expect("oversized event written");
        drop(event);
        assert_eq!(fs::read_to_string(live_path(directory.path(), "step-code")).expect("live file readable"), "0123456789abcdef");
    }

    #[test]
    fn keeping_no_generations_discards_the_previous_file() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let writer = RollingLogWriter::open(directory.path(), "step-code", 8, 0).expect("writer opens");
        write_events(&writer, 3, 8);
        assert_eq!(fs::read_to_string(live_path(directory.path(), "step-code")).expect("live file readable"), "0000003\n");
        assert_eq!(generations(directory.path()).len(), 1);
    }

    #[test]
    fn the_installed_logger_receives_log_records_and_tracing_events() {
        let directory = tempfile::tempdir().expect("temporary directory");
        init(directory.path()).expect("logger installed");
        // The workspace crates emit through the `log` facade.
        log::info!(target: "http", "POST /v1/chat/completions replayable");
        tracing::info!("tracing event reached the file");
        let content = fs::read_to_string(live_path(directory.path(), "step-code")).expect("live file readable");
        assert!(content.contains("replayable"), "{content}");
        assert!(content.contains("http"), "{content}");
        assert!(content.contains("tracing event reached the file"), "{content}");
    }
}
