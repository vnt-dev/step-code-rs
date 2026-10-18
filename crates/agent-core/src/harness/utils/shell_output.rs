//! Shell output capture with truncation and full-output spooling.
//!
//! Ported from `@step-harness/agent-core` `src/harness/utils/shell-output.ts`.
//! Stdout and stderr share one callback and interleave in arrival order; once
//! the output exceeds the truncation limits, the full output is spooled to a
//! temporary file while the in-memory tail is trimmed to
//! `DEFAULT_MAX_BYTES * 2` bytes.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::mpsc;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use crate::harness::types::{
    AbortSignal, CreateTempFileOptions, ExecutionEnv, ExecutionError, ExecutionErrorCode,
    FileContent, ShellExecOptions,
};
use crate::harness::utils::truncate::{
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncatedBy, TruncationResult, truncate_tail,
};

/// Progress snapshot of a shell capture.
#[derive(Clone, Debug)]
pub struct ShellCaptureProgress {
    pub output: String,
    pub truncation: TruncationResult,
    pub full_output_path: Option<String>,
    pub last_line_bytes: usize,
}

/// Callback invoked for each captured chunk with a progress accessor.
pub type OnChunkCallback = Arc<dyn Fn(&str, &dyn Fn() -> ShellCaptureProgress) + Send + Sync>;

/// Options for [`execute_shell_with_capture`].
#[derive(Clone, Default)]
pub struct ShellCaptureOptions {
    pub cwd: Option<String>,
    pub env: Option<HashMap<String, String>>,
    pub inherit_env: Option<bool>,
    /// Timeout in seconds.
    pub timeout: Option<f64>,
    pub signal: Option<AbortSignal>,
    pub on_chunk: Option<OnChunkCallback>,
    /// Return shell execution failures with captured output instead of an error.
    pub return_execution_errors: bool,
}

/// Result of a shell capture.
#[derive(Debug)]
pub struct ShellCaptureResult {
    pub output: String,
    pub truncation: TruncationResult,
    pub full_output_path: Option<String>,
    pub last_line_bytes: usize,
    pub exit_code: Option<i32>,
    pub cancelled: bool,
    pub truncated: bool,
    pub execution_error: Option<ExecutionError>,
}

/// Remove control characters and interlinear annotation characters that would
/// corrupt terminal output.
pub fn sanitize_binary_output(text: &str) -> String {
    text.chars()
        .filter(|character| {
            let code = *character as u32;
            if code == 0x09 || code == 0x0a || code == 0x0d {
                return true;
            }
            if code <= 0x1f {
                return false;
            }
            if (0xfff9..=0xfffb).contains(&code) {
                return false;
            }
            true
        })
        .collect()
}

/// Keep only the last `max_bytes` bytes of `text`, cutting on a UTF-8
/// character boundary.
fn trim_to_last_utf8_bytes(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_owned();
    }
    let mut start = text.len() - max_bytes;
    let bytes = text.as_bytes();
    while start < bytes.len() && (bytes[start] & 0xc0) == 0x80 {
        start += 1;
    }
    text[start..].to_owned()
}

fn to_execution_error(error: crate::harness::types::FileError) -> ExecutionError {
    let message = error.message.clone();
    ExecutionError::with_cause(ExecutionErrorCode::Unknown, message, error)
}

/// Messages driving the serialized full-output writer.
enum FullOutputMessage {
    Create(String),
    Append(String),
}

/// Serialized writer that spools the full output to a temporary file.
///
/// The writer task owns the append ordering, so concurrent chunk callbacks can
/// enqueue writes without blocking; [`FullOutputWriter::drain`] waits for the
/// queue to settle.
struct FullOutputWriter {
    sender: std::sync::Mutex<Option<mpsc::Sender<FullOutputMessage>>>,
    done: watch::Receiver<bool>,
    path: Arc<std::sync::Mutex<Option<String>>>,
    error: Arc<std::sync::Mutex<Option<ExecutionError>>>,
}

impl FullOutputWriter {
    fn spawn(env: Arc<dyn ExecutionEnv>) -> Arc<Self> {
        let (sender, mut receiver) = mpsc::channel(128);
        let (done_tx, done_rx) = watch::channel(false);
        let path = Arc::new(std::sync::Mutex::new(None));
        let error = Arc::new(std::sync::Mutex::new(None));
        let path_for_task = path.clone();
        let error_for_task = error.clone();

        tokio::spawn(async move {
            let mut current_path: Option<String> = None;
            while let Some(message) = receiver.recv().await {
                match message {
                    FullOutputMessage::Create(initial_content) => {
                        let temp_file = env
                            .create_temp_file(Some(&CreateTempFileOptions {
                                prefix: Some("bash-".to_owned()),
                                suffix: Some(".log".to_owned()),
                                signal: None,
                            }))
                            .await;
                        match temp_file {
                            Ok(created) => {
                                current_path = Some(created.clone());
                                *path_for_task.lock().expect("path mutex poisoned") =
                                    Some(created.clone());
                                if let Err(file_error) = env
                                    .append_file(&created, FileContent::Text(initial_content), None)
                                    .await
                                {
                                    *error_for_task.lock().expect("error mutex poisoned") =
                                        Some(to_execution_error(file_error));
                                    break;
                                }
                            }
                            Err(file_error) => {
                                *error_for_task.lock().expect("error mutex poisoned") =
                                    Some(to_execution_error(file_error));
                                break;
                            }
                        }
                    }
                    FullOutputMessage::Append(text) => {
                        let Some(current_path) = &current_path else {
                            *error_for_task.lock().expect("error mutex poisoned") =
                                Some(ExecutionError::new(
                                    ExecutionErrorCode::Unknown,
                                    "Full output path was not created",
                                ));
                            break;
                        };
                        if let Err(file_error) = env
                            .append_file(current_path, FileContent::Text(text), None)
                            .await
                        {
                            *error_for_task.lock().expect("error mutex poisoned") =
                                Some(to_execution_error(file_error));
                            break;
                        }
                    }
                }
            }
            let _ = done_tx.send(true);
        });

        Arc::new(FullOutputWriter {
            sender: std::sync::Mutex::new(Some(sender)),
            done: done_rx,
            path,
            error,
        })
    }

    fn create(&self, initial_content: String) {
        if let Some(sender) = self.sender.lock().expect("sender mutex poisoned").as_ref() {
            self.send(sender, FullOutputMessage::Create(initial_content));
        }
    }

    fn append(&self, text: String) {
        if let Some(sender) = self.sender.lock().expect("sender mutex poisoned").as_ref() {
            self.send(sender, FullOutputMessage::Append(text));
        }
    }

    fn send(&self, sender: &mpsc::Sender<FullOutputMessage>, message: FullOutputMessage) {
        if sender.try_send(message).is_err() {
            let mut error = self.error.lock().expect("writer error poisoned");
            if error.is_none() {
                *error = Some(ExecutionError::new(
                    ExecutionErrorCode::Unknown,
                    "Full-output writer cannot keep up with process output",
                ));
            }
        }
    }

    fn failed(&self) -> bool {
        self.error.lock().expect("writer error poisoned").is_some()
    }

    fn path(&self) -> Option<String> {
        self.path.lock().expect("path mutex poisoned").clone()
    }

    /// Take the writer's failure, if any. The writer stops on the first error.
    fn take_error(&self) -> Option<ExecutionError> {
        self.error.lock().expect("error mutex poisoned").take()
    }

    /// Wait for all queued writes to settle.
    async fn drain(&self) {
        // Dropping the last sender closes the channel; the task then exits and
        // signals completion.
        self.sender.lock().expect("sender mutex poisoned").take();
        let mut done = self.done.clone();
        while !*done.borrow_and_update() {
            if done.changed().await.is_err() {
                break;
            }
        }
    }
}

/// Mutable capture state shared between the chunk callback and the progress
/// accessor.
struct CaptureState {
    tail_output: String,
    total_bytes: usize,
    completed_lines: usize,
    has_open_line: bool,
    current_line_bytes: usize,
    full_output_path: Option<String>,
}

/// Execute a shell command through `env`, capturing merged stdout/stderr.
pub async fn execute_shell_with_capture(
    env: &Arc<dyn ExecutionEnv>,
    command: &str,
    options: Option<&ShellCaptureOptions>,
) -> Result<ShellCaptureResult, ExecutionError> {
    const MAX_OUTPUT_BYTES: usize = DEFAULT_MAX_BYTES * 2;

    let capture_signal = options
        .and_then(|options| options.signal.as_ref())
        .map(CancellationToken::child_token)
        .unwrap_or_default();
    let progress_state = Arc::new(std::sync::Mutex::new(CaptureState {
        tail_output: String::new(),
        total_bytes: 0,
        completed_lines: 0,
        has_open_line: false,
        current_line_bytes: 0,
        full_output_path: None,
    }));
    let writer: Arc<std::sync::Mutex<Option<Arc<FullOutputWriter>>>> =
        Arc::new(std::sync::Mutex::new(None));
    let accepting_output = Arc::new(std::sync::atomic::AtomicBool::new(true));

    let create_progress: Arc<dyn Fn() -> ShellCaptureProgress + Send + Sync> = {
        let progress_state = progress_state.clone();
        Arc::new(move || {
            let state = progress_state.lock().expect("progress mutex poisoned");
            let tail_truncation = truncate_tail(&state.tail_output, &Default::default());
            let total_lines = state.completed_lines + usize::from(state.has_open_line);
            let truncated =
                total_lines > DEFAULT_MAX_LINES || state.total_bytes > DEFAULT_MAX_BYTES;
            let truncation = TruncationResult {
                truncated,
                truncated_by: if truncated {
                    Some(tail_truncation.truncated_by.unwrap_or(
                        if state.total_bytes > DEFAULT_MAX_BYTES {
                            TruncatedBy::Bytes
                        } else {
                            TruncatedBy::Lines
                        },
                    ))
                } else {
                    None
                },
                total_lines,
                total_bytes: state.total_bytes,
                ..tail_truncation
            };
            ShellCaptureProgress {
                output: if truncated {
                    truncation.content.clone()
                } else {
                    state.tail_output.clone()
                },
                truncation,
                full_output_path: state.full_output_path.clone(),
                last_line_bytes: state.current_line_bytes,
            }
        })
    };

    let env_owned = env.clone();
    let on_chunk_impl = {
        let writer = writer.clone();
        let accepting_output = accepting_output.clone();
        let progress_state = progress_state.clone();
        let user_on_chunk = options.and_then(|options| options.on_chunk.clone());
        let create_progress = create_progress.clone();
        let env = env_owned;
        let capture_signal = capture_signal.clone();
        move |chunk: &str| {
            if !accepting_output.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }
            let text = sanitize_binary_output(chunk).replace('\r', "");

            {
                let mut state = progress_state.lock().expect("progress mutex poisoned");
                state.total_bytes += text.len();
                state.completed_lines += text.matches('\n').count();
                if let Some(last_newline) = text.rfind('\n') {
                    let trailing_text = &text[last_newline + 1..];
                    state.current_line_bytes = trailing_text.len();
                    state.has_open_line = !trailing_text.is_empty();
                } else if !text.is_empty() {
                    state.current_line_bytes += text.len();
                    state.has_open_line = true;
                }
                state.tail_output.push_str(&text);

                let total_lines = state.completed_lines + usize::from(state.has_open_line);
                let exceeds =
                    state.total_bytes > DEFAULT_MAX_BYTES || total_lines > DEFAULT_MAX_LINES;
                let existing_writer = writer.lock().expect("writer mutex poisoned").clone();
                if let Some(existing_writer) = existing_writer {
                    existing_writer.append(text.clone());
                    if existing_writer.failed() {
                        capture_signal.cancel();
                    }
                } else if exceeds {
                    let new_writer = FullOutputWriter::spawn(env.clone());
                    new_writer.create(state.tail_output.clone());
                    state.full_output_path = new_writer.path();
                    *writer.lock().expect("writer mutex poisoned") = Some(new_writer);
                }
                state.tail_output = trim_to_last_utf8_bytes(&state.tail_output, MAX_OUTPUT_BYTES);
            }

            if let Some(user_on_chunk) = &user_on_chunk {
                user_on_chunk(&text, create_progress.as_ref());
            }
        }
    };

    let on_chunk = Arc::new(on_chunk_impl);
    let exec_options = ShellExecOptions {
        cwd: options.and_then(|options| options.cwd.clone()),
        env: options.and_then(|options| options.env.clone()),
        inherit_env: options.and_then(|options| options.inherit_env),
        timeout: options.and_then(|options| options.timeout),
        signal: Some(capture_signal),
        on_stdout: Some(on_chunk.clone()),
        on_stderr: Some(on_chunk),
    };

    let result = env.exec(command, Some(&exec_options)).await;
    accepting_output.store(false, std::sync::atomic::Ordering::SeqCst);

    // If truncation happened, make sure the full output file exists even when
    // the tail was never spooled during streaming.
    let spool_after_exec = {
        let state = progress_state.lock().expect("progress mutex poisoned");
        let total_lines = state.completed_lines + usize::from(state.has_open_line);
        total_lines > DEFAULT_MAX_LINES || state.total_bytes > DEFAULT_MAX_BYTES
    };
    if spool_after_exec && writer.lock().expect("writer mutex poisoned").is_none() {
        let initial_content = progress_state
            .lock()
            .expect("progress mutex poisoned")
            .tail_output
            .clone();
        let new_writer = FullOutputWriter::spawn(env.clone());
        new_writer.create(initial_content);
        *writer.lock().expect("writer mutex poisoned") = Some(new_writer);
    }

    // Wait for all queued writes, then surface any write failure.
    let active_writer = writer.lock().expect("writer mutex poisoned").clone();
    if let Some(active_writer) = &active_writer {
        active_writer.drain().await;
        if let Some(error) = active_writer.take_error() {
            return Err(error);
        }
        // The temp-file path only becomes known once the writer has settled.
        let path = active_writer.path();
        progress_state
            .lock()
            .expect("progress mutex poisoned")
            .full_output_path = path;
    }

    let progress = create_progress();

    match result {
        Err(execution_error) => {
            let aborted = execution_error.code == ExecutionErrorCode::Aborted
                || options
                    .and_then(|options| options.signal.as_ref())
                    .is_some_and(|signal| signal.is_cancelled());
            if aborted {
                Ok(ShellCaptureResult {
                    exit_code: None,
                    cancelled: true,
                    execution_error: None,
                    ..capture_result_from(progress)
                })
            } else if options.is_some_and(|options| options.return_execution_errors) {
                Ok(ShellCaptureResult {
                    exit_code: None,
                    cancelled: false,
                    execution_error: Some(execution_error),
                    ..capture_result_from(progress)
                })
            } else {
                Err(execution_error)
            }
        }
        Ok(exec_result) => {
            let cancelled = options
                .and_then(|options| options.signal.as_ref())
                .is_some_and(|signal| signal.is_cancelled());
            Ok(ShellCaptureResult {
                exit_code: if cancelled {
                    None
                } else {
                    Some(exec_result.exit_code)
                },
                cancelled,
                execution_error: None,
                ..capture_result_from(progress)
            })
        }
    }
}

fn capture_result_from(progress: ShellCaptureProgress) -> ShellCaptureResult {
    ShellCaptureResult {
        truncated: progress.truncation.truncated,
        output: progress.output,
        truncation: progress.truncation,
        full_output_path: progress.full_output_path,
        last_line_bytes: progress.last_line_bytes,
        exit_code: None,
        cancelled: false,
        execution_error: None,
    }
}

/// Convenience re-export of the abort signal type.
pub type CaptureSignal = CancellationToken;
