//! The `bash` tool: shell command execution with output capture.
//!
//! Ported from `@step-harness/agent-core` `src\harness\tools\bash.ts`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::Value as Json;

use crate::harness::tools::tool_context::{ExecutionToolContext, ToolEnvContext};
use crate::harness::types::ContextToolExecutor;
use crate::harness::types::{AbortSignal, ExecutionEnv, ExecutionError, ExecutionErrorCode};
use crate::harness::utils::shell_output::{
    ShellCaptureOptions, ShellCaptureProgress, execute_shell_with_capture,
};
use crate::harness::utils::truncate::{
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncatedBy, TruncationResult, format_size,
};
use crate::types::{
    AgentTool, AgentToolResult, ToolError, ToolErrorMessage, ToolUpdateCallback, text_content,
};

use crate::harness::timeout::{MAX_TIMEOUT_SECONDS, resolve_timeout};
/// Throttle window for streamed progress updates.
pub const BASH_UPDATE_THROTTLE: Duration = Duration::from_millis(100);

/// JSON Schema for the bash tool parameters.
pub fn bash_parameters_schema() -> Json {
    serde_json::json!({
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Bash command to execute"
            },
            "timeout": {
                "type": "number",
                "exclusiveMinimum": 0,
                "maximum": MAX_TIMEOUT_SECONDS,
                "description": "Timeout in seconds; fractional seconds are supported (optional, no default timeout)"
            }
        },
        "required": ["command"]
    })
}

/// Validated bash tool arguments.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BashToolInput {
    pub command: String,
    #[serde(default)]
    pub timeout: Option<f64>,
}

impl BashToolInput {
    /// Parse validated tool-call arguments.
    pub fn from_args(args: &serde_json::Map<String, Json>) -> Result<Self, ToolError> {
        serde_json::from_value(Json::Object(args.clone())).map_err(ToolError::from)
    }
}

/// Details attached to a bash tool result.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BashToolDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncation: Option<TruncationResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full_output_path: Option<String>,
}

/// The command execution plan handed to the prepare hook.
#[derive(Clone, Debug)]
pub struct BashExecution {
    pub command: String,
    pub cwd: String,
    pub env: HashMap<String, String>,
    pub inherit_env: bool,
}

/// Hook that can adjust the execution plan before the command runs.
#[async_trait::async_trait]
pub trait BashPrepare<TContext>: Send + Sync {
    async fn prepare(
        &self,
        execution: &mut BashExecution,
        context: &TContext,
        signal: Option<&AbortSignal>,
    );
}

/// Options for [`create_bash_tool`].
pub struct BashToolOptions<TContext> {
    /// Prepended to the command as a separate line of the script.
    pub command_prefix: Option<String>,
    pub prepare: Option<Arc<dyn BashPrepare<TContext>>>,
}

impl<TContext> Default for BashToolOptions<TContext> {
    fn default() -> Self {
        BashToolOptions {
            command_prefix: None,
            prepare: None,
        }
    }
}

fn validate_timeout(timeout: Option<f64>) -> Result<(), ToolError> {
    resolve_timeout(timeout)
        .map(|_| ())
        .map_err(ToolError::from)
}

/// State for throttled progress updates.
struct BashUpdateState {
    progress: Option<ShellCaptureProgress>,
    dirty: bool,
    timer_pending: bool,
    last_update: Instant,
}

struct BashTimerGuard(Arc<std::sync::Mutex<Option<tokio::task::JoinHandle<()>>>>);
impl Drop for BashTimerGuard {
    fn drop(&mut self) {
        if let Some(task) = self
            .0
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
        {
            task.abort();
        }
    }
}

fn emit_output_update(state: &std::sync::Mutex<BashUpdateState>, on_update: &ToolUpdateCallback) {
    let mut state = state.lock().expect("bash update state poisoned");
    if !state.dirty {
        return;
    }
    state.dirty = false;
    state.last_update = Instant::now();
    let Some(progress) = state.progress.clone() else {
        return;
    };
    drop(state);
    let output = progress.output.clone();
    on_update(AgentToolResult {
        content: vec![text_content(output)],
        details: bash_details_json(&progress),
        usage: None,
        added_tool_names: None,
        terminate: None,
    });
}

fn bash_details_json(progress: &ShellCaptureProgress) -> Json {
    if !progress.truncation.truncated {
        return Json::Null;
    }
    serde_json::to_value(BashToolDetails {
        truncation: Some(progress.truncation.clone()),
        full_output_path: progress.full_output_path.clone(),
    })
    .unwrap_or(Json::Null)
}

struct BashToolExecutor<TContext: ToolEnvContext> {
    options: Option<BashToolOptions<TContext>>,
}

#[async_trait::async_trait]
impl<TContext: ToolEnvContext + Send + Sync + 'static> ContextToolExecutor<TContext, Json>
    for BashToolExecutor<TContext>
{
    async fn execute(
        &self,
        _tool_call_id: &str,
        params: &serde_json::Map<String, Json>,
        signal: Option<&AbortSignal>,
        on_update: Option<&ToolUpdateCallback<Json>>,
        context: &TContext,
    ) -> Result<AgentToolResult<Json>, ToolError> {
        let input = BashToolInput::from_args(params)?;
        validate_timeout(input.timeout)?;

        let env = context.env();
        let execution = BashExecution {
            command: match &self
                .options
                .as_ref()
                .and_then(|options| options.command_prefix.clone())
            {
                Some(prefix) => format!("{prefix}\n{}", input.command),
                None => input.command.clone(),
            },
            cwd: env.cwd().to_owned(),
            env: HashMap::new(),
            inherit_env: true,
        };
        let mut execution = execution;
        if let Some(prepare) = self
            .options
            .as_ref()
            .and_then(|options| options.prepare.as_ref())
        {
            prepare.prepare(&mut execution, context, signal).await;
        }

        let update_state = Arc::new(std::sync::Mutex::new(BashUpdateState {
            progress: None,
            dirty: false,
            timer_pending: false,
            last_update: Instant::now()
                .checked_sub(Duration::from_secs(3600))
                .unwrap_or_else(Instant::now),
        }));

        // Emit an initial empty update so consumers can render the tool slot.
        if let Some(on_update) = on_update {
            on_update(AgentToolResult {
                content: Vec::new(),
                details: Json::Null,
                usage: None,
                added_tool_names: None,
                terminate: None,
            });
        }

        let timer = Arc::new(std::sync::Mutex::new(None));
        let _timer_guard = BashTimerGuard(timer.clone());
        let on_chunk_timer = timer.clone();
        let on_chunk_state = update_state.clone();
        let on_chunk_callback = on_update.cloned();
        let on_chunk = move |_chunk: &str, get_progress: &dyn Fn() -> ShellCaptureProgress| {
            let progress = get_progress();
            let (should_emit, delay) = {
                let mut state = on_chunk_state.lock().expect("bash update state poisoned");
                state.progress = Some(progress);
                state.dirty = true;
                let elapsed = state.last_update.elapsed();
                if elapsed >= BASH_UPDATE_THROTTLE {
                    (true, Duration::ZERO)
                } else if state.timer_pending {
                    (false, Duration::ZERO)
                } else {
                    (false, BASH_UPDATE_THROTTLE - elapsed)
                }
            };
            if should_emit {
                if let Some(on_update) = &on_chunk_callback {
                    emit_output_update(&on_chunk_state, on_update);
                }
            } else if delay > Duration::ZERO
                && let Some(on_update) = on_chunk_callback.clone()
            {
                let timer_state = on_chunk_state.clone();
                on_chunk_state
                    .lock()
                    .expect("bash update state poisoned")
                    .timer_pending = true;
                let task = tokio::spawn(async move {
                    tokio::time::sleep(delay).await;
                    {
                        let mut state = timer_state.lock().expect("bash update state poisoned");
                        state.timer_pending = false;
                    }
                    emit_output_update(&timer_state, &on_update);
                });
                if let Some(previous) = on_chunk_timer
                    .lock()
                    .expect("bash timer poisoned")
                    .replace(task)
                {
                    previous.abort();
                }
            }
        };

        let capture = execute_shell_with_capture(
            env,
            &execution.command,
            Some(&ShellCaptureOptions {
                cwd: Some(execution.cwd.clone()),
                env: Some(execution.env.clone()),
                inherit_env: Some(execution.inherit_env),
                timeout: input.timeout,
                signal: signal.cloned(),
                on_chunk: Some(Arc::new(on_chunk)),
                return_execution_errors: true,
            }),
        )
        .await?;

        // Final update carrying the complete capture.
        {
            let mut state = update_state.lock().expect("bash update state poisoned");
            state.progress = Some(ShellCaptureProgress {
                output: capture.output.clone(),
                truncation: capture.truncation.clone(),
                full_output_path: capture.full_output_path.clone(),
                last_line_bytes: capture.last_line_bytes,
            });
            state.dirty = true;
        }
        if let Some(on_update) = on_update {
            emit_output_update(&update_state, on_update);
        }

        let mut output_text = capture.output.clone();
        let details: Json = if capture.truncation.truncated {
            let truncation = &capture.truncation;
            let start_line = truncation.total_lines - truncation.output_lines + 1;
            let end_line = truncation.total_lines;
            if truncation.last_line_partial {
                let last_line_size = format_size(capture.last_line_bytes);
                output_text.push_str(&format!(
                    "\n\n[Showing last {} of line {} (line is {}). Full output: {}]",
                    format_size(truncation.output_bytes),
                    end_line,
                    last_line_size,
                    capture.full_output_path.clone().unwrap_or_default()
                ));
            } else if truncation.truncated_by == Some(TruncatedBy::Lines) {
                output_text.push_str(&format!(
                    "\n\n[Showing lines {start_line}-{end_line} of {}. Full output: {}]",
                    truncation.total_lines,
                    capture.full_output_path.clone().unwrap_or_default()
                ));
            } else {
                output_text.push_str(&format!(
                    "\n\n[Showing lines {start_line}-{end_line} of {} ({} limit). Full output: {}]",
                    truncation.total_lines,
                    format_size(DEFAULT_MAX_BYTES),
                    capture.full_output_path.clone().unwrap_or_default()
                ));
            }
            serde_json::to_value(BashToolDetails {
                truncation: Some(truncation.clone()),
                full_output_path: capture.full_output_path.clone(),
            })
            .unwrap_or(Json::Null)
        } else {
            Json::Null
        };

        let append_status = |status: &str| -> String {
            if output_text.is_empty() {
                status.to_owned()
            } else {
                format!("{output_text}\n\n{status}")
            }
        };
        if capture.cancelled {
            return Err(Box::new(ToolErrorMessage(append_status("Command aborted"))));
        }
        if let Some(execution_error) = &capture.execution_error {
            if execution_error.code == ExecutionErrorCode::Timeout {
                return Err(Box::new(BashTimeoutError {
                    message: append_status(&format!(
                        "Command timed out after {} seconds",
                        input.timeout.unwrap_or_default()
                    )),
                    cause: execution_error.clone(),
                }));
            }
            return Err(Box::new(execution_error.clone()));
        }
        if let Some(exit_code) = capture.exit_code
            && exit_code != 0
        {
            return Err(Box::new(ToolErrorMessage(append_status(&format!(
                "Command exited with code {exit_code}"
            )))));
        }

        Ok(AgentToolResult {
            content: vec![text_content(if output_text.is_empty() {
                "(no output)".to_owned()
            } else {
                output_text
            })],
            details,
            usage: None,
            added_tool_names: None,
            terminate: None,
        })
    }
}

/// A bash timeout error carrying the underlying execution error as its cause.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct BashTimeoutError {
    pub message: String,
    #[source]
    pub cause: ExecutionError,
}

/// Create the `bash` tool.
pub fn create_bash_tool<TContext: ToolEnvContext + Send + Sync + 'static>(
    options: Option<BashToolOptions<TContext>>,
) -> crate::harness::types::AgentHarnessTool<TContext, Json> {
    crate::harness::types::AgentHarnessTool {
        name: "bash".to_owned(),
        label: "bash".to_owned(),
        description: format!(
            "Execute a bash command in the current working directory. Returns stdout and stderr. Output is truncated to last {DEFAULT_MAX_LINES} lines or {}KB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds.",
            DEFAULT_MAX_BYTES / 1024
        ),
        parameters: bash_parameters_schema(),
        execution_mode: None,
        prepare_arguments: None,
        execute: Arc::new(BashToolExecutor { options }),
    }
}

/// Convenience constructor producing an [`crate::types::AgentTool`] bound to one
/// execution environment.
pub fn create_bash_tool_with_env(env: Arc<dyn ExecutionEnv>) -> AgentTool {
    create_bash_tool::<ExecutionToolContext>(None).into_agent_tool(ExecutionToolContext::new(env))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_input_with_and_without_timeout() {
        let input: BashToolInput =
            serde_json::from_value(serde_json::json!({"command": "ls"})).unwrap();
        assert_eq!(input.command, "ls");
        assert_eq!(input.timeout, None);

        let input: BashToolInput =
            serde_json::from_value(serde_json::json!({"command": "ls", "timeout": 0.5})).unwrap();
        assert_eq!(input.timeout, Some(0.5));
    }

    #[test]
    fn validates_timeouts() {
        assert!(validate_timeout(None).is_ok());
        assert!(validate_timeout(Some(1.0)).is_ok());
        assert!(validate_timeout(Some(0.0005)).is_ok());
        assert!(validate_timeout(Some(MAX_TIMEOUT_SECONDS)).is_ok());
        assert!(validate_timeout(Some(MAX_TIMEOUT_SECONDS + 0.0001)).is_err());
        assert!(validate_timeout(Some(1e-20)).is_err());
        assert!(validate_timeout(Some(0.0)).is_err());
        assert!(validate_timeout(Some(-1.0)).is_err());
        assert!(validate_timeout(Some(f64::NAN)).is_err());
        assert!(validate_timeout(Some(MAX_TIMEOUT_SECONDS + 1.0)).is_err());
    }

    #[test]
    fn details_are_null_without_truncation() {
        let progress = ShellCaptureProgress {
            output: "x".to_owned(),
            truncation: TruncationResult {
                content: "x".to_owned(),
                truncated: false,
                truncated_by: None,
                total_lines: 1,
                total_bytes: 1,
                output_lines: 1,
                output_bytes: 1,
                last_line_partial: false,
                first_line_exceeds_limit: false,
                max_lines: DEFAULT_MAX_LINES,
                max_bytes: DEFAULT_MAX_BYTES,
            },
            full_output_path: None,
            last_line_bytes: 1,
        };
        assert_eq!(bash_details_json(&progress), Json::Null);
    }
}
