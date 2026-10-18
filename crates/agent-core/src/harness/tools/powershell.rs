//! The `powershell` tool: PowerShell command execution on Windows.
//!
//! Ported functionally from `@step-harness/coding-agent`
//! `src/core/tools/powershell.ts`. The TypeScript source configures a
//! PowerShell shell profile; this port passes the script to
//! `powershell.exe -EncodedCommand` (base64-encoded UTF-16LE), which avoids
//! every nested-quoting problem when the command itself is launched through
//! the environment's shell.

use std::sync::Arc;

use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;

use crate::harness::tools::tool_context::ToolEnvContext;
use crate::harness::types::{AbortSignal, AgentHarnessTool, ContextToolExecutor, ExecutionEnv};
use crate::harness::utils::shell_output::{ShellCaptureOptions, execute_shell_with_capture};
use crate::harness::utils::truncate::format_size;
use crate::types::{AgentToolResult, ToolError, ToolErrorMessage, ToolUpdateCallback};

use crate::harness::timeout::{MAX_TIMEOUT_SECONDS, resolve_timeout};

/// Prepended to every script so PowerShell emits UTF-8.
const UTF8_OUTPUT_PREFIX: &str =
    "try { [Console]::OutputEncoding=[System.Text.Encoding]::UTF8 } catch {}\n";

/// JSON Schema for the powershell tool parameters.
pub fn powershell_parameters_schema() -> Json {
    serde_json::json!({
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "PowerShell command to execute"
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

/// Validated powershell tool arguments.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PowerShellToolInput {
    pub command: String,
    #[serde(default)]
    pub timeout: Option<f64>,
}

/// Details attached to a powershell tool result.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PowerShellToolDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncated: Option<bool>,
}

/// Encode a script as base64 UTF-16LE for `-EncodedCommand`.
fn encode_command(script: &str) -> String {
    let full = format!("{UTF8_OUTPUT_PREFIX}{script}");
    let utf16: Vec<u16> = full.encode_utf16().collect();
    let mut bytes = Vec::with_capacity(utf16.len() * 2);
    for unit in utf16 {
        bytes.extend_from_slice(&unit.to_le_bytes());
    }
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// The PowerShell executable for this platform.
fn powershell_executable() -> &'static str {
    if cfg!(windows) {
        "powershell.exe"
    } else {
        "pwsh"
    }
}

struct PowerShellToolExecutor;

#[async_trait::async_trait]
impl<TContext: ToolEnvContext + Send + Sync + 'static> ContextToolExecutor<TContext, Json>
    for PowerShellToolExecutor
{
    async fn execute(
        &self,
        _tool_call_id: &str,
        params: &serde_json::Map<String, Json>,
        signal: Option<&AbortSignal>,
        _on_update: Option<&ToolUpdateCallback<Json>>,
        context: &TContext,
    ) -> Result<AgentToolResult<Json>, ToolError> {
        let input: PowerShellToolInput =
            serde_json::from_value(Json::Object(params.clone())).map_err(ToolError::from)?;
        resolve_timeout(input.timeout).map_err(ToolError::from)?;
        let env = context.env();

        let command = format!(
            "{} -NoProfile -EncodedCommand {}",
            powershell_executable(),
            encode_command(&input.command)
        );

        let capture = execute_shell_with_capture(
            env,
            &command,
            Some(&ShellCaptureOptions {
                cwd: Some(env.cwd().to_owned()),
                env: Some(Default::default()),
                inherit_env: Some(true),
                timeout: input.timeout,
                signal: signal.cloned(),
                on_chunk: None,
                return_execution_errors: true,
            }),
        )
        .await?;

        let mut output_text = capture.output.clone();
        let truncated = capture.truncation.truncated;
        if truncated {
            output_text.push_str(&format!(
                "\n\n[Output truncated at {} limit]",
                format_size(crate::harness::utils::truncate::DEFAULT_MAX_BYTES)
            ));
        }

        if capture.cancelled {
            return Err(Box::new(ToolErrorMessage(if output_text.is_empty() {
                "Command aborted".to_owned()
            } else {
                format!("{output_text}\n\nCommand aborted")
            })));
        }
        if let Some(execution_error) = &capture.execution_error {
            return Err(Box::new(execution_error.clone()));
        }
        if let Some(exit_code) = capture.exit_code
            && exit_code != 0
        {
            return Err(Box::new(ToolErrorMessage(if output_text.is_empty() {
                format!("Command exited with code {exit_code}")
            } else {
                format!("{output_text}\n\nCommand exited with code {exit_code}")
            })));
        }

        Ok(AgentToolResult::text(
            if output_text.is_empty() {
                "(no output)".to_owned()
            } else {
                output_text
            },
            serde_json::to_value(PowerShellToolDetails {
                truncated: Some(truncated),
            })
            .unwrap_or(Json::Null),
        ))
    }
}

/// Create the `powershell` tool.
pub fn create_powershell_tool<TContext: ToolEnvContext + Send + Sync + 'static>()
-> AgentHarnessTool<TContext, Json> {
    AgentHarnessTool {
        name: "powershell".to_owned(),
        label: "powershell".to_owned(),
        description: "Execute a PowerShell command on Windows (pwsh elsewhere). The script is passed to PowerShell via an encoded command, so quoting is preserved exactly. Output is truncated at the byte limit.".to_owned(),
        parameters: powershell_parameters_schema(),
        execution_mode: None,
        prepare_arguments: None,
        execute: Arc::new(PowerShellToolExecutor),
    }
}

/// Convenience constructor producing an [`crate::types::AgentTool`] bound to one
/// execution environment.
pub fn create_powershell_tool_with_env(env: Arc<dyn ExecutionEnv>) -> crate::types::AgentTool {
    use crate::harness::tools::tool_context::ExecutionToolContext;
    create_powershell_tool::<ExecutionToolContext>().into_agent_tool(ExecutionToolContext::new(env))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_scripts_as_base64_utf16le() {
        let encoded = encode_command("Write-Host hi");
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .unwrap();
        let units: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();
        let decoded = String::from_utf16(&units).unwrap();
        assert!(decoded.starts_with(UTF8_OUTPUT_PREFIX));
        assert!(decoded.ends_with("Write-Host hi"));
    }

    #[test]
    fn parses_timeout() {
        let input: PowerShellToolInput =
            serde_json::from_value(serde_json::json!({"command": "ls", "timeout": 5})).unwrap();
        assert_eq!(input.command, "ls");
        assert_eq!(input.timeout, Some(5.0));
    }
}
