//! The `ls` tool: directory listing with directory indicators.
//!
//! Ported functionally from `@step-harness/coding-agent`
//! `src/core/tools/ls.ts`. Entries are sorted case-insensitively, directories
//! carry a `/` suffix, and output is capped by entry count and byte size.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value as Json;

use crate::harness::tools::path_utils::resolve_tool_path;
use crate::harness::tools::tool_context::ToolEnvContext;
use crate::harness::types::{AbortSignal, AgentHarnessTool, ContextToolExecutor, ExecutionEnv};
use crate::harness::utils::truncate::{
    DEFAULT_MAX_BYTES, TruncationResult, format_size, truncate_head,
};
use crate::types::{AgentToolResult, ToolError, ToolUpdateCallback};

/// Default maximum number of listed entries.
pub const DEFAULT_ENTRY_LIMIT: usize = 500;

/// JSON Schema for the ls tool parameters.
pub fn ls_parameters_schema() -> Json {
    serde_json::json!({
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory to list (default: current directory)"
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "description": "Maximum number of entries to return (default: 500)"
            }
        }
    })
}

/// Validated ls tool arguments.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LsToolInput {
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Details attached to an ls tool result.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LsToolDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncation: Option<TruncationResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry_limit_reached: Option<usize>,
}

struct LsToolExecutor;

#[async_trait::async_trait]
impl<TContext: ToolEnvContext + Send + Sync + 'static> ContextToolExecutor<TContext, Json>
    for LsToolExecutor
{
    async fn execute(
        &self,
        _tool_call_id: &str,
        params: &serde_json::Map<String, Json>,
        signal: Option<&AbortSignal>,
        _on_update: Option<&ToolUpdateCallback<Json>>,
        context: &TContext,
    ) -> Result<AgentToolResult<Json>, ToolError> {
        let input: LsToolInput =
            serde_json::from_value(Json::Object(params.clone())).map_err(ToolError::from)?;
        let env = context.env();
        let requested = input.path.as_deref().unwrap_or(".");
        let absolute_path = resolve_tool_path(env.as_ref(), requested, signal).await?;
        let effective_limit = input.limit.unwrap_or(DEFAULT_ENTRY_LIMIT);

        let mut entries = env.list_dir(&absolute_path, signal).await?;
        entries.sort_by(|left, right| {
            left.name
                .to_lowercase()
                .cmp(&right.name.to_lowercase())
                .then_with(|| left.name.cmp(&right.name))
        });

        let mut names: Vec<String> = Vec::new();
        let mut entry_limit_reached = None;
        for entry in &entries {
            if names.len() >= effective_limit {
                entry_limit_reached = Some(effective_limit);
                break;
            }
            let suffix = match entry.kind {
                crate::harness::types::FileKind::Directory => "/",
                _ => "",
            };
            names.push(format!("{}{suffix}", entry.name));
        }

        if names.is_empty() {
            return Ok(AgentToolResult::text("(empty directory)", Json::Null));
        }

        let truncation = truncate_head(&names.join("\n"), &Default::default());
        let mut output = truncation.content.clone();
        let mut details = LsToolDetails::default();
        let mut notices: Vec<String> = Vec::new();
        if let Some(limit) = entry_limit_reached {
            notices.push(format!(
                "{limit} entries limit reached. Use limit={} for more",
                limit * 2
            ));
            details.entry_limit_reached = Some(limit);
        }
        if truncation.truncated {
            notices.push(format!("{} limit reached", format_size(DEFAULT_MAX_BYTES)));
            details.truncation = Some(truncation);
        }
        if !notices.is_empty() {
            output.push_str(&format!("\n\n[{}]", notices.join(". ")));
        }

        Ok(AgentToolResult::text(
            output,
            serde_json::to_value(details).unwrap_or(Json::Null),
        ))
    }
}

/// Create the `ls` tool.
pub fn create_ls_tool<TContext: ToolEnvContext + Send + Sync + 'static>()
-> AgentHarnessTool<TContext, Json> {
    AgentHarnessTool {
        name: "ls".to_owned(),
        label: "ls".to_owned(),
        description: format!(
            "List directory contents. Returns entries sorted alphabetically, with '/' suffix for directories. Includes dotfiles. Output is truncated to {DEFAULT_ENTRY_LIMIT} entries or {}KB (whichever is hit first).",
            DEFAULT_MAX_BYTES / 1024
        ),
        parameters: ls_parameters_schema(),
        execution_mode: None,
        prepare_arguments: None,
        execute: Arc::new(LsToolExecutor),
    }
}

/// Convenience constructor producing an [`crate::types::AgentTool`] bound to one
/// execution environment.
pub fn create_ls_tool_with_env(env: Arc<dyn ExecutionEnv>) -> crate::types::AgentTool {
    use crate::harness::tools::tool_context::ExecutionToolContext;
    create_ls_tool::<ExecutionToolContext>().into_agent_tool(ExecutionToolContext::new(env))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_path_and_limit() {
        let input: LsToolInput =
            serde_json::from_value(serde_json::json!({"path": "src", "limit": 10})).unwrap();
        assert_eq!(input.path.as_deref(), Some("src"));
        assert_eq!(input.limit, Some(10));

        let input: LsToolInput = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(input.path, None);
        assert_eq!(input.limit, None);
    }
}
