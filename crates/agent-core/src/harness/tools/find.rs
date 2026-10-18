//! The `find` tool: glob-based file search respecting ignore files.
//!
//! Ported functionally from `@step-harness/coding-agent`
//! `src/core/tools/find.ts`. The TypeScript source shells out to `fd` or
//! `git ls-files` when available; this port always uses the `ignore` crate's
//! walker, which honors `.gitignore` and hidden-file rules in-process.

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

/// Default maximum number of results.
pub const DEFAULT_FIND_LIMIT: usize = 1000;

/// JSON Schema for the find tool parameters.
pub fn find_parameters_schema() -> Json {
    serde_json::json!({
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern to match files, e.g. '*.ts', '**/*.json', or 'src/**/*.spec.ts'"
            },
            "path": {
                "type": "string",
                "description": "Directory to search in (default: current directory)"
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "description": "Maximum number of results (default: 1000)"
            }
        },
        "required": ["pattern"]
    })
}

/// Validated find tool arguments.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FindToolInput {
    pub pattern: String,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Details attached to a find tool result.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FindToolDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncation: Option<TruncationResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_limit_reached: Option<usize>,
}

struct FindToolExecutor;

#[async_trait::async_trait]
impl<TContext: ToolEnvContext + Send + Sync + 'static> ContextToolExecutor<TContext, Json>
    for FindToolExecutor
{
    async fn execute(
        &self,
        _tool_call_id: &str,
        params: &serde_json::Map<String, Json>,
        signal: Option<&AbortSignal>,
        _on_update: Option<&ToolUpdateCallback<Json>>,
        context: &TContext,
    ) -> Result<AgentToolResult<Json>, ToolError> {
        let input: FindToolInput =
            serde_json::from_value(Json::Object(params.clone())).map_err(ToolError::from)?;
        let env = context.env();
        let requested = input.path.as_deref().unwrap_or(".");
        let root = resolve_tool_path(env.as_ref(), requested, signal).await?;
        let effective_limit = input.limit.unwrap_or(DEFAULT_FIND_LIMIT);

        let matcher = globset::GlobBuilder::new(&input.pattern)
            .literal_separator(true)
            .build()
            .map_err(|error| crate::types::tool_error(format!("invalid glob pattern: {error}")))?
            .compile_matcher();

        let cancelled = signal.map(|signal| signal.is_cancelled()).unwrap_or(false);
        if cancelled {
            return Err(crate::types::tool_error("Operation aborted"));
        }

        let root_for_walk = root.clone();
        let matches = tokio::task::spawn_blocking(move || {
            let mut matches: Vec<String> = Vec::new();
            let walker = ignore::WalkBuilder::new(&root_for_walk)
                .standard_filters(true)
                .hidden(true)
                .follow_links(false)
                .build();
            for entry in walker.flatten() {
                let is_file = entry
                    .file_type()
                    .map(|kind| kind.is_file())
                    .unwrap_or(false);
                if !is_file {
                    continue;
                }
                let path = entry.path();
                let relative = path
                    .strip_prefix(&root_for_walk)
                    .unwrap_or(path)
                    .to_string_lossy()
                    .replace('\\', "/");
                let file_name = path
                    .file_name()
                    .map(|name| name.to_string_lossy().to_string())
                    .unwrap_or_default();
                if matcher.is_match(&relative) || matcher.is_match(&file_name) {
                    matches.push(relative);
                }
            }
            matches.sort();
            matches
        })
        .await
        .map_err(|error| crate::types::tool_error(format!("find failed: {error}")))?;

        if matches.is_empty() {
            return Ok(AgentToolResult::text(
                "No files found matching pattern",
                Json::Null,
            ));
        }

        let mut matches = matches;
        let mut result_limit_reached = None;
        if matches.len() > effective_limit {
            matches.truncate(effective_limit);
            result_limit_reached = Some(effective_limit);
        }

        let truncation = truncate_head(&matches.join("\n"), &Default::default());
        let mut output = truncation.content.clone();
        let mut details = FindToolDetails::default();
        let mut notices: Vec<String> = Vec::new();
        if let Some(limit) = result_limit_reached {
            notices.push(format!(
                "{limit} results limit reached. Use limit={} for more",
                limit * 2
            ));
            details.result_limit_reached = Some(limit);
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

/// Create the `find` tool.
pub fn create_find_tool<TContext: ToolEnvContext + Send + Sync + 'static>()
-> AgentHarnessTool<TContext, Json> {
    AgentHarnessTool {
        name: "find".to_owned(),
        label: "find".to_owned(),
        description: format!(
            "Search for files by glob pattern. Returns matching file paths relative to the search directory. Respects .gitignore. Output is truncated to {DEFAULT_FIND_LIMIT} results or {}KB (whichever is hit first).",
            DEFAULT_MAX_BYTES / 1024
        ),
        parameters: find_parameters_schema(),
        execution_mode: None,
        prepare_arguments: None,
        execute: Arc::new(FindToolExecutor),
    }
}

/// Convenience constructor producing an [`crate::types::AgentTool`] bound to one
/// execution environment.
pub fn create_find_tool_with_env(env: Arc<dyn ExecutionEnv>) -> crate::types::AgentTool {
    use crate::harness::tools::tool_context::ExecutionToolContext;
    create_find_tool::<ExecutionToolContext>().into_agent_tool(ExecutionToolContext::new(env))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_patterns() {
        let input: FindToolInput =
            serde_json::from_value(serde_json::json!({"pattern": "**/*.ts", "limit": 5})).unwrap();
        assert_eq!(input.pattern, "**/*.ts");
        assert_eq!(input.limit, Some(5));
    }

    #[test]
    fn requires_a_pattern() {
        let error = serde_json::from_value::<FindToolInput>(serde_json::json!({}));
        assert!(error.is_err());
    }
}
