//! The `grep` tool: regex/literal content search respecting ignore files.
//!
//! Ported functionally from `@step-harness/coding-agent`
//! `src/core/tools/grep.ts`. The TypeScript source shells out to `rg` or
//! `git grep` when available; this port always searches in-process with the
//! `ignore` walker and the `regex` crate.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value as Json;

use crate::harness::tools::path_utils::resolve_tool_path;
use crate::harness::tools::tool_context::ToolEnvContext;
use crate::harness::types::{AbortSignal, AgentHarnessTool, ContextToolExecutor, ExecutionEnv};
use crate::harness::utils::truncate::{TruncationResult, truncate_head};
use crate::types::{AgentToolResult, ToolError, ToolUpdateCallback};

/// Default maximum number of matches.
pub const DEFAULT_MATCH_LIMIT: usize = 100;
/// Files larger than this are skipped (binary/large-asset guard).
pub const MAX_SEARCH_FILE_BYTES: u64 = 10 * 1024 * 1024;

/// JSON Schema for the grep tool parameters.
pub fn grep_parameters_schema() -> Json {
    serde_json::json!({
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Search pattern (regex or literal string)"
            },
            "path": {
                "type": "string",
                "description": "Directory or file to search (default: current directory)"
            },
            "glob": {
                "type": "string",
                "description": "Filter files by glob pattern, e.g. '*.ts' or '**/*.spec.ts'"
            },
            "ignoreCase": {
                "type": "boolean",
                "description": "Case-insensitive search (default: false)"
            },
            "literal": {
                "type": "boolean",
                "description": "Treat pattern as literal string instead of regex (default: false)"
            },
            "context": {
                "type": "integer",
                "minimum": 0,
                "description": "Number of lines to show before and after each match (default: 0)"
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "description": "Maximum number of matches to return (default: 100)"
            }
        },
        "required": ["pattern"]
    })
}

/// Validated grep tool arguments.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GrepToolInput {
    pub pattern: String,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub glob: Option<String>,
    #[serde(default)]
    pub ignore_case: Option<bool>,
    #[serde(default)]
    pub literal: Option<bool>,
    #[serde(default)]
    pub context: Option<usize>,
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Details attached to a grep tool result.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GrepToolDetails {
    pub match_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncation: Option<TruncationResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_limit_reached: Option<usize>,
}

/// One matched line with optional context.
struct MatchLine {
    path: String,
    line_number: usize,
    text: String,
    /// `Some(true)` for the matching line, `Some(false)` for context.
    is_match: bool,
}

struct GrepToolExecutor;

#[async_trait::async_trait]
impl<TContext: ToolEnvContext + Send + Sync + 'static> ContextToolExecutor<TContext, Json>
    for GrepToolExecutor
{
    async fn execute(
        &self,
        _tool_call_id: &str,
        params: &serde_json::Map<String, Json>,
        signal: Option<&AbortSignal>,
        _on_update: Option<&ToolUpdateCallback<Json>>,
        context: &TContext,
    ) -> Result<AgentToolResult<Json>, ToolError> {
        let input: GrepToolInput =
            serde_json::from_value(Json::Object(params.clone())).map_err(ToolError::from)?;
        let env = context.env();
        let requested = input.path.as_deref().unwrap_or(".");
        let root = resolve_tool_path(env.as_ref(), requested, signal).await?;
        let effective_limit = input.limit.unwrap_or(DEFAULT_MATCH_LIMIT);
        let context_lines = input.context.unwrap_or(0);

        let pattern = if input.literal.unwrap_or(false) {
            regex::escape(&input.pattern)
        } else {
            input.pattern.clone()
        };
        let mut builder = regex::RegexBuilder::new(&pattern);
        builder.case_insensitive(input.ignore_case.unwrap_or(false));
        let regex = builder
            .build()
            .map_err(|error| crate::types::tool_error(format!("invalid regex pattern: {error}")))?;

        let glob_matcher = match &input.glob {
            Some(glob) => Some(
                globset::GlobBuilder::new(glob)
                    .literal_separator(false)
                    .build()
                    .map_err(|error| crate::types::tool_error(format!("invalid glob: {error}")))?
                    .compile_matcher(),
            ),
            None => None,
        };

        if signal.map(|signal| signal.is_cancelled()).unwrap_or(false) {
            return Err(crate::types::tool_error("Operation aborted"));
        }

        let root_for_walk = root.clone();
        let matches = tokio::task::spawn_blocking(move || {
            let mut matches: Vec<MatchLine> = Vec::new();
            let root_is_file = std::fs::metadata(&root_for_walk)
                .map(|metadata| metadata.is_file())
                .unwrap_or(false);
            let walker = ignore::WalkBuilder::new(&root_for_walk)
                .standard_filters(true)
                .hidden(true)
                .follow_links(false)
                .build();
            for entry in walker.flatten() {
                let is_file = if root_is_file {
                    entry.depth() == 0
                } else {
                    entry
                        .file_type()
                        .map(|kind| kind.is_file())
                        .unwrap_or(false)
                };
                if !is_file {
                    continue;
                }
                let path = entry.path();
                let metadata = match std::fs::metadata(path) {
                    Ok(metadata) => metadata,
                    Err(_) => continue,
                };
                if metadata.len() > MAX_SEARCH_FILE_BYTES {
                    continue;
                }
                if let Some(matcher) = &glob_matcher {
                    let relative = path
                        .strip_prefix(&root_for_walk)
                        .unwrap_or(path)
                        .to_string_lossy()
                        .replace('\\', "/");
                    let file_name = path
                        .file_name()
                        .map(|name| name.to_string_lossy().to_string())
                        .unwrap_or_default();
                    if !matcher.is_match(&relative) && !matcher.is_match(&file_name) {
                        continue;
                    }
                }
                let Ok(content) = std::fs::read_to_string(path) else {
                    continue; // binary or unreadable
                };
                let lines: Vec<&str> = content.lines().collect();
                for (index, line) in lines.iter().enumerate() {
                    if !regex.is_match(line) {
                        continue;
                    }
                    let display_path = path
                        .strip_prefix(&root_for_walk)
                        .unwrap_or(path)
                        .to_string_lossy()
                        .replace('\\', "/");
                    let display_path = if display_path.is_empty() {
                        path.file_name()
                            .map(|name| name.to_string_lossy().to_string())
                            .unwrap_or_default()
                    } else {
                        display_path
                    };
                    if context_lines > 0 {
                        let start = index.saturating_sub(context_lines);
                        let end = (index + context_lines + 1).min(lines.len());
                        for (offset, line_index) in (start..end).enumerate() {
                            let is_match = line_index == index;
                            matches.push(MatchLine {
                                path: display_path.clone(),
                                line_number: line_index + 1,
                                text: lines[line_index].to_owned(),
                                is_match,
                            });
                            let _ = offset;
                        }
                    } else {
                        matches.push(MatchLine {
                            path: display_path.clone(),
                            line_number: index + 1,
                            text: (*line).to_owned(),
                            is_match: true,
                        });
                    }
                }
            }
            matches
        })
        .await
        .map_err(|error| crate::types::tool_error(format!("grep failed: {error}")))?;

        let match_count = matches.iter().filter(|line| line.is_match).count();
        if matches.is_empty() {
            return Ok(AgentToolResult::text(
                "No matches found",
                serde_json::to_value(GrepToolDetails {
                    match_count: 0,
                    truncation: None,
                    result_limit_reached: None,
                })
                .unwrap_or(Json::Null),
            ));
        }

        let mut matches = matches;
        let mut result_limit_reached = None;
        if match_count > effective_limit {
            // Keep whole match groups (match plus its context lines).
            let mut kept = 0;
            let mut cut = matches.len();
            for (index, line) in matches.iter().enumerate() {
                if line.is_match {
                    kept += 1;
                    if kept > effective_limit {
                        cut = index;
                        break;
                    }
                }
            }
            matches.truncate(cut);
            result_limit_reached = Some(effective_limit);
        }

        let rendered: Vec<String> = matches
            .iter()
            .map(|line| {
                let separator = if line.is_match { ":" } else { "-" };
                format!(
                    "{}{}{}:{}",
                    line.path, separator, line.line_number, line.text
                )
            })
            .collect();
        let truncation = truncate_head(&rendered.join("\n"), &Default::default());
        let mut output = truncation.content.clone();
        let mut notices: Vec<String> = Vec::new();
        if let Some(limit) = result_limit_reached {
            notices.push(format!(
                "{limit} matches limit reached. Use limit={} for more",
                limit * 2
            ));
        }
        if truncation.truncated {
            notices.push("output size limit reached".to_owned());
        }
        if !notices.is_empty() {
            output.push_str(&format!("\n\n[{}]", notices.join(". ")));
        }

        Ok(AgentToolResult::text(
            output,
            serde_json::to_value(GrepToolDetails {
                match_count,
                truncation: Some(truncation),
                result_limit_reached,
            })
            .unwrap_or(Json::Null),
        ))
    }
}

/// Create the `grep` tool.
pub fn create_grep_tool<TContext: ToolEnvContext + Send + Sync + 'static>()
-> AgentHarnessTool<TContext, Json> {
    AgentHarnessTool {
        name: "grep".to_owned(),
        label: "grep".to_owned(),
        description: "Search file contents with a regex or literal string. Returns matching lines as path:line: text. Respects .gitignore. Use the glob parameter to filter by file pattern, and context to show surrounding lines.".to_owned(),
        parameters: grep_parameters_schema(),
        execution_mode: None,
        prepare_arguments: None,
        execute: Arc::new(GrepToolExecutor),
    }
}

/// Convenience constructor producing an [`crate::types::AgentTool`] bound to one
/// execution environment.
pub fn create_grep_tool_with_env(env: Arc<dyn ExecutionEnv>) -> crate::types::AgentTool {
    use crate::harness::tools::tool_context::ExecutionToolContext;
    create_grep_tool::<ExecutionToolContext>().into_agent_tool(ExecutionToolContext::new(env))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_arguments() {
        let input: GrepToolInput = serde_json::from_value(serde_json::json!({
            "pattern": "fn main",
            "path": "src",
            "glob": "*.rs",
            "ignoreCase": true,
            "literal": false,
            "context": 2,
            "limit": 50
        }))
        .unwrap();
        assert_eq!(input.pattern, "fn main");
        assert_eq!(input.glob.as_deref(), Some("*.rs"));
        assert_eq!(input.ignore_case, Some(true));
        assert_eq!(input.context, Some(2));
    }

    #[test]
    fn requires_a_pattern() {
        assert!(serde_json::from_value::<GrepToolInput>(serde_json::json!({})).is_err());
    }
}
