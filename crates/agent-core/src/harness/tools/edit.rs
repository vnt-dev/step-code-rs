//! The `edit` tool: exact-text replacement edits with diff details.
//!
//! Ported from `@step-harness/agent-core` `src\harness\tools\edit.ts`.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value as Json;

use crate::harness::tools::edit_diff::{
    Edit, apply_edits_to_normalized_content, detect_line_ending, generate_diff_string,
    generate_unified_patch, normalize_to_lf, restore_line_endings, strip_bom,
};
use crate::harness::tools::mutation_queue::with_file_mutation_queue;
use crate::harness::tools::path_utils::resolve_tool_path;
use crate::harness::tools::tool_context::ToolEnvContext;
use crate::harness::types::{
    AbortSignal, AgentHarnessTool, ContextToolExecutor, ExecutionEnv, FileError, FileKind,
};
use crate::types::{AgentToolResult, ToolError, ToolUpdateCallback, text_content};

/// JSON Schema for the edit tool parameters.
pub fn edit_parameters_schema() -> Json {
    serde_json::json!({
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to edit (relative or absolute)"
            },
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "oldText": {
                            "type": "string",
                            "description": "Exact text for one targeted replacement. It must be unique in the original file and must not overlap with any other edits[].oldText in the same call."
                        },
                        "newText": {
                            "type": "string",
                            "description": "Replacement text for this targeted edit."
                        }
                    },
                    "required": ["oldText", "newText"]
                },
                "description": "One or more targeted replacements. Each edit is matched against the original file, not incrementally. Do not include overlapping or nested edits. If two changes touch the same block or nearby lines, merge them into one edit instead."
            }
        },
        "required": ["path", "edits"]
    })
}

/// Validated edit tool arguments.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EditToolInput {
    pub path: String,
    pub edits: Vec<Edit>,
}

/// Details attached to an edit tool result.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EditToolDetails {
    pub diff: String,
    pub patch: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_changed_line: Option<usize>,
}

/// Normalize legacy and malformed edit arguments before schema validation.
///
/// Handles models that send `edits` as a JSON string, a single edit object, or
/// the legacy top-level `oldText`/`newText` pair.
pub fn prepare_edit_arguments(input: Json) -> Json {
    let Json::Object(mut args) = input else {
        return input;
    };

    // edits as a JSON string.
    if let Some(Json::String(text)) = args.get("edits").cloned()
        && let Ok(parsed) = serde_json::from_str::<Json>(&text)
    {
        match parsed {
            Json::Array(_) => {
                args.insert("edits".to_owned(), parsed);
            }
            Json::Object(object) if is_single_edit(&object) => {
                args.insert("edits".to_owned(), Json::Array(vec![Json::Object(object)]));
            }
            _ => {}
        }
    } else if let Some(Json::Object(object)) = args.get("edits").cloned()
        && is_single_edit(&object)
    {
        args.insert("edits".to_owned(), Json::Array(vec![Json::Object(object)]));
    }

    // Legacy top-level oldText/newText pair.
    let legacy_old_text = args.get("oldText").and_then(Json::as_str);
    let legacy_new_text = args.get("newText").and_then(Json::as_str);
    if let (Some(old_text), Some(new_text)) = (legacy_old_text, legacy_new_text) {
        let old_text = old_text.to_owned();
        let new_text = new_text.to_owned();
        let mut edits: Vec<Json> = match args.get("edits") {
            Some(Json::Array(existing)) => existing.clone(),
            _ => Vec::new(),
        };
        edits.push(serde_json::json!({ "oldText": old_text, "newText": new_text }));
        args.remove("oldText");
        args.remove("newText");
        args.insert("edits".to_owned(), Json::Array(edits));
    }

    Json::Object(args)
}

fn is_single_edit(value: &serde_json::Map<String, Json>) -> bool {
    value.get("oldText").and_then(Json::as_str).is_some()
        && value.get("newText").and_then(Json::as_str).is_some()
}

fn validate_edit_input(input: &EditToolInput) -> Result<(), ToolError> {
    if input.edits.is_empty() {
        return Err(crate::types::tool_error(
            "Edit tool input is invalid. edits must contain at least one replacement.",
        ));
    }
    Ok(())
}

fn edit_access_error(path: &str, error: FileError) -> ToolError {
    Box::new(EditAccessError {
        message: format!(
            "Could not edit file: {path}. Error code: {}.",
            error.code_name()
        ),
        cause: error,
    })
}

/// An edit failure wrapping a filesystem error.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct EditAccessError {
    pub message: String,
    #[source]
    pub cause: FileError,
}

struct EditToolExecutor;

#[async_trait::async_trait]
impl<TContext: ToolEnvContext + Send + Sync + 'static> ContextToolExecutor<TContext, Json>
    for EditToolExecutor
{
    async fn execute(
        &self,
        _tool_call_id: &str,
        params: &serde_json::Map<String, Json>,
        signal: Option<&AbortSignal>,
        _on_update: Option<&ToolUpdateCallback<Json>>,
        context: &TContext,
    ) -> Result<AgentToolResult<Json>, ToolError> {
        let input: EditToolInput =
            serde_json::from_value(Json::Object(params.clone())).map_err(ToolError::from)?;
        validate_edit_input(&input)?;
        let env = context.env().clone();

        let absolute_path = resolve_tool_path(env.as_ref(), &input.path, signal).await?;
        let edits = input.edits.clone();
        let path = input.path.clone();
        let env_for_mutation = env.clone();
        let queue_path = absolute_path.clone();
        let outcome = with_file_mutation_queue(&env, &queue_path, move || {
            let env = env_for_mutation;
            let edits = edits;
            let path = path;
            let absolute_path = absolute_path;
            async move {
                if signal.is_some_and(|signal| signal.is_cancelled()) {
                    return Err(crate::types::tool_error("Operation aborted"));
                }
                let info = env
                    .file_info(&absolute_path, signal)
                    .await
                    .map_err(|error| edit_access_error(&path, error))?;
                if info.kind != FileKind::File && info.kind != FileKind::Symlink {
                    return Err(crate::types::tool_error(format!(
                        "Could not edit file: {path}. Path is not a file."
                    )));
                }

                let original = env
                    .read_text_file(&absolute_path, signal)
                    .await
                    .map_err(|error| edit_access_error(&path, error))?;
                if signal.is_some_and(|signal| signal.is_cancelled()) {
                    return Err(crate::types::tool_error("Operation aborted"));
                }

                let (bom, content) = strip_bom(&original);
                let original_ending = detect_line_ending(content);
                let normalized_content = normalize_to_lf(content);
                let applied = apply_edits_to_normalized_content(&normalized_content, &edits, &path)
                    .map_err(crate::types::tool_error)?;
                if signal.is_some_and(|signal| signal.is_cancelled()) {
                    return Err(crate::types::tool_error("Operation aborted"));
                }

                let final_content = format!(
                    "{bom}{}",
                    restore_line_endings(&applied.new_content, original_ending)
                );
                env.write_file(
                    &absolute_path,
                    crate::harness::types::FileContent::Text(final_content),
                    signal,
                )
                .await
                .map_err(|error| edit_access_error(&path, error))?;
                if signal.is_some_and(|signal| signal.is_cancelled()) {
                    return Err(crate::types::tool_error("Operation aborted"));
                }

                let diff_result =
                    generate_diff_string(&applied.base_content, &applied.new_content, 4);
                Ok(EditToolDetails {
                    diff: diff_result.diff,
                    patch: generate_unified_patch(
                        &path,
                        &applied.base_content,
                        &applied.new_content,
                        4,
                    ),
                    first_changed_line: diff_result.first_changed_line,
                })
            }
        })
        .await?;

        Ok(AgentToolResult {
            content: vec![text_content(format!(
                "Successfully replaced {} block(s) in {}",
                input.edits.len(),
                input.path
            ))],
            details: serde_json::to_value(&outcome).unwrap_or(Json::Null),
            usage: None,
            added_tool_names: None,
            terminate: None,
        })
    }
}

/// Create the `edit` tool.
pub fn create_edit_tool<TContext: ToolEnvContext + Send + Sync + 'static>()
-> AgentHarnessTool<TContext, Json> {
    AgentHarnessTool {
        name: "edit".to_owned(),
        label: "edit".to_owned(),
        description: "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes.".to_owned(),
        parameters: edit_parameters_schema(),
        execution_mode: None,
        prepare_arguments: Some(Arc::new(prepare_edit_arguments)),
        execute: Arc::new(EditToolExecutor),
    }
}

/// Convenience constructor producing an [`crate::types::AgentTool`] bound to one
/// execution environment.
pub fn create_edit_tool_with_env(env: Arc<dyn ExecutionEnv>) -> crate::types::AgentTool {
    use crate::harness::tools::tool_context::ExecutionToolContext;
    create_edit_tool::<ExecutionToolContext>().into_agent_tool(ExecutionToolContext::new(env))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepares_legacy_single_edit() {
        let prepared = prepare_edit_arguments(serde_json::json!({
            "path": "f.txt",
            "oldText": "a",
            "newText": "b"
        }));
        assert!(prepared.get("oldText").is_none());
        let edits = prepared["edits"].as_array().unwrap();
        assert_eq!(edits.len(), 1);
        assert_eq!(edits[0]["oldText"], "a");
    }

    #[test]
    fn prepares_edits_json_string() {
        let prepared = prepare_edit_arguments(serde_json::json!({
            "path": "f.txt",
            "edits": "[{\"oldText\":\"a\",\"newText\":\"b\"}]"
        }));
        assert_eq!(prepared["edits"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn prepares_single_edit_object() {
        let prepared = prepare_edit_arguments(serde_json::json!({
            "path": "f.txt",
            "edits": {"oldText": "a", "newText": "b"}
        }));
        assert_eq!(prepared["edits"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn validates_non_empty_edits() {
        let input = EditToolInput {
            path: "f".to_owned(),
            edits: vec![Edit {
                old_text: "a".to_owned(),
                new_text: "b".to_owned(),
            }],
        };
        assert!(validate_edit_input(&input).is_ok());
        let empty = EditToolInput {
            path: "f".to_owned(),
            edits: vec![],
        };
        assert!(validate_edit_input(&empty).is_err());
    }
}
