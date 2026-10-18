//! The `read` tool: file reading with text and image support.
//!
//! Ported from `@step-harness/agent-core` `src\harness\tools\read.ts`.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value as Json;
use step_harness_providers::types::{ImageContent, UserContentBlock};

use crate::harness::tools::image::{
    DefaultImageProcessor, detect_supported_image_mime_type, guess_supported_image_mime_type,
};
use crate::harness::tools::path_utils::resolve_read_tool_path;
use crate::harness::tools::tool_context::ToolEnvContext;
use crate::harness::types::{AbortSignal, AgentHarnessTool, ContextToolExecutor, ExecutionEnv};
use crate::harness::utils::truncate::{
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncatedBy, TruncationResult, format_size, truncate_head,
};
use crate::types::{AgentToolResult, ToolError, ToolUpdateCallback, text_content};

/// JSON Schema for the read tool parameters.
pub fn read_parameters_schema() -> Json {
    serde_json::json!({
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read (relative or absolute)"
            },
            "offset": {
                "type": "integer",
                "minimum": 1,
                "description": "Line number to start reading from (1-indexed)"
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "description": "Maximum number of lines to read"
            }
        },
        "required": ["path"]
    })
}

/// Validated read tool arguments.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReadToolInput {
    pub path: String,
    #[serde(default)]
    pub offset: Option<usize>,
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Details attached to a read tool result.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReadToolDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncation: Option<TruncationResult>,
}

/// Successful image processing result.
#[derive(Clone, Debug)]
pub enum ReadImageProcessorResult {
    Ok {
        data: String,
        mime_type: String,
        hints: Vec<String>,
    },
    Err {
        message: String,
    },
}

/// Optional image conversion/resizing implementation.
#[async_trait::async_trait]
pub trait ReadImageProcessor: Send + Sync {
    async fn process(
        &self,
        bytes: &[u8],
        mime_type: &str,
        options: ReadImageProcessorOptions,
    ) -> ReadImageProcessorResult;
}

/// Options passed to a [`ReadImageProcessor`].
#[derive(Clone, Copy, Debug)]
pub struct ReadImageProcessorOptions {
    /// Whether the processor should resize images. Default: true.
    pub auto_resize_images: bool,
}

/// Options for [`create_read_tool`].
#[derive(Clone, Default)]
pub struct ReadToolOptions {
    pub auto_resize_images: Option<bool>,
    pub image_processor: Option<Arc<dyn ReadImageProcessor>>,
}

struct ReadToolExecutor {
    options: Option<ReadToolOptions>,
}

#[async_trait::async_trait]
impl<TContext: ToolEnvContext + Send + Sync + 'static> ContextToolExecutor<TContext, Json>
    for ReadToolExecutor
{
    async fn execute(
        &self,
        _tool_call_id: &str,
        params: &serde_json::Map<String, Json>,
        signal: Option<&AbortSignal>,
        _on_update: Option<&ToolUpdateCallback<Json>>,
        context: &TContext,
    ) -> Result<AgentToolResult<Json>, ToolError> {
        let input: ReadToolInput =
            serde_json::from_value(Json::Object(params.clone())).map_err(ToolError::from)?;
        if input.offset == Some(0) || input.limit == Some(0) {
            return Err(crate::types::tool_error(
                "offset and limit must be positive integers",
            ));
        }
        let env = context.env();

        let absolute_path = resolve_read_tool_path(env.as_ref(), &input.path, signal).await?;
        let prefix = env.read_file_prefix(&absolute_path, 256, signal).await?;

        if let Some(mime_type) = guess_supported_image_mime_type(&prefix) {
            const MAX_IMAGE_BYTES: u64 = 32 * 1024 * 1024;
            if env.file_info(&absolute_path, signal).await?.size > MAX_IMAGE_BYTES {
                return Err(crate::types::tool_error(
                    "Image exceeds the 32 MiB read limit",
                ));
            }
            let bytes = env
                .read_file_prefix(&absolute_path, MAX_IMAGE_BYTES as usize + 1, signal)
                .await?;
            if bytes.len() > MAX_IMAGE_BYTES as usize {
                return Err(crate::types::tool_error(
                    "Image exceeds the 32 MiB read limit",
                ));
            }
            if detect_supported_image_mime_type(&bytes).is_none() {
                return Err(crate::types::tool_error(
                    "Unsupported animated PNG or malformed image header",
                ));
            }
            let processor = self
                .options
                .as_ref()
                .and_then(|options| options.image_processor.clone())
                .unwrap_or_else(|| Arc::new(DefaultImageProcessor));
            {
                let processed = processor
                    .process(
                        &bytes,
                        mime_type,
                        ReadImageProcessorOptions {
                            auto_resize_images: self
                                .options
                                .as_ref()
                                .and_then(|options| options.auto_resize_images)
                                .unwrap_or(true),
                        },
                    )
                    .await;
                match processed {
                    ReadImageProcessorResult::Err { message } => {
                        return Ok(AgentToolResult {
                            content: vec![text_content(format!(
                                "Read image file [{mime_type}]\n{message}"
                            ))],
                            details: Json::Null,
                            usage: None,
                            added_tool_names: None,
                            terminate: None,
                        });
                    }
                    ReadImageProcessorResult::Ok {
                        data,
                        mime_type,
                        hints,
                    } => {
                        let hint_text = if hints.is_empty() {
                            String::new()
                        } else {
                            format!("\n{}", hints.join("\n"))
                        };
                        return Ok(AgentToolResult {
                            content: vec![
                                text_content(format!("Read image file [{mime_type}]{hint_text}")),
                                UserContentBlock::Image(ImageContent { data, mime_type }),
                            ],
                            details: Json::Null,
                            usage: None,
                            added_tool_names: None,
                            terminate: None,
                        });
                    }
                }
            }
        }

        let start_line = input.offset.unwrap_or(1) - 1;
        let start_line_display = start_line.saturating_add(1);
        let page = env
            .read_text_page(
                &absolute_path,
                &crate::harness::types::TextFilePageOptions {
                    offset: start_line,
                    limit: input.limit,
                    max_bytes: DEFAULT_MAX_BYTES,
                    signal: signal.cloned(),
                },
            )
            .await?;
        let total_file_lines = page.total_lines;
        let user_limited_lines = input.limit.map(|_| page.selected_lines);
        let mut truncation = truncate_head(&page.content, &Default::default());
        if page.byte_truncated {
            truncation.truncated = true;
        }
        let mut details: Option<ReadToolDetails> = None;
        let output_text;
        if truncation.first_line_exceeds_limit {
            let first_line_size = format_size(page.first_line_bytes);
            output_text = format!(
                "[Line {start_line_display} is {first_line_size}, exceeds {} limit. Use bash: sed -n '{start_line_display}p' {} | head -c {DEFAULT_MAX_BYTES}]",
                format_size(DEFAULT_MAX_BYTES),
                input.path
            );
            details = Some(ReadToolDetails {
                truncation: Some(truncation),
            });
        } else if truncation.truncated {
            let end_line_display = start_line_display + truncation.output_lines - 1;
            let next_offset = end_line_display + 1;
            let mut text = truncation.content.clone();
            if truncation.truncated_by == Some(TruncatedBy::Lines) {
                text.push_str(&format!(
                    "\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines}. Use offset={next_offset} to continue.]"
                ));
            } else {
                text.push_str(&format!(
                    "\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines} ({} limit). Use offset={next_offset} to continue.]",
                    format_size(DEFAULT_MAX_BYTES)
                ));
            }
            output_text = text;
            details = Some(ReadToolDetails {
                truncation: Some(truncation),
            });
        } else if let Some(user_limited_lines) = user_limited_lines
            && page.has_more
        {
            let remaining = total_file_lines - start_line - user_limited_lines;
            let next_offset = start_line + user_limited_lines + 1;
            output_text = format!(
                "{}\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]",
                truncation.content
            );
        } else {
            output_text = truncation.content.clone();
        }

        Ok(AgentToolResult {
            content: vec![text_content(output_text)],
            details: match details {
                Some(details) => serde_json::to_value(details).unwrap_or(Json::Null),
                None => Json::Null,
            },
            usage: None,
            added_tool_names: None,
            terminate: None,
        })
    }
}

/// Create the `read` tool.
pub fn create_read_tool<TContext: ToolEnvContext + Send + Sync + 'static>(
    options: Option<ReadToolOptions>,
) -> AgentHarnessTool<TContext, Json> {
    AgentHarnessTool {
        name: "read".to_owned(),
        label: "read".to_owned(),
        description: format!(
            "Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). Images are sent as attachments. For text files, output is truncated to {DEFAULT_MAX_LINES} lines or {}KB (whichever is hit first). Use offset/limit for large files. When you need the full file, continue with offset until complete.",
            DEFAULT_MAX_BYTES / 1024
        ),
        parameters: read_parameters_schema(),
        execution_mode: None,
        prepare_arguments: None,
        execute: Arc::new(ReadToolExecutor { options }),
    }
}

/// Convenience constructor producing an [`crate::types::AgentTool`] bound to one
/// execution environment.
pub fn create_read_tool_with_env(
    env: Arc<dyn ExecutionEnv>,
    options: Option<ReadToolOptions>,
) -> crate::types::AgentTool {
    use crate::harness::tools::tool_context::ExecutionToolContext;
    create_read_tool::<ExecutionToolContext>(options)
        .into_agent_tool(ExecutionToolContext::new(env))
}

/// Text/image content helpers re-exported for convenience.
pub use step_harness_providers::types::{
    ImageContent as ReadImageContent, TextContent as ReadTextContent,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_numeric_offsets_and_limits() {
        let input: ReadToolInput =
            serde_json::from_value(serde_json::json!({"path": "f", "offset": 3, "limit": 10}))
                .unwrap();
        assert_eq!(input.offset, Some(3));
        assert_eq!(input.limit, Some(10));

        let input: ReadToolInput =
            serde_json::from_value(serde_json::json!({"path": "f"})).unwrap();
        assert_eq!(input.offset, None);
    }

    #[test]
    fn rejects_fractional_negative_and_overflowing_line_numbers() {
        for value in [
            serde_json::json!(1.5),
            serde_json::json!(-1),
            serde_json::json!(1e100),
        ] {
            assert!(
                serde_json::from_value::<ReadToolInput>(
                    serde_json::json!({"path": "f", "offset": value})
                )
                .is_err()
            );
        }
    }
}
