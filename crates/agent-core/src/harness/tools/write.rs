//! The `write` tool: full-file writes with parent directory creation.
//!
//! Ported from `@step-harness/agent-core` `src\harness\tools\write.ts`.

use std::sync::Arc;

use serde::Deserialize;
use serde_json::Value as Json;

use crate::harness::tools::mutation_queue::with_file_mutation_queue;
use crate::harness::tools::path_utils::resolve_tool_path;
use crate::harness::tools::tool_context::ToolEnvContext;
use crate::harness::types::{AbortSignal, AgentHarnessTool, ContextToolExecutor, ExecutionEnv};
use crate::types::{AgentToolResult, ToolError, ToolUpdateCallback};

/// JSON Schema for the write tool parameters.
pub fn write_parameters_schema() -> Json {
    serde_json::json!({
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to write (relative or absolute)"
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file"
            }
        },
        "required": ["path", "content"]
    })
}

/// Validated write tool arguments.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WriteToolInput {
    pub path: String,
    pub content: String,
}

struct WriteToolExecutor;

#[async_trait::async_trait]
impl<TContext: ToolEnvContext + Send + Sync + 'static> ContextToolExecutor<TContext, Json>
    for WriteToolExecutor
{
    async fn execute(
        &self,
        _tool_call_id: &str,
        params: &serde_json::Map<String, Json>,
        signal: Option<&AbortSignal>,
        _on_update: Option<&ToolUpdateCallback<Json>>,
        context: &TContext,
    ) -> Result<AgentToolResult<Json>, ToolError> {
        let input: WriteToolInput =
            serde_json::from_value(Json::Object(params.clone())).map_err(ToolError::from)?;
        let env = context.env().clone();

        let absolute_path = resolve_tool_path(env.as_ref(), &input.path, signal).await?;
        let content = input.content.clone();
        let env_for_mutation = env.clone();
        let queue_path = absolute_path.clone();
        with_file_mutation_queue(&env, &queue_path, move || {
            let env = env_for_mutation;
            let content = content;
            let path = absolute_path;
            async move {
                if signal.is_some_and(|signal| signal.is_cancelled()) {
                    return Err(crate::types::tool_error("Operation aborted"));
                }
                env.write_file(
                    &path,
                    crate::harness::types::FileContent::Text(content.clone()),
                    signal,
                )
                .await?;
                if signal.is_some_and(|signal| signal.is_cancelled()) {
                    return Err(crate::types::tool_error("Operation aborted"));
                }
                Ok(())
            }
        })
        .await?;

        Ok(AgentToolResult::text(
            format!(
                "Successfully wrote {} bytes to {}",
                input.content.len(),
                input.path
            ),
            Json::Null,
        ))
    }
}

/// Create the `write` tool.
pub fn create_write_tool<TContext: ToolEnvContext + Send + Sync + 'static>()
-> AgentHarnessTool<TContext, Json> {
    AgentHarnessTool {
        name: "write".to_owned(),
        label: "write".to_owned(),
        description: "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.".to_owned(),
        parameters: write_parameters_schema(),
        execution_mode: None,
        prepare_arguments: None,
        execute: Arc::new(WriteToolExecutor),
    }
}

/// Convenience constructor producing an [`crate::types::AgentTool`] bound to one
/// execution environment.
pub fn create_write_tool_with_env(env: Arc<dyn ExecutionEnv>) -> crate::types::AgentTool {
    use crate::harness::tools::tool_context::ExecutionToolContext;
    create_write_tool::<ExecutionToolContext>().into_agent_tool(ExecutionToolContext::new(env))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_write_input() {
        let input: WriteToolInput =
            serde_json::from_value(serde_json::json!({"path": "f.txt", "content": "hello"}))
                .unwrap();
        assert_eq!(input.path, "f.txt");
        assert_eq!(input.content, "hello");
    }
}
