//! Core agent types: messages, tools, events, loop configuration, and hooks.
//!
//! Ported from `@step-harness/agent-core` `src/types.ts`. TypeScript's
//! declaration-merging extension point for custom messages becomes the
//! [`AgentMessage`] enum; async callbacks become trait objects.

use std::collections::HashSet;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use step_harness_providers::types::{
    AssistantContent, AssistantMessage, Context, ImageContent, Message, Model, ModelThinkingLevel,
    SimpleStreamOptions, TextContent, ThinkingBudgets, Tool, ToolCall, ToolResultMessage,
    Transport, Usage, UserContentBlock, UserMessage,
};
use tokio_util::sync::CancellationToken;

use crate::harness::messages::{
    BashExecutionMessage, BranchSummaryMessage, CompactionSummaryMessage, CustomMessage,
};
use crate::harness::types::AbortSignal;

/// A JSON value used for opaque tool data.
pub type Json = serde_json::Value;

/// Error type returned by tool executors. Tools throw on failure in the
/// TypeScript source; Rust tools return `Err` and the loop converts it into an
/// error tool result.
pub type ToolError = Box<dyn std::error::Error + Send + Sync>;

/// A plain tool error carrying only a message.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct ToolErrorMessage(pub String);

/// Convenience constructor for message-only tool errors.
pub fn tool_error(message: impl Into<String>) -> ToolError {
    Box::new(ToolErrorMessage(message.into()))
}

/// How tool calls from a single assistant message are executed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ToolExecutionMode {
    /// Each tool call is prepared, executed, and finalized before the next starts.
    Sequential,
    /// Tool calls are prepared sequentially, then allowed tools execute concurrently.
    #[default]
    Parallel,
}

/// Controls how many queued user messages are injected at a queue drain point.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum QueueMode {
    /// Drain and inject only the oldest queued message.
    #[default]
    OneAtATime,
    /// Drain and inject every queued message at that point.
    All,
}

/// A conversation message: the LLM message union plus harness custom roles.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "camelCase")]
pub enum AgentMessage {
    User(UserMessage),
    Assistant(Box<AssistantMessage>),
    ToolResult(Box<ToolResultMessage>),
    BashExecution(BashExecutionMessage),
    Custom(Box<CustomMessage>),
    BranchSummary(BranchSummaryMessage),
    CompactionSummary(CompactionSummaryMessage),
}

/// Structural equality over the wire representation.
///
/// The provider `AssistantMessage` and `ToolResultMessage` types do not derive
/// `PartialEq` (their diagnostics payload is an open JSON map), so equality is
/// defined on the serialized form, which covers every field.
impl PartialEq for AgentMessage {
    fn eq(&self, other: &Self) -> bool {
        if std::mem::discriminant(self) != std::mem::discriminant(other) {
            return false;
        }
        match (self, other) {
            (AgentMessage::BashExecution(left), AgentMessage::BashExecution(right)) => {
                left == right
            }
            (AgentMessage::BranchSummary(left), AgentMessage::BranchSummary(right)) => {
                left == right
            }
            (AgentMessage::CompactionSummary(left), AgentMessage::CompactionSummary(right)) => {
                left == right
            }
            _ => serde_json::to_value(self).ok() == serde_json::to_value(other).ok(),
        }
    }
}

impl AgentMessage {
    pub fn user(
        content: impl Into<step_harness_providers::types::UserContent>,
        timestamp: u64,
    ) -> Self {
        AgentMessage::User(UserMessage {
            content: content.into(),
            timestamp,
        })
    }

    pub fn assistant(message: AssistantMessage) -> Self {
        AgentMessage::Assistant(Box::new(message))
    }

    pub fn tool_result(message: ToolResultMessage) -> Self {
        AgentMessage::ToolResult(Box::new(message))
    }

    pub fn as_user(&self) -> Option<&UserMessage> {
        match self {
            AgentMessage::User(message) => Some(message),
            _ => None,
        }
    }

    pub fn as_assistant(&self) -> Option<&AssistantMessage> {
        match self {
            AgentMessage::Assistant(message) => Some(message.as_ref()),
            _ => None,
        }
    }

    pub fn as_tool_result(&self) -> Option<&ToolResultMessage> {
        match self {
            AgentMessage::ToolResult(message) => Some(message.as_ref()),
            _ => None,
        }
    }

    pub fn as_bash_execution(&self) -> Option<&BashExecutionMessage> {
        match self {
            AgentMessage::BashExecution(message) => Some(message),
            _ => None,
        }
    }

    pub fn as_custom(&self) -> Option<&CustomMessage> {
        match self {
            AgentMessage::Custom(message) => Some(message.as_ref()),
            _ => None,
        }
    }

    pub fn as_branch_summary(&self) -> Option<&BranchSummaryMessage> {
        match self {
            AgentMessage::BranchSummary(message) => Some(message),
            _ => None,
        }
    }

    pub fn as_compaction_summary(&self) -> Option<&CompactionSummaryMessage> {
        match self {
            AgentMessage::CompactionSummary(message) => Some(message),
            _ => None,
        }
    }

    /// Wire role name of this message.
    pub fn role(&self) -> &'static str {
        match self {
            AgentMessage::User(_) => "user",
            AgentMessage::Assistant(_) => "assistant",
            AgentMessage::ToolResult(_) => "toolResult",
            AgentMessage::BashExecution(_) => "bashExecution",
            AgentMessage::Custom(_) => "custom",
            AgentMessage::BranchSummary(_) => "branchSummary",
            AgentMessage::CompactionSummary(_) => "compactionSummary",
        }
    }

    pub fn timestamp(&self) -> u64 {
        match self {
            AgentMessage::User(message) => message.timestamp,
            AgentMessage::Assistant(message) => message.timestamp,
            AgentMessage::ToolResult(message) => message.timestamp,
            AgentMessage::BashExecution(message) => message.timestamp,
            AgentMessage::Custom(message) => message.timestamp,
            AgentMessage::BranchSummary(message) => message.timestamp,
            AgentMessage::CompactionSummary(message) => message.timestamp,
        }
    }
}

impl From<Message> for AgentMessage {
    fn from(message: Message) -> Self {
        match message {
            Message::User(user) => AgentMessage::User(user),
            Message::Assistant(assistant) => AgentMessage::Assistant(assistant),
            Message::ToolResult(result) => AgentMessage::ToolResult(result),
        }
    }
}

impl From<AgentMessage> for Option<Message> {
    fn from(message: AgentMessage) -> Self {
        match message {
            AgentMessage::User(user) => Some(Message::User(user)),
            AgentMessage::Assistant(assistant) => Some(Message::Assistant(assistant)),
            AgentMessage::ToolResult(result) => Some(Message::ToolResult(result)),
            _ => None,
        }
    }
}

/// Final or partial result produced by a tool.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentToolResult<TDetails = Json> {
    /// Text or image content returned to the model.
    pub content: Vec<UserContentBlock>,
    /// Arbitrary structured details for logs or UI rendering.
    pub details: TDetails,
    /// Usage from the final tool execution itself, if available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Names of tools introduced by this result, available from this transcript point onward.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub added_tool_names: Option<Vec<String>>,
    /// Hint that the agent should stop after the current tool batch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminate: Option<bool>,
}

impl<TDetails> AgentToolResult<TDetails> {
    /// An error result carrying a single text block; details are the default value.
    pub fn error(message: impl Into<String>) -> Self
    where
        TDetails: Default,
    {
        AgentToolResult {
            content: vec![UserContentBlock::Text(TextContent {
                text: message.into(),
                text_signature: None,
            })],
            details: TDetails::default(),
            usage: None,
            added_tool_names: None,
            terminate: None,
        }
    }

    pub fn text(message: impl Into<String>, details: TDetails) -> Self {
        AgentToolResult {
            content: vec![UserContentBlock::Text(TextContent {
                text: message.into(),
                text_signature: None,
            })],
            details,
            usage: None,
            added_tool_names: None,
            terminate: None,
        }
    }
}

/// Callback used by tools to stream partial execution updates.
pub type ToolUpdateCallback<TDetails = Json> = Arc<dyn Fn(AgentToolResult<TDetails>) + Send + Sync>;

/// Compatibility shim for raw tool-call arguments, run before schema validation.
pub type PrepareArgumentsFn = Arc<dyn Fn(Json) -> Json + Send + Sync>;

/// Executor for an [`AgentTool`].
#[async_trait::async_trait]
pub trait ToolExecutor<TDetails = Json>: Send + Sync {
    async fn execute(
        &self,
        tool_call_id: &str,
        params: &serde_json::Map<String, Json>,
        signal: Option<&AbortSignal>,
        on_update: Option<&ToolUpdateCallback<TDetails>>,
    ) -> Result<AgentToolResult<TDetails>, ToolError>;
}

/// Tool definition used by the agent runtime.
#[derive(Clone)]
pub struct AgentTool<TDetails = Json> {
    pub name: String,
    /// Human-readable label for UI display.
    pub label: String,
    pub description: String,
    /// JSON Schema for the tool parameters.
    pub parameters: Json,
    /// Per-tool execution mode override.
    pub execution_mode: Option<ToolExecutionMode>,
    pub prepare_arguments: Option<PrepareArgumentsFn>,
    pub executor: Arc<dyn ToolExecutor<TDetails>>,
}

impl<TDetails> AgentTool<TDetails> {
    /// Project onto the provider-level [`Tool`] for LLM requests.
    pub fn to_tool(&self) -> Tool {
        Tool {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: self.parameters.clone(),
            constrained_sampling: None,
        }
    }
}

impl<TDetails> std::fmt::Debug for AgentTool<TDetails> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentTool")
            .field("name", &self.name)
            .field("label", &self.label)
            .field("description", &self.description)
            .field("parameters", &self.parameters)
            .field("execution_mode", &self.execution_mode)
            .finish_non_exhaustive()
    }
}

/// Context snapshot passed into the low-level agent loop.
#[derive(Clone, Debug, Default)]
pub struct AgentContext {
    /// System prompt included with the request.
    pub system_prompt: String,
    /// Transcript visible to the model.
    pub messages: Vec<AgentMessage>,
    /// Tools available for this run.
    pub tools: Option<Vec<AgentTool>>,
}

impl AgentContext {
    /// Build the provider-level [`Context`] from this snapshot.
    pub fn to_llm_context(&self, messages: Vec<Message>) -> Context {
        Context {
            system_prompt: Some(self.system_prompt.clone()),
            messages,
            tools: self
                .tools
                .as_ref()
                .map(|tools| tools.iter().map(AgentTool::to_tool).collect()),
        }
    }
}

/// Public agent state.
#[derive(Clone, Debug)]
pub struct AgentState {
    /// System prompt sent with each model request.
    pub system_prompt: String,
    /// Active model used for future turns.
    pub model: Model,
    /// Requested reasoning level for future turns.
    pub thinking_level: ModelThinkingLevel,
    /// Available tools.
    pub tools: Vec<AgentTool>,
    /// Conversation transcript.
    pub messages: Vec<AgentMessage>,
    /// True while the agent is processing a prompt or continuation, including
    /// awaited `agent_end` listeners.
    pub is_streaming: bool,
    /// Partial assistant message for the current streamed response, if any.
    pub streaming_message: Option<AgentMessage>,
    /// Tool call ids currently executing.
    pub pending_tool_calls: HashSet<String>,
    /// Error message from the most recent failed or aborted assistant turn.
    pub error_message: Option<String>,
}

impl Default for AgentState {
    fn default() -> Self {
        AgentState {
            system_prompt: String::new(),
            model: default_model(),
            thinking_level: ModelThinkingLevel::Off,
            tools: Vec::new(),
            messages: Vec::new(),
            is_streaming: false,
            streaming_message: None,
            pending_tool_calls: HashSet::new(),
            error_message: None,
        }
    }
}

fn default_model() -> Model {
    Model {
        id: "unknown".to_owned(),
        name: "unknown".to_owned(),
        api: step_harness_providers::types::Api::Custom("unknown".into()),
        provider: step_harness_providers::types::ProviderId::new("unknown"),
        base_url: String::new(),
        reasoning: false,
        thinking_level_map: None,
        input: Vec::new(),
        cost: Default::default(),
        context_window: 0,
        max_tokens: 0,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// Initial (partial) agent state accepted by [`crate::Agent::new`].
#[derive(Clone, Debug, Default)]
pub struct InitialAgentState {
    pub system_prompt: Option<String>,
    pub model: Option<Model>,
    pub thinking_level: Option<ModelThinkingLevel>,
    pub tools: Option<Vec<AgentTool>>,
    pub messages: Option<Vec<AgentMessage>>,
}

/// Events emitted by the agent for UI updates.
///
/// `agent_end` is the last event of a run, but awaited `agent_end` listeners
/// still count toward run settlement. Payload-heavy variants are boxed to keep
/// the enum small.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum AgentEvent {
    AgentStart,
    AgentEnd {
        messages: Vec<AgentMessage>,
    },
    TurnStart,
    TurnEnd {
        message: Box<AgentMessage>,
        tool_results: Vec<ToolResultMessage>,
    },
    MessageStart {
        message: Box<AgentMessage>,
    },
    /// Assistant only: streaming chunk with the accompanying event.
    MessageUpdate {
        message: Box<AgentMessage>,
        assistant_message_event: Box<step_harness_providers::event::AssistantMessageEvent>,
    },
    MessageEnd {
        message: Box<AgentMessage>,
    },
    ToolExecutionStart {
        tool_call_id: String,
        tool_name: String,
        args: Json,
    },
    ToolExecutionUpdate {
        tool_call_id: String,
        tool_name: String,
        args: Json,
        partial_result: Json,
    },
    ToolExecutionEnd {
        tool_call_id: String,
        tool_name: String,
        result: Json,
        is_error: bool,
    },
}

/// Stream function used by the agent loop.
///
/// Contract: must not panic for request/model/runtime failures; failures are
/// encoded in the returned stream via protocol events and a final assistant
/// message with stop reason `Error` or `Aborted`.
#[async_trait::async_trait]
pub trait StreamFn: Send + Sync {
    async fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
    ) -> step_harness_providers::event::AssistantMessageEventStream;
}

/// Converts agent messages to LLM-compatible messages before each LLM call.
#[async_trait::async_trait]
pub trait ConvertToLlm: Send + Sync {
    async fn convert(&self, messages: &[AgentMessage]) -> Vec<Message>;
}

/// Default conversion: keep user, assistant, and tool-result messages.
pub struct DefaultConvertToLlm;

#[async_trait::async_trait]
impl ConvertToLlm for DefaultConvertToLlm {
    async fn convert(&self, messages: &[AgentMessage]) -> Vec<Message> {
        messages
            .iter()
            .filter_map(|message| match message {
                AgentMessage::User(user) => Some(Message::User(user.clone())),
                AgentMessage::Assistant(assistant) => Some(Message::Assistant(assistant.clone())),
                AgentMessage::ToolResult(result) => Some(Message::ToolResult(result.clone())),
                _ => None,
            })
            .collect()
    }
}

/// Optional transform applied to the context before `ConvertToLlm`.
#[async_trait::async_trait]
pub trait TransformContext: Send + Sync {
    async fn transform(
        &self,
        messages: &[AgentMessage],
        signal: Option<&AbortSignal>,
    ) -> Vec<AgentMessage>;
}

/// Resolves an API key dynamically for each LLM call.
#[async_trait::async_trait]
pub trait GetApiKey: Send + Sync {
    async fn get_api_key(&self, provider: &str) -> Option<String>;
}

/// Called after each turn fully completes and `turn_end` has been emitted.
#[async_trait::async_trait]
pub trait ShouldStopAfterTurn: Send + Sync {
    async fn should_stop_after_turn(
        &self,
        context: &ShouldStopAfterTurnContext,
        signal: Option<&AbortSignal>,
    ) -> bool;
}

/// Context passed to `ShouldStopAfterTurn`.
pub struct ShouldStopAfterTurnContext {
    /// The assistant message that completed the turn.
    pub message: AssistantMessage,
    /// Tool result messages passed to the preceding `turn_end` event.
    pub tool_results: Vec<ToolResultMessage>,
    /// Current agent context after the turn's messages have been appended.
    pub context: AgentContext,
    /// Messages this loop invocation will return if it exits at this point.
    pub new_messages: Vec<AgentMessage>,
}

/// Called after `turn_end` when the loop will continue, immediately before the
/// next turn starts.
#[async_trait::async_trait]
pub trait PrepareNextTurn: Send + Sync {
    async fn prepare_next_turn(
        &self,
        context: &ShouldStopAfterTurnContext,
        signal: Option<&AbortSignal>,
    ) -> Option<AgentLoopTurnUpdate>;
}

/// Signal-only variant of [`PrepareNextTurn`] used by `AgentOptions`.
#[async_trait::async_trait]
pub trait PrepareNextTurnSimple: Send + Sync {
    async fn prepare_next_turn(&self, signal: Option<&AbortSignal>) -> Option<AgentLoopTurnUpdate>;
}

/// Replacement runtime state used by the loop before starting another request.
#[derive(Clone, Debug, Default)]
pub struct AgentLoopTurnUpdate {
    pub context: Option<AgentContext>,
    pub model: Option<Model>,
    pub thinking_level: Option<ModelThinkingLevel>,
}

/// Returns steering messages to inject into the conversation mid-run.
#[async_trait::async_trait]
pub trait GetSteeringMessages: Send + Sync {
    async fn get_steering_messages(&self) -> Vec<AgentMessage>;
}

/// Returns follow-up messages to process after the agent would otherwise stop.
#[async_trait::async_trait]
pub trait GetFollowUpMessages: Send + Sync {
    async fn get_follow_up_messages(&self) -> Vec<AgentMessage>;
}

/// Result returned from `before_tool_call`.
#[derive(Clone, Debug, Default)]
pub struct BeforeToolCallResult {
    /// Prevents the tool from executing; an error tool result is emitted instead.
    pub block: bool,
    /// Text shown in the blocked error result.
    pub reason: Option<String>,
    /// Hint to stop after the current tool batch when this call is blocked.
    pub terminate: Option<bool>,
}

/// Context passed to `before_tool_call`.
pub struct BeforeToolCallContext {
    /// The assistant message that requested the tool call.
    pub assistant_message: AssistantMessage,
    /// The raw tool call block.
    pub tool_call: ToolCall,
    /// Validated tool arguments.
    pub args: serde_json::Map<String, Json>,
    /// Current agent context at preparation time.
    pub context: AgentContext,
}

/// Partial override returned from `after_tool_call`. Omitted fields keep the
/// original executed values; there is no deep merge.
#[derive(Clone, Debug, Default)]
pub struct AfterToolCallResult {
    pub content: Option<Vec<UserContentBlock>>,
    pub details: Option<Json>,
    pub is_error: Option<bool>,
    pub usage: Option<Usage>,
    pub terminate: Option<bool>,
}

/// Context passed to `after_tool_call`.
pub struct AfterToolCallContext {
    pub assistant_message: AssistantMessage,
    pub tool_call: ToolCall,
    pub args: serde_json::Map<String, Json>,
    /// The executed tool result before any overrides.
    pub result: AgentToolResult,
    pub is_error: bool,
    pub context: AgentContext,
}

/// Final content-only transform for every tool outcome.
#[async_trait::async_trait]
pub trait TransformToolResult: Send + Sync {
    async fn transform_tool_result(
        &self,
        content: Vec<UserContentBlock>,
        signal: Option<&AbortSignal>,
    ) -> Result<Vec<UserContentBlock>, ToolError>;
}

/// Configuration for the low-level agent loop.
pub struct AgentLoopConfig {
    pub model: Model,
    /// Bounded resampling for invalid tool argument JSON or tool calls leaked
    /// into plain text. Invalid arguments are never executed.
    /// Default 2; 0 disables.
    pub tool_call_leak_retries: Option<u32>,
    pub convert_to_llm: Arc<dyn ConvertToLlm>,
    pub transform_context: Option<Arc<dyn TransformContext>>,
    pub get_api_key: Option<Arc<dyn GetApiKey>>,
    pub should_stop_after_turn: Option<Arc<dyn ShouldStopAfterTurn>>,
    pub prepare_next_turn: Option<Arc<dyn PrepareNextTurn>>,
    pub get_steering_messages: Option<Arc<dyn GetSteeringMessages>>,
    pub get_follow_up_messages: Option<Arc<dyn GetFollowUpMessages>>,
    /// Tool execution mode; per-tool `execution_mode` overrides it.
    pub tool_execution: ToolExecutionMode,
    pub before_tool_call: Option<Arc<dyn BeforeToolCall>>,
    pub after_tool_call: Option<Arc<dyn AfterToolCall>>,
    pub transform_tool_result: Option<Arc<dyn TransformToolResult>>,
    /// Provider request options forwarded to the stream function.
    pub stream_options: SimpleStreamOptions,
}

impl AgentLoopConfig {
    /// The stream options for one request, with the resolved API key and signal applied.
    pub fn stream_options_for_request(
        &self,
        api_key: Option<String>,
        signal: Option<&CancellationToken>,
    ) -> SimpleStreamOptions {
        let mut options = self.stream_options.clone();
        options.base.base.api_key = api_key.or_else(|| options.base.base.api_key.clone());
        options.base.base.signal = signal.cloned();
        options
    }
}

/// A single tool call content block emitted by an assistant message.
pub type AgentToolCall = ToolCall;

/// Options for constructing an [`crate::Agent`].
pub struct AgentOptions {
    pub initial_state: Option<InitialAgentState>,
    pub convert_to_llm: Option<Arc<dyn ConvertToLlm>>,
    pub transform_context: Option<Arc<dyn TransformContext>>,
    pub stream_fn: Arc<dyn StreamFn>,
    pub get_api_key: Option<Arc<dyn GetApiKey>>,
    pub on_payload: Option<Arc<dyn step_harness_providers::types::PayloadObserver>>,
    pub on_response: Option<Arc<dyn step_harness_providers::types::ResponseObserver>>,
    pub before_tool_call: Option<Arc<dyn BeforeToolCall>>,
    pub after_tool_call: Option<Arc<dyn AfterToolCall>>,
    pub transform_tool_result: Option<Arc<dyn TransformToolResult>>,
    pub should_stop_after_turn: Option<Arc<dyn ShouldStopAfterTurn>>,
    pub prepare_next_turn: Option<Arc<dyn PrepareNextTurnSimple>>,
    pub prepare_next_turn_with_context: Option<Arc<dyn PrepareNextTurn>>,
    pub steering_mode: QueueMode,
    pub follow_up_mode: QueueMode,
    pub session_id: Option<String>,
    pub thinking_budgets: Option<ThinkingBudgets>,
    pub transport: Transport,
    pub max_retry_delay_ms: Option<u64>,
    pub tool_execution: ToolExecutionMode,
}

/// `beforeToolCall` hook: called after argument validation, can block execution.
#[async_trait::async_trait]
pub trait BeforeToolCall: Send + Sync {
    async fn before_tool_call(
        &self,
        context: &BeforeToolCallContext,
        signal: Option<&AbortSignal>,
    ) -> Option<BeforeToolCallResult>;
}

/// `afterToolCall` hook: called after execution, before terminal tool events.
#[async_trait::async_trait]
pub trait AfterToolCall: Send + Sync {
    async fn after_tool_call(
        &self,
        context: &AfterToolCallContext,
        signal: Option<&AbortSignal>,
    ) -> Option<AfterToolCallResult>;
}

/// Stream function that immediately fails; used in tests where no stream runs.
pub struct NoopStreamFn;

#[async_trait::async_trait]
impl StreamFn for NoopStreamFn {
    async fn stream(
        &self,
        model: &Model,
        _context: &Context,
        _options: &SimpleStreamOptions,
    ) -> step_harness_providers::event::AssistantMessageEventStream {
        step_harness_providers::event::lazy_stream(model, || async {
            Err::<step_harness_providers::event::AssistantMessageEventStream, _>("no stream")
        })
    }
}

/// Sink for agent events produced by the loop.
#[async_trait::async_trait]
pub trait AgentEventSink: Send + Sync {
    async fn emit(&self, event: AgentEvent);
}

/// Text content helper.
pub fn text_content(text: impl Into<String>) -> UserContentBlock {
    UserContentBlock::Text(TextContent {
        text: text.into(),
        text_signature: None,
    })
}

/// Image content helper from base64 data and a MIME type.
pub fn image_content(data: impl Into<String>, mime_type: impl Into<String>) -> UserContentBlock {
    UserContentBlock::Image(ImageContent {
        data: data.into(),
        mime_type: mime_type.into(),
    })
}

/// Collect the text of all text blocks in assistant content.
pub fn assistant_text(content: &[AssistantContent]) -> String {
    content
        .iter()
        .filter_map(|block| block.text())
        .map(|text| text.text.as_str())
        .collect::<Vec<_>>()
        .join("")
}

/// Convert a model thinking level (which includes the explicit "off" state)
/// into the optional provider reasoning parameter.
pub fn thinking_level_to_reasoning(
    level: ModelThinkingLevel,
) -> Option<step_harness_providers::types::ThinkingLevel> {
    use step_harness_providers::types::{ModelThinkingLevel as Level, ThinkingLevel};
    match level {
        Level::Off => None,
        Level::Minimal => Some(ThinkingLevel::Minimal),
        Level::Low => Some(ThinkingLevel::Low),
        Level::Medium => Some(ThinkingLevel::Medium),
        Level::High => Some(ThinkingLevel::High),
        Level::Xhigh => Some(ThinkingLevel::Xhigh),
        Level::Max => Some(ThinkingLevel::Max),
    }
}
