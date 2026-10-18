//! Core types for the unified model system.
//!
//! Field-type policy (mirrors the TypeScript source but with precise numeric types):
//! - token counts, context windows, budgets: `u64`
//! - prices ($/million tokens) and computed costs: `f64`
//! - timestamps: `u64` (Unix milliseconds)
//! - HTTP status codes: `u16`

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::utils::diagnostics::AssistantMessageDiagnostic;

/// The wire dialect / API implementation identifier. Known values are typed
/// variants; custom OpenAI-compatible servers use [`Api::Custom`].
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(from = "String", into = "String")]
pub enum Api {
    AnthropicMessages,
    OpenaiCompletions,
    OpenaiResponses,
    Custom(Arc<str>),
}

impl Api {
    pub const ANTHROPIC_MESSAGES: &'static str = "anthropic-messages";
    pub const OPENAI_COMPLETIONS: &'static str = "openai-completions";
    pub const OPENAI_RESPONSES: &'static str = "openai-responses";

    pub fn as_str(&self) -> &str {
        match self {
            Api::AnthropicMessages => Self::ANTHROPIC_MESSAGES,
            Api::OpenaiCompletions => Self::OPENAI_COMPLETIONS,
            Api::OpenaiResponses => Self::OPENAI_RESPONSES,
            Api::Custom(value) => value,
        }
    }
}

impl From<String> for Api {
    fn from(value: String) -> Self {
        value.as_str().into()
    }
}

impl From<&str> for Api {
    fn from(value: &str) -> Self {
        match value {
            Self::ANTHROPIC_MESSAGES => Api::AnthropicMessages,
            Self::OPENAI_COMPLETIONS => Api::OpenaiCompletions,
            Self::OPENAI_RESPONSES => Api::OpenaiResponses,
            other => Api::Custom(Arc::from(other)),
        }
    }
}

impl From<Api> for String {
    fn from(value: Api) -> Self {
        value.as_str().to_owned()
    }
}

impl fmt::Display for Api {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Provider identity. A cheap clonable string newtype; well-known ids are
/// exposed as associated constants.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default)]
#[serde(from = "String", into = "String")]
pub struct ProviderId(Arc<str>);

impl ProviderId {
    pub const ANTHROPIC: &'static str = "anthropic";
    pub const OPENAI: &'static str = "openai";
    pub const GITHUB_COPILOT: &'static str = "github-copilot";
    pub const AMAZON_BEDROCK: &'static str = "amazon-bedrock";
    pub const GOOGLE_VERTEX: &'static str = "google-vertex";
    pub const OPENROUTER: &'static str = "openrouter";

    pub fn new(id: impl AsRef<str>) -> Self {
        Self(Arc::from(id.as_ref()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for ProviderId {
    fn from(value: &str) -> Self {
        ProviderId::new(value)
    }
}

impl From<String> for ProviderId {
    fn from(value: String) -> Self {
        ProviderId::new(value)
    }
}

/// Lets `HashMap<ProviderId, _>` be looked up with `&str` keys.
impl std::borrow::Borrow<str> for ProviderId {
    fn borrow(&self) -> &str {
        &self.0
    }
}

impl From<ProviderId> for String {
    fn from(value: ProviderId) -> Self {
        value.0.to_string()
    }
}

impl fmt::Display for ProviderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ToolChoice {
    Auto,
    None,
}

/// Reasoning effort levels understood across providers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ThinkingLevel {
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

/// A thinking level including the explicit "off" state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ModelThinkingLevel {
    Off,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl ModelThinkingLevel {
    pub const ORDERED: [ModelThinkingLevel; 7] = [
        ModelThinkingLevel::Off,
        ModelThinkingLevel::Minimal,
        ModelThinkingLevel::Low,
        ModelThinkingLevel::Medium,
        ModelThinkingLevel::High,
        ModelThinkingLevel::Xhigh,
        ModelThinkingLevel::Max,
    ];

    /// Wire name of this level (matches the TypeScript string union).
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelThinkingLevel::Off => "off",
            ModelThinkingLevel::Minimal => "minimal",
            ModelThinkingLevel::Low => "low",
            ModelThinkingLevel::Medium => "medium",
            ModelThinkingLevel::High => "high",
            ModelThinkingLevel::Xhigh => "xhigh",
            ModelThinkingLevel::Max => "max",
        }
    }
}

impl fmt::Display for ModelThinkingLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Maps pi thinking levels to provider/model-specific values. Missing keys use
/// provider defaults; `None` marks a level as unsupported.
pub type ThinkingLevelMap = HashMap<ModelThinkingLevel, Option<String>>;

/// A `chat_template_kwargs` value; `$var` entries are substituted with
/// provider-controlled thinking values by the completions adapter.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatTemplateKwargValue {
    Var {
        #[serde(rename = "$var")]
        var: ChatTemplateVar,
        #[serde(rename = "omitWhenOff", skip_serializing_if = "Option::is_none")]
        omit_when_off: Option<bool>,
    },
    Literal(serde_json::Value),
}

/// Wire values are dotted (`thinking.enabled`), not snake_case.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChatTemplateVar {
    #[serde(rename = "thinking.enabled")]
    ThinkingEnabled,
    #[serde(rename = "thinking.effort")]
    ThinkingEffort,
    #[serde(rename = "thinking.budget")]
    ThinkingBudget,
}

impl ChatTemplateVar {
    pub fn as_str(&self) -> &'static str {
        match self {
            ChatTemplateVar::ThinkingEnabled => "thinking.enabled",
            ChatTemplateVar::ThinkingEffort => "thinking.effort",
            ChatTemplateVar::ThinkingBudget => "thinking.budget",
        }
    }
}

impl ChatTemplateKwargValue {
    pub fn as_literal(&self) -> Option<&serde_json::Value> {
        match self {
            ChatTemplateKwargValue::Literal(value) => Some(value),
            _ => None,
        }
    }

    pub fn var(&self) -> Option<(ChatTemplateVar, bool)> {
        match self {
            ChatTemplateKwargValue::Var { var, omit_when_off } => {
                Some((*var, omit_when_off.unwrap_or(false)))
            }
            _ => None,
        }
    }
}

/// Top-level request field used to cap reasoning tokens on OpenAI-compatible servers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThinkingTokenBudgetField {
    #[serde(rename = "thinking_token_budget")]
    Vllm,
    #[serde(rename = "thinking_budget")]
    Qwen,
    #[serde(rename = "thinking_budget_tokens")]
    LlamaCpp,
}

/// Token budgets for each thinking level (token-based providers only).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ThinkingBudgets {
    pub minimal: Option<u64>,
    pub low: Option<u64>,
    pub medium: Option<u64>,
    pub high: Option<u64>,
}

/// Prompt cache retention preference. Providers map this to their supported values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CacheRetention {
    #[default]
    Short,
    None,
    Long,
}

/// Preferred transport for providers that support multiple transports.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Transport {
    Sse,
    Websocket,
    WebsocketCached,
    Auto,
}

/// Session-affinity header format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SessionAffinityFormat {
    Openai,
    #[serde(rename = "openai-nosession")]
    OpenaiNosession,
    Openrouter,
}

/// Provider-scoped environment overrides. Values take precedence over process env.
pub type ProviderEnv = HashMap<String, String>;

/// Extra HTTP headers for provider requests. `None` values suppress a
/// provider/API default header with the same name.
pub type ProviderHeaders = HashMap<String, Option<String>>;

/// Summary of a received HTTP response, surfaced to response observers.
#[derive(Clone, Debug)]
pub struct ProviderResponse {
    pub status: u16,
    pub headers: HashMap<String, String>,
}

/// A JSON value used for opaque provider data.
pub type Json = serde_json::Value;

/// Async observer for inspecting or replacing provider payloads before sending.
/// Return `Some` to replace the payload, `None` to keep it unchanged.
#[async_trait::async_trait]
pub trait PayloadObserver: Send + Sync {
    async fn on_payload(&self, payload: Json, model: &Model) -> Option<Json>;
}

/// Async observer invoked after an HTTP response is received.
#[async_trait::async_trait]
pub trait ResponseObserver: Send + Sync {
    async fn on_response(&self, response: &ProviderResponse, model: &Model);
}

/// Transforms fully assembled model/auth/request headers before provider
/// dispatch (the `Models`-level `transformHeaders` hook).
#[async_trait::async_trait]
pub trait HeaderTransformer: Send + Sync {
    async fn transform(&self, headers: ProviderHeaders) -> ProviderHeaders;
}

/// Authentication, HTTP transport, and lifecycle callbacks shared by provider requests.
#[derive(Clone, Default)]
pub struct ProviderRequestOptions {
    /// Optional host-owned request metrics. Contains no prompt or credential data.
    pub request_metrics: Option<std::sync::Arc<crate::request_metrics::RequestTracker>>,
    /// Cooperative cancellation for the whole logical request.
    pub signal: Option<CancellationTokenRef>,
    pub api_key: Option<String>,
    /// Optional HTTP client override (custom proxy, TLS, or instrumentation).
    pub http_client: Option<reqwest::Client>,
    /// Provider-scoped environment values. These take precedence over process env
    /// for provider configuration such as regional settings and proxy variables.
    pub env: Option<ProviderEnv>,
    /// Optional custom HTTP headers merged with provider defaults; caller values
    /// override default headers. A `None` value suppresses a default header.
    pub headers: Option<ProviderHeaders>,
    /// HTTP request timeout in milliseconds.
    pub timeout_ms: Option<u64>,
    /// Maximum retry attempts for providers that support client-side retries.
    pub max_retries: Option<u32>,
    /// Maximum delay in milliseconds to wait when the server requests a long
    /// retry delay. Default 60000; 0 disables the cap.
    pub max_retry_delay_ms: Option<u64>,
}

pub use tokio_util::sync::CancellationToken;
/// Shared cancellation token handle. Cloning a [`CancellationToken`] is cheap
/// and produces another handle to the same cancellation source.
pub type CancellationTokenRef = CancellationToken;

#[derive(Debug, thiserror::Error)]
#[error("request aborted")]
pub struct AbortedError;

/// Result of an operation raced against a cancellation token.
pub type AbortableResult<T> = Result<T, AbortedError>;

impl ProviderRequestOptions {
    /// The cancellation token, or a never-cancelled placeholder.
    pub fn token(&self) -> CancellationToken {
        self.signal.clone().unwrap_or_default()
    }
}

impl fmt::Debug for ProviderRequestOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderRequestOptions")
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field(
                "env_names",
                &self
                    .env
                    .as_ref()
                    .map(|values| values.keys().collect::<Vec<_>>()),
            )
            .field(
                "header_names",
                &self
                    .headers
                    .as_ref()
                    .map(|values| values.keys().collect::<Vec<_>>()),
            )
            .field("timeout_ms", &self.timeout_ms)
            .field("max_retries", &self.max_retries)
            .field("max_retry_delay_ms", &self.max_retry_delay_ms)
            .finish_non_exhaustive()
    }
}

/// Full snapshots preserve the legacy event contract. Metadata-only snapshots
/// avoid repeatedly copying growing content; consumers accumulate event deltas.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EventSnapshotMode {
    #[default]
    Full,
    MetadataOnly,
}

/// Request options for streaming calls. Extends [`ProviderRequestOptions`] with
/// sampling and caching controls.
#[derive(Clone, Default)]
pub struct StreamOptions {
    pub base: ProviderRequestOptions,
    pub event_snapshot_mode: EventSnapshotMode,
    pub temperature: Option<f64>,
    /// Arbitrary sampling parameters merged into the request body after the
    /// named request fields, so keys here override them. Only applied by
    /// OpenAI-compatible adapters.
    pub sampling_params: Option<serde_json::Map<String, Json>>,
    pub max_tokens: Option<u64>,
    /// Preferred transport for providers that support multiple transports.
    pub transport: Option<Transport>,
    /// Prompt cache retention preference. Default: short.
    pub cache_retention: Option<CacheRetention>,
    /// Optional session identifier for providers that support session-based caching.
    pub session_id: Option<String>,
    /// WebSocket connect timeout in milliseconds (connection/open handshake only).
    pub websocket_connect_timeout_ms: Option<u64>,
    /// Optional metadata included in API requests; providers extract what they know.
    pub metadata: Option<serde_json::Map<String, Json>>,
    /// Observer for inspecting or replacing provider payloads before sending.
    pub on_payload: Option<Arc<dyn PayloadObserver>>,
    /// Observer invoked after an HTTP response is received.
    pub on_response: Option<Arc<dyn ResponseObserver>>,
    /// Final header transform applied before provider dispatch.
    pub transform_headers: Option<Arc<dyn HeaderTransformer>>,
}

impl fmt::Debug for StreamOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamOptions")
            .field("base", &self.base)
            .field("event_snapshot_mode", &self.event_snapshot_mode)
            .field("temperature", &self.temperature)
            .field("sampling_params", &self.sampling_params)
            .field("max_tokens", &self.max_tokens)
            .field("transport", &self.transport)
            .field("cache_retention", &self.cache_retention)
            .field("session_id", &self.session_id)
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

/// Deferred-response polling options.
pub type DeferredFetchOptions = ProviderRequestOptions;

/// Deferred-response window requested from capable providers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeferredWindow {
    #[serde(rename = "15m")]
    FifteenMinutes,
    #[serde(rename = "1h")]
    OneHour,
    #[serde(rename = "24h")]
    TwentyFourHours,
}

/// Unified options with reasoning passed to `stream_simple()`.
#[derive(Clone, Default)]
pub struct SimpleStreamOptions {
    pub base: StreamOptions,
    /// Provider-neutral tool selection for simple requests.
    pub tool_choice: Option<ToolChoice>,
    pub reasoning: Option<ThinkingLevel>,
    /// Ask a capable provider to return a durable handle and continue asynchronously.
    pub deferred: Option<DeferredRequest>,
    /// Custom token budgets for thinking levels (token-based providers only).
    pub thinking_budgets: Option<ThinkingBudgets>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DeferredRequest {
    Enabled(bool),
    Window(DeferredWindow),
}

impl fmt::Debug for SimpleStreamOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SimpleStreamOptions")
            .field("base", &self.base)
            .field("tool_choice", &self.tool_choice)
            .field("reasoning", &self.reasoning)
            .field("deferred", &self.deferred)
            .field("thinking_budgets", &self.thinking_budgets)
            .finish()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextContent {
    pub text: String,
    /// e.g. OpenAI Responses message metadata (legacy id or `TextSignatureV1` JSON).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_signature: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingContent {
    pub thinking: String,
    /// Provider-specific opaque reasoning replay data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_signature: Option<String>,
    /// True when the thinking content was redacted by safety filters. The opaque
    /// encrypted payload lives in `thinking_signature`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub redacted: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageContent {
    /// Base64-encoded image data.
    pub data: String,
    pub mime_type: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Map<String, Json>,
    /// Google-specific: opaque signature for reusing thought context.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
    /// OpenAI Responses namespace for calls to dynamically loaded tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
}

/// Content blocks of an assistant message.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum AssistantContent {
    Text(TextContent),
    Thinking(ThinkingContent),
    ToolCall(ToolCall),
}

impl AssistantContent {
    pub fn text(&self) -> Option<&TextContent> {
        match self {
            AssistantContent::Text(content) => Some(content),
            _ => None,
        }
    }

    pub fn thinking(&self) -> Option<&ThinkingContent> {
        match self {
            AssistantContent::Thinking(content) => Some(content),
            _ => None,
        }
    }

    pub fn tool_call(&self) -> Option<&ToolCall> {
        match self {
            AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        }
    }
}

/// Content blocks allowed in user and tool-result messages.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum UserContentBlock {
    Text(TextContent),
    Image(ImageContent),
}

impl UserContentBlock {
    pub fn text(&self) -> Option<&TextContent> {
        match self {
            UserContentBlock::Text(content) => Some(content),
            _ => None,
        }
    }

    pub fn image(&self) -> Option<&ImageContent> {
        match self {
            UserContentBlock::Image(content) => Some(content),
            _ => None,
        }
    }
}

/// User message content: plain text or mixed text/image blocks.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UserContent {
    Text(String),
    Blocks(Vec<UserContentBlock>),
}

impl UserContent {
    pub fn as_text(&self) -> Option<&str> {
        match self {
            UserContent::Text(text) => Some(text),
            _ => None,
        }
    }

    pub fn blocks(&self) -> Option<&[UserContentBlock]> {
        match self {
            UserContent::Blocks(blocks) => Some(blocks),
            _ => None,
        }
    }

    /// Iterate `(text, image)` blocks as a flat block slice; plain text becomes
    /// a single borrowed block via `Cow` is avoided — use [`Self::flatten_blocks`].
    pub fn flatten_blocks(&self) -> Vec<UserContentBlock> {
        match self {
            UserContent::Text(text) => vec![UserContentBlock::Text(TextContent {
                text: text.clone(),
                text_signature: None,
            })],
            UserContent::Blocks(blocks) => blocks.clone(),
        }
    }
}

/// Per-million-token price rates in USD.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCostRates {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
}

/// Request-wide pricing tier: applies when total input usage exceeds the threshold.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCostTier {
    /// Use this tier for requests whose total input usage exceeds this token count.
    pub input_tokens_above: u64,
    #[serde(flatten)]
    pub rates: ModelCostRates,
}

/// Model pricing, optionally with request-wide tiers.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCost {
    #[serde(flatten)]
    pub rates: ModelCostRates,
    /// The highest matching input threshold applies to the full request.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tiers: Vec<ModelCostTier>,
}

/// Token usage reported by providers, with computed cost.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageCost {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
    pub total: f64,
}

/// Token usage reported by providers. Token counts are `u64`; costs are USD `f64`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    /// Subset of `cache_write` written with 1h retention (Anthropic only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_1h: Option<u64>,
    /// Reasoning tokens when the provider reports them; a subset of `output`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<u64>,
    pub total_tokens: u64,
    #[serde(default)]
    pub cost: UsageCost,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StopReason {
    Pending,
    Stop,
    Length,
    ToolUse,
    Error,
    Aborted,
    Deferred,
}

impl StopReason {
    pub fn is_terminal_success(&self) -> bool {
        matches!(
            self,
            StopReason::Stop | StopReason::Length | StopReason::ToolUse | StopReason::Deferred
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UserMessage {
    pub content: UserContent,
    /// Unix timestamp in milliseconds.
    pub timestamp: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantMessage {
    pub content: Vec<AssistantContent>,
    pub api: Api,
    pub provider: ProviderId,
    pub model: String,
    /// Concrete response model when different from the requested one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_model: Option<String>,
    /// Provider-specific response/message identifier when exposed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// Redacted provider/runtime diagnostics for failures and recoveries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<Vec<AssistantMessageDiagnostic>>,
    #[serde(default)]
    pub usage: Usage,
    pub stop_reason: StopReason,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deferred: Option<DeferredHandle>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_stop_reason: Option<String>,
    /// Provider indication of whether the model explicitly ended its turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_turn: Option<bool>,
    /// Unix timestamp in milliseconds.
    pub timestamp: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultMessage {
    pub tool_call_id: String,
    pub tool_name: String,
    /// Text and image blocks.
    pub content: Vec<UserContentBlock>,
    /// Opaque tool execution details.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Json>,
    /// Usage from the tool execution itself; not part of LLM context accounting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Tool names that became available after this result (deferred tool loading).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub added_tool_names: Option<Vec<String>>,
    #[serde(default)]
    pub is_error: bool,
    /// Unix timestamp in milliseconds.
    pub timestamp: u64,
}

/// A conversation message. Assistant and tool-result variants are boxed to
/// keep the enum small.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "camelCase")]
pub enum Message {
    User(UserMessage),
    Assistant(Box<AssistantMessage>),
    ToolResult(Box<ToolResultMessage>),
}

impl Message {
    pub fn user(content: impl Into<UserContent>, timestamp: u64) -> Self {
        Message::User(UserMessage {
            content: content.into(),
            timestamp,
        })
    }

    pub fn as_user(&self) -> Option<&UserMessage> {
        match self {
            Message::User(message) => Some(message),
            _ => None,
        }
    }

    pub fn as_assistant(&self) -> Option<&AssistantMessage> {
        match self {
            Message::Assistant(message) => Some(message.as_ref()),
            _ => None,
        }
    }

    pub fn as_tool_result(&self) -> Option<&ToolResultMessage> {
        match self {
            Message::ToolResult(message) => Some(message.as_ref()),
            _ => None,
        }
    }

    pub fn timestamp(&self) -> u64 {
        match self {
            Message::User(message) => message.timestamp,
            Message::Assistant(message) => message.timestamp,
            Message::ToolResult(message) => message.timestamp,
        }
    }

    /// Construct a boxed tool-result message variant.
    pub fn tool_result(message: ToolResultMessage) -> Self {
        Message::ToolResult(Box::new(message))
    }

    /// Construct a boxed assistant message variant.
    pub fn assistant(message: AssistantMessage) -> Self {
        Message::Assistant(Box::new(message))
    }
}

impl From<String> for UserContent {
    fn from(value: String) -> Self {
        UserContent::Text(value)
    }
}

impl From<&str> for UserContent {
    fn from(value: &str) -> Self {
        UserContent::Text(value.to_owned())
    }
}

impl From<Vec<UserContentBlock>> for UserContent {
    fn from(value: Vec<UserContentBlock>) -> Self {
        UserContent::Blocks(value)
    }
}

/// Durable handle for a provider-side deferred response.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeferredHandle {
    pub provider: String,
    pub model_id: String,
    pub api: String,
    /// Provider token, such as a response id plus row id.
    pub id: String,
    /// Unix timestamp in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub poll_after_ms: Option<u64>,
    /// Provider conversion data required to reconstruct the final message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Json>,
}

/// OpenAI grammar variants for constrained sampling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GrammarFormat {
    #[serde(rename = "openai_lark")]
    OpenaiLark,
    #[serde(rename = "openai_regex")]
    OpenaiRegex,
}

pub type GrammarVariants = HashMap<GrammarFormat, String>;

/// Provider-side constrained sampling configuration for a tool.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConstrainedSamplingConfig {
    /// Maps to the API concept of `strict` json-schema constrained sampling.
    JsonSchema {
        strict: StrictMode,
    },
    Grammar {
        variants: GrammarVariants,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StrictMode {
    Prefer,
    Require,
}

/// Input modalities a model accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum InputModality {
    Text,
    Image,
}

/// A tool exposed to the model. `parameters` is a JSON Schema object
/// (the Rust counterpart of the TypeScript typebox `TSchema`).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: Json,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constrained_sampling: Option<ConstrainedSamplingConfig>,
}

/// A request context: system prompt, conversation, and tools.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Context {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}

/// Compatibility settings for OpenAI-compatible completions APIs. Field names
/// match the TypeScript source; wire compat is only relevant via serde when
/// declarations are deserialized.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct OpenaiCompletionsCompat {
    pub supports_store: Option<bool>,
    pub supports_developer_role: Option<bool>,
    pub supports_reasoning_effort: Option<bool>,
    pub supports_usage_in_streaming: Option<bool>,
    pub supports_finish_reason: Option<bool>,
    pub max_tokens_field: Option<MaxTokensField>,
    pub requires_tool_result_name: Option<bool>,
    pub requires_assistant_after_tool_result: Option<bool>,
    pub requires_thinking_as_text: Option<bool>,
    pub requires_reasoning_content_on_assistant_messages: Option<bool>,
    pub thinking_format: Option<CompletionsThinkingFormat>,
    pub chat_template_kwargs: Option<HashMap<String, ChatTemplateKwargValue>>,
    pub chat_template_args: Option<HashMap<String, ChatTemplateKwargValue>>,
    #[serde(rename = "openRouterRouting")]
    pub open_router_routing: Option<OpenRouterRouting>,
    #[serde(rename = "vercelGatewayRouting")]
    pub vercel_gateway_routing: Option<VercelGatewayRouting>,
    #[serde(rename = "zaiToolStream")]
    pub zai_tool_stream: Option<bool>,
    pub thinking_token_budget_field: Option<ThinkingTokenBudgetField>,
    /// Alias for `thinking_token_budget_field: "thinking_token_budget"` (vLLM).
    pub supports_thinking_token_budget: Option<bool>,
    pub supports_openai_grammar_tools: Option<bool>,
    pub supports_strict_mode: Option<bool>,
    pub cache_control_format: Option<CacheControlFormat>,
    pub send_session_affinity_headers: Option<bool>,
    pub deferred_tools_mode: Option<DeferredToolsMode>,
    pub session_affinity_format: Option<SessionAffinityFormat>,
    pub supports_long_cache_retention: Option<bool>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaxTokensField {
    MaxCompletionTokens,
    MaxTokens,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CompletionsThinkingFormat {
    Openai,
    Openrouter,
    Deepseek,
    Together,
    Baseten,
    Zai,
    Qwen,
    #[serde(rename = "chat-template")]
    ChatTemplate,
    #[serde(rename = "qwen-chat-template")]
    QwenChatTemplate,
    #[serde(rename = "string-thinking")]
    StringThinking,
    #[serde(rename = "ant-ling")]
    AntLing,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CacheControlFormat {
    Anthropic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeferredToolsMode {
    Kimi,
}

/// Compatibility settings for OpenAI Responses APIs.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct OpenaiResponsesCompat {
    pub supports_developer_role: Option<bool>,
    pub session_affinity_format: Option<SessionAffinityFormat>,
    pub supports_long_cache_retention: Option<bool>,
    pub supports_strict_mode: Option<bool>,
    pub supports_openai_grammar_tools: Option<bool>,
    pub supports_additional_tools: Option<bool>,
    pub supports_tool_search: Option<bool>,
    pub supports_explicit_prompt_cache_mode: Option<bool>,
}

/// Compatibility settings for Anthropic Messages-compatible APIs.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct AnthropicMessagesCompat {
    pub supports_eager_tool_input_streaming: Option<bool>,
    pub supports_long_cache_retention: Option<bool>,
    pub send_session_affinity_headers: Option<bool>,
    pub supports_cache_control_on_tools: Option<bool>,
    pub supports_temperature: Option<bool>,
    pub force_adaptive_thinking: Option<bool>,
    pub allow_empty_signature: Option<bool>,
    pub supports_strict_tools: Option<bool>,
    pub allowed_fallback_models: Option<Vec<AnthropicAllowedFallbackModel>>,
    pub supports_tool_references: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnthropicAllowedFallbackModel {
    pub provider: ProviderId,
    pub model: String,
    pub cost: ModelCost,
}

/// OpenRouter routing preferences, sent as the `provider` request field.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OpenRouterRouting {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allow_fallbacks: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub require_parameters: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_collection: Option<OpenRouterDataCollection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zdr: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enforce_distillable_text: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub order: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub only: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ignore: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantizations: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sort: Option<OpenRouterSort>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_price: Option<OpenRouterMaxPrice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preferred_min_throughput: Option<OpenRouterThroughput>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preferred_max_latency: Option<OpenRouterLatency>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenRouterDataCollection {
    Deny,
    Allow,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenRouterSort {
    By(String),
    Structured {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        by: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        partition: Option<String>,
    },
}

/// Price bound: numeric or string, per OpenRouter's accepted forms.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PriceBound {
    Number(f64),
    Text(String),
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OpenRouterMaxPrice {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<PriceBound>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion: Option<PriceBound>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<PriceBound>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio: Option<PriceBound>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request: Option<PriceBound>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenRouterThroughput {
    TokensPerSecond(f64),
    Percentile {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        p50: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        p75: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        p90: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        p99: Option<f64>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenRouterLatency {
    Seconds(f64),
    Percentile {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        p50: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        p75: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        p90: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        p99: Option<f64>,
    },
}

/// Vercel AI Gateway routing preferences.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VercelGatewayRouting {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub only: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub order: Option<Vec<String>>,
}

/// Per-API compat overrides. The TypeScript source types `Model.compat`
/// conditionally on `model.api`; Rust has no conditional field types, so this
/// is the superset struct and the per-API accessors project the relevant
/// subset. Wire-compatible with each per-API compat object in JSON.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct CompatSettings {
    // -- OpenAI completions --
    pub supports_store: Option<bool>,
    pub supports_developer_role: Option<bool>,
    pub supports_reasoning_effort: Option<bool>,
    pub supports_usage_in_streaming: Option<bool>,
    pub supports_finish_reason: Option<bool>,
    pub max_tokens_field: Option<MaxTokensField>,
    pub requires_tool_result_name: Option<bool>,
    pub requires_assistant_after_tool_result: Option<bool>,
    pub requires_thinking_as_text: Option<bool>,
    pub requires_reasoning_content_on_assistant_messages: Option<bool>,
    pub thinking_format: Option<CompletionsThinkingFormat>,
    pub chat_template_kwargs: Option<HashMap<String, ChatTemplateKwargValue>>,
    pub chat_template_args: Option<HashMap<String, ChatTemplateKwargValue>>,
    #[serde(rename = "openRouterRouting")]
    pub open_router_routing: Option<OpenRouterRouting>,
    #[serde(rename = "vercelGatewayRouting")]
    pub vercel_gateway_routing: Option<VercelGatewayRouting>,
    #[serde(rename = "zaiToolStream")]
    pub zai_tool_stream: Option<bool>,
    pub thinking_token_budget_field: Option<ThinkingTokenBudgetField>,
    /// Alias for `thinking_token_budget_field: "thinking_token_budget"` (vLLM).
    pub supports_thinking_token_budget: Option<bool>,
    pub supports_openai_grammar_tools: Option<bool>,
    pub supports_strict_mode: Option<bool>,
    pub cache_control_format: Option<CacheControlFormat>,
    pub send_session_affinity_headers: Option<bool>,
    pub deferred_tools_mode: Option<DeferredToolsMode>,
    pub session_affinity_format: Option<SessionAffinityFormat>,
    pub supports_long_cache_retention: Option<bool>,

    // -- OpenAI responses --
    pub supports_additional_tools: Option<bool>,
    pub supports_tool_search: Option<bool>,
    pub supports_explicit_prompt_cache_mode: Option<bool>,

    // -- Anthropic messages --
    pub supports_eager_tool_input_streaming: Option<bool>,
    pub supports_cache_control_on_tools: Option<bool>,
    pub supports_temperature: Option<bool>,
    pub force_adaptive_thinking: Option<bool>,
    pub allow_empty_signature: Option<bool>,
    pub supports_strict_tools: Option<bool>,
    pub allowed_fallback_models: Option<Vec<AnthropicAllowedFallbackModel>>,
    pub supports_tool_references: Option<bool>,
}

/// The compat object on a model; the per-API compat structs below are
/// projections, not separate wire shapes.
pub type ApiCompat = CompatSettings;

impl CompatSettings {
    /// Project the OpenAI-completions compat fields.
    pub fn openai_completions(&self) -> OpenaiCompletionsCompat {
        OpenaiCompletionsCompat {
            supports_store: self.supports_store,
            supports_developer_role: self.supports_developer_role,
            supports_reasoning_effort: self.supports_reasoning_effort,
            supports_usage_in_streaming: self.supports_usage_in_streaming,
            supports_finish_reason: self.supports_finish_reason,
            max_tokens_field: self.max_tokens_field,
            requires_tool_result_name: self.requires_tool_result_name,
            requires_assistant_after_tool_result: self.requires_assistant_after_tool_result,
            requires_thinking_as_text: self.requires_thinking_as_text,
            requires_reasoning_content_on_assistant_messages: self
                .requires_reasoning_content_on_assistant_messages,
            thinking_format: self.thinking_format,
            chat_template_kwargs: self.chat_template_kwargs.clone(),
            chat_template_args: self.chat_template_args.clone(),
            open_router_routing: self.open_router_routing.clone(),
            vercel_gateway_routing: self.vercel_gateway_routing.clone(),
            zai_tool_stream: self.zai_tool_stream,
            thinking_token_budget_field: self.thinking_token_budget_field,
            supports_thinking_token_budget: self.supports_thinking_token_budget,
            supports_openai_grammar_tools: self.supports_openai_grammar_tools,
            supports_strict_mode: self.supports_strict_mode,
            cache_control_format: self.cache_control_format,
            send_session_affinity_headers: self.send_session_affinity_headers,
            deferred_tools_mode: self.deferred_tools_mode,
            session_affinity_format: self.session_affinity_format,
            supports_long_cache_retention: self.supports_long_cache_retention,
        }
    }

    /// Project the OpenAI-responses compat fields.
    pub fn openai_responses(&self) -> OpenaiResponsesCompat {
        OpenaiResponsesCompat {
            supports_developer_role: self.supports_developer_role,
            session_affinity_format: self.session_affinity_format,
            supports_long_cache_retention: self.supports_long_cache_retention,
            supports_strict_mode: self.supports_strict_mode,
            supports_openai_grammar_tools: self.supports_openai_grammar_tools,
            supports_additional_tools: self.supports_additional_tools,
            supports_tool_search: self.supports_tool_search,
            supports_explicit_prompt_cache_mode: self.supports_explicit_prompt_cache_mode,
        }
    }

    /// Project the Anthropic-messages compat fields.
    pub fn anthropic_messages(&self) -> AnthropicMessagesCompat {
        AnthropicMessagesCompat {
            supports_eager_tool_input_streaming: self.supports_eager_tool_input_streaming,
            supports_long_cache_retention: self.supports_long_cache_retention,
            send_session_affinity_headers: self.send_session_affinity_headers,
            supports_cache_control_on_tools: self.supports_cache_control_on_tools,
            supports_temperature: self.supports_temperature,
            force_adaptive_thinking: self.force_adaptive_thinking,
            allow_empty_signature: self.allow_empty_signature,
            supports_strict_tools: self.supports_strict_tools,
            allowed_fallback_models: self.allowed_fallback_models.clone(),
            supports_tool_references: self.supports_tool_references,
        }
    }
}

/// A model in the unified model system.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Model {
    pub id: String,
    pub name: String,
    pub api: Api,
    pub provider: ProviderId,
    pub base_url: String,
    pub reasoning: bool,
    /// Maps pi thinking levels to provider/model-specific values. Missing keys
    /// use provider defaults; `None` marks a level as unsupported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_level_map: Option<ThinkingLevelMap>,
    pub input: Vec<InputModality>,
    #[serde(default)]
    pub cost: ModelCost,
    /// Context window in tokens.
    pub context_window: u64,
    /// Maximum output tokens.
    pub max_tokens: u64,
    /// Default sampling parameters; per-request keys override these.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_params: Option<serde_json::Map<String, Json>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// Compatibility overrides; when absent, adapters auto-detect from `base_url`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compat: Option<ApiCompat>,
}

impl Model {
    pub fn supports_images(&self) -> bool {
        self.input.contains(&InputModality::Image)
    }

    /// The projected OpenAI-completions compat fields, when compat is present.
    pub fn openai_completions_compat(&self) -> Option<OpenaiCompletionsCompat> {
        self.compat.as_ref().map(CompatSettings::openai_completions)
    }

    pub fn openai_responses_compat(&self) -> Option<OpenaiResponsesCompat> {
        self.compat.as_ref().map(CompatSettings::openai_responses)
    }

    pub fn anthropic_compat(&self) -> Option<AnthropicMessagesCompat> {
        self.compat.as_ref().map(CompatSettings::anthropic_messages)
    }
}

/// Token usage JSON coercion helper: providers occasionally report counts as
/// floats or strings; only exact integers are accepted.
pub fn json_as_token_count(value: &Json) -> Option<u64> {
    match value {
        Json::Number(number) => number
            .as_u64()
            .or_else(|| number.as_i64().map(|count| count.max(0) as u64))
            .or_else(|| {
                number
                    .as_f64()
                    .filter(|float| {
                        float.is_sign_positive()
                            && float.fract() == 0.0
                            && *float <= u64::MAX as f64
                    })
                    .map(|float| float as u64)
            }),
        Json::String(text) => text.trim().parse::<u64>().ok(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_roundtrips_known_and_custom_values() {
        for value in [
            "anthropic-messages",
            "openai-completions",
            "openai-responses",
            "custom-dialect",
        ] {
            let api: Api = serde_json::from_value(Json::String(value.to_owned())).unwrap();
            let serialized = serde_json::to_value(&api).unwrap();
            assert_eq!(serialized, Json::String(value.to_owned()));
        }
        assert_eq!(Api::from("openai-responses"), Api::OpenaiResponses);
        assert_eq!(Api::Custom("x".into()).to_string(), "x");
    }

    #[test]
    fn message_json_matches_typescript_shape() {
        let message: Message = serde_json::from_value(serde_json::json!({
            "role": "toolResult",
            "toolCallId": "call_1",
            "toolName": "search",
            "content": [{ "type": "text", "text": "hit" }],
            "isError": false,
            "timestamp": 1728000000000u64
        }))
        .unwrap();
        let Message::ToolResult(result) = &message else {
            panic!("expected tool result");
        };
        assert_eq!(result.tool_call_id, "call_1");
        let serialized = serde_json::to_value(&message).unwrap();
        assert_eq!(serialized["toolCallId"], "call_1");
        assert!(serialized.get("isError").is_some());
    }

    #[test]
    fn usage_uses_integer_tokens_and_float_costs() {
        let usage: Usage = serde_json::from_value(serde_json::json!({
			"input": 10,
			"output": 20,
			"cacheRead": 0,
			"cacheWrite": 0,
			"totalTokens": 30,
			"cost": { "input": 0.001, "output": 0.002, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.003 }
		}))
		.unwrap();
        assert_eq!(usage.input, 10);
        assert_eq!(usage.cost.total, 0.003);
    }

    #[test]
    fn token_count_coercion() {
        assert_eq!(json_as_token_count(&Json::from(12u64)), Some(12));
        assert_eq!(json_as_token_count(&serde_json::json!(12.0_f64)), Some(12));
        assert_eq!(json_as_token_count(&serde_json::json!("42")), Some(42));
        assert_eq!(json_as_token_count(&serde_json::json!(12.5)), None);
    }
}
