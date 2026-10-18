//! Anthropic Messages API adapter.
//!
//! Rust port of the TypeScript `anthropic-messages.ts` adapter. Builds the
//! `POST {base_url}/v1/messages` payload, streams the SSE event protocol, and
//! normalizes usage/cost. The SSE framing (`event:`/`data:` lines) is handled by
//! the `eventsource-stream` layer in `api::http`; this module owns the
//! Anthropic event-protocol state machine (message_start → content_block_* →
//! message_delta → message_stop) exposed as the pure `process_stream_frame`
//! reducer so it is testable without a network.

use std::collections::{HashMap, HashSet};

use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde_json::{Value, json};

use crate::api::constrained_sampling::{
    get_json_schema_tool_parameters, resolve_json_schema_strict_sampling,
};
use crate::api::copilot_headers::{build_copilot_dynamic_headers, has_copilot_vision_input};
use crate::api::http::{SseEvent, http_client_for, post_json};
use crate::api::simple_options::{
    adjust_max_tokens_for_thinking, build_base_options, clamp_max_tokens_to_context,
};
use crate::api::transform_messages::transform_messages;
use crate::event::{
    AssistantMessageEvent, AssistantMessageEventStream, assistant_message_event_stream,
    thinking_text_at,
};
use crate::models::{ProviderStreams, calculate_cost};
use crate::types::{
    AssistantContent, AssistantMessage, CacheRetention, Context, Message, Model,
    ModelThinkingLevel, ProviderHeaders, SimpleStreamOptions, StopReason, StreamOptions,
    TextContent, ThinkingContent, ToolCall, ToolResultMessage, Usage, UserContentBlock,
    json_as_token_count,
};
use crate::utils::deferred_tools::split_deferred_tools;
use crate::utils::headers::provider_headers_to_record;
use crate::utils::json_parse::{parse_json_with_repair, parse_streaming_json};
use crate::utils::provider_retry::{ProviderRetryOptions, retry_provider_request_tracked};
use crate::utils::sanitize::sanitize_surrogates;
use crate::utils::user_agent::get_user_agent;

// ---- wire constants --------------------------------------------------------

const FINE_GRAINED_TOOL_STREAMING_BETA: &str = "fine-grained-tool-streaming-2025-05-14";
const INTERLEAVED_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";
const SERVER_SIDE_FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";
const OAUTH_BETA: &str = "oauth-2025-04-20";
const DEFAULT_THINKING_BUDGET_TOKENS: u64 = 1024;

/// SSE event names that carry an Anthropic message event (everything else —
/// `ping`, `error`, and proxy events — is filtered or handled separately).
const ANTHROPIC_MESSAGE_EVENTS: [&str; 6] = [
    "message_start",
    "message_delta",
    "message_stop",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
];

// ---- option value types ----------------------------------------------------

/// Adaptive-thinking effort level (`output_config.effort`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnthropicEffort {
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl AnthropicEffort {
    pub fn as_str(&self) -> &'static str {
        match self {
            AnthropicEffort::Low => "low",
            AnthropicEffort::Medium => "medium",
            AnthropicEffort::High => "high",
            AnthropicEffort::Xhigh => "xhigh",
            AnthropicEffort::Max => "max",
        }
    }

    fn from_str(value: &str) -> Option<Self> {
        Some(match value {
            "low" => AnthropicEffort::Low,
            "medium" => AnthropicEffort::Medium,
            "high" => AnthropicEffort::High,
            "xhigh" => AnthropicEffort::Xhigh,
            "max" => AnthropicEffort::Max,
            _ => return None,
        })
    }
}

/// How thinking content is returned (`thinking.display`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnthropicThinkingDisplay {
    Summarized,
    Omitted,
}

impl AnthropicThinkingDisplay {
    pub fn as_str(&self) -> &'static str {
        match self {
            AnthropicThinkingDisplay::Summarized => "summarized",
            AnthropicThinkingDisplay::Omitted => "omitted",
        }
    }
}

/// Anthropic `tool_choice` behavior.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AnthropicToolChoice {
    Auto,
    Any,
    None,
    Tool { name: String },
}

impl AnthropicToolChoice {
    fn to_value(&self) -> Value {
        match self {
            AnthropicToolChoice::Auto => json!({ "type": "auto" }),
            AnthropicToolChoice::Any => json!({ "type": "any" }),
            AnthropicToolChoice::None => json!({ "type": "none" }),
            AnthropicToolChoice::Tool { name } => json!({ "type": "tool", "name": name }),
        }
    }
}

impl From<crate::types::ToolChoice> for AnthropicToolChoice {
    fn from(value: crate::types::ToolChoice) -> Self {
        match value {
            crate::types::ToolChoice::Auto => AnthropicToolChoice::Auto,
            crate::types::ToolChoice::None => AnthropicToolChoice::None,
        }
    }
}

/// Options for [`stream`]. Extends [`StreamOptions`] with Anthropic-specific
/// controls; shared sampling and transport options live in `base`.
#[derive(Clone, Default)]
pub struct AnthropicOptions {
    pub base: StreamOptions,
    /// Enable extended thinking. Adaptive models decide how much; older models
    /// use budget-based thinking with `thinking_budget_tokens`.
    pub thinking_enabled: Option<bool>,
    /// Token budget for extended thinking (older models only; default 1024).
    pub thinking_budget_tokens: Option<u64>,
    /// Effort level for adaptive thinking models.
    pub effort: Option<AnthropicEffort>,
    /// How thinking content is returned (default `summarized`).
    pub thinking_display: Option<AnthropicThinkingDisplay>,
    /// Request the interleaved-thinking beta for non-adaptive models (default true).
    pub interleaved_thinking: Option<bool>,
    /// Anthropic tool choice behavior.
    pub tool_choice: Option<AnthropicToolChoice>,
}

impl std::fmt::Debug for AnthropicOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicOptions")
            .field("base", &self.base)
            .field("thinking_enabled", &self.thinking_enabled)
            .field("thinking_budget_tokens", &self.thinking_budget_tokens)
            .field("effort", &self.effort)
            .field("thinking_display", &self.thinking_display)
            .field("interleaved_thinking", &self.interleaved_thinking)
            .field("tool_choice", &self.tool_choice)
            .finish()
    }
}

// ---- compat resolution -----------------------------------------------------

/// Resolved `anthropic-compatible` compat flags with TypeScript defaults.
#[derive(Clone, Copy, Debug)]
struct ResolvedCompat {
    supports_eager_tool_input_streaming: bool,
    supports_long_cache_retention: bool,
    send_session_affinity_headers: bool,
    supports_cache_control_on_tools: bool,
    supports_temperature: bool,
    allow_empty_signature: bool,
    supports_strict_tools: bool,
    supports_tool_references: bool,
    force_adaptive_thinking: bool,
}

/// Port of `getAnthropicCompat`: apply model overrides with the same defaults.
fn get_anthropic_compat(model: &Model) -> ResolvedCompat {
    let compat = model.anthropic_compat();
    ResolvedCompat {
        supports_eager_tool_input_streaming: compat
            .as_ref()
            .and_then(|c| c.supports_eager_tool_input_streaming)
            .unwrap_or(true),
        supports_long_cache_retention: compat
            .as_ref()
            .and_then(|c| c.supports_long_cache_retention)
            .unwrap_or(true),
        send_session_affinity_headers: compat
            .as_ref()
            .and_then(|c| c.send_session_affinity_headers)
            .unwrap_or(false),
        supports_cache_control_on_tools: compat
            .as_ref()
            .and_then(|c| c.supports_cache_control_on_tools)
            .unwrap_or(true),
        supports_temperature: compat
            .as_ref()
            .and_then(|c| c.supports_temperature)
            .unwrap_or(true),
        allow_empty_signature: compat
            .as_ref()
            .and_then(|c| c.allow_empty_signature)
            .unwrap_or(false),
        supports_strict_tools: compat
            .as_ref()
            .and_then(|c| c.supports_strict_tools)
            .unwrap_or(false),
        supports_tool_references: compat
            .as_ref()
            .and_then(|c| c.supports_tool_references)
            .unwrap_or_else(|| default_supports_tool_references(model)),
        force_adaptive_thinking: compat
            .as_ref()
            .and_then(|c| c.force_adaptive_thinking)
            .unwrap_or(false),
    }
}

/// Default for `supportsToolReferences`: first-party Anthropic models except
/// Haiku (rejects client-side tool_reference blocks) and models that predate
/// tool search (Claude 3.x, Opus/Sonnet 4.0, Opus 4.1).
fn default_supports_tool_references(model: &Model) -> bool {
    if model.provider.as_str() != crate::types::ProviderId::ANTHROPIC || model.id.contains("haiku")
    {
        return false;
    }
    let Some((major, minor)) = parse_claude_version(&model.id) else {
        return false;
    };
    major > 4 || (major == 4 && minor >= 5)
}

/// Parse `claude-(opus|sonnet|fable)-<major>[-<minor>]` from a model id.
/// Returns `None` when the id does not match. `minor` is only honored when it is
/// a short numeric run (`length < 8`), matching the TypeScript regex + guard.
fn parse_claude_version(id: &str) -> Option<(u64, u64)> {
    let rest = id.strip_prefix("claude-")?;
    let rest = rest
        .strip_prefix("opus-")
        .or_else(|| rest.strip_prefix("sonnet-"))
        .or_else(|| rest.strip_prefix("fable-"))?;
    // major digits
    let major_end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    if major_end == 0 {
        return None;
    }
    let major: u64 = rest[..major_end].parse().ok()?;
    let after = &rest[major_end..];
    // A separator `-` introduces the optional minor; anything else (end or a
    // non-dash continuation such as a date suffix) means minor is absent.
    let minor = if let Some(minor_str) = after.strip_prefix('-') {
        let minor_end = minor_str
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(minor_str.len());
        if minor_end == 0 || minor_end >= 8 {
            0
        } else {
            minor_str[..minor_end].parse().unwrap_or(0)
        }
    } else {
        0
    };
    Some((major, minor))
}

/// Cache-control resolution: `(retention, optional cache_control object)`.
fn get_cache_control(model: &Model, cache_retention: Option<CacheRetention>) -> Option<Value> {
    let retention = cache_retention.unwrap_or_default();
    if matches!(retention, CacheRetention::None) {
        return None;
    }
    let compat = get_anthropic_compat(model);
    let ttl = matches!(retention, CacheRetention::Long) && compat.supports_long_cache_retention;
    if ttl {
        Some(json!({ "type": "ephemeral", "ttl": "1h" }))
    } else {
        Some(json!({ "type": "ephemeral" }))
    }
}

// ---- SSE state machine -----------------------------------------------------

/// One content block under construction, mirroring a `content_block` slot.
struct StreamBlock {
    /// Anthropic content-block index (the `index` field of stream events).
    index: u64,
    /// Scratch accumulation of `input_json_delta` fragments (tool calls only).
    partial_json: String,
}

/// Mutable state threaded through the Anthropic event protocol.
struct StreamState {
    snapshot_mode: crate::types::EventSnapshotMode,
    output: AssistantMessage,
    blocks: Vec<StreamBlock>,
    /// Model used for cost calculation; may swap to a fallback model's rates.
    usage_model: Model,
    saw_message_start: bool,
    saw_message_end: bool,
}

impl StreamState {
    fn snapshot(&self) -> AssistantMessage {
        crate::event::snapshot_message(&self.output, self.snapshot_mode)
    }

    fn new(model: &Model) -> Self {
        StreamState {
            snapshot_mode: Default::default(),
            output: AssistantMessage {
                content: Vec::new(),
                api: model.api.clone(),
                provider: model.provider.clone(),
                model: model.id.clone(),
                response_model: None,
                response_id: None,
                diagnostics: None,
                usage: Usage::default(),
                stop_reason: StopReason::Pending,
                deferred: None,
                error_message: None,
                raw_stop_reason: None,
                end_turn: None,
                timestamp: crate::utils::now_ms(),
            },
            blocks: Vec::new(),
            usage_model: model.clone(),
            saw_message_start: false,
            saw_message_end: false,
        }
    }

    fn block_position(&self, index: u64) -> Option<usize> {
        self.blocks.iter().position(|block| block.index == index)
    }
}

/// A failure that terminates the stream: `aborted` selects stop reason
/// `Aborted` (vs `Error`) and the message becomes `error_message`.
struct StreamFailure {
    aborted: bool,
    message: String,
}

impl StreamFailure {
    fn error(message: impl Into<String>) -> Self {
        StreamFailure {
            aborted: false,
            message: message.into(),
        }
    }

    fn aborted() -> Self {
        StreamFailure {
            aborted: true,
            message: "Request was aborted".to_owned(),
        }
    }
}

/// Token field extraction with the shared numeric coercion helper.
fn json_token(value: &Value, key: &str) -> Option<u64> {
    value.get(key).and_then(json_as_token_count)
}

fn json_string(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(|entry| entry.as_str())
        .map(str::to_owned)
}

/// Map an Anthropic `stop_reason` to a [`StopReason`] plus an optional error
/// message. Unknown reasons surface as an error (matching the TS throw).
fn map_stop_reason(
    reason: &str,
    stop_details: Option<&Value>,
) -> Result<(StopReason, Option<String>), String> {
    match reason {
        "end_turn" | "pause_turn" | "stop_sequence" => Ok((StopReason::Stop, None)),
        "max_tokens" => Ok((StopReason::Length, None)),
        "tool_use" => Ok((StopReason::ToolUse, None)),
        "refusal" => {
            let explanation = stop_details
                .and_then(|details| json_string(details, "explanation"))
                .unwrap_or_else(|| "The model refused to complete the request".to_owned());
            Ok((StopReason::Error, Some(explanation)))
        }
        "sensitive" => Ok((
            StopReason::Error,
            Some("Provider stopped with: sensitive".to_owned()),
        )),
        other => Err(format!("Unhandled stop reason: {other}")),
    }
}

/// Reduce one parsed Anthropic stream event into zero or more assistant events,
/// mutating `state`. Pure — no I/O — so the protocol is unit-testable.
fn reduce_event(
    state: &mut StreamState,
    model: &Model,
    event: &Value,
) -> Result<Vec<AssistantMessageEvent>, String> {
    let mut events = Vec::new();
    let event_type = event
        .get("type")
        .and_then(|value| value.as_str())
        .unwrap_or("");
    match event_type {
        "message_start" => {
            let message = event.get("message").cloned().unwrap_or(Value::Null);
            if let Some(id) = json_string(&message, "id") {
                state.output.response_id = Some(id);
            }
            if let Some(model_id) = json_string(&message, "model") {
                state.output.model = model_id;
            }
            // Fallback pricing when the response model differs from the request.
            let fallbacks = model
                .anthropic_compat()
                .and_then(|compat| compat.allowed_fallback_models);
            if state.output.model != model.id
                && let Some(fallbacks) = fallbacks.as_ref()
                && let Some(fallback) = fallbacks.iter().find(|fallback| {
                    fallback.provider == model.provider && fallback.model == state.output.model
                })
            {
                let mut usage_model = model.clone();
                usage_model.id = state.output.model.clone();
                usage_model.cost = fallback.cost.clone();
                state.usage_model = usage_model;
            }
            let usage = message.get("usage").cloned().unwrap_or(Value::Null);
            state.output.usage.input = json_token(&usage, "input_tokens").unwrap_or(0);
            state.output.usage.output = json_token(&usage, "output_tokens").unwrap_or(0);
            state.output.usage.cache_read =
                json_token(&usage, "cache_read_input_tokens").unwrap_or(0);
            state.output.usage.cache_write =
                json_token(&usage, "cache_creation_input_tokens").unwrap_or(0);
            state.output.usage.cache_write_1h = usage
                .get("cache_creation")
                .and_then(|creation| json_token(creation, "ephemeral_1h_input_tokens"));
            recompute_total(&mut state.output.usage);
            calculate_cost(&state.usage_model, &mut state.output.usage);
        }
        "content_block_start" => {
            let index = event
                .get("index")
                .and_then(json_as_token_count)
                .unwrap_or(0);
            let block = event.get("content_block").cloned().unwrap_or(Value::Null);
            let block_type = block
                .get("type")
                .and_then(|value| value.as_str())
                .unwrap_or("");
            let content_index = state.output.content.len();
            match block_type {
                "text" => {
                    let text = json_string(&block, "text").unwrap_or_default();
                    state.blocks.push(StreamBlock {
                        index,
                        partial_json: String::new(),
                    });
                    state
                        .output
                        .content
                        .push(AssistantContent::Text(TextContent {
                            text,
                            text_signature: None,
                        }));
                    events.push(AssistantMessageEvent::TextStart {
                        content_index,
                        partial: state.snapshot(),
                    });
                }
                "thinking" => {
                    let thinking = json_string(&block, "thinking").unwrap_or_default();
                    let signature = json_string(&block, "signature").unwrap_or_default();
                    state.blocks.push(StreamBlock {
                        index,
                        partial_json: String::new(),
                    });
                    state
                        .output
                        .content
                        .push(AssistantContent::Thinking(ThinkingContent {
                            thinking,
                            thinking_signature: Some(signature),
                            redacted: false,
                        }));
                    events.push(AssistantMessageEvent::ThinkingStart {
                        content_index,
                        partial: state.snapshot(),
                    });
                }
                "redacted_thinking" => {
                    let data = json_string(&block, "data").unwrap_or_default();
                    state.blocks.push(StreamBlock {
                        index,
                        partial_json: String::new(),
                    });
                    state
                        .output
                        .content
                        .push(AssistantContent::Thinking(ThinkingContent {
                            thinking: "[Reasoning redacted]".to_owned(),
                            thinking_signature: Some(data),
                            redacted: true,
                        }));
                    events.push(AssistantMessageEvent::ThinkingStart {
                        content_index,
                        partial: state.snapshot(),
                    });
                }
                "tool_use" => {
                    let id = json_string(&block, "id").unwrap_or_default();
                    let name = json_string(&block, "name").unwrap_or_default();
                    let arguments = block
                        .get("input")
                        .and_then(|input| input.as_object())
                        .cloned()
                        .unwrap_or_default();
                    state.blocks.push(StreamBlock {
                        index,
                        partial_json: String::new(),
                    });
                    state
                        .output
                        .content
                        .push(AssistantContent::ToolCall(ToolCall {
                            id,
                            name,
                            arguments,
                            thought_signature: None,
                            namespace: None,
                        }));
                    events.push(AssistantMessageEvent::ToolCallStart {
                        content_index,
                        partial: state.snapshot(),
                    });
                }
                _ => {}
            }
        }
        "content_block_delta" => {
            let index = event
                .get("index")
                .and_then(json_as_token_count)
                .unwrap_or(0);
            let delta = event.get("delta").cloned().unwrap_or(Value::Null);
            let delta_type = delta
                .get("type")
                .and_then(|value| value.as_str())
                .unwrap_or("");
            let Some(position) = state.block_position(index) else {
                return Ok(events);
            };
            match delta_type {
                "text_delta" => {
                    let text = json_string(&delta, "text").unwrap_or_default();
                    if let Some(AssistantContent::Text(block)) =
                        state.output.content.get_mut(position)
                    {
                        block.text.push_str(&text);
                        events.push(AssistantMessageEvent::TextDelta {
                            content_index: position,
                            delta: text,
                            partial: state.snapshot(),
                        });
                    }
                }
                "thinking_delta" => {
                    let thinking = json_string(&delta, "thinking").unwrap_or_default();
                    if let Some(AssistantContent::Thinking(block)) =
                        state.output.content.get_mut(position)
                    {
                        block.thinking.push_str(&thinking);
                        events.push(AssistantMessageEvent::ThinkingDelta {
                            content_index: position,
                            delta: thinking,
                            partial: state.snapshot(),
                        });
                    }
                }
                "signature_delta" => {
                    let signature = json_string(&delta, "signature").unwrap_or_default();
                    if let Some(AssistantContent::Thinking(block)) =
                        state.output.content.get_mut(position)
                    {
                        let current = block.thinking_signature.get_or_insert_with(String::new);
                        current.push_str(&signature);
                    }
                }
                "input_json_delta" => {
                    let partial = json_string(&delta, "partial_json").unwrap_or_default();
                    if let Some(block) = state.blocks.get_mut(position)
                        && let Some(AssistantContent::ToolCall(call)) =
                            state.output.content.get_mut(position)
                    {
                        block.partial_json.push_str(&partial);
                        let parsed = parse_streaming_json(Some(&block.partial_json));
                        if let Some(object) = parsed.as_object() {
                            call.arguments = object.clone();
                        }
                        events.push(AssistantMessageEvent::ToolCallDelta {
                            content_index: position,
                            delta: partial,
                            partial: state.snapshot(),
                        });
                    }
                }
                _ => {}
            }
        }
        "content_block_stop" => {
            let index = event
                .get("index")
                .and_then(json_as_token_count)
                .unwrap_or(0);
            let Some(position) = state.block_position(index) else {
                return Ok(events);
            };
            match state.output.content.get(position).cloned() {
                Some(AssistantContent::Text(block)) => {
                    events.push(AssistantMessageEvent::TextEnd {
                        content_index: position,
                        content: block.text,
                        partial: state.snapshot(),
                    });
                }
                Some(AssistantContent::Thinking(_)) => {
                    events.push(AssistantMessageEvent::ThinkingEnd {
                        content_index: position,
                        content: thinking_text_at(&state.output.content, position),
                        partial: state.snapshot(),
                    });
                }
                Some(AssistantContent::ToolCall(call)) => {
                    // Finalize arguments from the accumulated partial JSON, then
                    // strip the scratch buffer so replay only carries parsed args.
                    let mut call = call;
                    if let Some(block) = state.blocks.get(position)
                        && !block.partial_json.is_empty()
                    {
                        call.arguments =
                            crate::utils::json_parse::parse_tool_arguments(&block.partial_json)
                                .map_err(|error| error.to_string())?;
                    }
                    if let Some(AssistantContent::ToolCall(slot)) =
                        state.output.content.get_mut(position)
                    {
                        slot.arguments = call.arguments.clone();
                    }
                    events.push(AssistantMessageEvent::ToolCallEnd {
                        content_index: position,
                        tool_call: call,
                        partial: state.snapshot(),
                    });
                }
                None => {}
            }
            // Scratch buffer is never persisted; drop it after finalization.
            if let Some(block) = state.blocks.get_mut(position) {
                block.partial_json.clear();
            }
        }
        "message_delta" => {
            if let Some(delta) = event.get("delta")
                && let Some(reason) = delta.get("stop_reason").and_then(|value| value.as_str())
            {
                state.output.raw_stop_reason = Some(reason.to_owned());
                let (stop_reason, error_message) =
                    map_stop_reason(reason, delta.get("stop_details"))?;
                state.output.stop_reason = stop_reason;
                if let Some(message) = error_message {
                    state.output.error_message = Some(message);
                }
            }
            if let Some(usage) = event.get("usage") {
                if let Some(value) = json_token(usage, "input_tokens") {
                    state.output.usage.input = value;
                }
                if let Some(value) = json_token(usage, "output_tokens") {
                    state.output.usage.output = value;
                }
                if let Some(value) = json_token(usage, "cache_read_input_tokens") {
                    state.output.usage.cache_read = value;
                }
                if let Some(value) = json_token(usage, "cache_creation_input_tokens") {
                    state.output.usage.cache_write = value;
                }
                if let Some(details) = usage.get("output_tokens_details")
                    && let Some(value) = json_token(details, "thinking_tokens")
                {
                    state.output.usage.reasoning = Some(value);
                }
            }
            recompute_total(&mut state.output.usage);
            calculate_cost(&state.usage_model, &mut state.output.usage);
        }
        _ => {}
    }
    Ok(events)
}

fn recompute_total(usage: &mut Usage) {
    usage.total_tokens = usage
        .input
        .saturating_add(usage.output)
        .saturating_add(usage.cache_read)
        .saturating_add(usage.cache_write);
}

/// Process one framed SSE event: filter non-message events, handle `error`,
/// parse with repair, and reduce. Returns the emitted assistant events.
fn process_stream_frame(
    state: &mut StreamState,
    model: &Model,
    frame: &SseEvent,
) -> Result<Vec<AssistantMessageEvent>, StreamFailure> {
    if frame.event == "error" {
        return Err(StreamFailure::error(frame.data.clone()));
    }
    if !ANTHROPIC_MESSAGE_EVENTS.contains(&frame.event.as_str()) {
        return Ok(Vec::new());
    }
    let value = parse_json_with_repair(&frame.data).map_err(|error| {
        StreamFailure::error(format!(
            "Could not parse Anthropic SSE event {}: {error}; data={}",
            frame.event, frame.data
        ))
    })?;
    let event_type = value
        .get("type")
        .and_then(|entry| entry.as_str())
        .unwrap_or("");
    if event_type == "message_start" {
        state.saw_message_start = true;
    } else if event_type == "message_stop" {
        state.saw_message_end = true;
    }
    reduce_event(state, model, &value).map_err(StreamFailure::error)
}

// ---- request payload building ---------------------------------------------

/// Normalize a tool call id to Anthropic's `[a-zA-Z0-9_-]{1,64}` pattern.
fn normalize_tool_call_id(id: &str) -> String {
    let mut normalized: String = id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .take(64)
        .collect();
    if normalized.is_empty() {
        normalized = "_".to_owned();
    }
    normalized
}

/// Convert user/tool-result content blocks to Anthropic's string-or-blocks form.
fn convert_content_blocks(content: &[UserContentBlock]) -> Value {
    let has_images = content
        .iter()
        .any(|block| matches!(block, UserContentBlock::Image(_)));
    if !has_images {
        let text = content
            .iter()
            .filter_map(|block| block.text())
            .map(|block| block.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        return Value::String(sanitize_surrogates(&text));
    }

    // With images, Anthropic rejects whitespace-only text blocks; drop them.
    let mut blocks: Vec<Value> = content
        .iter()
        .filter(
            |block| !matches!(block, UserContentBlock::Text(text) if text.text.trim().is_empty()),
        )
        .map(|block| match block {
            UserContentBlock::Text(text) => {
                json!({ "type": "text", "text": sanitize_surrogates(&text.text) })
            }
            UserContentBlock::Image(image) => json!({
                "type": "image",
                "source": { "type": "base64", "media_type": image.mime_type, "data": image.data },
            }),
        })
        .collect();

    let has_text = blocks
        .iter()
        .any(|block| block.get("type").and_then(|value| value.as_str()) == Some("text"));
    if !has_text {
        blocks.insert(0, json!({ "type": "text", "text": "(see attached image)" }));
    }
    Value::Array(blocks)
}

/// Convert one tool-result message to a `tool_result` block plus any displaced
/// sibling content (tool references are emitted as their own blocks).
fn convert_tool_result(
    msg: &ToolResultMessage,
    deferred_tool_names: &HashSet<String>,
    loaded_tool_names: &mut HashSet<String>,
) -> (Value, Vec<Value>) {
    let mut references: Vec<Value> = Vec::new();
    for name in msg.added_tool_names.iter().flatten() {
        if !deferred_tool_names.contains(name) || loaded_tool_names.contains(name) {
            continue;
        }
        loaded_tool_names.insert(name.clone());
        references.push(json!({ "type": "tool_reference", "tool_name": name }));
    }

    let has_references = !references.is_empty();
    let converted = convert_content_blocks(&msg.content);
    // Anthropic rejects tool references mixed with ordinary tool-result content,
    // so the references replace the content and the content moves to siblings.
    let tool_result = json!({
        "type": "tool_result",
        "tool_use_id": msg.tool_call_id,
        "content": if has_references { Value::Array(references) } else { converted.clone() },
        "is_error": msg.is_error,
    });

    let sibling_content = if !has_references {
        Vec::new()
    } else {
        match &converted {
            Value::String(text) => vec![json!({ "type": "text", "text": text })],
            Value::Array(blocks) => blocks.clone(),
            other => vec![other.clone()],
        }
    };
    (tool_result, sibling_content)
}

/// Convert transformed messages to Anthropic `MessageParam[]`.
fn convert_messages(
    messages: &[Message],
    cache_control: Option<&Value>,
    allow_empty_signature: bool,
    deferred_tool_names: &HashSet<String>,
) -> Vec<Value> {
    let mut params: Vec<Value> = Vec::new();
    let mut loaded_tool_names: HashSet<String> = HashSet::new();

    let mut index = 0;
    while index < messages.len() {
        let msg = &messages[index];
        match msg {
            Message::User(user) => {
                match &user.content {
                    crate::types::UserContent::Text(text) => {
                        if !text.trim().is_empty() {
                            params.push(
                                json!({ "role": "user", "content": sanitize_surrogates(text) }),
                            );
                        }
                    }
                    crate::types::UserContent::Blocks(blocks) => {
                        let converted: Vec<Value> = blocks
                            .iter()
                            .map(|block| match block {
                                UserContentBlock::Text(text) => {
                                    json!({ "type": "text", "text": sanitize_surrogates(&text.text) })
                                }
                                UserContentBlock::Image(image) => json!({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": image.mime_type,
                                        "data": image.data,
                                    },
                                }),
                            })
                            .filter(|block| {
                                block.get("type").and_then(|value| value.as_str()) != Some("text")
                                    || block
                                        .get("text")
                                        .and_then(|value| value.as_str())
                                        .is_none_or(|text| !text.trim().is_empty())
                            })
                            .collect();
                        if !converted.is_empty() {
                            params.push(json!({ "role": "user", "content": converted }));
                        }
                    }
                }
                index += 1;
            }
            Message::Assistant(assistant) => {
                let mut blocks: Vec<Value> = Vec::new();
                for block in &assistant.content {
                    match block {
                        AssistantContent::Text(text) => {
                            if text.text.trim().is_empty() {
                                continue;
                            }
                            blocks.push(
                                json!({ "type": "text", "text": sanitize_surrogates(&text.text) }),
                            );
                        }
                        AssistantContent::Thinking(thinking) => {
                            if thinking.redacted {
                                blocks.push(json!({
                                    "type": "redacted_thinking",
                                    "data": thinking.thinking_signature.clone().unwrap_or_default(),
                                }));
                                continue;
                            }
                            let signature = thinking.thinking_signature.clone().unwrap_or_default();
                            let has_signature = !signature.trim().is_empty();
                            if thinking.thinking.trim().is_empty() && !has_signature {
                                continue;
                            }
                            if has_signature {
                                blocks.push(json!({
                                    "type": "thinking",
                                    "thinking": sanitize_surrogates(&thinking.thinking),
                                    "signature": signature,
                                }));
                            } else if allow_empty_signature {
                                blocks.push(json!({
                                    "type": "thinking",
                                    "thinking": sanitize_surrogates(&thinking.thinking),
                                    "signature": "",
                                }));
                            } else {
                                blocks.push(json!({ "type": "text", "text": sanitize_surrogates(&thinking.thinking) }));
                            }
                        }
                        AssistantContent::ToolCall(call) => {
                            blocks.push(json!({
                                "type": "tool_use",
                                "id": call.id,
                                "name": call.name,
                                "input": Value::Object(call.arguments.clone()),
                            }));
                        }
                    }
                }
                if !blocks.is_empty() {
                    params.push(json!({ "role": "assistant", "content": blocks }));
                }
                index += 1;
            }
            Message::ToolResult(_) => {
                // Collect all consecutive tool-result messages.
                let mut tool_results: Vec<Value> = Vec::new();
                let mut sibling_content: Vec<Value> = Vec::new();
                while let Some(Message::ToolResult(result)) = messages.get(index) {
                    let (tool_result, siblings) =
                        convert_tool_result(result, deferred_tool_names, &mut loaded_tool_names);
                    tool_results.push(tool_result);
                    sibling_content.extend(siblings);
                    index += 1;
                }
                let mut content = tool_results;
                content.extend(sibling_content);
                params.push(json!({ "role": "user", "content": content }));
            }
        }
    }

    // Cache the conversation history by tagging the last user message's last block.
    if let Some(cache_control) = cache_control
        && let Some(last) = params.last_mut()
        && last.get("role").and_then(|role| role.as_str()) == Some("user")
    {
        match last.get_mut("content") {
            Some(Value::Array(blocks)) => {
                if let Some(last_block) = blocks.last_mut()
                    && let Some(object) = last_block.as_object_mut()
                    && matches!(
                        object.get("type").and_then(|value| value.as_str()),
                        Some("text") | Some("image") | Some("tool_result")
                    )
                {
                    object.insert("cache_control".to_owned(), cache_control.clone());
                }
            }
            Some(Value::String(text)) => {
                let text = text.clone();
                *last = json!({
                    "role": "user",
                    "content": [{ "type": "text", "text": text, "cache_control": cache_control }],
                });
            }
            _ => {}
        }
    }

    params
}

/// Convert tools to Anthropic tool definitions. `strict` sampling errors
/// (`require` + unsupported schema) surface as an error.
fn convert_tools(
    tools: &[crate::types::Tool],
    supports_eager_tool_input_streaming: bool,
    supports_strict_tools: bool,
    cache_control: Option<&Value>,
    defer_loading: bool,
) -> Result<Vec<Value>, String> {
    let mut converted = Vec::with_capacity(tools.len());
    for (index, tool) in tools.iter().enumerate() {
        let strict = resolve_json_schema_strict_sampling(tool, supports_strict_tools)?;
        let parameters = get_json_schema_tool_parameters(tool, strict);
        let properties = parameters.get("properties").cloned().unwrap_or(json!({}));
        let required = parameters.get("required").cloned().unwrap_or(json!([]));
        let mut legacy = serde_json::Map::new();
        legacy.insert("type".to_owned(), json!("object"));
        legacy.insert("properties".to_owned(), properties);
        legacy.insert("required".to_owned(), required);
        if let Some(defs) = parameters.get("$defs") {
            legacy.insert("$defs".to_owned(), defs.clone());
        }
        if let Some(definitions) = parameters.get("definitions") {
            legacy.insert("definitions".to_owned(), definitions.clone());
        }
        let input_schema = if strict == Some(true) {
            let mut merged = parameters.as_object().cloned().unwrap_or_default();
            for (key, value) in legacy {
                merged.insert(key, value);
            }
            Value::Object(merged)
        } else {
            Value::Object(legacy)
        };

        let mut tool_json = serde_json::Map::new();
        tool_json.insert("name".to_owned(), json!(tool.name));
        tool_json.insert("description".to_owned(), json!(tool.description));
        if supports_eager_tool_input_streaming {
            tool_json.insert("eager_input_streaming".to_owned(), json!(true));
        }
        if strict == Some(true) {
            tool_json.insert("strict".to_owned(), json!(true));
        }
        tool_json.insert("input_schema".to_owned(), input_schema);
        if defer_loading {
            tool_json.insert("defer_loading".to_owned(), json!(true));
        }
        if let Some(cache_control) = cache_control
            && index == tools.len() - 1
        {
            tool_json.insert("cache_control".to_owned(), cache_control.clone());
        }
        converted.push(Value::Object(tool_json));
    }
    Ok(converted)
}

/// Whether the model explicitly maps `off` to `null` (unsupported), in which
/// case `thinking: { type: "disabled" }` is withheld.
fn off_is_supported(model: &Model) -> bool {
    match &model.thinking_level_map {
        None => true,
        Some(map) => map.get(&ModelThinkingLevel::Off) != Some(&None),
    }
}

/// Build the `POST /v1/messages` request body.
fn build_params(
    model: &Model,
    context: &Context,
    options: &AnthropicOptions,
) -> Result<Value, String> {
    let cache_control = get_cache_control(model, options.base.cache_retention);
    let compat = get_anthropic_compat(model);
    let normalize = |id: &str| normalize_tool_call_id(id);
    let transformed = transform_messages(
        &context.messages,
        model,
        Some(&|id: &str, _: &Model, _: &_| normalize(id)),
    );

    let mut placement = split_deferred_tools(
        &Context {
            system_prompt: context.system_prompt.clone(),
            messages: transformed.clone(),
            tools: context.tools.clone(),
        },
        compat.supports_tool_references,
        |name: &str| name.to_owned(),
    );
    let mut immediate_tools = std::mem::take(&mut placement.immediate);
    let mut deferred_tools: Vec<crate::types::Tool> = placement.deferred.into_values().collect();
    if immediate_tools.is_empty() && !deferred_tools.is_empty() {
        immediate_tools = std::mem::take(&mut deferred_tools);
    }
    let deferred_tool_names: HashSet<String> = deferred_tools
        .iter()
        .map(|tool| tool.name.clone())
        .collect();

    let messages = convert_messages(
        &transformed,
        cache_control.as_ref(),
        compat.allow_empty_signature,
        &deferred_tool_names,
    );

    let mut params = serde_json::Map::new();
    params.insert("model".to_owned(), json!(model.id));
    params.insert("messages".to_owned(), Value::Array(messages));
    params.insert(
        "max_tokens".to_owned(),
        json!(options.base.max_tokens.unwrap_or(model.max_tokens)),
    );
    params.insert("stream".to_owned(), json!(true));

    if let Some(system_prompt) = &context.system_prompt {
        let mut system_block = serde_json::Map::new();
        system_block.insert("type".to_owned(), json!("text"));
        system_block.insert("text".to_owned(), json!(sanitize_surrogates(system_prompt)));
        if let Some(cache_control) = &cache_control {
            system_block.insert("cache_control".to_owned(), cache_control.clone());
        }
        params.insert(
            "system".to_owned(),
            Value::Array(vec![Value::Object(system_block)]),
        );
    }

    // Temperature is incompatible with extended thinking and unsupported on
    // some models (e.g. Claude Opus 4.7+).
    if let Some(temperature) = options.base.temperature
        && options.thinking_enabled != Some(true)
        && compat.supports_temperature
    {
        params.insert("temperature".to_owned(), json!(temperature));
    }

    if !immediate_tools.is_empty() || !deferred_tools.is_empty() {
        let tools_cache_control = if compat.supports_cache_control_on_tools {
            cache_control.as_ref()
        } else {
            None
        };
        let mut tools = convert_tools(
            &immediate_tools,
            compat.supports_eager_tool_input_streaming,
            compat.supports_strict_tools,
            tools_cache_control,
            false,
        )?;
        tools.extend(convert_tools(
            &deferred_tools,
            compat.supports_eager_tool_input_streaming,
            compat.supports_strict_tools,
            None,
            true,
        )?);
        params.insert("tools".to_owned(), Value::Array(tools));
    }

    // Thinking mode: adaptive, budget-based, or explicitly disabled.
    if model.reasoning {
        if options.thinking_enabled == Some(true) {
            let display = options
                .thinking_display
                .unwrap_or(AnthropicThinkingDisplay::Summarized);
            if compat.force_adaptive_thinking {
                params.insert(
                    "thinking".to_owned(),
                    json!({ "type": "adaptive", "display": display.as_str() }),
                );
                if let Some(effort) = options.effort {
                    params.insert(
                        "output_config".to_owned(),
                        json!({ "effort": effort.as_str() }),
                    );
                }
            } else {
                let raw_budget = options.thinking_budget_tokens.unwrap_or(0);
                let budget = if raw_budget == 0 {
                    DEFAULT_THINKING_BUDGET_TOKENS
                } else {
                    raw_budget
                };
                params.insert(
                    "thinking".to_owned(),
                    json!({ "type": "enabled", "budget_tokens": budget, "display": display.as_str() }),
                );
            }
        } else if options.thinking_enabled == Some(false) && off_is_supported(model) {
            params.insert("thinking".to_owned(), json!({ "type": "disabled" }));
        }
    }

    if let Some(metadata) = &options.base.metadata
        && let Some(user_id) = metadata.get("user_id").and_then(|value| value.as_str())
    {
        params.insert("metadata".to_owned(), json!({ "user_id": user_id }));
    }

    if let Some(tool_choice) = &options.tool_choice {
        params.insert("tool_choice".to_owned(), tool_choice.to_value());
    }

    let fallbacks = model
        .anthropic_compat()
        .and_then(|compat| compat.allowed_fallback_models);
    if let Some(fallbacks) = fallbacks.as_ref()
        && !fallbacks.is_empty()
    {
        params.insert(
            "fallbacks".to_owned(),
            Value::Array(
                fallbacks
                    .iter()
                    .map(|fallback| json!({ "model": fallback.model }))
                    .collect(),
            ),
        );
    }

    Ok(Value::Object(params))
}

// ---- request headers -------------------------------------------------------

/// Case-insensitive "header present with a non-empty value" check.
fn has_header(headers: &ProviderHeaders, name: &str) -> bool {
    let expected = name.to_ascii_lowercase();
    headers.iter().any(|(key, value)| {
        key.to_ascii_lowercase() == expected
            && value
                .as_deref()
                .is_some_and(|value| !value.trim().is_empty())
    })
}

/// Assert that some auth is available: an API key, or an auth header supplied
/// by the caller. Returns the error message on failure.
fn assert_request_auth(
    provider: &str,
    api_key: Option<&str>,
    headers: Option<&ProviderHeaders>,
) -> Result<(), String> {
    if api_key.is_some() {
        return Ok(());
    }
    if let Some(headers) = headers
        && (has_header(headers, "authorization")
            || has_header(headers, "x-api-key")
            || has_header(headers, "cf-aig-authorization"))
    {
        return Ok(());
    }
    Err(format!("No API key for provider: {provider}"))
}

fn is_oauth_token(api_key: &str) -> bool {
    api_key.contains("sk-ant-oat")
}

/// Assemble the request headers (port of `createClient`'s three auth branches).
fn build_request_headers(
    model: &Model,
    options: &AnthropicOptions,
    context: &Context,
) -> Result<reqwest::header::HeaderMap, String> {
    let compat = get_anthropic_compat(model);
    let interleaved_thinking = options.interleaved_thinking.unwrap_or(true);
    let needs_interleaved_beta = interleaved_thinking && !compat.force_adaptive_thinking;
    let use_fine_grained = context
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty())
        && !compat.supports_eager_tool_input_streaming;
    let use_server_side_fallback = model
        .anthropic_compat()
        .and_then(|compat| compat.allowed_fallback_models)
        .is_some_and(|fallbacks| !fallbacks.is_empty());

    let mut beta_features: Vec<&str> = Vec::new();
    if use_fine_grained {
        beta_features.push(FINE_GRAINED_TOOL_STREAMING_BETA);
    }
    if needs_interleaved_beta {
        beta_features.push(INTERLEAVED_THINKING_BETA);
    }
    if use_server_side_fallback {
        beta_features.push(SERVER_SIDE_FALLBACK_BETA);
    }

    let model_headers: ProviderHeaders = model
        .headers
        .as_ref()
        .map(|headers| {
            headers
                .iter()
                .map(|(name, value)| (name.clone(), Some(value.clone())))
                .collect()
        })
        .unwrap_or_default();
    let options_headers = options.base.base.headers.clone().unwrap_or_default();

    let mut merged: ProviderHeaders = HashMap::new();
    merged.insert("User-Agent".to_owned(), Some(get_user_agent()));
    merged.insert("accept".to_owned(), Some("application/json".to_owned()));
    merged.insert(
        "anthropic-dangerous-direct-browser-access".to_owned(),
        Some("true".to_owned()),
    );

    if model.provider.as_str() == crate::types::ProviderId::GITHUB_COPILOT {
        if let Some(api_key) = &options.base.base.api_key {
            merged.insert(
                "Authorization".to_owned(),
                Some(format!("Bearer {api_key}")),
            );
        }
        if !beta_features.is_empty() {
            merged.insert("anthropic-beta".to_owned(), Some(beta_features.join(",")));
        }
        merge_into(&mut merged, &model_headers);
        let dynamic = build_copilot_dynamic_headers(
            &context.messages,
            has_copilot_vision_input(&context.messages),
        );
        for (name, value) in dynamic {
            merged.insert(name, Some(value));
        }
        merge_into(&mut merged, &options_headers);
    } else if options
        .base
        .base
        .api_key
        .as_deref()
        .is_some_and(is_oauth_token)
    {
        if let Some(api_key) = &options.base.base.api_key {
            merged.insert(
                "Authorization".to_owned(),
                Some(format!("Bearer {api_key}")),
            );
        }
        let mut betas = vec![OAUTH_BETA];
        betas.extend(beta_features.iter().copied());
        merged.insert("anthropic-beta".to_owned(), Some(betas.join(",")));
        merge_into(&mut merged, &model_headers);
        merge_into(&mut merged, &options_headers);
    } else {
        // API key or header-owned auth.
        let cache_retention = options.base.cache_retention.unwrap_or_default();
        if let Some(session_id) = &options.base.session_id
            && !matches!(cache_retention, CacheRetention::None)
            && compat.send_session_affinity_headers
        {
            merged.insert("x-session-affinity".to_owned(), Some(session_id.clone()));
        }
        if let Some(api_key) = &options.base.base.api_key {
            merged.insert("x-api-key".to_owned(), Some(api_key.clone()));
        }
        if !beta_features.is_empty() {
            merged.insert("anthropic-beta".to_owned(), Some(beta_features.join(",")));
        }
        merge_into(&mut merged, &model_headers);
        merge_into(&mut merged, &options_headers);
    }

    crate::utils::headers::provider_headers_to_header_map(&merged)
        .map_err(|error| error.to_string())
}

/// Merge `override_` into `base` with case-insensitive replacement.
fn merge_into(base: &mut ProviderHeaders, override_: &ProviderHeaders) {
    for (name, value) in override_ {
        let lower = name.to_ascii_lowercase();
        base.retain(|existing, _| existing.to_ascii_lowercase() != lower);
        base.insert(name.clone(), value.clone());
    }
}

fn messages_url(model: &Model) -> String {
    format!("{}/v1/messages", model.base_url.trim_end_matches('/'))
}

// ---- streaming -------------------------------------------------------------

/// Map a terminal [`StopReason`] to the stream's done reason.
fn done_reason(stop_reason: StopReason) -> crate::event::DoneReason {
    match stop_reason {
        StopReason::Stop => crate::event::DoneReason::Stop,
        StopReason::Length => crate::event::DoneReason::Length,
        StopReason::ToolUse => crate::event::DoneReason::ToolUse,
        StopReason::Deferred => crate::event::DoneReason::Deferred,
        // Error/Aborted never reach the Done event; the caller routes them to Error.
        StopReason::Error | StopReason::Aborted | StopReason::Pending => {
            crate::event::DoneReason::Stop
        }
    }
}

/// Drive the HTTP request and SSE loop, pushing events to `sender`. Returns the
/// finalized message on success, or a [`StreamFailure`] describing the terminal
/// error (with `state.output` carrying any accumulated content).
async fn drive_stream(
    model: &Model,
    context: &Context,
    options: &AnthropicOptions,
    state: &mut StreamState,
    sender: &mut crate::event::AssistantMessageEventSender,
) -> Result<(), StreamFailure> {
    assert_request_auth(
        &model.provider.to_string(),
        options.base.base.api_key.as_deref(),
        options.base.base.headers.as_ref(),
    )
    .map_err(StreamFailure::error)?;

    let payload = build_params(model, context, options).map_err(StreamFailure::error)?;
    let payload = match &options.base.on_payload {
        Some(observer) => observer
            .on_payload(payload.clone(), model)
            .await
            .unwrap_or(payload),
        None => payload,
    };

    let client = http_client_for(&options.base.base)
        .map_err(|error| StreamFailure::error(error.to_string()))?;
    let headers = build_request_headers(model, options, context).map_err(StreamFailure::error)?;
    let url = messages_url(model);
    let token = options.base.base.signal.clone().unwrap_or_default();
    let retry = ProviderRetryOptions {
        max_retries: options.base.base.max_retries,
        max_retry_delay_ms: options.base.base.max_retry_delay_ms,
        signal: options.base.base.signal.clone(),
    };
    let payload_for_request = payload.clone();
    let timeout_ms = options.base.base.timeout_ms;
    let response = retry_provider_request_tracked(
        || {
            let client = client.clone();
            let headers = headers.clone();
            let url = url.clone();
            let payload = payload_for_request.clone();
            async move {
                let response = post_json(client, &url, &headers, &payload, timeout_ms).await?;
                let status = response.status().as_u16();
                if !(200..300).contains(&status) {
                    return Err(crate::api::http::error_from_response(response).await);
                }
                Ok(response)
            }
        },
        retry,
        options.base.base.request_metrics.as_ref(),
    )
    .await;

    let response = match response {
        Ok(response) => response,
        Err(error) => {
            return Err(StreamFailure::error(crate::error::format_provider_error(
                &error, None,
            )));
        }
    };

    if let Some(metrics) = &options.base.base.request_metrics { metrics.response(response.status().as_u16()); }
    if let Some(observer) = &options.base.on_response {
        let header_record =
            provider_headers_to_record(Some(&headers_to_provider(response.headers())));
        observer
            .on_response(
                &crate::types::ProviderResponse {
                    status: response.status().as_u16(),
                    headers: header_record.unwrap_or_default(),
                },
                model,
            )
            .await;
    }

    // `start` marks an empty initial message; deltas build content.
    let mut start_partial = state.output.clone();
    start_partial.content = Vec::new();
    sender
        .push(AssistantMessageEvent::Start {
            partial: start_partial,
        })
        .await;

    let metrics = options.base.base.request_metrics.clone();
    let mut event_stream = response.bytes_stream().map(move |result| {
        if result.is_ok() && let Some(metrics) = &metrics { metrics.frame(); }
        result
    }).eventsource().map(|result| {
        result
            .map(|event| SseEvent {
                event: event.event,
                data: event.data,
            })
            .map_err(crate::api::http::sse_error)
    });

    loop {
        let frame = tokio::select! {
            biased;
            _ = token.cancelled() => return Err(StreamFailure::aborted()),
            frame = event_stream.next() => match frame {
                Some(frame) => frame,
                None => break,
            },
        };
        let frame = frame.map_err(|error| {
            StreamFailure::error(crate::error::format_provider_error(&error, None))
        })?;
        let events = process_stream_frame(state, model, &frame)?;
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&frame.data) {
            if let Some(raw) = value.get("usage").or_else(|| value.get("message").and_then(|m| m.get("usage"))).filter(|v|!v.is_null()) { sender.report_usage(&state.output.usage, raw); }
        }
        for event in events {
            if !sender.push(event).await {
                return Ok(());
            }
        }
    }

    if token.is_cancelled() {
        return Err(StreamFailure::aborted());
    }
    if state.saw_message_start && !state.saw_message_end {
        return Err(StreamFailure::error(
            "Anthropic stream ended before message_stop",
        ));
    }
    if state.output.stop_reason == StopReason::Pending {
        return Err(StreamFailure::error(
            "Anthropic stream ended without a stop reason",
        ));
    }
    if matches!(
        state.output.stop_reason,
        StopReason::Error | StopReason::Aborted
    ) {
        let message = state
            .output
            .error_message
            .clone()
            .unwrap_or_else(|| "An unknown error occurred".to_owned());
        return Err(StreamFailure::error(message));
    }
    Ok(())
}

/// Convert a reqwest header map to the `ProviderHeaders` shape (all present).
fn headers_to_provider(headers: &reqwest::header::HeaderMap) -> ProviderHeaders {
    headers
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.as_str().to_owned(), Some(value.to_owned())))
        })
        .collect()
}

/// Run the adapter: build the initial message, drive the stream, and push the
/// terminal `Done`/`Error` event.
async fn run_stream(
    model: &Model,
    context: &Context,
    options: &AnthropicOptions,
    mut sender: crate::event::AssistantMessageEventSender,
) {
    let mut state = StreamState::new(model);
    state.snapshot_mode = options.base.event_snapshot_mode;
    let result = drive_stream(model, context, options, &mut state, &mut sender).await;
    match result {
        Ok(()) => {
            let reason = done_reason(state.output.stop_reason);
            sender
                .push(AssistantMessageEvent::Done {
                    reason,
                    message: state.output,
                })
                .await;
        }
        Err(failure) => {
            state.output.stop_reason = if failure.aborted {
                StopReason::Aborted
            } else {
                StopReason::Error
            };
            state.output.error_message = Some(failure.message);
            let reason = if failure.aborted {
                crate::event::ErrorReason::Aborted
            } else {
                crate::event::ErrorReason::Error
            };
            sender
                .push(AssistantMessageEvent::Error {
                    reason,
                    error: state.output,
                })
                .await;
        }
    }
}

/// Stream an Anthropic Messages completion.
pub fn stream(
    model: &Model,
    context: &Context,
    options: &AnthropicOptions,
) -> AssistantMessageEventStream {
    let (mut sender, stream) = assistant_message_event_stream();
    let model = model.clone();
    let context = context.clone();
    let options = options.clone();
    crate::execution_timing::spawn(async move {
        let mut options = options;
        let caller = options.base.base.token();
        let consumer = sender.cancellation_token();
        let cancellation = crate::utils::combine_abort_tokens(&[&caller, &consumer]);
        options.base.base.signal = Some(cancellation.token.clone());
        sender.set_signal(cancellation.token.clone());
        sender.set_metrics(options.base.base.request_metrics.clone());

        run_stream(&model, &context, &options, sender).await;
    });
    stream
}

/// Map a [`crate::types::ThinkingLevel`] to an Anthropic effort, honoring an
/// explicit `thinking_level_map` entry before the level-based default.
fn map_thinking_level_to_effort(
    model: &Model,
    level: crate::types::ThinkingLevel,
) -> AnthropicEffort {
    let model_level = match level {
        crate::types::ThinkingLevel::Minimal => ModelThinkingLevel::Minimal,
        crate::types::ThinkingLevel::Low => ModelThinkingLevel::Low,
        crate::types::ThinkingLevel::Medium => ModelThinkingLevel::Medium,
        crate::types::ThinkingLevel::High => ModelThinkingLevel::High,
        crate::types::ThinkingLevel::Xhigh => ModelThinkingLevel::Xhigh,
        crate::types::ThinkingLevel::Max => ModelThinkingLevel::Max,
    };
    if let Some(mapped) = model
        .thinking_level_map
        .as_ref()
        .and_then(|map| map.get(&model_level))
        .and_then(|mapped| mapped.as_deref())
        .and_then(AnthropicEffort::from_str)
    {
        return mapped;
    }
    match level {
        crate::types::ThinkingLevel::Minimal | crate::types::ThinkingLevel::Low => {
            AnthropicEffort::Low
        }
        crate::types::ThinkingLevel::Medium => AnthropicEffort::Medium,
        crate::types::ThinkingLevel::High
        | crate::types::ThinkingLevel::Xhigh
        | crate::types::ThinkingLevel::Max => AnthropicEffort::High,
    }
}

/// Map simple (reasoning-level) options onto [`AnthropicOptions`], mirroring the
/// TypeScript `streamSimple` branch logic. Split out so it is unit-testable
/// without a network.
fn map_simple_options(
    model: &Model,
    context: &Context,
    options: &SimpleStreamOptions,
) -> AnthropicOptions {
    let base = build_base_options(
        model,
        context,
        Some(options),
        options.base.base.api_key.as_deref(),
    );
    let mut anthropic_options = AnthropicOptions {
        base,
        tool_choice: options.tool_choice.map(AnthropicToolChoice::from),
        ..Default::default()
    };

    let Some(reasoning) = options.reasoning else {
        anthropic_options.thinking_enabled = Some(false);
        return anthropic_options;
    };

    let compat = get_anthropic_compat(model);
    if compat.force_adaptive_thinking {
        anthropic_options.thinking_enabled = Some(true);
        anthropic_options.effort = Some(map_thinking_level_to_effort(model, reasoning));
        return anthropic_options;
    }

    let (adjusted_max_tokens, thinking_budget) = adjust_max_tokens_for_thinking(
        anthropic_options.base.max_tokens,
        model.max_tokens,
        reasoning,
        options.thinking_budgets.as_ref(),
    );
    let max_tokens = clamp_max_tokens_to_context(model, context, adjusted_max_tokens);
    anthropic_options.base.max_tokens = Some(max_tokens);
    anthropic_options.thinking_enabled = Some(true);
    anthropic_options.thinking_budget_tokens =
        Some(thinking_budget.min(max_tokens.saturating_sub(1024)));
    anthropic_options
}

/// Stream with simple (reasoning-level) options mapped onto [`AnthropicOptions`].
pub fn stream_simple(
    model: &Model,
    context: &Context,
    options: &SimpleStreamOptions,
) -> AssistantMessageEventStream {
    let anthropic_options = map_simple_options(model, context, options);
    stream(model, context, &anthropic_options)
}

// ---- provider registration -------------------------------------------------

/// The Anthropic Messages [`ProviderStreams`] implementation.
pub struct AnthropicMessagesApi;

impl ProviderStreams for AnthropicMessagesApi {
    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: &StreamOptions,
    ) -> AssistantMessageEventStream {
        let anthropic_options = AnthropicOptions {
            base: options.clone(),
            ..Default::default()
        };
        stream(model, context, &anthropic_options)
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        stream_simple(model, context, options)
    }
}

/// Shared [`AnthropicMessagesApi`] handle.
pub fn anthropic_messages_api() -> AnthropicMessagesApi {
    AnthropicMessagesApi
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::http::SseEvent;
    use crate::types::{
        ApiCompat, AssistantMessage, CacheRetention, InputModality, Message, ModelCost,
        ModelCostRates, ProviderId, SimpleStreamOptions, StopReason, StreamOptions,
        ThinkingBudgets, ThinkingLevel, ToolChoice, ToolResultMessage, Usage, UserContent,
        UserMessage,
    };
    use serde_json::{Value, json};
    use std::collections::HashMap;

    // ---- fixtures -----------------------------------------------------------

    fn base_model() -> Model {
        Model {
            id: "step-5-preview".to_owned(),
            name: "Step 5 Preview".to_owned(),
            api: crate::types::Api::AnthropicMessages,
            provider: ProviderId::new("step"),
            base_url: "https://api.stepfun.com/v1".to_owned(),
            reasoning: true,
            thinking_level_map: None,
            input: vec![InputModality::Text],
            cost: ModelCost::default(),
            context_window: 128_000,
            max_tokens: 8192,
            sampling_params: None,
            headers: None,
            compat: None,
        }
    }

    fn anthropic_model() -> Model {
        base_model()
    }

    fn user_message(text: &str) -> Message {
        Message::User(UserMessage {
            content: UserContent::Text(text.to_owned()),
            timestamp: 1,
        })
    }

    fn hello_context() -> Context {
        Context {
            system_prompt: None,
            messages: vec![user_message("Hello")],
            tools: None,
        }
    }

    fn sse(event: &str, data: Value) -> SseEvent {
        SseEvent {
            event: event.to_owned(),
            data: data.to_string(),
        }
    }

    fn message_start(id: &str, input: u64) -> SseEvent {
        sse(
            "message_start",
            json!({
                "type": "message_start",
                "message": {
                    "id": id,
                    "usage": {
                        "input_tokens": input,
                        "output_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    }
                }
            }),
        )
    }

    fn message_delta(stop_reason: &str, usage: Value) -> SseEvent {
        sse(
            "message_delta",
            json!({ "type": "message_delta", "delta": { "stop_reason": stop_reason }, "usage": usage }),
        )
    }

    fn message_stop() -> SseEvent {
        sse("message_stop", json!({ "type": "message_stop" }))
    }

    /// Drive the pure reducer over framed SSE events and finalize the terminal
    /// message exactly as [`drive_stream`]/[`run_stream`] would (Done vs Error).
    fn collect_frames(frames: &[SseEvent], model: &Model) -> AssistantMessage {
        let mut state = StreamState::new(model);
        let mut failure: Option<StreamFailure> = None;
        for frame in frames {
            match process_stream_frame(&mut state, model, frame) {
                Ok(_) => {}
                Err(f) => {
                    failure = Some(f);
                    break;
                }
            }
        }
        let result = if let Some(f) = failure {
            Err(f)
        } else if state.saw_message_start && !state.saw_message_end {
            Err(StreamFailure::error(
                "Anthropic stream ended before message_stop",
            ))
        } else if state.output.stop_reason == StopReason::Pending {
            Err(StreamFailure::error(
                "Anthropic stream ended without a stop reason",
            ))
        } else if matches!(
            state.output.stop_reason,
            StopReason::Error | StopReason::Aborted
        ) {
            Err(StreamFailure::error(
                state
                    .output
                    .error_message
                    .clone()
                    .unwrap_or_else(|| "An unknown error occurred".to_owned()),
            ))
        } else {
            Ok(())
        };
        match result {
            Ok(()) => state.output,
            Err(f) => {
                state.output.stop_reason = if f.aborted {
                    StopReason::Aborted
                } else {
                    StopReason::Error
                };
                state.output.error_message = Some(f.message);
                state.output
            }
        }
    }

    // ---- anthropic-sse-parsing ---------------------------------------------

    #[test]
    fn sse_rejects_malformed_final_tool_json() {
        let model = anthropic_model();
        // Raw (unparseable) SSE data: the outer JSON carries an invalid `\H`
        // escape and a control character inside `partial_json`, so the repair
        // path must run before the frame can be reduced.
        let malformed = r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"A\H\",\"text\":\"col1\tcol2\"}"}}"#;
        let frames = [
            message_start("msg_test", 12),
            sse(
                "content_block_start",
                json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": { "type": "tool_use", "id": "toolu_test", "name": "edit", "input": {} }
                }),
            ),
            SseEvent {
                event: "content_block_delta".to_owned(),
                data: malformed.to_owned(),
            },
            sse(
                "content_block_stop",
                json!({ "type": "content_block_stop", "index": 0 }),
            ),
            message_delta(
                "tool_use",
                json!({ "input_tokens": 12, "output_tokens": 5 }),
            ),
            message_stop(),
        ];
        let result = collect_frames(&frames, &model);
        assert_eq!(result.stop_reason, StopReason::Error);
        assert!(
            result
                .error_message
                .as_deref()
                .unwrap()
                .contains("Invalid tool argument JSON")
        );
    }

    #[test]
    fn sse_preserves_content_from_content_block_start() {
        let model = anthropic_model();
        let frames = [
            message_start("msg_initial_content", 12),
            sse(
                "content_block_start",
                json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "text", "text": "Initial text" } }),
            ),
            sse(
                "content_block_delta",
                json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "text_delta", "text": " plus delta" } }),
            ),
            sse(
                "content_block_stop",
                json!({ "type": "content_block_stop", "index": 0 }),
            ),
            sse(
                "content_block_start",
                json!({ "type": "content_block_start", "index": 1, "content_block": { "type": "thinking", "thinking": "Initial thinking", "signature": "initial signature" } }),
            ),
            sse(
                "content_block_delta",
                json!({ "type": "content_block_delta", "index": 1, "delta": { "type": "thinking_delta", "thinking": " plus delta" } }),
            ),
            sse(
                "content_block_delta",
                json!({ "type": "content_block_delta", "index": 1, "delta": { "type": "signature_delta", "signature": " plus delta" } }),
            ),
            sse(
                "content_block_stop",
                json!({ "type": "content_block_stop", "index": 1 }),
            ),
            message_delta(
                "end_turn",
                json!({ "input_tokens": 12, "output_tokens": 5 }),
            ),
            message_stop(),
        ];
        let result = collect_frames(&frames, &model);
        assert_eq!(result.stop_reason, StopReason::Stop);
        match &result.content[0] {
            AssistantContent::Text(text) => assert_eq!(text.text, "Initial text plus delta"),
            other => panic!("expected text block, got {other:?}"),
        }
        match &result.content[1] {
            AssistantContent::Thinking(thinking) => {
                assert_eq!(thinking.thinking, "Initial thinking plus delta");
                assert_eq!(
                    thinking.thinking_signature.as_deref(),
                    Some("initial signature plus delta")
                );
            }
            other => panic!("expected thinking block, got {other:?}"),
        }
    }

    #[test]
    fn sse_preserves_refusal_stop_details() {
        let model = anthropic_model();
        let explanation =
            "This request triggered restrictions and was blocked under the Usage Policy.";
        let frames = [
            message_start("msg_refusal", 412),
            sse(
                "message_delta",
                json!({
                    "type": "message_delta",
                    "delta": { "stop_reason": "refusal", "stop_details": { "type": "refusal", "category": "cyber", "explanation": explanation } },
                    "usage": { "input_tokens": 412, "output_tokens": 0 }
                }),
            ),
            message_stop(),
        ];
        let result = collect_frames(&frames, &model);
        assert_eq!(result.stop_reason, StopReason::Error);
        assert_eq!(result.raw_stop_reason.as_deref(), Some("refusal"));
        assert_eq!(result.error_message.as_deref(), Some(explanation));
    }

    #[test]
    fn sse_preserves_sensitive_stop_reason() {
        let model = anthropic_model();
        let frames = [
            message_start("msg_sensitive", 12),
            message_delta(
                "sensitive",
                json!({ "input_tokens": 12, "output_tokens": 0 }),
            ),
            message_stop(),
        ];
        let result = collect_frames(&frames, &model);
        assert_eq!(result.stop_reason, StopReason::Error);
        assert_eq!(result.raw_stop_reason.as_deref(), Some("sensitive"));
        assert_eq!(
            result.error_message.as_deref(),
            Some("Provider stopped with: sensitive")
        );
    }

    #[test]
    fn sse_message_delta_without_usage_is_noop() {
        let model = anthropic_model();
        let frames = [
            message_start("msg_test", 12),
            sse(
                "content_block_start",
                json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "text", "text": "" } }),
            ),
            sse(
                "content_block_delta",
                json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "text_delta", "text": "Hello" } }),
            ),
            sse(
                "content_block_stop",
                json!({ "type": "content_block_stop", "index": 0 }),
            ),
            // message_delta without a usage object.
            sse(
                "message_delta",
                json!({ "type": "message_delta", "delta": { "stop_reason": "end_turn" } }),
            ),
            message_stop(),
        ];
        let result = collect_frames(&frames, &model);
        assert_eq!(result.stop_reason, StopReason::Stop);
        assert!(result.error_message.is_none());
        match &result.content[0] {
            AssistantContent::Text(text) => assert_eq!(text.text, "Hello"),
            other => panic!("expected text block, got {other:?}"),
        }
        assert_eq!(result.usage.input, 12);
        assert_eq!(result.usage.total_tokens, 12);
    }

    #[test]
    fn sse_ignores_unknown_events_after_message_stop() {
        let model = anthropic_model();
        let frames = [
            message_start("msg_test", 12),
            sse(
                "content_block_start",
                json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "text", "text": "" } }),
            ),
            sse(
                "content_block_delta",
                json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "text_delta", "text": "Hello" } }),
            ),
            sse(
                "content_block_stop",
                json!({ "type": "content_block_stop", "index": 0 }),
            ),
            message_delta(
                "end_turn",
                json!({ "input_tokens": 12, "output_tokens": 5 }),
            ),
            message_stop(),
            sse("done", json!("[DONE]")),
            sse("proxy.stats", json!("not json")),
            sse("ping", json!({})),
        ];
        let result = collect_frames(&frames, &model);
        assert_eq!(result.stop_reason, StopReason::Stop);
        assert!(result.error_message.is_none());
        match &result.content[0] {
            AssistantContent::Text(text) => assert_eq!(text.text, "Hello"),
            other => panic!("expected text block, got {other:?}"),
        }
    }

    #[test]
    fn sse_stream_ended_before_message_stop_errors() {
        let model = anthropic_model();
        let frames = [
            message_start("msg_test", 12),
            message_delta(
                "end_turn",
                json!({ "input_tokens": 12, "output_tokens": 5 }),
            ),
            // no message_stop
        ];
        let result = collect_frames(&frames, &model);
        assert_eq!(result.stop_reason, StopReason::Error);
        assert_eq!(
            result.error_message.as_deref(),
            Some("Anthropic stream ended before message_stop")
        );
    }

    // ---- anthropic-cache-write-1h-cost -------------------------------------

    fn cache_creation_model() -> Model {
        let mut model = anthropic_model();
        model.cost = ModelCost {
            rates: ModelCostRates {
                input: 5.0,
                output: 25.0,
                cache_read: 0.5,
                cache_write: 6.25,
            },
            tiers: Vec::new(),
        };
        model
    }

    fn cache_creation_frames(cache_creation: Option<Value>) -> Vec<SseEvent> {
        let mut start_usage = json!({
            "input_tokens": 100,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 1_000_000,
        });
        if let Some(cache_creation) = cache_creation {
            start_usage
                .as_object_mut()
                .expect("object")
                .insert("cache_creation".to_owned(), cache_creation);
        }
        vec![
            sse(
                "message_start",
                json!({ "type": "message_start", "message": { "id": "msg_test", "usage": start_usage } }),
            ),
            sse(
                "content_block_start",
                json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "text", "text": "" } }),
            ),
            sse(
                "content_block_delta",
                json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "text_delta", "text": "Hi" } }),
            ),
            sse(
                "content_block_stop",
                json!({ "type": "content_block_stop", "index": 0 }),
            ),
            message_delta(
                "end_turn",
                json!({ "input_tokens": 100, "output_tokens": 5, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 1_000_000 }),
            ),
            message_stop(),
        ]
    }

    #[test]
    fn cache_write_1h_priced_at_2x_input() {
        let model = cache_creation_model();
        let frames = cache_creation_frames(Some(json!({
            "ephemeral_5m_input_tokens": 600_000,
            "ephemeral_1h_input_tokens": 400_000,
        })));
        let result = collect_frames(&frames, &model);
        assert_eq!(result.usage.cache_write, 1_000_000);
        assert_eq!(result.usage.cache_write_1h, Some(400_000));
        // 600k * 6.25/Mtok + 400k * 10/Mtok = 3.75 + 4.0 = 7.75
        assert!(
            (result.usage.cost.cache_write - 7.75).abs() < 1e-9,
            "got {}",
            result.usage.cost.cache_write
        );
    }

    #[test]
    fn cache_write_falls_back_to_5m_rate_without_breakdown() {
        let model = cache_creation_model();
        let frames = cache_creation_frames(None);
        let result = collect_frames(&frames, &model);
        assert_eq!(result.usage.cache_write, 1_000_000);
        assert_eq!(result.usage.cache_write_1h.unwrap_or(0), 0);
        // 1M * 6.25/Mtok = 6.25
        assert!(
            (result.usage.cost.cache_write - 6.25).abs() < 1e-9,
            "got {}",
            result.usage.cost.cache_write
        );
    }

    // ---- build_params helpers ----------------------------------------------

    fn build(options: &AnthropicOptions) -> Value {
        build_params(&anthropic_model(), &hello_context(), options).expect("build_params succeeds")
    }

    fn simple_options(
        temperature: Option<f64>,
        reasoning: Option<ThinkingLevel>,
    ) -> SimpleStreamOptions {
        let mut base = StreamOptions {
            temperature,
            ..Default::default()
        };
        base.base.api_key = Some("fake-key".to_owned());
        SimpleStreamOptions {
            base,
            tool_choice: None,
            reasoning,
            deferred: None,
            thinking_budgets: None,
        }
    }

    // ---- anthropic-temperature-compat --------------------------------------

    #[test]
    fn temperature_forwarded_by_default() {
        let options = map_simple_options(
            &anthropic_model(),
            &hello_context(),
            &simple_options(Some(0.0), None),
        );
        let payload = build(&options);
        assert_eq!(payload.get("temperature"), Some(&json!(0.0)));
    }

    #[test]
    fn temperature_omitted_when_unsupported() {
        let mut model = base_model();
        model.id = "vendor--claude-opus-4-7".to_owned();
        model.provider = ProviderId::new("vendor-proxy");
        model.base_url = "http://127.0.0.1:9".to_owned();
        model.context_window = 200_000;
        model.max_tokens = 32_000;
        model.compat = Some(ApiCompat {
            supports_temperature: Some(false),
            ..Default::default()
        });
        let options =
            map_simple_options(&model, &hello_context(), &simple_options(Some(1.0), None));
        let payload =
            build_params(&model, &hello_context(), &options).expect("build_params succeeds");
        assert!(payload.get("temperature").is_none());
    }

    // ---- anthropic-force-adaptive-thinking ---------------------------------

    fn custom_proxy_model(compat: Option<ApiCompat>) -> Model {
        let mut model = base_model();
        model.id = "vendor--claude-opus-latest".to_owned();
        model.provider = ProviderId::new("vendor-proxy");
        model.base_url = "http://127.0.0.1:9".to_owned();
        model.context_window = 200_000;
        model.max_tokens = 32_000;
        model.compat = compat;
        model
    }

    fn adaptive_payload(model: &Model, reasoning: Option<ThinkingLevel>) -> Value {
        let options = map_simple_options(model, &hello_context(), &simple_options(None, reasoning));
        build_params(model, &hello_context(), &options).expect("build_params succeeds")
    }

    #[test]
    fn force_adaptive_off_uses_budget_thinking() {
        let model = custom_proxy_model(None);
        let payload = adaptive_payload(&model, Some(ThinkingLevel::Medium));
        assert_eq!(payload["thinking"]["type"], json!("enabled"));
        assert!(payload.get("output_config").is_none());
    }

    #[test]
    fn force_adaptive_on_uses_adaptive_thinking() {
        let model = custom_proxy_model(Some(ApiCompat {
            force_adaptive_thinking: Some(true),
            ..Default::default()
        }));
        let payload = adaptive_payload(&model, Some(ThinkingLevel::Medium));
        assert_eq!(
            payload["thinking"],
            json!({ "type": "adaptive", "display": "summarized" })
        );
        assert_eq!(payload["output_config"], json!({ "effort": "medium" }));
    }

    #[test]
    fn force_adaptive_uses_native_xhigh_for_fable() {
        let mut model = anthropic_model();
        model.compat = Some(ApiCompat {
            force_adaptive_thinking: Some(true),
            ..Default::default()
        });
        let mut level_map: crate::types::ThinkingLevelMap = HashMap::new();
        level_map.insert(ModelThinkingLevel::Off, None);
        level_map.insert(ModelThinkingLevel::Xhigh, Some("xhigh".to_owned()));
        level_map.insert(ModelThinkingLevel::Max, Some("max".to_owned()));
        model.thinking_level_map = Some(level_map);
        let payload = adaptive_payload(&model, Some(ThinkingLevel::Xhigh));
        assert_eq!(
            payload["thinking"],
            json!({ "type": "adaptive", "display": "summarized" })
        );
        assert_eq!(payload["output_config"], json!({ "effort": "xhigh" }));
    }

    #[test]
    fn force_adaptive_false_opts_out() {
        let mut model = anthropic_model();
        model.compat = Some(ApiCompat {
            force_adaptive_thinking: Some(false),
            ..Default::default()
        });
        let payload = adaptive_payload(&model, Some(ThinkingLevel::Medium));
        assert_eq!(payload["thinking"]["type"], json!("enabled"));
        assert!(payload.get("output_config").is_none());
    }

    #[test]
    fn force_adaptive_off_reasoning_sends_disabled() {
        let model = custom_proxy_model(Some(ApiCompat {
            force_adaptive_thinking: Some(true),
            ..Default::default()
        }));
        let payload = adaptive_payload(&model, None);
        assert_eq!(payload["thinking"], json!({ "type": "disabled" }));
        assert!(payload.get("output_config").is_none());
    }

    // ---- anthropic-empty-thinking-signature-compat -------------------------

    fn signature_model(allow_empty_signature: Option<bool>) -> Model {
        let mut model = base_model();
        model.id = "mimo-v2.5-pro".to_owned();
        model.provider = ProviderId::new("xiaomi-token-plan-ams");
        model.base_url = "http://127.0.0.1:9/anthropic".to_owned();
        model.context_window = 1_048_576;
        model.max_tokens = 1024;
        model.compat = allow_empty_signature.map(|value| ApiCompat {
            allow_empty_signature: Some(value),
            ..Default::default()
        });
        model
    }

    fn signature_context(signature: &str, thinking: &str) -> Context {
        let assistant = AssistantMessage {
            content: vec![AssistantContent::Thinking(ThinkingContent {
                thinking: thinking.to_owned(),
                thinking_signature: Some(signature.to_owned()),
                redacted: false,
            })],
            api: crate::types::Api::AnthropicMessages,
            provider: ProviderId::new("xiaomi-token-plan-ams"),
            model: "mimo-v2.5-pro".to_owned(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            timestamp: 1,
        };
        Context {
            system_prompt: None,
            messages: vec![
                user_message("first"),
                Message::Assistant(Box::new(assistant)),
                user_message("second"),
            ],
            tools: None,
        }
    }

    fn assistant_content(payload: &Value) -> Value {
        payload["messages"]
            .as_array()
            .expect("messages array")
            .iter()
            .find(|message| message["role"] == "assistant")
            .expect("assistant message")["content"]
            .clone()
    }

    #[test]
    fn empty_signature_thinking_becomes_text_by_default() {
        let model = signature_model(None);
        let context = signature_context("", "internal reasoning");
        let options = AnthropicOptions::default();
        let payload = build_params(&model, &context, &options).expect("build_params succeeds");
        assert_eq!(
            assistant_content(&payload),
            json!([{ "type": "text", "text": "internal reasoning" }])
        );
    }

    #[test]
    fn empty_thinking_text_preserved_with_signature() {
        let model = signature_model(None);
        let context = signature_context("signed-thinking", "");
        let options = AnthropicOptions::default();
        let payload = build_params(&model, &context, &options).expect("build_params succeeds");
        assert_eq!(
            assistant_content(&payload),
            json!([{ "type": "thinking", "thinking": "", "signature": "signed-thinking" }])
        );
    }

    #[test]
    fn empty_signature_preserved_when_allowed() {
        let model = signature_model(Some(true));
        let context = signature_context(" ", "internal reasoning");
        let options = AnthropicOptions::default();
        let payload = build_params(&model, &context, &options).expect("build_params succeeds");
        assert_eq!(
            assistant_content(&payload),
            json!([{ "type": "thinking", "thinking": "internal reasoning", "signature": "" }])
        );
    }

    // ---- anthropic-eager-tool-input-compat ---------------------------------

    fn eager_model(compat: Option<ApiCompat>) -> Model {
        let mut model = base_model();
        model.id = "claude-opus-4-8".to_owned();
        model.provider = ProviderId::new("test-anthropic");
        model.base_url = "http://127.0.0.1:9".to_owned();
        model.context_window = 200_000;
        model.max_tokens = 32_000;
        model.compat = Some(ApiCompat {
            force_adaptive_thinking: Some(true),
            ..compat.unwrap_or_default()
        });
        model
    }

    fn tool_context(parameters: Value) -> Context {
        Context {
            system_prompt: None,
            messages: vec![user_message("Use the tool")],
            tools: Some(vec![crate::types::Tool {
                name: "lookup".to_owned(),
                description: "Look up a value".to_owned(),
                parameters,
                constrained_sampling: None,
            }]),
        }
    }

    fn eager_options() -> AnthropicOptions {
        let mut base = StreamOptions {
            cache_retention: Some(CacheRetention::None),
            ..Default::default()
        };
        base.base.api_key = Some("test-key".to_owned());
        AnthropicOptions {
            base,
            ..Default::default()
        }
    }

    fn beta_header(headers: &reqwest::header::HeaderMap) -> Option<String> {
        headers
            .get("anthropic-beta")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned)
    }

    #[test]
    fn eager_tool_input_streaming_by_default() {
        let model = eager_model(None);
        let context = tool_context(
            json!({ "type": "object", "properties": { "value": { "type": "string" } }, "required": ["value"] }),
        );
        let options = eager_options();
        let payload = build_params(&model, &context, &options).expect("build_params succeeds");
        let tool = &payload["tools"][0];
        assert_eq!(tool["eager_input_streaming"], json!(true));
        let headers = build_request_headers(&model, &options, &context).expect("headers");
        assert_eq!(beta_header(&headers), None);
    }

    #[test]
    fn eager_disabled_uses_fine_grained_beta() {
        let model = eager_model(Some(ApiCompat {
            supports_eager_tool_input_streaming: Some(false),
            ..Default::default()
        }));
        let context = tool_context(
            json!({ "type": "object", "properties": { "value": { "type": "string" } }, "required": ["value"] }),
        );
        let options = eager_options();
        let payload = build_params(&model, &context, &options).expect("build_params succeeds");
        assert!(payload["tools"][0].get("eager_input_streaming").is_none());
        let headers = build_request_headers(&model, &options, &context).expect("headers");
        assert_eq!(
            beta_header(&headers).as_deref(),
            Some("fine-grained-tool-streaming-2025-05-14")
        );
    }

    #[test]
    fn no_fine_grained_beta_without_tools() {
        let model = eager_model(Some(ApiCompat {
            supports_eager_tool_input_streaming: Some(false),
            ..Default::default()
        }));
        let context = Context {
            system_prompt: None,
            messages: vec![user_message("Use the tool")],
            tools: None,
        };
        let options = eager_options();
        let payload = build_params(&model, &context, &options).expect("build_params succeeds");
        assert!(payload.get("tools").is_none());
        let headers = build_request_headers(&model, &options, &context).expect("headers");
        assert_eq!(beta_header(&headers), None);
    }

    #[test]
    fn strict_tools_send_full_schema_legacy_keeps_shape() {
        let model = eager_model(Some(ApiCompat {
            supports_strict_tools: Some(true),
            ..Default::default()
        }));
        let context = tool_context(json!({
            "type": "object",
            "properties": { "value": { "type": "string" } },
            "required": ["value"],
            "additionalProperties": false,
            "title": "LookupInput",
        }));
        let options = eager_options();
        let payload = build_params(&model, &context, &options).expect("build_params succeeds");
        assert_eq!(
            payload["tools"][0]["input_schema"],
            json!({ "type": "object", "properties": { "value": { "type": "string" } }, "required": ["value"] })
        );
    }

    #[test]
    fn strict_json_schema_tool_gets_strict_flag_and_schema() {
        let model = eager_model(Some(ApiCompat {
            supports_strict_tools: Some(true),
            ..Default::default()
        }));
        let mut context = tool_context(json!({
            "type": "object",
            "properties": { "value": { "type": "string" }, "optional": { "type": "number" } },
            "required": ["value"],
            "title": "StrictLookupInput",
        }));
        if let Some(tools) = context.tools.as_mut() {
            tools[0].constrained_sampling =
                Some(crate::types::ConstrainedSamplingConfig::JsonSchema {
                    strict: crate::types::StrictMode::Prefer,
                });
        }
        let options = eager_options();
        let payload = build_params(&model, &context, &options).expect("build_params succeeds");
        let tool = &payload["tools"][0];
        assert_eq!(tool["strict"], json!(true));
        let schema = &tool["input_schema"];
        assert_eq!(schema["additionalProperties"], json!(false));
        assert_eq!(schema["required"], json!(["value", "optional"]));
        assert_eq!(
            schema["properties"]["optional"],
            json!({ "anyOf": [{ "type": "number" }, { "type": "null" }] })
        );
        assert_eq!(schema["title"], json!("StrictLookupInput"));
    }

    #[test]
    fn legacy_input_schema_keeps_local_definitions() {
        let model = eager_model(None);
        let context = tool_context(json!({
            "type": "object",
            "properties": {
                "priority": { "$ref": "#/$defs/Priority" },
                "owner": { "$ref": "#/definitions/Owner" },
            },
            "required": ["priority"],
            "$defs": { "Priority": { "type": "string", "enum": ["low", "high"] } },
            "definitions": { "Owner": { "type": "object", "properties": { "name": { "type": "string" } } } },
        }));
        let options = eager_options();
        let payload = build_params(&model, &context, &options).expect("build_params succeeds");
        let schema = &payload["tools"][0]["input_schema"];
        assert_eq!(schema["type"], json!("object"));
        assert_eq!(schema["required"], json!(["priority"]));
        assert_eq!(
            schema["$defs"],
            json!({ "Priority": { "type": "string", "enum": ["low", "high"] } })
        );
        assert_eq!(
            schema["definitions"],
            json!({ "Owner": { "type": "object", "properties": { "name": { "type": "string" } } } })
        );
    }

    // ---- anthropic-tool-result-empty-text ----------------------------------

    fn tool_result_context(content: Vec<UserContentBlock>) -> Context {
        let mut model = anthropic_model();
        model.input = vec![InputModality::Text, InputModality::Image];
        model.base_url = "http://127.0.0.1:9".to_owned();
        let _ = &model;
        let assistant = AssistantMessage {
            content: vec![AssistantContent::ToolCall(ToolCall {
                id: "call_1".to_owned(),
                name: "screenshot".to_owned(),
                arguments: Default::default(),
                thought_signature: None,
                namespace: None,
            })],
            api: crate::types::Api::AnthropicMessages,
            provider: ProviderId::new("anthropic"),
            model: "claude-opus-4-6".to_owned(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: Usage::default(),
            stop_reason: StopReason::ToolUse,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            timestamp: 2,
        };
        let tool_result = ToolResultMessage {
            tool_call_id: "call_1".to_owned(),
            tool_name: "screenshot".to_owned(),
            content,
            details: None,
            usage: None,
            added_tool_names: None,
            is_error: false,
            timestamp: 3,
        };
        Context {
            system_prompt: None,
            messages: vec![
                user_message("Look"),
                Message::Assistant(Box::new(assistant)),
                Message::ToolResult(Box::new(tool_result)),
            ],
            tools: None,
        }
    }

    fn tool_result_image_model() -> Model {
        let mut model = anthropic_model();
        model.input = vec![InputModality::Text, InputModality::Image];
        model.base_url = "http://127.0.0.1:9".to_owned();
        model
    }

    fn tool_result_block_content(payload: &Value) -> Value {
        for message in payload["messages"].as_array().expect("messages") {
            if let Some(blocks) = message["content"].as_array() {
                for block in blocks {
                    if block["type"] == "tool_result" {
                        return block["content"].clone();
                    }
                }
            }
        }
        panic!("no tool_result block in payload");
    }

    fn text_block(text: &str) -> UserContentBlock {
        UserContentBlock::Text(TextContent {
            text: text.to_owned(),
            text_signature: None,
        })
    }

    #[test]
    fn tool_result_drops_whitespace_only_text_blocks() {
        let model = tool_result_image_model();
        let context = tool_result_context(vec![
            text_block("head"),
            text_block(""),
            text_block(" \n"),
            text_block("[notice]"),
            UserContentBlock::Image(crate::types::ImageContent {
                data: "AAAA".to_owned(),
                mime_type: "image/png".to_owned(),
            }),
        ]);
        let options = AnthropicOptions::default();
        let payload = build_params(&model, &context, &options).expect("build_params succeeds");
        let content = tool_result_block_content(&payload);
        let types: Vec<&str> = content
            .as_array()
            .expect("array")
            .iter()
            .map(|block| block["type"].as_str().expect("type"))
            .collect();
        assert_eq!(types, vec!["text", "text", "image"]);
        let texts: Vec<&str> = content
            .as_array()
            .unwrap()
            .iter()
            .filter(|block| block["type"] == "text")
            .map(|block| block["text"].as_str().unwrap())
            .collect();
        assert_eq!(texts, vec!["head", "[notice]"]);
    }

    #[test]
    fn tool_result_keeps_image_placeholder_when_text_empty() {
        let model = tool_result_image_model();
        let context = tool_result_context(vec![
            text_block(""),
            UserContentBlock::Image(crate::types::ImageContent {
                data: "AAAA".to_owned(),
                mime_type: "image/png".to_owned(),
            }),
        ]);
        let options = AnthropicOptions::default();
        let payload = build_params(&model, &context, &options).expect("build_params succeeds");
        let content = tool_result_block_content(&payload);
        assert_eq!(
            content[0],
            json!({ "type": "text", "text": "(see attached image)" })
        );
        assert_eq!(content[1]["type"], json!("image"));
        assert_eq!(content[1]["source"]["type"], json!("base64"));
        assert_eq!(content[1]["source"]["media_type"], json!("image/png"));
        assert_eq!(content[1]["source"]["data"], json!("AAAA"));
    }

    // ---- normalize_tool_call_id --------------------------------------------

    #[test]
    fn tool_call_id_normalized_to_anthropic_pattern() {
        assert_eq!(normalize_tool_call_id("call_abc-123"), "call_abc-123");
        assert_eq!(
            normalize_tool_call_id("call with spaces!"),
            "call_with_spaces_"
        );
        assert_eq!(normalize_tool_call_id(&"x".repeat(100)), "x".repeat(64));
        assert_eq!(normalize_tool_call_id("!!!"), "___");
    }

    // ---- provider registration ---------------------------------------------

    #[tokio::test]
    async fn provider_streams_adapts_generic_options() {
        let api = anthropic_messages_api();
        // `stream` with plain StreamOptions should not panic; it produces a stream
        // that terminates with an auth error (no API key configured).
        let model = anthropic_model();
        let options = StreamOptions::default();
        let mut stream = api.stream(&model, &hello_context(), &options);
        let result = stream.result().await.unwrap();
        assert_eq!(result.stop_reason, StopReason::Error);
        assert!(
            result
                .error_message
                .is_some_and(|message| message.contains("No API key"))
        );
    }

    #[test]
    fn tool_choice_serializes_wire_shapes() {
        assert_eq!(
            AnthropicToolChoice::Auto.to_value(),
            json!({ "type": "auto" })
        );
        assert_eq!(
            AnthropicToolChoice::Any.to_value(),
            json!({ "type": "any" })
        );
        assert_eq!(
            AnthropicToolChoice::None.to_value(),
            json!({ "type": "none" })
        );
        assert_eq!(
            AnthropicToolChoice::Tool {
                name: "edit".to_owned()
            }
            .to_value(),
            json!({ "type": "tool", "name": "edit" })
        );
        assert_eq!(
            AnthropicToolChoice::from(ToolChoice::Auto),
            AnthropicToolChoice::Auto
        );
        assert_eq!(
            AnthropicToolChoice::from(ToolChoice::None),
            AnthropicToolChoice::None
        );
    }

    #[test]
    fn default_supports_tool_references_matches_regex() {
        let mut model = base_model();
        model.provider = ProviderId::new("anthropic");
        // Claude 4.5+ supports tool references.
        model.id = "claude-sonnet-4-5-20250929".to_owned();
        assert!(default_supports_tool_references(&model));
        // Claude 4.0 does not.
        model.id = "claude-opus-4-0".to_owned();
        assert!(!default_supports_tool_references(&model));
        // Claude 4.1 does not.
        model.id = "claude-opus-4-1".to_owned();
        assert!(!default_supports_tool_references(&model));
        // Claude 4.5 does.
        model.id = "claude-opus-4-5".to_owned();
        assert!(default_supports_tool_references(&model));
        // Claude 5 does.
        model.id = "claude-sonnet-5".to_owned();
        assert!(default_supports_tool_references(&model));
        // Haiku never.
        model.id = "claude-haiku-4-5".to_owned();
        assert!(!default_supports_tool_references(&model));
        // Non-anthropic provider never.
        model.id = "claude-opus-4-5".to_owned();
        model.provider = ProviderId::new("vendor");
        assert!(!default_supports_tool_references(&model));
    }

    #[test]
    fn thinking_budgets_flow_through_simple_options() {
        let mut model = anthropic_model();
        // Large ceilings so the custom 20k budget fits without answer-room clamping.
        model.max_tokens = 100_000;
        model.context_window = 200_000;
        let mut options = simple_options(None, Some(ThinkingLevel::High));
        options.thinking_budgets = Some(ThinkingBudgets {
            high: Some(20_000),
            ..Default::default()
        });
        let mapped = map_simple_options(&model, &hello_context(), &options);
        assert_eq!(mapped.thinking_enabled, Some(true));
        assert_eq!(mapped.thinking_budget_tokens, Some(20_000));
    }
}
