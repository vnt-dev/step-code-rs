//! OpenAI Responses API adapter: request assembly (params, headers, retry) and
//! the `stream`/`stream_simple` entry points.
//!
//! Ported from `packages/providers/src/api/openai-responses.ts`.

use std::collections::HashMap;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Value, json};

use crate::api::constrained_sampling::create_grammar_tool_input_properties;
use crate::api::copilot_headers::{build_copilot_dynamic_headers, has_copilot_vision_input};
use crate::api::http::{http_client_for};
use crate::api::openai_prompt_cache::clamp_openai_prompt_cache_key;
use crate::api::openai_responses_shared::{
    ConvertResponsesMessagesOptions, ConvertResponsesToolsOptions, DeferredToolsMode,
    ResponsesStreamError, ResponsesStreamOptions, convert_responses_messages,
    convert_responses_tools, process_responses_stream,
};
use crate::api::simple_options::build_base_options;
use crate::error::{ProviderHttpError, format_provider_error};
use crate::event::{
    AssistantMessageEvent, AssistantMessageEventSender, AssistantMessageEventStream, DoneReason,
    ErrorReason, assistant_message_event_stream,
};
use crate::models::{ProviderStreams, clamp_thinking_level};
use crate::types::{
    AssistantMessage, CacheRetention, Context, Model, ModelThinkingLevel, ProviderHeaders,
    ProviderResponse, SessionAffinityFormat, SimpleStreamOptions, StopReason, StreamOptions,
    ThinkingLevel, Usage,
};
use crate::utils::deferred_tools::split_deferred_tools;
use crate::utils::provider_retry::{ProviderRetryOptions, retry_provider_request_tracked};
use crate::utils::user_agent::get_user_agent;

/// Providers whose composite `call_id|item_id` tool-call ids replay verbatim.
const OPENAI_TOOL_CALL_PROVIDERS: &[&str] = &["openai", "openai-codex", "opencode"];

/// OpenAI Responses rejects `max_output_tokens` below 16.
/// https://github.com/earendil-works/pi/issues/6265
const OPENAI_RESPONSES_MIN_OUTPUT_TOKENS: u64 = 16;

/// The prefix provider error messages are composed with.
const PROVIDER_ERROR_PREFIX: &str = "OpenAI API error";

// =============================================================================
// Options
// =============================================================================

/// Reasoning effort accepted by the Responses API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReasoningEffort {
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl ReasoningEffort {
    /// The wire value.
    pub fn wire(self) -> &'static str {
        match self {
            ReasoningEffort::Minimal => "minimal",
            ReasoningEffort::Low => "low",
            ReasoningEffort::Medium => "medium",
            ReasoningEffort::High => "high",
            ReasoningEffort::Xhigh => "xhigh",
            ReasoningEffort::Max => "max",
        }
    }

    fn from_thinking_level(level: ThinkingLevel) -> Self {
        match level {
            ThinkingLevel::Minimal => ReasoningEffort::Minimal,
            ThinkingLevel::Low => ReasoningEffort::Low,
            ThinkingLevel::Medium => ReasoningEffort::Medium,
            ThinkingLevel::High => ReasoningEffort::High,
            ThinkingLevel::Xhigh => ReasoningEffort::Xhigh,
            ThinkingLevel::Max => ReasoningEffort::Max,
        }
    }

    fn to_model_thinking_level(self) -> ModelThinkingLevel {
        match self {
            ReasoningEffort::Minimal => ModelThinkingLevel::Minimal,
            ReasoningEffort::Low => ModelThinkingLevel::Low,
            ReasoningEffort::Medium => ModelThinkingLevel::Medium,
            ReasoningEffort::High => ModelThinkingLevel::High,
            ReasoningEffort::Xhigh => ModelThinkingLevel::Xhigh,
            ReasoningEffort::Max => ModelThinkingLevel::Max,
        }
    }
}

/// Reasoning summary verbosity accepted by the Responses API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReasoningSummary {
    Auto,
    Detailed,
    Concise,
}

impl ReasoningSummary {
    /// The wire value.
    pub fn wire(self) -> &'static str {
        match self {
            ReasoningSummary::Auto => "auto",
            ReasoningSummary::Detailed => "detailed",
            ReasoningSummary::Concise => "concise",
        }
    }
}

/// OpenAI Responses-specific stream options. Contains [`StreamOptions`]
/// so the base fields read and write through unchanged.
#[derive(Clone, Default)]
pub struct OpenAIResponsesOptions {
    pub base: StreamOptions,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub reasoning_summary: Option<ReasoningSummary>,
    /// Opaque `service_tier` request value (a provider-specific union).
    pub service_tier: Option<Value>,
    /// Opaque `tool_choice` request value (a provider-specific union).
    pub tool_choice: Option<Value>,
}

impl std::fmt::Debug for OpenAIResponsesOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIResponsesOptions")
            .field("base", &self.base)
            .field("reasoning_effort", &self.reasoning_effort)
            .field("reasoning_summary", &self.reasoning_summary)
            .field("service_tier", &self.service_tier)
            .field("tool_choice", &self.tool_choice)
            .finish()
    }
}

/// The resolved `OpenAIResponsesCompat` block: defaults applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResolvedCompat {
    supports_developer_role: bool,
    session_affinity_format: SessionAffinityFormat,
    supports_long_cache_retention: bool,
    supports_strict_mode: bool,
    supports_openai_grammar_tools: bool,
    supports_additional_tools: bool,
    supports_tool_search: bool,
    supports_explicit_prompt_cache_mode: bool,
}

/// Session affinity defaults to the OpenRouter format for OpenRouter endpoints.
fn detect_session_affinity_format(model: &Model) -> SessionAffinityFormat {
    if model.provider.as_str() == "openrouter" || model.base_url.contains("openrouter.ai") {
        SessionAffinityFormat::Openrouter
    } else {
        SessionAffinityFormat::Openai
    }
}

fn get_compat(model: &Model) -> ResolvedCompat {
    let compat = model.openai_responses_compat().unwrap_or_default();
    ResolvedCompat {
        supports_developer_role: compat.supports_developer_role.unwrap_or(true),
        session_affinity_format: compat
            .session_affinity_format
            .unwrap_or_else(|| detect_session_affinity_format(model)),
        supports_long_cache_retention: compat.supports_long_cache_retention.unwrap_or(true),
        supports_strict_mode: compat.supports_strict_mode.unwrap_or(false),
        supports_openai_grammar_tools: compat.supports_openai_grammar_tools.unwrap_or(false),
        supports_additional_tools: compat.supports_additional_tools.unwrap_or(false),
        supports_tool_search: compat.supports_tool_search.unwrap_or(false),
        supports_explicit_prompt_cache_mode: compat
            .supports_explicit_prompt_cache_mode
            .unwrap_or(false),
    }
}

/// `24h` retention when supported and the caller asked for long retention.
fn get_prompt_cache_retention(
    compat: ResolvedCompat,
    cache_retention: CacheRetention,
) -> Option<&'static str> {
    (cache_retention == CacheRetention::Long && compat.supports_long_cache_retention)
        .then_some("24h")
}

/// True when the thinking level map does not explicitly disable `off`.
fn supports_reasoning_off(model: &Model) -> bool {
    model
        .thinking_level_map
        .as_ref()
        .and_then(|map| map.get(&ModelThinkingLevel::Off))
        != Some(&None)
}

/// The reasoning effort `off` maps to (`thinkingLevelMap.off ?? "none"`).
fn reasoning_off_effort(model: &Model) -> String {
    model
        .thinking_level_map
        .as_ref()
        .and_then(|map| map.get(&ModelThinkingLevel::Off))
        .and_then(|value| value.clone())
        .unwrap_or_else(|| "none".to_owned())
}

/// The effort a level maps to (`thinkingLevelMap[level] ?? level`).
fn mapped_reasoning_effort(model: &Model, effort: ReasoningEffort) -> String {
    model
        .thinking_level_map
        .as_ref()
        .and_then(|map| map.get(&effort.to_model_thinking_level()))
        .and_then(|value| value.clone())
        .unwrap_or_else(|| effort.wire().to_owned())
}

fn get_service_tier_cost_multiplier(model_id: &str, service_tier: &str) -> f64 {
    match service_tier {
        "flex" => 0.5,
        "priority" => {
            if model_id == "gpt-5.5" {
                2.5
            } else {
                2.0
            }
        }
        _ => 1.0,
    }
}

/// The closest supported effort for a requested level (`clampThinkingLevel`).
fn clamp_effort(model: &Model, effort: ReasoningEffort) -> ModelThinkingLevel {
    clamp_thinking_level(model, effort.to_model_thinking_level())
}

fn model_level_to_effort(level: ModelThinkingLevel) -> Option<ReasoningEffort> {
    match level {
        ModelThinkingLevel::Off => None,
        ModelThinkingLevel::Minimal => Some(ReasoningEffort::Minimal),
        ModelThinkingLevel::Low => Some(ReasoningEffort::Low),
        ModelThinkingLevel::Medium => Some(ReasoningEffort::Medium),
        ModelThinkingLevel::High => Some(ReasoningEffort::High),
        ModelThinkingLevel::Xhigh => Some(ReasoningEffort::Xhigh),
        ModelThinkingLevel::Max => Some(ReasoningEffort::Max),
    }
}

// =============================================================================
// Errors
// =============================================================================

/// A failure while streaming: either an HTTP failure or a plain message.
#[derive(Debug)]
enum StreamError {
    Http(ProviderHttpError),
    Message(String),
}

/// Compose the user-facing message, mirroring
/// `formatProviderError(normalizeProviderError(error), "OpenAI API error")`.
fn format_stream_error(error: &StreamError) -> String {
    match error {
        StreamError::Http(http) => format_provider_error(http, Some(PROVIDER_ERROR_PREFIX)),
        StreamError::Message(message) => message.clone(),
    }
}

impl From<ProviderHttpError> for StreamError {
    fn from(error: ProviderHttpError) -> Self {
        StreamError::Http(error)
    }
}

impl From<ResponsesStreamError> for StreamError {
    fn from(error: ResponsesStreamError) -> Self {
        match error {
            ResponsesStreamError::Http(http) => StreamError::Http(http),
            ResponsesStreamError::Aborted => StreamError::Message("Request was aborted".to_owned()),
            ResponsesStreamError::Provider(message) => StreamError::Message(message),
        }
    }
}

// =============================================================================
// Request assembly
// =============================================================================

/// The Responses endpoint for a model's base URL.
fn responses_url(model: &Model) -> String {
    format!("{}/responses", model.base_url.trim_end_matches('/'))
}

/// True when a non-empty header with this (case-insensitive) name is present.
fn has_header(headers: Option<&ProviderHeaders>, name: &str) -> bool {
    headers.into_iter().flatten().any(|(key, value)| {
        key.eq_ignore_ascii_case(name)
            && value
                .as_deref()
                .is_some_and(|value| !value.trim().is_empty())
    })
}

/// Resolve the API key, or the placeholder used when the caller authenticates
/// through an `authorization`/`cf-aig-authorization` header.
fn get_client_api_key(
    provider: &str,
    api_key: Option<&str>,
    headers: Option<&ProviderHeaders>,
) -> Result<String, String> {
    if let Some(api_key) = api_key {
        return Ok(api_key.to_owned());
    }
    if has_header(headers, "authorization") || has_header(headers, "cf-aig-authorization") {
        return Ok("unused".to_owned());
    }
    Err(format!("No API key for provider: {provider}"))
}

/// Build the request headers: defaults first, then model headers, Copilot
/// dynamics, session affinity, and finally the caller's headers (which win).
fn build_headers(
    model: &Model,
    context: &Context,
    api_key: &str,
    options: &OpenAIResponsesOptions,
    cache_session_id: Option<&str>,
    compat: ResolvedCompat,
) -> Result<reqwest::header::HeaderMap, crate::utils::headers::HeaderError> {
    let mut headers = ProviderHeaders::new();
    fn insert_header(headers: &mut ProviderHeaders, name: &str, value: &str) {
        headers.retain(|key, _| !key.eq_ignore_ascii_case(name));
        headers.insert(name.to_owned(), Some(value.to_owned()));
    }
    insert_header(&mut headers, "Authorization", &format!("Bearer {api_key}"));
    insert_header(&mut headers, "User-Agent", &get_user_agent());
    if let Some(model_headers) = &model.headers {
        for (name, value) in model_headers {
            insert_header(&mut headers, name, value);
        }
    }

    if model.provider.as_str() == "github-copilot" {
        let has_images = has_copilot_vision_input(&context.messages);
        for (name, value) in build_copilot_dynamic_headers(&context.messages, has_images) {
            insert_header(&mut headers, &name, &value);
        }
    }

    if let Some(session_id) = cache_session_id {
        match compat.session_affinity_format {
            SessionAffinityFormat::Openrouter => {
                insert_header(&mut headers, "x-session-id", session_id);
            }
            _ => {
                if compat.session_affinity_format == SessionAffinityFormat::Openai {
                    insert_header(&mut headers, "session_id", session_id);
                }
                insert_header(&mut headers, "x-client-request-id", session_id);
            }
        }
    }

    // Merged last so caller headers override the defaults; `None` suppresses.
    if let Some(options_headers) = &options.base.base.headers {
        for (name, value) in options_headers {
            match value {
                Some(value) => insert_header(&mut headers, name, value),
                None => {
                    headers.retain(|key, _| !key.eq_ignore_ascii_case(name));
                    headers.insert(name.clone(), None);
                }
            }
        }
    }
    crate::utils::headers::provider_headers_to_header_map(&headers)
}

/// Insert a field into a JSON object without panicking on non-objects.
fn set_field(target: &mut Value, key: &str, value: Value) {
    if let Some(object) = target.as_object_mut() {
        object.insert(key.to_owned(), value);
    }
}

/// Build the Responses request payload.
fn build_params(
    model: &Model,
    context: &Context,
    options: Option<&OpenAIResponsesOptions>,
    compat: ResolvedCompat,
    grammar_tool_input_properties: &HashMap<String, String>,
) -> Result<Value, String> {
    let deferred_tools_mode = if compat.supports_additional_tools {
        Some(DeferredToolsMode::AdditionalTools)
    } else if compat.supports_tool_search {
        Some(DeferredToolsMode::ToolSearch)
    } else {
        None
    };
    let tool_placement = split_deferred_tools(context, deferred_tools_mode.is_some(), |name| {
        name.to_owned()
    });
    let tool_options = ConvertResponsesToolsOptions {
        strict: None,
        supports_strict_mode: Some(compat.supports_strict_mode),
        supports_openai_grammar_tools: Some(compat.supports_openai_grammar_tools),
        defer_loading: None,
    };
    let messages = convert_responses_messages(
        model,
        context,
        OPENAI_TOOL_CALL_PROVIDERS,
        Some(&ConvertResponsesMessagesOptions {
            include_system_prompt: None,
            grammar_tool_input_properties: Some(grammar_tool_input_properties.clone()),
            deferred_tools: Some(tool_placement.deferred.clone()),
            deferred_tools_mode,
            tool_options: Some(tool_options.clone()),
        }),
    )?;

    let cache_retention = options
        .and_then(|options| options.base.cache_retention)
        .unwrap_or(CacheRetention::Short);
    let mut params = json!({
        "model": model.id,
        "input": messages,
        "stream": true,
        "store": false,
    });
    if cache_retention != CacheRetention::None
        && let Some(key) = clamp_openai_prompt_cache_key(
            options.and_then(|options| options.base.session_id.as_deref()),
        )
    {
        set_field(&mut params, "prompt_cache_key", json!(key));
    }
    if let Some(retention) = get_prompt_cache_retention(compat, cache_retention) {
        set_field(&mut params, "prompt_cache_retention", json!(retention));
    }
    if cache_retention == CacheRetention::None && compat.supports_explicit_prompt_cache_mode {
        set_field(
            &mut params,
            "prompt_cache_options",
            json!({ "mode": "explicit" }),
        );
    }

    if let Some(max_tokens) = options.and_then(|options| options.base.max_tokens) {
        set_field(
            &mut params,
            "max_output_tokens",
            json!(max_tokens.max(OPENAI_RESPONSES_MIN_OUTPUT_TOKENS)),
        );
    }
    if let Some(temperature) = options.and_then(|options| options.base.temperature) {
        set_field(&mut params, "temperature", json!(temperature));
    }
    if let Some(service_tier) = options.and_then(|options| options.service_tier.clone()) {
        set_field(&mut params, "service_tier", service_tier);
    }
    if !tool_placement.immediate.is_empty() {
        let tools = convert_responses_tools(&tool_placement.immediate, Some(&tool_options))?;
        set_field(&mut params, "tools", json!(tools));
    }
    if let Some(tool_choice) = options.and_then(|options| options.tool_choice.clone()) {
        set_field(&mut params, "tool_choice", tool_choice);
    }

    if model.reasoning {
        let effort = options.and_then(|options| options.reasoning_effort);
        let summary = options.and_then(|options| options.reasoning_summary);
        if effort.is_some() || summary.is_some() {
            let effort = match effort {
                Some(effort) => mapped_reasoning_effort(model, effort),
                None => "medium".to_owned(),
            };
            let mut reasoning = json!({ "effort": effort });
            set_field(
                &mut reasoning,
                "summary",
                json!(summary.unwrap_or(ReasoningSummary::Auto).wire()),
            );
            set_field(&mut params, "reasoning", reasoning);
            set_field(
                &mut params,
                "include",
                json!(["reasoning.encrypted_content"]),
            );
        } else if model.provider.as_str() != "github-copilot" && supports_reasoning_off(model) {
            set_field(
                &mut params,
                "reasoning",
                json!({ "effort": reasoning_off_effort(model) }),
            );
        }
        if model.provider.as_str() == "xai" {
            set_field(
                &mut params,
                "include",
                json!(["reasoning.encrypted_content"]),
            );
        }
    }

    // Last so custom keys override the named request fields.
    if let Some(sampling_params) = options.and_then(|options| options.base.sampling_params.as_ref())
        && let Some(object) = params.as_object_mut()
    {
        for (key, value) in sampling_params {
            object.insert(key.clone(), value.clone());
        }
    }

    Ok(params)
}

// =============================================================================
// Streaming
// =============================================================================

/// A zeroed assistant message bound to `model`: the stream's initial state.
fn initial_output(model: &Model) -> AssistantMessage {
    AssistantMessage {
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
    }
}

fn done_reason(stop_reason: StopReason) -> DoneReason {
    match stop_reason {
        StopReason::Stop => DoneReason::Stop,
        StopReason::Length => DoneReason::Length,
        StopReason::ToolUse => DoneReason::ToolUse,
        StopReason::Deferred => DoneReason::Deferred,
        // Terminal failures never reach the done path.
        StopReason::Pending | StopReason::Error | StopReason::Aborted => DoneReason::Stop,
    }
}

/// POST the payload, retrying transport/HTTP failures the way the SDKs do.
async fn send_request(
    client: reqwest::Client,
    url: &str,
    headers: &reqwest::header::HeaderMap,
    payload: &Value,
    options: &OpenAIResponsesOptions,
) -> Result<crate::api::http::SseResponse, StreamError> {
    let attempt = || async {
        let mut sse = crate::api::http::post_json_sse_tracked(
            client.clone(),
            url,
            headers,
            payload,
            options.base.base.timeout_ms,
            options.base.base.request_metrics.clone(),
        )
        .await;
        // A transport failure or non-2xx response arrives as a leading stream
        // error; surface it so the retry policy can act on it.
        match sse.events.next().await {
            Some(Err(error)) => Err(error),
            Some(Ok(first)) => {
                sse.events =
                    Box::pin(futures::stream::once(async move { Ok(first) }).chain(sse.events));
                Ok(sse)
            }
            None => Ok(sse),
        }
    };
    retry_provider_request_tracked(
        attempt,
        ProviderRetryOptions {
            max_retries: options.base.base.max_retries,
            max_retry_delay_ms: options.base.base.max_retry_delay_ms,
            signal: options.base.base.signal.clone(),
        },
        options.base.base.request_metrics.as_ref(),
    )
    .await
    .map_err(StreamError::Http)
}

/// Run one request attempt: build params, send, and process the stream.
async fn attempt_stream(
    model: &Model,
    context: &Context,
    options: &OpenAIResponsesOptions,
    output: &mut AssistantMessage,
    sender: &mut AssistantMessageEventSender,
) -> Result<(), StreamError> {
    let api_key = get_client_api_key(
        model.provider.as_str(),
        options.base.base.api_key.as_deref(),
        options.base.base.headers.as_ref(),
    )
    .map_err(StreamError::Message)?;
    let cache_retention = options
        .base
        .cache_retention
        .unwrap_or(CacheRetention::Short);
    let cache_session_id = (cache_retention != CacheRetention::None)
        .then(|| options.base.session_id.clone())
        .flatten();
    let compat = get_compat(model);
    let grammar_tool_input_properties = create_grammar_tool_input_properties(
        context.tools.as_deref(),
        compat.supports_openai_grammar_tools,
    )
    .map_err(StreamError::Message)?;
    let mut params = build_params(
        model,
        context,
        Some(options),
        compat,
        &grammar_tool_input_properties,
    )
    .map_err(StreamError::Message)?;
    if let Some(observer) = &options.base.on_payload
        && let Some(next) = observer.on_payload(params.clone(), model).await
    {
        params = next;
    }

    let headers = build_headers(
        model,
        context,
        &api_key,
        options,
        cache_session_id.as_deref(),
        compat,
    )
    .map_err(|error| StreamError::Message(error.to_string()))?;
    let client = http_client_for(&options.base.base)?;
    let sse = send_request(client, &responses_url(model), &headers, &params, options).await?;

    if let Some(metrics) = &options.base.base.request_metrics { metrics.response(sse.status); }
    if let Some(observer) = &options.base.on_response {
        observer
            .on_response(
                &ProviderResponse {
                    status: sse.status,
                    headers: crate::utils::headers::header_map_to_record(&sse.headers),
                },
                model,
            )
            .await;
    }

    // `start` marks an empty initial message; deltas build content and the
    // final event is authoritative. The content is mutated in place as chunks
    // arrive, so give `start` its own empty content array.
    let mut partial = output.clone();
    partial.content.clear();
    sender.push(AssistantMessageEvent::Start { partial }).await;

    let service_tier_cost_multiplier =
        |service_tier: &str| get_service_tier_cost_multiplier(&model.id, service_tier);
    let stream_options = ResponsesStreamOptions {
        snapshot_mode: options.base.event_snapshot_mode,
        service_tier: options.service_tier.as_ref().and_then(Value::as_str),
        grammar_tool_input_properties: Some(&grammar_tool_input_properties),
        service_tier_cost_multiplier: Some(&service_tier_cost_multiplier),
        signal: options.base.base.signal.as_ref(),
    };
    process_responses_stream(sse.events, output, sender, model, Some(&stream_options))
        .await
        .map_err(StreamError::from)
}

/// Emit the terminal error event for `output`.
async fn fail(
    output: &mut AssistantMessage,
    options: &OpenAIResponsesOptions,
    error: &StreamError,
    sender: &mut AssistantMessageEventSender,
) {
    let aborted = options
        .base
        .base
        .signal
        .as_ref()
        .is_some_and(|signal| signal.is_cancelled());
    output.stop_reason = if aborted {
        StopReason::Aborted
    } else {
        StopReason::Error
    };
    output.error_message = Some(format_stream_error(error));
    sender
        .push(AssistantMessageEvent::Error {
            reason: if aborted {
                ErrorReason::Aborted
            } else {
                ErrorReason::Error
            },
            error: output.clone(),
        })
        .await;
}

/// Drive one stream to a terminal event.
async fn run_stream(
    model: &Model,
    context: &Context,
    options: &OpenAIResponsesOptions,
    sender: &mut AssistantMessageEventSender,
) {
    let mut output = initial_output(model);
    if let Err(error) = attempt_stream(model, context, options, &mut output, sender).await {
        fail(&mut output, options, &error, sender).await;
        return;
    }

    if options
        .base
        .base
        .signal
        .as_ref()
        .is_some_and(|signal| signal.is_cancelled())
    {
        fail(
            &mut output,
            options,
            &StreamError::Message("Request was aborted".to_owned()),
            sender,
        )
        .await;
        return;
    }
    if output.stop_reason == StopReason::Pending {
        fail(
            &mut output,
            options,
            &StreamError::Message("OpenAI Responses stream ended without a stop reason".to_owned()),
            sender,
        )
        .await;
        return;
    }
    if matches!(output.stop_reason, StopReason::Aborted | StopReason::Error) {
        let message = output
            .error_message
            .clone()
            .unwrap_or_else(|| "An unknown error occurred".to_owned());
        fail(&mut output, options, &StreamError::Message(message), sender).await;
        return;
    }

    sender
        .push(AssistantMessageEvent::Done {
            reason: done_reason(output.stop_reason),
            message: output,
        })
        .await;
}

// =============================================================================
// Public entry points
// =============================================================================

/// Generate a stream of assistant-message events from the OpenAI Responses API.
pub fn stream(
    model: &Model,
    context: &Context,
    options: &OpenAIResponsesOptions,
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

        run_stream(&model, &context, &options, &mut sender).await;
    });
    stream
}

/// Generate a stream from unified options, mapping reasoning and tool choice.
pub fn stream_simple(
    model: &Model,
    context: &Context,
    options: &SimpleStreamOptions,
) -> AssistantMessageEventStream {
    let base = build_base_options(
        model,
        context,
        Some(options),
        options.base.base.api_key.as_deref(),
    );
    let mut responses_options = OpenAIResponsesOptions {
        base,
        reasoning_effort: None,
        reasoning_summary: None,
        service_tier: None,
        tool_choice: None,
    };
    responses_options.tool_choice = options
        .tool_choice
        .and_then(|choice| serde_json::to_value(choice).ok())
        .filter(|value| !value.is_null());
    if let Some(reasoning) = options.reasoning {
        let clamped = clamp_effort(model, ReasoningEffort::from_thinking_level(reasoning));
        responses_options.reasoning_effort = model_level_to_effort(clamped);
    }
    stream(model, context, &responses_options)
}

/// The adapter value registered for the `openai-responses` dialect.
pub struct OpenAIResponsesApi;

#[async_trait]
impl ProviderStreams for OpenAIResponsesApi {
    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: &StreamOptions,
    ) -> AssistantMessageEventStream {
        let responses_options = OpenAIResponsesOptions {
            base: options.clone(),
            reasoning_effort: None,
            reasoning_summary: None,
            service_tier: None,
            tool_choice: None,
        };
        stream(model, context, &responses_options)
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

/// The shared `openai-responses` dialect adapter. Dispatch is by `model.api`,
/// never by provider identity.
pub fn openai_responses_api() -> OpenAIResponsesApi {
    OpenAIResponsesApi
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::openai_responses_shared::convert_responses_messages;
    use crate::types::{
        Api, InputModality, Message, ModelCost, ProviderId, TextContent, Tool, ToolCall,
        ToolResultMessage, UserContent, UserContentBlock, UserMessage,
    };

    fn responses_model() -> Model {
        Model {
            id: "step-5-preview".to_owned(),
            name: "Step 5 Preview".to_owned(),
            api: Api::OpenaiResponses,
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

    fn test_context() -> Context {
        Context {
            system_prompt: Some("You are a helpful assistant.".to_owned()),
            messages: vec![Message::User(UserMessage {
                content: UserContent::Text("Hello".to_owned()),
                timestamp: 1,
            })],
            tools: Some(vec![Tool {
                name: "ping".to_owned(),
                description: "Ping a host".to_owned(),
                parameters: json!({"type": "object", "properties": {"host": {"type": "string"}}}),
                constrained_sampling: None,
            }]),
        }
    }

    fn options() -> OpenAIResponsesOptions {
        OpenAIResponsesOptions {
            base: keyed_stream_options("test-key"),
            ..Default::default()
        }
    }

    fn keyed_stream_options(api_key: &str) -> StreamOptions {
        StreamOptions {
            base: crate::types::ProviderRequestOptions {
                api_key: Some(api_key.to_owned()),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    // ---- params --------------------------------------------------------------

    #[test]
    fn params_match_responses_wire_shape() {
        let model = responses_model();
        let context = test_context();
        let compat = get_compat(&model);
        let params =
            build_params(&model, &context, Some(&options()), compat, &HashMap::new()).unwrap();
        assert_eq!(params["model"], json!("step-5-preview"));
        assert_eq!(params["stream"], json!(true));
        assert_eq!(params["store"], json!(false));
        assert!(params["input"].is_array());
        assert!(
            params["input"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item.get("role").and_then(Value::as_str) == Some("user"))
        );
        // Responses tools carry `name` at the top level.
        assert_eq!(params["tools"][0]["name"], json!("ping"));
        assert_eq!(params["tools"][0]["type"], json!("function"));
        // No session id: no prompt cache key.
        assert!(params.get("prompt_cache_key").is_none());
    }

    #[test]
    fn prompt_cache_key_clamped_to_64_chars() {
        let model = responses_model();
        let context = test_context();
        let mut options = options();
        options.base.session_id = Some("x".repeat(67));
        let compat = get_compat(&model);
        let params =
            build_params(&model, &context, Some(&options), compat, &HashMap::new()).unwrap();
        assert_eq!(params["prompt_cache_key"], json!("x".repeat(64)));
    }

    #[test]
    fn prompt_cache_retention_long_sends_24h() {
        let model = responses_model();
        let context = test_context();
        let mut options = options();
        options.base.cache_retention = Some(CacheRetention::Long);
        let compat = get_compat(&model);
        let params =
            build_params(&model, &context, Some(&options), compat, &HashMap::new()).unwrap();
        assert_eq!(params["prompt_cache_retention"], json!("24h"));
    }

    #[test]
    fn cache_retention_none_disables_implicit_cache_when_supported() {
        let model = responses_model();
        let context = test_context();
        let mut options = options();
        options.base.cache_retention = Some(CacheRetention::None);
        let compat = get_compat(&model);
        let params =
            build_params(&model, &context, Some(&options), compat, &HashMap::new()).unwrap();
        assert!(params.get("prompt_cache_key").is_none());
        assert!(params.get("prompt_cache_retention").is_none());

        let mut compat_model = model.clone();
        compat_model.compat = Some(crate::types::ApiCompat {
            supports_explicit_prompt_cache_mode: Some(true),
            ..Default::default()
        });
        let compat = get_compat(&compat_model);
        let params = build_params(
            &compat_model,
            &context,
            Some(&options),
            compat,
            &HashMap::new(),
        )
        .unwrap();
        assert_eq!(params["prompt_cache_options"], json!({"mode": "explicit"}));
    }

    #[test]
    fn tool_choice_forwarded_verbatim() {
        let model = responses_model();
        let context = test_context();
        let mut options = options();
        options.tool_choice = Some(json!("required"));
        let compat = get_compat(&model);
        let params =
            build_params(&model, &context, Some(&options), compat, &HashMap::new()).unwrap();
        assert_eq!(params["tool_choice"], json!("required"));
        assert_eq!(params["tools"][0]["name"], json!("ping"));
    }

    #[test]
    fn max_output_tokens_has_a_floor_of_16() {
        let model = responses_model();
        let context = test_context();
        let mut options = options();
        options.base.max_tokens = Some(4);
        let compat = get_compat(&model);
        let params =
            build_params(&model, &context, Some(&options), compat, &HashMap::new()).unwrap();
        assert_eq!(params["max_output_tokens"], json!(16));

        options.base.max_tokens = Some(4096);
        let params =
            build_params(&model, &context, Some(&options), compat, &HashMap::new()).unwrap();
        assert_eq!(params["max_output_tokens"], json!(4096));
    }

    #[test]
    fn temperature_and_service_tier_forwarded() {
        let model = responses_model();
        let context = test_context();
        let mut options = options();
        options.base.temperature = Some(0.25);
        options.service_tier = Some(json!("flex"));
        let compat = get_compat(&model);
        let params =
            build_params(&model, &context, Some(&options), compat, &HashMap::new()).unwrap();
        assert_eq!(params["temperature"], json!(0.25));
        assert_eq!(params["service_tier"], json!("flex"));
    }

    #[test]
    fn sampling_params_override_named_fields() {
        let model = responses_model();
        let context = test_context();
        let mut options = options();
        options.base.temperature = Some(0.25);
        let mut sampling = serde_json::Map::new();
        sampling.insert("temperature".to_owned(), json!(0.9));
        sampling.insert("top_p".to_owned(), json!(0.5));
        options.base.sampling_params = Some(sampling);
        let compat = get_compat(&model);
        let params =
            build_params(&model, &context, Some(&options), compat, &HashMap::new()).unwrap();
        assert_eq!(params["temperature"], json!(0.9));
        assert_eq!(params["top_p"], json!(0.5));
    }

    #[test]
    fn reasoning_effort_none_when_off_supported() {
        let model = responses_model();
        let context = test_context();
        let compat = get_compat(&model);
        let params =
            build_params(&model, &context, Some(&options()), compat, &HashMap::new()).unwrap();
        assert_eq!(params["reasoning"], json!({"effort": "none"}));
        assert!(params.get("include").is_none());
    }

    #[test]
    fn reasoning_omitted_when_off_unsupported() {
        let mut model = responses_model();
        model.thinking_level_map = Some([(ModelThinkingLevel::Off, None)].into_iter().collect());
        let context = test_context();
        let compat = get_compat(&model);
        let params =
            build_params(&model, &context, Some(&options()), compat, &HashMap::new()).unwrap();
        assert!(params.get("reasoning").is_none());
    }

    #[test]
    fn reasoning_effort_mapped_through_thinking_level_map() {
        let mut model = responses_model();
        model.thinking_level_map = Some(
            [(ModelThinkingLevel::High, Some("high-effort".to_owned()))]
                .into_iter()
                .collect(),
        );
        let context = test_context();
        let mut options = options();
        options.reasoning_effort = Some(ReasoningEffort::High);
        let compat = get_compat(&model);
        let params =
            build_params(&model, &context, Some(&options), compat, &HashMap::new()).unwrap();
        assert_eq!(params["reasoning"]["effort"], json!("high-effort"));
        assert_eq!(params["reasoning"]["summary"], json!("auto"));
        assert_eq!(params["include"], json!(["reasoning.encrypted_content"]));
    }

    #[test]
    fn reasoning_summary_forwarded() {
        let model = responses_model();
        let context = test_context();
        let mut options = options();
        options.reasoning_summary = Some(ReasoningSummary::Detailed);
        let compat = get_compat(&model);
        let params =
            build_params(&model, &context, Some(&options), compat, &HashMap::new()).unwrap();
        assert_eq!(params["reasoning"]["summary"], json!("detailed"));
    }

    #[test]
    fn copilot_omits_default_reasoning_effort() {
        let mut model = responses_model();
        model.provider = ProviderId::new("github-copilot");
        let context = test_context();
        let compat = get_compat(&model);
        let params =
            build_params(&model, &context, Some(&options()), compat, &HashMap::new()).unwrap();
        assert!(params.get("reasoning").is_none());
    }

    #[test]
    fn xai_always_requests_encrypted_reasoning() {
        let mut model = responses_model();
        model.provider = ProviderId::new("xai");
        let context = test_context();
        let compat = get_compat(&model);
        let params =
            build_params(&model, &context, Some(&options()), compat, &HashMap::new()).unwrap();
        assert_eq!(params["include"], json!(["reasoning.encrypted_content"]));
    }

    #[test]
    fn empty_tools_omit_the_tools_field() {
        let mut context = test_context();
        context.tools = None;
        let model = responses_model();
        let compat = get_compat(&model);
        let params =
            build_params(&model, &context, Some(&options()), compat, &HashMap::new()).unwrap();
        assert!(params.get("tools").is_none());
    }

    #[test]
    fn deferred_tools_use_additional_tools_mode() {
        let mut model = responses_model();
        model.compat = Some(crate::types::ApiCompat {
            supports_additional_tools: Some(true),
            ..Default::default()
        });
        let ping = Tool {
            name: "ping".to_owned(),
            description: "Ping a host".to_owned(),
            parameters: json!({"type": "object", "properties": {"host": {"type": "string"}}}),
            constrained_sampling: None,
        };
        let context = Context {
            system_prompt: None,
            messages: vec![
                Message::User(UserMessage {
                    content: UserContent::Text("hi".to_owned()),
                    timestamp: 1,
                }),
                Message::ToolResult(Box::new(ToolResultMessage {
                    tool_call_id: "call-1".to_owned(),
                    tool_name: "search".to_owned(),
                    content: vec![UserContentBlock::Text(TextContent {
                        text: "loaded".to_owned(),
                        text_signature: None,
                    })],
                    details: None,
                    usage: None,
                    added_tool_names: Some(vec!["ping".to_owned()]),
                    is_error: false,
                    timestamp: 2,
                })),
            ],
            tools: Some(vec![ping]),
        };
        let compat = get_compat(&model);
        let params =
            build_params(&model, &context, Some(&options()), compat, &HashMap::new()).unwrap();
        // The deferred tool is not sent as an immediate tool...
        assert!(params.get("tools").is_none());
        // ...but announced through an additional_tools input item.
        assert!(
            params["input"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item.get("type").and_then(Value::as_str) == Some("additional_tools"))
        );
    }

    // ---- headers -------------------------------------------------------------

    #[test]
    fn session_affinity_headers_for_openai_format() {
        let model = responses_model();
        let context = test_context();
        let compat = get_compat(&model);
        let headers = build_headers(
            &model,
            &context,
            "test-key",
            &options(),
            Some("session-123"),
            compat,
        )
        .unwrap();
        assert_eq!(headers.get("session_id").unwrap(), "session-123");
        assert_eq!(headers.get("x-client-request-id").unwrap(), "session-123");
        assert_eq!(headers.get("authorization").unwrap(), "Bearer test-key");
    }

    #[test]
    fn openrouter_uses_x_session_id() {
        let mut model = responses_model();
        model.provider = ProviderId::new("proxy");
        model.base_url = "https://proxy.example.com/v1".to_owned();
        model.compat = Some(crate::types::ApiCompat {
            session_affinity_format: Some(SessionAffinityFormat::Openrouter),
            ..Default::default()
        });
        let context = test_context();
        let compat = get_compat(&model);
        let headers = build_headers(
            &model,
            &context,
            "test-key",
            &options(),
            Some("session-proxy"),
            compat,
        )
        .unwrap();
        assert_eq!(headers.get("x-session-id").unwrap(), "session-proxy");
        assert!(headers.get("session_id").is_none());
        assert!(headers.get("x-client-request-id").is_none());
    }

    #[test]
    fn openai_nosession_omits_session_id_header() {
        let mut model = responses_model();
        model.provider = ProviderId::new("opencode");
        model.base_url = "https://proxy.example.com/v1".to_owned();
        model.compat = Some(crate::types::ApiCompat {
            session_affinity_format: Some(SessionAffinityFormat::OpenaiNosession),
            ..Default::default()
        });
        let context = test_context();
        let compat = get_compat(&model);
        let headers = build_headers(
            &model,
            &context,
            "test-key",
            &options(),
            Some("session-123"),
            compat,
        )
        .unwrap();
        assert!(headers.get("session_id").is_none());
        assert_eq!(headers.get("x-client-request-id").unwrap(), "session-123");
    }

    #[test]
    fn openrouter_endpoint_autodetects_session_affinity_format() {
        let mut model = responses_model();
        model.provider = ProviderId::new("openrouter");
        model.base_url = "https://openrouter.ai/api/v1".to_owned();
        let context = test_context();
        let compat = get_compat(&model);
        assert_eq!(
            compat.session_affinity_format,
            SessionAffinityFormat::Openrouter
        );
        let headers = build_headers(
            &model,
            &context,
            "test-key",
            &options(),
            Some("session-or"),
            compat,
        )
        .unwrap();
        assert_eq!(headers.get("x-session-id").unwrap(), "session-or");
    }

    #[test]
    fn caller_headers_override_defaults() {
        let model = responses_model();
        let context = test_context();
        let compat = get_compat(&model);
        let mut options = options();
        let mut headers = ProviderHeaders::new();
        headers.insert("session_id".to_owned(), Some("override-session".to_owned()));
        headers.insert(
            "x-client-request-id".to_owned(),
            Some("override-request".to_owned()),
        );
        options.base.base.headers = Some(headers);
        let built = build_headers(
            &model,
            &context,
            "test-key",
            &options,
            Some("session-123"),
            compat,
        )
        .unwrap();
        assert_eq!(built.get("session_id").unwrap(), "override-session");
        assert_eq!(
            built.get("x-client-request-id").unwrap(),
            "override-request"
        );
    }

    #[test]
    fn suppressed_headers_are_removed() {
        let model = responses_model();
        let context = test_context();
        let compat = get_compat(&model);
        let mut options = options();
        let mut headers = ProviderHeaders::new();
        headers.insert("User-Agent".to_owned(), None);
        options.base.base.headers = Some(headers);
        let built = build_headers(&model, &context, "test-key", &options, None, compat).unwrap();
        assert!(built.get("user-agent").is_none());
    }

    #[test]
    fn copilot_dynamic_headers_added() {
        let mut model = responses_model();
        model.provider = ProviderId::new("github-copilot");
        let context = test_context();
        let compat = get_compat(&model);
        let built = build_headers(&model, &context, "test-key", &options(), None, compat).unwrap();
        assert_eq!(built.get("x-initiator").unwrap(), "user");
        assert_eq!(built.get("openai-intent").unwrap(), "conversation-edits");
    }

    // ---- api key -------------------------------------------------------------

    #[test]
    fn api_key_resolution() {
        let empty = ProviderHeaders::new();
        assert_eq!(
            get_client_api_key("step", Some("k"), None),
            Ok("k".to_owned())
        );
        assert_eq!(
            get_client_api_key("step", None, Some(&empty)),
            Err("No API key for provider: step".to_owned())
        );
        let mut with_auth = ProviderHeaders::new();
        with_auth.insert("Authorization".to_owned(), Some("Bearer x".to_owned()));
        assert_eq!(
            get_client_api_key("step", None, Some(&with_auth)),
            Ok("unused".to_owned())
        );
        let mut with_cf = ProviderHeaders::new();
        with_cf.insert("cf-aig-authorization".to_owned(), Some("token".to_owned()));
        assert_eq!(
            get_client_api_key("step", None, Some(&with_cf)),
            Ok("unused".to_owned())
        );
        // An empty header value does not count as caller-supplied auth.
        let mut blank = ProviderHeaders::new();
        blank.insert("authorization".to_owned(), Some(String::new()));
        assert!(get_client_api_key("step", None, Some(&blank)).is_err());
    }

    // ---- service tier pricing ------------------------------------------------

    #[test]
    fn service_tier_cost_multipliers() {
        assert_eq!(get_service_tier_cost_multiplier("gpt-5.5", "flex"), 0.5);
        assert_eq!(get_service_tier_cost_multiplier("gpt-5.5", "priority"), 2.5);
        assert_eq!(get_service_tier_cost_multiplier("gpt-5.4", "priority"), 2.0);
        assert_eq!(get_service_tier_cost_multiplier("gpt-5.4", "default"), 1.0);
    }

    // ---- error formatting ----------------------------------------------------

    #[test]
    fn http_errors_get_the_provider_prefix() {
        let error = ProviderHttpError::from_response(
            403,
            Some(
                r#"{"error":{"message":"gateway WAF blocked request","type":"forbidden"}}"#
                    .to_owned(),
            ),
            "403 status code".to_owned(),
        );
        let message = format_stream_error(&StreamError::Http(error));
        assert!(message.contains("OpenAI API error"), "{message}");
        assert!(message.contains("403"), "{message}");
        assert!(message.contains("gateway WAF blocked request"), "{message}");
    }

    #[test]
    fn plain_messages_pass_through() {
        assert_eq!(
            format_stream_error(&StreamError::Message("boom".to_owned())),
            "boom"
        );
    }

    // ---- stop reason / done mapping -----------------------------------------

    #[test]
    fn done_reason_mapping() {
        assert_eq!(done_reason(StopReason::Stop), DoneReason::Stop);
        assert_eq!(done_reason(StopReason::Length), DoneReason::Length);
        assert_eq!(done_reason(StopReason::ToolUse), DoneReason::ToolUse);
    }

    // ---- stream_simple -------------------------------------------------------

    #[test]
    fn stream_simple_maps_reasoning_and_tool_choice() {
        // No network: the option mapping is asserted through the params it
        // produces (the same conversion `stream_simple` performs).
        let model = responses_model();
        let context = test_context();
        let simple = SimpleStreamOptions {
            base: keyed_stream_options("test-key"),
            tool_choice: Some(crate::types::ToolChoice::Auto),
            reasoning: Some(ThinkingLevel::High),
            deferred: None,
            thinking_budgets: None,
        };
        let base = build_base_options(
            &model,
            &context,
            Some(&simple),
            simple.base.base.api_key.as_deref(),
        );
        let mut responses = OpenAIResponsesOptions {
            base,
            ..Default::default()
        };
        responses.tool_choice = simple
            .tool_choice
            .and_then(|choice| serde_json::to_value(choice).ok())
            .filter(|value| !value.is_null());
        let clamped = clamp_effort(
            &model,
            ReasoningEffort::from_thinking_level(ThinkingLevel::High),
        );
        responses.reasoning_effort = model_level_to_effort(clamped);

        assert_eq!(responses.tool_choice, Some(json!("auto")));
        assert_eq!(responses.reasoning_effort, Some(ReasoningEffort::High));
        let compat = get_compat(&model);
        let params =
            build_params(&model, &context, Some(&responses), compat, &HashMap::new()).unwrap();
        assert_eq!(params["reasoning"]["effort"], json!("high"));
        assert_eq!(params["tool_choice"], json!("auto"));
    }

    #[test]
    fn stream_simple_clamps_reasoning_to_off() {
        let mut model = responses_model();
        model.reasoning = false;
        // A non-reasoning model only supports `off`, so no effort is requested.
        let clamped = clamp_effort(&model, ReasoningEffort::High);
        assert_eq!(clamped, ModelThinkingLevel::Off);
        assert_eq!(model_level_to_effort(clamped), None);
    }

    // ---- provider streams ---------------------------------------------------

    #[tokio::test]
    async fn provider_streams_adapts_generic_options() {
        let api = openai_responses_api();
        let mut model = responses_model();
        model.base_url = spawn_server(
            "400 Bad Request",
            "application/json",
            json!({"error": {"message": "bad request"}}).to_string(),
        )
        .await;
        let context = test_context();
        let mut stream_options = keyed_stream_options("test-key");
        stream_options.base.http_client = Some(no_proxy_client());
        let mut stream = ProviderStreams::stream(&api, &model, &context, &stream_options);
        // A failing endpoint terminates the stream with an error event rather
        // than panicking.
        let result = stream.result().await.unwrap();
        assert_eq!(result.stop_reason, StopReason::Error);
        assert!(result.error_message.is_some());
    }

    #[tokio::test]
    async fn provider_streams_simple_maps_reasoning() {
        let api = openai_responses_api();
        let mut model = responses_model();
        model.base_url = spawn_server(
            "200 OK",
            "text/event-stream",
            sse_body(&text_response_events()),
        )
        .await;
        let context = test_context();
        let simple = SimpleStreamOptions {
            base: keyed_stream_options("test-key"),
            reasoning: Some(ThinkingLevel::Medium),
            ..Default::default()
        };
        let mut simple = simple;
        simple.base.base.http_client = Some(no_proxy_client());
        let mut stream = ProviderStreams::stream_simple(&api, &model, &context, &simple);
        let mut saw_done = false;
        while let Some(event) = stream.next().await {
            if event.is_terminal() {
                saw_done = matches!(event, AssistantMessageEvent::Done { .. });
                break;
            }
        }
        assert!(saw_done);
    }

    // ---- end-to-end streams over a loopback SSE server -----------------------

    /// Serve one canned response (head + optional body) and return its URL.
    async fn spawn_server(status: &str, content_type: &str, body: String) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let status = status.to_owned();
        let content_type = content_type.to_owned();
        tokio::spawn(async move {
            if let Ok((mut socket, _)) = listener.accept().await {
                let mut buffer = Vec::new();
                let mut chunk = [0u8; 4096];
                let header_end = loop {
                    if let Some(position) =
                        buffer.windows(4).position(|window| window == b"\r\n\r\n")
                    {
                        break position + 4;
                    }
                    match socket.read(&mut chunk).await {
                        Ok(0) | Err(_) => break buffer.len(),
                        Ok(read) => buffer.extend_from_slice(&chunk[..read]),
                    }
                };
                let content_length = String::from_utf8_lossy(&buffer)
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())
                            .flatten()
                    })
                    .unwrap_or(0);
                while buffer.len() < header_end + content_length {
                    match socket.read(&mut chunk).await {
                        Ok(0) | Err(_) => break,
                        Ok(read) => buffer.extend_from_slice(&chunk[..read]),
                    }
                }
                let response = format!(
                    "HTTP/1.1 {status}\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.flush().await;
            }
        });
        format!("http://127.0.0.1:{}", address.port())
    }

    fn no_proxy_client() -> reqwest::Client {
        reqwest::Client::builder().no_proxy().build().unwrap()
    }

    fn sse_body(events: &[Value]) -> String {
        format!(
            "{}\n\n",
            events
                .iter()
                .map(|event| format!("data: {event}"))
                .collect::<Vec<_>>()
                .join("\n\n")
        )
    }

    fn text_response_events() -> Vec<Value> {
        vec![
            json!({ "type": "response.created", "response": { "id": "resp_conf_1" } }),
            json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": { "type": "message", "id": "msg_conf_1", "status": "in_progress", "role": "assistant", "content": [] },
            }),
            json!({ "type": "response.output_text.delta", "output_index": 0, "item_id": "msg_conf_1", "content_index": 0, "delta": "Hello" }),
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "msg_conf_1",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{ "type": "output_text", "text": "Hello", "annotations": [] }],
                },
            }),
            json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_conf_1",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "total_tokens": 12,
                        "input_tokens_details": { "cached_tokens": 0 },
                    },
                },
            }),
        ]
    }

    #[tokio::test]
    async fn conformance_stream_events() {
        let mut model = responses_model();
        model.base_url = spawn_server(
            "200 OK",
            "text/event-stream",
            sse_body(&text_response_events()),
        )
        .await;
        let mut options = options();
        options.base.base.http_client = Some(no_proxy_client());

        let mut stream = stream(&model, &test_context(), &options);
        let mut types = Vec::new();
        while let Some(event) = stream.next().await {
            types.push(match &event {
                AssistantMessageEvent::Start { .. } => "start",
                AssistantMessageEvent::TextStart { .. } => "text_start",
                AssistantMessageEvent::TextDelta { .. } => "text_delta",
                AssistantMessageEvent::TextEnd { .. } => "text_end",
                AssistantMessageEvent::Done { .. } => "done",
                AssistantMessageEvent::Error { .. } => "error",
                _ => "other",
            });
            if event.is_terminal() {
                break;
            }
        }
        assert_eq!(
            types,
            vec!["start", "text_start", "text_delta", "text_end", "done"]
        );

        let result = stream.result().await.unwrap();
        assert_eq!(
            serde_json::to_value(&result.content).unwrap(),
            json!([{
                "type": "text",
                "text": "Hello",
                "textSignature": r#"{"v":1,"id":"msg_conf_1"}"#,
            }])
        );
        assert_eq!(result.stop_reason, StopReason::Stop);
        assert_eq!(result.response_id.as_deref(), Some("resp_conf_1"));
        assert!(result.error_message.is_none());
        assert_eq!(result.usage.input, 10);
        assert_eq!(result.usage.output, 2);
    }

    #[tokio::test]
    async fn conformance_error_normalization() {
        let mut model = responses_model();
        let body = json!({
            "error": { "message": "gateway WAF blocked request", "type": "forbidden", "code": "forbidden" }
        })
        .to_string();
        model.base_url = spawn_server("403 Forbidden", "application/json", body).await;
        let mut options = options();
        options.base.base.http_client = Some(no_proxy_client());

        let mut stream = stream(&model, &test_context(), &options);
        let mut last = "none";
        while let Some(event) = stream.next().await {
            last = match &event {
                AssistantMessageEvent::Error { .. } => "error",
                AssistantMessageEvent::Done { .. } => "done",
                _ => "other",
            };
            if event.is_terminal() {
                break;
            }
        }
        assert_eq!(last, "error");
        let result = stream.result().await.unwrap();
        assert_eq!(result.stop_reason, StopReason::Error);
        let message = result.error_message.unwrap_or_default();
        assert!(message.contains("OpenAI API error"), "{message}");
        assert!(message.contains("403"), "{message}");
        assert!(message.contains("gateway WAF blocked request"), "{message}");
    }

    #[tokio::test]
    async fn missing_api_key_errors_without_network() {
        let model = responses_model();
        let mut stream = stream(&model, &test_context(), &OpenAIResponsesOptions::default());
        let mut saw_error = false;
        while let Some(event) = stream.next().await {
            if let AssistantMessageEvent::Error { error, .. } = &event {
                saw_error = true;
                assert_eq!(
                    error.error_message.as_deref(),
                    Some("No API key for provider: step")
                );
            }
        }
        assert!(saw_error);
    }

    #[tokio::test]
    async fn service_tier_pricing_end_to_end() {
        let mut model = responses_model();
        model.id = "gpt-5.5".to_owned();
        model.cost = ModelCost {
            rates: crate::types::ModelCostRates {
                input: 1.0,
                output: 2.0,
                cache_read: 0.5,
                cache_write: 1.25,
            },
            tiers: Vec::new(),
        };
        model.base_url = spawn_server(
            "200 OK",
            "text/event-stream",
            sse_body(&[json!({
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "service_tier": "priority",
                    "usage": {
                        "input_tokens": 100_000,
                        "output_tokens": 100_000,
                        "total_tokens": 200_000,
                        "input_tokens_details": { "cached_tokens": 0 },
                    },
                },
            })]),
        )
        .await;
        let mut options = options();
        options.base.base.http_client = Some(no_proxy_client());
        options.service_tier = Some(json!("priority"));

        let result = stream(&model, &test_context(), &options)
            .result()
            .await
            .unwrap();
        let scale = 100_000.0 / 1_000_000.0;
        assert!((result.usage.cost.input - 1.0 * 2.5 * scale).abs() < 1e-12);
        assert!((result.usage.cost.output - 2.0 * 2.5 * scale).abs() < 1e-12);
        assert!((result.usage.cost.total - 3.0 * 2.5 * scale).abs() < 1e-12);
    }

    // ---- replay --------------------------------------------------------------

    fn assistant_turn(model: &Model, model_id: &str, namespace: Option<&str>) -> Message {
        Message::Assistant(Box::new(AssistantMessage {
            content: vec![crate::types::AssistantContent::ToolCall(ToolCall {
                id: "call_test|fc_test".to_owned(),
                name: "lookup".to_owned(),
                arguments: json!({"value": "hello"}).as_object().cloned().unwrap(),
                thought_signature: None,
                namespace: namespace.map(str::to_owned),
            })],
            api: model.api.clone(),
            provider: model.provider.clone(),
            model: model_id.to_owned(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: Usage::default(),
            stop_reason: StopReason::ToolUse,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            timestamp: 1,
        }))
    }

    #[tokio::test]
    async fn assistant_tool_call_replays_as_function_call() {
        let model = responses_model();
        let context = Context {
            system_prompt: None,
            messages: vec![assistant_turn(&model, &model.id, Some("dynamic_tools"))],
            tools: None,
        };
        let items = convert_responses_messages(&model, &context, &["openai"], None).unwrap();
        let function_call = items
            .iter()
            .find(|item| item.get("type").and_then(Value::as_str) == Some("function_call"))
            .expect("function_call item");
        assert_eq!(function_call["id"], json!("fc_test"));
        assert_eq!(function_call["call_id"], json!("call_test"));
        assert_eq!(function_call["arguments"], json!(r#"{"value":"hello"}"#));
        assert_eq!(function_call["namespace"], json!("dynamic_tools"));

        // A different target model drops the namespace: it cannot replay the
        // load item that created it.
        let context = Context {
            system_prompt: None,
            messages: vec![assistant_turn(&model, "other-model", Some("dynamic_tools"))],
            tools: None,
        };
        let items = convert_responses_messages(&model, &context, &["openai"], None).unwrap();
        let function_call = items
            .iter()
            .find(|item| item.get("type").and_then(Value::as_str) == Some("function_call"))
            .expect("function_call item");
        assert!(function_call.get("namespace").is_none());
    }

    #[tokio::test]
    async fn calculate_cost_is_applied_to_finalized_usage() {
        // Smoke test for the cost path the stream processor uses.
        let mut model = responses_model();
        model.cost = ModelCost {
            rates: crate::types::ModelCostRates {
                input: 1.0,
                output: 2.0,
                cache_read: 0.5,
                cache_write: 1.25,
            },
            tiers: Vec::new(),
        };
        let mut usage = Usage {
            input: 1_000_000,
            output: 1_000_000,
            cache_read: 0,
            cache_write: 0,
            cache_write_1h: None,
            reasoning: None,
            total_tokens: 2_000_000,
            cost: Default::default(),
        };
        let cost = crate::models::calculate_cost(&model, &mut usage);
        assert!((cost.input - 1.0).abs() < 1e-12);
        assert!((cost.output - 2.0).abs() < 1e-12);
        assert!((cost.total - 3.0).abs() < 1e-12);
    }
}
