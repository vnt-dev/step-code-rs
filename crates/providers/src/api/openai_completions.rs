//! OpenAI Chat Completions API adapter.
//!
//! Rust port of `packages/providers/src/api/openai-completions.ts`. The adapter
//! speaks the `chat.completions` wire dialect: `messages`/`tools`/`tool_choice`
//! request bodies, `chat.completion.chunk` SSE deltas, and the provider-specific
//! `compat` surface (thinking formats, token fields, session affinity, deferred
//! tools, grammar tools).
//!
//! Structure (all pure helpers are unit-testable without network):
//! - [`resolve_compat`] — URL/provider auto-detection merged with `model.compat`.
//! - [`build_params`] — the request payload.
//! - [`convert_messages`] / [`convert_tools`] — conversation and tool encoding.
//! - `StreamingState` — chunk → assistant-event mapping and block bookkeeping.
//! - [`stream`] / [`stream_simple`] — the network entry points.

use async_openai::types::chat::{
    ChatCompletionTool, ChatCompletionTools, CustomGrammarFormatParam, CustomToolChatCompletions,
    CustomToolProperties, CustomToolPropertiesFormat, FunctionObject, GrammarSyntax,
};
use std::collections::{HashMap, HashSet};

use futures::Stream;
use futures::StreamExt;
use serde_json::{Value, json};

use crate::api::constrained_sampling::{
    GrammarToolInputJsonBuffer, create_grammar_tool_input_properties, get_grammar_tool_input,
    get_json_schema_tool_parameters, resolve_grammar_constrained_sampling,
    resolve_json_schema_strict_sampling,
};
use crate::api::copilot_headers::{build_copilot_dynamic_headers, has_copilot_vision_input};
use crate::api::http::{SseEvent, SseResponse, http_client_for};
use crate::api::openai_prompt_cache::clamp_openai_prompt_cache_key;
use crate::api::simple_options::{
    build_base_options, clamp_thinking_budget_to_answer_room, thinking_budget_for_level,
};
use crate::api::transform_messages::transform_messages;
use crate::error::{ProviderHttpError, format_provider_error};
use crate::event::{
    AssistantMessageEvent, AssistantMessageEventSender, AssistantMessageEventStream, DoneReason,
    ErrorReason, assistant_message_event_stream, thinking_text_at,
};
use crate::models::calculate_cost;
use crate::types::{
    Api, AssistantContent, AssistantMessage, CacheControlFormat, CacheRetention,
    ChatTemplateKwargValue, ChatTemplateVar, CompletionsThinkingFormat, Context, DeferredToolsMode,
    GrammarFormat, MaxTokensField, Message, Model, ModelThinkingLevel, ProviderHeaders, ProviderId,
    ProviderResponse, SessionAffinityFormat, SimpleStreamOptions, StopReason, StreamOptions,
    ThinkingBudgets, ThinkingContent, ThinkingLevel, ThinkingTokenBudgetField, Tool, ToolCall,
    Usage, UserContent, UserContentBlock, VercelGatewayRouting,
};
use crate::utils::hash::short_hash;
use crate::utils::headers::merge_provider_headers;
use crate::utils::json_parse::parse_streaming_json;
use crate::utils::now_ms;
use crate::utils::provider_retry::{ProviderRetryOptions, retry_provider_request_tracked};
use crate::utils::sanitize::sanitize_surrogates;
use crate::utils::user_agent::get_user_agent;

/// The dialect this adapter implements (dispatch is by `model.api`, never by
/// provider identity).
pub const API: &str = Api::OPENAI_COMPLETIONS;

/// Reasoning effort levels accepted by the OpenAI `reasoning_effort` field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ReasoningEffort {
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl ReasoningEffort {
    pub fn as_str(&self) -> &'static str {
        match self {
            ReasoningEffort::Minimal => "minimal",
            ReasoningEffort::Low => "low",
            ReasoningEffort::Medium => "medium",
            ReasoningEffort::High => "high",
            ReasoningEffort::Xhigh => "xhigh",
            ReasoningEffort::Max => "max",
        }
    }

    /// The provider-neutral thinking level for this effort.
    pub fn to_thinking_level(&self) -> ThinkingLevel {
        match self {
            ReasoningEffort::Minimal => ThinkingLevel::Minimal,
            ReasoningEffort::Low => ThinkingLevel::Low,
            ReasoningEffort::Medium => ThinkingLevel::Medium,
            ReasoningEffort::High => ThinkingLevel::High,
            ReasoningEffort::Xhigh => ThinkingLevel::Xhigh,
            ReasoningEffort::Max => ThinkingLevel::Max,
        }
    }

    /// The model thinking level (used for `thinking_level_map` lookups).
    pub fn to_model_thinking_level(&self) -> ModelThinkingLevel {
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

impl From<ThinkingLevel> for ReasoningEffort {
    fn from(level: ThinkingLevel) -> Self {
        match level {
            ThinkingLevel::Minimal => ReasoningEffort::Minimal,
            ThinkingLevel::Low => ReasoningEffort::Low,
            ThinkingLevel::Medium => ReasoningEffort::Medium,
            ThinkingLevel::High => ReasoningEffort::High,
            ThinkingLevel::Xhigh => ReasoningEffort::Xhigh,
            ThinkingLevel::Max => ReasoningEffort::Max,
        }
    }
}

impl std::fmt::Display for ReasoningEffort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Options for [`stream`]: the generic stream options plus the three
/// completions-specific knobs.
#[derive(Clone, Default)]
pub struct OpenAICompletionsOptions {
    pub base: StreamOptions,
    /// Raw `tool_choice` value (`"auto"`, `"none"`, `"required"`, or an object).
    pub tool_choice: Option<Value>,
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Token budgets per thinking level. Used when `compat.thinkingTokenBudgetField`
    /// or `compat.supportsThinkingTokenBudget` is set, or by `{ "$var": "thinking.budget" }`.
    pub thinking_budgets: Option<ThinkingBudgets>,
}

impl std::fmt::Debug for OpenAICompletionsOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAICompletionsOptions")
            .field("base", &self.base)
            .field("tool_choice", &self.tool_choice)
            .field("reasoning_effort", &self.reasoning_effort)
            .field("thinking_budgets", &self.thinking_budgets)
            .finish()
    }
}

/// The metadata keys [`options_from_stream_options`] reads when a generic
/// [`StreamOptions`] is adapted into [`OpenAICompletionsOptions`] (the
/// `ProviderStreams` entry point only carries the generic options).
const TOOL_CHOICE_METADATA_KEYS: [&str; 2] = ["toolChoice", "tool_choice"];
const REASONING_EFFORT_METADATA_KEYS: [&str; 2] = ["reasoningEffort", "reasoning_effort"];
const THINKING_BUDGETS_METADATA_KEYS: [&str; 2] = ["thinkingBudgets", "thinking_budgets"];

/// Adapt generic [`StreamOptions`] into completions options, reading the three
/// completions-specific knobs from `options.metadata` (camelCase or snake_case).
pub fn options_from_stream_options(options: &StreamOptions) -> OpenAICompletionsOptions {
    let metadata = options.metadata.as_ref();
    let tool_choice = metadata.and_then(|metadata| {
        TOOL_CHOICE_METADATA_KEYS
            .iter()
            .find_map(|key| metadata.get(*key))
            .cloned()
    });
    let reasoning_effort = metadata.and_then(|metadata| {
        REASONING_EFFORT_METADATA_KEYS
            .iter()
            .find_map(|key| metadata.get(*key))
            .and_then(|value| serde_json::from_value::<ReasoningEffort>(value.clone()).ok())
    });
    let thinking_budgets = metadata.and_then(|metadata| {
        THINKING_BUDGETS_METADATA_KEYS
            .iter()
            .find_map(|key| metadata.get(*key))
            .and_then(|value| serde_json::from_value::<ThinkingBudgets>(value.clone()).ok())
    });
    OpenAICompletionsOptions {
        base: options.clone(),
        tool_choice,
        reasoning_effort,
        thinking_budgets,
    }
}

// ---------------------------------------------------------------------------
// Compat resolution
// ---------------------------------------------------------------------------

/// The resolved compat: auto-detected defaults with every `model.compat`
/// override applied. Mirrors `ResolvedOpenAICompletionsCompat` in the source.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedOpenAICompletionsCompat {
    pub supports_store: bool,
    pub supports_developer_role: bool,
    pub supports_reasoning_effort: bool,
    pub supports_usage_in_streaming: bool,
    pub supports_finish_reason: bool,
    pub max_tokens_field: MaxTokensField,
    pub requires_tool_result_name: bool,
    pub requires_assistant_after_tool_result: bool,
    pub requires_thinking_as_text: bool,
    pub requires_reasoning_content_on_assistant_messages: bool,
    pub thinking_format: CompletionsThinkingFormat,
    pub open_router_routing: Option<crate::types::OpenRouterRouting>,
    pub vercel_gateway_routing: Option<VercelGatewayRouting>,
    pub chat_template_kwargs: HashMap<String, ChatTemplateKwargValue>,
    pub chat_template_args: HashMap<String, ChatTemplateKwargValue>,
    pub zai_tool_stream: bool,
    pub supports_thinking_token_budget: bool,
    pub thinking_token_budget_field: Option<ThinkingTokenBudgetField>,
    pub supports_openai_grammar_tools: bool,
    pub supports_strict_mode: bool,
    pub cache_control_format: Option<CacheControlFormat>,
    pub send_session_affinity_headers: bool,
    pub deferred_tools_mode: Option<DeferredToolsMode>,
    pub session_affinity_format: SessionAffinityFormat,
    pub supports_long_cache_retention: bool,
}

/// Auto-detect compatibility settings from provider name and baseUrl. Explicit
/// `model.compat` entries override these detected values.
pub fn detect_compat(model: &Model) -> ResolvedOpenAICompletionsCompat {
    let provider = model.provider.as_str();
    let base_url = model.base_url.as_str();
    let is_step = provider == "step"
        || base_url.contains("api.stepfun.com")
        || base_url.contains("api.stepfun.ai");

    let is_zai = provider == "zai"
        || provider == "zai-coding-cn"
        || base_url.contains("api.z.ai")
        || base_url.contains("open.bigmodel.cn");
    let is_together = provider == "together"
        || base_url.contains("api.together.ai")
        || base_url.contains("api.together.xyz");
    let is_moonshot = provider == "moonshotai"
        || provider == "moonshotai-cn"
        || base_url.contains("api.moonshot.");
    let is_open_router = provider == ProviderId::OPENROUTER || base_url.contains("openrouter.ai");
    let is_cloudflare_workers_ai =
        provider == "cloudflare-workers-ai" || base_url.contains("api.cloudflare.com");
    let is_cloudflare_ai_gateway =
        provider == "cloudflare-ai-gateway" || base_url.contains("gateway.ai.cloudflare.com");
    let is_nvidia = provider == "nvidia" || base_url.contains("integrate.api.nvidia.com");
    let is_ant_ling = provider == "ant-ling" || base_url.contains("api.ant-ling.com");
    let is_deepseek = provider == "deepseek" || base_url.to_lowercase().contains("deepseek.com");

    let is_non_standard = is_nvidia
        || provider == "cerebras"
        || base_url.contains("cerebras.ai")
        || provider == "xai"
        || base_url.contains("api.x.ai")
        || is_together
        || base_url.contains("chutes.ai")
        || is_deepseek
        || is_zai
        || is_moonshot
        || provider == "opencode"
        || base_url.contains("opencode.ai")
        || is_cloudflare_workers_ai
        || is_cloudflare_ai_gateway
        || is_ant_ling;

    let use_max_tokens = is_step
        || base_url.contains("chutes.ai")
        || is_deepseek
        || is_moonshot
        || is_cloudflare_ai_gateway
        || is_together
        || is_nvidia
        || is_ant_ling
        || is_zai;

    let is_grok = provider == "xai" || base_url.contains("api.x.ai");
    let is_open_router_developer_role_model =
        is_open_router && (model.id.starts_with("anthropic/") || model.id.starts_with("openai/"));
    let cache_control_format =
        if provider == ProviderId::OPENROUTER && model.id.starts_with("anthropic/") {
            Some(CacheControlFormat::Anthropic)
        } else {
            None
        };

    ResolvedOpenAICompletionsCompat {
        supports_store: !is_non_standard && !is_step,
        supports_developer_role: !is_step
            && (is_open_router_developer_role_model || (!is_non_standard && !is_open_router)),
        supports_reasoning_effort: !is_grok
            && !is_zai
            && !is_moonshot
            && !is_together
            && !is_cloudflare_ai_gateway
            && !is_nvidia
            && !is_ant_ling,
        supports_usage_in_streaming: true,
        supports_finish_reason: true,
        max_tokens_field: if use_max_tokens {
            MaxTokensField::MaxTokens
        } else {
            MaxTokensField::MaxCompletionTokens
        },
        requires_tool_result_name: false,
        requires_assistant_after_tool_result: false,
        requires_thinking_as_text: false,
        requires_reasoning_content_on_assistant_messages: is_deepseek,
        thinking_format: if is_deepseek {
            CompletionsThinkingFormat::Deepseek
        } else if is_zai {
            CompletionsThinkingFormat::Zai
        } else if is_together {
            CompletionsThinkingFormat::Together
        } else if is_ant_ling {
            CompletionsThinkingFormat::AntLing
        } else if is_open_router {
            CompletionsThinkingFormat::Openrouter
        } else {
            CompletionsThinkingFormat::Openai
        },
        open_router_routing: None,
        vercel_gateway_routing: None,
        chat_template_kwargs: HashMap::new(),
        chat_template_args: HashMap::new(),
        zai_tool_stream: false,
        supports_thinking_token_budget: false,
        thinking_token_budget_field: None,
        supports_strict_mode: !is_moonshot
            && !is_together
            && !is_cloudflare_ai_gateway
            && !is_nvidia,
        supports_openai_grammar_tools: false,
        cache_control_format,
        send_session_affinity_headers: false,
        deferred_tools_mode: None,
        session_affinity_format: if is_open_router {
            SessionAffinityFormat::Openrouter
        } else {
            SessionAffinityFormat::Openai
        },
        supports_long_cache_retention: !(is_together
            || is_cloudflare_workers_ai
            || is_cloudflare_ai_gateway
            || is_nvidia
            || is_ant_ling),
    }
}

/// Get resolved compatibility settings for a model: auto-detected from
/// provider/URL, then overridden by explicit `model.compat`.
pub fn resolve_compat(model: &Model) -> ResolvedOpenAICompletionsCompat {
    let detected = detect_compat(model);
    let Some(compat) = model.openai_completions_compat() else {
        return detected;
    };
    ResolvedOpenAICompletionsCompat {
        supports_store: compat.supports_store.unwrap_or(detected.supports_store),
        supports_developer_role: compat
            .supports_developer_role
            .unwrap_or(detected.supports_developer_role),
        supports_reasoning_effort: compat
            .supports_reasoning_effort
            .unwrap_or(detected.supports_reasoning_effort),
        supports_usage_in_streaming: compat
            .supports_usage_in_streaming
            .unwrap_or(detected.supports_usage_in_streaming),
        supports_finish_reason: compat
            .supports_finish_reason
            .unwrap_or(detected.supports_finish_reason),
        max_tokens_field: compat.max_tokens_field.unwrap_or(detected.max_tokens_field),
        requires_tool_result_name: compat
            .requires_tool_result_name
            .unwrap_or(detected.requires_tool_result_name),
        requires_assistant_after_tool_result: compat
            .requires_assistant_after_tool_result
            .unwrap_or(detected.requires_assistant_after_tool_result),
        requires_thinking_as_text: compat
            .requires_thinking_as_text
            .unwrap_or(detected.requires_thinking_as_text),
        requires_reasoning_content_on_assistant_messages: compat
            .requires_reasoning_content_on_assistant_messages
            .unwrap_or(detected.requires_reasoning_content_on_assistant_messages),
        thinking_format: compat.thinking_format.unwrap_or(detected.thinking_format),
        open_router_routing: compat.open_router_routing.or(detected.open_router_routing),
        vercel_gateway_routing: compat
            .vercel_gateway_routing
            .or(detected.vercel_gateway_routing),
        chat_template_kwargs: compat
            .chat_template_kwargs
            .unwrap_or(detected.chat_template_kwargs),
        chat_template_args: compat
            .chat_template_args
            .unwrap_or(detected.chat_template_args),
        zai_tool_stream: compat.zai_tool_stream.unwrap_or(detected.zai_tool_stream),
        supports_thinking_token_budget: compat
            .supports_thinking_token_budget
            .unwrap_or(detected.supports_thinking_token_budget),
        thinking_token_budget_field: compat
            .thinking_token_budget_field
            .or(detected.thinking_token_budget_field),
        supports_openai_grammar_tools: compat
            .supports_openai_grammar_tools
            .unwrap_or(detected.supports_openai_grammar_tools),
        supports_strict_mode: compat
            .supports_strict_mode
            .unwrap_or(detected.supports_strict_mode),
        cache_control_format: compat
            .cache_control_format
            .or(detected.cache_control_format),
        send_session_affinity_headers: compat
            .send_session_affinity_headers
            .unwrap_or(detected.send_session_affinity_headers),
        deferred_tools_mode: compat.deferred_tools_mode.or(detected.deferred_tools_mode),
        session_affinity_format: compat
            .session_affinity_format
            .unwrap_or(detected.session_affinity_format),
        supports_long_cache_retention: compat
            .supports_long_cache_retention
            .unwrap_or(detected.supports_long_cache_retention),
    }
}

// ---------------------------------------------------------------------------
// Request payload
// ---------------------------------------------------------------------------

fn thinking_token_budget_field_name(field: ThinkingTokenBudgetField) -> &'static str {
    match field {
        ThinkingTokenBudgetField::Vllm => "thinking_token_budget",
        ThinkingTokenBudgetField::Qwen => "thinking_budget",
        ThinkingTokenBudgetField::LlamaCpp => "thinking_budget_tokens",
    }
}

/// `map?.[level] ?? fallback` semantics: a null mapping is treated as absent.
fn nullish_mapped_level(model: &Model, level: ModelThinkingLevel) -> Option<String> {
    model
        .thinking_level_map
        .as_ref()
        .and_then(|map| map.get(&level))
        .cloned()
        .flatten()
}

/// `map?.[level]` semantics: `None` when the key is absent, `Some(None)` when
/// the level is explicitly unsupported.
fn mapped_level_value(model: &Model, level: ModelThinkingLevel) -> Option<Option<String>> {
    model
        .thinking_level_map
        .as_ref()
        .and_then(|map| map.get(&level))
        .cloned()
}

/// Resolve the effort string for a requested level: an explicit mapping wins,
/// an explicit `null` mapping suppresses the field, and an absent key falls
/// back to the requested effort.
fn resolve_effort(model: &Model, requested: ReasoningEffort) -> Option<String> {
    match mapped_level_value(model, requested.to_model_thinking_level()) {
        Some(Some(value)) => Some(value),
        Some(None) => None,
        None => Some(requested.as_str().to_owned()),
    }
}

/// Build the `chat.completions` request payload.
pub fn build_params(
    model: &Model,
    context: &Context,
    options: Option<&OpenAICompletionsOptions>,
    compat: &ResolvedOpenAICompletionsCompat,
    cache_retention: CacheRetention,
    grammar_tool_input_properties: &HashMap<String, String>,
) -> Result<Value, String> {
    let mut messages = convert_messages(model, context, compat, grammar_tool_input_properties)?;
    let mut params = serde_json::Map::new();
    params.insert("model".to_owned(), json!(model.id));
    params.insert(
        "messages".to_owned(),
        Value::Array(std::mem::take(&mut messages)),
    );
    params.insert("stream".to_owned(), json!(true));

    let wants_prompt_cache_key = (model.base_url.contains("api.openai.com")
        && cache_retention != CacheRetention::None)
        || (cache_retention == CacheRetention::Long && compat.supports_long_cache_retention);
    if wants_prompt_cache_key
        && let Some(key) = clamp_openai_prompt_cache_key(
            options.and_then(|options| options.base.session_id.as_deref()),
        )
    {
        params.insert("prompt_cache_key".to_owned(), json!(key));
    }
    if cache_retention == CacheRetention::Long && compat.supports_long_cache_retention {
        params.insert("prompt_cache_retention".to_owned(), json!("24h"));
    }

    if compat.supports_usage_in_streaming {
        params.insert(
            "stream_options".to_owned(),
            json!({ "include_usage": true }),
        );
    }
    if compat.supports_store {
        params.insert("store".to_owned(), json!(false));
    }
    if let Some(max_tokens) = options.and_then(|options| options.base.max_tokens) {
        let field = match compat.max_tokens_field {
            MaxTokensField::MaxTokens => "max_tokens",
            MaxTokensField::MaxCompletionTokens => "max_completion_tokens",
        };
        params.insert(field.to_owned(), json!(max_tokens));
    }
    if let Some(temperature) = options.and_then(|options| options.base.temperature) {
        params.insert("temperature".to_owned(), json!(temperature));
    }

    // Deferred tools: in Kimi mode the transcript-loaded definitions are dropped
    // from the request tools and replayed as system tool messages.
    let deferred_tool_names: HashSet<String> =
        if compat.deferred_tools_mode == Some(DeferredToolsMode::Kimi) {
            context
                .messages
                .iter()
                .filter_map(Message::as_tool_result)
                .flat_map(|result| result.added_tool_names.iter().flatten().cloned())
                .collect()
        } else {
            HashSet::new()
        };
    let active_tools: Vec<Tool> = context
        .tools
        .iter()
        .flatten()
        .filter(|tool| !deferred_tool_names.contains(&tool.name))
        .cloned()
        .collect();
    if !active_tools.is_empty() {
        let tools = convert_tools(&active_tools, compat)?;
        params.insert("tools".to_owned(), Value::Array(tools));
        if compat.zai_tool_stream {
            params.insert("tool_stream".to_owned(), json!(true));
        }
    } else if has_tool_history(&context.messages) {
        // Anthropic (via LiteLLM/proxy) requires tools when the conversation has
        // tool_calls/tool_results.
        params.insert("tools".to_owned(), json!([]));
    }

    let cache_control = compat_cache_control(compat, cache_retention);
    if let Some(cache_control) = &cache_control {
        apply_anthropic_cache_control(&mut params, cache_control);
    }

    if let Some(tool_choice) = options.and_then(|options| options.tool_choice.clone()) {
        params.insert("tool_choice".to_owned(), tool_choice);
    }

    let reasoning_effort = options.and_then(|options| options.reasoning_effort);
    let thinking_budget = resolve_clamped_thinking_budget(model, options, &params);
    apply_thinking_params(
        model,
        reasoning_effort,
        compat,
        thinking_budget,
        &mut params,
    );

    // Cap reasoning with a top-level budget field. Independent of thinkingFormat:
    // the same server can serve zai, qwen or chat-template models.
    if let Some(field) = compat.thinking_token_budget_field.or_else(|| {
        compat
            .supports_thinking_token_budget
            .then_some(ThinkingTokenBudgetField::Vllm)
    }) && let Some(budget) = thinking_budget
    {
        params.insert(
            thinking_token_budget_field_name(field).to_owned(),
            json!(budget),
        );
    }

    // OpenRouter provider routing preferences.
    if let Some(routing) = &compat.open_router_routing {
        params.insert("provider".to_owned(), json!(routing));
    }

    // Vercel AI Gateway provider routing preferences.
    if let Some(routing) = &compat.vercel_gateway_routing
        && (routing.only.is_some() || routing.order.is_some())
    {
        let mut gateway = serde_json::Map::new();
        if let Some(only) = &routing.only {
            gateway.insert("only".to_owned(), json!(only));
        }
        if let Some(order) = &routing.order {
            gateway.insert("order".to_owned(), json!(order));
        }
        params.insert("providerOptions".to_owned(), json!({ "gateway": gateway }));
    }

    // Last so custom keys override the named request fields.
    if let Some(sampling_params) = options.and_then(|options| options.base.sampling_params.clone())
    {
        for (key, value) in sampling_params {
            params.insert(key, value);
        }
    }

    Ok(Value::Object(params))
}

fn resolve_clamped_thinking_budget(
    model: &Model,
    options: Option<&OpenAICompletionsOptions>,
    params: &serde_json::Map<String, Value>,
) -> Option<u64> {
    let options = options?;
    let reasoning_effort = options.reasoning_effort?;
    if !model.reasoning {
        return None;
    }
    let ceiling = params
        .get("max_tokens")
        .and_then(Value::as_u64)
        .or_else(|| params.get("max_completion_tokens").and_then(Value::as_u64))
        .unwrap_or(model.max_tokens);
    let budget = clamp_thinking_budget_to_answer_room(
        thinking_budget_for_level(
            reasoning_effort.to_thinking_level(),
            options.thinking_budgets.as_ref(),
        ),
        ceiling,
    );
    if budget > 0 { Some(budget) } else { None }
}

/// The per-format thinking/reasoning request fields.
fn apply_thinking_params(
    model: &Model,
    reasoning_effort: Option<ReasoningEffort>,
    compat: &ResolvedOpenAICompletionsCompat,
    thinking_budget: Option<u64>,
    params: &mut serde_json::Map<String, Value>,
) {
    let nullish_effort = |effort: ReasoningEffort| {
        nullish_mapped_level(model, effort.to_model_thinking_level())
            .unwrap_or_else(|| effort.as_str().to_owned())
    };
    let off_is_null = || mapped_level_value(model, ModelThinkingLevel::Off) == Some(None);

    match compat.thinking_format {
        CompletionsThinkingFormat::Zai if model.reasoning => {
            params.insert(
                "thinking".to_owned(),
                if reasoning_effort.is_some() {
                    json!({ "type": "enabled", "clear_thinking": false })
                } else {
                    json!({ "type": "disabled" })
                },
            );
            if let Some(effort) = reasoning_effort
                && compat.supports_reasoning_effort
                && let Some(value) = resolve_effort(model, effort)
            {
                params.insert("reasoning_effort".to_owned(), json!(value));
            }
        }
        CompletionsThinkingFormat::Qwen if model.reasoning => {
            params.insert(
                "enable_thinking".to_owned(),
                json!(reasoning_effort.is_some()),
            );
            if let Some(effort) = reasoning_effort
                && compat.supports_reasoning_effort
            {
                params.insert("reasoning_effort".to_owned(), json!(nullish_effort(effort)));
            }
        }
        CompletionsThinkingFormat::QwenChatTemplate if model.reasoning => {
            params.insert(
                "chat_template_kwargs".to_owned(),
                json!({ "enable_thinking": reasoning_effort.is_some(), "preserve_thinking": true }),
            );
        }
        CompletionsThinkingFormat::ChatTemplate if model.reasoning => {
            if let Some(values) = build_chat_template_values(
                model,
                reasoning_effort,
                &compat.chat_template_kwargs,
                thinking_budget,
            ) {
                params.insert("chat_template_kwargs".to_owned(), Value::Object(values));
            }
        }
        CompletionsThinkingFormat::Baseten if model.reasoning => {
            if let Some(args) = build_chat_template_values(
                model,
                reasoning_effort,
                &compat.chat_template_args,
                thinking_budget,
            ) {
                params.insert("chat_template_args".to_owned(), Value::Object(args));
            }
            if compat.supports_reasoning_effort {
                // `mappedEffort === undefined ? requestedEffort : mappedEffort`: a
                // null mapping suppresses the field.
                let effort = match reasoning_effort {
                    Some(effort) => resolve_effort(model, effort),
                    None => mapped_level_value(model, ModelThinkingLevel::Off).flatten(),
                };
                if let Some(effort) = effort {
                    params.insert("reasoning_effort".to_owned(), json!(effort));
                }
            }
        }
        CompletionsThinkingFormat::Deepseek if model.reasoning => {
            if reasoning_effort.is_some() {
                params.insert("thinking".to_owned(), json!({ "type": "enabled" }));
            } else if !off_is_null() {
                params.insert("thinking".to_owned(), json!({ "type": "disabled" }));
            }
            if let Some(effort) = reasoning_effort
                && compat.supports_reasoning_effort
            {
                params.insert("reasoning_effort".to_owned(), json!(nullish_effort(effort)));
            }
        }
        CompletionsThinkingFormat::Openrouter if model.reasoning => {
            // OpenRouter normalizes reasoning across providers via a nested object.
            if let Some(effort) = reasoning_effort {
                params.insert(
                    "reasoning".to_owned(),
                    json!({ "effort": nullish_effort(effort) }),
                );
            } else if !off_is_null() {
                let effort = nullish_mapped_level(model, ModelThinkingLevel::Off)
                    .unwrap_or_else(|| "none".to_owned());
                params.insert("reasoning".to_owned(), json!({ "effort": effort }));
            }
        }
        CompletionsThinkingFormat::AntLing if model.reasoning => {
            if let Some(effort) = reasoning_effort
                && let Some(Some(value)) =
                    mapped_level_value(model, effort.to_model_thinking_level())
            {
                params.insert("reasoning".to_owned(), json!({ "effort": value }));
            }
        }
        CompletionsThinkingFormat::Together if model.reasoning => {
            params.insert(
                "reasoning".to_owned(),
                json!({ "enabled": reasoning_effort.is_some() }),
            );
            if let Some(effort) = reasoning_effort
                && compat.supports_reasoning_effort
            {
                params.insert("reasoning_effort".to_owned(), json!(nullish_effort(effort)));
            }
        }
        CompletionsThinkingFormat::StringThinking if model.reasoning => {
            if let Some(effort) = reasoning_effort {
                params.insert("thinking".to_owned(), json!(nullish_effort(effort)));
            } else if !off_is_null() {
                let effort = nullish_mapped_level(model, ModelThinkingLevel::Off)
                    .unwrap_or_else(|| "none".to_owned());
                params.insert("thinking".to_owned(), json!(effort));
            }
        }
        // OpenAI-style reasoning_effort.
        _ if model.reasoning && compat.supports_reasoning_effort => {
            if let Some(effort) = reasoning_effort {
                params.insert("reasoning_effort".to_owned(), json!(nullish_effort(effort)));
            } else if let Some(Some(off)) = mapped_level_value(model, ModelThinkingLevel::Off) {
                params.insert("reasoning_effort".to_owned(), json!(off));
            }
        }
        _ => {}
    }
}

fn build_chat_template_values(
    model: &Model,
    reasoning_effort: Option<ReasoningEffort>,
    values: &HashMap<String, ChatTemplateKwargValue>,
    thinking_budget: Option<u64>,
) -> Option<serde_json::Map<String, Value>> {
    let mut resolved = serde_json::Map::new();
    for (key, value) in values {
        if let Some(resolved_value) =
            resolve_chat_template_kwarg_value(model, reasoning_effort, value, thinking_budget)
        {
            resolved.insert(key.clone(), resolved_value);
        }
    }
    if resolved.is_empty() {
        None
    } else {
        Some(resolved)
    }
}

fn resolve_chat_template_kwarg_value(
    model: &Model,
    reasoning_effort: Option<ReasoningEffort>,
    value: &ChatTemplateKwargValue,
    thinking_budget: Option<u64>,
) -> Option<Value> {
    // The wire form is `{"$var": "thinking.enabled", "omitWhenOff": true}`. The
    // typed `Var` variant is preferred; the dotted var names also arrive as a
    // plain literal object (the foundation enum renames them snake_case), so
    // both shapes are resolved here.
    if let ChatTemplateKwargValue::Var { var, omit_when_off } = value {
        let var = match var {
            ChatTemplateVar::ThinkingEnabled => "thinking.enabled",
            ChatTemplateVar::ThinkingBudget => "thinking.budget",
            ChatTemplateVar::ThinkingEffort => "thinking.effort",
        };
        return resolve_template_var(
            model,
            reasoning_effort,
            var,
            omit_when_off.unwrap_or(false),
            thinking_budget,
        );
    }

    let Some(object) = value.as_literal().and_then(Value::as_object) else {
        return value.as_literal().cloned();
    };
    let Some(var) = object.get("$var").and_then(Value::as_str) else {
        return value.as_literal().cloned();
    };
    let omit_when_off = object
        .get("omitWhenOff")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    resolve_template_var(model, reasoning_effort, var, omit_when_off, thinking_budget)
}

/// Substitute one chat-template variable with a provider-controlled value.
fn resolve_template_var(
    model: &Model,
    reasoning_effort: Option<ReasoningEffort>,
    var: &str,
    omit_when_off: bool,
    thinking_budget: Option<u64>,
) -> Option<Value> {
    if reasoning_effort.is_none() && omit_when_off {
        return None;
    }
    match var {
        "thinking.enabled" => Some(json!(reasoning_effort.is_some())),
        "thinking.budget" => thinking_budget.map(|budget| json!(budget)),
        // Any other variable resolves through the thinking level map.
        _ => {
            let mapped = match reasoning_effort {
                Some(effort) => mapped_level_value(model, effort.to_model_thinking_level()),
                None => mapped_level_value(model, ModelThinkingLevel::Off),
            };
            match mapped {
                None => reasoning_effort.map(|effort| json!(effort.as_str())),
                Some(Some(value)) => Some(json!(value)),
                Some(None) => None,
            }
        }
    }
}

fn compat_cache_control(
    compat: &ResolvedOpenAICompletionsCompat,
    cache_retention: CacheRetention,
) -> Option<Value> {
    if compat.cache_control_format != Some(CacheControlFormat::Anthropic)
        || cache_retention == CacheRetention::None
    {
        return None;
    }
    let ttl = cache_retention == CacheRetention::Long && compat.supports_long_cache_retention;
    if ttl {
        Some(json!({ "type": "ephemeral", "ttl": "1h" }))
    } else {
        Some(json!({ "type": "ephemeral" }))
    }
}

/// Attach Anthropic-style `cache_control` markers to the system prompt, the
/// last tool, and the last conversation message.
fn apply_anthropic_cache_control(
    params: &mut serde_json::Map<String, Value>,
    cache_control: &Value,
) {
    add_cache_control_to_system_prompt(params, cache_control);
    add_cache_control_to_last_tool(params, cache_control);
    add_cache_control_to_last_conversation_message(params, cache_control);
}

fn add_cache_control_to_system_prompt(
    params: &mut serde_json::Map<String, Value>,
    cache_control: &Value,
) {
    let Some(Value::Array(messages)) = params.get_mut("messages") else {
        return;
    };
    for message in messages.iter_mut() {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if (role == "system" || role == "developer")
            && add_cache_control_to_text_content(message, cache_control)
        {
            return;
        }
    }
}

fn add_cache_control_to_last_tool(
    params: &mut serde_json::Map<String, Value>,
    cache_control: &Value,
) {
    let Some(Value::Array(tools)) = params.get_mut("tools") else {
        return;
    };
    if let Some(last) = tools.last_mut().and_then(Value::as_object_mut) {
        last.insert("cache_control".to_owned(), cache_control.clone());
    }
}

fn add_cache_control_to_last_conversation_message(
    params: &mut serde_json::Map<String, Value>,
    cache_control: &Value,
) {
    let Some(Value::Array(messages)) = params.get_mut("messages") else {
        return;
    };
    for message in messages.iter_mut().rev() {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if matches!(role, "user" | "assistant" | "tool")
            && add_cache_control_to_text_content(message, cache_control)
        {
            return;
        }
    }
}

/// Attach `cache_control` to the last text part of a message. Returns whether a
/// marker was placed.
fn add_cache_control_to_text_content(message: &mut Value, cache_control: &Value) -> bool {
    let Some(object) = message.as_object_mut() else {
        return false;
    };
    match object.get_mut("content") {
        Some(Value::String(text)) => {
            if text.is_empty() {
                return false;
            }
            let text = std::mem::take(text);
            object.insert(
                "content".to_owned(),
                json!([{ "type": "text", "text": text, "cache_control": cache_control }]),
            );
            true
        }
        Some(Value::Array(parts)) => {
            for part in parts.iter_mut().rev() {
                if part.get("type").and_then(Value::as_str) == Some("text")
                    && let Some(part) = part.as_object_mut()
                {
                    part.insert("cache_control".to_owned(), cache_control.clone());
                    return true;
                }
            }
            false
        }
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Message conversion
// ---------------------------------------------------------------------------

/// Sanitize a tool call id to the charset OpenAI accepts.
fn sanitize_tool_call_id_part(part: &str) -> String {
    part.chars()
        .map(|char| {
            if char.is_ascii_alphanumeric() || char == '_' || char == '-' {
                char
            } else {
                '_'
            }
        })
        .collect()
}

/// Handle pipe-separated ids from the OpenAI Responses API
/// (`{call_id}|{item_id}`), preserving item-level uniqueness for Chat
/// Completions, which requires distinct tool call ids.
fn normalize_tool_call_id(id: &str, provider: &str) -> String {
    if let Some(separator) = id.find('|') {
        let call_id = sanitize_tool_call_id_part(&id[..separator]);
        let item_id = sanitize_tool_call_id_part(&id[separator + 1..]);
        let combined = if item_id.is_empty() {
            call_id.clone()
        } else {
            format!("{call_id}_{item_id}")
        };
        if combined.chars().count() <= 40 {
            return combined;
        }
        let hash = short_hash(id);
        let hash = &hash[..hash.len().min(8)];
        let prefix: String = call_id
            .chars()
            .take(40usize.saturating_sub(hash.len() + 1).max(1))
            .collect();
        return format!("{prefix}_{hash}");
    }
    if provider == ProviderId::OPENAI {
        return id.chars().take(40).collect();
    }
    id.to_owned()
}

fn has_tool_history(messages: &[Message]) -> bool {
    messages.iter().any(|message| match message {
        Message::ToolResult(_) => true,
        Message::Assistant(assistant) => assistant
            .content
            .iter()
            .any(|block| block.tool_call().is_some()),
        Message::User(_) => false,
    })
}

fn get_tools_by_name(tools: Option<&[Tool]>, names: &[String]) -> Vec<Tool> {
    let Some(tools) = tools else {
        return Vec::new();
    };
    let by_name: HashMap<&str, &Tool> = tools
        .iter()
        .map(|tool| (tool.name.as_str(), tool))
        .collect();
    let mut seen = HashSet::new();
    names
        .iter()
        .filter(|name| seen.insert(name.as_str()))
        .filter_map(|name| by_name.get(name.as_str()).map(|tool| (*tool).clone()))
        .collect()
}

/// Convert the conversation into `chat.completions` messages.
pub fn convert_messages(
    model: &Model,
    context: &Context,
    compat: &ResolvedOpenAICompletionsCompat,
    grammar_tool_input_properties: &HashMap<String, String>,
) -> Result<Vec<Value>, String> {
    let mut params: Vec<Value> = Vec::new();
    let provider = model.provider.as_str();
    let transformed = transform_messages(
        &context.messages,
        model,
        Some(&|id, _, _| normalize_tool_call_id(id, provider)),
    );

    if let Some(system_prompt) = context
        .system_prompt
        .as_deref()
        .filter(|prompt| !prompt.is_empty())
    {
        let role = if model.reasoning && compat.supports_developer_role {
            "developer"
        } else {
            "system"
        };
        params.push(json!({ "role": role, "content": sanitize_surrogates(system_prompt) }));
    }

    let mut last_role: Option<&str> = None;
    let mut index = 0;
    while index < transformed.len() {
        let message = &transformed[index];
        // Some providers don't allow user messages directly after tool results;
        // insert a synthetic assistant message to bridge the gap.
        if compat.requires_assistant_after_tool_result
            && last_role == Some("toolResult")
            && matches!(message, Message::User(_))
        {
            params.push(
                json!({ "role": "assistant", "content": "I have processed the tool results." }),
            );
        }

        match message {
            Message::User(user) => {
                match &user.content {
                    UserContent::Text(text) => {
                        params
                            .push(json!({ "role": "user", "content": sanitize_surrogates(text) }));
                    }
                    UserContent::Blocks(blocks) => {
                        let content: Vec<Value> = blocks
                            .iter()
                            .map(|block| match block {
                                UserContentBlock::Text(text) => {
                                    json!({ "type": "text", "text": sanitize_surrogates(&text.text) })
                                }
                                UserContentBlock::Image(image) => json!({
                                    "type": "image_url",
                                    "image_url": { "url": format!("data:{};base64,{}", image.mime_type, image.data) },
                                }),
                            })
                            .collect();
                        if content.is_empty() {
                            index += 1;
                            continue;
                        }
                        params.push(json!({ "role": "user", "content": content }));
                    }
                }
                last_role = Some("user");
            }
            Message::Assistant(assistant) => {
                if let Some(message) = convert_assistant_message(
                    assistant,
                    model,
                    compat,
                    grammar_tool_input_properties,
                )? {
                    params.push(message);
                }
                last_role = Some("assistant");
            }
            Message::ToolResult(_) => {
                let mut image_blocks: Vec<Value> = Vec::new();
                let mut deferred_tool_names: Vec<String> = Vec::new();
                let mut cursor = index;

                while cursor < transformed.len()
                    && let Some(tool_message) = transformed[cursor].as_tool_result()
                {
                    let text_result: String = tool_message
                        .content
                        .iter()
                        .filter_map(|block| block.text())
                        .map(|text| text.text.as_str())
                        .collect::<Vec<_>>()
                        .join("\n");
                    let has_images = tool_message
                        .content
                        .iter()
                        .any(|block| block.image().is_some());

                    // Always send a tool result with text (or a placeholder when
                    // the result only carries images).
                    let tool_result_text = if !text_result.is_empty() {
                        text_result
                    } else if has_images {
                        "(see attached image)".to_owned()
                    } else {
                        "(no tool output)".to_owned()
                    };
                    let mut tool_result = serde_json::Map::new();
                    tool_result.insert("role".to_owned(), json!("tool"));
                    tool_result.insert(
                        "content".to_owned(),
                        json!(sanitize_surrogates(&tool_result_text)),
                    );
                    tool_result.insert("tool_call_id".to_owned(), json!(tool_message.tool_call_id));
                    if compat.requires_tool_result_name {
                        tool_result.insert("name".to_owned(), json!(tool_message.tool_name));
                    }
                    params.push(Value::Object(tool_result));

                    if compat.deferred_tools_mode == Some(DeferredToolsMode::Kimi) {
                        deferred_tool_names
                            .extend(tool_message.added_tool_names.iter().flatten().cloned());
                    }

                    if has_images && model.supports_images() {
                        for block in &tool_message.content {
                            if let Some(image) = block.image() {
                                image_blocks.push(json!({
                                    "type": "image_url",
                                    "image_url": { "url": format!("data:{};base64,{}", image.mime_type, image.data) },
                                }));
                            }
                        }
                    }
                    cursor += 1;
                }

                index = cursor - 1;

                if !image_blocks.is_empty() {
                    if compat.requires_assistant_after_tool_result {
                        params.push(json!({ "role": "assistant", "content": "I have processed the tool results." }));
                    }
                    let mut content = vec![
                        json!({ "type": "text", "text": "Attached image(s) from tool result:" }),
                    ];
                    content.extend(image_blocks);
                    params.push(json!({ "role": "user", "content": content }));
                    last_role = Some("user");
                } else {
                    last_role = Some("toolResult");
                }

                if !deferred_tool_names.is_empty() {
                    let deferred_tools =
                        get_tools_by_name(context.tools.as_deref(), &deferred_tool_names);
                    if !deferred_tools.is_empty() {
                        // Kimi accepts a system message with tools but omits the
                        // standard content field.
                        params.push(json!({
                            "role": "system",
                            "tools": convert_tools(&deferred_tools, compat)?,
                        }));
                    }
                }
            }
        }

        index += 1;
    }

    Ok(params)
}

fn convert_assistant_message(
    assistant: &AssistantMessage,
    model: &Model,
    compat: &ResolvedOpenAICompletionsCompat,
    grammar_tool_input_properties: &HashMap<String, String>,
) -> Result<Option<Value>, String> {
    let mut message = serde_json::Map::new();
    message.insert("role".to_owned(), json!("assistant"));
    // Some providers don't accept null content; use an empty string instead.
    message.insert(
        "content".to_owned(),
        if compat.requires_assistant_after_tool_result {
            json!("")
        } else {
            Value::Null
        },
    );

    let assistant_text_parts: Vec<Value> = assistant
        .content
        .iter()
        .filter_map(AssistantContent::text)
        .filter(|block| !block.text.trim().is_empty())
        .map(|block| json!({ "type": "text", "text": sanitize_surrogates(&block.text) }))
        .collect();
    let assistant_text: String = assistant_text_parts
        .iter()
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect();

    let thinking_blocks: Vec<&ThinkingContent> = assistant
        .content
        .iter()
        .filter_map(AssistantContent::thinking)
        .collect();
    let tool_calls: Vec<&ToolCall> = assistant
        .content
        .iter()
        .filter_map(AssistantContent::tool_call)
        .collect();

    let signed_reasoning_details = thinking_blocks
        .iter()
        .find_map(|block| parse_openai_reasoning_details(block.thinking_signature.as_deref()));
    let legacy_reasoning_details: Vec<Value> = tool_calls
        .iter()
        .filter_map(|call| {
            parse_legacy_encrypted_reasoning_detail(call.thought_signature.as_deref())
        })
        .collect();
    let preserved_reasoning_details = signed_reasoning_details.or_else(|| {
        if legacy_reasoning_details.is_empty() {
            None
        } else {
            Some(legacy_reasoning_details)
        }
    });

    let non_empty_thinking_blocks: Vec<&&ThinkingContent> = thinking_blocks
        .iter()
        .filter(|block| !block.thinking.trim().is_empty())
        .collect();

    if !non_empty_thinking_blocks.is_empty() {
        if compat.requires_thinking_as_text {
            // Convert thinking blocks to plain text (no tags, to avoid the model
            // mimicking them).
            let thinking_text = non_empty_thinking_blocks
                .iter()
                .map(|block| sanitize_surrogates(&block.thinking))
                .collect::<Vec<_>>()
                .join("\n\n");
            let mut content = vec![json!({ "type": "text", "text": thinking_text })];
            content.extend(assistant_text_parts);
            message.insert("content".to_owned(), Value::Array(content));
        } else {
            // Always send assistant content as a plain string (the Chat
            // Completions standard); content-block arrays make some models
            // mirror the structure literally.
            if !assistant_text.is_empty() {
                message.insert("content".to_owned(), json!(assistant_text));
            }

            // `reasoning_details` is the structured alternative to a raw
            // reasoning field.
            if preserved_reasoning_details.is_none() {
                // Use the signature from the first thinking block when available
                // (llama.cpp server + gpt-oss).
                let mut signature = non_empty_thinking_blocks[0].thinking_signature.clone();
                if model.provider.as_str() == "opencode-go"
                    && signature.as_deref() == Some("reasoning")
                {
                    signature = Some("reasoning_content".to_owned());
                }
                if let Some(signature) = signature.filter(|field| is_reasoning_field(field)) {
                    let joined = non_empty_thinking_blocks
                        .iter()
                        .map(|block| block.thinking.as_str())
                        .collect::<Vec<_>>()
                        .join("\n");
                    message.insert(signature, json!(joined));
                }
            }
        }
    } else if !assistant_text.is_empty() {
        message.insert("content".to_owned(), json!(assistant_text));
    }

    if !tool_calls.is_empty() {
        let calls: Vec<Value> = tool_calls
            .iter()
            .map(|call| {
                if let Some(property) = grammar_tool_input_properties.get(&call.name) {
                    Ok(json!({
                        "id": call.id,
                        "type": "custom",
                        "custom": {
                            "name": call.name,
                            "input": sanitize_surrogates(&get_grammar_tool_input(&call.name, &call.arguments, property)?),
                        },
                    }))
                } else {
                    Ok(json!({
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".to_owned()),
                        },
                    }))
                }
            })
            .collect::<Result<_, String>>()?;
        message.insert("tool_calls".to_owned(), Value::Array(calls));
    }

    if let Some(details) = preserved_reasoning_details {
        message.insert("reasoning_details".to_owned(), Value::Array(details));
    }
    if compat.requires_reasoning_content_on_assistant_messages
        && model.reasoning
        && message.get("reasoning_content").is_none()
    {
        message.insert("reasoning_content".to_owned(), json!(""));
    }

    // Skip assistant messages with no content and no tool calls: some providers
    // require "either content or tool_calls, but not none" (e.g. aborted turns).
    let has_content = match message.get("content") {
        Some(Value::String(text)) => !text.is_empty(),
        Some(Value::Array(parts)) => !parts.is_empty(),
        _ => false,
    };
    if !has_content && message.get("tool_calls").is_none() {
        return Ok(None);
    }

    Ok(Some(Value::Object(message)))
}

fn is_reasoning_field(field: &str) -> bool {
    matches!(field, "reasoning" | "reasoning_content" | "reasoning_text")
}

/// Convert tools into `chat.completions` tool definitions.
pub fn convert_tools(
    tools: &[Tool],
    compat: &ResolvedOpenAICompletionsCompat,
) -> Result<Vec<Value>, String> {
    tools
        .iter()
        .map(|tool| {
            if let Some(grammar) =
                resolve_grammar_constrained_sampling(tool, compat.supports_openai_grammar_tools)?
            {
                let syntax = match grammar.format {
                    GrammarFormat::OpenaiLark => GrammarSyntax::Lark,
                    GrammarFormat::OpenaiRegex => GrammarSyntax::Regex,
                };
                let wire = ChatCompletionTools::Custom(CustomToolChatCompletions {
                    custom: CustomToolProperties {
                        name: tool.name.clone(),
                        description: Some(tool.description.clone()),
                        format: CustomToolPropertiesFormat::Grammar {
                            grammar: CustomGrammarFormatParam {
                                syntax,
                                definition: grammar.definition,
                            },
                        },
                    },
                });
                return serde_json::to_value(wire).map_err(|error| error.to_string());
            }

            let strict = resolve_json_schema_strict_sampling(tool, compat.supports_strict_mode)?;
            let wire = ChatCompletionTools::Function(ChatCompletionTool {
                function: FunctionObject {
                    name: tool.name.clone(),
                    description: Some(tool.description.clone()),
                    parameters: Some(get_json_schema_tool_parameters(tool, strict)),
                    // Omit unsupported fields for compatible providers.
                    strict: compat
                        .supports_strict_mode
                        .then_some(strict.unwrap_or(false)),
                },
            });
            serde_json::to_value(wire).map_err(|error| error.to_string())
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Reasoning details (OpenRouter replay metadata)
// ---------------------------------------------------------------------------

fn is_reasoning_detail_object(value: &Value) -> bool {
    value.is_object()
}

fn has_valid_common_reasoning_detail_fields(candidate: &serde_json::Map<String, Value>) -> bool {
    let id_ok = candidate
        .get("id")
        .is_none_or(|value| value.is_null() || value.is_string());
    let format_ok = candidate.get("format").is_none_or(Value::is_string);
    let index_ok = candidate.get("index").is_none_or(Value::is_number);
    id_ok && format_ok && index_ok
}

fn is_openai_reasoning_detail(detail: &Value) -> bool {
    let Some(object) = detail.as_object() else {
        return false;
    };
    if !is_reasoning_detail_object(detail) || !has_valid_common_reasoning_detail_fields(object) {
        return false;
    }
    match object.get("type").and_then(Value::as_str) {
        Some("reasoning.summary") => object.get("summary").is_some_and(Value::is_string),
        Some("reasoning.encrypted") => object.get("data").is_some_and(Value::is_string),
        Some("reasoning.text") => {
            object.get("text").is_some_and(Value::is_string)
                && object
                    .get("signature")
                    .is_none_or(|value| value.is_null() || value.is_string())
        }
        _ => false,
    }
}

fn parse_openai_reasoning_details(signature: Option<&str>) -> Option<Vec<Value>> {
    let signature = signature?;
    let parsed: Value = serde_json::from_str(signature).ok()?;
    let details = parsed.as_array()?;
    if details.is_empty() || !details.iter().all(is_openai_reasoning_detail) {
        return None;
    }
    Some(details.clone())
}

fn parse_legacy_encrypted_reasoning_detail(signature: Option<&str>) -> Option<Value> {
    let signature = signature?;
    let parsed: Value = serde_json::from_str(signature).ok()?;
    if !is_openai_reasoning_detail(&parsed) {
        return None;
    }
    let object = parsed.as_object()?;
    if object.get("type").and_then(Value::as_str) != Some("reasoning.encrypted") {
        return None;
    }
    let id = object
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())?;
    let data = object
        .get("data")
        .and_then(Value::as_str)
        .filter(|data| !data.is_empty())?;
    let _ = (id, data);
    Some(parsed)
}

fn fill_missing_common_reasoning_detail_fields(
    target: &mut serde_json::Map<String, Value>,
    source: &Value,
) {
    let source = source.as_object();
    let is_nullish = |value: Option<&Value>| matches!(value, None | Some(Value::Null));
    if is_nullish(target.get("id"))
        && let Some(id) = source.and_then(|source| source.get("id"))
        && !id.is_null()
    {
        target.insert("id".to_owned(), id.clone());
    }
    // `format ||= source.format` — an empty string is falsy too.
    let format_falsy = target
        .get("format")
        .is_none_or(|value| value.is_null() || value.as_str() == Some(""));
    if format_falsy
        && let Some(format) = source.and_then(|source| source.get("format"))
        && !format.is_null()
    {
        target.insert("format".to_owned(), format.clone());
    }
    if is_nullish(target.get("index"))
        && let Some(index) = source.and_then(|source| source.get("index"))
        && !index.is_null()
    {
        target.insert("index".to_owned(), index.clone());
    }
}

/// Append a reasoning detail, merging consecutive text/summary deltas into
/// logical entries while encrypted entries stay discrete.
fn append_openai_reasoning_detail(details: &mut Vec<Value>, detail: Value) {
    let detail_type = detail
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if let Some(Value::Object(last)) = details.last_mut() {
        let last_type = last.get("type").and_then(Value::as_str).unwrap_or_default();
        if detail_type == "reasoning.text" && last_type == "reasoning.text" {
            if let Some(Value::String(last_text)) = last.get_mut("text")
                && let Some(text) = detail.get("text").and_then(Value::as_str)
            {
                last_text.push_str(text);
            }
            if last
                .get("signature")
                .is_none_or(|value| value.is_null() || value.as_str() == Some(""))
                && let Some(signature) = detail.get("signature")
                && !signature.is_null()
            {
                last.insert("signature".to_owned(), signature.clone());
            }
            fill_missing_common_reasoning_detail_fields(last, &detail);
            return;
        }
        if detail_type == "reasoning.summary" && last_type == "reasoning.summary" {
            if let Some(Value::String(last_summary)) = last.get_mut("summary")
                && let Some(summary) = detail.get("summary").and_then(Value::as_str)
            {
                last_summary.push_str(summary);
            }
            fill_missing_common_reasoning_detail_fields(last, &detail);
            return;
        }
    }
    details.push(detail);
}

// ---------------------------------------------------------------------------
// Usage and stop reasons
// ---------------------------------------------------------------------------

/// Parse a chunk `usage` object into assistant usage, computing cost.
pub fn parse_chunk_usage(raw_usage: &Value, model: &Model) -> Usage {
    let token = |key: &str| {
        raw_usage
            .get(key)
            .and_then(crate::types::json_as_token_count)
    };
    let prompt_tokens = token("prompt_tokens").unwrap_or(0);
    // Providers disagree on placement: OpenAI/OpenRouter use
    // prompt_tokens_details.cached_tokens, DeepSeek uses prompt_cache_hit_tokens,
    // and Kimi documents top-level usage.cached_tokens.
    let cache_read_tokens = raw_usage
        .get("prompt_tokens_details")
        .and_then(|details| details.get("cached_tokens"))
        .and_then(crate::types::json_as_token_count)
        .or_else(|| token("prompt_cache_hit_tokens"))
        .or_else(|| token("cached_tokens"))
        .unwrap_or(0);
    let cache_write_tokens = raw_usage
        .get("prompt_tokens_details")
        .and_then(|details| details.get("cache_write_tokens"))
        .and_then(crate::types::json_as_token_count)
        .unwrap_or(0);

    // cached_tokens is cache-read (hits); writes are counted separately and are
    // never subtracted from reads.
    let input = prompt_tokens
        .saturating_sub(cache_read_tokens)
        .saturating_sub(cache_write_tokens);
    // OpenAI completion_tokens already includes reasoning_tokens.
    let output_tokens = token("completion_tokens").unwrap_or(0);
    let reasoning_tokens = raw_usage
        .get("completion_tokens_details")
        .and_then(|details| details.get("reasoning_tokens"))
        .and_then(crate::types::json_as_token_count);
    let mut usage = Usage {
        input,
        output: output_tokens,
        cache_read: cache_read_tokens,
        cache_write: cache_write_tokens,
        cache_write_1h: None,
        reasoning: reasoning_tokens,
        total_tokens: input + output_tokens + cache_read_tokens + cache_write_tokens,
        cost: Default::default(),
    };
    calculate_cost(model, &mut usage);
    usage
}

/// Map a provider `finish_reason` to a stop reason plus an optional error message.
pub fn map_stop_reason(reason: &str) -> (StopReason, Option<String>) {
    match reason {
        "stop" | "end" => (StopReason::Stop, None),
        "length" => (StopReason::Length, None),
        "function_call" | "tool_calls" => (StopReason::ToolUse, None),
        "content_filter" => (
            StopReason::Error,
            Some("Provider finish_reason: content_filter".to_owned()),
        ),
        "network_error" => (
            StopReason::Error,
            Some("Provider finish_reason: network_error".to_owned()),
        ),
        other => (
            StopReason::Error,
            Some(format!("Provider finish_reason: {other}")),
        ),
    }
}

fn done_reason(stop_reason: StopReason) -> DoneReason {
    match stop_reason {
        StopReason::Length => DoneReason::Length,
        StopReason::ToolUse => DoneReason::ToolUse,
        StopReason::Deferred => DoneReason::Deferred,
        _ => DoneReason::Stop,
    }
}

// ---------------------------------------------------------------------------
// Streaming state machine
// ---------------------------------------------------------------------------

/// Scratch bookkeeping for a streamed tool call block. Arguments themselves live
/// in `output.content`; this holds the parse buffers.
struct ToolCallScratch {
    partial_args: Option<String>,
    custom_input: Option<CustomInputScratch>,
    stream_index: Option<u64>,
}

struct CustomInputScratch {
    property: String,
    buffer: GrammarToolInputJsonBuffer,
}

enum StreamBlock {
    Text,
    Thinking,
    ToolCall(ToolCallScratch),
}

/// Internal failure of a stream: an HTTP/transport error or a logical error.
#[derive(Debug)]
enum StreamError {
    Http(ProviderHttpError),
    Message(String),
    Aborted,
}

impl StreamError {
    fn aborted() -> Self {
        StreamError::Aborted
    }

    fn is_aborted(&self) -> bool {
        matches!(self, StreamError::Aborted)
    }

    fn display_message(&self) -> String {
        match self {
            StreamError::Http(error) => format_provider_error(error, None),
            StreamError::Message(message) => message.clone(),
            StreamError::Aborted => "Request was aborted".to_owned(),
        }
    }
}

impl From<ProviderHttpError> for StreamError {
    fn from(error: ProviderHttpError) -> Self {
        StreamError::Http(error)
    }
}

impl From<String> for StreamError {
    fn from(message: String) -> Self {
        StreamError::Message(message)
    }
}

/// Incremental state for one `chat.completions` stream: the message under
/// construction plus per-block scratch buffers.
struct StreamingState {
    snapshot_mode: crate::types::EventSnapshotMode,
    output: AssistantMessage,
    blocks: Vec<StreamBlock>,
    text_block: Option<usize>,
    thinking_block: Option<usize>,
    tool_calls_by_index: HashMap<u64, usize>,
    tool_calls_by_id: HashMap<String, usize>,
    has_finish_reason: bool,
    /// `reasoning_details` are replay metadata, not user-visible deltas: kept in
    /// memory while streaming and serialized once when the block is finalized.
    streamed_reasoning_details: Option<Vec<Value>>,
}

impl StreamingState {
    fn snapshot(&self) -> AssistantMessage {
        crate::event::snapshot_message(&self.output, self.snapshot_mode)
    }

    fn new(output: AssistantMessage) -> Self {
        StreamingState {
            snapshot_mode: Default::default(),
            output,
            blocks: Vec::new(),
            text_block: None,
            thinking_block: None,
            tool_calls_by_index: HashMap::new(),
            tool_calls_by_id: HashMap::new(),
            has_finish_reason: false,
            streamed_reasoning_details: None,
        }
    }

    fn ensure_text_block(&mut self, events: &mut Vec<AssistantMessageEvent>) -> usize {
        if let Some(index) = self.text_block {
            return index;
        }
        let index = self.blocks.len();
        self.blocks.push(StreamBlock::Text);
        self.output
            .content
            .push(AssistantContent::Text(crate::types::TextContent {
                text: String::new(),
                text_signature: None,
            }));
        self.text_block = Some(index);
        events.push(AssistantMessageEvent::TextStart {
            content_index: index,
            partial: self.snapshot(),
        });
        index
    }

    fn ensure_thinking_block(
        &mut self,
        signature: String,
        events: &mut Vec<AssistantMessageEvent>,
    ) -> usize {
        if let Some(index) = self.thinking_block {
            return index;
        }
        let index = self.blocks.len();
        self.blocks.push(StreamBlock::Thinking);
        self.output
            .content
            .push(AssistantContent::Thinking(ThinkingContent {
                thinking: String::new(),
                thinking_signature: Some(signature),
                redacted: false,
            }));
        self.thinking_block = Some(index);
        events.push(AssistantMessageEvent::ThinkingStart {
            content_index: index,
            partial: self.snapshot(),
        });
        index
    }

    fn ensure_tool_call_block(
        &mut self,
        tool_call: &Value,
        grammar_tool_input_properties: &HashMap<String, String>,
        events: &mut Vec<AssistantMessageEvent>,
    ) -> usize {
        let stream_index = tool_call.get("index").and_then(Value::as_u64);
        let id = tool_call
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let name = tool_call
            .get("function")
            .and_then(|function| function.get("name"))
            .and_then(Value::as_str)
            .or_else(|| {
                tool_call
                    .get("custom")
                    .and_then(|custom| custom.get("name"))
                    .and_then(Value::as_str)
            })
            .unwrap_or_default();

        let index = match stream_index
            .and_then(|stream_index| self.tool_calls_by_index.get(&stream_index).copied())
            .or_else(|| {
                // Deltas without a stream index are tracked by tool call id.
                if id.is_empty() {
                    None
                } else {
                    self.tool_calls_by_id.get(id).copied()
                }
            }) {
            Some(index) => index,
            None => {
                // The "input" fallback should not be taken; if the model makes up
                // a tool we don't know about, we at least have a place to stash it.
                let custom_input_property =
                    if tool_call.get("custom").is_some() && tool_call.get("function").is_none() {
                        Some(
                            grammar_tool_input_properties
                                .get(name)
                                .cloned()
                                .unwrap_or_else(|| "input".to_owned()),
                        )
                    } else {
                        None
                    };
                let index = self.blocks.len();
                let mut arguments = serde_json::Map::new();
                if let Some(property) = &custom_input_property {
                    arguments.insert(property.clone(), json!(""));
                }
                self.blocks.push(StreamBlock::ToolCall(ToolCallScratch {
                    partial_args: if custom_input_property.is_some() {
                        None
                    } else {
                        Some(String::new())
                    },
                    custom_input: custom_input_property.as_ref().map(|property| {
                        CustomInputScratch {
                            property: property.clone(),
                            buffer: GrammarToolInputJsonBuffer::default(),
                        }
                    }),
                    stream_index,
                }));
                self.output
                    .content
                    .push(AssistantContent::ToolCall(ToolCall {
                        id: id.to_owned(),
                        name: name.to_owned(),
                        arguments,
                        thought_signature: None,
                        namespace: None,
                    }));
                if let Some(stream_index) = stream_index {
                    self.tool_calls_by_index.insert(stream_index, index);
                }
                if !id.is_empty() {
                    self.tool_calls_by_id.insert(id.to_owned(), index);
                }
                events.push(AssistantMessageEvent::ToolCallStart {
                    content_index: index,
                    partial: self.snapshot(),
                });
                index
            }
        };
        if let Some(stream_index) = stream_index
            && let StreamBlock::ToolCall(scratch) = &mut self.blocks[index]
            && scratch.stream_index.is_none()
        {
            scratch.stream_index = Some(stream_index);
            self.tool_calls_by_index.insert(stream_index, index);
        }
        if !id.is_empty() {
            self.tool_calls_by_id.insert(id.to_owned(), index);
        }
        if let AssistantContent::ToolCall(call) = &mut self.output.content[index]
            && call.name.is_empty()
            && !name.is_empty()
        {
            call.name = name.to_owned();
        }
        // A custom tool call discovered without a grammar property still streams
        // its input through the grammar JSON buffer.
        if tool_call.get("custom").is_some() && tool_call.get("function").is_none() {
            let needs_custom_input = match &self.blocks[index] {
                StreamBlock::ToolCall(scratch) => scratch.custom_input.is_none(),
                _ => false,
            };
            if needs_custom_input {
                let block_name = self
                    .output
                    .content
                    .get(index)
                    .and_then(AssistantContent::tool_call)
                    .map(|call| call.name.clone())
                    .unwrap_or_default();
                let property = grammar_tool_input_properties
                    .get(&block_name)
                    .cloned()
                    .unwrap_or_else(|| "input".to_owned());
                if let StreamBlock::ToolCall(scratch) = &mut self.blocks[index] {
                    scratch.custom_input = Some(CustomInputScratch {
                        property: property.clone(),
                        buffer: GrammarToolInputJsonBuffer::default(),
                    });
                    scratch.partial_args = None;
                }
                if let AssistantContent::ToolCall(call) = &mut self.output.content[index] {
                    call.arguments = serde_json::Map::from_iter([(property, json!(""))]);
                }
            }
        }
        index
    }

    fn custom_tool_call_input(&self, index: usize) -> String {
        let Some(StreamBlock::ToolCall(scratch)) = &self.blocks.get(index) else {
            return String::new();
        };
        let Some(custom) = &scratch.custom_input else {
            return String::new();
        };
        match self
            .output
            .content
            .get(index)
            .and_then(AssistantContent::tool_call)
        {
            Some(call) => match call.arguments.get(&custom.property) {
                Some(Value::String(value)) => value.clone(),
                _ => String::new(),
            },
            None => String::new(),
        }
    }

    fn append_custom_tool_call_input(
        &mut self,
        index: usize,
        next_input: &str,
        close: bool,
    ) -> Result<Option<String>, StreamError> {
        let Some(StreamBlock::ToolCall(scratch)) = self.blocks.get_mut(index) else {
            return Ok(None);
        };
        let Some(custom) = &mut scratch.custom_input else {
            return Ok(None);
        };
        let delta = custom
            .buffer
            .append_json_delta(&custom.property, next_input, close)
            .map_err(StreamError::Message)?;
        if let Some(AssistantContent::ToolCall(call)) = self.output.content.get_mut(index) {
            call.arguments =
                serde_json::Map::from_iter([(custom.property.clone(), json!(next_input))]);
        }
        Ok(delta)
    }

    fn apply_streamed_reasoning_details(&mut self, index: usize) {
        let Some(details) = &self.streamed_reasoning_details else {
            return;
        };
        if let Some(AssistantContent::Thinking(thinking)) = self.output.content.get_mut(index) {
            thinking.thinking_signature =
                Some(serde_json::to_string(details).unwrap_or_else(|_| "[]".to_owned()));
        }
    }

    /// Apply one `chat.completion.chunk`, returning the events it produced (with
    /// partial snapshots taken after each mutation, as the source does).
    fn apply_chunk(
        &mut self,
        chunk: &Value,
        model: &Model,
        grammar_tool_input_properties: &HashMap<String, String>,
    ) -> Result<Vec<AssistantMessageEvent>, StreamError> {
        let mut events = Vec::new();

        // Each chunk in a streamed completion carries the same id.
        if let Some(id) = chunk.get("id").and_then(Value::as_str)
            && !id.is_empty()
            && self.output.response_id.is_none()
        {
            self.output.response_id = Some(id.to_owned());
        }
        if let Some(chunk_model) = chunk.get("model").and_then(Value::as_str)
            && !chunk_model.is_empty()
            && chunk_model != model.id
            && self.output.response_model.is_none()
        {
            self.output.response_model = Some(chunk_model.to_owned());
        }
        if let Some(usage) = chunk.get("usage").filter(|usage| !usage.is_null()) {
            self.output.usage = parse_chunk_usage(usage, model);
        }

        let Some(choice) = chunk
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
        else {
            return Ok(events);
        };

        // Fallback: some providers (e.g. Moonshot) return usage in
        // choice.usage instead of the standard chunk.usage.
        if chunk
            .get("usage")
            .filter(|usage| !usage.is_null())
            .is_none()
            && let Some(usage) = choice.get("usage").filter(|usage| !usage.is_null())
        {
            self.output.usage = parse_chunk_usage(usage, model);
        }

        if let Some(finish_reason) = choice.get("finish_reason").and_then(Value::as_str) {
            self.output.raw_stop_reason = Some(finish_reason.to_owned());
            let (stop_reason, error_message) = map_stop_reason(finish_reason);
            self.output.stop_reason = stop_reason;
            if let Some(error_message) = error_message {
                self.output.error_message = Some(error_message);
            }
            self.has_finish_reason = true;
        }

        let Some(delta) = choice.get("delta").filter(|delta| !delta.is_null()) else {
            return Ok(events);
        };

        if let Some(content) = delta.get("content").and_then(Value::as_str)
            && !content.is_empty()
        {
            let index = self.ensure_text_block(&mut events);
            if let Some(AssistantContent::Text(text)) = self.output.content.get_mut(index) {
                text.text.push_str(content);
            }
            events.push(AssistantMessageEvent::TextDelta {
                content_index: index,
                delta: content.to_owned(),
                partial: self.snapshot(),
            });
        }

        // Some endpoints return reasoning in reasoning_content (llama.cpp) or
        // reasoning (other OpenAI-compatible endpoints). Use the first non-empty
        // field to avoid duplication (chutes.ai returns both with the same text).
        let mut reasoning: Option<(String, String)> = None;
        for field in ["reasoning_content", "reasoning", "reasoning_text"] {
            if let Some(value) = delta.get(field).and_then(Value::as_str)
                && !value.is_empty()
            {
                reasoning = Some((field.to_owned(), value.to_owned()));
                break;
            }
        }
        if let Some((field, text)) = reasoning {
            let signature = if model.provider.as_str() == "opencode-go" && field == "reasoning" {
                "reasoning_content".to_owned()
            } else {
                field
            };
            let index = self.ensure_thinking_block(signature, &mut events);
            if let Some(AssistantContent::Thinking(thinking)) = self.output.content.get_mut(index) {
                thinking.thinking.push_str(&text);
            }
            events.push(AssistantMessageEvent::ThinkingDelta {
                content_index: index,
                delta: text,
                partial: self.snapshot(),
            });
        }

        if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
            for tool_call in tool_calls {
                let index = self.ensure_tool_call_block(
                    tool_call,
                    grammar_tool_input_properties,
                    &mut events,
                );

                let id = tool_call
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if !id.is_empty()
                    && let AssistantContent::ToolCall(call) = &mut self.output.content[index]
                    && call.id.is_empty()
                {
                    call.id = id.to_owned();
                    self.tool_calls_by_id.insert(id.to_owned(), index);
                }

                let name = tool_call
                    .get("function")
                    .and_then(|function| function.get("name"))
                    .and_then(Value::as_str)
                    .or_else(|| {
                        tool_call
                            .get("custom")
                            .and_then(|custom| custom.get("name"))
                            .and_then(Value::as_str)
                    });
                if let Some(name) = name
                    && !name.is_empty()
                    && let AssistantContent::ToolCall(call) = &mut self.output.content[index]
                    && call.name.is_empty()
                {
                    call.name = name.to_owned();
                }

                let mut delta_text = String::new();
                let function_arguments = tool_call
                    .get("function")
                    .and_then(|function| function.get("arguments"))
                    .and_then(Value::as_str)
                    .filter(|arguments| !arguments.is_empty());
                if let Some(arguments) = function_arguments {
                    delta_text = arguments.to_owned();
                    let parsed = {
                        // The block always exists: `ensure_tool_call_block` just
                        // created or resolved it.
                        let Some(StreamBlock::ToolCall(scratch)) = self.blocks.get_mut(index)
                        else {
                            continue;
                        };
                        let partial = scratch.partial_args.get_or_insert_with(String::new);
                        partial.push_str(arguments);
                        parse_streaming_json(Some(partial.as_str()))
                    };
                    if let Some(AssistantContent::ToolCall(call)) =
                        self.output.content.get_mut(index)
                    {
                        call.arguments = parsed.as_object().cloned().unwrap_or_default();
                    }
                } else if let Some(custom_input) = tool_call
                    .get("custom")
                    .and_then(|custom| custom.get("input"))
                    .and_then(Value::as_str)
                    .filter(|input| !input.is_empty())
                {
                    let next_input =
                        format!("{}{}", self.custom_tool_call_input(index), custom_input);
                    if let Some(delta) =
                        self.append_custom_tool_call_input(index, &next_input, false)?
                    {
                        delta_text = delta;
                    }
                }
                events.push(AssistantMessageEvent::ToolCallDelta {
                    content_index: index,
                    delta: delta_text,
                    partial: self.snapshot(),
                });
            }
        }

        if let Some(reasoning_details) = delta.get("reasoning_details").and_then(Value::as_array) {
            for detail in reasoning_details {
                if !is_openai_reasoning_detail(detail) {
                    continue;
                }
                self.ensure_thinking_block(String::new(), &mut events);
                // OpenRouter streams reasoning_details as deltas: consecutive
                // text/summary deltas merge into logical entries while encrypted
                // entries stay opaque and discrete.
                let details = self.streamed_reasoning_details.get_or_insert_with(Vec::new);
                append_openai_reasoning_detail(details, detail.clone());
            }
        }

        Ok(events)
    }

    /// Emit the terminal events for every open block.
    fn finish_blocks(&mut self) -> Result<Vec<AssistantMessageEvent>, StreamError> {
        let mut events = Vec::new();
        for index in 0..self.blocks.len() {
            let kind = match &self.blocks[index] {
                StreamBlock::Text => 0u8,
                StreamBlock::Thinking => 1,
                StreamBlock::ToolCall(scratch) => {
                    if scratch.custom_input.is_some() {
                        3
                    } else {
                        2
                    }
                }
            };
            match kind {
                0 => {
                    let content = self
                        .output
                        .content
                        .get(index)
                        .and_then(AssistantContent::text)
                        .map(|text| text.text.clone())
                        .unwrap_or_default();
                    events.push(AssistantMessageEvent::TextEnd {
                        content_index: index,
                        content,
                        partial: self.snapshot(),
                    });
                }
                1 => {
                    self.apply_streamed_reasoning_details(index);
                    events.push(AssistantMessageEvent::ThinkingEnd {
                        content_index: index,
                        content: thinking_text_at(&self.output.content, index),
                        partial: self.snapshot(),
                    });
                }
                3 => {
                    let next_input = self.custom_tool_call_input(index);
                    if let Ok(Some(delta)) =
                        self.append_custom_tool_call_input(index, &next_input, true)
                    {
                        events.push(AssistantMessageEvent::ToolCallDelta {
                            content_index: index,
                            delta,
                            partial: self.snapshot(),
                        });
                    }
                    if let Some(AssistantContent::ToolCall(call)) = self.output.content.get(index) {
                        events.push(AssistantMessageEvent::ToolCallEnd {
                            content_index: index,
                            tool_call: call.clone(),
                            partial: self.snapshot(),
                        });
                    }
                }
                _ => {
                    if let Some(StreamBlock::ToolCall(scratch)) = self.blocks.get_mut(index)
                        && let Some(partial_args) = &scratch.partial_args
                    {
                        let parsed = crate::utils::json_parse::parse_tool_arguments(partial_args)
                            .map_err(|error| StreamError::Message(error.to_string()))?;
                        if let Some(AssistantContent::ToolCall(call)) =
                            self.output.content.get_mut(index)
                        {
                            call.arguments = parsed;
                        }
                    }
                    if let Some(AssistantContent::ToolCall(call)) = self.output.content.get(index) {
                        events.push(AssistantMessageEvent::ToolCallEnd {
                            content_index: index,
                            tool_call: call.clone(),
                            partial: self.snapshot(),
                        });
                    }
                }
            }
        }
        Ok(events)
    }

    fn into_message(self) -> AssistantMessage {
        self.output
    }
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

fn has_header(headers: Option<&ProviderHeaders>, name: &str) -> bool {
    let Some(headers) = headers else {
        return false;
    };
    let expected = name.to_lowercase();
    headers.iter().any(|(key, value)| {
        key.to_lowercase() == expected
            && value
                .as_deref()
                .is_some_and(|value| !value.trim().is_empty())
    })
}

fn get_client_api_key(
    provider: &str,
    api_key: Option<&str>,
    headers: Option<&ProviderHeaders>,
) -> Result<String, StreamError> {
    if let Some(api_key) = api_key {
        return Ok(api_key.to_owned());
    }
    if has_header(headers, "authorization") || has_header(headers, "cf-aig-authorization") {
        return Ok("unused".to_owned());
    }
    Err(StreamError::Message(format!(
        "No API key for provider: {provider}"
    )))
}

/// Build the request headers: provider defaults, Copilot dynamic headers,
/// session affinity, then the caller's headers (which override everything).
fn build_request_headers(
    model: &Model,
    context: &Context,
    compat: &ResolvedOpenAICompletionsCompat,
    cache_session_id: Option<&str>,
    options_headers: Option<&ProviderHeaders>,
    api_key: &str,
) -> ProviderHeaders {
    let mut headers: ProviderHeaders = HashMap::new();
    headers.insert("User-Agent".to_owned(), Some(get_user_agent()));
    if let Some(model_headers) = &model.headers {
        for (name, value) in model_headers {
            headers.insert(name.clone(), Some(value.clone()));
        }
    }

    if model.provider.as_str() == ProviderId::GITHUB_COPILOT {
        let has_images = has_copilot_vision_input(&context.messages);
        for (name, value) in build_copilot_dynamic_headers(&context.messages, has_images) {
            headers.insert(name, Some(value));
        }
    }

    if let Some(session_id) = cache_session_id
        && compat.send_session_affinity_headers
    {
        match compat.session_affinity_format {
            SessionAffinityFormat::Openrouter => {
                headers.insert("x-session-id".to_owned(), Some(session_id.to_owned()));
            }
            _ => {
                if compat.session_affinity_format == SessionAffinityFormat::Openai {
                    headers.insert("session_id".to_owned(), Some(session_id.to_owned()));
                }
                headers.insert(
                    "x-client-request-id".to_owned(),
                    Some(session_id.to_owned()),
                );
                headers.insert("x-session-affinity".to_owned(), Some(session_id.to_owned()));
            }
        }
    }

    // Merge options headers last so they can override defaults.
    let merged = merge_provider_headers(Some(&headers), options_headers).unwrap_or(headers);
    let mut merged = merged;
    if !has_header(Some(&merged), "authorization")
        && !has_header(Some(&merged), "cf-aig-authorization")
    {
        merged.insert(
            "Authorization".to_owned(),
            Some(format!("Bearer {api_key}")),
        );
    }
    merged
}

/// POST the payload, returning the SSE stream. Non-2xx responses become an
/// error so [`retry_provider_request_tracked`] can apply the SDK retry policy.
async fn request_sse(
    client: reqwest::Client,
    url: &str,
    headers: &reqwest::header::HeaderMap,
    payload: &Value,
    timeout_ms: Option<u64>,
    metrics: Option<std::sync::Arc<crate::request_metrics::RequestTracker>>,
) -> Result<SseResponse, ProviderHttpError> {
    let mut response = crate::api::http::post_json_sse_tracked(client, url, headers, payload, timeout_ms, metrics).await;
    if (200..300).contains(&response.status) {
        return Ok(response);
    }
    match response.events.next().await {
        Some(Err(error)) => Err(error),
        _ => Err(ProviderHttpError::from_response(
            response.status,
            None,
            format!("{} status code", response.status),
        )),
    }
}

// ---------------------------------------------------------------------------
// Stream driver
// ---------------------------------------------------------------------------

/// Drive the SSE event stream into assistant events, terminating with `done`
/// (or `error`). Split out of [`stream`] so tests can feed a synthetic SSE
/// response without any network.
async fn run_stream_events<S>(
    mut events: S,
    model: &Model,
    compat: &ResolvedOpenAICompletionsCompat,
    grammar_tool_input_properties: &HashMap<String, String>,
    token: &tokio_util::sync::CancellationToken,
    mut state: StreamingState,
    sender: &mut AssistantMessageEventSender,
) where
    S: Stream<Item = Result<SseEvent, ProviderHttpError>> + Unpin,
{
    // The `start` event marks an empty initial message; deltas build content and
    // `done` carries the authoritative final message. Consumers may serialize
    // this event after the producer advanced, so `start` gets its own empty
    // content array rather than aliasing the live one.
    let mut start_partial = state.output.clone();
    start_partial.content = Vec::new();
    sender
        .push(AssistantMessageEvent::Start {
            partial: start_partial,
        })
        .await;

    let outcome = consume_chunks(
        &mut events,
        model,
        compat,
        grammar_tool_input_properties,
        token,
        &mut state,
        sender,
    )
    .await;

    match outcome {
        Ok(()) => {
            let reason = done_reason(state.output.stop_reason);
            let message = state.into_message();
            sender
                .push(AssistantMessageEvent::Done { reason, message })
                .await;
        }
        Err(error) => {
            // Finalize thinking signatures so replay keeps the reasoning details.
            for index in 0..state.blocks.len() {
                if matches!(state.blocks[index], StreamBlock::Thinking) {
                    state.apply_streamed_reasoning_details(index);
                }
            }
            let aborted = token.is_cancelled() || error.is_aborted();
            let mut message = state.into_message();
            message.stop_reason = if aborted {
                StopReason::Aborted
            } else {
                StopReason::Error
            };
            message.error_message = Some(error.display_message());
            sender
                .push(AssistantMessageEvent::Error {
                    reason: if aborted {
                        ErrorReason::Aborted
                    } else {
                        ErrorReason::Error
                    },
                    error: message,
                })
                .await;
        }
    }
}

async fn consume_chunks<S>(
    events: &mut S,
    model: &Model,
    compat: &ResolvedOpenAICompletionsCompat,
    grammar_tool_input_properties: &HashMap<String, String>,
    token: &tokio_util::sync::CancellationToken,
    state: &mut StreamingState,
    sender: &mut AssistantMessageEventSender,
) -> Result<(), StreamError>
where
    S: Stream<Item = Result<SseEvent, ProviderHttpError>> + Unpin,
{
    loop {
        let event = tokio::select! {
            biased;
            _ = token.cancelled() => return Err(StreamError::aborted()),
            event = events.next() => event,
        };
        let Some(event) = event else {
            break;
        };
        let event = event?;
        let data = event.data.trim();
        if data.is_empty() {
            continue;
        }
        if data == "[DONE]" {
            break;
        }
        let chunk: Value = serde_json::from_str(data)
            .map_err(|error| StreamError::Message(format!("Invalid JSON in SSE data: {error}")))?;
        let updates = state.apply_chunk(&chunk, model, grammar_tool_input_properties)?;
        if let Some(raw) = chunk.get("usage").filter(|v|!v.is_null()) { sender.report_usage(&state.output.usage, raw); }
        for event in updates {
            if !sender.push(event).await {
                return Err(StreamError::aborted());
            }
        }
    }

    for event in state.finish_blocks()? {
        if !sender.push(event).await {
            return Err(StreamError::aborted());
        }
    }

    if token.is_cancelled() {
        return Err(StreamError::aborted());
    }
    if state.output.stop_reason == StopReason::Aborted {
        return Err(StreamError::aborted());
    }
    if !state.has_finish_reason && !compat.supports_finish_reason {
        state.output.stop_reason = if state
            .output
            .content
            .iter()
            .any(|block| block.tool_call().is_some())
        {
            StopReason::ToolUse
        } else {
            StopReason::Stop
        };
    }
    if state.output.stop_reason == StopReason::Error {
        return Err(StreamError::Message(
            state
                .output
                .error_message
                .clone()
                .unwrap_or_else(|| "Provider returned an error stop reason".to_owned()),
        ));
    }
    if (compat.supports_finish_reason && !state.has_finish_reason)
        || state.output.stop_reason == StopReason::Pending
    {
        return Err(StreamError::Message(
            "Stream ended without finish_reason".to_owned(),
        ));
    }
    Ok(())
}

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
        timestamp: now_ms(),
    }
}

fn error_event(model: &Model, aborted: bool, message: String) -> AssistantMessageEvent {
    let mut output = initial_output(model);
    output.stop_reason = if aborted {
        StopReason::Aborted
    } else {
        StopReason::Error
    };
    output.error_message = Some(message);
    AssistantMessageEvent::Error {
        reason: if aborted {
            ErrorReason::Aborted
        } else {
            ErrorReason::Error
        },
        error: output,
    }
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Stream a completion from an OpenAI-compatible chat completions endpoint.
pub fn stream(
    model: &Model,
    context: &Context,
    options: &OpenAICompletionsOptions,
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

        if let Err(error) = run_stream(&model, &context, &options, &mut sender).await {
            let aborted = error.is_aborted()
                || options
                    .base
                    .base
                    .signal
                    .as_ref()
                    .is_some_and(|signal| signal.is_cancelled());
            sender
                .push(error_event(&model, aborted, error.display_message()))
                .await;
        }
    });
    stream
}

async fn run_stream(
    model: &Model,
    context: &Context,
    options: &OpenAICompletionsOptions,
    sender: &mut AssistantMessageEventSender,
) -> Result<(), StreamError> {
    let api_key = get_client_api_key(
        model.provider.as_str(),
        options.base.base.api_key.as_deref(),
        options.base.base.headers.as_ref(),
    )?;
    let compat = resolve_compat(model);
    let grammar_tool_input_properties = create_grammar_tool_input_properties(
        context.tools.as_deref(),
        compat.supports_openai_grammar_tools,
    )
    .map_err(StreamError::Message)?;
    let cache_retention = options.base.cache_retention.unwrap_or_default();
    let cache_session_id = if cache_retention == CacheRetention::None {
        None
    } else {
        options.base.session_id.as_deref()
    };
    let headers = build_request_headers(
        model,
        context,
        &compat,
        cache_session_id,
        options.base.base.headers.as_ref(),
        &api_key,
    );
    let mut params = build_params(
        model,
        context,
        Some(options),
        &compat,
        cache_retention,
        &grammar_tool_input_properties,
    )?;
    if let Some(observer) = &options.base.on_payload
        && let Some(replaced) = observer.on_payload(params.clone(), model).await
    {
        params = replaced;
    }

    let client = http_client_for(&options.base.base)?;
    let url = format!("{}/chat/completions", model.base_url.trim_end_matches('/'));
    let header_map = crate::utils::headers::provider_headers_to_header_map(&headers)
        .map_err(|error| StreamError::Message(error.to_string()))?;
    let response = retry_provider_request_tracked(
        || {
            request_sse(
                client.clone(),
                &url,
                &header_map,
                &params,
                options.base.base.timeout_ms,
                options.base.base.request_metrics.clone(),
            )
        },
        ProviderRetryOptions {
            max_retries: options.base.base.max_retries,
            max_retry_delay_ms: options.base.base.max_retry_delay_ms,
            signal: options.base.base.signal.clone(),
        },
        options.base.base.request_metrics.as_ref(),
    )
    .await?;

    if let Some(metrics) = &options.base.base.request_metrics { metrics.response(response.status); }
    if let Some(observer) = &options.base.on_response {
        observer
            .on_response(
                &ProviderResponse {
                    status: response.status,
                    headers: crate::utils::headers::header_map_to_record(&response.headers),
                },
                model,
            )
            .await;
    }

    let token = options.base.base.token();
    let output = initial_output(model);
    let mut state = StreamingState::new(output);
    state.snapshot_mode = options.base.event_snapshot_mode;
    run_stream_events(
        response.events,
        model,
        &compat,
        &grammar_tool_input_properties,
        &token,
        state,
        sender,
    )
    .await;
    Ok(())
}

/// Stream with the provider-neutral simple options: tool choice, clamped
/// reasoning level, thinking budgets, and sampling/max-token defaults.
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
    let tool_choice = options.tool_choice.map(|choice| match choice {
        crate::types::ToolChoice::Auto => json!("auto"),
        crate::types::ToolChoice::None => json!("none"),
    });
    let clamped_reasoning = options
        .reasoning
        .map(|level| crate::models::clamp_thinking_level(model, model_thinking_level(level)));
    let reasoning_effort = match clamped_reasoning {
        Some(ModelThinkingLevel::Off) | None => None,
        Some(level) => Some(ReasoningEffort::from(thinking_level(level))),
    };

    stream(
        model,
        context,
        &OpenAICompletionsOptions {
            base,
            tool_choice,
            reasoning_effort,
            thinking_budgets: options.thinking_budgets,
        },
    )
}

fn model_thinking_level(level: ThinkingLevel) -> ModelThinkingLevel {
    ReasoningEffort::from(level).to_model_thinking_level()
}

fn thinking_level(level: ModelThinkingLevel) -> ThinkingLevel {
    match level {
        ModelThinkingLevel::Off => ThinkingLevel::Minimal,
        ModelThinkingLevel::Minimal => ThinkingLevel::Minimal,
        ModelThinkingLevel::Low => ThinkingLevel::Low,
        ModelThinkingLevel::Medium => ThinkingLevel::Medium,
        ModelThinkingLevel::High => ThinkingLevel::High,
        ModelThinkingLevel::Xhigh => ThinkingLevel::Xhigh,
        ModelThinkingLevel::Max => ThinkingLevel::Max,
    }
}

/// The dialect adapter: dispatch is by `model.api`, never by provider identity.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenAICompletionsApi;

impl OpenAICompletionsApi {
    /// The wire dialect this adapter implements.
    pub fn api(&self) -> Api {
        Api::OpenaiCompletions
    }
}

impl crate::models::ProviderStreams for OpenAICompletionsApi {
    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: &StreamOptions,
    ) -> AssistantMessageEventStream {
        stream(model, context, &options_from_stream_options(options))
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

/// Construct the dialect adapter.
pub fn openai_completions_api() -> OpenAICompletionsApi {
    OpenAICompletionsApi
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ProviderStreams;
    use crate::types::{
        AssistantContent, InputModality, ModelCost, ProviderId, ProviderRequestOptions,
        TextContent, ToolChoice, ToolResultMessage, UserContent, UserMessage,
    };
    use crate::utils::json_parse::parse_json_with_repair;
    use futures::StreamExt;
    use std::sync::Arc;

    // -- helpers ------------------------------------------------------------

    fn model(provider: &str, base_url: &str, reasoning: bool) -> Model {
        Model {
            id: "test-model".to_owned(),
            name: "Test Model".to_owned(),
            api: Api::OpenaiCompletions,
            provider: ProviderId::new(provider),
            base_url: base_url.to_owned(),
            reasoning,
            thinking_level_map: None,
            input: vec![InputModality::Text],
            cost: ModelCost::default(),
            context_window: 128_000,
            max_tokens: 4096,
            sampling_params: None,
            headers: None,
            compat: None,
        }
    }

    fn openai_model() -> Model {
        model(ProviderId::OPENAI, "https://api.openai.com/v1", false)
    }

    fn compat_settings() -> crate::types::CompatSettings {
        crate::types::CompatSettings::default()
    }

    fn tool(name: &str) -> Tool {
        Tool {
            name: name.to_owned(),
            description: format!("The {name} tool"),
            parameters: json!({ "type": "object", "properties": { "value": { "type": "string" } } }),
            constrained_sampling: None,
        }
    }

    fn user(text: &str, timestamp: u64) -> Message {
        Message::User(UserMessage {
            content: UserContent::Text(text.to_owned()),
            timestamp,
        })
    }

    fn usage() -> Usage {
        Usage::default()
    }

    fn assistant(content: Vec<AssistantContent>, stop_reason: StopReason) -> Message {
        Message::Assistant(Box::new(AssistantMessage {
            content,
            api: Api::OpenaiCompletions,
            provider: ProviderId::new("openai"),
            model: "test-model".to_owned(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: usage(),
            stop_reason,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            timestamp: 2,
        }))
    }

    fn tool_call(id: &str, name: &str) -> AssistantContent {
        AssistantContent::ToolCall(ToolCall {
            id: id.to_owned(),
            name: name.to_owned(),
            arguments: Default::default(),
            thought_signature: None,
            namespace: None,
        })
    }

    fn tool_result(
        tool_call_id: &str,
        tool_name: &str,
        added_tool_names: Option<Vec<String>>,
    ) -> Message {
        Message::ToolResult(Box::new(ToolResultMessage {
            tool_call_id: tool_call_id.to_owned(),
            tool_name: tool_name.to_owned(),
            content: vec![UserContentBlock::Text(crate::types::TextContent {
                text: "done".to_owned(),
                text_signature: None,
            })],
            details: None,
            usage: None,
            added_tool_names,
            is_error: false,
            timestamp: 3,
        }))
    }

    fn grammar_tool() -> Tool {
        let mut variants = std::collections::HashMap::new();
        variants.insert(GrammarFormat::OpenaiLark, "start: /[a-z]+/".to_owned());
        Tool {
            name: "sample_tool".to_owned(),
            description: "Sample tool".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": { "payload": { "type": "string" } },
                "required": ["payload"],
                "additionalProperties": false
            }),
            constrained_sampling: Some(crate::types::ConstrainedSamplingConfig::Grammar {
                variants,
            }),
        }
    }

    fn stream_options() -> OpenAICompletionsOptions {
        OpenAICompletionsOptions {
            base: StreamOptions {
                base: ProviderRequestOptions {
                    api_key: Some("test-key".to_owned()),
                    ..Default::default()
                },
                ..Default::default()
            },
            tool_choice: None,
            reasoning_effort: None,
            thinking_budgets: None,
        }
    }

    fn build_payload(
        target: &Model,
        context: &Context,
        options: &OpenAICompletionsOptions,
    ) -> Value {
        let compat = resolve_compat(target);
        let grammar = create_grammar_tool_input_properties(
            context.tools.as_deref(),
            compat.supports_openai_grammar_tools,
        )
        .expect("grammar properties resolve");
        build_params(
            target,
            context,
            Some(options),
            &compat,
            options.base.cache_retention.unwrap_or_default(),
            &grammar,
        )
        .expect("params build")
    }

    fn simple_context() -> Context {
        Context {
            system_prompt: Some("You are a helpful assistant.".to_owned()),
            messages: vec![user("Hello", 1)],
            tools: None,
        }
    }

    fn sse_events(chunks: &[Value]) -> Vec<Result<SseEvent, ProviderHttpError>> {
        let mut events: Vec<Result<SseEvent, ProviderHttpError>> = chunks
            .iter()
            .map(|chunk| {
                Ok(SseEvent {
                    event: String::new(),
                    data: serde_json::to_string(chunk).expect("chunk serializes"),
                })
            })
            .collect();
        events.push(Ok(SseEvent {
            event: String::new(),
            data: "[DONE]".to_owned(),
        }));
        events
    }

    fn event_type(event: &AssistantMessageEvent) -> &'static str {
        match event {
            AssistantMessageEvent::Start { .. } => "start",
            AssistantMessageEvent::TextStart { .. } => "text_start",
            AssistantMessageEvent::TextDelta { .. } => "text_delta",
            AssistantMessageEvent::TextEnd { .. } => "text_end",
            AssistantMessageEvent::ThinkingStart { .. } => "thinking_start",
            AssistantMessageEvent::ThinkingDelta { .. } => "thinking_delta",
            AssistantMessageEvent::ThinkingEnd { .. } => "thinking_end",
            AssistantMessageEvent::ToolCallStart { .. } => "toolcall_start",
            AssistantMessageEvent::ToolCallDelta { .. } => "toolcall_delta",
            AssistantMessageEvent::ToolCallEnd { .. } => "toolcall_end",
            AssistantMessageEvent::Done { .. } => "done",
            AssistantMessageEvent::Error { .. } => "error",
        }
    }

    /// Drive a synthetic SSE response through the stream driver, returning the
    /// emitted event types and the final message.
    async fn drive_events_with_grammar(
        events: Vec<Result<SseEvent, ProviderHttpError>>,
        grammar: HashMap<String, String>,
    ) -> (Vec<String>, AssistantMessage) {
        let target = openai_model();
        let compat = resolve_compat(&target);
        let token = tokio_util::sync::CancellationToken::new();
        let state = StreamingState::new(initial_output(&target));
        let (mut sender, mut stream) = assistant_message_event_stream();
        tokio::spawn(async move {
            run_stream_events(
                futures::stream::iter(events),
                &target,
                &compat,
                &grammar,
                &token,
                state,
                &mut sender,
            )
            .await;
        });
        let mut types = Vec::new();
        while let Some(event) = stream.next().await {
            types.push(event_type(&event).to_owned());
        }
        let message = stream.result().await.unwrap();
        (types, message)
    }

    async fn drive_chunks(chunks: &[Value]) -> (Vec<String>, AssistantMessage) {
        drive_events_with_grammar(sse_events(chunks), HashMap::new()).await
    }

    async fn drive_chunks_with_grammar(
        chunks: &[Value],
        grammar: HashMap<String, String>,
    ) -> (Vec<String>, AssistantMessage) {
        drive_events_with_grammar(sse_events(chunks), grammar).await
    }

    fn chunk(delta: Value, finish_reason: Option<&str>) -> Value {
        let mut choice = serde_json::Map::new();
        choice.insert("index".to_owned(), json!(0));
        choice.insert("delta".to_owned(), delta);
        if let Some(finish_reason) = finish_reason {
            choice.insert("finish_reason".to_owned(), json!(finish_reason));
        }
        json!({ "id": "chatcmpl-test", "choices": [choice] })
    }

    struct PayloadCapture(std::sync::Mutex<Option<Value>>);

    impl PayloadCapture {
        fn new() -> Arc<Self> {
            Arc::new(PayloadCapture(std::sync::Mutex::new(None)))
        }

        fn take(&self) -> Value {
            self.0
                .lock()
                .expect("capture lock")
                .clone()
                .expect("payload captured")
        }
    }

    #[async_trait::async_trait]
    impl crate::types::PayloadObserver for PayloadCapture {
        async fn on_payload(&self, payload: Value, _model: &Model) -> Option<Value> {
            *self.0.lock().expect("capture lock") = Some(payload);
            None
        }
    }

    /// Run `stream` against an unreachable loopback address and capture the
    /// payload the observer sees (the request itself fails afterwards).
    async fn capture_payload(
        target: &Model,
        context: &Context,
        options: &OpenAICompletionsOptions,
    ) -> Value {
        let mut options = options.clone();
        let capture = PayloadCapture::new();
        options.base.on_payload = Some(capture.clone());
        let mut stream = stream(target, context, &options);
        while stream.next().await.is_some() {}
        capture.take()
    }

    // -- conformance suite --------------------------------------------------

    #[test]
    fn adapter_self_reports_the_openai_completions_dialect() {
        let adapter = openai_completions_api();
        assert_eq!(adapter.api(), Api::OpenaiCompletions);
        assert_eq!(adapter.api().as_str(), API);
        // Dispatch is by dialect, never by provider identity.
        assert_eq!(adapter.api(), openai_model().api);
    }

    #[tokio::test]
    async fn request_snapshot_matches_chat_completions_wire_shape() {
        let target = model(ProviderId::OPENAI, "http://127.0.0.1:9/v1", false);
        let context = Context {
            system_prompt: Some("You are a helpful assistant.".to_owned()),
            messages: vec![user("Hello", 1)],
            tools: Some(vec![tool("read")]),
        };
        let payload = capture_payload(&target, &context, &stream_options()).await;

        assert_eq!(payload["model"], json!("test-model"));
        assert_eq!(payload["stream"], json!(true));
        let roles: Vec<&str> = payload["messages"]
            .as_array()
            .expect("messages array")
            .iter()
            .map(|message| message["role"].as_str().expect("role"))
            .collect();
        assert!(roles.contains(&"user"), "roles: {roles:?}");
        assert!(
            roles
                .iter()
                .any(|role| *role == "system" || *role == "developer"),
            "roles: {roles:?}"
        );
        let tools = payload["tools"].as_array().expect("tools array");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], json!("function"));
        assert_eq!(tools[0]["function"]["name"], json!("read"));
    }

    #[tokio::test]
    async fn stream_events_follow_the_canonical_order() {
        let chunks = [
            chunk(json!({ "role": "assistant", "content": "Hello" }), None),
            json!({
                "id": "chatcmpl-test",
                "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "prompt_tokens_details": { "cached_tokens": 0 },
                    "completion_tokens_details": { "reasoning_tokens": 0 }
                }
            }),
        ];
        let (types, message) = drive_chunks(&chunks).await;

        assert_eq!(
            types,
            vec!["start", "text_start", "text_delta", "text_end", "done"]
        );
        assert_eq!(message.content.len(), 1);
        assert_eq!(message.content[0].text().expect("text block").text, "Hello");
        assert_eq!(message.stop_reason, StopReason::Stop);
        assert_eq!(message.raw_stop_reason.as_deref(), Some("stop"));
        assert!(message.error_message.is_none());
        assert_eq!(message.usage.input, 1);
        assert_eq!(message.usage.output, 1);
    }

    #[tokio::test]
    async fn wire_error_normalizes_status_and_body() {
        let error = ProviderHttpError::from_response(
            429,
            Some(
                json!({ "message": "gateway WAF blocked request", "type": "rate_limit_error" })
                    .to_string(),
            ),
            "429 status code (no body)",
        );
        let (types, message) = drive_events_with_grammar(vec![Err(error)], HashMap::new()).await;

        assert_eq!(types.last().map(String::as_str), Some("error"));
        assert_eq!(message.stop_reason, StopReason::Error);
        let error_message = message.error_message.expect("error message");
        assert!(error_message.contains("429"), "{error_message}");
        assert!(
            error_message.contains("gateway WAF blocked request"),
            "{error_message}"
        );
        // Normalization surfaces the body, not the opaque SDK placeholder.
        assert_ne!(error_message, "429 status code (no body)");
    }

    #[tokio::test]
    async fn cross_profile_reuse_keeps_one_wire_shape() {
        // Two different provider identities sharing the dialect.
        let canonical = openai_model();
        let mut gateway = model("groq", "https://api.groq.com/openai/v1", false);
        gateway.id = "llama-3.3-70b-versatile".to_owned();
        assert_ne!(canonical.provider, gateway.provider);
        assert_ne!(canonical.base_url, gateway.base_url);

        let context = Context {
            system_prompt: Some("You are a helpful assistant.".to_owned()),
            messages: vec![user("Hello", 1)],
            tools: Some(vec![tool("read")]),
        };
        let canonical_payload = build_payload(&canonical, &context, &stream_options());
        let gateway_payload = build_payload(&gateway, &context, &stream_options());

        assert_eq!(canonical_payload["model"], json!("test-model"));
        assert_eq!(gateway_payload["model"], json!("llama-3.3-70b-versatile"));
        assert_eq!(canonical_payload["stream"], json!(true));
        assert_eq!(gateway_payload["stream"], json!(true));
        assert_eq!(
            canonical_payload["tools"][0]["function"]["name"],
            json!("read")
        );
        assert_eq!(
            gateway_payload["tools"][0]["function"]["name"],
            json!("read")
        );
        assert_eq!(canonical_payload["messages"], gateway_payload["messages"]);
    }

    // -- constrained sampling ----------------------------------------------

    #[test]
    fn converts_supported_constraints_and_falls_back_when_unsupported() {
        let mut strict_tool = tool("sample_tool");
        strict_tool.constrained_sampling =
            Some(crate::types::ConstrainedSamplingConfig::JsonSchema {
                strict: crate::types::StrictMode::Prefer,
            });
        let compat = resolve_compat(&openai_model());
        let converted =
            convert_tools(&[strict_tool.clone()], &compat).expect("strict tool converts");
        assert_eq!(converted[0]["function"]["strict"], json!(true));

        // `require` errors when strict tools are unsupported.
        let mut required = strict_tool.clone();
        required.constrained_sampling = Some(crate::types::ConstrainedSamplingConfig::JsonSchema {
            strict: crate::types::StrictMode::Require,
        });
        let mut unsupported = compat.clone();
        unsupported.supports_strict_mode = false;
        let error = convert_tools(&[required], &unsupported).expect_err("require must error");
        assert!(
            error.contains("requires JSON-schema constrained sampling"),
            "{error}"
        );

        // Grammar tools become OpenAI custom tools with lark grammars.
        let mut grammar_compat = compat.clone();
        grammar_compat.supports_openai_grammar_tools = true;
        let converted =
            convert_tools(&[grammar_tool()], &grammar_compat).expect("grammar tool converts");
        assert_eq!(converted[0]["type"], json!("custom"));
        assert_eq!(converted[0]["custom"]["format"]["type"], json!("grammar"));
        assert_eq!(
            converted[0]["custom"]["format"]["grammar"]["syntax"],
            json!("lark")
        );
        assert_eq!(
            converted[0]["custom"]["format"]["grammar"]["definition"],
            json!("start: /[a-z]+/")
        );

        // Empty grammar variants error.
        let mut empty_variants = grammar_tool();
        empty_variants.constrained_sampling =
            Some(crate::types::ConstrainedSamplingConfig::Grammar {
                variants: std::collections::HashMap::new(),
            });
        let error = convert_tools(&[empty_variants], &grammar_compat)
            .expect_err("empty grammar must error");
        assert!(error.contains("no supported grammar variant"), "{error}");

        // Fallback: a plain function tool without `strict`.
        let mut fallback_compat = compat.clone();
        fallback_compat.supports_strict_mode = false;
        let converted =
            convert_tools(&[grammar_tool()], &fallback_compat).expect("fallback converts");
        assert_eq!(converted[0]["type"], json!("function"));
        assert_eq!(converted[0]["function"]["name"], json!("sample_tool"));
        assert!(converted[0]["function"].get("strict").is_none());
    }

    #[test]
    fn falls_back_when_a_schema_cannot_be_made_strict() {
        let mut unsupported_tool = tool("sample_tool");
        unsupported_tool.parameters = json!({ "type": "object", "allOf": [] });
        unsupported_tool.constrained_sampling =
            Some(crate::types::ConstrainedSamplingConfig::JsonSchema {
                strict: crate::types::StrictMode::Prefer,
            });
        let compat = resolve_compat(&openai_model());
        let converted =
            convert_tools(&[unsupported_tool.clone()], &compat).expect("fallback converts");
        assert_eq!(converted[0]["function"]["strict"], json!(false));
        assert_eq!(
            converted[0]["function"]["parameters"],
            unsupported_tool.parameters
        );

        unsupported_tool.constrained_sampling =
            Some(crate::types::ConstrainedSamplingConfig::JsonSchema {
                strict: crate::types::StrictMode::Require,
            });
        let error = convert_tools(&[unsupported_tool], &compat).expect_err("require must error");
        assert!(error.contains("allOf schemas are unsupported"), "{error}");
    }

    #[test]
    fn replays_grammar_calls_as_custom_tool_calls() {
        let target = openai_model();
        let mut compat = resolve_compat(&target);
        compat.supports_openai_grammar_tools = true;
        let grammar_properties =
            create_grammar_tool_input_properties(Some(&[grammar_tool()]), true)
                .expect("grammar properties");
        assert_eq!(
            grammar_properties.get("sample_tool").map(String::as_str),
            Some("payload")
        );

        let valid_call = AssistantContent::ToolCall(ToolCall {
            id: "call_1".to_owned(),
            name: "sample_tool".to_owned(),
            arguments: serde_json::Map::from_iter([("payload".to_owned(), json!("abc"))]),
            thought_signature: None,
            namespace: None,
        });
        let context = Context {
            messages: vec![
                user("use the tool", 1),
                assistant(vec![valid_call], StopReason::ToolUse),
                tool_result("call_1", "sample_tool", None),
            ],
            ..Default::default()
        };
        let messages =
            convert_messages(&target, &context, &compat, &grammar_properties).expect("messages");
        let assistant_message = messages
            .iter()
            .find(|message| message["role"] == "assistant")
            .expect("assistant message");
        assert_eq!(
            assistant_message["tool_calls"][0],
            json!({
                "id": "call_1",
                "type": "custom",
                "custom": { "name": "sample_tool", "input": "abc" }
            })
        );

        // A non-string grammar argument is rejected.
        for invalid in [json!({}), json!({ "payload": 42 })] {
            let call = AssistantContent::ToolCall(ToolCall {
                id: "call_1".to_owned(),
                name: "sample_tool".to_owned(),
                arguments: invalid.as_object().cloned().expect("object"),
                thought_signature: None,
                namespace: None,
            });
            let context = Context {
                messages: vec![assistant(vec![call], StopReason::ToolUse)],
                ..Default::default()
            };
            let error = convert_messages(&target, &context, &compat, &grammar_properties)
                .expect_err("invalid grammar arguments must error");
            assert_eq!(
                error,
                "Grammar tool call \"sample_tool\" requires argument \"payload\" to be a string."
            );
        }
    }

    #[tokio::test]
    async fn streams_custom_tool_call_input_as_json_deltas() {
        let chunks = [
            chunk(
                json!({
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_1",
                        "type": "custom",
                        "custom": { "name": "sample_tool", "input": "" }
                    }]
                }),
                None,
            ),
            chunk(
                json!({ "tool_calls": [{ "index": 0, "custom": { "input": "a" } }] }),
                None,
            ),
            chunk(
                json!({ "tool_calls": [{ "index": 0, "custom": { "input": "\"\\n" } }] }),
                None,
            ),
            chunk(
                json!({ "tool_calls": [{ "index": 0, "custom": { "input": "b" } }] }),
                Some("tool_calls"),
            ),
        ];
        let grammar = HashMap::from_iter([("sample_tool".to_owned(), "payload".to_owned())]);
        let (types, message) = drive_chunks_with_grammar(&chunks, grammar).await;

        assert_eq!(
            types,
            vec![
                "start",
                "toolcall_start",
                "toolcall_delta",
                "toolcall_delta",
                "toolcall_delta",
                "toolcall_delta",
                "toolcall_delta",
                "toolcall_end",
                "done"
            ]
        );
        let call = message.content[0].tool_call().expect("tool call block");
        assert_eq!(call.id, "call_1");
        assert_eq!(call.name, "sample_tool");
        // The deltas carry a quote and a literal backslash-n; the JSON buffer
        // escapes them in the streamed arguments.
        let expected: Value = serde_json::from_str(r#"{"payload": "a\"\\nb"}"#).expect("json");
        assert_eq!(Value::Object(call.arguments.clone()), expected);
        assert_eq!(call.arguments["payload"], json!("a\"\\nb"));
    }

    // -- cache retention ----------------------------------------------------

    fn retention_model(compat: crate::types::CompatSettings) -> Model {
        let mut target = model(
            "test-openai-completions",
            "https://my-proxy.example.com/v1",
            false,
        );
        target.compat = Some(compat);
        target
    }

    fn retention_options(
        cache_retention: CacheRetention,
        session_id: &str,
    ) -> OpenAICompletionsOptions {
        let mut options = stream_options();
        options.base.cache_retention = Some(cache_retention);
        options.base.session_id = Some(session_id.to_owned());
        options
    }

    #[tokio::test]
    async fn long_retention_sets_prompt_cache_key_and_retention_for_proxy_base_urls() {
        let target = retention_model(compat_settings());
        let payload = capture_payload(
            &target,
            &simple_context(),
            &retention_options(CacheRetention::Long, "session-completions"),
        )
        .await;
        assert_eq!(payload["prompt_cache_key"], json!("session-completions"));
        assert_eq!(payload["prompt_cache_retention"], json!("24h"));
    }

    #[tokio::test]
    async fn long_retention_is_omitted_when_unsupported() {
        let mut compat = compat_settings();
        compat.supports_long_cache_retention = Some(false);
        let target = retention_model(compat);
        let payload = capture_payload(
            &target,
            &simple_context(),
            &retention_options(CacheRetention::Long, "session-completions-false"),
        )
        .await;
        assert!(payload.get("prompt_cache_key").is_none());
        assert!(payload.get("prompt_cache_retention").is_none());
    }

    #[tokio::test]
    async fn none_retention_disables_prompt_caching() {
        let target = openai_model();
        let payload = capture_payload(
            &target,
            &simple_context(),
            &retention_options(CacheRetention::None, "session-none"),
        )
        .await;
        assert!(payload.get("prompt_cache_key").is_none());
        assert!(payload.get("prompt_cache_retention").is_none());
    }

    #[tokio::test]
    async fn openai_base_url_sets_prompt_cache_key_without_retention() {
        let target = openai_model();
        let payload = capture_payload(
            &target,
            &simple_context(),
            &retention_options(CacheRetention::Short, "session-openai"),
        )
        .await;
        assert_eq!(payload["prompt_cache_key"], json!("session-openai"));
        assert!(payload.get("prompt_cache_retention").is_none());
    }

    // -- cross-provider handoff --------------------------------------------

    #[tokio::test]
    async fn cross_provider_handoff_normalizes_ids_and_thinking() {
        let target = model(ProviderId::OPENAI, "https://api.openai.com/v1", false);
        // A context produced by the OpenAI Responses dialect: pipe-separated tool
        // call ids and an unsigned thinking block.
        let source_assistant = Message::Assistant(Box::new(AssistantMessage {
            content: vec![
                AssistantContent::Thinking(ThinkingContent {
                    thinking: "let me think".to_owned(),
                    thinking_signature: None,
                    redacted: false,
                }),
                AssistantContent::Text(TextContent {
                    text: "calling the tool".to_owned(),
                    text_signature: None,
                }),
                tool_call("call_1|ctc_1", "double_number"),
            ],
            api: Api::OpenaiResponses,
            provider: ProviderId::new(ProviderId::OPENAI),
            model: "gpt-5.4".to_owned(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: usage(),
            stop_reason: StopReason::ToolUse,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            timestamp: 2,
        }));
        let context = Context {
            system_prompt: Some("You are a helpful assistant.".to_owned()),
            messages: vec![
                user("Please double 21", 1),
                source_assistant,
                tool_result("call_1|ctc_1", "double_number", None),
                user("thanks", 4),
            ],
            tools: Some(vec![Tool {
                name: "double_number".to_owned(),
                description: "Doubles a number".to_owned(),
                parameters: json!({ "type": "object", "properties": { "value": { "type": "number" } } }),
                constrained_sampling: None,
            }]),
        };

        let compat = resolve_compat(&target);
        let messages =
            convert_messages(&target, &context, &compat, &HashMap::new()).expect("messages");
        let roles: Vec<&str> = messages
            .iter()
            .map(|message| message["role"].as_str().expect("role"))
            .collect();
        assert_eq!(roles, vec!["system", "user", "assistant", "tool", "user"]);

        let replayed = &messages[2];
        // The pipe-separated id is normalized while preserving item uniqueness.
        assert_eq!(replayed["tool_calls"][0]["id"], json!("call_1_ctc_1"));
        assert_eq!(replayed["tool_calls"][0]["type"], json!("function"));
        // Unsigned cross-model thinking replays as plain assistant text.
        let content = replayed["content"].as_str().expect("string content");
        assert!(content.contains("let me think"), "{content}");
        assert!(content.contains("calling the tool"), "{content}");
        // The tool result points at the normalized id.
        assert_eq!(messages[3]["tool_call_id"], json!("call_1_ctc_1"));
    }

    #[test]
    fn openai_tool_call_ids_truncate_to_forty_characters() {
        let long_id = "c".repeat(64);
        assert_eq!(
            normalize_tool_call_id(&long_id, ProviderId::OPENAI)
                .chars()
                .count(),
            40
        );
        // Non-OpenAI providers keep the id verbatim.
        assert_eq!(normalize_tool_call_id(&long_id, "groq"), long_id);
        // Pipe ids without an item part collapse to the call id.
        assert_eq!(normalize_tool_call_id("call_1|", "openai"), "call_1");
        // Over-long combined ids get a stable hash suffix.
        let normalized =
            normalize_tool_call_id(&format!("{}|{}", "c".repeat(40), "d".repeat(40)), "groq");
        assert!(normalized.chars().count() <= 40, "{normalized}");
        assert!(normalized.contains('_'), "{normalized}");
    }

    // -- deferred tools -----------------------------------------------------

    fn deferred_context(tools: Vec<Tool>, added: &[(&str, Vec<String>)]) -> Context {
        let mut messages = vec![
            user("Hello", 1),
            assistant(vec![tool_call("call_1", "base_tool")], StopReason::ToolUse),
        ];
        for (tool_call_id, names) in added {
            messages.push(tool_result(tool_call_id, "base_tool", Some(names.clone())));
        }
        messages.push(user("next", 4));
        Context {
            messages,
            tools: Some(tools),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn kimi_defers_tools_to_system_tool_messages() {
        let mut compat = compat_settings();
        compat.deferred_tools_mode = Some(DeferredToolsMode::Kimi);
        let target = retention_model(compat);
        let context = deferred_context(
            vec![tool("base_tool"), tool("late_tool")],
            &[("call_1", vec!["late_tool".to_owned()])],
        );
        let payload = build_payload(&target, &context, &stream_options());

        let tool_names: Vec<&str> = payload["tools"]
            .as_array()
            .expect("tools")
            .iter()
            .map(|entry| entry["function"]["name"].as_str().expect("name"))
            .collect();
        assert_eq!(tool_names, vec!["base_tool"]);

        let messages = payload["messages"].as_array().expect("messages");
        let roles: Vec<&str> = messages
            .iter()
            .map(|m| m["role"].as_str().expect("role"))
            .collect();
        assert_eq!(roles, vec!["user", "assistant", "tool", "system", "user"]);
        let deferred = &messages[3];
        assert!(
            deferred.get("content").is_none(),
            "kimi tool message omits content"
        );
        let deferred_names: Vec<&str> = deferred["tools"]
            .as_array()
            .expect("tools")
            .iter()
            .map(|entry| entry["function"]["name"].as_str().expect("name"))
            .collect();
        assert_eq!(deferred_names, vec!["late_tool"]);
    }

    #[tokio::test]
    async fn kimi_emits_deferred_schemas_after_all_tool_results_in_a_batch() {
        let mut compat = compat_settings();
        compat.deferred_tools_mode = Some(DeferredToolsMode::Kimi);
        let target = retention_model(compat);
        let context = deferred_context(
            vec![tool("base_tool"), tool("late_tool"), tool("later_tool")],
            &[
                ("call_1", vec!["late_tool".to_owned()]),
                ("call_2", vec!["later_tool".to_owned()]),
            ],
        );
        let payload = build_payload(&target, &context, &stream_options());
        let messages = payload["messages"].as_array().expect("messages");
        let roles: Vec<&str> = messages
            .iter()
            .map(|m| m["role"].as_str().expect("role"))
            .collect();
        assert_eq!(
            roles,
            vec!["user", "assistant", "tool", "tool", "system", "user"]
        );
        let deferred_names: Vec<&str> = messages[4]["tools"]
            .as_array()
            .expect("tools")
            .iter()
            .map(|entry| entry["function"]["name"].as_str().expect("name"))
            .collect();
        assert_eq!(deferred_names, vec!["late_tool", "later_tool"]);
    }

    #[tokio::test]
    async fn completions_tools_are_unchanged_without_kimi_mode() {
        let target = retention_model(compat_settings());
        let context = deferred_context(
            vec![tool("base_tool"), tool("late_tool")],
            &[("call_1", vec!["late_tool".to_owned()])],
        );
        let payload = build_payload(&target, &context, &stream_options());
        let tool_names: Vec<&str> = payload["tools"]
            .as_array()
            .expect("tools")
            .iter()
            .map(|entry| entry["function"]["name"].as_str().expect("name"))
            .collect();
        assert_eq!(tool_names, vec!["base_tool", "late_tool"]);
        assert!(
            !payload["messages"]
                .as_array()
                .expect("messages")
                .iter()
                .any(|message| message.get("tools").is_some())
        );
    }

    // -- compat detection ---------------------------------------------------

    type CompatPredicate = fn(&ResolvedOpenAICompletionsCompat) -> bool;
    type PayloadPredicate = fn(&Value);

    #[test]
    fn detects_compat_from_provider_and_base_url() {
        let cases: Vec<(&str, &str, CompatPredicate)> = vec![
            ("step", "https://proxy.example.com/v1", |compat| {
                !compat.supports_developer_role
                    && compat.max_tokens_field == MaxTokensField::MaxTokens
            }),
            ("custom", "https://api.stepfun.com/step_plan/v1", |compat| {
                !compat.supports_developer_role
                    && compat.max_tokens_field == MaxTokensField::MaxTokens
            }),
            ("custom", "https://api.stepfun.ai/v1", |compat| {
                !compat.supports_developer_role
                    && compat.max_tokens_field == MaxTokensField::MaxTokens
            }),
            ("openai", "https://api.openai.com/v1", |compat| {
                compat.supports_store
                    && compat.supports_developer_role
                    && compat.supports_reasoning_effort
                    && compat.max_tokens_field == MaxTokensField::MaxCompletionTokens
                    && compat.thinking_format == CompletionsThinkingFormat::Openai
                    && compat.supports_strict_mode
                    && compat.supports_long_cache_retention
            }),
            ("deepseek", "https://api.deepseek.com/v1", |compat| {
                compat.max_tokens_field == MaxTokensField::MaxTokens
                    && compat.thinking_format == CompletionsThinkingFormat::Deepseek
                    && compat.requires_reasoning_content_on_assistant_messages
            }),
            ("zai", "https://api.z.ai/v1", |compat| {
                compat.thinking_format == CompletionsThinkingFormat::Zai
                    && !compat.supports_reasoning_effort
                    && compat.max_tokens_field == MaxTokensField::MaxTokens
                    && !compat.supports_store
            }),
            ("together", "https://api.together.xyz/v1", |compat| {
                compat.thinking_format == CompletionsThinkingFormat::Together
                    && !compat.supports_strict_mode
                    && !compat.supports_long_cache_retention
            }),
            ("openrouter", "https://openrouter.ai/api/v1", |compat| {
                compat.thinking_format == CompletionsThinkingFormat::Openrouter
                    && compat.session_affinity_format == SessionAffinityFormat::Openrouter
                    && !compat.supports_developer_role
            }),
            ("moonshotai", "https://api.moonshot.ai/v1", |compat| {
                !compat.supports_strict_mode
                    && !compat.supports_reasoning_effort
                    && compat.max_tokens_field == MaxTokensField::MaxTokens
            }),
            ("nvidia", "https://integrate.api.nvidia.com/v1", |compat| {
                !compat.supports_strict_mode
                    && !compat.supports_reasoning_effort
                    && !compat.supports_long_cache_retention
            }),
            ("cerebras", "https://api.cerebras.ai/v1", |compat| {
                !compat.supports_store
            }),
            ("ant-ling", "https://api.ant-ling.com/v1", |compat| {
                compat.thinking_format == CompletionsThinkingFormat::AntLing
            }),
            ("xai", "https://api.x.ai/v1", |compat| {
                !compat.supports_reasoning_effort
            }),
        ];
        for (provider, base_url, predicate) in cases {
            let compat = resolve_compat(&model(provider, base_url, false));
            assert!(
                predicate(&compat),
                "compat mismatch for {provider} ({base_url}): {compat:?}"
            );
        }

        // Anthropic models served through OpenRouter get Anthropic cache control
        // and the developer role.
        let mut anthropic_on_openrouter =
            model(ProviderId::OPENROUTER, "https://openrouter.ai/api/v1", true);
        anthropic_on_openrouter.id = "anthropic/claude-sonnet-4.5".to_owned();
        let compat = resolve_compat(&anthropic_on_openrouter);
        assert_eq!(
            compat.cache_control_format,
            Some(CacheControlFormat::Anthropic)
        );
        assert!(compat.supports_developer_role);
    }

    #[test]
    fn explicit_compat_overrides_detected_values() {
        let mut compat = compat_settings();
        compat.supports_store = Some(false);
        compat.thinking_format = Some(CompletionsThinkingFormat::Qwen);
        compat.max_tokens_field = Some(MaxTokensField::MaxTokens);
        let mut target = openai_model();
        target.compat = Some(compat);
        let resolved = resolve_compat(&target);
        assert!(!resolved.supports_store);
        assert_eq!(resolved.thinking_format, CompletionsThinkingFormat::Qwen);
        assert_eq!(resolved.max_tokens_field, MaxTokensField::MaxTokens);
        // Untouched fields keep their detected values.
        assert!(resolved.supports_developer_role);
    }

    #[test]
    fn step_title_payload_uses_system_role_and_native_token_budget() {
        let target = model("step", "https://api.stepfun.com/step_plan/v1", true);
        let context = Context {
            system_prompt: Some("Only generate a title".to_owned()),
            messages: vec![user("广州天气", 0)],
            tools: None,
        };
        let mut options = stream_options();
        options.base.max_tokens = Some(1024);
        let payload = build_payload(&target, &context, &options);
        assert_eq!(payload["messages"][0]["role"], "system");
        assert_eq!(payload["max_tokens"], 1024);
        assert!(payload.get("max_completion_tokens").is_none());
    }

    // -- thinking formats ---------------------------------------------------

    fn reasoning_model(format: CompletionsThinkingFormat) -> Model {
        let mut compat = compat_settings();
        compat.thinking_format = Some(format);
        let mut target = model("custom", "https://my-proxy.example.com/v1", true);
        target.compat = Some(compat);
        target
    }

    fn high_effort_options() -> OpenAICompletionsOptions {
        let mut options = stream_options();
        options.reasoning_effort = Some(ReasoningEffort::High);
        options
    }

    #[tokio::test]
    async fn thinking_formats_map_reasoning_effort() {
        let cases: Vec<(CompletionsThinkingFormat, PayloadPredicate)> = vec![
            (CompletionsThinkingFormat::Openai, |payload| {
                assert_eq!(payload["reasoning_effort"], json!("high"));
            }),
            (CompletionsThinkingFormat::Openrouter, |payload| {
                assert_eq!(payload["reasoning"]["effort"], json!("high"));
            }),
            (CompletionsThinkingFormat::Deepseek, |payload| {
                assert_eq!(payload["thinking"]["type"], json!("enabled"));
                assert_eq!(payload["reasoning_effort"], json!("high"));
            }),
            (CompletionsThinkingFormat::Together, |payload| {
                assert_eq!(payload["reasoning"]["enabled"], json!(true));
                assert_eq!(payload["reasoning_effort"], json!("high"));
            }),
            (CompletionsThinkingFormat::Zai, |payload| {
                assert_eq!(payload["thinking"]["type"], json!("enabled"));
                assert_eq!(payload["thinking"]["clear_thinking"], json!(false));
                assert_eq!(payload["reasoning_effort"], json!("high"));
            }),
            (CompletionsThinkingFormat::Qwen, |payload| {
                assert_eq!(payload["enable_thinking"], json!(true));
                assert_eq!(payload["reasoning_effort"], json!("high"));
            }),
            (CompletionsThinkingFormat::QwenChatTemplate, |payload| {
                assert_eq!(
                    payload["chat_template_kwargs"],
                    json!({ "enable_thinking": true, "preserve_thinking": true })
                );
            }),
            (CompletionsThinkingFormat::StringThinking, |payload| {
                assert_eq!(payload["thinking"], json!("high"));
            }),
            (CompletionsThinkingFormat::Baseten, |payload| {
                assert_eq!(payload["reasoning_effort"], json!("high"));
            }),
        ];
        for (format, assert_fields) in cases {
            let target = reasoning_model(format);
            let payload = build_payload(&target, &simple_context(), &high_effort_options());
            assert_fields(&payload);
        }
    }

    #[tokio::test]
    async fn ant_ling_requires_an_explicit_effort_mapping() {
        // Without a mapping entry nothing is emitted.
        let target = reasoning_model(CompletionsThinkingFormat::AntLing);
        let payload = build_payload(&target, &simple_context(), &high_effort_options());
        assert!(payload.get("reasoning").is_none());

        let mut level_map = std::collections::HashMap::new();
        level_map.insert(ModelThinkingLevel::High, Some("high_effort".to_owned()));
        let mut target = reasoning_model(CompletionsThinkingFormat::AntLing);
        target.thinking_level_map = Some(level_map);
        let payload = build_payload(&target, &simple_context(), &high_effort_options());
        assert_eq!(payload["reasoning"], json!({ "effort": "high_effort" }));
    }

    #[tokio::test]
    async fn chat_template_kwargs_substitute_thinking_variables() {
        let mut compat = compat_settings();
        compat.thinking_format = Some(CompletionsThinkingFormat::ChatTemplate);
        compat.supports_thinking_token_budget = Some(true);
        compat.chat_template_kwargs = Some(
            serde_json::from_value(json!({
                "enable_thinking": { "$var": "thinking.enabled" },
                "effort": { "$var": "thinking.effort" },
                "budget": { "$var": "thinking.budget" },
                "omit_me": { "$var": "thinking.enabled", "omitWhenOff": true },
                "literal": 7
            }))
            .expect("kwargs deserialize"),
        );
        let mut target = model("custom", "https://my-proxy.example.com/v1", true);
        target.compat = Some(compat);

        let mut options = high_effort_options();
        options.base.max_tokens = Some(4096);
        let payload = build_payload(&target, &simple_context(), &options);
        let kwargs = &payload["chat_template_kwargs"];
        assert_eq!(kwargs["enable_thinking"], json!(true));
        assert_eq!(kwargs["effort"], json!("high"));
        // The budget is capped to leave room for the answer.
        assert_eq!(kwargs["budget"], json!(4096 - 1024));
        assert_eq!(kwargs["omit_me"], json!(true));
        assert_eq!(kwargs["literal"], json!(7));
        // The top-level budget field is emitted too.
        assert_eq!(payload["thinking_token_budget"], json!(4096 - 1024));

        // With reasoning off, `omitWhenOff` drops the key and the budget disappears.
        let payload = build_payload(&target, &simple_context(), &stream_options());
        let kwargs = &payload["chat_template_kwargs"];
        assert_eq!(kwargs["enable_thinking"], json!(false));
        assert!(kwargs.get("omit_me").is_none());
        assert!(kwargs.get("budget").is_none());
        assert!(payload.get("thinking_token_budget").is_none());
    }

    #[tokio::test]
    async fn non_reasoning_models_emit_no_thinking_fields() {
        let mut compat = compat_settings();
        compat.thinking_format = Some(CompletionsThinkingFormat::Openai);
        let mut target = openai_model();
        target.compat = Some(compat);
        let payload = build_payload(&target, &simple_context(), &high_effort_options());
        assert!(payload.get("reasoning_effort").is_none());
        assert!(payload.get("thinking").is_none());
    }

    // -- usage and stop reasons --------------------------------------------

    #[test]
    fn usage_parsing_maps_cache_and_reasoning_tokens() {
        let target = openai_model();
        let usage = parse_chunk_usage(
            &json!({
                "prompt_tokens": 100,
                "completion_tokens": 30,
                "prompt_tokens_details": { "cached_tokens": 40 },
                "completion_tokens_details": { "reasoning_tokens": 10 }
            }),
            &target,
        );
        assert_eq!(usage.input, 60);
        assert_eq!(usage.output, 30);
        assert_eq!(usage.cache_read, 40);
        assert_eq!(usage.cache_write, 0);
        assert_eq!(usage.reasoning, Some(10));
        assert_eq!(usage.total_tokens, 130);
    }

    #[test]
    fn usage_parsing_handles_provider_specific_placements() {
        let target = openai_model();
        // DeepSeek reports cache hits at the top level.
        let deepseek = parse_chunk_usage(
            &json!({ "prompt_tokens": 50, "completion_tokens": 5, "prompt_cache_hit_tokens": 20 }),
            &target,
        );
        assert_eq!(deepseek.input, 30);
        assert_eq!(deepseek.cache_read, 20);

        // Kimi documents top-level usage.cached_tokens on the final chunk.
        let kimi = parse_chunk_usage(
            &json!({ "prompt_tokens": 50, "cached_tokens": 20 }),
            &target,
        );
        assert_eq!(kimi.cache_read, 20);
        assert_eq!(kimi.input, 30);

        // OpenRouter-compatible providers report cache writes separately; writes
        // are never subtracted from reads.
        let openrouter = parse_chunk_usage(
            &json!({
                "prompt_tokens": 100,
                "completion_tokens": 5,
                "prompt_tokens_details": { "cached_tokens": 40, "cache_write_tokens": 10 }
            }),
            &target,
        );
        assert_eq!(openrouter.cache_read, 40);
        assert_eq!(openrouter.cache_write, 10);
        assert_eq!(openrouter.input, 50);
        assert_eq!(openrouter.total_tokens, 50 + 5 + 40 + 10);

        // Token counts are coerced from floats and strings.
        let coerced = parse_chunk_usage(
            &json!({ "prompt_tokens": "12", "completion_tokens": 3.0 }),
            &target,
        );
        assert_eq!(coerced.input, 12);
        assert_eq!(coerced.output, 3);
    }

    #[test]
    fn stop_reason_mapping_covers_provider_values() {
        assert_eq!(map_stop_reason("stop"), (StopReason::Stop, None));
        assert_eq!(map_stop_reason("end"), (StopReason::Stop, None));
        assert_eq!(map_stop_reason("length"), (StopReason::Length, None));
        assert_eq!(map_stop_reason("tool_calls"), (StopReason::ToolUse, None));
        assert_eq!(
            map_stop_reason("function_call"),
            (StopReason::ToolUse, None)
        );
        let (reason, message) = map_stop_reason("content_filter");
        assert_eq!(reason, StopReason::Error);
        assert_eq!(
            message.as_deref(),
            Some("Provider finish_reason: content_filter")
        );
        let (reason, message) = map_stop_reason("weird_value");
        assert_eq!(reason, StopReason::Error);
        assert_eq!(
            message.as_deref(),
            Some("Provider finish_reason: weird_value")
        );
    }

    // -- streaming details --------------------------------------------------

    #[tokio::test]
    async fn reasoning_content_streams_as_thinking_blocks() {
        let chunks = [
            chunk(json!({ "reasoning_content": "think " }), None),
            chunk(json!({ "reasoning": "ing" }), None),
            chunk(json!({ "content": "answer" }), Some("stop")),
        ];
        let (types, message) = drive_chunks(&chunks).await;

        assert_eq!(
            types,
            vec![
                "start",
                "thinking_start",
                "thinking_delta",
                "thinking_delta",
                "text_start",
                "text_delta",
                "thinking_end",
                "text_end",
                "done"
            ]
        );
        let thinking = message.content[0].thinking().expect("thinking block");
        assert_eq!(thinking.thinking, "think ing");
        // The field name is kept as the signature for replay.
        assert_eq!(
            thinking.thinking_signature.as_deref(),
            Some("reasoning_content")
        );
        assert_eq!(
            message.content[1].text().expect("text block").text,
            "answer"
        );
    }

    #[tokio::test]
    async fn function_tool_calls_accumulate_partial_json() {
        let chunks = [
            chunk(
                json!({ "tool_calls": [{ "index": 0, "id": "call_1", "type": "function", "function": { "name": "read", "arguments": "" } }] }),
                None,
            ),
            chunk(
                json!({ "tool_calls": [{ "index": 0, "function": { "arguments": "{\"path\":" } }] }),
                None,
            ),
            chunk(
                json!({ "tool_calls": [{ "index": 0, "function": { "arguments": "\"/src\"}" } }] }),
                Some("tool_calls"),
            ),
        ];
        let (types, message) = drive_chunks(&chunks).await;

        assert_eq!(
            types,
            vec![
                "start",
                "toolcall_start",
                "toolcall_delta",
                "toolcall_delta",
                "toolcall_delta",
                "toolcall_end",
                "done"
            ]
        );
        let call = message.content[0].tool_call().expect("tool call");
        assert_eq!(call.id, "call_1");
        assert_eq!(call.name, "read");
        assert_eq!(call.arguments["path"], json!("/src"));
        assert_eq!(message.stop_reason, StopReason::ToolUse);
    }

    #[tokio::test]
    async fn final_tool_arguments_must_be_complete_objects() {
        for arguments in [r#"{"path":"unfinished"#, "[]", "{invalid}"] {
            let chunks = [chunk(
                json!({ "tool_calls": [{ "index": 0, "id": "call_1", "type": "function",
                    "function": { "name": "read", "arguments": arguments } }] }),
                Some("tool_calls"),
            )];
            let (events, message) = drive_chunks(&chunks).await;
            assert_eq!(message.stop_reason, StopReason::Error);
            assert!(
                !events
                    .iter()
                    .any(|event| event == "toolcall_end" || event == "done")
            );
            assert!(
                message
                    .error_message
                    .as_deref()
                    .is_some_and(|message| !message.is_empty())
            );
        }
    }

    #[tokio::test]
    async fn tool_calls_tracked_by_id_when_index_is_missing() {
        let chunks = [
            chunk(
                json!({ "tool_calls": [{ "id": "call_a", "function": { "name": "read", "arguments": "{\"a\":1}" } }] }),
                None,
            ),
            chunk(
                json!({ "tool_calls": [{ "id": "call_a", "function": { "arguments": "{\"b\":2}" } }] }),
                Some("tool_calls"),
            ),
        ];
        let (_types, message) = drive_chunks(&chunks).await;
        // Both deltas land on the same block.
        assert_eq!(message.content.len(), 1);
        assert_eq!(message.content[0].tool_call().expect("call").id, "call_a");
    }

    #[tokio::test]
    async fn stream_without_finish_reason_errors_when_required() {
        let chunks = [chunk(json!({ "content": "partial" }), None)];
        let (types, message) = drive_chunks(&chunks).await;
        assert_eq!(types.last().map(String::as_str), Some("error"));
        assert_eq!(message.stop_reason, StopReason::Error);
        assert_eq!(
            message.error_message.as_deref(),
            Some("Stream ended without finish_reason")
        );
    }

    #[tokio::test]
    async fn missing_finish_reason_is_tolerated_when_compat_allows_it() {
        let mut target = openai_model();
        let mut compat = compat_settings();
        compat.supports_finish_reason = Some(false);
        target.compat = Some(compat);
        let grammar = HashMap::new();
        let resolved = resolve_compat(&target);
        let token = tokio_util::sync::CancellationToken::new();
        let state = StreamingState::new(initial_output(&target));
        let (mut sender, mut stream) = assistant_message_event_stream();
        tokio::spawn(async move {
            run_stream_events(
                futures::stream::iter(sse_events(&[chunk(json!({ "content": "hi" }), None)])),
                &target,
                &resolved,
                &grammar,
                &token,
                state,
                &mut sender,
            )
            .await;
        });
        let mut types = Vec::new();
        while let Some(event) = stream.next().await {
            types.push(event_type(&event).to_owned());
        }
        let message = stream.result().await.unwrap();
        assert_eq!(types.last().map(String::as_str), Some("done"));
        assert_eq!(message.stop_reason, StopReason::Stop);
    }

    #[tokio::test]
    async fn usage_only_chunks_and_done_sentinel_are_handled() {
        let chunks = [
            chunk(json!({ "content": "hi" }), Some("stop")),
            json!({ "id": "chatcmpl-test", "choices": [], "usage": { "prompt_tokens": 4, "completion_tokens": 2 } }),
        ];
        let (types, message) = drive_chunks(&chunks).await;
        assert_eq!(
            types,
            vec!["start", "text_start", "text_delta", "text_end", "done"]
        );
        assert_eq!(message.usage.input, 4);
        assert_eq!(message.usage.output, 2);
        assert_eq!(message.response_id.as_deref(), Some("chatcmpl-test"));
    }

    #[tokio::test]
    async fn aborting_the_token_emits_an_aborted_error() {
        let token = tokio_util::sync::CancellationToken::new();
        token.cancel();
        let target = openai_model();
        let compat = resolve_compat(&target);
        let state = StreamingState::new(initial_output(&target));
        let (mut sender, mut stream) = assistant_message_event_stream();
        tokio::spawn(async move {
            run_stream_events(
                futures::stream::iter(sse_events(&[chunk(
                    json!({ "content": "hi" }),
                    Some("stop"),
                )])),
                &target,
                &compat,
                &HashMap::new(),
                &token,
                state,
                &mut sender,
            )
            .await;
        });
        let mut types = Vec::new();
        while let Some(event) = stream.next().await {
            types.push(event_type(&event).to_owned());
        }
        let message = stream.result().await.unwrap();
        assert_eq!(types.last().map(String::as_str), Some("error"));
        assert_eq!(message.stop_reason, StopReason::Aborted);
    }

    // -- session affinity ---------------------------------------------------

    #[test]
    fn session_affinity_headers_follow_the_compat_format() {
        let target = model("custom", "https://my-proxy.example.com/v1", false);
        let context = simple_context();
        let cases: Vec<(SessionAffinityFormat, Vec<&str>)> = vec![
            (
                SessionAffinityFormat::Openai,
                vec!["session_id", "x-client-request-id", "x-session-affinity"],
            ),
            (
                SessionAffinityFormat::OpenaiNosession,
                vec!["x-client-request-id", "x-session-affinity"],
            ),
            (SessionAffinityFormat::Openrouter, vec!["x-session-id"]),
        ];
        for (format, expected) in cases {
            let mut compat = resolve_compat(&target);
            compat.send_session_affinity_headers = true;
            compat.session_affinity_format = format;
            let headers =
                build_request_headers(&target, &context, &compat, Some("session-1"), None, "key");
            for name in &expected {
                assert_eq!(
                    headers.get(*name).and_then(|value| value.as_deref()),
                    Some("session-1"),
                    "{format:?}/{name}"
                );
            }
            match format {
                SessionAffinityFormat::Openai => assert!(!headers.contains_key("x-session-id")),
                SessionAffinityFormat::OpenaiNosession => {
                    assert!(!headers.contains_key("session_id"))
                }
                SessionAffinityFormat::Openrouter => {
                    assert!(!headers.contains_key("session_id"));
                    assert!(!headers.contains_key("x-client-request-id"));
                }
            }
            assert_eq!(
                headers.get("Authorization").and_then(|v| v.as_deref()),
                Some("Bearer key")
            );
        }
    }

    #[test]
    fn caller_headers_override_defaults_and_suppress_authorization() {
        let target = model("custom", "https://my-proxy.example.com/v1", false);
        let context = simple_context();
        let compat = resolve_compat(&target);
        let mut caller = ProviderHeaders::new();
        caller.insert("User-Agent".to_owned(), Some("custom-agent".to_owned()));
        caller.insert(
            "Authorization".to_owned(),
            Some("Bearer upstream".to_owned()),
        );
        caller.insert("X-Extra".to_owned(), Some("1".to_owned()));
        let headers =
            build_request_headers(&target, &context, &compat, None, Some(&caller), "unused");
        assert_eq!(
            headers.get("User-Agent").and_then(|v| v.as_deref()),
            Some("custom-agent")
        );
        assert_eq!(
            headers.get("Authorization").and_then(|v| v.as_deref()),
            Some("Bearer upstream")
        );
        assert_eq!(headers.get("X-Extra").and_then(|v| v.as_deref()), Some("1"));
    }

    #[test]
    fn copilot_dynamic_headers_are_merged() {
        let mut target = model(
            ProviderId::GITHUB_COPILOT,
            "https://api.githubcopilot.com/chat/completions",
            false,
        );
        target.headers = Some(std::collections::HashMap::from([(
            "Editor-Version".to_owned(),
            "vscode/1.0".to_owned(),
        )]));
        let context = simple_context();
        let compat = resolve_compat(&target);
        let headers = build_request_headers(&target, &context, &compat, None, None, "key");
        assert_eq!(
            headers.get("X-Initiator").and_then(|v| v.as_deref()),
            Some("user")
        );
        assert_eq!(
            headers.get("Openai-Intent").and_then(|v| v.as_deref()),
            Some("conversation-edits")
        );
        assert_eq!(
            headers.get("Editor-Version").and_then(|v| v.as_deref()),
            Some("vscode/1.0")
        );
    }

    // -- simple options and errors -----------------------------------------

    #[tokio::test]
    async fn stream_simple_maps_simple_options() {
        let mut target = model(ProviderId::OPENAI, "http://127.0.0.1:9/v1", true);
        let mut level_map = std::collections::HashMap::new();
        level_map.insert(ModelThinkingLevel::High, Some("high_effort".to_owned()));
        target.thinking_level_map = Some(level_map);
        let context = simple_context();

        let mut options = SimpleStreamOptions::default();
        options.base.base.api_key = Some("test-key".to_owned());
        options.tool_choice = Some(ToolChoice::None);
        options.reasoning = Some(ThinkingLevel::High);
        let capture = PayloadCapture::new();
        options.base.on_payload = Some(capture.clone());
        let mut stream = stream_simple(&target, &context, &options);
        while stream.next().await.is_some() {}
        let payload = capture.take();

        assert_eq!(payload["tool_choice"], json!("none"));
        assert_eq!(payload["reasoning_effort"], json!("high_effort"));
        // max_tokens is clamped into the remaining context window.
        assert!(payload.get("max_completion_tokens").is_some());
        assert!(payload.get("max_tokens").is_none());
    }

    #[tokio::test]
    async fn stream_simple_omits_reasoning_for_non_reasoning_models() {
        let target = model(ProviderId::OPENAI, "http://127.0.0.1:9/v1", false);
        let context = simple_context();
        let mut options = SimpleStreamOptions::default();
        options.base.base.api_key = Some("test-key".to_owned());
        options.reasoning = Some(ThinkingLevel::Xhigh);
        let capture = PayloadCapture::new();
        options.base.on_payload = Some(capture.clone());
        let mut stream = stream_simple(&target, &context, &options);
        while stream.next().await.is_some() {}
        let payload = capture.take();
        assert!(payload.get("reasoning_effort").is_none());
        assert!(payload.get("thinking_token_budget").is_none());
    }

    #[tokio::test]
    async fn missing_api_key_emits_an_error_event() {
        let target = openai_model();
        let options = OpenAICompletionsOptions::default();
        let mut stream = stream(&target, &simple_context(), &options);
        let mut types = Vec::new();
        while let Some(event) = stream.next().await {
            types.push(event_type(&event).to_owned());
        }
        let message = stream.result().await.unwrap();
        assert_eq!(types, vec!["error"]);
        assert_eq!(message.stop_reason, StopReason::Error);
        assert_eq!(
            message.error_message.as_deref(),
            Some("No API key for provider: openai")
        );
    }

    #[tokio::test]
    async fn unreachable_endpoint_emits_an_error_event() {
        let target = model("custom", "http://127.0.0.1:9/v1", false);
        let mut options = stream_options();
        options.base.base.max_retries = Some(0);
        let mut stream = stream(&target, &simple_context(), &options);
        let mut types = Vec::new();
        while let Some(event) = stream.next().await {
            types.push(event_type(&event).to_owned());
        }
        let message = stream.result().await.unwrap();
        assert_eq!(types, vec!["error"]);
        assert_eq!(message.stop_reason, StopReason::Error);
        assert!(message.error_message.is_some());
    }

    #[tokio::test]
    async fn adapts_generic_stream_options_from_metadata() {
        let base = StreamOptions {
            base: ProviderRequestOptions {
                api_key: Some("k".to_owned()),
                ..Default::default()
            },
            metadata: Some(serde_json::Map::from_iter([
                ("toolChoice".to_owned(), json!("required")),
                ("reasoningEffort".to_owned(), json!("xhigh")),
                ("thinkingBudgets".to_owned(), json!({ "high": 4096 })),
            ])),
            ..Default::default()
        };
        let options = options_from_stream_options(&base);
        assert_eq!(options.tool_choice, Some(json!("required")));
        assert_eq!(options.reasoning_effort, Some(ReasoningEffort::Xhigh));
        assert_eq!(
            options.thinking_budgets,
            Some(ThinkingBudgets {
                high: Some(4096),
                ..Default::default()
            })
        );

        // The trait entry point adapts the same way.
        let adapter = openai_completions_api();
        let stream = adapter.stream(&openai_model(), &simple_context(), &base);
        drop(stream);
        let simple = adapter.stream_simple(
            &openai_model(),
            &simple_context(),
            &SimpleStreamOptions::default(),
        );
        drop(simple);
    }

    #[test]
    fn tool_choice_and_sampling_params_reach_the_payload() {
        let target = openai_model();
        let mut options = stream_options();
        options.tool_choice = Some(json!("required"));
        options.base.temperature = Some(0.25);
        options.base.sampling_params = Some(serde_json::Map::from_iter([(
            "top_p".to_owned(),
            json!(0.9),
        )]));
        options.base.max_tokens = Some(2048);
        let payload = build_payload(&target, &simple_context(), &options);
        assert_eq!(payload["tool_choice"], json!("required"));
        assert_eq!(payload["temperature"], json!(0.25));
        assert_eq!(payload["top_p"], json!(0.9));
        assert_eq!(payload["max_completion_tokens"], json!(2048));
    }

    #[test]
    fn anthropic_cache_control_marks_system_tools_and_last_message() {
        let mut compat = compat_settings();
        compat.cache_control_format = Some(CacheControlFormat::Anthropic);
        let target = retention_model(compat);
        let context = Context {
            system_prompt: Some("You are a helpful assistant.".to_owned()),
            messages: vec![user("Hello", 1)],
            tools: Some(vec![tool("read")]),
        };
        let payload = build_payload(&target, &context, &stream_options());
        let messages = payload["messages"].as_array().expect("messages");
        let system = messages
            .iter()
            .find(|message| message["role"] == "system")
            .expect("system message");
        assert_eq!(
            system["content"][0]["cache_control"],
            json!({ "type": "ephemeral" })
        );
        let last_user = messages.last().expect("last message");
        assert_eq!(
            last_user["content"][0]["cache_control"],
            json!({ "type": "ephemeral" })
        );
        assert_eq!(
            payload["tools"][0]["cache_control"],
            json!({ "type": "ephemeral" })
        );

        // Long retention adds the 1h TTL.
        let mut options = stream_options();
        options.base.cache_retention = Some(CacheRetention::Long);
        let payload = build_payload(&target, &context, &options);
        assert_eq!(
            payload["tools"][0]["cache_control"],
            json!({ "type": "ephemeral", "ttl": "1h" })
        );
    }

    #[test]
    fn assistant_messages_without_content_or_tool_calls_are_skipped() {
        let target = openai_model();
        let compat = resolve_compat(&target);
        let context = Context {
            messages: vec![assistant(Vec::new(), StopReason::Stop), user("hello", 2)],
            ..Default::default()
        };
        let messages =
            convert_messages(&target, &context, &compat, &HashMap::new()).expect("messages");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], json!("user"));
    }

    #[test]
    fn assistant_after_tool_result_bridges_user_messages() {
        let mut compat = compat_settings();
        compat.requires_assistant_after_tool_result = Some(true);
        let target = retention_model(compat);
        let context = Context {
            messages: vec![
                user("hi", 1),
                assistant(vec![tool_call("call_1", "read")], StopReason::ToolUse),
                tool_result("call_1", "read", None),
                user("next", 4),
            ],
            ..Default::default()
        };
        let payload = build_payload(&target, &context, &stream_options());
        let messages = payload["messages"].as_array().expect("messages");
        let roles: Vec<&str> = messages
            .iter()
            .map(|m| m["role"].as_str().expect("role"))
            .collect();
        assert_eq!(
            roles,
            vec!["user", "assistant", "tool", "assistant", "user"]
        );
        assert_eq!(
            messages[3]["content"],
            json!("I have processed the tool results.")
        );
        // Assistant content becomes an empty string instead of null.
        assert_eq!(messages[1]["content"], json!(""));
    }

    #[test]
    fn tool_result_placeholders_cover_image_only_and_empty_results() {
        let mut target = openai_model();
        target.input = vec![InputModality::Text, InputModality::Image];
        let compat = resolve_compat(&target);
        let image_only = Message::ToolResult(Box::new(ToolResultMessage {
            tool_call_id: "call_1".to_owned(),
            tool_name: "read".to_owned(),
            content: vec![UserContentBlock::Image(crate::types::ImageContent {
                data: "aW1hZ2U=".to_owned(),
                mime_type: "image/png".to_owned(),
            })],
            details: None,
            usage: None,
            added_tool_names: None,
            is_error: false,
            timestamp: 3,
        }));
        let empty = Message::ToolResult(Box::new(ToolResultMessage {
            tool_call_id: "call_2".to_owned(),
            tool_name: "read".to_owned(),
            content: Vec::new(),
            details: None,
            usage: None,
            added_tool_names: None,
            is_error: false,
            timestamp: 3,
        }));
        let context = Context {
            messages: vec![image_only, empty],
            ..Default::default()
        };
        let messages =
            convert_messages(&target, &context, &compat, &HashMap::new()).expect("messages");
        assert_eq!(messages[0]["content"], json!("(see attached image)"));
        assert_eq!(messages[1]["content"], json!("(no tool output)"));
    }

    #[test]
    fn reasoning_details_replay_as_structured_metadata() {
        let target = openai_model();
        let compat = resolve_compat(&target);
        let details = json!([
            { "type": "reasoning.text", "text": "first", "id": "rs_1" },
            { "type": "reasoning.encrypted", "data": "opaque", "id": "rs_2" }
        ]);
        let source = Message::Assistant(Box::new(AssistantMessage {
            content: vec![
                AssistantContent::Thinking(ThinkingContent {
                    thinking: "first".to_owned(),
                    thinking_signature: Some(
                        serde_json::to_string(&details).expect("details serialize"),
                    ),
                    redacted: false,
                }),
                tool_call("call_1", "read"),
            ],
            api: Api::OpenaiCompletions,
            provider: ProviderId::new(ProviderId::OPENAI),
            model: "test-model".to_owned(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: usage(),
            stop_reason: StopReason::ToolUse,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            timestamp: 2,
        }));
        let context = Context {
            messages: vec![source],
            ..Default::default()
        };
        let messages =
            convert_messages(&target, &context, &compat, &HashMap::new()).expect("messages");
        assert_eq!(
            messages[0]["reasoning_details"],
            Value::Array(details.as_array().cloned().expect("array"))
        );
        // No raw reasoning field when structured details are present.
        assert!(messages[0].get("reasoning_content").is_none());
    }

    #[test]
    fn deepseek_assistant_messages_carry_reasoning_content() {
        let target = model("deepseek", "https://api.deepseek.com/v1", true);
        let compat = resolve_compat(&target);
        assert!(compat.requires_reasoning_content_on_assistant_messages);
        let context = Context {
            messages: vec![assistant(
                vec![AssistantContent::Text(TextContent {
                    text: "hello".to_owned(),
                    text_signature: None,
                })],
                StopReason::Stop,
            )],
            ..Default::default()
        };
        let messages =
            convert_messages(&target, &context, &compat, &HashMap::new()).expect("messages");
        assert_eq!(messages[0]["reasoning_content"], json!(""));
    }

    #[test]
    fn thinking_as_text_inlines_thinking_blocks() {
        let mut compat = compat_settings();
        compat.requires_thinking_as_text = Some(true);
        // The assistant helper replays as provider "openai" / model "test-model",
        // so the target must match for the thinking block to survive the
        // cross-provider transform.
        let mut target = openai_model();
        target.compat = Some(compat);
        let compat = resolve_compat(&target);
        let context = Context {
            messages: vec![assistant(
                vec![
                    AssistantContent::Thinking(ThinkingContent {
                        thinking: "step one".to_owned(),
                        thinking_signature: None,
                        redacted: false,
                    }),
                    AssistantContent::Text(TextContent {
                        text: "answer".to_owned(),
                        text_signature: None,
                    }),
                ],
                StopReason::Stop,
            )],
            ..Default::default()
        };
        let messages =
            convert_messages(&target, &context, &compat, &HashMap::new()).expect("messages");
        assert_eq!(messages[0]["content"][0]["text"], json!("step one"));
        assert_eq!(messages[0]["content"][1]["text"], json!("answer"));
    }

    #[test]
    fn openrouter_and_vercel_routing_reach_the_payload() {
        let mut compat = compat_settings();
        compat.open_router_routing =
            Some(serde_json::from_value(json!({ "only": ["anthropic"] })).expect("routing"));
        compat.vercel_gateway_routing =
            Some(serde_json::from_value(json!({ "order": ["bedrock"] })).expect("routing"));
        let target = retention_model(compat);
        let payload = build_payload(&target, &simple_context(), &stream_options());
        assert_eq!(payload["provider"], json!({ "only": ["anthropic"] }));
        assert_eq!(
            payload["providerOptions"],
            json!({ "gateway": { "order": ["bedrock"] } })
        );
    }

    #[test]
    fn usage_in_streaming_and_store_follow_compat() {
        let mut compat = compat_settings();
        compat.supports_usage_in_streaming = Some(false);
        compat.supports_store = Some(false);
        let target = retention_model(compat);
        let payload = build_payload(&target, &simple_context(), &stream_options());
        assert!(payload.get("stream_options").is_none());
        assert!(payload.get("store").is_none());

        let target = openai_model();
        let payload = build_payload(&target, &simple_context(), &stream_options());
        assert_eq!(payload["stream_options"], json!({ "include_usage": true }));
        assert_eq!(payload["store"], json!(false));
    }

    #[test]
    fn tool_history_without_tools_sends_an_empty_tools_array() {
        let target = openai_model();
        let context = Context {
            messages: vec![
                assistant(vec![tool_call("call_1", "read")], StopReason::ToolUse),
                tool_result("call_1", "read", None),
            ],
            tools: None,
            ..Default::default()
        };
        let payload = build_payload(&target, &context, &stream_options());
        assert_eq!(payload["tools"], json!([]));
    }

    #[test]
    fn sse_frames_are_parsed_from_raw_text() {
        // The driver consumes `data:` payloads; verify the framing contract the
        // harness builds (`data: <json>` blocks terminated by `[DONE]`).
        let body = format!(
            "{}\n\n",
            [chunk(json!({ "content": "hi" }), Some("stop"))]
                .iter()
                .map(|chunk| format!("data: {}", serde_json::to_string(chunk).expect("json")))
                .collect::<Vec<_>>()
                .join("\n\n")
        );
        let frames: Vec<&str> = body
            .split("\n\n")
            .filter(|frame| !frame.trim().is_empty())
            .collect();
        assert_eq!(frames.len(), 1);
        let data = frames[0].trim_start_matches("data: ").trim();
        let parsed: Value = parse_json_with_repair(data).expect("frame parses");
        assert_eq!(parsed["choices"][0]["delta"]["content"], json!("hi"));
    }
}
