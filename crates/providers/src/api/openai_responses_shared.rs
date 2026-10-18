//! OpenAI Responses API shared logic: conversation/tool conversion into the
//! Responses `input`/`tools` wire shapes, and the SSE stream processor that
//! maps Responses stream events onto assistant-message events.
//!
//! Ported from `packages/providers/src/api/openai-responses-shared.ts`.
//!
//! The wire shapes produced here match the OpenAI Responses API exactly:
//! `input` items (`message` with `input_text`/`input_image`, `reasoning` items
//! carrying `encrypted_content`, `function_call`/`function_call_output`,
//! `custom_tool_call`/`custom_tool_call_output`, `additional_tools`,
//! `tool_search_call`/`tool_search_output`) and `tools` entries (`function`
//! with optional `strict`, `custom` with a grammar `format`).

use async_openai::types::responses::{
    CustomGrammarFormatParam, CustomToolParam, CustomToolParamFormat, FunctionTool, GrammarSyntax,
    MessagePhase, OutputItem, OutputMessageContent, ReasoningItemContent, SummaryPart,
    Tool as WireTool,
};
use std::collections::{HashMap, HashSet};

use futures::{Stream, StreamExt};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::api::constrained_sampling::{
    GrammarToolInputJsonBuffer, get_grammar_tool_input, get_json_schema_tool_parameters,
    resolve_grammar_constrained_sampling, resolve_json_schema_strict_sampling,
};
use crate::api::http::SseEvent;
use crate::api::transform_messages::transform_messages;
use crate::error::ProviderHttpError;
use crate::event::{AssistantMessageEvent, AssistantMessageEventSender, thinking_text_at};
use crate::models::calculate_cost;
use crate::types::{
    AssistantContent, AssistantMessage, Context, ImageContent, Message, Model, StopReason,
    TextContent, ThinkingContent, Tool, ToolCall, Usage, UsageCost, UserContent, UserContentBlock,
    json_as_token_count,
};
use crate::utils::hash::short_hash;
use crate::utils::json_parse::{parse_json_with_repair, parse_streaming_json};
use crate::utils::sanitize::sanitize_surrogates;

// =============================================================================
// Text signatures
// =============================================================================

/// The `phase` carried by an assistant message item (`commentary` while the
/// model is still working, `final_answer` once it explicitly ended its turn).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextPhase {
    Commentary,
    FinalAnswer,
}

impl TextPhase {
    /// The wire value used by the Responses API.
    pub fn wire(self) -> &'static str {
        match self {
            TextPhase::Commentary => "commentary",
            TextPhase::FinalAnswer => "final_answer",
        }
    }

    fn from_wire(value: &str) -> Option<Self> {
        match value {
            "commentary" => Some(TextPhase::Commentary),
            "final_answer" => Some(TextPhase::FinalAnswer),
            _ => None,
        }
    }
}

/// Encode the `TextSignatureV1` envelope persisted on assistant text blocks.
fn encode_text_signature_v1(id: &str, phase: Option<TextPhase>) -> String {
    let mut payload = json!({ "v": 1, "id": id });
    if let Some(phase) = phase {
        set_field(&mut payload, "phase", json!(phase.wire()));
    }
    payload.to_string()
}

struct ParsedTextSignature {
    id: String,
    phase: Option<TextPhase>,
}

/// Parse a persisted text signature. The field carries either a legacy plain
/// message id or the JSON `TextSignatureV1` envelope `{v:1, id, phase}`.
fn parse_text_signature(signature: Option<&str>) -> Option<ParsedTextSignature> {
    let signature = signature?;
    if signature.starts_with('{')
        && let Ok(parsed) = parse_json_with_repair(signature)
        && parsed.get("v") == Some(&json!(1))
        && let Some(id) = parsed.get("id").and_then(Value::as_str)
    {
        let phase = parsed
            .get("phase")
            .and_then(Value::as_str)
            .and_then(TextPhase::from_wire);
        return Some(ParsedTextSignature {
            id: id.to_owned(),
            phase,
        });
    }
    Some(ParsedTextSignature {
        id: signature.to_owned(),
        phase: None,
    })
}

// =============================================================================
// Conversion options
// =============================================================================

/// Where deferred tool definitions are announced to the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeferredToolsMode {
    /// Sent as an `additional_tools` input item.
    AdditionalTools,
    /// Sent as a client-executed `tool_search_call`/`tool_search_output` pair.
    ToolSearch,
}

/// Options for [`convert_responses_messages`].
#[derive(Clone, Debug, Default)]
pub struct ConvertResponsesMessagesOptions {
    /// Emit the system prompt as a leading `developer`/`system` message.
    /// Default: true.
    pub include_system_prompt: Option<bool>,
    /// Grammar tool input property per tool name (grammar-constrained tools).
    pub grammar_tool_input_properties: Option<HashMap<String, String>>,
    /// Deferred tool definitions keyed by tool name.
    pub deferred_tools: Option<HashMap<String, Tool>>,
    /// How deferred tools are announced; `None` disables deferred loading.
    pub deferred_tools_mode: Option<DeferredToolsMode>,
    /// Options used when deferred tools are re-serialized.
    pub tool_options: Option<ConvertResponsesToolsOptions>,
}

/// Options for [`convert_responses_tools`].
#[derive(Clone, Debug, Default)]
pub struct ConvertResponsesToolsOptions {
    /// Fallback `strict` value for function tools without a constrained-sampling
    /// decision. Default: false.
    pub strict: Option<bool>,
    /// Whether the provider accepts the `strict` field. Default: true.
    pub supports_strict_mode: Option<bool>,
    /// Whether the provider accepts grammar `custom` tools. Default: false.
    pub supports_openai_grammar_tools: Option<bool>,
    /// Mark every tool with `defer_loading: true`.
    pub defer_loading: Option<bool>,
}

// =============================================================================
// Message conversion
// =============================================================================

/// Insert a field into a JSON object without panicking on non-objects.
fn set_field(target: &mut Value, key: &str, value: Value) {
    if let Some(object) = target.as_object_mut() {
        object.insert(key.to_owned(), value);
    }
}

/// Insert an optional string field, omitting the key when absent (mirrors the
/// TypeScript behavior of dropping `undefined` keys).
fn set_optional_field(target: &mut Value, key: &str, value: Option<&str>) {
    if let Some(value) = value {
        set_field(target, key, json!(value));
    }
}

/// OpenAI ids accept `[a-zA-Z0-9_-]` up to 64 characters with no trailing `_`.
fn normalize_id_part(part: &str) -> String {
    let sanitized: String = part
        .chars()
        .map(|char| {
            if char.is_ascii_alphanumeric() || char == '_' || char == '-' {
                char
            } else {
                '_'
            }
        })
        .collect();
    let bounded: String = if sanitized.chars().count() > 64 {
        sanitized.chars().take(64).collect()
    } else {
        sanitized
    };
    bounded.trim_end_matches('_').to_owned()
}

/// Hash a foreign (cross-provider) tool item id into a bounded `fc_<hash>` shape.
fn build_foreign_responses_item_id(item_id: &str) -> String {
    let normalized = format!("fc_{}", short_hash(item_id));
    if normalized.chars().count() > 64 {
        normalized.chars().take(64).collect()
    } else {
        normalized
    }
}

/// Normalize a `call_id|item_id` tool-call id for replay against `model`.
fn normalize_tool_call_id(
    id: &str,
    model: &Model,
    source: &AssistantMessage,
    allowed: &HashSet<&str>,
) -> String {
    if !allowed.contains(model.provider.as_str()) {
        return normalize_id_part(id);
    }
    let Some((call_id, item_id)) = id.split_once('|') else {
        return normalize_id_part(id);
    };
    let normalized_call_id = normalize_id_part(call_id);
    let is_foreign = source.provider != model.provider || source.api != model.api;
    let mut normalized_item_id = if is_foreign {
        build_foreign_responses_item_id(item_id)
    } else {
        normalize_id_part(item_id)
    };
    // OpenAI Responses API requires the item id to start with "fc".
    if !normalized_item_id.starts_with("fc_") {
        normalized_item_id = normalize_id_part(&format!("fc_{normalized_item_id}"));
    }
    format!("{normalized_call_id}|{normalized_item_id}")
}

/// Render tool-result content as either plain text or an input-content array.
fn convert_tool_result_output(model: &Model, content: &[UserContentBlock]) -> Value {
    let text_result = content
        .iter()
        .filter_map(|block| match block {
            UserContentBlock::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    let images: Vec<&ImageContent> = content
        .iter()
        .filter_map(|block| match block {
            UserContentBlock::Image(image) => Some(image),
            _ => None,
        })
        .collect();
    let has_text = !text_result.is_empty();

    if images.is_empty() || !model.supports_images() {
        let text = if has_text {
            text_result
        } else if !images.is_empty() {
            "(see attached image)".to_owned()
        } else {
            "(no tool output)".to_owned()
        };
        return json!(sanitize_surrogates(&text));
    }

    let mut output = Vec::with_capacity(images.len() + 1);
    if has_text {
        output.push(json!({ "type": "input_text", "text": sanitize_surrogates(&text_result) }));
    }
    for image in images {
        output.push(json!({
            "type": "input_image",
            "detail": "auto",
            "image_url": format!("data:{};base64,{}", image.mime_type, image.data),
        }));
    }
    Value::Array(output)
}

/// Convert a conversation into Responses `input` items.
///
/// `allowed_tool_call_providers` lists the providers whose composite
/// `call_id|item_id` tool-call ids may be replayed verbatim (after
/// normalization); other providers get sanitized ids.
pub fn convert_responses_messages(
    model: &Model,
    context: &Context,
    allowed_tool_call_providers: &[&str],
    options: Option<&ConvertResponsesMessagesOptions>,
) -> Result<Vec<Value>, String> {
    let mut items: Vec<Value> = Vec::new();
    let mut loaded_tool_names: HashSet<String> = HashSet::new();
    let allowed: HashSet<&str> = allowed_tool_call_providers.iter().copied().collect();

    let normalize_id = |id: &str, _target: &Model, source: &AssistantMessage| {
        normalize_tool_call_id(id, model, source, &allowed)
    };
    let transformed = transform_messages(&context.messages, model, Some(&normalize_id));

    let include_system_prompt = options
        .and_then(|options| options.include_system_prompt)
        .unwrap_or(true);
    if include_system_prompt && let Some(system_prompt) = &context.system_prompt {
        let supports_developer_role = model
            .openai_responses_compat()
            .and_then(|compat| compat.supports_developer_role)
            .unwrap_or(true);
        let role = if model.reasoning && supports_developer_role {
            "developer"
        } else {
            "system"
        };
        items.push(json!({ "role": role, "content": sanitize_surrogates(system_prompt) }));
    }

    let grammar_properties =
        || options.and_then(|options| options.grammar_tool_input_properties.as_ref());
    let deferred_tools = || options.and_then(|options| options.deferred_tools.as_ref());
    let deferred_tools_mode = options.and_then(|options| options.deferred_tools_mode);

    let mut msg_index = 0usize;
    for message in &transformed {
        match message {
            Message::User(user) => match &user.content {
                UserContent::Text(text) => {
                    items.push(json!({
                        "role": "user",
                        "content": [{ "type": "input_text", "text": sanitize_surrogates(text) }],
                    }));
                }
                UserContent::Blocks(blocks) => {
                    let content: Vec<Value> = blocks
                        .iter()
                        .map(|block| match block {
                            UserContentBlock::Text(text) => {
                                json!({ "type": "input_text", "text": sanitize_surrogates(&text.text) })
                            }
                            UserContentBlock::Image(image) => json!({
                                "type": "input_image",
                                "detail": "auto",
                                "image_url": format!("data:{};base64,{}", image.mime_type, image.data),
                            }),
                        })
                        .collect();
                    if content.is_empty() {
                        continue;
                    }
                    items.push(json!({ "role": "user", "content": content }));
                }
            },
            Message::Assistant(assistant) => {
                let is_same_provider_and_api =
                    assistant.provider == model.provider && assistant.api == model.api;
                let is_same_model = is_same_provider_and_api && assistant.model == model.id;
                let is_different_model = is_same_provider_and_api && assistant.model != model.id;
                let mut output: Vec<Value> = Vec::new();
                let mut text_block_index = 0usize;

                for block in &assistant.content {
                    match block {
                        AssistantContent::Thinking(thinking) => {
                            if let Some(signature) = &thinking.thinking_signature {
                                let reasoning_item =
                                    parse_json_with_repair(signature).map_err(|error| {
                                        format!("invalid thinking signature JSON: {error}")
                                    })?;
                                output.push(reasoning_item);
                            }
                        }
                        AssistantContent::Text(text) => {
                            let parsed = parse_text_signature(text.text_signature.as_deref());
                            let fallback_message_id = if text_block_index == 0 {
                                format!("msg_pi_{msg_index}")
                            } else {
                                format!("msg_pi_{msg_index}_{text_block_index}")
                            };
                            text_block_index += 1;
                            // OpenAI requires the message id to be at most 64 characters.
                            let message_id = match &parsed {
                                Some(parsed) if parsed.id.chars().count() > 64 => {
                                    format!("msg_{}", short_hash(&parsed.id))
                                }
                                Some(parsed) => parsed.id.clone(),
                                None => fallback_message_id,
                            };
                            let mut item = json!({
                                "type": "message",
                                "role": "assistant",
                                "content": [{
                                    "type": "output_text",
                                    "text": sanitize_surrogates(&text.text),
                                    "annotations": [],
                                }],
                                "status": "completed",
                                "id": message_id,
                            });
                            if let Some(phase) = parsed.as_ref().and_then(|parsed| parsed.phase) {
                                set_field(&mut item, "phase", json!(phase.wire()));
                            }
                            output.push(item);
                        }
                        AssistantContent::ToolCall(call) => {
                            let (call_id, item_id_raw) = match call.id.split_once('|') {
                                Some((call_id, item_id)) => (call_id, Some(item_id)),
                                None => (call.id.as_str(), None),
                            };
                            let custom_input_property = grammar_properties()
                                .and_then(|properties| properties.get(&call.name))
                                .map(String::as_str);
                            let mut item_id = item_id_raw;
                            let starts_with_fc = item_id.is_some_and(|id| id.starts_with("fc_"));
                            // Drop ids that would trip OpenAI's fc_/rs_ pairing validation:
                            // a different model's fc_* id, and any non-fc_* id replayed as a
                            // function_call (custom-tool ids such as ctc_* included).
                            if (is_different_model && starts_with_fc)
                                || (custom_input_property.is_none() && !starts_with_fc)
                            {
                                item_id = None;
                            }
                            let can_replay_namespace = is_same_model
                                || deferred_tools()
                                    .is_some_and(|tools| tools.contains_key(&call.name));

                            let mut item = if let Some(property) = custom_input_property {
                                let input =
                                    get_grammar_tool_input(&call.name, &call.arguments, property)?;
                                json!({
                                    "type": "custom_tool_call",
                                    "call_id": call_id,
                                    "name": call.name,
                                    "input": sanitize_surrogates(&input),
                                })
                            } else {
                                json!({
                                    "type": "function_call",
                                    "call_id": call_id,
                                    "name": call.name,
                                    "arguments": Value::Object(call.arguments.clone()).to_string(),
                                })
                            };
                            set_optional_field(&mut item, "id", item_id);
                            if can_replay_namespace {
                                set_optional_field(
                                    &mut item,
                                    "namespace",
                                    call.namespace.as_deref(),
                                );
                            }
                            output.push(item);
                        }
                    }
                }
                if output.is_empty() {
                    continue;
                }
                items.extend(output);
            }
            Message::ToolResult(result) => {
                let call_id = result
                    .tool_call_id
                    .split('|')
                    .next()
                    .unwrap_or(result.tool_call_id.as_str());
                let output = convert_tool_result_output(model, &result.content);
                let is_grammar_tool = grammar_properties()
                    .is_some_and(|properties| properties.contains_key(&result.tool_name));
                if is_grammar_tool {
                    items.push(json!({ "type": "custom_tool_call_output", "call_id": call_id, "output": output }));
                } else {
                    items.push(json!({ "type": "function_call_output", "call_id": call_id, "output": output }));
                }

                let mut deferred: Vec<Tool> = Vec::new();
                for name in result.added_tool_names.iter().flatten() {
                    let Some(tool) = deferred_tools().and_then(|tools| tools.get(name)) else {
                        continue;
                    };
                    if !loaded_tool_names.insert(name.clone()) {
                        continue;
                    }
                    deferred.push(tool.clone());
                }
                match deferred_tools_mode {
                    Some(DeferredToolsMode::AdditionalTools) if !deferred.is_empty() => {
                        let tools = convert_responses_tools(
                            &deferred,
                            options.and_then(|options| options.tool_options.as_ref()),
                        )?;
                        items.push(json!({ "type": "additional_tools", "role": "developer", "tools": tools }));
                    }
                    Some(DeferredToolsMode::ToolSearch) if !deferred.is_empty() => {
                        let names: Vec<&str> =
                            deferred.iter().map(|tool| tool.name.as_str()).collect();
                        let search_call_id = format!(
                            "pi_tool_load_{}",
                            short_hash(&format!("{}:{}", result.tool_call_id, names.join(",")))
                        );
                        items.push(json!({
                            "type": "tool_search_call",
                            "call_id": search_call_id,
                            "execution": "client",
                            "status": "completed",
                            "arguments": { "query": names.join(" "), "limit": names.len() },
                        }));
                        let mut tool_options = options
                            .and_then(|options| options.tool_options.clone())
                            .unwrap_or_default();
                        tool_options.defer_loading = Some(true);
                        let tools = convert_responses_tools(&deferred, Some(&tool_options))?;
                        items.push(json!({
                            "type": "tool_search_output",
                            "call_id": search_call_id,
                            "execution": "client",
                            "status": "completed",
                            "tools": tools,
                        }));
                    }
                    _ => {}
                }
            }
        }
        msg_index += 1;
    }

    Ok(items)
}

// =============================================================================
// Tool conversion
// =============================================================================

/// Convert tools into Responses `tools` entries: grammar tools become `custom`
/// tools with a `format`, everything else becomes a `function` tool with an
/// optional `strict` flag.
pub fn convert_responses_tools(
    tools: &[Tool],
    options: Option<&ConvertResponsesToolsOptions>,
) -> Result<Vec<Value>, String> {
    let default_strict = options.and_then(|options| options.strict).unwrap_or(false);
    let supports_strict_mode = options
        .and_then(|options| options.supports_strict_mode)
        .unwrap_or(true);
    let supports_openai_grammar_tools = options
        .and_then(|options| options.supports_openai_grammar_tools)
        .unwrap_or(false);
    let defer_loading = options
        .and_then(|options| options.defer_loading)
        .unwrap_or(false);

    let mut converted = Vec::with_capacity(tools.len());
    for tool in tools {
        if let Some(grammar) =
            resolve_grammar_constrained_sampling(tool, supports_openai_grammar_tools)?
        {
            let syntax = match grammar.format {
                crate::types::GrammarFormat::OpenaiLark => GrammarSyntax::Lark,
                crate::types::GrammarFormat::OpenaiRegex => GrammarSyntax::Regex,
            };
            let wire = WireTool::Custom(CustomToolParam {
                name: tool.name.clone(),
                description: Some(tool.description.clone()),
                format: CustomToolParamFormat::Grammar(CustomGrammarFormatParam {
                    syntax,
                    definition: grammar.definition,
                }),
                defer_loading: defer_loading.then_some(true),
                ..Default::default()
            });
            converted.push(serde_json::to_value(wire).map_err(|error| error.to_string())?);
            continue;
        }

        let constrained_strict = resolve_json_schema_strict_sampling(tool, supports_strict_mode)?;
        let strict = constrained_strict.unwrap_or(default_strict);
        let wire = WireTool::Function(FunctionTool {
            name: tool.name.clone(),
            description: Some(tool.description.clone()),
            parameters: Some(get_json_schema_tool_parameters(tool, Some(strict))),
            strict: supports_strict_mode.then_some(strict),
            defer_loading: defer_loading.then_some(true),
            ..Default::default()
        });
        converted.push(serde_json::to_value(wire).map_err(|error| error.to_string())?);
    }
    Ok(converted)
}

// =============================================================================
// Stream processing
// =============================================================================

/// Options for [`process_responses_stream`].
#[derive(Default)]
pub struct ResponsesStreamOptions<'a> {
    pub snapshot_mode: crate::types::EventSnapshotMode,
    /// Requested service tier; used when the response omits its own.
    pub service_tier: Option<&'a str>,
    /// Grammar tool input property per tool name.
    pub grammar_tool_input_properties: Option<&'a HashMap<String, String>>,
    /// Cost multiplier for a resolved service tier (model-aware pricing).
    pub service_tier_cost_multiplier: Option<&'a (dyn Fn(&str) -> f64 + Sync)>,
    /// Cooperative cancellation for the stream.
    pub signal: Option<&'a CancellationToken>,
}

/// Failure while consuming a Responses stream.
#[derive(Debug, thiserror::Error)]
pub enum ResponsesStreamError {
    /// A provider-reported stream failure (error event, failed response, or a
    /// stream that ended before a terminal response event).
    #[error("{0}")]
    Provider(String),
    /// An HTTP/transport failure surfaced by the SSE transport.
    #[error(transparent)]
    Http(#[from] ProviderHttpError),
    /// The request was aborted.
    #[error("request aborted")]
    Aborted,
}

impl ResponsesStreamError {
    fn provider(message: impl Into<String>) -> Self {
        ResponsesStreamError::Provider(message.into())
    }
}

/// The Responses stream events this adapter acts on; every other event is
/// ignored, matching the TypeScript dispatch chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResponsesEventType {
    ResponseCreated,
    OutputItemAdded,
    ReasoningSummaryTextDelta,
    ReasoningSummaryPartDone,
    ReasoningTextDelta,
    OutputTextDelta,
    RefusalDelta,
    FunctionCallArgumentsDelta,
    FunctionCallArgumentsDone,
    CustomToolCallInputDelta,
    CustomToolCallInputDone,
    OutputItemDone,
    Completed,
    Incomplete,
    Failed,
    Error,
    Unknown,
}

impl ResponsesEventType {
    fn from_wire(value: &str) -> Self {
        match value {
            "response.created" => ResponsesEventType::ResponseCreated,
            "response.output_item.added" => ResponsesEventType::OutputItemAdded,
            "response.reasoning_summary_text.delta" => {
                ResponsesEventType::ReasoningSummaryTextDelta
            }
            "response.reasoning_summary_part.done" => ResponsesEventType::ReasoningSummaryPartDone,
            "response.reasoning_text.delta" => ResponsesEventType::ReasoningTextDelta,
            "response.output_text.delta" => ResponsesEventType::OutputTextDelta,
            "response.refusal.delta" => ResponsesEventType::RefusalDelta,
            "response.function_call_arguments.delta" => {
                ResponsesEventType::FunctionCallArgumentsDelta
            }
            "response.function_call_arguments.done" => {
                ResponsesEventType::FunctionCallArgumentsDone
            }
            "response.custom_tool_call_input.delta" => ResponsesEventType::CustomToolCallInputDelta,
            "response.custom_tool_call_input.done" => ResponsesEventType::CustomToolCallInputDone,
            "response.output_item.done" => ResponsesEventType::OutputItemDone,
            "response.completed" => ResponsesEventType::Completed,
            "response.incomplete" => ResponsesEventType::Incomplete,
            "response.failed" => ResponsesEventType::Failed,
            "error" => ResponsesEventType::Error,
            _ => ResponsesEventType::Unknown,
        }
    }
}

fn wire_string(value: Option<&Value>) -> String {
    value.and_then(Value::as_str).unwrap_or_default().to_owned()
}

fn message_phase(phase: Option<MessagePhase>) -> Option<TextPhase> {
    phase.map(|phase| match phase {
        MessagePhase::Commentary => TextPhase::Commentary,
        MessagePhase::FinalAnswer => TextPhase::FinalAnswer,
    })
}

fn message_item_text(content: &[OutputMessageContent]) -> String {
    content
        .iter()
        .map(|part| match part {
            OutputMessageContent::OutputText(part) => part.text.as_str(),
            OutputMessageContent::Refusal(part) => part.refusal.as_str(),
        })
        .collect()
}

/// Decode supported output items using SDK wire types. Compatible gateways may
/// omit metadata or empty fields during item-added events; fill only those
/// defaults, retaining the original item separately for replay signatures.
fn parse_output_item(value: &Value) -> Result<Option<OutputItem>, ResponsesStreamError> {
    let Some(kind) = value.get("type").and_then(Value::as_str) else {
        return Ok(None);
    };
    if !matches!(
        kind,
        "reasoning" | "message" | "function_call" | "custom_tool_call"
    ) {
        return Ok(None);
    }
    let mut normalized = value.clone();
    let Some(object) = normalized.as_object_mut() else {
        return Ok(None);
    };
    match kind {
        "reasoning" => {
            object.entry("summary").or_insert_with(|| json!([]));
            for (key, kind) in [("summary", "summary_text"), ("content", "reasoning_text")] {
                if let Some(parts) = object.get_mut(key).and_then(Value::as_array_mut) {
                    for part in parts {
                        if let Some(part) = part.as_object_mut() {
                            part.entry("type").or_insert_with(|| json!(kind));
                            part.entry("text").or_insert_with(|| json!(""));
                        }
                    }
                }
            }
        }
        "message" => {
            object.entry("id").or_insert_with(|| json!(""));
            object.entry("role").or_insert_with(|| json!("assistant"));
            object
                .entry("status")
                .or_insert_with(|| json!("in_progress"));
            object.entry("content").or_insert_with(|| json!([]));
            if let Some(parts) = object.get_mut("content").and_then(Value::as_array_mut) {
                for part in parts {
                    if let Some(part) = part.as_object_mut()
                        && part.get("type").and_then(Value::as_str) == Some("output_text")
                    {
                        part.entry("annotations").or_insert_with(|| json!([]));
                    }
                }
            }
        }
        _ => {
            for key in ["call_id", "name"] {
                object.entry(key).or_insert_with(|| json!(""));
            }
            let key = if kind == "function_call" {
                "arguments"
            } else {
                "input"
            };
            object.entry(key).or_insert_with(|| json!(""));
            if kind == "custom_tool_call" {
                object.entry("id").or_insert_with(|| json!(""));
            }
        }
    }
    serde_json::from_value(normalized)
        .map(Some)
        .map_err(|error| {
            ResponsesStreamError::provider(format!("Invalid Responses output item: {error}"))
        })
}

/// An open output slot: the content index of the block being streamed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputSlot {
    Thinking { content_index: usize },
    Text { content_index: usize },
    ToolCall { content_index: usize },
}

/// Streaming scratch state for a tool-call block. `partial_json` mirrors the
/// TypeScript `partialJson` scratch field: it exists only while the call is
/// streaming and is dropped when the block is finalized.
struct ToolCallScratch {
    partial_json: Option<String>,
    custom_input: Option<CustomInputScratch>,
}

struct CustomInputScratch {
    property: String,
    buffer: GrammarToolInputJsonBuffer,
}

/// Maps a Responses status (plus an incomplete reason) onto a stop reason.
fn map_stop_reason(
    status: Option<&str>,
    incomplete_reason: Option<&str>,
) -> Result<(StopReason, Option<String>), String> {
    let Some(status) = status else {
        return Ok((StopReason::Stop, None));
    };
    match status {
        "completed" => Ok((StopReason::Stop, None)),
        "incomplete" => {
            if incomplete_reason == Some("max_output_tokens") {
                Ok((StopReason::Length, None))
            } else {
                let message = match incomplete_reason {
                    Some(reason) => format!("Response incomplete: {reason}"),
                    None => "Response incomplete without a provider reason".to_owned(),
                };
                Ok((StopReason::Error, Some(message)))
            }
        }
        "failed" | "cancelled" => Ok((StopReason::Error, None)),
        // These two are wonky: the response is not actually finished, but the
        // stream treats them as a stop.
        "in_progress" | "queued" => Ok((StopReason::Stop, None)),
        other => Err(format!("Unhandled stop reason: {other}")),
    }
}

fn output_index_of(event: &Value) -> usize {
    event
        .get("output_index")
        .and_then(Value::as_u64)
        .map(|index| index as usize)
        .unwrap_or(0)
}

/// Stateful processor for one Responses stream.
struct ResponsesStreamRunner<'a> {
    output: &'a mut AssistantMessage,
    sender: &'a mut AssistantMessageEventSender,
    model: &'a Model,
    options: Option<&'a ResponsesStreamOptions<'a>>,
    slots: HashMap<usize, OutputSlot>,
    /// Reasoning item id -> content index, for terminal signature backfill.
    reasoning_by_item_id: HashMap<String, usize>,
    /// Streaming scratch state keyed by content index.
    scratches: HashMap<usize, ToolCallScratch>,
    saw_terminal_response_event: bool,
}

impl<'a> ResponsesStreamRunner<'a> {
    async fn emit(&mut self, event: AssistantMessageEvent) {
        if !self.sender.push(event).await {
            self.sender.cancellation_token().cancel();
        }
    }

    fn push_block(&mut self, block: AssistantContent) -> usize {
        self.output.content.push(block);
        self.output.content.len() - 1
    }

    fn grammar_input_property(&self, tool_name: &str) -> Option<String> {
        self.options
            .and_then(|options| options.grammar_tool_input_properties)
            .and_then(|properties| properties.get(tool_name))
            .cloned()
    }

    /// A `final_answer` phase means the model explicitly ended its turn.
    fn apply_message_phase_stop_reason(&mut self, phase: Option<TextPhase>) {
        if phase == Some(TextPhase::FinalAnswer) {
            self.output.stop_reason = StopReason::Stop;
        }
    }

    /// Open an output slot for a new item, emitting its `*_start` event.
    async fn create_slot(
        &mut self,
        output_index: usize,
        item: &Value,
    ) -> Result<Option<OutputSlot>, ResponsesStreamError> {
        let Some(parsed) = parse_output_item(item)? else {
            return Ok(None);
        };
        let slot = match parsed {
            OutputItem::Reasoning(_) => {
                let content_index = self.push_block(AssistantContent::Thinking(ThinkingContent {
                    thinking: String::new(),
                    thinking_signature: None,
                    redacted: false,
                }));
                self.emit(AssistantMessageEvent::ThinkingStart {
                    content_index,
                    partial: crate::event::snapshot_message(
                        self.output,
                        self.options
                            .map_or(Default::default(), |options| options.snapshot_mode),
                    ),
                })
                .await;
                OutputSlot::Thinking { content_index }
            }
            OutputItem::Message(message) => {
                self.apply_message_phase_stop_reason(message_phase(message.phase));
                let content_index = self.push_block(AssistantContent::Text(TextContent {
                    text: String::new(),
                    text_signature: None,
                }));
                self.emit(AssistantMessageEvent::TextStart {
                    content_index,
                    partial: crate::event::snapshot_message(
                        self.output,
                        self.options
                            .map_or(Default::default(), |options| options.snapshot_mode),
                    ),
                })
                .await;
                OutputSlot::Text { content_index }
            }
            OutputItem::FunctionCall(call) => {
                let tool_call = ToolCall {
                    id: format!("{}|{}", call.call_id, call.id.unwrap_or_default()),
                    name: call.name,
                    arguments: serde_json::Map::new(),
                    thought_signature: None,
                    namespace: call.namespace,
                };
                let content_index = self.push_block(AssistantContent::ToolCall(tool_call));
                self.scratches.insert(
                    content_index,
                    ToolCallScratch {
                        partial_json: Some(call.arguments),
                        custom_input: None,
                    },
                );
                self.emit(AssistantMessageEvent::ToolCallStart {
                    content_index,
                    partial: crate::event::snapshot_message(
                        self.output,
                        self.options
                            .map_or(Default::default(), |options| options.snapshot_mode),
                    ),
                })
                .await;
                OutputSlot::ToolCall { content_index }
            }
            OutputItem::CustomToolCall(call) => {
                let input_property = self
                    .grammar_input_property(&call.name)
                    .unwrap_or_else(|| "input".to_owned());
                let mut arguments = serde_json::Map::new();
                arguments.insert(input_property.clone(), json!(call.input));
                let tool_call = ToolCall {
                    id: format!("{}|{}", call.call_id, call.id),
                    name: call.name,
                    arguments,
                    thought_signature: None,
                    namespace: call.namespace,
                };
                let content_index = self.push_block(AssistantContent::ToolCall(tool_call));
                self.scratches.insert(
                    content_index,
                    ToolCallScratch {
                        partial_json: None,
                        custom_input: Some(CustomInputScratch {
                            property: input_property,
                            buffer: GrammarToolInputJsonBuffer::default(),
                        }),
                    },
                );
                self.emit(AssistantMessageEvent::ToolCallStart {
                    content_index,
                    partial: crate::event::snapshot_message(
                        self.output,
                        self.options
                            .map_or(Default::default(), |options| options.snapshot_mode),
                    ),
                })
                .await;
                OutputSlot::ToolCall { content_index }
            }
            _ => return Ok(None),
        };
        self.slots.insert(output_index, slot);
        Ok(Some(slot))
    }

    async fn get_or_create_slot(
        &mut self,
        output_index: usize,
        item: &Value,
    ) -> Result<Option<OutputSlot>, ResponsesStreamError> {
        if let Some(slot) = self.slots.get(&output_index) {
            return Ok(Some(*slot));
        }
        self.create_slot(output_index, item).await
    }

    fn slot(&self, output_index: usize) -> Option<OutputSlot> {
        self.slots.get(&output_index).copied()
    }

    fn scratch(&self, content_index: usize) -> Option<&ToolCallScratch> {
        self.scratches.get(&content_index)
    }

    fn scratch_mut(&mut self, content_index: usize) -> Option<&mut ToolCallScratch> {
        self.scratches.get_mut(&content_index)
    }

    /// The accumulated custom-tool input string for a streaming block.
    fn custom_tool_call_input(&self, content_index: usize) -> String {
        let Some(scratch) = self.scratch(content_index) else {
            return String::new();
        };
        let Some(custom_input) = scratch.custom_input.as_ref() else {
            return String::new();
        };
        match self.output.content.get(content_index) {
            Some(AssistantContent::ToolCall(call)) => {
                match call.arguments.get(&custom_input.property) {
                    Some(Value::String(value)) => value.clone(),
                    _ => String::new(),
                }
            }
            _ => String::new(),
        }
    }

    /// Append a custom-tool input delta, re-emitting it as a JSON delta.
    fn append_custom_tool_call_input(
        &mut self,
        content_index: usize,
        next_input: &str,
        close: bool,
    ) -> Result<Option<String>, String> {
        // Access the scratch map directly (not through a `&mut self` helper) so
        // the content block can be mutated in the same scope.
        let Some(scratch) = self.scratches.get_mut(&content_index) else {
            return Ok(None);
        };
        let Some(custom_input) = scratch.custom_input.as_mut() else {
            return Ok(None);
        };
        let delta =
            custom_input
                .buffer
                .append_json_delta(&custom_input.property, next_input, close)?;
        let property = custom_input.property.clone();
        if let Some(AssistantContent::ToolCall(call)) = self.output.content.get_mut(content_index) {
            call.arguments.insert(property, json!(next_input));
        }
        Ok(delta)
    }

    async fn push_tool_call_delta(&mut self, content_index: usize, delta: Option<String>) {
        if let Some(delta) = delta {
            self.emit(AssistantMessageEvent::ToolCallDelta {
                content_index,
                delta,
                partial: crate::event::snapshot_message(
                    self.output,
                    self.options
                        .map_or(Default::default(), |options| options.snapshot_mode),
                ),
            })
            .await;
        }
    }

    async fn handle(
        &mut self,
        event_type: ResponsesEventType,
        event: &Value,
    ) -> Result<(), ResponsesStreamError> {
        match event_type {
            ResponsesEventType::ResponseCreated => {
                if let Some(id) = event.pointer("/response/id").and_then(Value::as_str) {
                    self.output.response_id = Some(id.to_owned());
                }
            }
            ResponsesEventType::OutputItemAdded => {
                let output_index = output_index_of(event);
                if let Some(item) = event.get("item") {
                    self.create_slot(output_index, item).await?;
                }
            }
            ResponsesEventType::ReasoningSummaryTextDelta
            | ResponsesEventType::ReasoningTextDelta
            | ResponsesEventType::ReasoningSummaryPartDone => {
                let Some(OutputSlot::Thinking { content_index }) =
                    self.slot(output_index_of(event))
                else {
                    return Ok(());
                };
                let delta = if event_type == ResponsesEventType::ReasoningSummaryPartDone {
                    // A finished summary part separates the next part.
                    "\n\n".to_owned()
                } else {
                    wire_string(event.get("delta"))
                };
                if delta.is_empty() {
                    return Ok(());
                }
                if let Some(AssistantContent::Thinking(block)) =
                    self.output.content.get_mut(content_index)
                {
                    block.thinking.push_str(&delta);
                }
                self.emit(AssistantMessageEvent::ThinkingDelta {
                    content_index,
                    delta,
                    partial: crate::event::snapshot_message(
                        self.output,
                        self.options
                            .map_or(Default::default(), |options| options.snapshot_mode),
                    ),
                })
                .await;
            }
            ResponsesEventType::OutputTextDelta | ResponsesEventType::RefusalDelta => {
                let Some(OutputSlot::Text { content_index }) = self.slot(output_index_of(event))
                else {
                    return Ok(());
                };
                let delta = wire_string(event.get("delta"));
                if delta.is_empty() {
                    return Ok(());
                }
                if let Some(AssistantContent::Text(block)) =
                    self.output.content.get_mut(content_index)
                {
                    block.text.push_str(&delta);
                }
                self.emit(AssistantMessageEvent::TextDelta {
                    content_index,
                    delta,
                    partial: crate::event::snapshot_message(
                        self.output,
                        self.options
                            .map_or(Default::default(), |options| options.snapshot_mode),
                    ),
                })
                .await;
            }
            ResponsesEventType::FunctionCallArgumentsDelta => {
                let Some(OutputSlot::ToolCall { content_index }) =
                    self.slot(output_index_of(event))
                else {
                    return Ok(());
                };
                let delta = wire_string(event.get("delta"));
                let has_partial = self
                    .scratch(content_index)
                    .is_some_and(|scratch| scratch.partial_json.is_some());
                if delta.is_empty() || !has_partial {
                    return Ok(());
                }
                if let Some(scratch) = self.scratch_mut(content_index)
                    && let Some(partial_json) = scratch.partial_json.as_mut()
                {
                    partial_json.push_str(&delta);
                    let parsed = parse_streaming_json(Some(partial_json));
                    if let Some(AssistantContent::ToolCall(call)) =
                        self.output.content.get_mut(content_index)
                    {
                        call.arguments = parsed.as_object().cloned().unwrap_or_default();
                    }
                }
                self.emit(AssistantMessageEvent::ToolCallDelta {
                    content_index,
                    delta,
                    partial: crate::event::snapshot_message(
                        self.output,
                        self.options
                            .map_or(Default::default(), |options| options.snapshot_mode),
                    ),
                })
                .await;
            }
            ResponsesEventType::FunctionCallArgumentsDone => {
                let Some(OutputSlot::ToolCall { content_index }) =
                    self.slot(output_index_of(event))
                else {
                    return Ok(());
                };
                let has_partial = self
                    .scratch(content_index)
                    .is_some_and(|scratch| scratch.partial_json.is_some());
                if !has_partial {
                    return Ok(());
                }
                let arguments = wire_string(event.get("arguments"));
                let previous_partial = self
                    .scratch(content_index)
                    .and_then(|scratch| scratch.partial_json.clone())
                    .unwrap_or_default();
                if let Some(scratch) = self.scratch_mut(content_index) {
                    scratch.partial_json = Some(arguments.clone());
                }
                let parsed = crate::utils::json_parse::parse_tool_arguments(&arguments)
                    .map_err(|error| ResponsesStreamError::provider(error.to_string()))?;
                if let Some(AssistantContent::ToolCall(call)) =
                    self.output.content.get_mut(content_index)
                {
                    call.arguments = parsed;
                }
                // Emit only the tail the deltas had not produced yet.
                if let Some(tail) = arguments.strip_prefix(&previous_partial)
                    && !tail.is_empty()
                {
                    self.emit(AssistantMessageEvent::ToolCallDelta {
                        content_index,
                        delta: tail.to_owned(),
                        partial: crate::event::snapshot_message(
                            self.output,
                            self.options
                                .map_or(Default::default(), |options| options.snapshot_mode),
                        ),
                    })
                    .await;
                }
            }
            ResponsesEventType::CustomToolCallInputDelta => {
                let Some(OutputSlot::ToolCall { content_index }) =
                    self.slot(output_index_of(event))
                else {
                    return Ok(());
                };
                let delta = wire_string(event.get("delta"));
                let next_input = format!("{}{}", self.custom_tool_call_input(content_index), delta);
                let emitted = self
                    .append_custom_tool_call_input(content_index, &next_input, false)
                    .map_err(ResponsesStreamError::provider)?;
                self.push_tool_call_delta(content_index, emitted).await;
            }
            ResponsesEventType::CustomToolCallInputDone => {
                let Some(OutputSlot::ToolCall { content_index }) =
                    self.slot(output_index_of(event))
                else {
                    return Ok(());
                };
                let input = wire_string(event.get("input"));
                let emitted = self
                    .append_custom_tool_call_input(content_index, &input, true)
                    .map_err(ResponsesStreamError::provider)?;
                self.push_tool_call_delta(content_index, emitted).await;
            }
            ResponsesEventType::OutputItemDone => {
                let output_index = output_index_of(event);
                let Some(item_value) = event.get("item") else {
                    return Ok(());
                };
                let Some(item) = parse_output_item(item_value)? else {
                    return Ok(());
                };
                if let OutputItem::Message(message) = &item {
                    self.apply_message_phase_stop_reason(message_phase(message.phase));
                }
                let slot = self.get_or_create_slot(output_index, item_value).await?;
                match item {
                    OutputItem::Reasoning(reasoning) => {
                        let Some(OutputSlot::Thinking { content_index }) = slot else {
                            return Ok(());
                        };
                        let summary_text = reasoning
                            .summary
                            .iter()
                            .map(|part| match part {
                                SummaryPart::SummaryText(part) => part.text.as_str(),
                            })
                            .collect::<Vec<_>>()
                            .join("\n\n");
                        let content_text = reasoning
                            .content
                            .as_deref()
                            .unwrap_or_default()
                            .iter()
                            .map(|part| match part {
                                ReasoningItemContent::ReasoningText(part) => part.text.as_str(),
                            })
                            .collect::<Vec<_>>()
                            .join("\n\n");
                        let thinking = if !summary_text.is_empty() {
                            summary_text
                        } else if !content_text.is_empty() {
                            content_text
                        } else {
                            match self.output.content.get(content_index) {
                                Some(AssistantContent::Thinking(block)) => block.thinking.clone(),
                                _ => String::new(),
                            }
                        };
                        if let Some(AssistantContent::Thinking(block)) =
                            self.output.content.get_mut(content_index)
                        {
                            block.thinking = thinking.clone();
                            // The whole reasoning item is the replay signature.
                            block.thinking_signature = Some(item_value.to_string());
                        }
                        if let Some(id) = reasoning.id.filter(|id| !id.is_empty()) {
                            self.reasoning_by_item_id.insert(id, content_index);
                        }
                        self.emit(AssistantMessageEvent::ThinkingEnd {
                            content_index,
                            content: thinking_text_at(&self.output.content, content_index),
                            partial: crate::event::snapshot_message(
                                self.output,
                                self.options
                                    .map_or(Default::default(), |options| options.snapshot_mode),
                            ),
                        })
                        .await;
                        self.slots.remove(&output_index);
                    }
                    OutputItem::Message(message) => {
                        let Some(OutputSlot::Text { content_index }) = slot else {
                            return Ok(());
                        };
                        let signature =
                            encode_text_signature_v1(&message.id, message_phase(message.phase));
                        if let Some(AssistantContent::Text(block)) =
                            self.output.content.get_mut(content_index)
                        {
                            block.text = message_item_text(&message.content);
                            block.text_signature = Some(signature);
                        }
                        self.emit(AssistantMessageEvent::TextEnd {
                            content_index,
                            content: message_item_text(&message.content),
                            partial: crate::event::snapshot_message(
                                self.output,
                                self.options
                                    .map_or(Default::default(), |options| options.snapshot_mode),
                            ),
                        })
                        .await;
                        self.slots.remove(&output_index);
                    }
                    OutputItem::FunctionCall(call) => {
                        let Some(OutputSlot::ToolCall { content_index }) = slot else {
                            return Ok(());
                        };
                        let has_partial = self
                            .scratch(content_index)
                            .is_some_and(|scratch| scratch.partial_json.is_some());
                        if !has_partial {
                            return Ok(());
                        }
                        let arguments = if !call.arguments.is_empty() {
                            call.arguments
                        } else {
                            self.scratch(content_index)
                                .and_then(|scratch| scratch.partial_json.clone())
                                .unwrap_or_else(|| "{}".to_owned())
                        };
                        let parsed = crate::utils::json_parse::parse_tool_arguments(&arguments)
                            .map_err(|error| ResponsesStreamError::provider(error.to_string()))?;
                        let tool_call = if let Some(AssistantContent::ToolCall(block)) =
                            self.output.content.get_mut(content_index)
                        {
                            block.arguments = parsed;
                            if let Some(namespace) = call.namespace {
                                block.namespace = Some(namespace);
                            }
                            block.clone()
                        } else {
                            return Ok(());
                        };
                        if let Some(scratch) = self.scratch_mut(content_index) {
                            // Strip the scratch buffer so replay only carries
                            // parsed arguments.
                            scratch.partial_json = None;
                        }
                        self.emit(AssistantMessageEvent::ToolCallEnd {
                            content_index,
                            tool_call,
                            partial: crate::event::snapshot_message(
                                self.output,
                                self.options
                                    .map_or(Default::default(), |options| options.snapshot_mode),
                            ),
                        })
                        .await;
                        self.slots.remove(&output_index);
                    }
                    OutputItem::CustomToolCall(call) => {
                        let Some(OutputSlot::ToolCall { content_index }) = slot else {
                            return Ok(());
                        };
                        let has_custom_input = self
                            .scratch(content_index)
                            .is_some_and(|scratch| scratch.custom_input.is_some());
                        if !has_custom_input {
                            return Ok(());
                        }
                        // A missing input field falls back to what streamed so far.
                        let input = if item_value.get("input").is_some_and(Value::is_string) {
                            call.input
                        } else {
                            self.custom_tool_call_input(content_index)
                        };
                        let emitted = self
                            .append_custom_tool_call_input(content_index, &input, true)
                            .map_err(ResponsesStreamError::provider)?;
                        self.push_tool_call_delta(content_index, emitted).await;
                        let tool_call = if let Some(AssistantContent::ToolCall(block)) =
                            self.output.content.get_mut(content_index)
                        {
                            if let Some(namespace) = call.namespace {
                                block.namespace = Some(namespace);
                            }
                            block.clone()
                        } else {
                            return Ok(());
                        };
                        if let Some(scratch) = self.scratch_mut(content_index) {
                            scratch.custom_input = None;
                        }
                        self.emit(AssistantMessageEvent::ToolCallEnd {
                            content_index,
                            tool_call,
                            partial: crate::event::snapshot_message(
                                self.output,
                                self.options
                                    .map_or(Default::default(), |options| options.snapshot_mode),
                            ),
                        })
                        .await;
                        self.slots.remove(&output_index);
                    }
                    _ => {}
                }
            }
            ResponsesEventType::Completed | ResponsesEventType::Incomplete => {
                if let Some(response) = event.get("response") {
                    self.finalize_response(response)?;
                }
            }
            ResponsesEventType::Failed => {
                self.saw_terminal_response_event = true;
                let response = event.get("response");
                self.output.raw_stop_reason = response
                    .and_then(|response| response.get("status"))
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                let error = response.and_then(|response| response.get("error"));
                let details = response.and_then(|response| response.get("incomplete_details"));
                let message = if let Some(error) = error {
                    format!(
                        "{}: {}",
                        wire_string(error.get("code")),
                        wire_string(error.get("message"))
                    )
                } else if let Some(reason) = details
                    .and_then(|details| details.get("reason"))
                    .and_then(Value::as_str)
                {
                    format!("incomplete: {reason}")
                } else {
                    "Unknown error (no error details in response)".to_owned()
                };
                return Err(ResponsesStreamError::provider(message));
            }
            ResponsesEventType::Error => {
                return Err(ResponsesStreamError::provider(format!(
                    "Error Code {}: {}",
                    wire_string(event.get("code")),
                    wire_string(event.get("message"))
                )));
            }
            ResponsesEventType::Unknown => {}
        }
        Ok(())
    }

    /// Apply the terminal response: usage, cost, service-tier pricing, and the
    /// mapped stop reason.
    fn finalize_response(&mut self, response: &Value) -> Result<(), ResponsesStreamError> {
        self.saw_terminal_response_event = true;
        self.backfill_reasoning_signatures(response.get("output"));
        if let Some(id) = response.get("id").and_then(Value::as_str) {
            self.output.response_id = Some(id.to_owned());
        }
        if let Some(usage) = response.get("usage") {
            let input_tokens = usage
                .get("input_tokens")
                .and_then(json_as_token_count)
                .unwrap_or(0);
            let output_tokens = usage
                .get("output_tokens")
                .and_then(json_as_token_count)
                .unwrap_or(0);
            let total_tokens = usage
                .get("total_tokens")
                .and_then(json_as_token_count)
                .unwrap_or(0);
            let details = usage.get("input_tokens_details");
            let cached_tokens = details
                .and_then(|details| details.get("cached_tokens"))
                .and_then(json_as_token_count)
                .unwrap_or(0);
            let cache_write_tokens = details
                .and_then(|details| details.get("cache_write_tokens"))
                .and_then(json_as_token_count)
                .unwrap_or(0);
            let reasoning_tokens = usage
                .get("output_tokens_details")
                .and_then(|details| details.get("reasoning_tokens"))
                .and_then(json_as_token_count);
            self.output.usage = Usage {
                // OpenAI includes cached and cache-write tokens in input_tokens,
                // so subtract both.
                input: input_tokens
                    .saturating_sub(cached_tokens)
                    .saturating_sub(cache_write_tokens),
                output: output_tokens,
                cache_read: cached_tokens,
                cache_write: cache_write_tokens,
                cache_write_1h: None,
                reasoning: Some(reasoning_tokens.unwrap_or(0)),
                total_tokens,
                cost: UsageCost::default(),
            };
        }
        calculate_cost(self.model, &mut self.output.usage);

        let response_service_tier = response.get("service_tier").and_then(Value::as_str);
        let service_tier =
            response_service_tier.or_else(|| self.options.and_then(|options| options.service_tier));
        if let (Some(service_tier), Some(multiplier)) = (
            service_tier,
            self.options
                .and_then(|options| options.service_tier_cost_multiplier),
        ) {
            let multiplier = multiplier(service_tier);
            if multiplier != 1.0 {
                let usage = &mut self.output.usage;
                usage.cost.input *= multiplier;
                usage.cost.output *= multiplier;
                usage.cost.cache_read *= multiplier;
                usage.cost.cache_write *= multiplier;
                usage.cost.total = usage.cost.input
                    + usage.cost.output
                    + usage.cost.cache_read
                    + usage.cost.cache_write;
            }
        }

        let status = response.get("status").and_then(Value::as_str);
        let incomplete_reason = response
            .get("incomplete_details")
            .and_then(|details| details.get("reason"))
            .and_then(Value::as_str);
        self.output.raw_stop_reason = match (status, incomplete_reason) {
            (Some(status), Some(reason)) => Some(format!("{status}.{reason}")),
            (Some(status), None) => Some(status.to_owned()),
            (None, _) => None,
        };
        let (stop_reason, error_message) =
            map_stop_reason(status, incomplete_reason).map_err(ResponsesStreamError::provider)?;
        self.output.stop_reason = stop_reason;
        self.output.error_message = error_message;
        if self
            .output
            .content
            .iter()
            .any(|block| block.tool_call().is_some())
            && self.output.stop_reason == StopReason::Stop
        {
            self.output.stop_reason = StopReason::ToolUse;
        }
        Ok(())
    }

    /// Azure OpenAI can omit `reasoning.encrypted_content` from
    /// `response.output_item.done` and provide it only in the terminal response.
    /// Backfill the persisted signature so `store: false` multi-turn replay
    /// stays stateless.
    fn backfill_reasoning_signatures(&mut self, response_output: Option<&Value>) {
        for item in response_output
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if item.get("type").and_then(Value::as_str) != Some("reasoning") {
                continue;
            }
            let Some(encrypted_content) = item.get("encrypted_content").and_then(Value::as_str)
            else {
                continue;
            };
            let Some(id) = item.get("id").and_then(Value::as_str) else {
                continue;
            };
            let Some(&content_index) = self.reasoning_by_item_id.get(id) else {
                continue;
            };
            let Some(AssistantContent::Thinking(block)) = self.output.content.get(content_index)
            else {
                continue;
            };
            let Some(signature) = block.thinking_signature.as_deref() else {
                continue;
            };
            let Ok(mut stored_item) = parse_json_with_repair(signature) else {
                continue;
            };
            if stored_item.get("encrypted_content").is_some() {
                continue;
            }
            set_field(
                &mut stored_item,
                "encrypted_content",
                json!(encrypted_content),
            );
            if let Some(AssistantContent::Thinking(block)) =
                self.output.content.get_mut(content_index)
            {
                block.thinking_signature = Some(stored_item.to_string());
            }
        }
    }
}

/// Pull the next SSE event, racing the cancellation token when one is set.
async fn next_sse_event<S>(
    events: &mut std::pin::Pin<Box<S>>,
    signal: Option<&CancellationToken>,
) -> Result<Option<Result<SseEvent, ProviderHttpError>>, ResponsesStreamError>
where
    S: Stream<Item = Result<SseEvent, ProviderHttpError>>,
{
    match signal {
        Some(signal) => {
            tokio::select! {
                biased;
                () = signal.cancelled() => Err(ResponsesStreamError::Aborted),
                event = events.next() => Ok(event),
            }
        }
        None => Ok(events.next().await),
    }
}

/// Consume a Responses SSE stream, emitting assistant-message events and
/// finalizing `output` from the terminal response event.
pub async fn process_responses_stream<S>(
    events: S,
    output: &mut AssistantMessage,
    sender: &mut AssistantMessageEventSender,
    model: &Model,
    options: Option<&ResponsesStreamOptions<'_>>,
) -> Result<(), ResponsesStreamError>
where
    S: Stream<Item = Result<SseEvent, ProviderHttpError>>,
{
    let mut runner = ResponsesStreamRunner {
        output,
        sender,
        model,
        options,
        slots: HashMap::new(),
        reasoning_by_item_id: HashMap::new(),
        scratches: HashMap::new(),
        saw_terminal_response_event: false,
    };
    let mut events = Box::pin(events);
    while let Some(event) =
        next_sse_event(&mut events, options.and_then(|options| options.signal)).await?
    {
        let event = event?;
        // The OpenAI SDK's SSE decoder ends the stream on the sentinel frame.
        if event.data.trim() == "[DONE]" {
            break;
        }
        let value: Value = parse_json_with_repair(&event.data).map_err(|error| {
            ResponsesStreamError::provider(format!("malformed Responses stream event: {error}"))
        })?;
        let event_type = ResponsesEventType::from_wire(
            value
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        );
        runner.handle(event_type, &value).await?;
        if let Some(raw) = value.get("response").and_then(|v| v.get("usage")).filter(|v|!v.is_null()) {
            runner.sender.report_usage(&runner.output.usage, raw);
        }
        if runner.sender.is_closed() || runner.sender.cancellation_token().is_cancelled() {
            return Err(ResponsesStreamError::Aborted);
        }
    }
    if !runner.saw_terminal_response_event {
        return Err(ResponsesStreamError::provider(
            "OpenAI Responses stream ended before a terminal response event",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Api, AssistantMessage, ModelCost, ProviderId, StopReason, Usage};

    fn responses_model() -> Model {
        Model {
            id: "step-5-preview".to_owned(),
            name: "Step 5 Preview".to_owned(),
            api: Api::OpenaiResponses,
            provider: ProviderId::new("step"),
            base_url: "https://api.stepfun.com/v1".to_owned(),
            reasoning: true,
            thinking_level_map: None,
            input: vec![crate::types::InputModality::Text],
            cost: ModelCost::default(),
            context_window: 128_000,
            max_tokens: 8192,
            sampling_params: None,
            headers: None,
            compat: None,
        }
    }

    fn assistant_message(model: &Model) -> AssistantMessage {
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
            timestamp: 1,
        }
    }

    fn sse_events(events: Vec<Value>) -> impl Stream<Item = Result<SseEvent, ProviderHttpError>> {
        futures::stream::iter(events.into_iter().map(|event| {
            Ok(SseEvent {
                event: String::new(),
                data: event.to_string(),
            })
        }))
    }

    async fn run_events(
        model: &Model,
        output: &mut AssistantMessage,
        events: Vec<Value>,
        options: Option<&ResponsesStreamOptions<'_>>,
    ) -> (Vec<AssistantMessageEvent>, Result<(), ResponsesStreamError>) {
        let (mut sender, mut stream) = crate::event::assistant_message_event_stream();
        let result =
            process_responses_stream(sse_events(events), output, &mut sender, model, options).await;
        // Drop the producer so the event channel closes and the drain ends.
        drop(sender);
        let mut collected = Vec::new();
        while let Some(event) = stream.next().await {
            collected.push(event);
        }
        (collected, result)
    }

    // ---- text signatures ----------------------------------------------------

    #[test]
    fn text_signature_round_trip() {
        assert_eq!(
            encode_text_signature_v1("msg_1", None),
            r#"{"v":1,"id":"msg_1"}"#
        );
        assert_eq!(
            encode_text_signature_v1("msg_1", Some(TextPhase::FinalAnswer)),
            r#"{"v":1,"id":"msg_1","phase":"final_answer"}"#
        );
        let parsed =
            parse_text_signature(Some(r#"{"v":1,"id":"msg_2","phase":"commentary"}"#)).unwrap();
        assert_eq!(parsed.id, "msg_2");
        assert_eq!(parsed.phase, Some(TextPhase::Commentary));
        // Legacy plain-id signatures keep the id and carry no phase.
        let legacy = parse_text_signature(Some("msg_legacy")).unwrap();
        assert_eq!(legacy.id, "msg_legacy");
        assert_eq!(legacy.phase, None);
        // A non-v1 JSON envelope falls back to the legacy plain-string form.
        let fallback = parse_text_signature(Some(r#"{"v":2,"id":"x"}"#)).unwrap();
        assert_eq!(fallback.id, r#"{"v":2,"id":"x"}"#);
        assert!(parse_text_signature(None).is_none());
    }

    // ---- message conversion -------------------------------------------------

    #[test]
    fn fallback_message_ids_are_unique_per_text_block() {
        let model = responses_model();
        let mut assistant = assistant_message(&model);
        assistant.api = Api::AnthropicMessages;
        assistant.provider = ProviderId::new("anthropic");
        assistant.model = "claude-opus-4-8".to_owned();
        assistant.content = vec![
            AssistantContent::Thinking(ThinkingContent {
                thinking: "private reasoning".to_owned(),
                thinking_signature: None,
                redacted: false,
            }),
            AssistantContent::Text(TextContent {
                text: "visible answer".to_owned(),
                text_signature: None,
            }),
            AssistantContent::Text(TextContent {
                text: "second block".to_owned(),
                text_signature: None,
            }),
        ];
        let context = Context {
            system_prompt: Some("You are concise.".to_owned()),
            messages: vec![
                Message::User(crate::types::UserMessage {
                    content: crate::types::UserContent::Text("hello".to_owned()),
                    timestamp: 1,
                }),
                Message::Assistant(Box::new(assistant)),
            ],
            tools: None,
        };
        let items = convert_responses_messages(&model, &context, &["openai"], None).unwrap();
        let messages: Vec<(String, String)> = items
            .iter()
            .filter(|item| item.get("type").and_then(Value::as_str) == Some("message"))
            .map(|item| {
                (
                    item["id"].as_str().unwrap_or_default().to_owned(),
                    item["content"][0]["text"]
                        .as_str()
                        .unwrap_or_default()
                        .to_owned(),
                )
            })
            .collect();
        // The unsigned thinking block becomes plain text cross-model, so all
        // three blocks are message items with unique fallback ids.
        assert_eq!(
            messages,
            vec![
                ("msg_pi_1".to_owned(), "private reasoning".to_owned()),
                ("msg_pi_1_1".to_owned(), "visible answer".to_owned()),
                ("msg_pi_1_2".to_owned(), "second block".to_owned()),
            ]
        );
    }

    #[test]
    fn empty_tool_result_uses_placeholder() {
        let model = responses_model();
        let mut assistant = assistant_message(&model);
        assistant.content = vec![AssistantContent::ToolCall(ToolCall {
            id: "tool-1".to_owned(),
            name: "bash".to_owned(),
            arguments: serde_json::Map::new(),
            thought_signature: None,
            namespace: None,
        })];
        assistant.stop_reason = StopReason::ToolUse;
        let context = Context {
            system_prompt: None,
            messages: vec![
                Message::User(crate::types::UserMessage {
                    content: crate::types::UserContent::Text("Run the command".to_owned()),
                    timestamp: 1,
                }),
                Message::Assistant(Box::new(assistant)),
                Message::ToolResult(Box::new(crate::types::ToolResultMessage {
                    tool_call_id: "tool-1".to_owned(),
                    tool_name: "bash".to_owned(),
                    content: vec![UserContentBlock::Text(TextContent {
                        text: String::new(),
                        text_signature: None,
                    })],
                    details: None,
                    usage: None,
                    added_tool_names: None,
                    is_error: false,
                    timestamp: 3,
                })),
            ],
            tools: None,
        };
        let items = convert_responses_messages(
            &model,
            &context,
            &["openai", "openai-codex", "opencode"],
            None,
        )
        .unwrap();
        let output = items
            .iter()
            .find(|item| item.get("type").and_then(Value::as_str) == Some("function_call_output"))
            .expect("function_call_output item");
        assert_eq!(output["output"], json!("(no tool output)"));
    }

    #[test]
    fn tool_result_output_shapes() {
        let mut vision_model = responses_model();
        vision_model.input = vec![
            crate::types::InputModality::Text,
            crate::types::InputModality::Image,
        ];
        let image = UserContentBlock::Image(ImageContent {
            data: "aaaa".to_owned(),
            mime_type: "image/png".to_owned(),
        });
        let text = UserContentBlock::Text(TextContent {
            text: "result".to_owned(),
            text_signature: None,
        });

        // A non-vision model never sees image payloads.
        let model = responses_model();
        assert_eq!(
            convert_tool_result_output(&model, std::slice::from_ref(&image)),
            json!("(see attached image)")
        );
        assert_eq!(
            convert_tool_result_output(
                &model,
                &[UserContentBlock::Text(TextContent {
                    text: String::new(),
                    text_signature: None,
                })]
            ),
            json!("(no tool output)")
        );
        // A vision model receives text plus image parts.
        let output = convert_tool_result_output(&vision_model, &[text.clone(), image.clone()]);
        assert_eq!(output[0]["type"], json!("input_text"));
        assert_eq!(output[0]["text"], json!("result"));
        assert_eq!(output[1]["type"], json!("input_image"));
        assert_eq!(output[1]["image_url"], json!("data:image/png;base64,aaaa"));
        // Multiple text blocks join with newlines; without images the output
        // stays a plain string.
        assert_eq!(
            convert_tool_result_output(&vision_model, &[text.clone(), text]),
            json!("result\nresult")
        );
    }

    #[test]
    fn foreign_tool_call_item_ids_are_hashed_into_fc_shape() {
        let mut model = responses_model();
        model.provider = ProviderId::new("openai");
        let raw_id = "call_4VnzVawQXPB9MgYib7CiQFEY|I9b95oN1wD/cHXKTw3PpRkL6KkCtzTJhUxMouMWYwHeTo2j3htzfSk7YPx2vifiIM4g3A8XXyOj8q4Bt6SLUG7gqY1E3ELkrkVQNHglRfUmWj84lqxJY+Puieb3VKyX0FB+83TUzn91cDMF/4gzt990IzqVrc+nIb9RRscRD070Du16q1glydVjWR0SBJsE6TbY/esOjFpqplogQqrajm1eI++f3eLi73R6q7hVusY0QbeFySVxABCjhN0lXB04caBe1rzHjYzul6MAXj7uq+0r17VLq+yrtyYhN12wkmFqHeqTyEei6EFPbMy24Nc+IbJlkP0OCg02W+gOnyBFcbi2ctvJFSOhSjt1CqBdqCnnhwUqXjbWiT0wh3DmLScRgTHmGkaI+oAcQQjfic65nxj+TnEkReA==";
        let mut assistant = assistant_message(&model);
        assistant.api = Api::OpenaiResponses;
        assistant.provider = ProviderId::new("github-copilot");
        assistant.model = "gpt-5.5".to_owned();
        assistant.content = vec![AssistantContent::ToolCall(ToolCall {
            id: raw_id.to_owned(),
            name: "edit".to_owned(),
            arguments: serde_json::json!({"path": "src/styles/app.css"})
                .as_object()
                .cloned()
                .unwrap(),
            thought_signature: None,
            namespace: None,
        })];
        let context = Context {
            system_prompt: None,
            messages: vec![Message::Assistant(Box::new(assistant))],
            tools: None,
        };
        let items = convert_responses_messages(&model, &context, &["openai"], None).unwrap();
        let function_call = items
            .iter()
            .find(|item| item.get("type").and_then(Value::as_str) == Some("function_call"))
            .expect("function_call item");
        let item_id = raw_id.split('|').nth(1).unwrap();
        assert_eq!(
            function_call["id"],
            json!(format!("fc_{}", short_hash(item_id)))
        );
        assert!(function_call["id"].as_str().unwrap().chars().count() <= 64);
        assert!(function_call["id"].as_str().unwrap().starts_with("fc_"));
    }

    #[test]
    fn system_prompt_role_follows_reasoning_and_compat() {
        let model = responses_model();
        let context = Context {
            system_prompt: Some("sys".to_owned()),
            messages: vec![Message::User(crate::types::UserMessage {
                content: crate::types::UserContent::Text("hi".to_owned()),
                timestamp: 1,
            })],
            tools: None,
        };
        let items = convert_responses_messages(&model, &context, &["openai"], None).unwrap();
        assert_eq!(items[0]["role"], json!("developer"));

        let mut non_reasoning = model.clone();
        non_reasoning.reasoning = false;
        let items =
            convert_responses_messages(&non_reasoning, &context, &["openai"], None).unwrap();
        assert_eq!(items[0]["role"], json!("system"));

        let mut compat_model = model.clone();
        compat_model.compat = Some(crate::types::ApiCompat {
            supports_developer_role: Some(false),
            ..Default::default()
        });
        let items = convert_responses_messages(&compat_model, &context, &["openai"], None).unwrap();
        assert_eq!(items[0]["role"], json!("system"));
    }

    #[test]
    fn user_blocks_become_input_text_and_image() {
        let mut model = responses_model();
        model.input = vec![
            crate::types::InputModality::Text,
            crate::types::InputModality::Image,
        ];
        let context = Context {
            system_prompt: None,
            messages: vec![Message::User(crate::types::UserMessage {
                content: crate::types::UserContent::Blocks(vec![
                    UserContentBlock::Text(TextContent {
                        text: "look".to_owned(),
                        text_signature: None,
                    }),
                    UserContentBlock::Image(ImageContent {
                        data: "aaaa".to_owned(),
                        mime_type: "image/png".to_owned(),
                    }),
                ]),
                timestamp: 1,
            })],
            tools: None,
        };
        let items = convert_responses_messages(&model, &context, &["openai"], None).unwrap();
        assert_eq!(items[0]["role"], json!("user"));
        assert_eq!(items[0]["content"][0]["type"], json!("input_text"));
        assert_eq!(items[0]["content"][1]["type"], json!("input_image"));
        assert_eq!(
            items[0]["content"][1]["image_url"],
            json!("data:image/png;base64,aaaa")
        );
        assert_eq!(items[0]["content"][1]["detail"], json!("auto"));
    }

    #[test]
    fn deferred_tools_emit_additional_tools_items() {
        let model = responses_model();
        let ping = Tool {
            name: "ping".to_owned(),
            description: "Ping a host".to_owned(),
            parameters: json!({"type": "object", "properties": {"host": {"type": "string"}}}),
            constrained_sampling: None,
        };
        let mut deferred = HashMap::new();
        deferred.insert("ping".to_owned(), ping.clone());
        let context = Context {
            system_prompt: None,
            messages: vec![Message::ToolResult(Box::new(
                crate::types::ToolResultMessage {
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
                    timestamp: 1,
                },
            ))],
            tools: Some(vec![ping]),
        };
        let items = convert_responses_messages(
            &model,
            &context,
            &["openai"],
            Some(&ConvertResponsesMessagesOptions {
                deferred_tools: Some(deferred),
                deferred_tools_mode: Some(DeferredToolsMode::AdditionalTools),
                ..Default::default()
            }),
        )
        .unwrap();
        let additional = items
            .iter()
            .find(|item| item.get("type").and_then(Value::as_str) == Some("additional_tools"))
            .expect("additional_tools item");
        assert_eq!(additional["role"], json!("developer"));
        assert_eq!(additional["tools"][0]["name"], json!("ping"));
    }

    #[test]
    fn deferred_tools_emit_tool_search_pair() {
        let model = responses_model();
        let ping = Tool {
            name: "ping".to_owned(),
            description: "Ping a host".to_owned(),
            parameters: json!({"type": "object", "properties": {"host": {"type": "string"}}}),
            constrained_sampling: None,
        };
        let mut deferred = HashMap::new();
        deferred.insert("ping".to_owned(), ping.clone());
        let context = Context {
            system_prompt: None,
            messages: vec![Message::ToolResult(Box::new(
                crate::types::ToolResultMessage {
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
                    timestamp: 1,
                },
            ))],
            tools: Some(vec![ping]),
        };
        let items = convert_responses_messages(
            &model,
            &context,
            &["openai"],
            Some(&ConvertResponsesMessagesOptions {
                deferred_tools: Some(deferred),
                deferred_tools_mode: Some(DeferredToolsMode::ToolSearch),
                ..Default::default()
            }),
        )
        .unwrap();
        let call = items
            .iter()
            .find(|item| item.get("type").and_then(Value::as_str) == Some("tool_search_call"))
            .expect("tool_search_call item");
        assert_eq!(call["execution"], json!("client"));
        assert_eq!(call["arguments"]["query"], json!("ping"));
        assert_eq!(call["arguments"]["limit"], json!(1));
        let output = items
            .iter()
            .find(|item| item.get("type").and_then(Value::as_str) == Some("tool_search_output"))
            .expect("tool_search_output item");
        assert_eq!(output["tools"][0]["defer_loading"], json!(true));
        assert_eq!(output["tools"][0]["name"], json!("ping"));
    }

    // ---- tool conversion ----------------------------------------------------

    #[test]
    fn function_tools_carry_name_at_top_level() {
        let tools = vec![Tool {
            name: "ping".to_owned(),
            description: "Ping a host".to_owned(),
            parameters: json!({"type": "object", "properties": {"host": {"type": "string"}}}),
            constrained_sampling: None,
        }];
        let converted = convert_responses_tools(&tools, None).unwrap();
        assert_eq!(converted[0]["type"], json!("function"));
        assert_eq!(converted[0]["name"], json!("ping"));
        // Strict mode is off by default and omitted when unsupported.
        assert_eq!(converted[0]["strict"], json!(false));
        assert!(converted[0].get("defer_loading").is_none());
    }

    #[test]
    fn strict_mode_converts_schema_and_sets_flag() {
        let tools = vec![Tool {
            name: "ping".to_owned(),
            description: "Ping a host".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {"host": {"type": "string"}, "port": {"type": "integer"}},
                "required": ["host"],
            }),
            constrained_sampling: None,
        }];
        let converted = convert_responses_tools(
            &tools,
            Some(&ConvertResponsesToolsOptions {
                strict: Some(true),
                supports_strict_mode: Some(true),
                ..Default::default()
            }),
        )
        .unwrap();
        assert_eq!(converted[0]["strict"], json!(true));
        assert_eq!(
            converted[0]["parameters"]["additionalProperties"],
            json!(false)
        );
        assert_eq!(
            converted[0]["parameters"]["required"],
            json!(["host", "port"])
        );
    }

    #[test]
    fn grammar_tools_become_custom_tools() {
        for (format, syntax, definition) in [
            (
                crate::types::GrammarFormat::OpenaiLark,
                "lark",
                "start: WORD",
            ),
            (
                crate::types::GrammarFormat::OpenaiRegex,
                "regex",
                "^[a-z]+$",
            ),
        ] {
            let tools = vec![Tool {
                name: "query".to_owned(),
                description: "Query".to_owned(),
                parameters: json!({ "type": "object", "properties": { "input": { "type": "string" } }, "required": ["input"] }),
                constrained_sampling: Some(crate::types::ConstrainedSamplingConfig::Grammar {
                    variants: HashMap::from([(format, definition.to_owned())]),
                }),
            }];
            let converted = convert_responses_tools(
                &tools,
                Some(&ConvertResponsesToolsOptions {
                    supports_openai_grammar_tools: Some(true),
                    defer_loading: Some(true),
                    ..Default::default()
                }),
            )
            .unwrap();
            assert_eq!(
                converted[0],
                json!({ "type": "custom", "name": "query", "description": "Query",
                "format": { "type": "grammar", "syntax": syntax, "definition": definition }, "defer_loading": true })
            );
            let mut compat = crate::api::openai_completions::resolve_compat(&responses_model());
            compat.supports_openai_grammar_tools = true;
            let chat = crate::api::openai_completions::convert_tools(&tools, &compat).unwrap();
            assert_eq!(
                chat[0],
                json!({ "type": "custom", "custom": { "name": "query", "description": "Query",
                "format": { "type": "grammar", "grammar": { "syntax": syntax, "definition": definition } } } })
            );
        }
    }

    // ---- stream processing --------------------------------------------------

    fn text_stream_events() -> Vec<Value> {
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
    async fn canned_text_stream_emits_stable_event_order() {
        let model = responses_model();
        let mut output = assistant_message(&model);
        let (events, result) = run_events(&model, &mut output, text_stream_events(), None).await;
        result.unwrap();
        let types: Vec<&str> = events
            .iter()
            .map(|event| match event {
                AssistantMessageEvent::Start { .. } => "start",
                AssistantMessageEvent::TextStart { .. } => "text_start",
                AssistantMessageEvent::TextDelta { .. } => "text_delta",
                AssistantMessageEvent::TextEnd { .. } => "text_end",
                AssistantMessageEvent::Done { .. } => "done",
                _ => "other",
            })
            .collect();
        // The `start` event is emitted by the adapter, not the stream processor,
        // so the processor yields text_start .. text_end here.
        assert_eq!(types, vec!["text_start", "text_delta", "text_end"]);
        assert_eq!(output.content.len(), 1);
        let AssistantContent::Text(text) = &output.content[0] else {
            panic!("expected text block");
        };
        assert_eq!(text.text, "Hello");
        assert_eq!(
            text.text_signature.as_deref(),
            Some(r#"{"v":1,"id":"msg_conf_1"}"#)
        );
        assert_eq!(output.stop_reason, StopReason::Stop);
        assert_eq!(output.response_id.as_deref(), Some("resp_conf_1"));
        assert_eq!(output.usage.input, 10);
        assert_eq!(output.usage.output, 2);
        assert_eq!(output.usage.total_tokens, 12);
    }

    #[tokio::test]
    async fn terminal_event_required() {
        let model = responses_model();
        let mut output = assistant_message(&model);
        let events = vec![
            json!({ "type": "response.created", "response": { "id": "resp_early_eof" } }),
            json!({ "type": "response.output_item.added", "output_index": 0, "item": { "type": "reasoning", "id": "rs_1", "summary": [] } }),
            json!({ "type": "response.reasoning_text.delta", "output_index": 0, "content_index": 0, "item_id": "rs_1", "delta": "partial" }),
        ];
        let (_, result) = run_events(&model, &mut output, events, None).await;
        let error = result.unwrap_err();
        assert_eq!(
            error.to_string(),
            "OpenAI Responses stream ended before a terminal response event"
        );
    }

    #[tokio::test]
    async fn done_sentinel_ends_stream() {
        let model = responses_model();
        let mut output = assistant_message(&model);
        let (mut sender, mut stream) = crate::event::assistant_message_event_stream();
        // The OpenAI SDK's SSE decoder stops on `[DONE]`; the adapter mirrors it.
        let result = process_responses_stream(
            futures::stream::iter(vec![
                Ok(SseEvent {
                    event: String::new(),
                    data: text_stream_events()[0].to_string(),
                }),
                Ok(SseEvent {
                    event: String::new(),
                    data: "[DONE]".to_owned(),
                }),
            ]),
            &mut output,
            &mut sender,
            &model,
            None,
        )
        .await;
        drop(sender);
        while stream.next().await.is_some() {}
        assert_eq!(
            result.unwrap_err().to_string(),
            "OpenAI Responses stream ended before a terminal response event"
        );
    }

    #[tokio::test]
    async fn completed_usage_subtracts_cached_and_write_tokens() {
        let model = responses_model();
        let mut output = assistant_message(&model);
        let events = vec![json!({
            "type": "response.completed",
            "response": {
                "id": "resp_completed",
                "status": "completed",
                "usage": {
                    "input_tokens": 20,
                    "output_tokens": 7,
                    "total_tokens": 27,
                    "input_tokens_details": { "cached_tokens": 2, "cache_write_tokens": 3 },
                },
            },
        })];
        let (_, result) = run_events(&model, &mut output, events, None).await;
        result.unwrap();
        assert_eq!(output.response_id.as_deref(), Some("resp_completed"));
        assert_eq!(output.stop_reason, StopReason::Stop);
        assert_eq!(output.raw_stop_reason.as_deref(), Some("completed"));
        assert_eq!(output.usage.input, 15);
        assert_eq!(output.usage.output, 7);
        assert_eq!(output.usage.cache_read, 2);
        assert_eq!(output.usage.cache_write, 3);
        assert_eq!(output.usage.total_tokens, 27);
        assert_eq!(output.usage.reasoning, Some(0));
    }

    #[tokio::test]
    async fn incomplete_max_output_tokens_maps_to_length() {
        let model = responses_model();
        let mut output = assistant_message(&model);
        let events = vec![json!({
            "type": "response.incomplete",
            "response": {
                "id": "resp_incomplete",
                "status": "incomplete",
                "incomplete_details": { "reason": "max_output_tokens" },
                "usage": {
                    "input_tokens": 30,
                    "output_tokens": 12,
                    "total_tokens": 42,
                    "input_tokens_details": { "cached_tokens": 5 },
                },
            },
        })];
        let (_, result) = run_events(&model, &mut output, events, None).await;
        result.unwrap();
        assert_eq!(output.stop_reason, StopReason::Length);
        assert_eq!(
            output.raw_stop_reason.as_deref(),
            Some("incomplete.max_output_tokens")
        );
        assert_eq!(output.usage.input, 25);
        assert_eq!(output.usage.cache_read, 5);
    }

    #[tokio::test]
    async fn incomplete_content_filter_is_an_error() {
        let model = responses_model();
        let mut output = assistant_message(&model);
        let events = vec![json!({
            "type": "response.incomplete",
            "response": {
                "id": "resp_incomplete",
                "status": "incomplete",
                "incomplete_details": { "reason": "content_filter" },
            },
        })];
        let (_, result) = run_events(&model, &mut output, events, None).await;
        result.unwrap();
        assert_eq!(output.stop_reason, StopReason::Error);
        assert_eq!(
            output.error_message.as_deref(),
            Some("Response incomplete: content_filter")
        );
    }

    #[tokio::test]
    async fn failed_response_surfaces_provider_error() {
        let model = responses_model();
        let mut output = assistant_message(&model);
        let events = vec![json!({
            "type": "response.failed",
            "response": {
                "id": "resp_failed",
                "status": "failed",
                "error": { "code": "server_error", "message": "boom" },
            },
        })];
        let (_, result) = run_events(&model, &mut output, events, None).await;
        assert_eq!(result.unwrap_err().to_string(), "server_error: boom");
        assert_eq!(output.raw_stop_reason.as_deref(), Some("failed"));
    }

    #[tokio::test]
    async fn message_phase_tracks_stop_reason() {
        let model = responses_model();
        for (phases, expected) in [
            (
                vec!["commentary", "commentary"],
                vec![StopReason::Pending, StopReason::Pending],
            ),
            (
                vec!["final_answer", "final_answer"],
                vec![StopReason::Stop, StopReason::Stop],
            ),
            (
                vec!["commentary", "final_answer"],
                vec![StopReason::Pending, StopReason::Stop],
            ),
        ] {
            let mut output = assistant_message(&model);
            let events = vec![
                json!({
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": { "type": "message", "id": "msg_phase", "role": "assistant", "status": "in_progress", "content": [], "phase": phases[0] },
                }),
                json!({
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {
                        "type": "message",
                        "id": "msg_phase",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{ "type": "output_text", "text": "answer", "annotations": [] }],
                        "phase": phases[1],
                    },
                }),
                json!({ "type": "response.completed", "response": { "id": "resp_phase", "status": "completed" } }),
            ];
            let (emitted, result) = run_events(&model, &mut output, events, None).await;
            result.unwrap();
            let observed: Vec<StopReason> = emitted
                .iter()
                .map(|event| event.message().stop_reason)
                .collect();
            assert_eq!(observed, expected);
            assert_eq!(output.stop_reason, StopReason::Stop);
        }
    }

    #[tokio::test]
    async fn invalid_sdk_output_items_fail_instead_of_disappearing() {
        let model = responses_model();
        for kind in ["response.output_item.added", "response.output_item.done"] {
            let mut output = assistant_message(&model);
            let events = vec![json!({
                "type": kind, "output_index": 0,
                "item": { "type": "function_call", "id": "fc_bad", "call_id": "call_bad",
                    "name": "lookup", "arguments": { "unexpected": "object" } },
            })];
            let (_, result) = run_events(&model, &mut output, events, None).await;
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("Invalid Responses output item")
            );
            assert!(output.content.is_empty());
        }
    }

    #[test]
    fn sdk_item_defaults_preserve_wire_data_and_unknown_items() {
        let raw = json!({ "type": "message", "id": "msg_test", "content": [
            { "type": "output_text", "text": "hello" },
            { "type": "refusal", "refusal": "refused" },
        ], "extension": { "future": true } });
        let original = raw.clone();
        let Some(OutputItem::Message(message)) = parse_output_item(&raw).unwrap() else {
            panic!("expected SDK message");
        };
        assert_eq!(message_item_text(&message.content), "hellorefused");
        assert_eq!(raw, original);
        assert!(
            parse_output_item(&json!({ "type": "future_item", "extra": 42 }))
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn sdk_custom_call_missing_final_input_keeps_streamed_text() {
        let model = responses_model();
        let mut output = assistant_message(&model);
        let item = json!({ "type": "custom_tool_call", "id": "ctc_test", "call_id": "call_test", "name": "query" });
        let events = vec![
            json!({ "type": "response.output_item.added", "output_index": 0, "item": item }),
            json!({ "type": "response.custom_tool_call_input.delta", "output_index": 0, "delta": "SELECT 1" }),
            json!({ "type": "response.output_item.done", "output_index": 0, "item": item }),
            json!({ "type": "response.completed", "response": { "id": "resp_test", "status": "completed" } }),
        ];
        let (_, result) = run_events(&model, &mut output, events, None).await;
        result.unwrap();
        assert_eq!(
            output.content[0].tool_call().unwrap().arguments["input"],
            json!("SELECT 1")
        );
    }

    #[tokio::test]
    async fn tool_calls_flip_stop_to_tool_use() {
        let model = responses_model();
        let mut output = assistant_message(&model);
        let events = vec![
            json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": { "type": "function_call", "id": "fc_test", "call_id": "call_test", "name": "lookup", "arguments": "" },
            }),
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_test",
                    "call_id": "call_test",
                    "name": "lookup",
                    "arguments": "{\"value\":\"hello\"}",
                    "namespace": "dynamic_tools",
                },
            }),
            json!({ "type": "response.completed", "response": { "id": "resp_test", "status": "completed" } }),
        ];
        let (_, result) = run_events(&model, &mut output, events, None).await;
        result.unwrap();
        let AssistantContent::ToolCall(call) = &output.content[0] else {
            panic!("expected toolCall block");
        };
        assert_eq!(call.id, "call_test|fc_test");
        assert_eq!(call.name, "lookup");
        assert_eq!(call.arguments["value"], json!("hello"));
        assert_eq!(call.namespace.as_deref(), Some("dynamic_tools"));
        assert_eq!(output.stop_reason, StopReason::ToolUse);
    }

    #[tokio::test]
    async fn final_function_call_rejects_incomplete_arguments() {
        let model = responses_model();
        for event_type in [
            "response.function_call_arguments.done",
            "response.output_item.done",
        ] {
            let mut output = assistant_message(&model);
            let item = json!({ "type": "function_call", "id": "fc_test", "call_id": "call_test",
                "name": "lookup", "arguments": "" });
            let arguments = r#"{"value":"unfinished"#;
            let mut done = item.clone();
            done["arguments"] = json!(arguments);
            let events = vec![
                json!({ "type": "response.output_item.added", "output_index": 0, "item": item }),
                json!({ "type": event_type, "output_index": 0, "arguments": arguments, "item": done }),
            ];
            let (emitted, result) = run_events(&model, &mut output, events, None).await;
            assert!(result.is_err());
            assert!(
                !emitted
                    .iter()
                    .any(|event| matches!(event, AssistantMessageEvent::ToolCallEnd { .. }))
            );
        }
    }

    #[tokio::test]
    async fn function_call_arguments_deltas_and_done() {
        let model = responses_model();
        let mut output = assistant_message(&model);
        let arguments = r#"{"path":"README.md","content":"updated"}"#;
        let events = vec![
            json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": { "type": "function_call", "id": "fc_test", "call_id": "call_test", "name": "edit", "arguments": "" },
            }),
            json!({ "type": "response.function_call_arguments.delta", "output_index": 0, "delta": "{\"path\":\"README.md\"" }),
            json!({ "type": "response.function_call_arguments.delta", "output_index": 0, "delta": ",\"content\":\"updated\"}" }),
            json!({ "type": "response.function_call_arguments.done", "output_index": 0, "arguments": arguments }),
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": { "type": "function_call", "id": "fc_test", "call_id": "call_test", "name": "edit", "arguments": arguments },
            }),
            json!({ "type": "response.completed", "response": { "id": "resp_test", "status": "completed" } }),
        ];
        let (emitted, result) = run_events(&model, &mut output, events, None).await;
        result.unwrap();
        let AssistantContent::ToolCall(call) = &output.content[0] else {
            panic!("expected toolCall block");
        };
        assert_eq!(call.arguments["path"], json!("README.md"));
        assert_eq!(call.arguments["content"], json!("updated"));
        // Two deltas plus the tail emitted by the `done` event.
        let deltas: Vec<&str> = emitted
            .iter()
            .filter_map(|event| match event {
                AssistantMessageEvent::ToolCallDelta { delta, .. } => Some(delta.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(deltas.len(), 2);
        assert!(
            emitted
                .iter()
                .any(|event| matches!(event, AssistantMessageEvent::ToolCallEnd { .. }))
        );
    }

    #[tokio::test]
    async fn custom_tool_call_streams_grammar_json_deltas() {
        let model = responses_model();
        let mut output = assistant_message(&model);
        let mut grammar = HashMap::new();
        grammar.insert("query".to_owned(), "input".to_owned());
        let options = ResponsesStreamOptions {
            grammar_tool_input_properties: Some(&grammar),
            ..Default::default()
        };
        let events = vec![
            json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": { "type": "custom_tool_call", "id": "ctc_test", "call_id": "call_test", "name": "query", "input": "" },
            }),
            json!({ "type": "response.custom_tool_call_input.delta", "output_index": 0, "delta": "hel" }),
            json!({ "type": "response.custom_tool_call_input.delta", "output_index": 0, "delta": "lo" }),
            json!({ "type": "response.custom_tool_call_input.done", "output_index": 0, "input": "hello" }),
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "custom_tool_call",
                    "id": "ctc_test",
                    "call_id": "call_test",
                    "name": "query",
                    "input": "hello",
                    "namespace": "dynamic_tools",
                },
            }),
            json!({ "type": "response.completed", "response": { "id": "resp_test", "status": "completed" } }),
        ];
        let (emitted, result) = run_events(&model, &mut output, events, Some(&options)).await;
        result.unwrap();
        let AssistantContent::ToolCall(call) = &output.content[0] else {
            panic!("expected toolCall block");
        };
        assert_eq!(call.id, "call_test|ctc_test");
        assert_eq!(call.arguments["input"], json!("hello"));
        assert_eq!(call.namespace.as_deref(), Some("dynamic_tools"));
        let deltas: Vec<&str> = emitted
            .iter()
            .filter_map(|event| match event {
                AssistantMessageEvent::ToolCallDelta { delta, .. } => Some(delta.as_str()),
                _ => None,
            })
            .collect();
        // The grammar buffer re-emits the custom input as JSON deltas: the
        // opening fragment, the middle chunk, and the closing fragment from the
        // `input.done` event.
        assert_eq!(deltas, vec!["{\"input\":\"hel", "lo", "\"}"]);
    }

    #[tokio::test]
    async fn reasoning_signature_backfilled_from_terminal_response() {
        let model = responses_model();
        let mut output = assistant_message(&model);
        let events = vec![
            json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": { "type": "reasoning", "id": "rs_backfill", "summary": [] },
            }),
            json!({ "type": "response.reasoning_summary_text.delta", "output_index": 0, "delta": "thought" }),
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "reasoning",
                    "id": "rs_backfill",
                    "summary": [{ "type": "summary_text", "text": "summary text" }],
                },
            }),
            json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_backfill",
                    "status": "completed",
                    "output": [{
                        "type": "reasoning",
                        "id": "rs_backfill",
                        "encrypted_content": "enc_secret",
                        "summary": [{ "type": "summary_text", "text": "summary text" }],
                    }],
                },
            }),
        ];
        let (_, result) = run_events(&model, &mut output, events, None).await;
        result.unwrap();
        let AssistantContent::Thinking(block) = &output.content[0] else {
            panic!("expected thinking block");
        };
        assert_eq!(block.thinking, "summary text");
        let signature: Value =
            serde_json::from_str(block.thinking_signature.as_deref().unwrap()).unwrap();
        assert_eq!(signature["id"], json!("rs_backfill"));
        assert_eq!(signature["encrypted_content"], json!("enc_secret"));
    }

    #[tokio::test]
    async fn unknown_events_are_ignored() {
        let model = responses_model();
        let mut output = assistant_message(&model);
        let mut events = text_stream_events();
        events.insert(
            1,
            json!({ "type": "response.output_text.annotation.added", "output_index": 0, "annotation": "x" }),
        );
        let (_, result) = run_events(&model, &mut output, events, None).await;
        result.unwrap();
        assert_eq!(output.stop_reason, StopReason::Stop);
    }

    #[tokio::test]
    async fn malformed_event_data_errors() {
        let model = responses_model();
        let mut output = assistant_message(&model);
        let (mut sender, mut stream) = crate::event::assistant_message_event_stream();
        let result = process_responses_stream(
            futures::stream::iter(vec![Ok(SseEvent {
                event: String::new(),
                data: "{not json".to_owned(),
            })]),
            &mut output,
            &mut sender,
            &model,
            None,
        )
        .await;
        drop(sender);
        while stream.next().await.is_some() {}
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn transport_error_propagates() {
        let model = responses_model();
        let mut output = assistant_message(&model);
        let (mut sender, mut stream) = crate::event::assistant_message_event_stream();
        let result = process_responses_stream(
            futures::stream::iter(vec![Err(ProviderHttpError::transport("SSE stream failed"))]),
            &mut output,
            &mut sender,
            &model,
            None,
        )
        .await;
        drop(sender);
        while stream.next().await.is_some() {}
        assert!(matches!(result, Err(ResponsesStreamError::Http(_))));
    }

    #[tokio::test]
    async fn service_tier_multiplier_scales_cost() {
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
        let mut output = assistant_message(&model);
        let multiplier = |tier: &str| if tier == "flex" { 0.5 } else { 1.0 };
        let options = ResponsesStreamOptions {
            service_tier: Some("flex"),
            service_tier_cost_multiplier: Some(&multiplier),
            ..Default::default()
        };
        let events = vec![json!({
            "type": "response.completed",
            "response": {
                "id": "resp_tier",
                "status": "completed",
                "service_tier": "flex",
                "usage": {
                    "input_tokens": 100_000,
                    "output_tokens": 100_000,
                    "total_tokens": 200_000,
                    "input_tokens_details": { "cached_tokens": 0 },
                },
            },
        })];
        let (_, result) = run_events(&model, &mut output, events, Some(&options)).await;
        result.unwrap();
        let scale = 100_000.0 / 1_000_000.0;
        assert!((output.usage.cost.input - 1.0 * 0.5 * scale).abs() < 1e-12);
        assert!((output.usage.cost.output - 2.0 * 0.5 * scale).abs() < 1e-12);
        assert!((output.usage.cost.total - 3.0 * 0.5 * scale).abs() < 1e-12);
    }

    #[tokio::test]
    async fn aborted_stream_returns_aborted() {
        let model = responses_model();
        let mut output = assistant_message(&model);
        let token = CancellationToken::new();
        token.cancel();
        let options = ResponsesStreamOptions {
            signal: Some(&token),
            ..Default::default()
        };
        let (mut sender, mut stream) = crate::event::assistant_message_event_stream();
        let result = process_responses_stream(
            sse_events(text_stream_events()),
            &mut output,
            &mut sender,
            &model,
            Some(&options),
        )
        .await;
        drop(sender);
        while stream.next().await.is_some() {}
        assert!(matches!(result, Err(ResponsesStreamError::Aborted)));
    }

    #[test]
    fn stop_reason_mapping() {
        assert_eq!(
            map_stop_reason(None, None).unwrap(),
            (StopReason::Stop, None)
        );
        assert_eq!(
            map_stop_reason(Some("completed"), None).unwrap(),
            (StopReason::Stop, None)
        );
        assert_eq!(
            map_stop_reason(Some("incomplete"), Some("max_output_tokens")).unwrap(),
            (StopReason::Length, None)
        );
        let (reason, message) =
            map_stop_reason(Some("incomplete"), Some("content_filter")).unwrap();
        assert_eq!(reason, StopReason::Error);
        assert_eq!(
            message.as_deref(),
            Some("Response incomplete: content_filter")
        );
        assert_eq!(
            map_stop_reason(Some("failed"), None).unwrap().0,
            StopReason::Error
        );
        assert_eq!(
            map_stop_reason(Some("queued"), None).unwrap().0,
            StopReason::Stop
        );
        assert!(map_stop_reason(Some("mystery"), None).is_err());
    }

    #[test]
    fn compat_projection_carries_session_affinity() {
        // The adapter reads the owned projection returned by
        // `Model::openai_responses_compat()`.
        let compat = crate::types::ApiCompat {
            session_affinity_format: Some(crate::types::SessionAffinityFormat::Openrouter),
            ..Default::default()
        };
        let projection = compat.openai_responses();
        assert_eq!(
            projection.session_affinity_format,
            Some(crate::types::SessionAffinityFormat::Openrouter)
        );
        assert_eq!(projection.supports_developer_role, None);
    }
}
