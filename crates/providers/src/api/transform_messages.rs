//! Cross-provider message normalization: image downgrades, thinking replay
//! rules, tool-call id normalization, and synthetic tool results for orphaned
//! calls.

use std::collections::HashMap;

use crate::types::{
    AssistantContent, AssistantMessage, ImageContent, Message, Model, TextContent, ToolCall,
    ToolResultMessage, UserContentBlock,
};

const NON_VISION_USER_IMAGE_PLACEHOLDER: &str = "(image omitted: model does not support images)";
const NON_VISION_TOOL_IMAGE_PLACEHOLDER: &str =
    "(tool image omitted: model does not support images)";

/// Cross-provider tool-call id normalizer (OpenAI Responses ids are 450+
/// chars with special characters; Anthropic requires `^[a-zA-Z0-9_-]+$` up to
/// 64 chars).
pub type ToolCallIdNormalizer<'a> = &'a dyn Fn(&str, &Model, &AssistantMessage) -> String;

/// Consecutive images collapse into a single placeholder; a text block equal
/// to the placeholder keeps the collapse.
fn replace_images_with_placeholders(
    content: &[UserContentBlock],
    placeholder: &str,
) -> Vec<UserContentBlock> {
    let mut result = Vec::with_capacity(content.len());
    let mut previous_was_placeholder = false;
    for block in content {
        match block {
            UserContentBlock::Image(_) => {
                if !previous_was_placeholder {
                    result.push(UserContentBlock::Text(TextContent {
                        text: placeholder.to_owned(),
                        text_signature: None,
                    }));
                }
                previous_was_placeholder = true;
            }
            UserContentBlock::Text(text) => {
                previous_was_placeholder = text.text == placeholder;
                result.push(block.clone());
            }
        }
    }
    result
}

fn downgrade_unsupported_images(messages: Vec<Message>, model: &Model) -> Vec<Message> {
    if model.supports_images() {
        return messages;
    }
    messages
        .into_iter()
        .map(|message| match message {
            Message::User(mut user) => {
                user.content = match user.content {
                    crate::types::UserContent::Blocks(blocks) => {
                        crate::types::UserContent::Blocks(replace_images_with_placeholders(
                            &blocks,
                            NON_VISION_USER_IMAGE_PLACEHOLDER,
                        ))
                    }
                    other => other,
                };
                Message::User(user)
            }
            Message::ToolResult(mut result) => {
                result.content = replace_images_with_placeholders(
                    &result.content,
                    NON_VISION_TOOL_IMAGE_PLACEHOLDER,
                );
                Message::ToolResult(result)
            }
            other => other,
        })
        .collect()
}

/// Normalize messages for cross-provider replay:
/// - downgrade images to placeholders for non-vision models;
/// - keep thinking blocks only for same-model replay (redacted thinking drops
///   cross-model; unsigned thinking becomes plain text);
/// - normalize tool call ids (OpenAI Responses ids are 450+ chars; Anthropic
///   requires `^[a-zA-Z0-9_-]+$` up to 64 chars);
/// - skip errored/aborted assistant turns;
/// - insert synthetic error tool results for orphaned tool calls.
pub fn transform_messages(
    messages: &[Message],
    model: &Model,
    normalize_tool_call_id: Option<ToolCallIdNormalizer<'_>>,
) -> Vec<Message> {
    let mut tool_call_id_map: HashMap<String, String> = HashMap::new();
    let image_aware = downgrade_unsupported_images(messages.to_vec(), model);

    let mut transformed: Vec<Message> = Vec::with_capacity(image_aware.len());
    for message in image_aware {
        match message {
            Message::User(message) => transformed.push(Message::User(message)),
            Message::ToolResult(mut result) => {
                if let Some(normalized) = tool_call_id_map.get(&result.tool_call_id)
                    && *normalized != result.tool_call_id
                {
                    result.tool_call_id = normalized.clone();
                }
                transformed.push(Message::ToolResult(result));
            }
            Message::Assistant(mut assistant) => {
                let is_same_model = assistant.provider == model.provider
                    && assistant.api == model.api
                    && assistant.model == model.id;

                let mut content: Vec<AssistantContent> =
                    Vec::with_capacity(assistant.content.len());
                for block in std::mem::take(&mut assistant.content) {
                    match block {
                        AssistantContent::Thinking(thinking) => {
                            // Redacted thinking is opaque encrypted content, only valid
                            // for the same model; drop it cross-model.
                            if thinking.redacted {
                                if is_same_model {
                                    content.push(AssistantContent::Thinking(thinking));
                                }
                                continue;
                            }
                            // Same model keeps signed thinking blocks for replay even
                            // when the text is empty (OpenAI encrypted reasoning).
                            if is_same_model && thinking.thinking_signature.is_some() {
                                content.push(AssistantContent::Thinking(thinking));
                                continue;
                            }
                            if thinking.thinking.trim().is_empty() {
                                continue;
                            }
                            if is_same_model {
                                content.push(AssistantContent::Thinking(thinking));
                            } else {
                                content.push(AssistantContent::Text(TextContent {
                                    text: thinking.thinking,
                                    text_signature: None,
                                }));
                            }
                        }
                        AssistantContent::Text(text) => {
                            // Signatures are only valid for the same model.
                            if is_same_model {
                                content.push(AssistantContent::Text(text));
                            } else {
                                content.push(AssistantContent::Text(TextContent {
                                    text: text.text,
                                    text_signature: None,
                                }));
                            }
                        }
                        AssistantContent::ToolCall(call) => {
                            let mut call = call;
                            if !is_same_model {
                                call.thought_signature = None;
                                if let Some(normalize) = normalize_tool_call_id {
                                    let normalized_id = normalize(&call.id, model, &assistant);
                                    if normalized_id != call.id {
                                        tool_call_id_map
                                            .insert(call.id.clone(), normalized_id.clone());
                                        call.id = normalized_id;
                                    }
                                }
                            }
                            content.push(AssistantContent::ToolCall(call));
                        }
                    }
                }
                assistant.content = content;
                transformed.push(Message::Assistant(assistant));
            }
        }
    }

    // Second pass: drop incomplete turns, insert synthetic tool results for
    // orphaned tool calls.
    let mut result: Vec<Message> = Vec::with_capacity(transformed.len());
    let mut pending_tool_calls: Vec<ToolCall> = Vec::new();
    let mut existing_tool_result_ids: std::collections::HashSet<String> =
        std::collections::HashSet::new();

    macro_rules! insert_synthetic_tool_results {
        () => {
            for call in std::mem::take(&mut pending_tool_calls) {
                if !existing_tool_result_ids.contains(&call.id) {
                    result.push(Message::tool_result(ToolResultMessage {
                        tool_call_id: call.id,
                        tool_name: call.name,
                        content: vec![UserContentBlock::Text(TextContent {
                            text: "No result provided".to_owned(),
                            text_signature: None,
                        })],
                        details: None,
                        usage: None,
                        added_tool_names: None,
                        is_error: true,
                        timestamp: crate::utils::now_ms(),
                    }));
                }
            }
            existing_tool_result_ids.clear();
        };
    }

    for message in transformed {
        match message {
            Message::Assistant(assistant) => {
                insert_synthetic_tool_results!();
                // Skip errored/aborted assistant messages entirely: they are
                // incomplete turns whose replay can cause provider errors
                // (e.g. OpenAI "reasoning without following item").
                if matches!(
                    assistant.stop_reason,
                    crate::types::StopReason::Error | crate::types::StopReason::Aborted
                ) {
                    continue;
                }
                let tool_calls: Vec<ToolCall> = assistant
                    .content
                    .iter()
                    .filter_map(AssistantContent::tool_call)
                    .cloned()
                    .collect();
                if !tool_calls.is_empty() {
                    pending_tool_calls = tool_calls;
                    existing_tool_result_ids.clear();
                }
                result.push(Message::Assistant(assistant));
            }
            Message::ToolResult(tool_result) => {
                existing_tool_result_ids.insert(tool_result.tool_call_id.clone());
                result.push(Message::ToolResult(tool_result));
            }
            Message::User(user) => {
                // A user message interrupts tool flow: synthesize results for
                // orphaned calls.
                insert_synthetic_tool_results!();
                result.push(Message::User(user));
            }
        }
    }

    // If the conversation ends with unresolved tool calls, synthesize results.
    insert_synthetic_tool_results!();

    result
}

#[allow(unused)]
fn unused_image_content_guard(_: &ImageContent) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Api, ModelCost, ProviderId, StopReason, ThinkingContent, Usage};

    fn model(supports_images: bool) -> Model {
        let mut input = vec![crate::types::InputModality::Text];
        if supports_images {
            input.push(crate::types::InputModality::Image);
        }
        Model {
            id: "m".to_owned(),
            name: "m".to_owned(),
            api: Api::AnthropicMessages,
            provider: ProviderId::new("test"),
            base_url: "http://localhost".to_owned(),
            reasoning: false,
            thinking_level_map: None,
            input,
            cost: ModelCost::default(),
            context_window: 1000,
            max_tokens: 100,
            sampling_params: None,
            headers: None,
            compat: None,
        }
    }

    fn assistant(content: Vec<AssistantContent>, stop_reason: StopReason) -> Message {
        Message::assistant(AssistantMessage {
            content,
            api: Api::AnthropicMessages,
            provider: ProviderId::new("test"),
            model: "m".to_owned(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: Usage::default(),
            stop_reason,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            timestamp: 1,
        })
    }

    #[test]
    fn images_downgraded_for_non_vision_models() {
        let messages = vec![Message::User(crate::types::UserMessage {
            content: crate::types::UserContent::Blocks(vec![
                UserContentBlock::Text(TextContent {
                    text: "look".to_owned(),
                    text_signature: None,
                }),
                UserContentBlock::Image(ImageContent {
                    data: "aaaa".to_owned(),
                    mime_type: "image/png".to_owned(),
                }),
                UserContentBlock::Image(ImageContent {
                    data: "bbbb".to_owned(),
                    mime_type: "image/png".to_owned(),
                }),
            ]),
            timestamp: 1,
        })];
        let transformed = transform_messages(&messages, &model(false), None);
        let Message::User(user) = &transformed[0] else {
            panic!("expected user message");
        };
        let blocks = user.content.blocks().unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(
            blocks[1].text().unwrap().text,
            NON_VISION_USER_IMAGE_PLACEHOLDER
        );
    }

    #[test]
    fn cross_model_thinking_becomes_text() {
        let messages = vec![assistant(
            vec![AssistantContent::Thinking(ThinkingContent {
                thinking: "deep thought".to_owned(),
                thinking_signature: Some("sig".to_owned()),
                redacted: false,
            })],
            StopReason::Stop,
        )];
        let mut other_model = model(true);
        other_model.id = "other".to_owned();
        let transformed = transform_messages(&messages, &other_model, None);
        let Message::Assistant(message) = &transformed[0] else {
            panic!("expected assistant");
        };
        assert_eq!(message.content.len(), 1);
        assert_eq!(message.content[0].text().unwrap().text, "deep thought");
    }

    #[test]
    fn redacted_thinking_dropped_cross_model_kept_same_model() {
        let thinking = || {
            AssistantContent::Thinking(ThinkingContent {
                thinking: String::new(),
                thinking_signature: Some("enc".to_owned()),
                redacted: true,
            })
        };
        let messages = vec![assistant(vec![thinking()], StopReason::Stop)];
        let same_model = transform_messages(&messages, &model(true), None);
        assert_eq!(same_model.len(), 1);
        let mut other = model(true);
        other.id = "other".to_owned();
        let cross = transform_messages(&messages, &other, None);
        assert_eq!(cross.len(), 1);
        let Message::Assistant(message) = &cross[0] else {
            panic!("expected assistant");
        };
        assert!(message.content.is_empty());
    }

    #[test]
    fn orphaned_tool_calls_get_synthetic_results() {
        let call = ToolCall {
            id: "call_1".to_owned(),
            name: "search".to_owned(),
            arguments: Default::default(),
            thought_signature: None,
            namespace: None,
        };
        let messages = vec![
            assistant(vec![AssistantContent::ToolCall(call)], StopReason::ToolUse),
            Message::User(crate::types::UserMessage {
                content: crate::types::UserContent::Text("next".to_owned()),
                timestamp: 2,
            }),
        ];
        let transformed = transform_messages(&messages, &model(true), None);
        assert_eq!(transformed.len(), 3);
        let Message::ToolResult(synthetic) = &transformed[1] else {
            panic!("expected synthetic tool result");
        };
        assert!(synthetic.is_error);
        assert_eq!(synthetic.tool_call_id, "call_1");
        assert_eq!(
            synthetic.content[0].text().unwrap().text,
            "No result provided"
        );
    }

    #[test]
    fn errored_assistant_turns_skipped() {
        let messages = vec![
            assistant(Vec::new(), StopReason::Error),
            assistant(Vec::new(), StopReason::Stop),
        ];
        let transformed = transform_messages(&messages, &model(true), None);
        assert_eq!(transformed.len(), 1);
    }

    #[test]
    fn tool_call_ids_normalized_cross_model() {
        let mut other = model(true);
        other.id = "other".to_owned();
        other.api = Api::OpenaiResponses;
        let call = ToolCall {
            id: "resp_call_450_chars".to_owned(),
            name: "search".to_owned(),
            arguments: Default::default(),
            thought_signature: None,
            namespace: None,
        };
        let messages = vec![
            assistant(vec![AssistantContent::ToolCall(call)], StopReason::ToolUse),
            Message::tool_result(ToolResultMessage {
                tool_call_id: "resp_call_450_chars".to_owned(),
                tool_name: "search".to_owned(),
                content: vec![UserContentBlock::Text(TextContent {
                    text: "ok".to_owned(),
                    text_signature: None,
                })],
                details: None,
                usage: None,
                added_tool_names: None,
                is_error: false,
                timestamp: 2,
            }),
        ];
        let normalize = |id: &str, _: &Model, _: &AssistantMessage| format!("normalized_{id}");
        let transformed = transform_messages(&messages, &other, Some(&normalize));
        let Message::ToolResult(result) = &transformed[1] else {
            panic!("expected tool result");
        };
        assert_eq!(result.tool_call_id, "normalized_resp_call_450_chars");
    }
}
