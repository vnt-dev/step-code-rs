//! Context token estimation from message/tool shapes.

use crate::types::{Context, Message, Tool, Usage, UserContent};

const CHARS_PER_TOKEN: usize = 4;
const ESTIMATED_IMAGE_CHARS: usize = 4800;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContextUsageEstimate {
    /// Estimated total context tokens.
    pub tokens: u64,
    /// Tokens reported by the most recent applicable assistant usage block.
    pub usage_tokens: u64,
    /// Estimated tokens after the most recent applicable assistant usage block.
    pub trailing_tokens: u64,
    /// Index of the applicable message that provided usage, or `None`.
    pub last_usage_index: Option<usize>,
}

/// Total context tokens for a reported usage block: `total_tokens` when the
/// provider reported it, else the sum of the component counters.
pub fn calculate_context_tokens(usage: &Usage) -> u64 {
    if usage.total_tokens != 0 {
        return usage.total_tokens;
    }
    usage
        .input
        .saturating_add(usage.output)
        .saturating_add(usage.cache_read)
        .saturating_add(usage.cache_write)
}

fn estimate_user_content_chars(content: &UserContent) -> usize {
    match content {
        UserContent::Text(text) => text.chars().count(),
        UserContent::Blocks(blocks) => blocks
            .iter()
            .map(|block| match block {
                crate::types::UserContentBlock::Text(text) => text.text.chars().count(),
                crate::types::UserContentBlock::Image(_) => ESTIMATED_IMAGE_CHARS,
            })
            .sum(),
    }
}

/// Estimate tokens for a text string (ceil(chars / 4)).
pub fn estimate_text_tokens(text: &str) -> u64 {
    text.chars().count().div_ceil(CHARS_PER_TOKEN) as u64
}

pub fn estimate_user_content_tokens(content: &UserContent) -> u64 {
    estimate_user_content_chars(content).div_ceil(CHARS_PER_TOKEN) as u64
}

/// Estimate tokens for a whole message (any role).
pub fn estimate_message_tokens(message: &Message) -> u64 {
    if let Message::User(message) = message {
        return estimate_user_content_tokens(&message.content);
    }
    if let Message::ToolResult(message) = message {
        return estimate_user_content_tokens(&UserContent::Blocks(message.content.clone()));
    }
    let Message::Assistant(message) = message else {
        unreachable!("remaining variant is assistant");
    };
    let chars: usize = message
        .content
        .iter()
        .map(|block| match block {
            crate::types::AssistantContent::Text(text) => text.text.chars().count(),
            crate::types::AssistantContent::Thinking(thinking) => thinking.thinking.chars().count(),
            crate::types::AssistantContent::ToolCall(call) => {
                call.name.chars().count()
                    + serde_json::to_string(&call.arguments)
                        .map(|s| s.len())
                        .unwrap_or_default()
            }
        })
        .sum();
    chars.div_ceil(CHARS_PER_TOKEN) as u64
}

struct LastUsage {
    usage: Usage,
    index: usize,
}

fn get_last_assistant_usage_info(messages: &[Message]) -> Option<LastUsage> {
    let mut latest_prefix_timestamp: u64 = 0;
    let mut usage_info: Option<LastUsage> = None;

    for (index, message) in messages.iter().enumerate() {
        if let Message::Assistant(assistant) = message {
            // A newer prefix message inserted after this response (e.g. a
            // compaction summary) invalidates its usage for the current prefix.
            let usage_applies_to_prefix = assistant.timestamp >= latest_prefix_timestamp;
            if usage_applies_to_prefix
                && !matches!(
                    assistant.stop_reason,
                    crate::types::StopReason::Aborted | crate::types::StopReason::Error
                )
                && calculate_context_tokens(&assistant.usage) > 0
            {
                usage_info = Some(LastUsage {
                    usage: assistant.usage.clone(),
                    index,
                });
            }
        }
        latest_prefix_timestamp = latest_prefix_timestamp.max(message.timestamp());
    }

    usage_info
}

fn estimate_messages(messages: &[Message]) -> ContextUsageEstimate {
    if let Some(info) = get_last_assistant_usage_info(messages) {
        let usage_tokens = calculate_context_tokens(&info.usage);
        let trailing_tokens: u64 = messages[info.index + 1..]
            .iter()
            .map(estimate_message_tokens)
            .sum();
        return ContextUsageEstimate {
            tokens: usage_tokens + trailing_tokens,
            usage_tokens,
            trailing_tokens,
            last_usage_index: Some(info.index),
        };
    }

    let tokens: u64 = messages.iter().map(estimate_message_tokens).sum();
    ContextUsageEstimate {
        tokens,
        usage_tokens: 0,
        trailing_tokens: tokens,
        last_usage_index: None,
    }
}

fn estimate_tools_tokens(tools: &[Tool]) -> u64 {
    if tools.is_empty() {
        return 0;
    }
    estimate_text_tokens(
        &serde_json::to_string(tools).unwrap_or_else(|_| "[unserializable]".to_owned()),
    )
}

/// Estimate context tokens for a full request context.
pub fn estimate_context_tokens(context: &Context) -> ContextUsageEstimate {
    let estimate = estimate_messages(&context.messages);
    if let Some(last_usage_index) = estimate.last_usage_index {
        let added_names: std::collections::HashSet<&str> = context.messages[last_usage_index + 1..]
            .iter()
            .filter_map(Message::as_tool_result)
            .flat_map(|result| result.added_tool_names.iter().flatten().map(String::as_str))
            .collect();
        let added_tools: Vec<Tool> = context
            .tools
            .iter()
            .flatten()
            .filter(|tool| added_names.contains(tool.name.as_str()))
            .cloned()
            .collect();
        let added_tool_tokens = estimate_tools_tokens(&added_tools);
        return ContextUsageEstimate {
            tokens: estimate.tokens + added_tool_tokens,
            usage_tokens: estimate.usage_tokens,
            trailing_tokens: estimate.trailing_tokens + added_tool_tokens,
            last_usage_index: Some(last_usage_index),
        };
    }

    let prefix_tokens = context
        .system_prompt
        .as_deref()
        .map(estimate_text_tokens)
        .unwrap_or(0)
        + estimate_tools_tokens(context.tools.as_deref().unwrap_or_default());

    ContextUsageEstimate {
        tokens: estimate.tokens + prefix_tokens,
        usage_tokens: estimate.usage_tokens,
        trailing_tokens: estimate.trailing_tokens + prefix_tokens,
        last_usage_index: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        AssistantContent, StopReason, TextContent, ToolResultMessage, UserContentBlock, UserMessage,
    };

    fn user(text: &str, timestamp: u64) -> Message {
        Message::User(UserMessage {
            content: UserContent::Text(text.to_owned()),
            timestamp,
        })
    }

    fn assistant(
        text: &str,
        total_tokens: u64,
        stop_reason: StopReason,
        timestamp: u64,
    ) -> Message {
        Message::assistant(crate::types::AssistantMessage {
            content: vec![AssistantContent::Text(TextContent {
                text: text.to_owned(),
                text_signature: None,
            })],
            api: crate::types::Api::OpenaiCompletions,
            provider: crate::types::ProviderId::new("test"),
            model: "m".to_owned(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: Usage {
                total_tokens,
                ..Usage::default()
            },
            stop_reason,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            timestamp,
        })
    }

    #[test]
    fn plain_messages_estimate_without_usage() {
        // Per-message ceil: ceil(5/4) + ceil(4/4) = 2 + 1 = 3.
        let estimate = estimate_messages(&[user("1234 ", 1), user("abcd", 2)]);
        assert_eq!(estimate.tokens, 3);
        assert_eq!(estimate.last_usage_index, None);
    }

    #[test]
    fn usage_anchors_estimates() {
        let estimate = estimate_messages(&[
            user("0123456789", 1),
            assistant("x", 100, StopReason::Stop, 2),
            user("12345", 3),
        ]);
        assert_eq!(estimate.usage_tokens, 100);
        assert_eq!(estimate.trailing_tokens, 2);
        assert_eq!(estimate.tokens, 102);
        assert_eq!(estimate.last_usage_index, Some(1));
    }

    #[test]
    fn aborted_and_errored_usage_ignored() {
        let estimate = estimate_messages(&[
            user("", 1),
            assistant("x", 100, StopReason::Aborted, 2),
            assistant("y", 200, StopReason::Error, 3),
        ]);
        assert_eq!(estimate.last_usage_index, None);
    }

    #[test]
    fn tool_results_count_added_tool_tokens() {
        let context = Context {
            system_prompt: None,
            messages: vec![
                user("hi", 1),
                assistant("x", 50, StopReason::Stop, 2),
                Message::tool_result(ToolResultMessage {
                    tool_call_id: "c1".to_owned(),
                    tool_name: "lazy".to_owned(),
                    content: vec![UserContentBlock::Text(TextContent {
                        text: "done".to_owned(),
                        text_signature: None,
                    })],
                    details: None,
                    usage: None,
                    added_tool_names: Some(vec!["lazy".to_owned()]),
                    is_error: false,
                    timestamp: 3,
                }),
            ],
            tools: Some(vec![Tool {
                name: "lazy".to_owned(),
                description: "d".to_owned(),
                parameters: serde_json::json!({"type": "object", "properties": {}}),
                constrained_sampling: None,
            }]),
        };
        let estimate = estimate_context_tokens(&context);
        assert_eq!(estimate.last_usage_index, Some(1));
        assert!(estimate.trailing_tokens > 1);
    }
}
