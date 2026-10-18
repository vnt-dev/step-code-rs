//! Content measurement, hashing, marker, classification, and in-place text-block
//! rewrite helpers shared by the projection modules. Everything here is pure and
//! depends only on `step_harness_providers` message types.
//!
//! Ported from `@step-harness/agent-core`
//! `src/harness/compaction/projection-content.ts`.

use std::sync::LazyLock;

use regex::Regex;
use step_harness_providers::types::{
    AssistantContent, Message, TextContent, UserContent, UserContentBlock, UserMessage,
};

// ============================================================================
// Rewrite markers
// ============================================================================

/// Machine-readable marker prefix for cut content.
pub const PROJECTION_CUT_MARKER_PREFIX: &str = "[context-compacted:";
/// Marker prefix for folded duplicate outputs.
pub const PROJECTION_REPEAT_MARKER_PREFIX: &str = "[repeated:";
/// Marker prefix for deduplicated older summaries.
pub const PROJECTION_SUMMARY_MARKER_PREFIX: &str = "[superseded-summary:";

/// True when `text` already carries a projection marker (never rewrite twice).
pub fn contains_projection_marker(text: &str) -> bool {
    text.contains(PROJECTION_CUT_MARKER_PREFIX)
        || text.starts_with(PROJECTION_REPEAT_MARKER_PREFIX)
        || text.starts_with(PROJECTION_SUMMARY_MARKER_PREFIX)
}

// ============================================================================
// Character counting
// ============================================================================

/// Number of Unicode scalar values ("characters") in `text`.
///
/// The TypeScript source uses `String.length` (UTF-16 code units). Rust has no
/// UTF-16 view, so the migration counts code points via [`str::chars`]. The two
/// agree on the Basic Multilingual Plane and diverge only for astral (non-BMP)
/// characters, which the TypeScript counts as two surrogate code units.
pub(crate) fn char_len(text: &str) -> usize {
    text.chars().count()
}

// ============================================================================
// Content sizing
// ============================================================================

const CHARS_PER_TOKEN: usize = 4;
const ESTIMATED_IMAGE_CHARS: usize = 4800;

fn safe_json_stringify<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "[unserializable]".to_owned())
}

/// Join the text blocks of a block slice with `\n`; image blocks are skipped.
fn join_text_blocks(blocks: &[UserContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|block| match block {
            UserContentBlock::Text(text) => Some(text.text.as_str()),
            UserContentBlock::Image(_) => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Estimated character size of a single message's content.
pub(crate) fn message_content_chars(message: &Message) -> usize {
    match message {
        Message::User(user) => user_content_chars(&user.content),
        Message::ToolResult(result) => result
            .content
            .iter()
            .map(|block| match block {
                UserContentBlock::Text(text) => char_len(&text.text),
                UserContentBlock::Image(_) => ESTIMATED_IMAGE_CHARS,
            })
            .sum(),
        Message::Assistant(assistant) => {
            let mut chars = 0;
            for block in &assistant.content {
                match block {
                    AssistantContent::Text(text) => chars += char_len(&text.text),
                    AssistantContent::Thinking(thinking) => chars += char_len(&thinking.thinking),
                    AssistantContent::ToolCall(call) => {
                        chars +=
                            char_len(&call.name) + char_len(&safe_json_stringify(&call.arguments));
                    }
                }
            }
            chars
        }
    }
}

fn user_content_chars(content: &UserContent) -> usize {
    match content {
        UserContent::Text(text) => char_len(text),
        UserContent::Blocks(blocks) => blocks
            .iter()
            .map(|block| match block {
                UserContentBlock::Text(text) => char_len(&text.text),
                UserContentBlock::Image(_) => ESTIMATED_IMAGE_CHARS,
            })
            .sum(),
    }
}

/// Total estimated content characters across `messages`.
pub fn total_content_chars(messages: &[Message]) -> usize {
    messages.iter().map(message_content_chars).sum()
}

fn projection_estimation_units(message: &Message) -> usize {
    use super::core::token_estimation_units as units;
    let blocks = |blocks: &[UserContentBlock]| {
        blocks
            .iter()
            .map(|block| match block {
                UserContentBlock::Text(text) => units(&text.text),
                UserContentBlock::Image(_) => ESTIMATED_IMAGE_CHARS,
            })
            .sum::<usize>()
    };
    match message {
        Message::User(user) => match &user.content {
            UserContent::Text(text) => units(text),
            UserContent::Blocks(content) => blocks(content),
        },
        Message::ToolResult(result) => blocks(&result.content),
        Message::Assistant(assistant) => assistant
            .content
            .iter()
            .map(|block| match block {
                AssistantContent::Text(text) => units(&text.text),
                AssistantContent::Thinking(thinking) => units(&thinking.thinking),
                AssistantContent::ToolCall(call) => {
                    units(&call.name) + units(&safe_json_stringify(&call.arguments))
                }
            })
            .sum(),
    }
}

/// ASCII/non-ASCII content estimate, deliberately deterministic.
pub fn estimate_projection_tokens(messages: &[Message]) -> u64 {
    messages
        .iter()
        .map(projection_estimation_units)
        .sum::<usize>()
        .div_ceil(CHARS_PER_TOKEN) as u64
}

/// ASCII/non-ASCII estimate for a single message.
pub(crate) fn estimate_message_tokens(message: &Message) -> u64 {
    projection_estimation_units(message).div_ceil(CHARS_PER_TOKEN) as u64
}

// ============================================================================
// Content hashing
// ============================================================================

fn hash32(text: &str, seed: u32) -> u32 {
    let mut hash = seed;
    for ch in text.chars() {
        // XOR with the code point, then multiply by the FNV-1a prime. `wrapping_*`
        // mirrors TypeScript's `>>> 0` / `Math.imul` 32-bit truncation.
        hash ^= ch as u32;
        hash = hash.wrapping_mul(16777619);
    }
    hash
}

/// Deterministic 16-hex-char content hash (double 32-bit FNV-1a).
///
/// Iterates by Unicode scalar value. The TypeScript original iterates UTF-16
/// code units, so the two produce identical hashes for BMP text but differ for
/// astral (non-BMP) characters, which Rust folds into a single code point while
/// TypeScript sees a surrogate pair. Deduplication only ever compares hashes
/// produced by this same function, so the divergence is self-consistent.
pub fn short_content_hash(text: &str) -> String {
    let primary_hash = hash32(text, 0x811c9dc5);
    let secondary_hash = hash32(text, 0x811c9dc5 ^ 0x9e3779b9);
    format!("{primary_hash:08x}{secondary_hash:08x}")
}

static WHITESPACE_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\s+").expect("valid whitespace regex"));

/// Whitespace-insensitive normalization applied before hashing/deduplication.
pub(crate) fn normalize_for_hash(text: &str) -> String {
    let collapsed = WHITESPACE_REGEX.replace_all(text, " ");
    collapsed.trim().to_owned()
}

// ============================================================================
// User / toolResult content access
// ============================================================================

/// Concatenated text of a user/toolResult message: the plain string for text
/// content, or the text blocks joined by `\n`. `None` for assistant messages.
///
/// Mirrors `textOfUserOrToolContent` over the message's content.
pub(crate) fn text_of_user_or_tool_message(message: &Message) -> Option<String> {
    match message {
        Message::User(user) => Some(match &user.content {
            UserContent::Text(text) => text.clone(),
            UserContent::Blocks(blocks) => join_text_blocks(blocks),
        }),
        Message::ToolResult(result) => Some(join_text_blocks(&result.content)),
        Message::Assistant(_) => None,
    }
}

/// True when a user/toolResult message carries at least one image block.
pub(crate) fn content_has_image(message: &Message) -> bool {
    match message {
        Message::User(user) => match &user.content {
            UserContent::Text(_) => false,
            UserContent::Blocks(blocks) => blocks
                .iter()
                .any(|b| matches!(b, UserContentBlock::Image(_))),
        },
        Message::ToolResult(result) => result
            .content
            .iter()
            .any(|b| matches!(b, UserContentBlock::Image(_))),
        Message::Assistant(_) => false,
    }
}

// ============================================================================
// In-place text-block rewriting
// ============================================================================

/// Rewrite the text blocks of at least `min_chars` characters in `message`.
///
/// Non-text blocks and blocks the callback declines (`None`) are kept as-is.
/// Returns `Some((new_message, cuts))` when at least one block was rewritten and
/// `None` when nothing changed. The content shape is preserved by construction:
/// plain-text content stays plain text, block content stays blocks (with each
/// rewritten block keeping its `text_signature`).
pub(crate) fn replace_text_blocks_in_message(
    message: &Message,
    min_chars: usize,
    rewrite: impl Fn(&str) -> Option<String>,
) -> Option<(Message, usize)> {
    match message {
        Message::User(user) => match &user.content {
            UserContent::Text(text) => {
                if char_len(text) < min_chars {
                    return None;
                }
                rewrite(text).map(|new_text| {
                    (
                        Message::User(UserMessage {
                            content: UserContent::Text(new_text),
                            timestamp: user.timestamp,
                        }),
                        1usize,
                    )
                })
            }
            UserContent::Blocks(blocks) => {
                let (new_blocks, cuts) = rewrite_blocks(blocks, min_chars, &rewrite);
                if cuts == 0 {
                    return None;
                }
                Some((
                    Message::User(UserMessage {
                        content: UserContent::Blocks(new_blocks),
                        timestamp: user.timestamp,
                    }),
                    cuts,
                ))
            }
        },
        Message::ToolResult(result) => {
            let (new_blocks, cuts) = rewrite_blocks(&result.content, min_chars, &rewrite);
            if cuts == 0 {
                return None;
            }
            let mut new_result = (**result).clone();
            new_result.content = new_blocks;
            Some((Message::ToolResult(Box::new(new_result)), cuts))
        }
        Message::Assistant(_) => None,
    }
}

fn rewrite_blocks(
    blocks: &[UserContentBlock],
    min_chars: usize,
    rewrite: &impl Fn(&str) -> Option<String>,
) -> (Vec<UserContentBlock>, usize) {
    let mut cuts = 0;
    let new_blocks = blocks
        .iter()
        .map(|block| match block {
            UserContentBlock::Text(text) if char_len(&text.text) >= min_chars => {
                match rewrite(&text.text) {
                    Some(new_text) => {
                        cuts += 1;
                        UserContentBlock::Text(TextContent {
                            text: new_text,
                            text_signature: text.text_signature.clone(),
                        })
                    }
                    None => block.clone(),
                }
            }
            _ => block.clone(),
        })
        .collect();
    (new_blocks, cuts)
}

/// Rebuild a user/toolResult message whose content is a single marker text block,
/// preserving the original content shape: plain-text content becomes the marker
/// string, block content becomes a single text block. Mirrors the marker branch
/// of `withUserOrToolContent` used by the fold/dedup rules.
pub(crate) fn with_marker_content(message: &Message, marker: String) -> Message {
    match message {
        Message::User(user) => {
            let content = match &user.content {
                UserContent::Text(_) => UserContent::Text(marker),
                UserContent::Blocks(_) => {
                    UserContent::Blocks(vec![UserContentBlock::Text(TextContent {
                        text: marker,
                        text_signature: None,
                    })])
                }
            };
            Message::User(UserMessage {
                content,
                timestamp: user.timestamp,
            })
        }
        Message::ToolResult(result) => {
            let mut new_result = (**result).clone();
            new_result.content = vec![UserContentBlock::Text(TextContent {
                text: marker,
                text_signature: None,
            })];
            Message::ToolResult(Box::new(new_result))
        }
        Message::Assistant(_) => message.clone(),
    }
}

// ============================================================================
// Message classification
// ============================================================================

/// `bashExecutionToText` conversions start with this shape ("Ran `cmd`").
static BASH_EXECUTION_TEXT_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^Ran `[^`\n]*`\n").expect("valid bash execution regex"));

/// Literal prefixes used by compaction/branch summary user messages.
const COMPACTION_SUMMARY_TEXT_START: &str =
    "The conversation history before this point was compacted into the following summary:";
const BRANCH_SUMMARY_TEXT_START: &str =
    "The following is a summary of a branch that this conversation came back from:";

pub(crate) fn is_bash_execution_user_message(message: &Message) -> bool {
    match message {
        Message::User(user) => {
            let text = match &user.content {
                UserContent::Text(text) => text.clone(),
                UserContent::Blocks(blocks) => join_text_blocks(blocks),
            };
            BASH_EXECUTION_TEXT_REGEX.is_match(&text)
        }
        _ => false,
    }
}

pub(crate) fn is_summary_user_message(message: &Message) -> bool {
    match text_of_user_or_tool_message(message) {
        Some(text) => {
            text.starts_with(COMPACTION_SUMMARY_TEXT_START)
                || text.starts_with(BRANCH_SUMMARY_TEXT_START)
        }
        None => false,
    }
}

/// Tool results and converted bash executions: the outputs rules a/b operate on.
pub(crate) fn is_tool_output_message(message: &Message) -> bool {
    matches!(message, Message::ToolResult(_)) || is_bash_execution_user_message(message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use step_harness_providers::types::{Api, ProviderId, ToolResultMessage};

    fn user_text(text: &str) -> Message {
        Message::User(UserMessage {
            content: UserContent::Text(text.to_owned()),
            timestamp: 0,
        })
    }

    fn tool_result(text: &str) -> Message {
        Message::ToolResult(Box::new(ToolResultMessage {
            tool_call_id: "call_1".to_owned(),
            tool_name: "bash".to_owned(),
            content: vec![UserContentBlock::Text(TextContent {
                text: text.to_owned(),
                text_signature: None,
            })],
            details: None,
            usage: None,
            added_tool_names: None,
            is_error: false,
            timestamp: 0,
        }))
    }

    fn assistant_thinking(thinking: &str) -> Message {
        Message::Assistant(Box::new(step_harness_providers::types::AssistantMessage {
            content: vec![AssistantContent::Thinking(
                step_harness_providers::types::ThinkingContent {
                    thinking: thinking.to_owned(),
                    thinking_signature: None,
                    redacted: false,
                },
            )],
            api: Api::AnthropicMessages,
            provider: ProviderId::new("test"),
            model: "m".to_owned(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: Default::default(),
            stop_reason: step_harness_providers::types::StopReason::Stop,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            timestamp: 0,
        }))
    }

    #[test]
    fn hash_is_stable_and_sixteen_hex_chars() {
        let hash = short_content_hash("hello world");
        assert_eq!(hash.len(), 16);
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
        // deterministic
        assert_eq!(hash, short_content_hash("hello world"));
        // different input, different hash
        assert_ne!(hash, short_content_hash("hello worlt"));
    }

    #[test]
    fn hash_distinguishes_seed_inputs() {
        // The two seeds should produce two different halves for non-trivial text.
        let hash = short_content_hash("some content to hash");
        let (primary, secondary) = hash.split_at(8);
        assert_ne!(primary, secondary);
    }

    #[test]
    fn normalize_collapses_whitespace_and_trims() {
        assert_eq!(normalize_for_hash("  a\t b \n c "), "a b c");
        // whitespace-only differences normalize identically
        assert_eq!(normalize_for_hash("a b"), normalize_for_hash("a  b"));
    }

    #[test]
    fn content_chars_use_chars_per_token() {
        let messages = vec![user_text(&"x".repeat(16))];
        assert_eq!(total_content_chars(&messages), 16);
        assert_eq!(estimate_projection_tokens(&messages), 4);
        // div_ceil rounds up
        let odd = vec![user_text(&"x".repeat(17))];
        assert_eq!(estimate_projection_tokens(&odd), 5);
    }

    #[test]
    fn tool_result_text_and_image_detection() {
        let with_text = tool_result("output");
        assert_eq!(text_of_user_or_tool_message(&with_text).unwrap(), "output");
        assert!(!content_has_image(&with_text));

        let with_image = Message::ToolResult(Box::new(ToolResultMessage {
            tool_call_id: "c".to_owned(),
            tool_name: "t".to_owned(),
            content: vec![
                UserContentBlock::Text(TextContent {
                    text: "hi".to_owned(),
                    text_signature: None,
                }),
                UserContentBlock::Image(step_harness_providers::types::ImageContent {
                    data: String::new(),
                    mime_type: "image/png".to_owned(),
                }),
            ],
            details: None,
            usage: None,
            added_tool_names: None,
            is_error: false,
            timestamp: 0,
        }));
        assert!(content_has_image(&with_image));
        assert_eq!(text_of_user_or_tool_message(&with_image).unwrap(), "hi");
    }

    #[test]
    fn marker_detection_and_classification() {
        assert!(contains_projection_marker("[context-compacted: x]"));
        assert!(contains_projection_marker("[repeated: x]"));
        assert!(contains_projection_marker("[superseded-summary: x]"));
        assert!(!contains_projection_marker("plain text"));

        assert!(is_bash_execution_user_message(&user_text(
            "Ran `ls -la`\nfile"
        )));
        assert!(!is_bash_execution_user_message(&user_text("hello")));
        assert!(is_tool_output_message(&tool_result("x")));
        assert!(is_summary_user_message(&user_text(&format!(
            "{COMPACTION_SUMMARY_TEXT_START} more"
        ))));
        assert!(!is_summary_user_message(&user_text("nope")));
    }

    #[test]
    fn assistant_tool_call_chars_count_name_plus_arguments() {
        let call = Message::Assistant(Box::new(step_harness_providers::types::AssistantMessage {
            content: vec![AssistantContent::ToolCall(
                step_harness_providers::types::ToolCall {
                    id: "c1".to_owned(),
                    name: "bash".to_owned(), // 4 chars
                    arguments: serde_json::json!({"cmd": "ls"})
                        .as_object()
                        .unwrap()
                        .clone(),
                    thought_signature: None,
                    namespace: None,
                },
            )],
            api: Api::AnthropicMessages,
            provider: ProviderId::new("test"),
            model: "m".to_owned(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: Default::default(),
            stop_reason: step_harness_providers::types::StopReason::Stop,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            timestamp: 0,
        }));
        // name (4) + json `{"cmd":"ls"}` (12) = 16
        assert_eq!(message_content_chars(&call), 16);
        assert_eq!(
            message_content_chars(&assistant_thinking(&"z".repeat(30))),
            30
        );
    }

    #[test]
    fn replace_text_blocks_respects_min_chars_and_preserves_shape() {
        let msg = user_text(&"a".repeat(50));
        // below min -> no rewrite
        assert!(replace_text_blocks_in_message(&msg, 100, |t| Some(t.to_uppercase())).is_none());
        // above min -> rewrite
        let (new_msg, cuts) =
            replace_text_blocks_in_message(&msg, 10, |t| Some(t.to_uppercase())).unwrap();
        assert_eq!(cuts, 1);
        assert_eq!(
            text_of_user_or_tool_message(&new_msg).unwrap(),
            "A".repeat(50)
        );
        // shape preserved: still plain text
        assert!(matches!(new_msg, Message::User(u) if matches!(u.content, UserContent::Text(_))));
    }
}
