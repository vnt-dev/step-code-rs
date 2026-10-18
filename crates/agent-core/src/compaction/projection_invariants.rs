//! Projection invariant zones and post-projection verification.
//!
//! Three zones are protected from rewriting (any violation observed afterwards
//! makes the caller fall back to the original messages):
//!   1. The current user turn (last user message).
//!   2. The active tool-call group (last assistant message and everything after it).
//!   3. The most recent `keep_recent_tokens` worth of tail messages.
//!
//! Ported from `@step-harness/agent-core`
//! `src/harness/compaction/projection-invariants.ts`.

use std::collections::HashSet;

use step_harness_providers::types::{AssistantContent, Message};

use super::projection_content::estimate_message_tokens;

/// Indexes the rules must not rewrite, plus the zone boundaries they derive from.
pub struct ProtectionZones {
    pub protected_indexes: HashSet<usize>,
    /// Index of the last user message, or `None` when there is no user message.
    ///
    /// The TypeScript source uses a `-1` sentinel; the migration uses `Option`.
    pub last_user_index: Option<usize>,
    pub active_group_start: usize,
    pub tail_start: usize,
}

/// Compute the protected zones for a message array.
pub fn compute_protection(messages: &[Message], keep_recent_tokens: u64) -> ProtectionZones {
    let mut last_user_index: Option<usize> = None;
    let mut last_assistant_index: Option<usize> = None;
    for i in (0..messages.len()).rev() {
        if last_user_index.is_none() && matches!(messages[i], Message::User(_)) {
            last_user_index = Some(i);
        }
        if last_assistant_index.is_none() && matches!(messages[i], Message::Assistant(_)) {
            last_assistant_index = Some(i);
        }
        if last_user_index.is_some() && last_assistant_index.is_some() {
            break;
        }
    }

    // Invariant 2: the last assistant message plus everything after it (its tool
    // results and any trailing steering/user content) forms the active group.
    let active_group_start = last_assistant_index.unwrap_or(messages.len());

    // Invariant 3: keep the trailing messages that fit fully within the
    // `keep_recent_tokens` budget untouched. The message straddling the budget
    // boundary sits mostly outside the window and stays projectable; the very
    // last messages are always additionally covered by invariants 1 and 2.
    let mut tail_start = 0usize;
    let mut accumulated_tokens: u64 = 0;
    for i in (0..messages.len()).rev() {
        accumulated_tokens += estimate_message_tokens(&messages[i]);
        if accumulated_tokens > keep_recent_tokens {
            tail_start = i + 1;
            break;
        }
    }

    let protected_from = active_group_start.min(tail_start);
    let mut protected_indexes = HashSet::new();
    for i in protected_from..messages.len() {
        protected_indexes.insert(i);
    }
    // Invariant 1: the current user turn is protected wherever it sits.
    if let Some(index) = last_user_index {
        protected_indexes.insert(index);
    }

    ProtectionZones {
        protected_indexes,
        last_user_index,
        active_group_start,
        tail_start,
    }
}

// ============================================================================
// Invariant verification (fail-safe)
// ============================================================================

/// `id:name` signature of an assistant message's tool calls, joined by `,`.
fn tool_call_signature(message: &step_harness_providers::types::AssistantMessage) -> String {
    message
        .content
        .iter()
        .filter_map(|block| match block {
            AssistantContent::ToolCall(call) => Some(format!("{}:{}", call.id, call.name)),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// True when two messages carry the same enum variant (role).
fn same_role(a: &Message, b: &Message) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
}

/// Deep content equality for a protected message.
///
/// The provider message types intentionally do not derive `PartialEq` (their
/// diagnostics/usage payloads are open JSON). The TypeScript source compares
/// protected messages by object reference (`before !== after`); since Rust rules
/// operate on owned clones, the migration instead deep-compares the serialized
/// form, which covers every field. This is only ever called on protected
/// indexes, where projection leaves the message byte-identical.
fn messages_deep_equal(a: &Message, b: &Message) -> bool {
    match (serde_json::to_value(a), serde_json::to_value(b)) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

/// Verify the structural invariants between the original and projected arrays.
/// Returns a violation description, or `None` when everything holds.
pub fn verify_projection_invariants(
    original: &[Message],
    projected: &[Message],
    protected_indexes: &HashSet<usize>,
    last_user_index: Option<usize>,
) -> Option<String> {
    if original.len() != projected.len() {
        return Some("message-count-changed".to_owned());
    }

    for i in 0..original.len() {
        let before = &original[i];
        let after = &projected[i];
        if !same_role(before, after) {
            return Some(format!("role-changed@{i}"));
        }
        if protected_indexes.contains(&i) && !messages_deep_equal(before, after) {
            return Some(format!("protected-message-modified@{i}"));
        }
        if let (Message::Assistant(before), Message::Assistant(after)) = (before, after)
            && tool_call_signature(before) != tool_call_signature(after)
        {
            return Some(format!("tool-calls-modified@{i}"));
        }
        if let (Message::ToolResult(before), Message::ToolResult(after)) = (before, after) {
            if before.tool_call_id != after.tool_call_id || before.tool_name != after.tool_name {
                return Some(format!("tool-result-identity-changed@{i}"));
            }
            if before.is_error != after.is_error {
                return Some(format!("tool-result-error-flag-changed@{i}"));
            }
        }
    }

    if let Some(index) = last_user_index
        && !messages_deep_equal(&original[index], &projected[index])
    {
        return Some("current-user-turn-modified".to_owned());
    }

    // toolCall/toolResult pairing: every call answered before must stay answered.
    let answered_before: HashSet<&str> = original
        .iter()
        .filter_map(|m| match m {
            Message::ToolResult(result) => Some(result.tool_call_id.as_str()),
            _ => None,
        })
        .collect();
    let answered_after: HashSet<&str> = projected
        .iter()
        .filter_map(|m| match m {
            Message::ToolResult(result) => Some(result.tool_call_id.as_str()),
            _ => None,
        })
        .collect();
    for message in projected {
        let Message::Assistant(assistant) = message else {
            continue;
        };
        for block in &assistant.content {
            if let AssistantContent::ToolCall(call) = block
                && answered_before.contains(call.id.as_str())
                && !answered_after.contains(call.id.as_str())
            {
                return Some(format!("tool-pairing-broken@{}", call.id));
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use step_harness_providers::types::{
        Api, AssistantMessage, ProviderId, StopReason, TextContent, ToolCall, UserContentBlock,
        UserMessage,
    };

    fn user(text: &str) -> Message {
        Message::User(UserMessage {
            content: step_harness_providers::types::UserContent::Text(text.to_owned()),
            timestamp: 0,
        })
    }

    fn tool_result(call_id: &str) -> Message {
        Message::ToolResult(Box::new(step_harness_providers::types::ToolResultMessage {
            tool_call_id: call_id.to_owned(),
            tool_name: "bash".to_owned(),
            content: vec![UserContentBlock::Text(TextContent {
                text: "ok".to_owned(),
                text_signature: None,
            })],
            details: None,
            usage: None,
            added_tool_names: None,
            is_error: false,
            timestamp: 0,
        }))
    }

    fn assistant_with_call(call_id: &str) -> Message {
        Message::Assistant(Box::new(AssistantMessage {
            content: vec![AssistantContent::ToolCall(ToolCall {
                id: call_id.to_owned(),
                name: "bash".to_owned(),
                arguments: Default::default(),
                thought_signature: None,
                namespace: None,
            })],
            api: Api::AnthropicMessages,
            provider: ProviderId::new("test"),
            model: "m".to_owned(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: Default::default(),
            stop_reason: StopReason::ToolUse,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            timestamp: 0,
        }))
    }

    #[test]
    fn protection_protects_tail_and_last_user() {
        let messages = vec![user("a"), user("b"), user("c")];
        // huge budget -> everything protected
        let protection = compute_protection(&messages, 1_000_000);
        assert_eq!(protection.protected_indexes.len(), 3);
        assert_eq!(protection.last_user_index, Some(2));
    }

    #[test]
    fn protection_active_group_starts_at_last_assistant() {
        let messages = vec![
            user("a"),
            assistant_with_call("c1"),
            tool_result("c1"),
            user("b"),
        ];
        let protection = compute_protection(&messages, 1_000_000);
        // last assistant is index 1, so active group covers 1..end
        assert_eq!(protection.active_group_start, 1);
        assert!(protection.protected_indexes.contains(&1));
        assert!(protection.protected_indexes.contains(&3));
        assert_eq!(protection.last_user_index, Some(3));
    }

    #[test]
    fn identical_arrays_pass_invariants() {
        let messages = vec![user("a"), assistant_with_call("c1"), tool_result("c1")];
        let protection = compute_protection(&messages, 1_000_000);
        assert!(
            verify_projection_invariants(
                &messages,
                &messages,
                &protection.protected_indexes,
                protection.last_user_index,
            )
            .is_none()
        );
    }

    #[test]
    fn role_change_is_detected() {
        let original = vec![user("a"), tool_result("c1")];
        let mut projected = original.clone();
        projected[1] = user("changed");
        let protection = compute_protection(&original, 1);
        let violation = verify_projection_invariants(
            &original,
            &projected,
            &protection.protected_indexes,
            protection.last_user_index,
        );
        assert_eq!(violation, Some("role-changed@1".to_owned()));
    }

    #[test]
    fn protected_modification_is_detected() {
        let original = vec![user("a"), user("b")];
        let mut projected = original.clone();
        // Protect everything, then modify index 0.
        let protection = compute_protection(&original, 1_000_000);
        projected[0] = user("changed");
        let violation = verify_projection_invariants(
            &original,
            &projected,
            &protection.protected_indexes,
            protection.last_user_index,
        );
        assert_eq!(violation, Some("protected-message-modified@0".to_owned()));
    }

    #[test]
    fn tool_pairing_break_is_detected() {
        let original = vec![assistant_with_call("c1"), tool_result("c1")];
        // Drop the tool result while keeping the call -> pairing broken.
        let mut projected = original.clone();
        projected.pop();
        let protection = compute_protection(&original, 1);
        let violation = verify_projection_invariants(
            &original,
            &projected,
            &protection.protected_indexes,
            protection.last_user_index,
        );
        // count changed first (2 -> 1)
        assert_eq!(violation, Some("message-count-changed".to_owned()));
    }
}
