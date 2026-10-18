//! The five projection rewrite rules. Each rule only rewrites message *content*
//! in place inside the working copy -- never structure, roles, or tool-call
//! blocks -- and always skips the protected indexes computed by
//! [`compute_protection`].
//!
//! Ported from `@step-harness/agent-core`
//! `src/harness/compaction/projection-rules.ts`.

use std::collections::HashMap;
use std::sync::LazyLock;

use regex::Regex;
use step_harness_providers::types::{AssistantContent, Message, TextContent};

use super::projection_content::{
    PROJECTION_CUT_MARKER_PREFIX, PROJECTION_REPEAT_MARKER_PREFIX,
    PROJECTION_SUMMARY_MARKER_PREFIX, char_len, contains_projection_marker, content_has_image,
    is_bash_execution_user_message, is_summary_user_message, is_tool_output_message,
    normalize_for_hash, replace_text_blocks_in_message, text_of_user_or_tool_message,
    with_marker_content,
};
use super::projection_invariants::ProtectionZones;
use super::projection_options::{ProjectionByRuleStats, ProjectionKnobs};
use super::projection_salient::{
    code_salient_line_regex, cut_text_with_salient_lines, salient_line_regex,
};

/// Minimum size for a result to participate in repeated-output folding.
const REPEAT_MIN_CHARS: usize = 200;

// ============================================================================
// Rule a: large toolResult / bashExecution outputs
// ============================================================================

fn apply_large_tool_result_rule(
    projected_messages: &mut [Message],
    protection: &ProtectionZones,
    knobs: &ProjectionKnobs,
    by_rule: &mut ProjectionByRuleStats,
) -> usize {
    let mut rewrites = 0;
    for (i, message) in projected_messages.iter_mut().enumerate() {
        if protection.protected_indexes.contains(&i) {
            continue;
        }
        if !is_tool_output_message(message) {
            continue;
        }
        let text = match text_of_user_or_tool_message(message) {
            Some(text) => text,
            None => continue,
        };
        if char_len(&text) < knobs.large_min_chars || contains_projection_marker(&text) {
            continue;
        }
        if let Some((new_message, _cuts)) =
            replace_text_blocks_in_message(message, knobs.large_min_chars, |block_text| {
                cut_text_with_salient_lines(
                    block_text,
                    knobs.head_chars,
                    knobs.tail_chars,
                    knobs.max_salient_lines,
                    salient_line_regex(),
                )
                .map(|cut| cut.text)
            })
        {
            *message = new_message;
            by_rule.tool_result_cuts += 1;
            rewrites += 1;
        }
    }
    rewrites
}

// ============================================================================
// Rule b: repeated tool/bash outputs
// ============================================================================

/// Fold repeated outputs, preserving the first and most recent occurrences.
fn apply_repeated_output_rule(
    projected_messages: &mut [Message],
    protection: &ProtectionZones,
    by_rule: &mut ProjectionByRuleStats,
) -> usize {
    // Preserve insertion order with a Vec while retaining O(1) key lookup.
    let mut duplicate_indexes_by_key: HashMap<String, Vec<usize>> = HashMap::new();
    let mut group_order: Vec<String> = Vec::new();
    for (i, message) in projected_messages.iter().enumerate() {
        if !is_tool_output_message(message) {
            continue;
        }
        if content_has_image(message) {
            continue;
        }
        let text = match text_of_user_or_tool_message(message) {
            Some(text) => text,
            None => continue,
        };
        if char_len(&text) < REPEAT_MIN_CHARS || contains_projection_marker(&text) {
            continue;
        }
        let identity_key = match message {
            Message::ToolResult(result) => {
                format!(
                    "toolResult|{}|{}",
                    result.tool_name,
                    i32::from(result.is_error)
                )
            }
            _ => "user-bash|_|0".to_owned(),
        };
        let group_key = format!("{}|{}", identity_key, short_content_hash_of(&text));
        let entry = duplicate_indexes_by_key
            .entry(group_key.clone())
            .or_insert_with(|| {
                group_order.push(group_key.clone());
                Vec::new()
            });
        entry.push(i);
    }

    let mut rewrites = 0;
    for group_key in &group_order {
        let Some(duplicate_indexes) = duplicate_indexes_by_key.get(group_key) else {
            continue;
        };
        if duplicate_indexes.len() < 3 {
            // first + most recent stay full; only middles fold
            continue;
        }
        let first_index = duplicate_indexes[0];
        let last_index = duplicate_indexes[duplicate_indexes.len() - 1];
        let content_hash = group_key.rsplit('|').next().unwrap_or(group_key.as_str());
        for &duplicate_index in duplicate_indexes {
            if duplicate_index == first_index || duplicate_index == last_index {
                continue;
            }
            if protection.protected_indexes.contains(&duplicate_index) {
                continue;
            }
            let message = &projected_messages[duplicate_index];
            if !is_tool_output_message(message) {
                continue;
            }
            let original_chars = text_of_user_or_tool_message(message).map_or(0, |t| char_len(&t));
            let marker = format!(
                "{PROJECTION_REPEAT_MARKER_PREFIX} last full output at index {last_index}; first at index {first_index}; hash={content_hash}; original_chars={original_chars}]"
            );
            projected_messages[duplicate_index] = with_marker_content(message, marker);
            by_rule.dedup_folds += 1;
            rewrites += 1;
        }
    }
    rewrites
}

/// Local wrapper so the hash call reads clearly at the group-key construction site.
fn short_content_hash_of(text: &str) -> String {
    super::projection_content::short_content_hash(&normalize_for_hash(text))
}

// ============================================================================
// Rule c: historical assistant thinking
// ============================================================================

struct ThinkingStripOutcome {
    content: Vec<AssistantContent>,
    dropped_blocks: usize,
}

/// Strip all thinking blocks from an assistant message, leaving one elision
/// marker (with the total dropped chars) at the first block's position.
/// Returns `None` when the message carries no thinking.
fn strip_thinking_blocks(
    message: &step_harness_providers::types::AssistantMessage,
) -> Option<ThinkingStripOutcome> {
    let mut dropped_blocks = 0usize;
    let mut dropped_chars = 0usize;
    let mut marker_index: Option<usize> = None;
    let mut content: Vec<AssistantContent> = Vec::new();
    for block in &message.content {
        if let AssistantContent::Thinking(thinking) = block {
            if dropped_blocks == 0 {
                marker_index = Some(content.len());
                content.push(AssistantContent::Text(TextContent {
                    text: String::new(),
                    text_signature: None,
                }));
            }
            dropped_blocks += 1;
            dropped_chars += char_len(&thinking.thinking);
            continue;
        }
        content.push(block.clone());
    }
    if dropped_blocks == 0 {
        return None;
    }
    if let Some(index) = marker_index {
        content[index] = AssistantContent::Text(TextContent {
            text: format!(
                "{PROJECTION_CUT_MARKER_PREFIX} thinking elided ({dropped_chars} chars)]"
            ),
            text_signature: None,
        });
    }
    Some(ThinkingStripOutcome {
        content,
        dropped_blocks,
    })
}

/// Drop historical assistant thinking, keeping the most recent blocks.
fn apply_thinking_drop_rule(
    projected_messages: &mut [Message],
    protection: &ProtectionZones,
    knobs: &ProjectionKnobs,
    by_rule: &mut ProjectionByRuleStats,
) -> usize {
    let thinking_bearing: Vec<usize> = (0..projected_messages.len())
        .filter(|&i| match &projected_messages[i] {
            Message::Assistant(assistant) => assistant
                .content
                .iter()
                .any(|b| matches!(b, AssistantContent::Thinking(_))),
            _ => false,
        })
        .collect();
    if thinking_bearing.len() <= knobs.keep_thinking_blocks {
        return 0;
    }

    // Keep the most recent `keep_thinking_blocks` entries; strip the rest.
    let keep_from = thinking_bearing.len() - knobs.keep_thinking_blocks;
    let mut rewrites = 0;
    for (position, &index) in thinking_bearing.iter().enumerate() {
        if position >= keep_from {
            continue;
        }
        if protection.protected_indexes.contains(&index) {
            continue;
        }
        let Message::Assistant(assistant) = &projected_messages[index] else {
            continue;
        };
        if let Some(stripped) = strip_thinking_blocks(assistant) {
            let mut new_assistant = (**assistant).clone();
            new_assistant.content = stripped.content;
            projected_messages[index] = Message::Assistant(Box::new(new_assistant));
            by_rule.thinking_drops += stripped.dropped_blocks;
            rewrites += 1;
        }
    }
    rewrites
}

// ============================================================================
// Rule d: repeated branch/compaction summaries
// ============================================================================

/// Deduplicate repeated summaries, keeping only the newest copy.
fn apply_summary_dedup_rule(
    projected_messages: &mut [Message],
    protection: &ProtectionZones,
    by_rule: &mut ProjectionByRuleStats,
) -> usize {
    let mut duplicate_indexes_by_hash: HashMap<String, Vec<usize>> = HashMap::new();
    let mut group_order: Vec<String> = Vec::new();
    for (i, message) in projected_messages.iter().enumerate() {
        if !is_summary_user_message(message) {
            continue;
        }
        let Some(text) = text_of_user_or_tool_message(message) else {
            continue;
        };
        let content_hash = short_content_hash_of(&text);
        let entry = duplicate_indexes_by_hash
            .entry(content_hash.clone())
            .or_insert_with(|| {
                group_order.push(content_hash.clone());
                Vec::new()
            });
        entry.push(i);
    }

    let mut rewrites = 0;
    for content_hash in &group_order {
        let Some(duplicate_indexes) = duplicate_indexes_by_hash.get(content_hash) else {
            continue;
        };
        if duplicate_indexes.len() < 2 {
            continue;
        }
        let newest_index = duplicate_indexes[duplicate_indexes.len() - 1];
        for &duplicate_index in duplicate_indexes {
            if duplicate_index == newest_index {
                continue;
            }
            if protection.protected_indexes.contains(&duplicate_index) {
                continue;
            }
            let message = &projected_messages[duplicate_index];
            if !is_summary_user_message(message) {
                continue;
            }
            let marker = format!(
                "{PROJECTION_SUMMARY_MARKER_PREFIX} identical summary retained at index {newest_index}; hash={content_hash}]"
            );
            projected_messages[duplicate_index] = with_marker_content(message, marker);
            by_rule.summary_dedups += 1;
            rewrites += 1;
        }
    }
    rewrites
}

// ============================================================================
// Rule e: large code / patch / JSON payloads
// ============================================================================

const FENCED_CODE_BLOCK_MARKER: &str = "```";
static DIFF_HEADER_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^diff --git ").expect("valid diff header regex"));
static DIFF_HUNK_HEADER_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^@@ -\d").expect("valid diff hunk header regex"));

fn looks_like_code_or_data(text: &str) -> bool {
    if text.contains(FENCED_CODE_BLOCK_MARKER) {
        return true;
    }
    if DIFF_HEADER_REGEX.is_match(text) || DIFF_HUNK_HEADER_REGEX.is_match(text) {
        return true;
    }
    matches!(text.trim_start().chars().next(), Some('{') | Some('['))
}

/// Line-boundary truncation of large code / patch / JSON payloads.
fn apply_large_code_rule(
    projected_messages: &mut [Message],
    protection: &ProtectionZones,
    knobs: &ProjectionKnobs,
    by_rule: &mut ProjectionByRuleStats,
) -> usize {
    let mut rewrites = 0;
    for (i, message) in projected_messages.iter_mut().enumerate() {
        if protection.protected_indexes.contains(&i) {
            continue;
        }
        if !matches!(message, Message::User(_) | Message::ToolResult(_)) {
            continue;
        }
        if is_bash_execution_user_message(message) || is_summary_user_message(message) {
            continue;
        }
        let text = match text_of_user_or_tool_message(message) {
            Some(text) => text,
            None => continue,
        };
        if char_len(&text) < knobs.large_min_chars || contains_projection_marker(&text) {
            continue;
        }
        if !looks_like_code_or_data(&text) {
            continue;
        }
        if let Some((new_message, _cuts)) =
            replace_text_blocks_in_message(message, knobs.large_min_chars, |block_text| {
                cut_text_with_salient_lines(
                    block_text,
                    knobs.head_chars,
                    knobs.tail_chars,
                    knobs.max_salient_lines,
                    code_salient_line_regex(),
                )
                .map(|cut| cut.text)
            })
        {
            *message = new_message;
            by_rule.code_cuts += 1;
            rewrites += 1;
        }
    }
    rewrites
}

// ============================================================================
// Orchestration
// ============================================================================

/// Apply all five rules to the working copy; returns the number of rewritten messages.
///
/// Order matches the TypeScript source: summary dedup (d) -> repeated-output
/// fold (b) -> large tool result (a) -> large code (e) -> thinking drop (c).
pub fn apply_projection_rules(
    projected_messages: &mut [Message],
    protection: &ProtectionZones,
    knobs: &ProjectionKnobs,
    by_rule: &mut ProjectionByRuleStats,
) -> usize {
    let mut rewrites = 0;
    rewrites += apply_summary_dedup_rule(projected_messages, protection, by_rule);
    rewrites += apply_repeated_output_rule(projected_messages, protection, by_rule);
    rewrites += apply_large_tool_result_rule(projected_messages, protection, knobs, by_rule);
    rewrites += apply_large_code_rule(projected_messages, protection, knobs, by_rule);
    rewrites += apply_thinking_drop_rule(projected_messages, protection, knobs, by_rule);
    rewrites
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use step_harness_providers::types::{
        Api, AssistantMessage, ProviderId, StopReason, ThinkingContent, UserContent,
        UserContentBlock, UserMessage,
    };

    fn user_text(text: &str) -> Message {
        Message::User(UserMessage {
            content: UserContent::Text(text.to_owned()),
            timestamp: 0,
        })
    }

    fn tool_result(text: &str) -> Message {
        Message::ToolResult(Box::new(step_harness_providers::types::ToolResultMessage {
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
        Message::Assistant(Box::new(AssistantMessage {
            content: vec![AssistantContent::Thinking(ThinkingContent {
                thinking: thinking.to_owned(),
                thinking_signature: None,
                redacted: false,
            })],
            api: Api::AnthropicMessages,
            provider: ProviderId::new("test"),
            model: "m".to_owned(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: Default::default(),
            stop_reason: StopReason::Stop,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            timestamp: 0,
        }))
    }

    fn permissive_protection() -> ProtectionZones {
        ProtectionZones {
            protected_indexes: HashSet::new(),
            last_user_index: None,
            active_group_start: usize::MAX,
            tail_start: usize::MAX,
        }
    }

    #[test]
    fn large_tool_result_is_cut() {
        // Many lines so head+tail leaves an omittable middle (a single giant line
        // would be fully covered by head+tail and correctly not cut).
        let mut big = String::new();
        for i in 0..300 {
            big.push_str(&format!("filler line number {i}\n"));
        }
        big.push_str("ERROR: something failed");
        let mut messages = vec![tool_result(&big)];
        let mut by_rule = create_stats();
        let knobs = resolve_default();
        let rewrites = apply_large_tool_result_rule(
            &mut messages,
            &permissive_protection(),
            &knobs,
            &mut by_rule,
        );
        assert_eq!(rewrites, 1);
        assert_eq!(by_rule.tool_result_cuts, 1);
        let text = text_of_user_or_tool_message(&messages[0]).unwrap();
        assert!(text.contains(PROJECTION_CUT_MARKER_PREFIX));
        assert!(char_len(&text) < char_len(&big));
    }

    #[test]
    fn repeated_output_folds_only_middles() {
        let body = "x".repeat(400);
        // Four identical outputs -> fold the two middle ones, keep first and last.
        let mut messages = vec![
            tool_result(&body),
            tool_result(&body),
            tool_result(&body),
            tool_result(&body),
        ];
        let mut by_rule = create_stats();
        let rewrites =
            apply_repeated_output_rule(&mut messages, &permissive_protection(), &mut by_rule);
        assert_eq!(rewrites, 2);
        assert_eq!(by_rule.dedup_folds, 2);
        assert!(!contains_projection_marker(
            &text_of_user_or_tool_message(&messages[0]).unwrap()
        ));
        assert!(contains_projection_marker(
            &text_of_user_or_tool_message(&messages[1]).unwrap()
        ));
        assert!(contains_projection_marker(
            &text_of_user_or_tool_message(&messages[2]).unwrap()
        ));
        assert!(!contains_projection_marker(
            &text_of_user_or_tool_message(&messages[3]).unwrap()
        ));
    }

    #[test]
    fn thinking_drop_keeps_most_recent() {
        let mut messages = vec![
            assistant_thinking(&"a".repeat(100)),
            assistant_thinking(&"b".repeat(100)),
            assistant_thinking(&"c".repeat(100)),
        ];
        let mut by_rule = create_stats();
        let mut knobs = resolve_default();
        knobs.keep_thinking_blocks = 1;
        let rewrites = apply_thinking_drop_rule(
            &mut messages,
            &permissive_protection(),
            &knobs,
            &mut by_rule,
        );
        assert_eq!(rewrites, 2);
        assert_eq!(by_rule.thinking_drops, 2);
        // The newest (index 2) keeps its thinking.
        if let Message::Assistant(a) = &messages[2] {
            assert!(
                a.content
                    .iter()
                    .any(|b| matches!(b, AssistantContent::Thinking(_)))
            );
        } else {
            panic!("expected assistant");
        }
    }

    #[test]
    fn summary_dedup_keeps_newest() {
        let summary = format!(
            "The conversation history before this point was compacted into the following summary:{}",
            "y".repeat(300)
        );
        let mut messages = vec![
            user_text(&summary),
            user_text(&summary),
            user_text(&summary),
        ];
        let mut by_rule = create_stats();
        let rewrites =
            apply_summary_dedup_rule(&mut messages, &permissive_protection(), &mut by_rule);
        assert_eq!(rewrites, 2);
        assert_eq!(by_rule.summary_dedups, 2);
        assert!(contains_projection_marker(
            &text_of_user_or_tool_message(&messages[0]).unwrap()
        ));
        assert!(!contains_projection_marker(
            &text_of_user_or_tool_message(&messages[2]).unwrap()
        ));
    }

    #[test]
    fn large_code_json_is_cut() {
        // Multi-line JSON payload starting with `{`.
        let mut json = String::from("{\n");
        for i in 0..400 {
            json.push_str(&format!("  \"key{i}\": \"value{i}\",\n"));
        }
        json.push('}');
        let mut messages = vec![user_text(&json)];
        let mut by_rule = create_stats();
        let knobs = resolve_default();
        let rewrites = apply_large_code_rule(
            &mut messages,
            &permissive_protection(),
            &knobs,
            &mut by_rule,
        );
        assert_eq!(rewrites, 1);
        assert_eq!(by_rule.code_cuts, 1);
    }

    #[test]
    fn plain_large_text_is_not_code_cut() {
        let prose = "the quick brown fox ".repeat(500);
        let mut messages = vec![user_text(&prose)];
        let mut by_rule = create_stats();
        let knobs = resolve_default();
        let rewrites = apply_large_code_rule(
            &mut messages,
            &permissive_protection(),
            &knobs,
            &mut by_rule,
        );
        assert_eq!(rewrites, 0);
    }

    #[test]
    fn orchestration_order_accumulates_all_rules() {
        // One message per rule family to confirm every rule fires.
        let summary = format!(
            "The conversation history before this point was compacted into the following summary:{}",
            "s".repeat(300)
        );
        let mut big_output = String::new();
        for i in 0..300 {
            big_output.push_str(&format!("filler line number {i}\n"));
        }
        big_output.push_str("ERROR: something failed");
        let mut json = String::from("{\n");
        for i in 0..400 {
            json.push_str(&format!("  \"key{i}\": \"value{i}\",\n"));
        }
        json.push('}');
        let mut messages = vec![
            user_text(&summary),
            user_text(&summary),
            assistant_thinking(&"t".repeat(100)),
            assistant_thinking(&"t".repeat(100)),
            assistant_thinking(&"t".repeat(100)),
            tool_result(&big_output),
            user_text(&json),
        ];
        let mut by_rule = create_stats();
        let knobs = resolve_default();
        let rewrites = apply_projection_rules(
            &mut messages,
            &permissive_protection(),
            &knobs,
            &mut by_rule,
        );
        assert!(rewrites >= 4);
        assert!(by_rule.summary_dedups >= 1);
        assert!(by_rule.thinking_drops >= 1);
        assert!(by_rule.tool_result_cuts >= 1);
        assert!(by_rule.code_cuts >= 1);
    }

    // -- small helpers --
    fn create_stats() -> ProjectionByRuleStats {
        ProjectionByRuleStats::default()
    }

    fn resolve_default() -> ProjectionKnobs {
        super::super::projection_options::resolve_projection_knobs(None)
    }
}
