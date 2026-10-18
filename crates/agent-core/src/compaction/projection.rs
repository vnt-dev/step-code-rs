//! Lightweight request-time context projection: composition entry point and
//! public API surface.
//!
//! [`project_context_for_request`] deterministically rewrites the LLM-facing
//! message array right before a model request to reclaim context window from
//! redundant content. It is a pure function: no I/O, no model calls, and no
//! session mutation. The session transcript and compaction entries are never
//! touched -- only the projected copy handed to the provider changes.
//!
//! Structural guarantee: projection only rewrites message *content* in place.
//! It never removes, inserts, or reorders messages, never changes roles, and
//! never touches tool-call blocks, so assistant `toolCall` / `toolResult`
//! pairing is preserved by construction and re-verified afterwards.
//!
//! Invariants (any violation returns the original messages unchanged):
//!   1. The current user turn (last user message) is never modified.
//!   2. The active tool-call group (last assistant message and everything
//!      after it) is never modified.
//!   3. The most recent `keep_recent_tokens` worth of tail messages are never
//!      modified.
//!
//! Module layout: this file is the composition entry point and public API
//! surface. The mechanics live in [`super::projection_options`] (option/stats
//! types and defaults), [`super::projection_rules`] (the five rewrite rules),
//! [`super::projection_invariants`] (protected zones + verification),
//! [`super::projection_salient`] (head+tail+salient-line cutting), and
//! [`super::projection_content`] (content sizing, hashing, classification).
//!
//! Ported from `@step-harness/agent-core`
//! `src/harness/compaction/projection.ts`.
//!
//! # Notable differences from the TypeScript source
//!
//! - Token counts are `u64`, character counts are `usize`, ratios are `f64`, and
//!   message indexes are `usize` (the `-1` sentinel becomes [`Option`]).
//! - The `try/catch` around the rewrite passes is unnecessary here: the passes
//!   are pure and panic-free, so the invariant verification is the sole
//!   fail-safe. Passthrough still returns a fresh clone of the input.

use step_harness_providers::types::Message;

use super::budget::{Rounding, ratio_tokens};
use super::projection_content::total_content_chars;
use super::projection_invariants::{ProtectionZones, compute_protection};
use super::projection_options::{
    ProjectionKnobs, create_by_rule_stats, resolve_projection_knobs, to_aggressive_knobs,
};
use super::projection_rules::apply_projection_rules;

// Re-export the public API surface (mirrors the TypeScript `projection.ts`).
pub use super::projection_content::{
    PROJECTION_CUT_MARKER_PREFIX, PROJECTION_REPEAT_MARKER_PREFIX,
    PROJECTION_SUMMARY_MARKER_PREFIX, estimate_projection_tokens, short_content_hash,
};
pub use super::projection_invariants::verify_projection_invariants;
pub use super::projection_options::{
    ContextProjectionMode, DEFAULT_CAP_RATIO, DEFAULT_KEEP_RECENT_TOKENS,
    DEFAULT_SOFT_THRESHOLD_RATIO, ProjectionByRuleStats, ProjectionOptions,
    ProjectionSkippedReason, ProjectionStats,
};
pub use super::projection_salient::{CutResult, cut_text_with_salient_lines};

/// Result of [`project_context_for_request`].
#[derive(Clone, Debug)]
pub struct ProjectionResult {
    pub messages: Vec<Message>,
    pub stats: ProjectionStats,
}

// ============================================================================
// Stats accumulation and trigger checks
// ============================================================================

/// The base stats shared by every outcome; per-branch overrides mutate it.
fn base_stats(
    original_tokens: u64,
    original_chars: usize,
    by_rule: ProjectionByRuleStats,
) -> ProjectionStats {
    ProjectionStats {
        applied: false,
        original_tokens,
        projected_tokens: original_tokens,
        original_chars,
        projected_chars: original_chars,
        by_rule,
        invariants_passed: true,
        invariant_violation: None,
        skipped_reason: None,
        aggressive_pass: false,
    }
}

fn find_skip_reason(
    messages_len: usize,
    context_window: u64,
    original_tokens: u64,
    soft_threshold_tokens: u64,
) -> Option<ProjectionSkippedReason> {
    if messages_len == 0 {
        return Some(ProjectionSkippedReason::EmptyMessages);
    }
    // A `u64` window is always finite; the TypeScript `<= 0` guard becomes `== 0`.
    if context_window == 0 {
        return Some(ProjectionSkippedReason::NoContextWindow);
    }
    if original_tokens < soft_threshold_tokens {
        return Some(ProjectionSkippedReason::BelowSoftThreshold);
    }
    None
}

// ============================================================================
// Rewrite passes
// ============================================================================

/// Budgets resolved once per run for the rewrite passes.
struct ProjectionRunBudgets {
    original_tokens: u64,
    cap_tokens: u64,
    keep_recent_tokens: u64,
    knobs: ProjectionKnobs,
}

struct ProjectionPassOutcome {
    projected_messages: Vec<Message>,
    protection: ProtectionZones,
    rewrites: usize,
    projected_tokens: u64,
    aggressive_pass: bool,
}

/// Estimated post-projection tokens: the usage-derived original minus the
/// estimated savings (chars/4) observed on the working copy. Savings are
/// subtracted from the possibly usage-derived original token count.
fn estimate_projected_tokens(original_tokens: u64, estimated_original: u64, current: u64) -> u64 {
    let savings = estimated_original.saturating_sub(current);
    original_tokens.saturating_sub(savings)
}

/// Run the default-knob pass, then the aggressive pass while still above the cap.
fn run_projection_passes(
    messages: &[Message],
    by_rule: &mut ProjectionByRuleStats,
    budgets: &ProjectionRunBudgets,
) -> ProjectionPassOutcome {
    let protection = compute_protection(messages, budgets.keep_recent_tokens);
    let estimated_original_tokens = estimate_projection_tokens(messages);
    let mut projected_messages = messages.to_vec();

    let mut rewrites = apply_projection_rules(
        &mut projected_messages,
        &protection,
        &budgets.knobs,
        by_rule,
    );
    let mut projected_tokens = estimate_projected_tokens(
        budgets.original_tokens,
        estimated_original_tokens,
        estimate_projection_tokens(&projected_messages),
    );

    let mut aggressive_pass = false;
    if projected_tokens > budgets.cap_tokens {
        aggressive_pass = true;
        let aggressive_knobs = to_aggressive_knobs(budgets.knobs);
        rewrites += apply_projection_rules(
            &mut projected_messages,
            &protection,
            &aggressive_knobs,
            by_rule,
        );
        projected_tokens = estimate_projected_tokens(
            budgets.original_tokens,
            estimated_original_tokens,
            estimate_projection_tokens(&projected_messages),
        );
    }

    ProjectionPassOutcome {
        projected_messages,
        protection,
        rewrites,
        projected_tokens,
        aggressive_pass,
    }
}

// ============================================================================
// Entry point
// ============================================================================

/// Verify invariants over a finished pass outcome and assemble the result.
fn finalize_projection_result(
    original_messages: &[Message],
    outcome: ProjectionPassOutcome,
    base: ProjectionStats,
) -> ProjectionResult {
    if outcome.rewrites == 0 {
        let mut stats = base;
        stats.skipped_reason = Some(ProjectionSkippedReason::NoReducibleContent);
        stats.aggressive_pass = outcome.aggressive_pass;
        return ProjectionResult {
            messages: original_messages.to_vec(),
            stats,
        };
    }

    if let Some(violation) = verify_projection_invariants(
        original_messages,
        &outcome.projected_messages,
        &outcome.protection.protected_indexes,
        outcome.protection.last_user_index,
    ) {
        let mut stats = base;
        stats.invariants_passed = false;
        stats.invariant_violation = Some(violation);
        stats.aggressive_pass = outcome.aggressive_pass;
        return ProjectionResult {
            messages: original_messages.to_vec(),
            stats,
        };
    }

    let mut stats = base;
    stats.applied = true;
    stats.projected_tokens = outcome.projected_tokens;
    stats.projected_chars = total_content_chars(&outcome.projected_messages);
    stats.aggressive_pass = outcome.aggressive_pass;
    ProjectionResult {
        messages: outcome.projected_messages,
        stats,
    }
}

/// Project the LLM-facing message array for the next model request.
///
/// Pure and deterministic: same inputs produce the same output, nothing is
/// persisted, and the input array/messages are never mutated. Invalid ratios,
/// skipped projection runs, and failed invariants return the original messages
/// unchanged. Invalid ratios are reported as `ProjectionSkippedReason::InvalidOptions`;
/// use `ProjectionOptions::validate` for a field-specific error.
pub fn project_context_for_request(
    messages: &[Message],
    options: Option<&ProjectionOptions>,
) -> ProjectionResult {
    let mut by_rule = create_by_rule_stats();
    let original_chars = total_content_chars(messages);
    let context_window = options.and_then(|o| o.context_window).unwrap_or(0);
    let original_tokens = options
        .and_then(|o| o.context_tokens)
        .unwrap_or_else(|| estimate_projection_tokens(messages));

    let soft_threshold_tokens = ratio_tokens(
        context_window,
        options
            .and_then(|o| o.soft_threshold_ratio)
            .unwrap_or(DEFAULT_SOFT_THRESHOLD_RATIO),
        Rounding::Ceil,
        "soft_threshold_ratio",
    );
    let cap_tokens = ratio_tokens(
        context_window,
        options
            .and_then(|o| o.cap_ratio)
            .unwrap_or(DEFAULT_CAP_RATIO),
        Rounding::Floor,
        "cap_ratio",
    );
    let (Ok(soft_threshold_tokens), Ok(cap_tokens)) = (soft_threshold_tokens, cap_tokens) else {
        let mut stats = base_stats(original_tokens, original_chars, by_rule);
        stats.skipped_reason = Some(ProjectionSkippedReason::InvalidOptions);
        return ProjectionResult {
            messages: messages.to_vec(),
            stats,
        };
    };

    if let Some(skipped_reason) = find_skip_reason(
        messages.len(),
        context_window,
        original_tokens,
        soft_threshold_tokens,
    ) {
        let mut stats = base_stats(original_tokens, original_chars, by_rule);
        stats.skipped_reason = Some(skipped_reason);
        return ProjectionResult {
            messages: messages.to_vec(),
            stats,
        };
    }

    let budgets = ProjectionRunBudgets {
        original_tokens,
        cap_tokens,
        keep_recent_tokens: options
            .and_then(|o| o.keep_recent_tokens)
            .unwrap_or(DEFAULT_KEEP_RECENT_TOKENS),
        knobs: resolve_projection_knobs(options),
    };

    let outcome = run_projection_passes(messages, &mut by_rule, &budgets);
    let base = base_stats(original_tokens, original_chars, by_rule);
    finalize_projection_result(messages, outcome, base)
}

#[cfg(test)]
mod tests {
    use super::*;
    use step_harness_providers::types::{
        Api, AssistantContent, AssistantMessage, ProviderId, StopReason, TextContent,
        ThinkingContent, UserContent, UserContentBlock, UserMessage,
    };

    #[test]
    fn invalid_ratios_leave_context_unchanged_and_report_the_bad_option() {
        let messages = vec![user_text("unchanged")];
        for ratio in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1, 0.0, 1.01] {
            for field in ["soft_threshold_ratio", "cap_ratio"] {
                let mut options = ProjectionOptions {
                    context_window: Some(100),
                    context_tokens: Some(100),
                    ..Default::default()
                };
                match field {
                    "soft_threshold_ratio" => options.soft_threshold_ratio = Some(ratio),
                    _ => options.cap_ratio = Some(ratio),
                }
                assert_eq!(options.validate().unwrap_err().field, field);
                let result = project_context_for_request(&messages, Some(&options));
                assert_eq!(
                    result.stats.skipped_reason,
                    Some(ProjectionSkippedReason::InvalidOptions)
                );
                assert!(!result.stats.applied);
                assert_eq!(
                    serde_json::to_value(&result.messages).unwrap(),
                    serde_json::to_value(&messages).unwrap()
                );
            }
        }
    }

    #[test]
    fn fractional_trigger_rounds_up_and_fractional_cap_rounds_down() {
        let messages = vec![user_text("unchanged")];
        let mut options = ProjectionOptions {
            context_window: Some(11),
            context_tokens: Some(6),
            soft_threshold_ratio: Some(0.6),
            ..Default::default()
        };
        assert_eq!(
            project_context_for_request(&messages, Some(&options))
                .stats
                .skipped_reason,
            Some(ProjectionSkippedReason::BelowSoftThreshold)
        );
        options.context_tokens = Some(7);
        assert_ne!(
            project_context_for_request(&messages, Some(&options))
                .stats
                .skipped_reason,
            Some(ProjectionSkippedReason::BelowSoftThreshold)
        );
        options.context_tokens = Some(8);
        options.cap_ratio = Some(0.75);
        assert!(
            !project_context_for_request(&messages, Some(&options))
                .stats
                .aggressive_pass
        );
        options.context_tokens = Some(9);
        assert!(
            project_context_for_request(&messages, Some(&options))
                .stats
                .aggressive_pass
        );
    }

    #[test]
    fn trigger_distinguishes_adjacent_counts_above_f64_integer_precision() {
        let count = (1u64 << 53) + 1;
        let options = ProjectionOptions {
            context_window: Some(count),
            context_tokens: Some(count - 1),
            soft_threshold_ratio: Some(1.0),
            ..Default::default()
        };
        let result = project_context_for_request(&[user_text("unchanged")], Some(&options));
        assert_eq!(
            result.stats.skipped_reason,
            Some(ProjectionSkippedReason::BelowSoftThreshold)
        );
    }

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

    fn opts(context_window: u64) -> ProjectionOptions {
        ProjectionOptions {
            context_window: Some(context_window),
            ..Default::default()
        }
    }

    #[test]
    fn empty_messages_are_skipped() {
        let result = project_context_for_request(&[], Some(&opts(100_000)));
        assert!(!result.stats.applied);
        assert_eq!(
            result.stats.skipped_reason,
            Some(ProjectionSkippedReason::EmptyMessages)
        );
        assert!(result.messages.is_empty());
    }

    #[test]
    fn missing_context_window_is_skipped() {
        let messages = vec![user_text("hello")];
        let result = project_context_for_request(&messages, None);
        assert!(!result.stats.applied);
        assert_eq!(
            result.stats.skipped_reason,
            Some(ProjectionSkippedReason::NoContextWindow)
        );
        // passthrough returns an equal clone
        assert_eq!(result.messages.len(), 1);
    }

    #[test]
    fn below_soft_threshold_is_skipped() {
        // Small content, huge window -> far below the 0.6 soft threshold.
        let messages = vec![user_text("tiny")];
        let result = project_context_for_request(&messages, Some(&opts(10_000_000)));
        assert!(!result.stats.applied);
        assert_eq!(
            result.stats.skipped_reason,
            Some(ProjectionSkippedReason::BelowSoftThreshold)
        );
    }

    #[test]
    fn nothing_reducible_reports_skip() {
        // Above threshold but content is too small / not reducible.
        let messages = vec![user_text(&"a".repeat(20))];
        let mut options = opts(10);
        options.context_tokens = Some(1_000); // force above soft threshold (0.6*10=6)
        let result = project_context_for_request(&messages, Some(&options));
        assert!(!result.stats.applied);
        assert_eq!(
            result.stats.skipped_reason,
            Some(ProjectionSkippedReason::NoReducibleContent)
        );
    }

    #[test]
    fn reducible_content_is_projected() {
        // Three identical large tool outputs force rule b to fold the middles.
        let body = "x".repeat(400);
        let messages = vec![tool_result(&body), tool_result(&body), tool_result(&body)];
        let mut options = opts(10);
        options.context_tokens = Some(1_000);
        options.keep_recent_tokens = Some(0); // do not protect the tail
        let result = project_context_for_request(&messages, Some(&options));
        assert!(result.stats.applied);
        assert!(result.stats.invariants_passed);
        assert!(result.stats.by_rule.dedup_folds >= 1);
        assert!(result.stats.projected_chars < result.stats.original_chars);
        // first and last preserved full, middle folded
        assert!(!contains_projection_marker_text(&result.messages[0]));
        assert!(contains_projection_marker_text(&result.messages[1]));
    }

    #[test]
    fn last_user_turn_is_never_modified() {
        // The current user turn is protected even if it looks reducible.
        let summary = format!(
            "The conversation history before this point was compacted into the following summary:{}",
            "s".repeat(300)
        );
        let messages = vec![
            user_text(&summary),
            user_text(&summary),
            user_text("current turn"),
        ];
        let mut options = opts(10);
        options.context_tokens = Some(1_000);
        options.keep_recent_tokens = Some(0); // do not protect the tail
        let result = project_context_for_request(&messages, Some(&options));
        // The older summaries get deduplicated, but the last user turn stays intact.
        assert!(result.stats.applied);
        assert!(result.stats.by_rule.summary_dedups >= 1);
        match (&messages[2], &result.messages[2]) {
            (Message::User(a), Message::User(b)) => assert_eq!(a.content_text(), b.content_text()),
            _ => panic!("expected user"),
        }
    }

    fn contains_projection_marker_text(message: &Message) -> bool {
        match message {
            Message::User(user) => match &user.content {
                UserContent::Text(text) => {
                    text.starts_with("[repeated:")
                        || text.starts_with("[superseded-summary:")
                        || text.contains("[context-compacted:")
                }
                UserContent::Blocks(blocks) => blocks.iter().any(|b| match b {
                    UserContentBlock::Text(t) => {
                        t.text.starts_with("[repeated:")
                            || t.text.starts_with("[superseded-summary:")
                            || t.text.contains("[context-compacted:")
                    }
                    _ => false,
                }),
            },
            Message::ToolResult(result) => result.content.iter().any(|b| match b {
                UserContentBlock::Text(t) => {
                    t.text.starts_with("[repeated:")
                        || t.text.starts_with("[superseded-summary:")
                        || t.text.contains("[context-compacted:")
                }
                _ => false,
            }),
            Message::Assistant(_) => false,
        }
    }

    // Helper trait so the last-user-turn assertion reads cleanly.
    trait ContentText {
        fn content_text(&self) -> String;
    }
    impl ContentText for UserMessage {
        fn content_text(&self) -> String {
            match &self.content {
                UserContent::Text(text) => text.clone(),
                UserContent::Blocks(blocks) => blocks
                    .iter()
                    .filter_map(|b| match b {
                        UserContentBlock::Text(t) => Some(t.text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
            }
        }
    }

    #[test]
    fn thinking_drop_via_entry_point() {
        let messages = vec![
            assistant_thinking(&"a".repeat(100)),
            assistant_thinking(&"b".repeat(100)),
            assistant_thinking(&"c".repeat(100)),
            user_text("done"),
        ];
        let mut options = opts(10);
        options.context_tokens = Some(1_000);
        options.keep_recent_tokens = Some(0); // do not protect the tail
        let result = project_context_for_request(&messages, Some(&options));
        assert!(result.stats.applied);
        assert!(result.stats.by_rule.thinking_drops >= 1);
    }

    #[test]
    fn aggressive_pass_runs_when_above_cap() {
        // A very large multi-line output with a tiny cap forces the aggressive pass.
        let mut body = String::new();
        for i in 0..500 {
            body.push_str(&format!("filler line number {i}\n"));
        }
        body.push_str("ERROR: something failed");
        let messages = vec![tool_result(&body)];
        let mut options = opts(10);
        options.context_tokens = Some(1_000_000); // far above cap (0.75*10)
        options.keep_recent_tokens = Some(0); // do not protect the tail
        let result = project_context_for_request(&messages, Some(&options));
        assert!(result.stats.applied);
        assert!(result.stats.aggressive_pass);
    }
}
