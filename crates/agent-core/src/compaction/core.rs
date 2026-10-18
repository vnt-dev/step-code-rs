//! The compaction pipeline: settings, token estimates, cut-point selection,
//! summary generation, and the compacted-history replacement.
//!
//! Ported from `@step-harness/agent-core` `src/harness/compaction/compaction.ts`.
//!
//! Notable differences from the TypeScript source:
//! - `models.completeSimple` is replaced by [`SimpleCompleter`], passed as
//!   `&dyn SimpleCompleter` alongside the [`Model`].
//! - Token counts are `u64`; the TypeScript `-1` sentinel for
//!   `turnStartIndex` is `Option<usize>`.
//! - `combineUsage` adds the optional `cacheWrite1h`/`reasoning` counters only
//!   when at least one side reports them, matching the conditional spreads.

use std::sync::LazyLock;

use step_harness_providers::types::{
    AssistantContent, AssistantMessage, CacheRetention, Context, Message, Model,
    ModelThinkingLevel, SimpleStreamOptions, StopReason, TextContent, Usage, UsageCost,
    UserContent, UserContentBlock, UserMessage,
};
use step_harness_providers::utils::retry::{RetryCallbacks, RetryPolicy, retry_assistant_call};
use step_harness_providers::utils::text::assistant_content_text;
use step_harness_providers::utils::{now_ms, uuid::uuidv7};

use crate::compaction::SimpleCompleter;
use crate::compaction::utils::{
    FileOperations, SUMMARIZATION_SYSTEM_PROMPT, compute_file_lists, create_file_ops,
    extract_file_ops_from_message, format_file_operations, safe_json_stringify,
    serialize_conversation,
};
use crate::harness::messages::{
    CustomMessageContent, convert_to_llm, create_branch_summary_message,
    create_compaction_summary_message,
};
use crate::harness::types::{AbortSignal, CompactionError, CompactionErrorCode};
use crate::session::context::{SessionContextBuildOptions, build_session_context};
use crate::session::types::{CompactionEntry, Entry, MessageEntry};
use crate::types::{AgentMessage, thinking_level_to_reasoning};

/// Compaction thresholds and retention settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompactionSettings {
    /// Enable automatic compaction decisions.
    pub enabled: bool,
    /// Tokens reserved for summary prompt and output.
    pub reserve_tokens: u64,
    /// Approximate recent-context tokens to keep after compaction.
    pub keep_recent_tokens: u64,
}

/// Default compaction settings used by the harness.
pub const DEFAULT_COMPACTION_SETTINGS: CompactionSettings = CompactionSettings {
    enabled: true,
    // Bumped from 16384: content-rich sessions were flirting with the
    // 0.8 * 16384 = 13107 maxTokens cap and getting rejected on length-stop.
    // 24576 gives ~19660-token headroom for the summary (or up to the ceiling
    // on models that expose a larger native output cap), while still leaving
    // ~keepRecentTokens for the retained tail.
    reserve_tokens: 24576,
    keep_recent_tokens: 20000,
};

/**
 * Hard upper bound on summary output tokens, regardless of `reserveTokens`.
 * Chosen to sit under Anthropic's 32k-per-response cap while giving rich
 * long-horizon sessions enough room to emit a full 8-section handoff without
 * hitting the `stopReason:"length"` guard.
 */
pub const SUMMARY_OUTPUT_TOKENS_CEILING: u64 = 32000;

/// Pick the summary output cap. Invalid fractions return zero for compatibility;
/// use [`try_pick_summary_max_tokens`] for a typed validation error.
pub fn pick_summary_max_tokens(
    model_max_tokens: u64,
    reserve_tokens: u64,
    reserve_fraction: f64,
) -> u64 {
    try_pick_summary_max_tokens(model_max_tokens, reserve_tokens, reserve_fraction).unwrap_or(0)
}

/// Resolve an integer summary budget from a finite reserve fraction in (0, 1].
/// A positive model limit preserves the existing model-bound budget behavior;
/// an unknown model limit uses the reserve fraction rounded down.
pub fn try_pick_summary_max_tokens(
    model_max_tokens: u64,
    reserve_tokens: u64,
    reserve_fraction: f64,
) -> Result<u64, super::RatioError> {
    let reserve_budget = super::budget::ratio_tokens(
        reserve_tokens,
        reserve_fraction,
        super::budget::Rounding::Floor,
        "reserve_fraction",
    )?;
    Ok(if model_max_tokens > 0 {
        model_max_tokens.min(SUMMARY_OUTPUT_TOKENS_CEILING)
    } else {
        reserve_budget.min(SUMMARY_OUTPUT_TOKENS_CEILING)
    })
}

/// Calculate total context tokens from provider usage.
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

fn get_assistant_usage(message: &AgentMessage) -> Option<Usage> {
    let AgentMessage::Assistant(assistant) = message else {
        return None;
    };
    if matches!(
        assistant.stop_reason,
        StopReason::Aborted | StopReason::Error
    ) {
        return None;
    }
    let usage = &assistant.usage;
    if calculate_context_tokens(usage) > 0 {
        Some(usage.clone())
    } else {
        None
    }
}

/// Return usage from the last valid assistant message in session entries.
pub fn get_last_assistant_usage(entries: &[Entry]) -> Option<Usage> {
    entries.iter().rev().find_map(|entry| match entry {
        Entry::Message(message) => get_assistant_usage(&message.message),
        _ => None,
    })
}

/// Estimated context-token usage for a message list.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ContextUsageEstimate {
    /// Estimated total context tokens.
    pub tokens: u64,
    /// Tokens reported by the most recent assistant usage block.
    pub usage_tokens: u64,
    /// Estimated tokens after the most recent assistant usage block.
    pub trailing_tokens: u64,
    /// Index of the message that provided usage, or `None` when none exists.
    pub last_usage_index: Option<usize>,
}

fn get_last_assistant_usage_info(messages: &[AgentMessage]) -> Option<(Usage, usize)> {
    messages
        .iter()
        .enumerate()
        .rev()
        .find_map(|(index, message)| get_assistant_usage(message).map(|usage| (usage, index)))
}

/// Estimate context tokens for messages using provider usage when available.
pub fn estimate_context_tokens(messages: &[AgentMessage]) -> ContextUsageEstimate {
    estimate_context_tokens_with(messages, &estimate_tokens)
}

/// Use a model-specific estimator for messages not covered by provider usage.
pub fn estimate_context_tokens_with(
    messages: &[AgentMessage],
    estimator: &dyn Fn(&AgentMessage) -> u64,
) -> ContextUsageEstimate {
    let Some((usage, index)) = get_last_assistant_usage_info(messages) else {
        let estimated: u64 = messages.iter().map(estimator).fold(0, u64::saturating_add);
        return ContextUsageEstimate {
            tokens: estimated,
            usage_tokens: 0,
            trailing_tokens: estimated,
            last_usage_index: None,
        };
    };

    let usage_tokens = calculate_context_tokens(&usage);
    let trailing_tokens: u64 = messages[index + 1..]
        .iter()
        .map(estimator)
        .fold(0, u64::saturating_add);

    ContextUsageEstimate {
        tokens: usage_tokens.saturating_add(trailing_tokens),
        usage_tokens,
        trailing_tokens,
        last_usage_index: Some(index),
    }
}

/// Return whether context usage exceeds the configured compaction threshold.
pub fn should_compact(
    context_tokens: u64,
    context_window: u64,
    settings: &CompactionSettings,
) -> bool {
    if !settings.enabled {
        return false;
    }
    context_tokens > context_window.saturating_sub(settings.reserve_tokens)
}

const ESTIMATED_IMAGE_CHARS: usize = 4800;
const CHARS_PER_TOKEN: usize = 4;

/// Heuristic units: ASCII keeps the historical chars/4 estimate; non-ASCII
/// receives one estimated token per UTF-8 byte to avoid underestimating CJK.
/// This remains a fallback, not a tokenizer for any particular model.
pub(crate) fn token_estimation_units(text: &str) -> usize {
    text.chars()
        .map(|ch| {
            if ch.is_ascii() {
                1
            } else {
                ch.len_utf8() * CHARS_PER_TOKEN
            }
        })
        .sum()
}

fn user_content_chars(content: &UserContent) -> usize {
    match content {
        UserContent::Text(text) => token_estimation_units(text),
        UserContent::Blocks(blocks) => blocks_chars(blocks),
    }
}

fn blocks_chars(blocks: &[UserContentBlock]) -> usize {
    blocks
        .iter()
        .map(|block| match block {
            UserContentBlock::Text(text) => token_estimation_units(&text.text),
            UserContentBlock::Image(_) => ESTIMATED_IMAGE_CHARS,
        })
        .sum()
}

/// Estimate content tokens using an ASCII/non-ASCII fallback heuristic.
/// Callers must budget separately for system prompts, schemas and protocol overhead.
pub fn estimate_tokens(message: &AgentMessage) -> u64 {
    let chars = match message {
        AgentMessage::User(user) => user_content_chars(&user.content),
        AgentMessage::Assistant(assistant) => assistant
            .content
            .iter()
            .map(|block| match block {
                AssistantContent::Text(text) => token_estimation_units(&text.text),
                AssistantContent::Thinking(thinking) => token_estimation_units(&thinking.thinking),
                AssistantContent::ToolCall(call) => {
                    token_estimation_units(&call.name)
                        + token_estimation_units(&safe_json_stringify(&call.arguments))
                }
            })
            .sum(),
        AgentMessage::ToolResult(result) => blocks_chars(&result.content),
        AgentMessage::Custom(custom) => match &custom.content {
            CustomMessageContent::Text(text) => token_estimation_units(text),
            CustomMessageContent::Blocks(blocks) => blocks_chars(blocks),
        },
        AgentMessage::BashExecution(bash) => {
            token_estimation_units(&bash.command) + token_estimation_units(&bash.output)
        }
        AgentMessage::BranchSummary(summary) => token_estimation_units(&summary.summary),
        AgentMessage::CompactionSummary(summary) => token_estimation_units(&summary.summary),
    };
    chars.div_ceil(CHARS_PER_TOKEN) as u64
}

fn find_valid_cut_points(entries: &[Entry], start_index: usize, end_index: usize) -> Vec<usize> {
    let end_index = end_index.min(entries.len());
    let start_index = start_index.min(end_index);
    let mut cut_points = Vec::new();
    for (index, entry) in entries.iter().enumerate().take(end_index).skip(start_index) {
        match entry {
            Entry::Message(message) => {
                if !matches!(message.message, AgentMessage::ToolResult(_)) {
                    cut_points.push(index);
                }
            }
            Entry::BranchSummary(_) => cut_points.push(index),
            _ => {}
        }
    }
    cut_points
}

/// Find the user-visible message that starts the turn containing an entry.
pub fn find_turn_start_index(
    entries: &[Entry],
    entry_index: usize,
    start_index: usize,
) -> Option<usize> {
    if entries.is_empty() || start_index >= entries.len() {
        return None;
    }
    let mut index = entry_index.min(entries.len().saturating_sub(1)) as isize;
    let start = start_index as isize;
    while index >= start {
        let current = index as usize;
        match &entries[current] {
            Entry::BranchSummary(_) => return Some(current),
            Entry::Message(message) => {
                if matches!(
                    message.message,
                    AgentMessage::User(_) | AgentMessage::BashExecution(_)
                ) {
                    return Some(current);
                }
            }
            _ => {}
        }
        index -= 1;
    }
    None
}

/// Cut point selected for compaction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CutPointResult {
    /// Index of the first entry retained after compaction.
    pub first_kept_entry_index: usize,
    /// Index of the turn-start entry when the cut splits a turn, otherwise `None`.
    pub turn_start_index: Option<usize>,
    /// Whether the selected cut point splits an in-progress turn.
    pub is_split_turn: bool,
}

/// Find the compaction cut point that keeps approximately the requested recent-token budget.
pub fn find_cut_point(
    entries: &[Entry],
    start_index: usize,
    end_index: usize,
    keep_recent_tokens: u64,
) -> CutPointResult {
    let end_index = end_index.min(entries.len());
    let start_index = start_index.min(end_index);
    let cut_points = find_valid_cut_points(entries, start_index, end_index);

    if cut_points.is_empty() {
        return CutPointResult {
            first_kept_entry_index: start_index,
            turn_start_index: None,
            is_split_turn: false,
        };
    }
    let mut accumulated_tokens: u64 = 0;
    let mut cut_index = cut_points[0];

    let mut index = end_index as isize - 1;
    while index >= start_index as isize {
        let current = index as usize;
        if let Entry::Message(message) = &entries[current] {
            accumulated_tokens =
                accumulated_tokens.saturating_add(estimate_tokens(&message.message));
            if accumulated_tokens >= keep_recent_tokens {
                for cut_point in &cut_points {
                    if *cut_point >= current {
                        cut_index = *cut_point;
                        break;
                    }
                }
                break;
            }
        }
        index -= 1;
    }
    while cut_index > start_index {
        let previous = &entries[cut_index - 1];
        if matches!(previous, Entry::Compaction(_) | Entry::Message(_)) {
            break;
        }
        cut_index -= 1;
    }
    let cut_entry = &entries[cut_index];
    let is_user_message = matches!(
        cut_entry,
        Entry::Message(message) if matches!(message.message, AgentMessage::User(_))
    );
    let turn_start_index = if is_user_message {
        None
    } else {
        find_turn_start_index(entries, cut_index, start_index)
    };

    CutPointResult {
        first_kept_entry_index: cut_index,
        turn_start_index,
        is_split_turn: !is_user_message && turn_start_index.is_some(),
    }
}

/// Shared 8-section handoff format used by every compaction summary prompt.
pub const SUMMARY_FORMAT: &str = r#"## User Goal
[The user's active objective(s) and acceptance criteria, preserving the user's own wording where it matters. List multiple goals as separate items. Do not present completed or abandoned goals as active.]

## Current State
### Done
- [Completed item — with its concrete result: what changed, where, and the evidence]

### In Progress
- [Started but unfinished item — with exactly where it stands and what remains]

### Blocked
- [Blocked item — with the precise blocker; or "(none)"]

## Files & Artifacts
- [`path/to/file` — created/modified/deleted/reverted; key functions/classes touched and why this file matters to the goal]

## Verification
- [Command or test that was run → PASS/FAIL/BLOCKED, exit code, and the most diagnostic output or error lines verbatim; or "(none run)"]

## Decisions & Constraints
- [User preference / technical decision / environment constraint — with brief rationale. Mark user-stated requirements vs your own inference (e.g. "(inferred)").]

## Failed Approaches
- [Approach that failed or was ruled out → why it failed (with the exact error where available) and what would have to change before retrying; or "(none)"]

## Next Actions
1. [Concrete next step: the action, the target file/command, and how to tell it is done]

## References
- [Issue/PR links, log or artifact paths, or other pointers that help resume the work; or "(none)"]"#;

/// Fidelity rules appended to every compaction summary prompt.
pub const SUMMARY_DETAIL_RULES: &str = r#"Detail requirements:
1. Preserve high-value strings EXACTLY: file paths, function/class names, commands, flags, exit codes, and error message lines must appear verbatim, never paraphrased.
2. Prefer results over process: record outcomes and current state ("test X fails with Y"), not a play-by-play of the steps taken.
3. Keep negative information: failures, dead ends, and disproven hypotheses are as important as successes — they stop the next model from repeating them.
4. Separate facts from inference: if something was not directly observed in the conversation, label it as inferred or unverified.
5. No empty statements: "fixed the bug" or "made progress" is useless — every item must say what changed, where, and what evidence supports it.
6. If you must shorten, first cut repetition, rhetoric, and background that is closed or superseded; cut the active User Goal, Verification results, and Next Actions last.

Be thorough. A complete handoff matters more than brevity, but every line must carry information the next model can act on."#;

pub static SUMMARIZATION_PROMPT: LazyLock<String> = LazyLock::new(|| {
    format!(
        "The messages above are a conversation to summarize. Produce a structured handoff summary that the next model instance will use to continue the work. The summarized messages will be discarded: your summary replaces them entirely.\n\nUse this EXACT format:\n\n{SUMMARY_FORMAT}\n\n{SUMMARY_DETAIL_RULES}"
    )
});

pub static UPDATE_SUMMARIZATION_PROMPT: LazyLock<String> = LazyLock::new(|| {
    format!(
        "The messages above are NEW conversation messages to incorporate into the existing handoff summary provided in <previous-summary> tags.\n\nUpdate the summary so it remains a complete handoff for the next model instance. RULES:\n- PRESERVE Done items from the previous summary together with their recorded results; do not drop or dilute them.\n- Move items from \"In Progress\" to \"Done\" ONLY when the new messages provide concrete evidence of completion; otherwise update where they stand.\n- Do NOT repeat or re-expand Failed Approaches already recorded; keep each recorded once and add newly failed approaches.\n- REMOVE Next Actions that were completed or are now obsolete; add new ones reflecting the current state.\n- ADD new files, verification results, decisions, and constraints from the new messages.\n- PRESERVE exact file paths, function names, commands, exit codes, and error lines.\n- Remove other content only when it is clearly superseded or no longer relevant to any active goal.\n\nUse this EXACT format:\n\n{SUMMARY_FORMAT}\n\n{SUMMARY_DETAIL_RULES}"
    )
});

pub static TURN_PREFIX_SUMMARIZATION_PROMPT: LazyLock<String> = LazyLock::new(|| {
    format!(
        "The messages above are the PREFIX of a single turn that was too large to keep in context. The most recent part of the turn (the suffix) is retained verbatim; your summary replaces only this prefix and is read together with the retained suffix.\n\nProduce a structured handoff summary of the prefix so the next model instance can understand the retained suffix and finish the turn. Scope every section to this turn's prefix; use \"(none)\" where the prefix has nothing to report.\n\nUse this EXACT format:\n\n{SUMMARY_FORMAT}\n\n{SUMMARY_DETAIL_RULES}"
    )
});

/// Run one summarization request with the isolation options every summary needs.
///
/// Summaries are standalone requests, so routing is isolated and cache writes
/// that cannot be reused are avoided: `cacheRetention` is forced to `None` and
/// a fresh session id is minted for each call. The retry policy and callbacks
/// are forwarded verbatim to [`retry_assistant_call`].
pub async fn complete_simple_with_retries(
    completer: &dyn SimpleCompleter,
    model: &Model,
    context: &Context,
    options: &SimpleStreamOptions,
    retry: Option<&RetryPolicy>,
    callbacks: Option<&(dyn RetryCallbacks + Send + Sync)>,
) -> AssistantMessage {
    let mut request_options = options.clone();
    request_options.base.cache_retention = Some(CacheRetention::None);
    request_options.base.session_id = Some(uuidv7());
    retry_assistant_call(
        || completer.complete_simple(model, context, &request_options),
        retry.copied(),
        request_options.base.base.signal.as_ref(),
        callbacks,
    )
    .await
}

fn combine_usage(first: &Usage, second: &Usage) -> Usage {
    Usage {
        input: first.input.saturating_add(second.input),
        output: first.output.saturating_add(second.output),
        cache_read: first.cache_read.saturating_add(second.cache_read),
        cache_write: first.cache_write.saturating_add(second.cache_write),
        cache_write_1h: combine_optional(first.cache_write_1h, second.cache_write_1h),
        reasoning: combine_optional(first.reasoning, second.reasoning),
        total_tokens: first.total_tokens.saturating_add(second.total_tokens),
        cost: UsageCost {
            input: first.cost.input + second.cost.input,
            output: first.cost.output + second.cost.output,
            cache_read: first.cost.cache_read + second.cost.cache_read,
            cache_write: first.cost.cache_write + second.cost.cache_write,
            total: first.cost.total + second.cost.total,
        },
    }
}

fn combine_optional(first: Option<u64>, second: Option<u64>) -> Option<u64> {
    match (first, second) {
        (None, None) => None,
        (left, right) => Some(left.unwrap_or(0).saturating_add(right.unwrap_or(0))),
    }
}

/// Summary text plus the provider usage of the call that produced it.
#[derive(Clone, Debug, PartialEq)]
pub struct SummaryWithUsage {
    /// Summary text.
    pub text: String,
    /// Usage reported by the summarization request.
    pub usage: Usage,
}

/// File-operation details stored on generated compaction entries.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionDetails {
    /// Files read in the compacted history.
    pub read_files: Vec<String>,
    /// Files modified in the compacted history.
    pub modified_files: Vec<String>,
}

/// Build the single user message that carries a summarization prompt.
pub(crate) fn summarization_user_message(text: String) -> Message {
    Message::User(UserMessage {
        content: UserContent::Blocks(vec![UserContentBlock::Text(TextContent {
            text,
            text_signature: None,
        })]),
        timestamp: now_ms(),
    })
}

fn summarization_options(
    max_tokens: u64,
    signal: Option<&AbortSignal>,
    thinking_level: Option<ModelThinkingLevel>,
    model: &Model,
) -> SimpleStreamOptions {
    let mut options = SimpleStreamOptions::default();
    options.base.max_tokens = Some(max_tokens);
    options.base.base.signal = signal.cloned();
    // Reasoning is only requested when the model supports it and a level above
    // "off" was asked for, matching the TypeScript conditional.
    options.reasoning = thinking_level
        .filter(|level| model.reasoning && *level != ModelThinkingLevel::Off)
        .and_then(thinking_level_to_reasoning);
    options
}

/// Generate or update a conversation summary for compaction.
#[allow(clippy::too_many_arguments)]
pub async fn generate_summary(
    current_messages: &[AgentMessage],
    completer: &dyn SimpleCompleter,
    model: &Model,
    reserve_tokens: u64,
    signal: Option<&AbortSignal>,
    custom_instructions: Option<&str>,
    previous_summary: Option<&str>,
    thinking_level: Option<ModelThinkingLevel>,
    retry: Option<&RetryPolicy>,
    callbacks: Option<&(dyn RetryCallbacks + Send + Sync)>,
) -> Result<String, CompactionError> {
    let result = generate_summary_with_usage(
        current_messages,
        completer,
        model,
        reserve_tokens,
        signal,
        custom_instructions,
        previous_summary,
        thinking_level,
        retry,
        callbacks,
    )
    .await?;
    Ok(result.text)
}

/// Generate or update a conversation summary and return its provider usage.
#[allow(clippy::too_many_arguments)]
pub async fn generate_summary_with_usage(
    current_messages: &[AgentMessage],
    completer: &dyn SimpleCompleter,
    model: &Model,
    reserve_tokens: u64,
    signal: Option<&AbortSignal>,
    custom_instructions: Option<&str>,
    previous_summary: Option<&str>,
    thinking_level: Option<ModelThinkingLevel>,
    retry: Option<&RetryPolicy>,
    callbacks: Option<&(dyn RetryCallbacks + Send + Sync)>,
) -> Result<SummaryWithUsage, CompactionError> {
    let max_tokens = pick_summary_max_tokens(model.max_tokens, reserve_tokens, 0.8);
    let mut base_prompt = match previous_summary {
        Some(_) => UPDATE_SUMMARIZATION_PROMPT.clone(),
        None => SUMMARIZATION_PROMPT.clone(),
    };
    if let Some(instructions) = custom_instructions {
        base_prompt = format!("{base_prompt}\n\nAdditional focus: {instructions}");
    }
    let llm_messages = convert_to_llm(current_messages);
    let conversation_text = serialize_conversation(&llm_messages, None);
    let mut prompt_text = format!("<conversation>\n{conversation_text}\n</conversation>\n\n");
    if let Some(previous_summary) = previous_summary {
        prompt_text.push_str(&format!(
            "<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
        ));
    }
    prompt_text.push_str(&base_prompt);

    let context = Context {
        system_prompt: Some(SUMMARIZATION_SYSTEM_PROMPT.to_owned()),
        messages: vec![summarization_user_message(prompt_text)],
        tools: None,
    };
    let options = summarization_options(max_tokens, signal, thinking_level, model);

    let response =
        complete_simple_with_retries(completer, model, &context, &options, retry, callbacks).await;
    if response.stop_reason == StopReason::Aborted {
        return Err(CompactionError::new(
            CompactionErrorCode::Aborted,
            response
                .error_message
                .as_deref()
                .unwrap_or("Summarization aborted"),
        ));
    }
    if response.stop_reason == StopReason::Error {
        return Err(CompactionError::new(
            CompactionErrorCode::SummarizationFailed,
            format!(
                "Summarization failed: {}",
                response.error_message.as_deref().unwrap_or("Unknown error")
            ),
        ));
    }
    if response.stop_reason == StopReason::Length {
        return Err(CompactionError::new(
            CompactionErrorCode::SummarizationFailed,
            format!(
                "Summarization failed: generation hit the {max_tokens}-token output cap and the summary is incomplete; raise reserveTokens if this recurs"
            ),
        ));
    }

    let text = assistant_content_text(&response.content, "\n");
    // Validate model text before split-turn scaffolding or file metadata can make it look nonempty.
    if text.trim().is_empty() {
        return Err(CompactionError::new(
            CompactionErrorCode::SummarizationFailed,
            "Summarization failed: empty summary",
        ));
    }

    Ok(SummaryWithUsage {
        text,
        usage: response.usage,
    })
}

/// Generated compaction data ready to be persisted as a compaction entry.
#[derive(Clone, Debug)]
pub struct CompactResult<T = serde_json::Value> {
    /// Summary text that replaces compacted history in future context.
    pub summary: String,
    /// Estimated context tokens before compaction.
    pub tokens_before: u64,
    /// Usage from the LLM call(s) that generated this summary, if available.
    pub usage: Option<Usage>,
    /// Retained recent messages stored directly on the compaction entry.
    pub retained_tail: Vec<AgentMessage>,
    /// Optional implementation-specific details stored with the compaction entry.
    pub details: Option<T>,
}

/// Prepared inputs for a compaction run.
#[derive(Clone, Debug)]
pub struct CompactionPreparation {
    /// Messages summarized into the history summary.
    pub messages_to_summarize: Vec<AgentMessage>,
    /// Prefix messages summarized separately when compaction splits a turn.
    pub turn_prefix_messages: Vec<AgentMessage>,
    /// Recent messages retained after compaction and stored on the compaction entry.
    pub retained_tail: Vec<AgentMessage>,
    /// Whether compaction splits a turn.
    pub is_split_turn: bool,
    /// Estimated context tokens before compaction.
    pub tokens_before: u64,
    /// Previous compaction summary used for iterative updates.
    pub previous_summary: Option<String>,
    /// File operations extracted from summarized history.
    pub file_ops: FileOperations,
    /// Settings used to prepare compaction.
    pub settings: CompactionSettings,
}

fn extract_file_operations(
    messages: &[AgentMessage],
    path_entries: &[Entry],
    prev_compaction_index: Option<usize>,
) -> FileOperations {
    let mut file_ops = create_file_ops();
    if let Some(Entry::Compaction(compaction)) =
        prev_compaction_index.and_then(|index| path_entries.get(index))
        && let Some(details) = &compaction.details
        && let Ok(details) = serde_json::from_value::<CompactionDetails>(details.clone())
    {
        file_ops.read.extend(details.read_files);
        file_ops.edited.extend(details.modified_files);
    }
    for message in messages {
        extract_file_ops_from_message(message, &mut file_ops);
    }
    file_ops
}

fn get_message_from_entry(entry: &Entry) -> Option<AgentMessage> {
    match entry {
        Entry::Message(message) => Some(message.message.clone()),
        Entry::BranchSummary(summary) => {
            Some(AgentMessage::BranchSummary(create_branch_summary_message(
                summary.summary.clone(),
                summary.from_id.clone(),
                summary.timestamp,
            )))
        }
        Entry::Compaction(compaction) => Some(AgentMessage::CompactionSummary(
            create_compaction_summary_message(
                compaction.summary.clone(),
                compaction.tokens_before,
                compaction.timestamp,
            ),
        )),
        _ => None,
    }
}

fn get_message_from_entry_for_compaction(entry: &Entry) -> Option<AgentMessage> {
    if matches!(entry, Entry::Compaction(_)) {
        return None;
    }
    get_message_from_entry(entry)
}

/// Prepare session entries for compaction, or return `None` when compaction is not applicable.
pub fn prepare_compaction(
    path_entries: &[Entry],
    settings: &CompactionSettings,
) -> Result<Option<CompactionPreparation>, CompactionError> {
    if path_entries.is_empty() || matches!(path_entries.last(), Some(Entry::Compaction(_))) {
        return Ok(None);
    }

    let mut prev_compaction: Option<(usize, &CompactionEntry)> = None;
    for (index, entry) in path_entries.iter().enumerate().rev() {
        if let Entry::Compaction(compaction) = entry {
            prev_compaction = Some((index, compaction));
            break;
        }
    }

    let mut previous_summary: Option<String> = None;
    let compactable_entries: Vec<Entry> = match prev_compaction {
        Some((index, compaction)) => {
            previous_summary = Some(compaction.summary.clone());
            let mut virtual_entries: Vec<Entry> = compaction
                .retained_tail
                .iter()
                .enumerate()
                .map(|(retained_index, message)| {
                    Entry::Message(MessageEntry {
                        id: format!("{}:retained:{retained_index}", compaction.id),
                        seq: compaction.seq,
                        parent_id: Some(if retained_index == 0 {
                            compaction.id.clone()
                        } else {
                            format!("{}:retained:{}", compaction.id, retained_index - 1)
                        }),
                        timestamp: message.timestamp(),
                        message: message.clone(),
                        terminate: false,
                    })
                })
                .collect();
            virtual_entries.extend_from_slice(&path_entries[index + 1..]);
            virtual_entries
        }
        None => path_entries.to_vec(),
    };
    let boundary_end = compactable_entries.len();

    let tokens_before = estimate_context_tokens(
        &build_session_context(path_entries, &SessionContextBuildOptions::default()).messages,
    )
    .tokens;

    let cut_point = find_cut_point(
        &compactable_entries,
        0,
        boundary_end,
        settings.keep_recent_tokens,
    );
    let history_end = if cut_point.is_split_turn {
        cut_point
            .turn_start_index
            .unwrap_or(cut_point.first_kept_entry_index)
            .min(boundary_end)
    } else {
        cut_point.first_kept_entry_index.min(boundary_end)
    };
    let mut messages_to_summarize: Vec<AgentMessage> = Vec::new();
    for entry in &compactable_entries[..history_end] {
        if let Some(message) = get_message_from_entry_for_compaction(entry) {
            messages_to_summarize.push(message);
        }
    }
    let mut turn_prefix_messages: Vec<AgentMessage> = Vec::new();
    if cut_point.is_split_turn {
        let prefix_start = cut_point
            .turn_start_index
            .unwrap_or(0)
            .min(cut_point.first_kept_entry_index);
        for entry in &compactable_entries[prefix_start..cut_point.first_kept_entry_index] {
            if let Some(message) = get_message_from_entry_for_compaction(entry) {
                turn_prefix_messages.push(message);
            }
        }
    }
    let mut retained_tail: Vec<AgentMessage> = Vec::new();
    for entry in
        &compactable_entries[cut_point.first_kept_entry_index.min(boundary_end)..boundary_end]
    {
        if let Some(message) = get_message_from_entry_for_compaction(entry) {
            retained_tail.push(message);
        }
    }
    let mut file_ops = extract_file_operations(
        &messages_to_summarize,
        path_entries,
        prev_compaction.map(|(index, _)| index),
    );
    if cut_point.is_split_turn {
        for message in &turn_prefix_messages {
            extract_file_ops_from_message(message, &mut file_ops);
        }
    }

    Ok(Some(CompactionPreparation {
        messages_to_summarize,
        turn_prefix_messages,
        retained_tail,
        is_split_turn: cut_point.is_split_turn,
        tokens_before,
        previous_summary,
        file_ops,
        settings: *settings,
    }))
}

/// Generate compaction summary data from prepared session history.
#[allow(clippy::too_many_arguments)]
pub async fn compact(
    preparation: &CompactionPreparation,
    completer: &dyn SimpleCompleter,
    model: &Model,
    custom_instructions: Option<&str>,
    signal: Option<&AbortSignal>,
    thinking_level: Option<ModelThinkingLevel>,
    retry: Option<&RetryPolicy>,
    callbacks: Option<&(dyn RetryCallbacks + Send + Sync)>,
) -> Result<CompactResult<CompactionDetails>, CompactionError> {
    let (summary, summary_usage) =
        if preparation.is_split_turn && !preparation.turn_prefix_messages.is_empty() {
            // With no new history to summarize, the previous checkpoint still carries the earlier context.
            let mut history_text = preparation
                .previous_summary
                .clone()
                .unwrap_or_else(|| "No prior history.".to_owned());
            let mut history_usage: Option<Usage> = None;
            if !preparation.messages_to_summarize.is_empty() {
                let history_result = generate_summary_with_usage(
                    &preparation.messages_to_summarize,
                    completer,
                    model,
                    preparation.settings.reserve_tokens,
                    signal,
                    custom_instructions,
                    preparation.previous_summary.as_deref(),
                    thinking_level,
                    retry,
                    callbacks,
                )
                .await?;
                history_text = history_result.text;
                history_usage = Some(history_result.usage);
            }
            let turn_prefix_result = generate_turn_prefix_summary(
                &preparation.turn_prefix_messages,
                completer,
                model,
                preparation.settings.reserve_tokens,
                signal,
                thinking_level,
                retry,
                callbacks,
            )
            .await?;
            let summary = format!(
                "{history_text}\n\n---\n\n**Turn Context (split turn):**\n\n{}",
                turn_prefix_result.text
            );
            let usage = match history_usage {
                Some(history_usage) => combine_usage(&history_usage, &turn_prefix_result.usage),
                None => turn_prefix_result.usage,
            };
            (summary, usage)
        } else {
            let summary_result = generate_summary_with_usage(
                &preparation.messages_to_summarize,
                completer,
                model,
                preparation.settings.reserve_tokens,
                signal,
                custom_instructions,
                preparation.previous_summary.as_deref(),
                thinking_level,
                retry,
                callbacks,
            )
            .await?;
            (summary_result.text, summary_result.usage)
        };

    let (read_files, modified_files) = compute_file_lists(&preparation.file_ops);
    let summary = format!(
        "{summary}{}",
        format_file_operations(&read_files, &modified_files)
    );

    Ok(CompactResult {
        summary,
        tokens_before: preparation.tokens_before,
        usage: Some(summary_usage),
        retained_tail: preparation.retained_tail.clone(),
        details: Some(CompactionDetails {
            read_files,
            modified_files,
        }),
    })
}

/// Summarize the prefix of a turn whose suffix is retained verbatim.
#[allow(clippy::too_many_arguments)]
async fn generate_turn_prefix_summary(
    messages: &[AgentMessage],
    completer: &dyn SimpleCompleter,
    model: &Model,
    reserve_tokens: u64,
    signal: Option<&AbortSignal>,
    thinking_level: Option<ModelThinkingLevel>,
    retry: Option<&RetryPolicy>,
    callbacks: Option<&(dyn RetryCallbacks + Send + Sync)>,
) -> Result<SummaryWithUsage, CompactionError> {
    // Smaller output budget for turn-prefix summaries: the suffix of the turn
    // is retained verbatim, so the summary only needs to describe the prefix.
    let max_tokens = pick_summary_max_tokens(model.max_tokens, reserve_tokens, 0.5);
    let llm_messages = convert_to_llm(messages);
    let conversation_text = serialize_conversation(&llm_messages, None);
    let turn_prefix_prompt = TURN_PREFIX_SUMMARIZATION_PROMPT.clone();
    let prompt_text =
        format!("<conversation>\n{conversation_text}\n</conversation>\n\n{turn_prefix_prompt}");
    let context = Context {
        system_prompt: Some(SUMMARIZATION_SYSTEM_PROMPT.to_owned()),
        messages: vec![summarization_user_message(prompt_text)],
        tools: None,
    };
    let options = summarization_options(max_tokens, signal, thinking_level, model);

    let response =
        complete_simple_with_retries(completer, model, &context, &options, retry, callbacks).await;
    if response.stop_reason == StopReason::Aborted {
        return Err(CompactionError::new(
            CompactionErrorCode::Aborted,
            response
                .error_message
                .as_deref()
                .unwrap_or("Turn prefix summarization aborted"),
        ));
    }
    if response.stop_reason == StopReason::Error {
        return Err(CompactionError::new(
            CompactionErrorCode::SummarizationFailed,
            format!(
                "Turn prefix summarization failed: {}",
                response.error_message.as_deref().unwrap_or("Unknown error")
            ),
        ));
    }
    if response.stop_reason == StopReason::Length {
        return Err(CompactionError::new(
            CompactionErrorCode::SummarizationFailed,
            format!(
                "Turn prefix summarization failed: generation hit the {max_tokens}-token output cap and the summary is incomplete; raise reserveTokens if this recurs"
            ),
        ));
    }

    let text = assistant_content_text(&response.content, "\n");
    // A valid history summary cannot substitute for a missing turn-prefix summary.
    if text.trim().is_empty() {
        return Err(CompactionError::new(
            CompactionErrorCode::SummarizationFailed,
            "Turn prefix summarization failed: empty summary",
        ));
    }

    Ok(SummaryWithUsage {
        text,
        usage: response.usage,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::utils::DEFAULT_TOOL_RESULT_TRUNCATION;
    use crate::session::types::ProvisionedEntry;
    use step_harness_providers::types::{Api, ProviderId, ToolCall};

    fn assistant_message(text: &str, usage: Usage, stop_reason: StopReason) -> AgentMessage {
        AgentMessage::Assistant(Box::new(provider_assistant_message(
            text,
            usage,
            stop_reason,
        )))
    }

    fn provider_assistant_message(
        text: &str,
        usage: Usage,
        stop_reason: StopReason,
    ) -> AssistantMessage {
        AssistantMessage {
            content: vec![AssistantContent::Text(TextContent {
                text: text.to_owned(),
                text_signature: None,
            })],
            api: Api::AnthropicMessages,
            provider: ProviderId::new("test"),
            model: "m".to_owned(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage,
            stop_reason,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            timestamp: 1,
        }
    }

    fn message_entry(
        id: &str,
        seq: u64,
        parent_id: Option<String>,
        message: AgentMessage,
    ) -> Entry {
        ProvisionedEntry::Message {
            id: id.to_owned(),
            message,
            terminate: false,
        }
        .into_entry(parent_id, seq, seq * 10)
    }

    fn usage(total: u64) -> Usage {
        Usage {
            input: total,
            output: 0,
            cache_read: 0,
            cache_write: 0,
            cache_write_1h: None,
            reasoning: None,
            total_tokens: total,
            cost: Default::default(),
        }
    }

    #[test]
    fn checked_summary_budget_rejects_invalid_fractions() {
        for fraction in [f64::NAN, f64::INFINITY, -0.5, 0.0, 1.1] {
            let error = try_pick_summary_max_tokens(4096, 24576, fraction).unwrap_err();
            assert_eq!(error.field, "reserve_fraction");
            assert_eq!(pick_summary_max_tokens(4096, 24576, fraction), 0);
        }
        assert_eq!(try_pick_summary_max_tokens(0, 10, 0.6).unwrap(), 6);
        assert_eq!(
            try_pick_summary_max_tokens(0, u64::MAX, 1.0).unwrap(),
            SUMMARY_OUTPUT_TOKENS_CEILING
        );
    }

    #[test]
    fn empty_cutpoint_is_none_and_unicode_estimation_is_not_chars_over_four() {
        assert_eq!(find_turn_start_index(&[], 0, 0), None);
        assert!(estimate_tokens(&AgentMessage::user("中文内容", 0)) >= 4);
        let estimated = estimate_context_tokens_with(&[AgentMessage::user("test", 0)], &|_| 123);
        assert_eq!(estimated.tokens, 123);
    }

    #[test]
    fn calculate_context_tokens_prefers_total_tokens() {
        assert_eq!(calculate_context_tokens(&usage(42)), 42);
        let mut partial = usage(0);
        partial.input = 10;
        partial.output = 5;
        partial.cache_read = 2;
        partial.cache_write = 1;
        assert_eq!(calculate_context_tokens(&partial), 18);
    }

    #[test]
    fn estimate_tokens_counts_characters_over_four() {
        let message = AgentMessage::user("abcdefgh", 1);
        assert_eq!(estimate_tokens(&message), 2);
        assert_eq!(estimate_tokens(&AgentMessage::user("", 1)), 0);
        let image = AgentMessage::user(
            step_harness_providers::types::UserContent::Blocks(vec![UserContentBlock::Image(
                step_harness_providers::types::ImageContent {
                    data: "x".to_owned(),
                    mime_type: "image/png".to_owned(),
                },
            )]),
            1,
        );
        assert_eq!(estimate_tokens(&image), 1200);
    }

    #[test]
    fn estimate_context_tokens_uses_last_usage_and_trailing_estimate() {
        let messages = vec![
            AgentMessage::user("old", 1),
            assistant_message("reply", usage(100), StopReason::Stop),
            AgentMessage::user("abcdefgh", 3),
        ];
        let estimate = estimate_context_tokens(&messages);
        assert_eq!(estimate.usage_tokens, 100);
        assert_eq!(estimate.trailing_tokens, 2);
        assert_eq!(estimate.tokens, 102);
        assert_eq!(estimate.last_usage_index, Some(1));
    }

    #[test]
    fn estimate_context_tokens_without_usage_estimates_everything() {
        let messages = vec![
            AgentMessage::user("abcdefgh", 1),
            AgentMessage::user("abcd", 2),
        ];
        let estimate = estimate_context_tokens(&messages);
        assert_eq!(estimate.tokens, 3);
        assert_eq!(estimate.usage_tokens, 0);
        assert_eq!(estimate.last_usage_index, None);
    }

    #[test]
    fn get_last_assistant_usage_skips_failed_and_aborted_turns() {
        let entries = vec![
            message_entry(
                "a",
                1,
                None,
                assistant_message("ok", usage(10), StopReason::Stop),
            ),
            message_entry(
                "b",
                2,
                Some("a".to_owned()),
                assistant_message("bad", usage(20), StopReason::Error),
            ),
        ];
        assert_eq!(get_last_assistant_usage(&entries), Some(usage(10)));
    }

    #[test]
    fn should_compact_respects_settings_and_reserve() {
        let settings = DEFAULT_COMPACTION_SETTINGS;
        assert!(should_compact(200_000, 200_000, &settings));
        assert!(!should_compact(100_000, 200_000, &settings));
        let disabled = CompactionSettings {
            enabled: false,
            ..settings
        };
        assert!(!should_compact(1_000_000, 1_000, &disabled));
    }

    #[test]
    fn pick_summary_max_tokens_bounds_by_model_and_ceiling() {
        // Unknown model limit: reserve fraction bounded by the ceiling.
        assert_eq!(pick_summary_max_tokens(0, 24_576, 0.8), 19_660);
        // A small model limit wins.
        assert_eq!(pick_summary_max_tokens(4_096, 24_576, 0.8), 4_096);
        // A huge model limit is capped by the ceiling.
        assert_eq!(pick_summary_max_tokens(1_000_000, 24_576, 0.8), 32_000);
        // Turn-prefix budget is half the reserve.
        assert_eq!(pick_summary_max_tokens(0, 24_576, 0.5), 12_288);
    }

    #[test]
    fn find_turn_start_index_walks_back_to_user_or_branch_summary() {
        let entries = vec![
            message_entry("u", 1, None, AgentMessage::user("hi", 1)),
            message_entry(
                "a",
                2,
                Some("u".to_owned()),
                assistant_message("reply", usage(1), StopReason::Stop),
            ),
        ];
        assert_eq!(find_turn_start_index(&entries, 1, 0), Some(0));
        assert_eq!(find_turn_start_index(&entries, 0, 0), Some(0));
        assert_eq!(find_turn_start_index(&entries, 1, 1), None);
    }

    #[test]
    fn find_cut_point_keeps_recent_tokens_at_a_valid_boundary() {
        let mut entries = vec![message_entry(
            "u1",
            1,
            None,
            AgentMessage::user("a".repeat(400), 1),
        )];
        entries.push(message_entry(
            "a1",
            2,
            Some("u1".to_owned()),
            assistant_message(&"b".repeat(400), usage(1), StopReason::Stop),
        ));
        entries.push(message_entry(
            "t1",
            3,
            Some("a1".to_owned()),
            AgentMessage::tool_result(step_harness_providers::types::ToolResultMessage {
                tool_call_id: "c".to_owned(),
                tool_name: "read".to_owned(),
                content: vec![UserContentBlock::Text(TextContent {
                    text: "x".to_owned(),
                    text_signature: None,
                })],
                details: None,
                usage: None,
                added_tool_names: None,
                is_error: false,
                timestamp: 3,
            }),
        ));
        entries.push(message_entry(
            "u2",
            4,
            Some("t1".to_owned()),
            AgentMessage::user("recent", 4),
        ));

        // Tool results are never cut points, so a 2-token budget lands the cut on
        // the last user message.
        let cut = find_cut_point(&entries, 0, entries.len(), 2);
        assert_eq!(cut.first_kept_entry_index, 3);
        assert!(!cut.is_split_turn);
        assert_eq!(cut.turn_start_index, None);
    }

    #[test]
    fn find_cut_point_detects_split_turns() {
        let entries = vec![
            message_entry("u1", 1, None, AgentMessage::user("hi", 1)),
            message_entry(
                "a1",
                2,
                Some("u1".to_owned()),
                assistant_message(&"c".repeat(4000), usage(1), StopReason::Stop),
            ),
        ];
        let cut = find_cut_point(&entries, 0, entries.len(), 10);
        assert_eq!(cut.first_kept_entry_index, 1);
        assert_eq!(cut.turn_start_index, Some(0));
        assert!(cut.is_split_turn);
    }

    #[test]
    fn find_cut_point_without_valid_cut_points_returns_start() {
        let entries = vec![message_entry(
            "t1",
            1,
            None,
            AgentMessage::tool_result(step_harness_providers::types::ToolResultMessage {
                tool_call_id: "c".to_owned(),
                tool_name: "read".to_owned(),
                content: vec![],
                details: None,
                usage: None,
                added_tool_names: None,
                is_error: false,
                timestamp: 1,
            }),
        )];
        let cut = find_cut_point(&entries, 0, entries.len(), 100);
        assert_eq!(
            cut,
            CutPointResult {
                first_kept_entry_index: 0,
                turn_start_index: None,
                is_split_turn: false,
            }
        );
    }

    #[test]
    fn prepare_compaction_returns_none_for_empty_or_compacted_paths() {
        assert!(
            prepare_compaction(&[], &DEFAULT_COMPACTION_SETTINGS)
                .unwrap()
                .is_none()
        );
        let compaction = ProvisionedEntry::Compaction {
            id: "c1".to_owned(),
            summary: "s".to_owned(),
            retained_tail: vec![],
            tokens_before: 10,
            details: None,
            usage: None,
        }
        .into_entry(None, 1, 10);
        assert!(
            prepare_compaction(&[compaction], &DEFAULT_COMPACTION_SETTINGS)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn prepare_compaction_splits_history_and_tail() {
        let mut entries = vec![message_entry(
            "u1",
            1,
            None,
            AgentMessage::user("a".repeat(400), 1),
        )];
        entries.push(message_entry(
            "a1",
            2,
            Some("u1".to_owned()),
            assistant_message(&"b".repeat(400), usage(500), StopReason::Stop),
        ));
        entries.push(message_entry(
            "u2",
            3,
            Some("a1".to_owned()),
            AgentMessage::user("recent", 3),
        ));

        // Keeping ~2 tokens retains only the trailing user message.
        let settings = CompactionSettings {
            keep_recent_tokens: 2,
            ..DEFAULT_COMPACTION_SETTINGS
        };
        let preparation = prepare_compaction(&entries, &settings)
            .unwrap()
            .expect("compaction applies");
        assert!(!preparation.is_split_turn);
        assert_eq!(preparation.messages_to_summarize.len(), 2);
        assert_eq!(preparation.retained_tail.len(), 1);
        assert!(preparation.previous_summary.is_none());
        // The assistant usage block (500) plus the estimated trailing message.
        assert_eq!(preparation.tokens_before, 502);
        assert_eq!(preparation.settings, settings);
    }

    #[test]
    fn prepare_compaction_reuses_previous_compaction_summary() {
        let compaction = ProvisionedEntry::Compaction {
            id: "c1".to_owned(),
            summary: "earlier".to_owned(),
            retained_tail: vec![AgentMessage::user("kept", 5)],
            tokens_before: 10,
            details: Some(serde_json::json!({
                "readFiles": ["kept.rs"],
                "modifiedFiles": ["changed.rs"]
            })),
            usage: None,
        }
        .into_entry(None, 1, 10);
        let entries = vec![
            message_entry("u0", 1, None, AgentMessage::user("before", 1)),
            compaction,
            message_entry(
                "u1",
                2,
                Some("c1".to_owned()),
                AgentMessage::user("after", 2),
            ),
        ];
        let settings = CompactionSettings {
            keep_recent_tokens: 1,
            ..DEFAULT_COMPACTION_SETTINGS
        };
        let preparation = prepare_compaction(&entries, &settings)
            .unwrap()
            .expect("compaction applies");
        assert_eq!(preparation.previous_summary.as_deref(), Some("earlier"));
        // The retained tail of the previous compaction participates again.
        assert!(
            preparation
                .messages_to_summarize
                .contains(&AgentMessage::user("kept", 5))
        );
        assert!(
            preparation
                .retained_tail
                .contains(&AgentMessage::user("after", 2))
        );
        assert!(preparation.file_ops.read.contains("kept.rs"));
        assert!(preparation.file_ops.edited.contains("changed.rs"));
    }

    #[test]
    fn prompts_are_composed_from_the_shared_format_and_rules() {
        assert!(
            SUMMARIZATION_PROMPT.starts_with("The messages above are a conversation to summarize.")
        );
        assert!(
            SUMMARIZATION_PROMPT.contains(&format!("{SUMMARY_FORMAT}\n\n{SUMMARY_DETAIL_RULES}"))
        );
        assert!(UPDATE_SUMMARIZATION_PROMPT.contains("<previous-summary>"));
        assert!(
            UPDATE_SUMMARIZATION_PROMPT
                .contains(&format!("{SUMMARY_FORMAT}\n\n{SUMMARY_DETAIL_RULES}"))
        );
        let turn_prefix_prompt = TURN_PREFIX_SUMMARIZATION_PROMPT.clone();
        assert!(turn_prefix_prompt.contains("the PREFIX of a single turn"));
        assert!(
            turn_prefix_prompt.contains(&format!("{SUMMARY_FORMAT}\n\n{SUMMARY_DETAIL_RULES}"))
        );
    }

    #[test]
    fn truncation_defaults_bound_each_tool_result() {
        let options = DEFAULT_TOOL_RESULT_TRUNCATION;
        assert_eq!(options.head_chars, 800);
        assert_eq!(options.tail_chars, 800);
        assert_eq!(options.max_salient_lines, 20);
        assert_eq!(options.max_salient_chars, 800);
    }

    #[test]
    fn combine_usage_sums_counters_and_keeps_optional_fields() {
        let mut first = usage(10);
        first.cache_write_1h = Some(1);
        first.reasoning = Some(2);
        let mut second = usage(20);
        second.cache_write_1h = Some(3);
        let combined = combine_usage(&first, &second);
        assert_eq!(combined.total_tokens, 30);
        assert_eq!(combined.input, 30);
        assert_eq!(combined.cache_write_1h, Some(4));
        assert_eq!(combined.reasoning, Some(2));
        assert_eq!(combine_usage(&usage(1), &usage(2)).reasoning, None);
    }

    #[test]
    fn tool_call_arguments_are_serialized_for_estimates() {
        let call = ToolCall {
            id: "c".to_owned(),
            name: "read".to_owned(),
            arguments: serde_json::json!({ "path": "a.rs" })
                .as_object()
                .unwrap()
                .clone(),
            thought_signature: None,
            namespace: None,
        };
        let message =
            AgentMessage::Assistant(Box::new(step_harness_providers::types::AssistantMessage {
                content: vec![AssistantContent::ToolCall(call)],
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
                timestamp: 1,
            }));
        // "read" (4) + `{"path":"a.rs"}` (14) = 18 chars -> 5 tokens.
        assert_eq!(estimate_tokens(&message), 5);
    }

    fn model() -> Model {
        Model {
            id: "m".to_owned(),
            name: "m".to_owned(),
            api: Api::AnthropicMessages,
            provider: ProviderId::new("test"),
            base_url: String::new(),
            reasoning: false,
            thinking_level_map: None,
            input: Vec::new(),
            cost: Default::default(),
            context_window: 128_000,
            // 0 means "unknown output limit", so the reserve fraction bounds the budget.
            max_tokens: 0,
            sampling_params: None,
            headers: None,
            compat: None,
        }
    }

    fn assistant_with_tool_call(name: &str, path: &str) -> AgentMessage {
        AgentMessage::Assistant(Box::new(step_harness_providers::types::AssistantMessage {
            content: vec![AssistantContent::ToolCall(ToolCall {
                id: "c1".to_owned(),
                name: name.to_owned(),
                arguments: serde_json::json!({ "path": path })
                    .as_object()
                    .unwrap()
                    .clone(),
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
            stop_reason: StopReason::Stop,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            timestamp: 2,
        }))
    }

    /// Completer that answers every request with a fixed text block.
    struct ScriptedCompleter {
        text: String,
    }

    #[async_trait::async_trait]
    impl SimpleCompleter for ScriptedCompleter {
        async fn complete_simple(
            &self,
            _model: &Model,
            _context: &Context,
            options: &SimpleStreamOptions,
        ) -> AssistantMessage {
            // Summaries are standalone requests: no reusable cache writes and a
            // fresh session id on every call.
            assert_eq!(options.base.cache_retention, Some(CacheRetention::None));
            assert!(options.base.session_id.is_some());
            assert!(options.base.max_tokens.unwrap_or(0) > 0);
            let mut message = provider_assistant_message("", usage(0), StopReason::Stop);
            message.content = vec![AssistantContent::Text(TextContent {
                text: self.text.clone(),
                text_signature: None,
            })];
            message.usage = usage(7);
            message
        }
    }

    #[tokio::test]
    async fn compact_appends_file_operations_and_returns_details() {
        let completer = ScriptedCompleter {
            text: "the summary".to_owned(),
        };
        let entries = vec![
            message_entry("u1", 1, None, AgentMessage::user("do it", 1)),
            message_entry(
                "a1",
                2,
                Some("u1".to_owned()),
                assistant_with_tool_call("read", "a.rs"),
            ),
            message_entry(
                "u2",
                3,
                Some("a1".to_owned()),
                AgentMessage::user("recent", 3),
            ),
        ];
        let settings = CompactionSettings {
            keep_recent_tokens: 1,
            ..DEFAULT_COMPACTION_SETTINGS
        };
        let preparation = prepare_compaction(&entries, &settings)
            .unwrap()
            .expect("compaction applies");
        assert!(!preparation.is_split_turn);

        let result = compact(
            &preparation,
            &completer,
            &model(),
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .unwrap();
        assert!(result.summary.starts_with("the summary"));
        assert!(
            result
                .summary
                .ends_with("<read-files>\na.rs\n</read-files>")
        );
        let details = result.details.expect("details are attached");
        assert_eq!(details.read_files, vec!["a.rs".to_owned()]);
        assert!(details.modified_files.is_empty());
        assert_eq!(result.usage.expect("usage is attached").total_tokens, 7);
        assert_eq!(result.retained_tail.len(), 1);
    }

    #[tokio::test]
    async fn compact_combines_history_and_turn_prefix_for_split_turns() {
        let completer = ScriptedCompleter {
            text: "prefix summary".to_owned(),
        };
        // One oversized assistant reply inside a single turn: the cut lands mid-turn.
        let entries = vec![
            message_entry("u1", 1, None, AgentMessage::user("hi", 1)),
            message_entry(
                "a1",
                2,
                Some("u1".to_owned()),
                assistant_message(&"c".repeat(4000), usage(0), StopReason::Stop),
            ),
        ];
        let settings = CompactionSettings {
            keep_recent_tokens: 10,
            ..DEFAULT_COMPACTION_SETTINGS
        };
        let preparation = prepare_compaction(&entries, &settings)
            .unwrap()
            .expect("compaction applies");
        assert!(preparation.is_split_turn);
        assert!(preparation.messages_to_summarize.is_empty());
        assert_eq!(preparation.turn_prefix_messages.len(), 1);

        let result = compact(
            &preparation,
            &completer,
            &model(),
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .unwrap();
        assert!(result.summary.starts_with("No prior history."));
        assert!(
            result
                .summary
                .contains("\n\n---\n\n**Turn Context (split turn):**\n\nprefix summary")
        );
        assert_eq!(result.usage.expect("usage is attached").total_tokens, 7);
        assert_eq!(result.retained_tail.len(), 1);
    }
}
