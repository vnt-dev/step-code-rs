//! Context compaction: threshold estimation, cut-point selection, summary
//! generation, and branch summarization.
//!
//! Ported from `@step-harness/agent-core` `src/harness/compaction/`. The
//! subsystem is split into three modules:
//!
//! - [`compaction`] — the compaction pipeline: settings, token estimates,
//!   cut-point selection, summary generation, and [`compact`].
//! - [`branch_summarization`] — summarizing an abandoned branch before the
//!   conversation navigates elsewhere.
//! - [`utils`] — file-operation accumulators, conversation serialization, and
//!   the shared summarization system prompt.
//! - [`projection`] — request-time context projection: a pure, deterministic
//!   rewrite of the LLM-facing message array (see
//!   [`project_context_for_request`]). Its mechanics are split across
//!   [`projection_options`], [`projection_content`], [`projection_rules`],
//!   [`projection_invariants`], and [`projection_salient`].
//!
//! # Notable differences from the TypeScript source
//!
//! - `models.completeSimple` becomes the [`SimpleCompleter`] trait, so callers
//!   supply any object that can run a one-shot assistant request (the real
//!   `Models` collection, a test double, or a proxy).
//! - The TypeScript `Result<T, CompactionError>` monad is
//!   [`std::result::Result`]; the `-1` "not found" sentinel used for
//!   `turnStartIndex` becomes `Option<usize>`.
//! - Token counts are `u64`, character counts are `usize`, and fractions are
//!   `f64`.
//! - Retry policy and callbacks are passed as `Option<&RetryPolicy>` /
//!   `Option<&(dyn RetryCallbacks + Send + Sync)>`; the TypeScript source
//!   forwards the same pair to `retryAssistantCall`.

use step_harness_providers::types::{AssistantMessage, Context, Model, SimpleStreamOptions};

pub mod branch_summarization;
mod budget;
pub use budget::RatioError;
pub mod core;
pub mod projection;
pub mod projection_content;
pub mod projection_invariants;
pub mod projection_options;
pub mod projection_rules;
pub mod projection_salient;
pub mod utils;

pub use branch_summarization::{
    BRANCH_SUMMARY_PREAMBLE, BRANCH_SUMMARY_PROMPT, BranchPreparation, BranchSummaryDetails,
    BranchSummaryResult, CollectEntriesResult, DEFAULT_BRANCH_RESERVE_TOKENS,
    GenerateBranchSummaryOptions, collect_entries_for_branch_summary, generate_branch_summary,
    prepare_branch_entries,
};
pub use core::{
    CompactResult, CompactionDetails, CompactionPreparation, CompactionSettings,
    ContextUsageEstimate, CutPointResult, DEFAULT_COMPACTION_SETTINGS, SUMMARIZATION_PROMPT,
    SUMMARY_DETAIL_RULES, SUMMARY_FORMAT, SUMMARY_OUTPUT_TOKENS_CEILING, SummaryWithUsage,
    TURN_PREFIX_SUMMARIZATION_PROMPT, UPDATE_SUMMARIZATION_PROMPT, calculate_context_tokens,
    compact, complete_simple_with_retries, estimate_context_tokens, estimate_context_tokens_with,
    estimate_tokens, find_cut_point, find_turn_start_index, generate_summary,
    generate_summary_with_usage, get_last_assistant_usage, pick_summary_max_tokens,
    prepare_compaction, should_compact, try_pick_summary_max_tokens,
};
pub use projection::{
    ContextProjectionMode, CutResult, DEFAULT_CAP_RATIO, DEFAULT_KEEP_RECENT_TOKENS,
    DEFAULT_SOFT_THRESHOLD_RATIO, PROJECTION_CUT_MARKER_PREFIX, PROJECTION_REPEAT_MARKER_PREFIX,
    PROJECTION_SUMMARY_MARKER_PREFIX, ProjectionByRuleStats, ProjectionOptions, ProjectionResult,
    ProjectionSkippedReason, ProjectionStats, cut_text_with_salient_lines,
    estimate_projection_tokens, project_context_for_request, short_content_hash,
    verify_projection_invariants,
};
pub use utils::{
    DEFAULT_TOOL_RESULT_TRUNCATION, FileOperations, SUMMARIZATION_SYSTEM_PROMPT,
    ToolResultTruncationOptions, compute_file_lists, create_file_ops,
    extract_file_ops_from_message, format_file_operations, safe_json_stringify,
    serialize_conversation,
};

/// One-shot assistant request used by the summarization entry points.
///
/// The Rust counterpart of the TypeScript `models.completeSimple(model,
/// context, options)` call. Implementations own auth resolution, provider
/// dispatch, and event streaming; this trait only exposes the non-streaming
/// "give me the final assistant message" shape the summarizers need.
///
/// Summarization requests are standalone: callers in this module force
/// `cacheRetention = None` and a fresh session id before dispatching, so
/// implementations do not need to special-case them.
#[async_trait::async_trait]
pub trait SimpleCompleter: Send + Sync {
    /// Run one non-streaming assistant request and return the final message.
    ///
    /// Failures are encoded in the returned [`AssistantMessage`] (`stop_reason`
    /// of `Error` or `Aborted`), never as an `Err`, matching the TypeScript
    /// contract that `completeSimple` resolves with a terminal message.
    async fn complete_simple(
        &self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
    ) -> AssistantMessage;
}
