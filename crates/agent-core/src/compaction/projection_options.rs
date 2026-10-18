//! Public option and statistics types for `project_context_for_request`, their
//! defaults, and the per-rule counter / tuning-knob shapes shared with the rule
//! implementations.
//!
//! Ported from `@step-harness/agent-core`
//! `src/harness/compaction/projection-options.ts`.

use serde::{Deserialize, Serialize};

// ============================================================================
// Modes and statistics
// ============================================================================

/// Feature-flag values for `step.compaction.contextProjection`.
///
/// The TypeScript source is a string union (`"off" | "lightweight-v1"`). Per the
/// migration spec this is modelled as an enum serialized with `camelCase`, so
/// the wire value of [`ContextProjectionMode::LightweightV1`] is
/// `"lightweightV1"` rather than the TypeScript `"lightweight-v1"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ContextProjectionMode {
    Off,
    LightweightV1,
}

/// Why a projection run did not rewrite anything.
///
/// Serialized `kebab-case` to match the TypeScript string-union values
/// (`"empty-messages"`, `"no-context-window"`, `"below-soft-threshold"`,
/// `"no-reducible-content"`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProjectionSkippedReason {
    EmptyMessages,
    NoContextWindow,
    BelowSoftThreshold,
    NoReducibleContent,
    InvalidOptions,
}

/// Per-rule rewrite counters. Keys follow the telemetry naming.
///
/// Field names are intentionally snake_case (the serde default) to match the
/// TypeScript telemetry keys exactly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectionByRuleStats {
    /// Rule a: large toolResult / bashExecution outputs cut to head+tail+salient lines.
    pub tool_result_cuts: usize,
    /// Rule b: repeated search/test outputs folded into a one-liner.
    pub dedup_folds: usize,
    /// Rule c: historical assistant thinking blocks dropped.
    pub thinking_drops: usize,
    /// Rule d: repeated branch/compaction summaries deduplicated.
    pub summary_dedups: usize,
    /// Rule e: large code / patch / JSON payloads cut on line boundaries.
    pub code_cuts: usize,
}

/// A fresh, zeroed per-rule counter set.
pub fn create_by_rule_stats() -> ProjectionByRuleStats {
    ProjectionByRuleStats::default()
}

/// Result statistics for one projection run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectionStats {
    /// True when at least one rewrite was applied and all invariants held.
    pub applied: bool,
    /// Context tokens before projection (usage-based when provided, else estimated).
    pub original_tokens: u64,
    /// Estimated context tokens after projection (originalTokens minus estimated savings).
    pub projected_tokens: u64,
    /// Total content characters before projection.
    pub original_chars: usize,
    /// Total content characters after projection.
    pub projected_chars: usize,
    /// Rewrite counts per projection rule.
    pub by_rule: ProjectionByRuleStats,
    /// True when the post-projection invariant verification passed (or nothing was rewritten).
    pub invariants_passed: bool,
    /// Set when verification failed; the original messages were returned unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub invariant_violation: Option<String>,
    /// Set when projection did not run at all.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skipped_reason: Option<ProjectionSkippedReason>,
    /// True when the second, more aggressive pass ran because the cap target was still exceeded.
    pub aggressive_pass: bool,
}

// ============================================================================
// Options and tuning knobs
// ============================================================================

/// Tuning knobs for `project_context_for_request`. All fields optional.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectionOptions {
    /// Model context window in tokens. Required for the trigger; `None`/0 disables projection.
    pub context_window: Option<u64>,
    ///
    /// Context tokens of the *unprojected* request, preferably derived from provider
    /// usage. When omitted, a chars/4 estimate over `messages` is used.
    pub context_tokens: Option<u64>,
    /// Trigger fraction, finite and in (0, 1]. Its token threshold is rounded up. Default 0.6.
    pub soft_threshold_ratio: Option<f64>,
    /// Cap fraction, finite and in (0, 1]. Its token budget is rounded down. Default 0.75.
    pub cap_ratio: Option<f64>,
    /// Token span at the tail that is never modified. Default 20000.
    pub keep_recent_tokens: Option<u64>,
    /// Number of most recent thinking-bearing assistant messages whose thinking is kept. Default 2.
    pub keep_thinking_blocks: Option<usize>,
    /// Character budget for the head slice of a cut message. Default 800.
    pub head_chars: Option<usize>,
    /// Character budget for the tail slice of a cut message. Default 800.
    pub tail_chars: Option<usize>,
    /// Maximum salient lines preserved between head and tail. Default 20.
    pub max_salient_lines: Option<usize>,
    /// Minimum text-block size (chars) before the large-content rules cut it. Default 4000.
    pub large_min_chars: Option<usize>,
}

impl ProjectionOptions {
    /// Reject invalid fractional settings before any context can be rewritten.
    pub fn validate(&self) -> Result<(), super::RatioError> {
        if let Some(ratio) = self.soft_threshold_ratio {
            super::budget::validate_ratio("soft_threshold_ratio", ratio)?;
        }
        if let Some(ratio) = self.cap_ratio {
            super::budget::validate_ratio("cap_ratio", ratio)?;
        }
        Ok(())
    }
}

/// Effective per-pass tuning values (defaults or aggressive-pass overrides).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProjectionKnobs {
    pub head_chars: usize,
    pub tail_chars: usize,
    pub max_salient_lines: usize,
    pub large_min_chars: usize,
    pub keep_thinking_blocks: usize,
}

// ============================================================================
// Defaults
// ============================================================================

/// Default soft threshold ratio: projection only runs at or above this fraction of the window.
pub const DEFAULT_SOFT_THRESHOLD_RATIO: f64 = 0.6;
/// Default cap ratio: the aggressive pass runs while still above this fraction of the window.
pub const DEFAULT_CAP_RATIO: f64 = 0.75;
/// Default tail token span that is never modified.
pub const DEFAULT_KEEP_RECENT_TOKENS: u64 = 20000;
const DEFAULT_KEEP_THINKING_BLOCKS: usize = 2;
const DEFAULT_HEAD_CHARS: usize = 800;
const DEFAULT_TAIL_CHARS: usize = 800;
const DEFAULT_MAX_SALIENT_LINES: usize = 20;
const DEFAULT_LARGE_MIN_CHARS: usize = 4000;

/// Aggressive second-pass knobs used when the first pass stays above the cap.
const AGGRESSIVE_HEAD_CHARS: usize = 400;
const AGGRESSIVE_TAIL_CHARS: usize = 400;
const AGGRESSIVE_MAX_SALIENT_LINES: usize = 10;
const AGGRESSIVE_LARGE_MIN_CHARS: usize = 2000;
const AGGRESSIVE_KEEP_THINKING_BLOCKS: usize = 1;

/// Resolve the effective first-pass knobs from `options`, filling in defaults.
pub fn resolve_projection_knobs(options: Option<&ProjectionOptions>) -> ProjectionKnobs {
    ProjectionKnobs {
        head_chars: options
            .and_then(|o| o.head_chars)
            .unwrap_or(DEFAULT_HEAD_CHARS),
        tail_chars: options
            .and_then(|o| o.tail_chars)
            .unwrap_or(DEFAULT_TAIL_CHARS),
        max_salient_lines: options
            .and_then(|o| o.max_salient_lines)
            .unwrap_or(DEFAULT_MAX_SALIENT_LINES),
        large_min_chars: options
            .and_then(|o| o.large_min_chars)
            .unwrap_or(DEFAULT_LARGE_MIN_CHARS),
        keep_thinking_blocks: options
            .and_then(|o| o.keep_thinking_blocks)
            .unwrap_or(DEFAULT_KEEP_THINKING_BLOCKS),
    }
}

/// Tighten each knob toward the aggressive-pass ceiling (never looser).
pub fn to_aggressive_knobs(knobs: ProjectionKnobs) -> ProjectionKnobs {
    ProjectionKnobs {
        head_chars: knobs.head_chars.min(AGGRESSIVE_HEAD_CHARS),
        tail_chars: knobs.tail_chars.min(AGGRESSIVE_TAIL_CHARS),
        max_salient_lines: knobs.max_salient_lines.min(AGGRESSIVE_MAX_SALIENT_LINES),
        large_min_chars: knobs.large_min_chars.min(AGGRESSIVE_LARGE_MIN_CHARS),
        keep_thinking_blocks: knobs
            .keep_thinking_blocks
            .min(AGGRESSIVE_KEEP_THINKING_BLOCKS),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_fill_every_knob() {
        let knobs = resolve_projection_knobs(None);
        assert_eq!(knobs.head_chars, 800);
        assert_eq!(knobs.tail_chars, 800);
        assert_eq!(knobs.max_salient_lines, 20);
        assert_eq!(knobs.large_min_chars, 4000);
        assert_eq!(knobs.keep_thinking_blocks, 2);
    }

    #[test]
    fn options_override_defaults() {
        let options = ProjectionOptions {
            head_chars: Some(123),
            keep_thinking_blocks: Some(5),
            ..Default::default()
        };
        let knobs = resolve_projection_knobs(Some(&options));
        assert_eq!(knobs.head_chars, 123);
        assert_eq!(knobs.keep_thinking_blocks, 5);
        // untouched fields keep their defaults
        assert_eq!(knobs.tail_chars, 800);
        assert_eq!(knobs.large_min_chars, 4000);
    }

    #[test]
    fn aggressive_knobs_only_tighten() {
        let base = resolve_projection_knobs(None);
        let aggressive = to_aggressive_knobs(base);
        assert_eq!(aggressive.head_chars, 400);
        assert_eq!(aggressive.tail_chars, 400);
        assert_eq!(aggressive.max_salient_lines, 10);
        assert_eq!(aggressive.large_min_chars, 2000);
        assert_eq!(aggressive.keep_thinking_blocks, 1);

        // A knob already below the aggressive ceiling is preserved.
        let tight = ProjectionKnobs {
            head_chars: 10,
            tail_chars: 10,
            max_salient_lines: 1,
            large_min_chars: 100,
            keep_thinking_blocks: 0,
        };
        assert_eq!(to_aggressive_knobs(tight), tight);
    }

    #[test]
    fn skipped_reason_serde_is_kebab_case() {
        let reason = ProjectionSkippedReason::NoContextWindow;
        let json = serde_json::to_value(reason).unwrap();
        assert_eq!(json, serde_json::json!("no-context-window"));
        let round: ProjectionSkippedReason = serde_json::from_value(json).unwrap();
        assert_eq!(round, reason);
    }

    #[test]
    fn by_rule_stats_default_zeroed() {
        assert_eq!(create_by_rule_stats(), ProjectionByRuleStats::default());
    }
}
