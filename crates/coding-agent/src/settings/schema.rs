//! Settings schema and merge semantics.
//!
//! Ported functionally from `@step-harness/coding-agent`
//! `src/core/settings-manager.ts` and `src/step/settings-manager.ts`. Every
//! field is optional: the on-disk file only contains keys the user set, and
//! getters apply defaults. Merging follows the TypeScript `deepMerge`
//! semantics — a missing key never overwrites, an explicit `null` does.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use step_harness_providers::types::ModelThinkingLevel;

/// Reasoning levels accepted in settings (superset of the model level, with
/// the same wire names).
pub use step_harness_providers::types::ModelThinkingLevel as ThinkingLevelSetting;

/// How many queued steering messages are injected at a drain point.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum QueueModeSetting {
    #[default]
    OneAtATime,
    All,
}

impl From<QueueModeSetting> for step_harness_agent_core::types::QueueMode {
    fn from(value: QueueModeSetting) -> Self {
        match value {
            QueueModeSetting::OneAtATime => step_harness_agent_core::types::QueueMode::OneAtATime,
            QueueModeSetting::All => step_harness_agent_core::types::QueueMode::All,
        }
    }
}

/// Auto-compaction behavior.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionSettingsFile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    /// Tokens kept free below the context window before compacting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reserve_tokens: Option<u64>,
    /// Recent conversation always kept verbatim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keep_recent_tokens: Option<u64>,
}

/// Branch summarization behavior.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BranchSummarySettingsFile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reserve_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skip_prompt: Option<bool>,
}

/// Provider-error retry behavior.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RetrySettingsFile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_retries: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_delay_ms: Option<u64>,
}

/// Approval policy selection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PermissionPresetSetting {
    /// Safe tools run; writes and commands require approval.
    #[default]
    #[serde(
        alias = "readOnly",
        alias = "read-only",
        alias = "readonly",
        alias = "strict"
    )]
    Ask,
    /// Ordinary tools run; dangerous commands still require approval.
    Bypass,
    /// Full access: tool calls run without confirmation.
    Autopilot,
}

impl PermissionPresetSetting {
    /// Normalize the accepted aliases from the TypeScript source.
    pub fn from_alias(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "ask" | "confirm" => Some(Self::Ask),
            "read-only" | "readonly" | "strict" | "plan" => Some(Self::Ask),
            "bypass" | "auto" | "bypasspermissions" | "acceptedits" => Some(Self::Bypass),
            "autopilot" => Some(Self::Autopilot),
            _ => None,
        }
    }
}

/// Permission-related settings.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PermissionsSettingsFile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preset: Option<PermissionPresetSetting>,
    /// Fallback decision when no UI can answer a confirmation request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub non_interactive_approval: Option<bool>,
}

/// Per-level thinking token budgets.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingBudgetsFile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimal: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub low: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub medium: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub high: Option<u64>,
}

/// The complete settings schema.
///
/// Only the fields a desktop UI needs are modeled; terminal-only settings
/// (theme, keybindings, TUI modes) are intentionally absent.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Settings {
    /// Default provider id (e.g. `anthropic`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_provider: Option<String>,
    /// Default model id within the default provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,
    /// Default reasoning level for new sessions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_thinking_level: Option<ThinkingLevelSetting>,
    /// Per-model reasoning overrides, keyed `provider/modelId`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_thinking_levels: Option<BTreeMap<String, ThinkingLevelSetting>>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compaction: Option<CompactionSettingsFile>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub branch_summary: Option<BranchSummarySettingsFile>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry: Option<RetrySettingsFile>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub permissions: Option<PermissionsSettingsFile>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_budgets: Option<ThinkingBudgetsFile>,

    /// How queued steering messages are delivered.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steering_mode: Option<QueueModeSetting>,
    /// How queued follow-up messages are delivered.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub follow_up_mode: Option<QueueModeSetting>,

    /// Model globs (`provider/id` or `id`, `*`/`?` wildcards) that scope the
    /// model picker. Empty means "all models".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled_models: Option<Vec<String>>,
    /// Built-in tool names enabled by default. Empty means all.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_tools: Option<Vec<String>>,

    /// Hide thinking blocks in the UI by default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hide_thinking_block: Option<bool>,
    /// Load `AGENTS.md`-style context files into the system prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_files: Option<bool>,
    /// Extra skill directories or markdown files.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skills: Option<Vec<String>>,
    /// Override for the session store root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_dir: Option<String>,
}

impl Settings {
    /// Deep-merge `override` onto `self`, following the TypeScript semantics:
    /// only keys present in `override` replace the base value.
    #[must_use]
    pub fn merged_with(&self, overrides: &Settings) -> Settings {
        let mut base = serde_json::to_value(self).unwrap_or(serde_json::Value::Null);
        let patch = serde_json::to_value(overrides).unwrap_or(serde_json::Value::Null);
        deep_merge(&mut base, patch);
        serde_json::from_value(base).unwrap_or_else(|_| self.clone())
    }
}

/// Recursively merge `patch` into `base`. Objects merge per key; everything
/// else (including explicit `null`) replaces.
pub fn deep_merge(base: &mut serde_json::Value, patch: serde_json::Value) {
    match (base, patch) {
        (serde_json::Value::Object(base_map), serde_json::Value::Object(patch_map)) => {
            for (key, value) in patch_map {
                match base_map.get_mut(&key) {
                    Some(existing) => deep_merge(existing, value),
                    None => {
                        base_map.insert(key, value);
                    }
                }
            }
        }
        (base, patch) => *base = patch,
    }
}

/// Default reserved tokens for auto-compaction.
pub const DEFAULT_COMPACTION_RESERVE_TOKENS: u64 = 16_384;
/// Default recent-conversation floor for auto-compaction.
pub const DEFAULT_COMPACTION_KEEP_RECENT_TOKENS: u64 = 20_000;
/// Default retry ceiling.
pub const DEFAULT_MAX_RETRIES: u32 = 3;
/// Default retry base delay.
pub const DEFAULT_RETRY_BASE_DELAY_MS: u64 = 2_000;
/// Default reasoning level for new sessions.
pub const DEFAULT_THINKING_LEVEL: ModelThinkingLevel = ModelThinkingLevel::Medium;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn merges_nested_objects_per_key() {
        let base: Settings = serde_json::from_value(json!({
            "defaultModel": "a",
            "compaction": {"enabled": true, "reserveTokens": 1000}
        }))
        .unwrap();
        let overrides: Settings = serde_json::from_value(json!({
            "defaultModel": "b",
            "compaction": {"reserveTokens": 2000}
        }))
        .unwrap();
        let merged = base.merged_with(&overrides);
        assert_eq!(merged.default_model.as_deref(), Some("b"));
        let compaction = merged.compaction.unwrap();
        assert_eq!(compaction.enabled, Some(true));
        assert_eq!(compaction.reserve_tokens, Some(2000));
    }

    #[test]
    fn absent_keys_never_overwrite() {
        // Struct-level merging cannot distinguish "absent" from "explicit
        // null"; that distinction only exists on raw file JSON, which the
        // settings manager merges directly (see manager tests).
        let base: Settings = serde_json::from_value(json!({"defaultModel": "a"})).unwrap();
        let overrides: Settings = serde_json::from_value(json!({})).unwrap();
        let merged = base.merged_with(&overrides);
        assert_eq!(merged.default_model.as_deref(), Some("a"));
    }

    #[test]
    fn raw_json_null_replaces_the_base_value() {
        let mut base = json!({"defaultModel": "a"});
        deep_merge(&mut base, json!({"defaultModel": null}));
        assert_eq!(base["defaultModel"], serde_json::Value::Null);
    }

    #[test]
    fn missing_keys_never_overwrite() {
        let base: Settings = serde_json::from_value(json!({"defaultModel": "a"})).unwrap();
        let overrides: Settings = serde_json::from_value(json!({"defaultProvider": "p"})).unwrap();
        let merged = base.merged_with(&overrides);
        assert_eq!(merged.default_model.as_deref(), Some("a"));
        assert_eq!(merged.default_provider.as_deref(), Some("p"));
    }

    #[test]
    fn legacy_read_only_preset_migrates_to_ask() {
        for legacy in ["readOnly", "read-only", "readonly", "strict"] {
            assert_eq!(
                serde_json::from_value::<PermissionPresetSetting>(json!(legacy)).unwrap(),
                PermissionPresetSetting::Ask
            );
        }
    }

    #[test]
    fn preset_aliases_normalize() {
        assert_eq!(
            PermissionPresetSetting::from_alias("confirm"),
            Some(PermissionPresetSetting::Ask)
        );
        assert_eq!(
            PermissionPresetSetting::from_alias("ReadOnly"),
            Some(PermissionPresetSetting::Ask)
        );
        assert_eq!(
            PermissionPresetSetting::from_alias("bypassPermissions"),
            Some(PermissionPresetSetting::Bypass)
        );
        assert_eq!(PermissionPresetSetting::from_alias("nonsense"), None);
    }
}
