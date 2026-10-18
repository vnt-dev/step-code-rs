//! Settings file loading, merge precedence, and persistence.

pub mod manager;
pub mod schema;

pub use manager::{SettingsDiagnostic, SettingsManager, SettingsScope};
pub use schema::{
    BranchSummarySettingsFile, CompactionSettingsFile, DEFAULT_COMPACTION_KEEP_RECENT_TOKENS,
    DEFAULT_COMPACTION_RESERVE_TOKENS, DEFAULT_MAX_RETRIES, DEFAULT_RETRY_BASE_DELAY_MS,
    DEFAULT_THINKING_LEVEL, PermissionPresetSetting, PermissionsSettingsFile, QueueModeSetting,
    RetrySettingsFile, Settings, ThinkingBudgetsFile, ThinkingLevelSetting, deep_merge,
};
