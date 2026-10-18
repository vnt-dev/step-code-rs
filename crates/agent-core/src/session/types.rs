//! Durable session types: entries, records, queries, and errors.
//!
//! Ported from `@step-harness/agent-core` `src\harness\session\types.ts`.
//! The TypeScript discriminated unions become serde-tagged enums; the
//! `ProvisionedEntry`/`NewRecord` write-side shapes (storage-assigned fields
//! omitted) are separate enums.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use step_harness_providers::types::{StopReason, Usage};

use crate::types::AgentMessage;

/// A JSON value used for opaque session data.
pub type JsonValue = serde_json::Value;

/// Stop reasons persisted in usage records (excludes the transient `pending`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SessionStopReason {
    Stop,
    Length,
    ToolUse,
    Error,
    Aborted,
    Deferred,
}

impl SessionStopReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            SessionStopReason::Stop => "stop",
            SessionStopReason::Length => "length",
            SessionStopReason::ToolUse => "toolUse",
            SessionStopReason::Error => "error",
            SessionStopReason::Aborted => "aborted",
            SessionStopReason::Deferred => "deferred",
        }
    }
}

impl TryFrom<StopReason> for SessionStopReason {
    type Error = ();

    fn try_from(value: StopReason) -> Result<Self, ()> {
        match value {
            StopReason::Stop => Ok(SessionStopReason::Stop),
            StopReason::Length => Ok(SessionStopReason::Length),
            StopReason::ToolUse => Ok(SessionStopReason::ToolUse),
            StopReason::Error => Ok(SessionStopReason::Error),
            StopReason::Aborted => Ok(SessionStopReason::Aborted),
            StopReason::Deferred => Ok(SessionStopReason::Deferred),
            StopReason::Pending => Err(()),
        }
    }
}

/// Generates entry ids (UUIDv7 by default).
pub trait IdGenerator: Send + Sync {
    fn next(&self) -> String;
}

/// Default id generator producing UUIDv7 strings.
#[derive(Clone, Copy, Debug, Default)]
pub struct UuidV7Generator;

impl IdGenerator for UuidV7Generator {
    fn next(&self) -> String {
        uuid::Uuid::now_v7().to_string()
    }
}

// ---------------------------------------------------------------------------
// Entries
// ---------------------------------------------------------------------------

/// A conversation message entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageEntry {
    pub id: String,
    pub seq: u64,
    pub parent_id: Option<String>,
    pub timestamp: u64,
    pub message: AgentMessage,
    /// Tool-result termination decision.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub terminate: bool,
}

/// A model change entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelChangeEntry {
    pub id: String,
    pub seq: u64,
    pub parent_id: Option<String>,
    pub timestamp: u64,
    pub provider: String,
    pub model_id: String,
}

/// A thinking level change entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingLevelEntry {
    pub id: String,
    pub seq: u64,
    pub parent_id: Option<String>,
    pub timestamp: u64,
    pub thinking_level: String,
}

/// An active-tools change entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ActiveToolsEntry {
    pub id: String,
    pub seq: u64,
    pub parent_id: Option<String>,
    pub timestamp: u64,
    pub active_tool_names: Vec<String>,
}

/// A compaction entry: the summary plus the retained tail messages.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionEntry {
    pub id: String,
    pub seq: u64,
    pub parent_id: Option<String>,
    pub timestamp: u64,
    pub summary: String,
    pub retained_tail: Vec<AgentMessage>,
    pub tokens_before: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<JsonValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// A branch summary entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BranchSummaryEntry {
    pub id: String,
    pub seq: u64,
    pub parent_id: Option<String>,
    pub timestamp: u64,
    pub from_id: String,
    pub summary: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<JsonValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// An application-defined custom entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomEntry {
    pub id: String,
    pub seq: u64,
    pub parent_id: Option<String>,
    pub timestamp: u64,
    pub custom_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<JsonValue>,
}

/// A durable entry in the session tree.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Entry {
    Message(MessageEntry),
    ModelChange(ModelChangeEntry),
    ThinkingLevelChange(ThinkingLevelEntry),
    ActiveToolsChange(ActiveToolsEntry),
    Compaction(CompactionEntry),
    BranchSummary(BranchSummaryEntry),
    Custom(CustomEntry),
}

impl Entry {
    pub fn id(&self) -> &str {
        match self {
            Entry::Message(entry) => &entry.id,
            Entry::ModelChange(entry) => &entry.id,
            Entry::ThinkingLevelChange(entry) => &entry.id,
            Entry::ActiveToolsChange(entry) => &entry.id,
            Entry::Compaction(entry) => &entry.id,
            Entry::BranchSummary(entry) => &entry.id,
            Entry::Custom(entry) => &entry.id,
        }
    }

    pub fn seq(&self) -> u64 {
        match self {
            Entry::Message(entry) => entry.seq,
            Entry::ModelChange(entry) => entry.seq,
            Entry::ThinkingLevelChange(entry) => entry.seq,
            Entry::ActiveToolsChange(entry) => entry.seq,
            Entry::Compaction(entry) => entry.seq,
            Entry::BranchSummary(entry) => entry.seq,
            Entry::Custom(entry) => entry.seq,
        }
    }

    pub fn parent_id(&self) -> Option<&str> {
        match self {
            Entry::Message(entry) => entry.parent_id.as_deref(),
            Entry::ModelChange(entry) => entry.parent_id.as_deref(),
            Entry::ThinkingLevelChange(entry) => entry.parent_id.as_deref(),
            Entry::ActiveToolsChange(entry) => entry.parent_id.as_deref(),
            Entry::Compaction(entry) => entry.parent_id.as_deref(),
            Entry::BranchSummary(entry) => entry.parent_id.as_deref(),
            Entry::Custom(entry) => entry.parent_id.as_deref(),
        }
    }

    pub fn timestamp(&self) -> u64 {
        match self {
            Entry::Message(entry) => entry.timestamp,
            Entry::ModelChange(entry) => entry.timestamp,
            Entry::ThinkingLevelChange(entry) => entry.timestamp,
            Entry::ActiveToolsChange(entry) => entry.timestamp,
            Entry::Compaction(entry) => entry.timestamp,
            Entry::BranchSummary(entry) => entry.timestamp,
            Entry::Custom(entry) => entry.timestamp,
        }
    }

    /// Wire type discriminant of this entry.
    pub fn entry_type(&self) -> &'static str {
        match self {
            Entry::Message(_) => "message",
            Entry::ModelChange(_) => "model_change",
            Entry::ThinkingLevelChange(_) => "thinking_level_change",
            Entry::ActiveToolsChange(_) => "active_tools_change",
            Entry::Compaction(_) => "compaction",
            Entry::BranchSummary(_) => "branch_summary",
            Entry::Custom(_) => "custom",
        }
    }

    /// Known entry type discriminants, in wire order.
    pub const ENTRY_TYPES: [&'static str; 7] = [
        "message",
        "model_change",
        "thinking_level_change",
        "active_tools_change",
        "compaction",
        "branch_summary",
        "custom",
    ];

    /// True when this entry carries a message.
    pub fn message(&self) -> Option<&AgentMessage> {
        match self {
            Entry::Message(entry) => Some(&entry.message),
            _ => None,
        }
    }
}

/// The write-side shape of an entry: storage-assigned fields are omitted.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProvisionedEntry {
    #[serde(rename_all = "camelCase")]
    Message {
        id: String,
        message: AgentMessage,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        terminate: bool,
    },
    #[serde(rename_all = "camelCase")]
    ModelChange {
        id: String,
        provider: String,
        model_id: String,
    },
    #[serde(rename_all = "camelCase")]
    ThinkingLevelChange { id: String, thinking_level: String },
    #[serde(rename_all = "camelCase")]
    ActiveToolsChange {
        id: String,
        active_tool_names: Vec<String>,
    },
    #[serde(rename_all = "camelCase")]
    Compaction {
        id: String,
        summary: String,
        retained_tail: Vec<AgentMessage>,
        tokens_before: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        details: Option<JsonValue>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<Usage>,
    },
    #[serde(rename_all = "camelCase")]
    BranchSummary {
        id: String,
        from_id: String,
        summary: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        details: Option<JsonValue>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<Usage>,
    },
    #[serde(rename_all = "camelCase")]
    Custom {
        id: String,
        custom_type: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        data: Option<JsonValue>,
    },
}

impl ProvisionedEntry {
    pub fn id(&self) -> &str {
        match self {
            ProvisionedEntry::Message { id, .. }
            | ProvisionedEntry::ModelChange { id, .. }
            | ProvisionedEntry::ThinkingLevelChange { id, .. }
            | ProvisionedEntry::ActiveToolsChange { id, .. }
            | ProvisionedEntry::Compaction { id, .. }
            | ProvisionedEntry::BranchSummary { id, .. }
            | ProvisionedEntry::Custom { id, .. } => id,
        }
    }

    /// Fill in the storage-assigned fields, producing a full entry.
    pub fn into_entry(self, parent_id: Option<String>, seq: u64, timestamp: u64) -> Entry {
        match self {
            ProvisionedEntry::Message {
                id,
                message,
                terminate,
            } => Entry::Message(MessageEntry {
                id,
                seq,
                parent_id,
                timestamp,
                message,
                terminate,
            }),
            ProvisionedEntry::ModelChange {
                id,
                provider,
                model_id,
            } => Entry::ModelChange(ModelChangeEntry {
                id,
                seq,
                parent_id,
                timestamp,
                provider,
                model_id,
            }),
            ProvisionedEntry::ThinkingLevelChange { id, thinking_level } => {
                Entry::ThinkingLevelChange(ThinkingLevelEntry {
                    id,
                    seq,
                    parent_id,
                    timestamp,
                    thinking_level,
                })
            }
            ProvisionedEntry::ActiveToolsChange {
                id,
                active_tool_names,
            } => Entry::ActiveToolsChange(ActiveToolsEntry {
                id,
                seq,
                parent_id,
                timestamp,
                active_tool_names,
            }),
            ProvisionedEntry::Compaction {
                id,
                summary,
                retained_tail,
                tokens_before,
                details,
                usage,
            } => Entry::Compaction(CompactionEntry {
                id,
                seq,
                parent_id,
                timestamp,
                summary,
                retained_tail,
                tokens_before,
                details,
                usage,
            }),
            ProvisionedEntry::BranchSummary {
                id,
                from_id,
                summary,
                details,
                usage,
            } => Entry::BranchSummary(BranchSummaryEntry {
                id,
                seq,
                parent_id,
                timestamp,
                from_id,
                summary,
                details,
                usage,
            }),
            ProvisionedEntry::Custom {
                id,
                custom_type,
                data,
            } => Entry::Custom(CustomEntry {
                id,
                seq,
                parent_id,
                timestamp,
                custom_type,
                data,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Records
// ---------------------------------------------------------------------------

/// Why compaction summary generation started.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionReason {
    Manual,
    Threshold,
    Overflow,
}

/// Outcome of a finished operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationOutcome {
    Completed,
    Aborted,
    Failed,
    Declined,
}

/// Which step of an operation an attempt belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepKind {
    Assistant,
    Compaction,
    BranchSummary,
}

/// Whether a tool invocation may be replayed during recovery.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayPolicy {
    Never,
    Safe,
}

/// Which queue a `queue_enqueued` record targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum QueueKind {
    Steer,
    FollowUp,
    NextRun,
}

/// The intent of a started operation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OperationIntent {
    #[serde(rename_all = "camelCase")]
    Run {
        original_prompt: Vec<AgentMessage>,
        initial_messages: Vec<ProvisionedEntry>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        system_prompt_override: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        resume_data: Option<HashMap<String, JsonValue>>,
    },
    #[serde(rename_all = "camelCase")]
    Compaction {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        custom_instructions: Option<String>,
        result_entry_id: String,
    },
    #[serde(rename_all = "camelCase")]
    Navigation {
        target_id: Option<String>,
        summarize: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        custom_instructions: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        summary_entry_id: Option<String>,
    },
}

/// A durable operation-start record. Its id is the operation's run id.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OperationStartedRecord {
    pub id: String,
    pub seq: u64,
    pub lane: String,
    pub timestamp: u64,
    pub source_leaf_id: Option<String>,
    pub intent: OperationIntent,
}

/// A record requesting abortion of a run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AbortRequestedRecord {
    pub id: String,
    pub seq: u64,
    pub lane: String,
    pub timestamp: u64,
    pub run_id: String,
}

/// A durable operation-finish record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OperationFinishedRecord {
    pub id: String,
    pub seq: u64,
    pub lane: String,
    pub timestamp: u64,
    pub run_id: String,
    pub outcome: OperationOutcome,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<OperationError>,
}

/// The error payload of a failed operation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OperationError {
    pub code: String,
    pub message: String,
}

/// A step attempt record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StepAttemptRecord {
    pub id: String,
    pub seq: u64,
    pub lane: String,
    pub timestamp: u64,
    pub run_id: String,
    pub step: StepKind,
    pub attempt: u32,
    pub result_entry_id: String,
    /// Required when `step` is `Compaction`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compaction_reason: Option<CompactionReason>,
}

impl StepAttemptRecord {
    /// The TypeScript source encodes the compaction reason requirement in the
    /// type system; validate it on the wire.
    pub fn is_valid(&self) -> bool {
        match self.step {
            StepKind::Compaction => self.compaction_reason.is_some(),
            _ => self.compaction_reason.is_none(),
        }
    }
}

/// A tool-start record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolStartedRecord {
    pub id: String,
    pub seq: u64,
    pub lane: String,
    pub timestamp: u64,
    pub run_id: String,
    pub assistant_entry_id: String,
    pub tool_index: usize,
    pub tool_call_id: String,
    pub tool_name: String,
    pub effective_args: serde_json::Map<String, JsonValue>,
    pub result_entry_id: String,
    pub replay: ReplayPolicy,
}

/// A queue-enqueue record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QueueEnqueuedRecord {
    pub id: String,
    pub seq: u64,
    pub lane: String,
    pub timestamp: u64,
    pub queue: QueueKind,
    /// Required for `steer`/`followUp`; absent for `nextRun`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    pub target: ProvisionedEntry,
}

/// A queue-cancellation record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QueueCancelledRecord {
    pub id: String,
    pub seq: u64,
    pub lane: String,
    pub timestamp: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    pub entry_id: String,
}

/// A deferred-write record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WriteDeferredRecord {
    pub id: String,
    pub seq: u64,
    pub lane: String,
    pub timestamp: u64,
    pub run_id: String,
    pub target: ProvisionedEntry,
}

/// The cause of a usage record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "cause", rename_all = "snake_case")]
pub enum UsageCause {
    #[serde(rename_all = "camelCase")]
    Assistant {
        run_id: String,
        entry_id: String,
        attempt: u32,
        stop_reason: SessionStopReason,
    },
    #[serde(rename_all = "camelCase")]
    Compaction {
        run_id: String,
        entry_id: String,
        attempt: u32,
        stop_reason: SessionStopReason,
    },
    #[serde(rename_all = "camelCase")]
    BranchSummary {
        run_id: String,
        entry_id: String,
        attempt: u32,
        stop_reason: SessionStopReason,
    },
    #[serde(rename_all = "camelCase")]
    DeferredFetch {
        run_id: String,
        entry_id: String,
        attempt: u32,
        stop_reason: SessionStopReason,
    },
    #[serde(rename_all = "camelCase")]
    Tool {
        run_id: String,
        entry_id: String,
        tool_call_id: String,
    },
    #[serde(rename_all = "camelCase")]
    Hook { run_id: String, entry_id: String },
    #[serde(rename_all = "camelCase")]
    Adjustment {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        run_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        entry_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        details: Option<JsonValue>,
    },
}

/// A usage accounting record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageRecord {
    pub id: String,
    pub seq: u64,
    pub lane: String,
    pub timestamp: u64,
    pub usage: Usage,
    #[serde(flatten)]
    pub cause: UsageCause,
}

/// A durable lane record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LaneRecord {
    OperationStarted(OperationStartedRecord),
    AbortRequested(AbortRequestedRecord),
    OperationFinished(OperationFinishedRecord),
    StepAttempt(StepAttemptRecord),
    ToolStarted(ToolStartedRecord),
    QueueEnqueued(QueueEnqueuedRecord),
    QueueCancelled(QueueCancelledRecord),
    WriteDeferred(WriteDeferredRecord),
    Usage(UsageRecord),
}

impl LaneRecord {
    pub fn id(&self) -> &str {
        match self {
            LaneRecord::OperationStarted(record) => &record.id,
            LaneRecord::AbortRequested(record) => &record.id,
            LaneRecord::OperationFinished(record) => &record.id,
            LaneRecord::StepAttempt(record) => &record.id,
            LaneRecord::ToolStarted(record) => &record.id,
            LaneRecord::QueueEnqueued(record) => &record.id,
            LaneRecord::QueueCancelled(record) => &record.id,
            LaneRecord::WriteDeferred(record) => &record.id,
            LaneRecord::Usage(record) => &record.id,
        }
    }

    pub fn seq(&self) -> u64 {
        match self {
            LaneRecord::OperationStarted(record) => record.seq,
            LaneRecord::AbortRequested(record) => record.seq,
            LaneRecord::OperationFinished(record) => record.seq,
            LaneRecord::StepAttempt(record) => record.seq,
            LaneRecord::ToolStarted(record) => record.seq,
            LaneRecord::QueueEnqueued(record) => record.seq,
            LaneRecord::QueueCancelled(record) => record.seq,
            LaneRecord::WriteDeferred(record) => record.seq,
            LaneRecord::Usage(record) => record.seq,
        }
    }

    pub fn lane(&self) -> &str {
        match self {
            LaneRecord::OperationStarted(record) => &record.lane,
            LaneRecord::AbortRequested(record) => &record.lane,
            LaneRecord::OperationFinished(record) => &record.lane,
            LaneRecord::StepAttempt(record) => &record.lane,
            LaneRecord::ToolStarted(record) => &record.lane,
            LaneRecord::QueueEnqueued(record) => &record.lane,
            LaneRecord::QueueCancelled(record) => &record.lane,
            LaneRecord::WriteDeferred(record) => &record.lane,
            LaneRecord::Usage(record) => &record.lane,
        }
    }

    /// The run id this record belongs to, when it is operation-owned.
    pub fn run_id(&self) -> Option<&str> {
        match self {
            // A started operation is identified by its own record id.
            LaneRecord::OperationStarted(record) => Some(&record.id),
            LaneRecord::AbortRequested(record) => Some(&record.run_id),
            LaneRecord::OperationFinished(record) => Some(&record.run_id),
            LaneRecord::StepAttempt(record) => Some(&record.run_id),
            LaneRecord::ToolStarted(record) => Some(&record.run_id),
            LaneRecord::QueueEnqueued(record) => record.run_id.as_deref(),
            LaneRecord::QueueCancelled(record) => record.run_id.as_deref(),
            LaneRecord::WriteDeferred(record) => Some(&record.run_id),
            LaneRecord::Usage(record) => match &record.cause {
                UsageCause::Assistant { run_id, .. }
                | UsageCause::Compaction { run_id, .. }
                | UsageCause::BranchSummary { run_id, .. }
                | UsageCause::DeferredFetch { run_id, .. }
                | UsageCause::Tool { run_id, .. }
                | UsageCause::Hook { run_id, .. } => Some(run_id),
                UsageCause::Adjustment { run_id, .. } => run_id.as_deref(),
            },
        }
    }

    /// Wire type discriminant of this record.
    pub fn record_type(&self) -> &'static str {
        match self {
            LaneRecord::OperationStarted(_) => "operation_started",
            LaneRecord::AbortRequested(_) => "abort_requested",
            LaneRecord::OperationFinished(_) => "operation_finished",
            LaneRecord::StepAttempt(_) => "step_attempt",
            LaneRecord::ToolStarted(_) => "tool_started",
            LaneRecord::QueueEnqueued(_) => "queue_enqueued",
            LaneRecord::QueueCancelled(_) => "queue_cancelled",
            LaneRecord::WriteDeferred(_) => "write_deferred",
            LaneRecord::Usage(_) => "usage",
        }
    }

    /// Known record type discriminants, in wire order.
    pub const RECORD_TYPES: [&'static str; 9] = [
        "operation_started",
        "abort_requested",
        "operation_finished",
        "step_attempt",
        "tool_started",
        "queue_enqueued",
        "queue_cancelled",
        "write_deferred",
        "usage",
    ];
}

/// The write-side shape of a record: storage-assigned fields are omitted.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum NewRecord {
    #[serde(rename_all = "camelCase")]
    OperationStarted {
        id: String,
        lane: String,
        source_leaf_id: Option<String>,
        intent: OperationIntent,
    },
    #[serde(rename_all = "camelCase")]
    AbortRequested {
        id: String,
        lane: String,
        run_id: String,
    },
    #[serde(rename_all = "camelCase")]
    OperationFinished {
        id: String,
        lane: String,
        run_id: String,
        outcome: OperationOutcome,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<OperationError>,
    },
    #[serde(rename_all = "camelCase")]
    StepAttempt {
        id: String,
        lane: String,
        run_id: String,
        step: StepKind,
        attempt: u32,
        result_entry_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        compaction_reason: Option<CompactionReason>,
    },
    #[serde(rename_all = "camelCase")]
    ToolStarted {
        id: String,
        lane: String,
        run_id: String,
        assistant_entry_id: String,
        tool_index: usize,
        tool_call_id: String,
        tool_name: String,
        effective_args: serde_json::Map<String, JsonValue>,
        result_entry_id: String,
        replay: ReplayPolicy,
    },
    #[serde(rename_all = "camelCase")]
    QueueEnqueued {
        id: String,
        lane: String,
        queue: QueueKind,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        run_id: Option<String>,
        target: ProvisionedEntry,
    },
    #[serde(rename_all = "camelCase")]
    QueueCancelled {
        id: String,
        lane: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        run_id: Option<String>,
        entry_id: String,
    },
    #[serde(rename_all = "camelCase")]
    WriteDeferred {
        id: String,
        lane: String,
        run_id: String,
        target: ProvisionedEntry,
    },
    #[serde(rename_all = "camelCase")]
    Usage {
        id: String,
        lane: String,
        usage: Usage,
        #[serde(flatten)]
        cause: UsageCause,
    },
}

impl NewRecord {
    pub fn id(&self) -> &str {
        match self {
            NewRecord::OperationStarted { id, .. }
            | NewRecord::AbortRequested { id, .. }
            | NewRecord::OperationFinished { id, .. }
            | NewRecord::StepAttempt { id, .. }
            | NewRecord::ToolStarted { id, .. }
            | NewRecord::QueueEnqueued { id, .. }
            | NewRecord::QueueCancelled { id, .. }
            | NewRecord::WriteDeferred { id, .. }
            | NewRecord::Usage { id, .. } => id,
        }
    }

    pub fn lane(&self) -> &str {
        match self {
            NewRecord::OperationStarted { lane, .. }
            | NewRecord::AbortRequested { lane, .. }
            | NewRecord::OperationFinished { lane, .. }
            | NewRecord::StepAttempt { lane, .. }
            | NewRecord::ToolStarted { lane, .. }
            | NewRecord::QueueEnqueued { lane, .. }
            | NewRecord::QueueCancelled { lane, .. }
            | NewRecord::WriteDeferred { lane, .. }
            | NewRecord::Usage { lane, .. } => lane,
        }
    }

    /// Fill in the storage-assigned fields, producing a full record.
    pub fn into_record(self, seq: u64, timestamp: u64) -> LaneRecord {
        match self {
            NewRecord::OperationStarted {
                id,
                lane,
                source_leaf_id,
                intent,
            } => LaneRecord::OperationStarted(OperationStartedRecord {
                id,
                seq,
                lane,
                timestamp,
                source_leaf_id,
                intent,
            }),
            NewRecord::AbortRequested { id, lane, run_id } => {
                LaneRecord::AbortRequested(AbortRequestedRecord {
                    id,
                    seq,
                    lane,
                    timestamp,
                    run_id,
                })
            }
            NewRecord::OperationFinished {
                id,
                lane,
                run_id,
                outcome,
                error,
            } => LaneRecord::OperationFinished(OperationFinishedRecord {
                id,
                seq,
                lane,
                timestamp,
                run_id,
                outcome,
                error,
            }),
            NewRecord::StepAttempt {
                id,
                lane,
                run_id,
                step,
                attempt,
                result_entry_id,
                compaction_reason,
            } => LaneRecord::StepAttempt(StepAttemptRecord {
                id,
                seq,
                lane,
                timestamp,
                run_id,
                step,
                attempt,
                result_entry_id,
                compaction_reason,
            }),
            NewRecord::ToolStarted {
                id,
                lane,
                run_id,
                assistant_entry_id,
                tool_index,
                tool_call_id,
                tool_name,
                effective_args,
                result_entry_id,
                replay,
            } => LaneRecord::ToolStarted(ToolStartedRecord {
                id,
                seq,
                lane,
                timestamp,
                run_id,
                assistant_entry_id,
                tool_index,
                tool_call_id,
                tool_name,
                effective_args,
                result_entry_id,
                replay,
            }),
            NewRecord::QueueEnqueued {
                id,
                lane,
                queue,
                run_id,
                target,
            } => LaneRecord::QueueEnqueued(QueueEnqueuedRecord {
                id,
                seq,
                lane,
                timestamp,
                queue,
                run_id,
                target,
            }),
            NewRecord::QueueCancelled {
                id,
                lane,
                run_id,
                entry_id,
            } => LaneRecord::QueueCancelled(QueueCancelledRecord {
                id,
                seq,
                lane,
                timestamp,
                run_id,
                entry_id,
            }),
            NewRecord::WriteDeferred {
                id,
                lane,
                run_id,
                target,
            } => LaneRecord::WriteDeferred(WriteDeferredRecord {
                id,
                seq,
                lane,
                timestamp,
                run_id,
                target,
            }),
            NewRecord::Usage {
                id,
                lane,
                usage,
                cause,
            } => LaneRecord::Usage(UsageRecord {
                id,
                seq,
                lane,
                timestamp,
                usage,
                cause,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

/// Query result order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum EntryOrder {
    #[default]
    NewestFirst,
    OldestFirst,
}

/// Pagination cursor for entry queries.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EntryCursor {
    pub after_seq: u64,
}

/// Entry query.
#[derive(Clone, Debug, Default)]
pub struct EntryQuery {
    pub entry_type: Option<String>,
    pub custom_type: Option<String>,
    pub order: Option<EntryOrder>,
    pub limit: Option<usize>,
    pub cursor: Option<EntryCursor>,
}

/// Bounds of a branch scan.
#[derive(Clone, Debug, Default)]
pub struct BranchBounds {
    /// Default: the view's lane leaf.
    pub start: Option<String>,
    /// Scan ends after the first match, inclusive.
    pub stop_at_type: Option<String>,
    pub stop_at_id: Option<String>,
}

/// A branch entry query: bounds plus an explicit start.
#[derive(Clone, Debug, Default)]
pub struct BranchEntryQuery {
    pub bounds: BranchBounds,
    pub entry_type: Option<String>,
    pub custom_type: Option<String>,
    pub order: Option<EntryOrder>,
    pub limit: Option<usize>,
    pub cursor: Option<EntryCursor>,
}

/// Record query.
#[derive(Clone, Debug, Default)]
pub struct RecordQuery {
    pub lane: Option<String>,
    pub record_type: Option<String>,
    pub run_id: Option<String>,
    pub operation_kind: Option<String>,
    pub after_seq: Option<u64>,
    pub order: Option<EntryOrder>,
    pub limit: Option<usize>,
}

/// Options for log queries.
#[derive(Clone, Debug, Default)]
pub struct LogOptions {
    pub after_seq: Option<u64>,
    pub limit: Option<usize>,
}

/// Session metadata.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionMetadata {
    pub id: String,
    /// Unix timestamp in milliseconds.
    pub created_at: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_session_id: Option<String>,
}

/// Session-wide counters replayed from usage records.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionStats {
    pub message_count: u64,
    pub cached_tokens: u64,
    pub uncached_tokens: u64,
    pub total_tokens: u64,
    pub cost_total: f64,
}

/// A lane and its current leaf.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LanePointer {
    pub lane: String,
    pub leaf_id: Option<String>,
}

/// One item of the durable mutation log.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum LogItem {
    Entry {
        seq: u64,
        entry: Entry,
    },
    Record {
        seq: u64,
        record: LaneRecord,
    },
    Lane {
        seq: u64,
        lane: String,
        leaf_id: Option<String>,
    },
    Fact {
        seq: u64,
        fact: FactName,
        name: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        project: Option<String>,
    },
    Label {
        seq: u64,
        target_id: String,
        label: Option<String>,
    },
}

/// Which global fact a log item sets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactName {
    Name,
    Label,
    Project,
}

/// Options for creating a session.
#[derive(Clone, Debug, Default)]
pub struct SessionCreateOptions {
    pub id: Option<String>,
    pub parent_session_id: Option<String>,
}

/// Fork scope and target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ForkOptions {
    /// Fork a single branch (the default).
    Branch {
        entry_id: Option<String>,
        position: Option<ForkPosition>,
    },
    /// Fork the whole tree: every entry, lane, and fact.
    Tree,
}

impl Default for ForkOptions {
    fn default() -> Self {
        ForkOptions::Branch {
            entry_id: None,
            position: None,
        }
    }
}

/// Where relative to the target entry a branch fork cuts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForkPosition {
    Before,
    At,
}

/// Session error codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionErrorCode {
    Aborted,
    NotFound,
    AlreadyExists,
    InvalidEntry,
    InvalidPayload,
    InvalidLane,
    InvalidQuery,
    InvalidForkTarget,
    Storage,
}

impl SessionErrorCode {
    pub fn as_str(&self) -> &'static str {
        match self {
            SessionErrorCode::Aborted => "aborted",
            SessionErrorCode::NotFound => "not_found",
            SessionErrorCode::AlreadyExists => "already_exists",
            SessionErrorCode::InvalidEntry => "invalid_entry",
            SessionErrorCode::InvalidPayload => "invalid_payload",
            SessionErrorCode::InvalidLane => "invalid_lane",
            SessionErrorCode::InvalidQuery => "invalid_query",
            SessionErrorCode::InvalidForkTarget => "invalid_fork_target",
            SessionErrorCode::Storage => "storage",
        }
    }
}

/// Error returned by session storage and repository operations.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct SessionError {
    pub code: SessionErrorCode,
    pub message: String,
    #[source]
    pub cause: Option<Arc<dyn std::error::Error + Send + Sync>>,
}

impl SessionError {
    pub fn new(code: SessionErrorCode, message: impl Into<String>) -> Self {
        SessionError {
            code,
            message: message.into(),
            cause: None,
        }
    }

    pub fn with_cause(
        code: SessionErrorCode,
        message: impl Into<String>,
        cause: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        SessionError {
            code,
            message: message.into(),
            cause: Some(Arc::new(cause)),
        }
    }
}

/// Validate a query limit: positive integers only.
pub fn assert_valid_limit(limit: Option<usize>) -> Result<(), SessionError> {
    if limit == Some(0) {
        return Err(SessionError::new(
            SessionErrorCode::InvalidQuery,
            "limit must be a positive integer",
        ));
    }
    Ok(())
}

/// Validate that `operation_kind` is only used with operation-start queries.
pub fn assert_valid_operation_kind(
    record_type: Option<&str>,
    operation_kind: Option<&str>,
) -> Result<(), SessionError> {
    if operation_kind.is_some() && record_type != Some("operation_started") {
        return Err(SessionError::new(
            SessionErrorCode::InvalidQuery,
            "operationKind requires type \"operation_started\"",
        ));
    }
    Ok(())
}

/// Validate that a durable payload is JSON-clean.
///
/// The TypeScript source walks the value looking for cycles, bigints, sparse
/// arrays, accessors, and non-finite numbers. Rust values are already typed
/// trees over [`serde_json::Value`], so only non-finite floats need a check
/// (and `serde_json` refuses to construct them); the walk stays as a
/// defensive guard for future opaque payloads.
pub fn assert_json_serializable(value: &JsonValue) -> Result<(), SessionError> {
    fn walk(value: &JsonValue) -> Result<(), SessionError> {
        match value {
            JsonValue::Number(number) => {
                if let Some(float) = number.as_f64()
                    && !float.is_finite()
                {
                    return Err(SessionError::new(
                        SessionErrorCode::InvalidPayload,
                        "Durable payload contains a non-finite number",
                    ));
                }
                Ok(())
            }
            JsonValue::Array(items) => {
                for item in items {
                    walk(item)?;
                }
                Ok(())
            }
            JsonValue::Object(entries) => {
                for item in entries.values() {
                    walk(item)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
    walk(value)
}
