//! Pure lane-state recovery: validate a lane's durable recovery slice, then
//! reduce it back into orchestration state.
//!
//! Ported from `@step-harness/agent-core` `src\harness\reducer.ts`. Everything
//! here is a pure function over data the caller already read from storage: no
//! IO, no clocks, no mutation of the inputs.
//!
//! # Notable differences from the TypeScript source
//!
//! - Corruption is a returned [`RecordLogCorruption`] value, not a thrown
//!   exception, so every validation helper returns `Result`.
//! - [`EffectiveLaneConfiguration`] models `model` and `active_tool_names` as
//!   optional: the TypeScript source always carries values because its harness
//!   defaults do, while "not persisted yet" is a distinct state here.
//!   `thinking_level` is a `String` for the same reason
//!   [`crate::session::context::SessionContext`] keeps it one: the persisted
//!   vocabulary is open and the TypeScript source only casts it.
//! - [`matches_provisioned_entry`] compares JSON representations instead of
//!   destructuring the three storage-assigned fields; see its doc comment.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use step_harness_providers::types::{AssistantMessage, DeferredHandle, StopReason, ToolCall};

use crate::session::context::SessionModel;
use crate::session::types::{
    CompactionReason, Entry, LaneRecord, OperationIntent, OperationStartedRecord, ProvisionedEntry,
    QueueEnqueuedRecord, QueueKind, StepAttemptRecord, StepKind, ToolStartedRecord, UsageCause,
};
use crate::types::AgentMessage;

/// Machine-readable category for a contradiction in a lane's durable recovery
/// slice. These indicate states the single-writer record protocol cannot
/// produce, not ordinary operation failures or incomplete-but-recoverable
/// intent/result prefixes. Restore must reject such states rather than repair or
/// continue it; the accompanying error message supplies human-readable detail.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecordLogCorruptionReason {
    MultipleOpenOperations,
    UnknownOperation,
    RecordAfterFinish,
    NonConsecutiveAttempt,
    InvalidCompactionReason,
    QueueAfterAbort,
    InvalidQueueCancellation,
    InconsistentStep,
    ToolCallMismatch,
    DuplicateToolInvocation,
    ProvisionedEntryMismatch,
    InvalidDeferredHandle,
}

impl RecordLogCorruptionReason {
    /// Wire name of this reason, matching the TypeScript string union.
    pub fn as_str(&self) -> &'static str {
        match self {
            RecordLogCorruptionReason::MultipleOpenOperations => "multipleOpenOperations",
            RecordLogCorruptionReason::UnknownOperation => "unknownOperation",
            RecordLogCorruptionReason::RecordAfterFinish => "recordAfterFinish",
            RecordLogCorruptionReason::NonConsecutiveAttempt => "nonConsecutiveAttempt",
            RecordLogCorruptionReason::InvalidCompactionReason => "invalidCompactionReason",
            RecordLogCorruptionReason::QueueAfterAbort => "queueAfterAbort",
            RecordLogCorruptionReason::InvalidQueueCancellation => "invalidQueueCancellation",
            RecordLogCorruptionReason::InconsistentStep => "inconsistentStep",
            RecordLogCorruptionReason::ToolCallMismatch => "toolCallMismatch",
            RecordLogCorruptionReason::DuplicateToolInvocation => "duplicateToolInvocation",
            RecordLogCorruptionReason::ProvisionedEntryMismatch => "provisionedEntryMismatch",
            RecordLogCorruptionReason::InvalidDeferredHandle => "invalidDeferredHandle",
        }
    }
}

impl std::fmt::Display for RecordLogCorruptionReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A lane recovery slice that contradicts the single-writer record protocol.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{reason}: {message}")]
pub struct RecordLogCorruption {
    pub reason: RecordLogCorruptionReason,
    pub message: String,
}

impl RecordLogCorruption {
    fn new(reason: RecordLogCorruptionReason, message: String) -> Self {
        RecordLogCorruption { reason, message }
    }
}

/// The bounded slice of durable state a lane's restore reads.
#[derive(Clone, Debug, PartialEq)]
pub struct RecordLogSlice {
    pub lane: String,
    pub open_operations: Vec<OperationStartedRecord>,
    pub records: Vec<LaneRecord>,
    /// Operation-owned entries plus entries fetched directly by provisioned or
    /// referenced ids.
    pub entries: Vec<Entry>,
}

/// Effective lane configuration derived from persisted entries, with harness
/// option fallbacks for values that were never persisted.
///
/// Not serializable: [`SessionModel`] is a plain in-memory type and the
/// configuration is recovery state, never a durable payload.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EffectiveLaneConfiguration {
    pub model: Option<SessionModel>,
    pub thinking_level: String,
    pub active_tool_names: Option<Vec<String>>,
}

/// Which part of an operation produced a terminal assistant failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureSource {
    Step,
    DeferredFetch,
}

/// A terminal assistant failure the lane ended on.
#[derive(Clone, Debug)]
pub struct TerminalFailureState {
    pub entry_id: String,
    pub source: FailureSource,
    pub message: AssistantMessage,
}

/// Recovery state of one tool call inside a tool batch.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCallState {
    pub tool_index: usize,
    pub tool_call: ToolCall,
    pub started: Option<ToolStartedRecord>,
    pub result_exists: bool,
    pub terminate: Option<bool>,
}

/// Recovery state of the newest assistant tool batch.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolBatchState {
    pub assistant_entry_id: String,
    pub calls: Vec<ToolCallState>,
    /// The assistant message hit the length stop reason.
    pub truncated: bool,
    /// At least one call has no result yet.
    pub unresolved: bool,
}

/// Which kind of durable work an operation performs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationKind {
    Run,
    Compaction,
    Navigation,
}

impl OperationKind {
    /// Wire name of this kind, matching the TypeScript string union.
    pub fn as_str(&self) -> &'static str {
        match self {
            OperationKind::Run => "run",
            OperationKind::Compaction => "compaction",
            OperationKind::Navigation => "navigation",
        }
    }
}

/// The intent kind of a started operation.
///
/// The TypeScript `LaneState.operation.kind` duplicates the intent's tag; the
/// Rust port derives it so the two cannot drift.
fn intent_kind(intent: &OperationIntent) -> OperationKind {
    match intent {
        OperationIntent::Run { .. } => OperationKind::Run,
        OperationIntent::Compaction { .. } => OperationKind::Compaction,
        OperationIntent::Navigation { .. } => OperationKind::Navigation,
    }
}

/// Recovery state of the in-flight durable step of an operation.
#[derive(Clone, Debug, PartialEq)]
pub struct StepState {
    pub kind: StepKind,
    pub attempts: u32,
    pub result_entry_id: String,
    pub compaction_reason: Option<CompactionReason>,
}

/// Summary of the newest entry the open operation appended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NewestOwn {
    pub entry_id: String,
    /// Wire entry type; `type` is a Rust keyword.
    pub entry_type: String,
    pub role: Option<String>,
    pub stop_reason: Option<StopReason>,
}

/// Whether an operation's declared result targets already exist.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OperationTargets {
    pub result: Option<bool>,
    pub summary: Option<bool>,
}

/// Recovery state of the lane's open operation.
#[derive(Clone, Debug)]
pub struct OperationState {
    pub id: String,
    pub kind: OperationKind,
    pub intent: OperationIntent,
    pub aborting: bool,
    pub step: Option<StepState>,
    pub tool_batch: Option<ToolBatchState>,
    pub missing_initial_messages: Vec<ProvisionedEntry>,
    pub pending_steer: Vec<ProvisionedEntry>,
    pub pending_follow_up: Vec<ProvisionedEntry>,
    pub pending_writes: Vec<ProvisionedEntry>,
    pub deferred: Option<DeferredHandle>,
    pub overflow_recovery_used: bool,
    pub newest_own: Option<NewestOwn>,
    pub targets: OperationTargets,
}

/// One lane's reconstructed orchestration state.
#[derive(Clone, Debug)]
pub struct LaneState {
    pub lane: String,
    pub leaf_id: Option<String>,
    pub operation: Option<OperationState>,
    pub pending_next_run: Vec<ProvisionedEntry>,
}

/// Everything [`reduce_lane_state`] needs: the recovery slice plus the bounded
/// effective-state lookups and harness fallbacks.
#[derive(Clone, Debug)]
pub struct LaneReductionInput {
    pub lane: String,
    pub open_operations: Vec<OperationStartedRecord>,
    pub records: Vec<LaneRecord>,
    pub entries: Vec<Entry>,
    pub leaf_id: Option<String>,
    /// Entries appended by the open operation, oldest first. Empty when idle.
    pub own_entries: Vec<Entry>,
    /// Bounded effective-state lookups at the operation anchor or idle leaf,
    /// oldest first.
    pub configuration_entries: Vec<Entry>,
    /// Harness option fallbacks used when no persisted value exists.
    pub defaults: EffectiveLaneConfiguration,
}

/// The result of reducing one lane.
#[derive(Clone, Debug)]
pub struct LaneReductionResult {
    pub lane_state: LaneState,
    pub effective_configuration: EffectiveLaneConfiguration,
    pub terminal_failure: Option<TerminalFailureState>,
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Validates a bounded lane recovery slice without reading or mutating session
/// state.
pub fn validate_record_log(input: &RecordLogSlice) -> Result<(), RecordLogCorruption> {
    validate_slice(
        &input.lane,
        &input.open_operations,
        &input.records,
        &input.entries,
    )
}

fn validate_slice(
    lane: &str,
    open_operations: &[OperationStartedRecord],
    records: &[LaneRecord],
    entries: &[Entry],
) -> Result<(), RecordLogCorruption> {
    if open_operations.len() > 1 {
        return Err(RecordLogCorruption::new(
            RecordLogCorruptionReason::MultipleOpenOperations,
            format!("Lane {lane} has at least two open operations"),
        ));
    }

    let entries_by_id = index_entries(entries);
    validate_deferred_handles(entries_by_id.values().copied())?;
    let mut starts: HashMap<&str, &OperationStartedRecord> = HashMap::new();
    let mut finished_at: HashMap<&str, u64> = HashMap::new();
    let mut aborted_at: HashMap<&str, u64> = HashMap::new();
    let mut queue_enqueues: HashMap<&str, &QueueEnqueuedRecord> = HashMap::new();
    let mut latest_attempt: HashMap<&str, &StepAttemptRecord> = HashMap::new();
    let mut tool_invocations: HashSet<String> = HashSet::new();
    let mut records: Vec<&LaneRecord> = records.iter().collect();
    records.sort_by_key(|record| record.seq());

    for record in records {
        if let LaneRecord::OperationStarted(started) = record {
            starts.insert(&started.id, started);
            validate_operation_result(&entries_by_id, started)?;
            continue;
        }

        if let Some(run_id) = record.run_id() {
            if !starts.contains_key(run_id) {
                return Err(RecordLogCorruption::new(
                    RecordLogCorruptionReason::UnknownOperation,
                    format!(
                        "Record {} references unknown operation {run_id}",
                        record.id()
                    ),
                ));
            }
            if let Some(finish_seq) = finished_at.get(run_id)
                && record.seq() > *finish_seq
            {
                return Err(RecordLogCorruption::new(
                    RecordLogCorruptionReason::RecordAfterFinish,
                    format!(
                        "Record {} follows the finish of operation {run_id}",
                        record.id()
                    ),
                ));
            }
        }

        match record {
            LaneRecord::OperationFinished(finished) => {
                finished_at.insert(&finished.run_id, finished.seq);
            }
            LaneRecord::AbortRequested(abort) => {
                aborted_at.insert(&abort.run_id, abort.seq);
            }
            LaneRecord::StepAttempt(attempt) => {
                validate_attempt_reason(attempt)?;
                let previous = latest_attempt.get(attempt.run_id.as_str()).copied();
                validate_attempt_sequence(attempt, previous, &entries_by_id)?;
                validate_attempt_result(&entries_by_id, attempt)?;
                latest_attempt.insert(&attempt.run_id, attempt);
            }
            LaneRecord::ToolStarted(started) => {
                validate_tool_start(started, &entries_by_id, &mut tool_invocations)?
            }
            LaneRecord::QueueEnqueued(enqueued) => {
                let aborted_seq = enqueued
                    .run_id
                    .as_deref()
                    .and_then(|run_id| aborted_at.get(run_id).copied());
                if enqueued.queue != QueueKind::NextRun
                    && aborted_seq.is_some_and(|aborted_seq| enqueued.seq > aborted_seq)
                {
                    return Err(RecordLogCorruption::new(
                        RecordLogCorruptionReason::QueueAfterAbort,
                        format!(
                            "{} item {} was enqueued after abort",
                            queue_kind_str(enqueued.queue),
                            enqueued.target.id()
                        ),
                    ));
                }
                queue_enqueues.insert(enqueued.target.id(), enqueued);
                validate_exact_provisioned_entry(&entries_by_id, &enqueued.target)?;
            }
            LaneRecord::QueueCancelled(cancelled) => {
                let matching =
                    queue_enqueues
                        .get(cancelled.entry_id.as_str())
                        .is_some_and(|enqueue| {
                            enqueue.seq < cancelled.seq
                                && enqueue.run_id.as_deref() == cancelled.run_id.as_deref()
                                && !entries_by_id.contains_key(cancelled.entry_id.as_str())
                        });
                if !matching {
                    return Err(RecordLogCorruption::new(
                        RecordLogCorruptionReason::InvalidQueueCancellation,
                        format!(
                            "Queue cancellation {} has no pending matching enqueue",
                            cancelled.id
                        ),
                    ));
                }
            }
            LaneRecord::WriteDeferred(deferred) => {
                validate_exact_provisioned_entry(&entries_by_id, &deferred.target)?
            }
            LaneRecord::OperationStarted(_) | LaneRecord::Usage(_) => {}
        }
    }
    Ok(())
}

fn index_entries(entries: &[Entry]) -> HashMap<&str, &Entry> {
    let mut by_id: HashMap<&str, &Entry> = HashMap::new();
    for entry in entries {
        by_id.insert(entry.id(), entry);
    }
    by_id
}

fn validate_deferred_handles<'a>(
    entries: impl Iterator<Item = &'a Entry>,
) -> Result<(), RecordLogCorruption> {
    for entry in entries {
        if let Entry::Message(message) = entry
            && let AgentMessage::Assistant(assistant) = &message.message
            && assistant.stop_reason == StopReason::Deferred
            && assistant.deferred.is_none()
        {
            return Err(RecordLogCorruption::new(
                RecordLogCorruptionReason::InvalidDeferredHandle,
                format!(
                    "Deferred assistant entry {} does not carry a handle",
                    message.id
                ),
            ));
        }
    }
    Ok(())
}

/// Deep-compare a stored entry against the provisioned intent it was written
/// from, ignoring the storage-assigned `parentId`, `seq`, and `timestamp`.
///
/// The TypeScript source destructures those three fields away and deep-compares
/// the remaining payload against the `ProvisionedEntry`. Both Rust types are
/// serde-tagged enums over the same field names, so the serialized entry minus
/// the three storage-assigned keys equals the serialized provisioned entry
/// exactly when the payloads match; the comparison is on
/// [`serde_json::Value`], which is key-order independent.
fn matches_provisioned_entry(entry: &Entry, target: &ProvisionedEntry) -> bool {
    let Ok(mut entry_value) = serde_json::to_value(entry) else {
        return false;
    };
    if let Some(object) = entry_value.as_object_mut() {
        object.remove("parentId");
        object.remove("seq");
        object.remove("timestamp");
    }
    serde_json::to_value(target).is_ok_and(|target| target == entry_value)
}

fn validate_exact_provisioned_entry(
    entries_by_id: &HashMap<&str, &Entry>,
    target: &ProvisionedEntry,
) -> Result<(), RecordLogCorruption> {
    if let Some(entry) = entries_by_id.get(target.id())
        && !matches_provisioned_entry(entry, target)
    {
        return Err(RecordLogCorruption::new(
            RecordLogCorruptionReason::ProvisionedEntryMismatch,
            format!(
                "Provisioned entry {} exists with content different from its intent",
                target.id()
            ),
        ));
    }
    Ok(())
}

fn validate_result_entry(
    entries_by_id: &HashMap<&str, &Entry>,
    result_entry_id: &str,
    matches: impl Fn(&Entry) -> bool,
    description: &str,
) -> Result<(), RecordLogCorruption> {
    if let Some(entry) = entries_by_id.get(result_entry_id)
        && !matches(entry)
    {
        return Err(RecordLogCorruption::new(
            RecordLogCorruptionReason::ProvisionedEntryMismatch,
            format!(
                "Provisioned {description} entry {result_entry_id} exists with different content"
            ),
        ));
    }
    Ok(())
}

fn validate_attempt_reason(record: &StepAttemptRecord) -> Result<(), RecordLogCorruption> {
    match record.step {
        StepKind::Compaction => {
            if !matches!(
                record.compaction_reason,
                Some(
                    CompactionReason::Manual
                        | CompactionReason::Threshold
                        | CompactionReason::Overflow
                )
            ) {
                return Err(RecordLogCorruption::new(
                    RecordLogCorruptionReason::InvalidCompactionReason,
                    format!(
                        "Compaction attempt {} has no valid compaction reason",
                        record.id
                    ),
                ));
            }
        }
        step => {
            if record.compaction_reason.is_some() {
                return Err(RecordLogCorruption::new(
                    RecordLogCorruptionReason::InvalidCompactionReason,
                    format!(
                        "{} attempt {} has a compaction reason",
                        step_kind_str(step),
                        record.id
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn validate_attempt_sequence(
    record: &StepAttemptRecord,
    previous: Option<&StepAttemptRecord>,
    entries_by_id: &HashMap<&str, &Entry>,
) -> Result<(), RecordLogCorruption> {
    let previous_result = previous.and_then(|previous| {
        entries_by_id
            .get(previous.result_entry_id.as_str())
            .copied()
    });
    let continues_series = previous.is_some_and(|previous| previous.step == record.step)
        && previous_result.is_none_or(|result| result.seq() >= record.seq);
    let expected_attempt = if continues_series {
        previous.map_or(1, |previous| previous.attempt + 1)
    } else {
        1
    };
    if record.attempt != expected_attempt {
        return Err(RecordLogCorruption::new(
            RecordLogCorruptionReason::NonConsecutiveAttempt,
            format!(
                "{} attempt {} is {}; expected {expected_attempt}",
                step_kind_str(record.step),
                record.id,
                record.attempt
            ),
        ));
    }
    if !continues_series || record.step == StepKind::Assistant {
        return Ok(());
    }
    // `continues_series` implies a previous attempt of the same step kind.
    let previous = previous.expect("continuing attempt series has a previous attempt");
    if record.result_entry_id != previous.result_entry_id {
        return Err(RecordLogCorruption::new(
            RecordLogCorruptionReason::InconsistentStep,
            format!(
                "{} attempts disagree on their result entry id",
                step_kind_str(record.step)
            ),
        ));
    }
    if record.compaction_reason != previous.compaction_reason {
        return Err(RecordLogCorruption::new(
            RecordLogCorruptionReason::InconsistentStep,
            format!(
                "{} attempts disagree on their compaction reason",
                step_kind_str(record.step)
            ),
        ));
    }
    Ok(())
}

fn validate_attempt_result(
    entries_by_id: &HashMap<&str, &Entry>,
    record: &StepAttemptRecord,
) -> Result<(), RecordLogCorruption> {
    match record.step {
        StepKind::Assistant => validate_result_entry(
            entries_by_id,
            &record.result_entry_id,
            |entry| {
                matches!(
                    entry,
                    Entry::Message(message) if matches!(&message.message, AgentMessage::Assistant(_))
                )
            },
            "assistant result",
        ),
        StepKind::Compaction => validate_result_entry(
            entries_by_id,
            &record.result_entry_id,
            |entry| matches!(entry, Entry::Compaction(_)),
            "compaction result",
        ),
        StepKind::BranchSummary => validate_result_entry(
            entries_by_id,
            &record.result_entry_id,
            |entry| matches!(entry, Entry::BranchSummary(_)),
            "branch-summary result",
        ),
    }
}

fn validate_tool_start(
    record: &ToolStartedRecord,
    entries_by_id: &HashMap<&str, &Entry>,
    invocations: &mut HashSet<String>,
) -> Result<(), RecordLogCorruption> {
    let invocation = format!("{}\u{0}{}", record.assistant_entry_id, record.tool_index);
    if !invocations.insert(invocation) {
        return Err(RecordLogCorruption::new(
            RecordLogCorruptionReason::DuplicateToolInvocation,
            format!(
                "Tool invocation {}:{} is duplicated",
                record.assistant_entry_id, record.tool_index
            ),
        ));
    }

    let assistant = match entries_by_id.get(record.assistant_entry_id.as_str()) {
        Some(Entry::Message(message)) => match &message.message {
            AgentMessage::Assistant(assistant) => Some(assistant),
            _ => None,
        },
        _ => None,
    };
    let Some(assistant) = assistant else {
        return Err(RecordLogCorruption::new(
            RecordLogCorruptionReason::ToolCallMismatch,
            format!(
                "Tool start {} does not reference an assistant entry",
                record.id
            ),
        ));
    };
    let tool_calls: Vec<&ToolCall> = assistant
        .content
        .iter()
        .filter_map(|content| content.tool_call())
        .collect();
    match tool_calls.get(record.tool_index) {
        Some(tool_call)
            if tool_call.id == record.tool_call_id && tool_call.name == record.tool_name => {}
        _ => {
            return Err(RecordLogCorruption::new(
                RecordLogCorruptionReason::ToolCallMismatch,
                format!(
                    "Tool start {} does not match its assistant tool-call ordinal",
                    record.id
                ),
            ));
        }
    }

    validate_result_entry(
        entries_by_id,
        &record.result_entry_id,
        |entry| {
            matches!(
                entry,
                Entry::Message(message)
                    if matches!(&message.message, AgentMessage::ToolResult(result)
                        if result.tool_call_id == record.tool_call_id
                            && result.tool_name == record.tool_name)
            )
        },
        "tool result",
    )
}

fn validate_operation_result(
    entries_by_id: &HashMap<&str, &Entry>,
    record: &OperationStartedRecord,
) -> Result<(), RecordLogCorruption> {
    match &record.intent {
        OperationIntent::Run {
            initial_messages, ..
        } => {
            for target in initial_messages {
                validate_exact_provisioned_entry(entries_by_id, target)?;
            }
        }
        OperationIntent::Compaction {
            result_entry_id, ..
        } => validate_result_entry(
            entries_by_id,
            result_entry_id,
            |entry| matches!(entry, Entry::Compaction(_)),
            "manual compaction",
        )?,
        OperationIntent::Navigation {
            summary_entry_id, ..
        } => {
            if let Some(summary_entry_id) = summary_entry_id {
                validate_result_entry(
                    entries_by_id,
                    summary_entry_id,
                    |entry| matches!(entry, Entry::BranchSummary(_)),
                    "navigation summary",
                )?;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Reduction
// ---------------------------------------------------------------------------

/// Purely reconstructs one lane's orchestration state from its bounded recovery
/// inputs.
pub fn reduce_lane_state(
    input: &LaneReductionInput,
) -> Result<LaneReductionResult, RecordLogCorruption> {
    validate_slice(
        &input.lane,
        &input.open_operations,
        &input.records,
        &input.entries,
    )?;

    let mut records: Vec<&LaneRecord> = input.records.iter().collect();
    records.sort_by_key(|record| record.seq());
    let mut own_entries: Vec<&Entry> = input.own_entries.iter().collect();
    own_entries.sort_by_key(|entry| entry.seq());

    let mut entries_by_id = index_entries(&input.entries);
    for entry in &input.own_entries {
        entries_by_id.insert(entry.id(), entry);
    }

    let cancelled_queue_ids: HashSet<&str> = records
        .iter()
        .filter_map(|record| match record {
            LaneRecord::QueueCancelled(cancelled) => Some(cancelled.entry_id.as_str()),
            _ => None,
        })
        .collect();
    let pending_queue_records: Vec<&QueueEnqueuedRecord> = records
        .iter()
        .filter_map(|record| match record {
            LaneRecord::QueueEnqueued(enqueued)
                if !entries_by_id.contains_key(enqueued.target.id())
                    && !cancelled_queue_ids.contains(enqueued.target.id()) =>
            {
                Some(enqueued)
            }
            _ => None,
        })
        .collect();

    let started = input.open_operations.first();
    let captured_initial_message_ids: HashSet<&str> = match started {
        Some(started) => match &started.intent {
            OperationIntent::Run {
                initial_messages, ..
            } => initial_messages.iter().map(ProvisionedEntry::id).collect(),
            _ => HashSet::new(),
        },
        None => HashSet::new(),
    };
    let pending_next_run: Vec<ProvisionedEntry> = pending_queue_records
        .iter()
        .filter(|enqueued| {
            enqueued.queue == QueueKind::NextRun
                && !captured_initial_message_ids.contains(enqueued.target.id())
        })
        .map(|enqueued| enqueued.target.clone())
        .collect();
    let effective_configuration = derive_effective_configuration(input);

    let Some(started) = started else {
        return Ok(LaneReductionResult {
            lane_state: LaneState {
                lane: input.lane.clone(),
                leaf_id: input.leaf_id.clone(),
                operation: None,
                pending_next_run,
            },
            effective_configuration,
            terminal_failure: None,
        });
    };

    let operation_records: Vec<&LaneRecord> = records
        .iter()
        .filter(|record| match record {
            LaneRecord::OperationStarted(operation) => operation.id == started.id,
            _ => record.run_id() == Some(started.id.as_str()),
        })
        .copied()
        .collect();
    let aborting = operation_records
        .iter()
        .any(|record| matches!(record, LaneRecord::AbortRequested(_)));
    let pending_steer = if aborting {
        Vec::new()
    } else {
        queue_targets(&pending_queue_records, QueueKind::Steer, Some(&started.id))
    };
    let pending_follow_up = if aborting {
        Vec::new()
    } else {
        queue_targets(
            &pending_queue_records,
            QueueKind::FollowUp,
            Some(&started.id),
        )
    };
    let pending_writes: Vec<ProvisionedEntry> = operation_records
        .iter()
        .filter_map(|record| match record {
            LaneRecord::WriteDeferred(deferred)
                if !entries_by_id.contains_key(deferred.target.id()) =>
            {
                Some(deferred.target.clone())
            }
            _ => None,
        })
        .collect();
    let missing_initial_messages: Vec<ProvisionedEntry> = match &started.intent {
        OperationIntent::Run {
            initial_messages, ..
        } => initial_messages
            .iter()
            .filter(|target| !entries_by_id.contains_key(target.id()))
            .cloned()
            .collect(),
        _ => Vec::new(),
    };

    let newest_attempt = operation_records
        .iter()
        .rev()
        .find_map(|record| match record {
            LaneRecord::StepAttempt(attempt) => Some(attempt),
            _ => None,
        });
    let step = newest_attempt
        .filter(|attempt| !entries_by_id.contains_key(attempt.result_entry_id.as_str()))
        .map(|attempt| StepState {
            kind: attempt.step,
            attempts: attempt.attempt,
            result_entry_id: attempt.result_entry_id.clone(),
            compaction_reason: (attempt.step == StepKind::Compaction)
                .then_some(attempt.compaction_reason)
                .flatten(),
        });

    let mut consumed_input_ids: HashSet<&str> = HashSet::new();
    if let OperationIntent::Run {
        initial_messages, ..
    } = &started.intent
    {
        for target in initial_messages {
            consumed_input_ids.insert(target.id());
        }
    }
    for record in &operation_records {
        if let LaneRecord::QueueEnqueued(enqueued) = record
            && enqueued.queue != QueueKind::NextRun
        {
            consumed_input_ids.insert(enqueued.target.id());
        }
    }
    let mut newest_consumed_input_sequence: Option<u64> = None;
    for id in &consumed_input_ids {
        if let Some(entry) = entries_by_id.get(id)
            && matches!(entry, Entry::Message(_))
        {
            newest_consumed_input_sequence = Some(
                newest_consumed_input_sequence
                    .map_or(entry.seq(), |current| current.max(entry.seq())),
            );
        }
    }
    let overflow_recovery_used = operation_records.iter().any(|record| {
        matches!(
            record,
            LaneRecord::StepAttempt(attempt)
                if attempt.step == StepKind::Compaction
                    && attempt.compaction_reason == Some(CompactionReason::Overflow)
                    && newest_consumed_input_sequence
                        .is_none_or(|sequence| attempt.seq > sequence)
        )
    });

    let newest_own_entry = own_entries.last().copied();
    let newest_own = derive_newest_own(newest_own_entry);
    let deferred = newest_own_entry.and_then(|entry| match entry {
        Entry::Message(message) => match &message.message {
            AgentMessage::Assistant(assistant) if assistant.stop_reason == StopReason::Deferred => {
                assistant.deferred.clone()
            }
            _ => None,
        },
        _ => None,
    });
    let mut targets = OperationTargets::default();
    match &started.intent {
        OperationIntent::Compaction {
            result_entry_id, ..
        } => targets.result = Some(entries_by_id.contains_key(result_entry_id.as_str())),
        OperationIntent::Navigation {
            summary_entry_id: Some(summary_entry_id),
            ..
        } => targets.summary = Some(entries_by_id.contains_key(summary_entry_id.as_str())),
        _ => {}
    }

    let deferred_write_ids: HashSet<&str> = operation_records
        .iter()
        .filter_map(|record| match record {
            LaneRecord::WriteDeferred(deferred) => Some(deferred.target.id()),
            _ => None,
        })
        .collect();
    let mut terminal_failure: Option<TerminalFailureState> = None;
    if let Some(newest) = newest_own_entry
        && let Entry::Message(message) = newest
        && let AgentMessage::Assistant(assistant) = &message.message
        && assistant.stop_reason == StopReason::Error
        && !deferred_write_ids.contains(newest.id())
    {
        let produced_by_step = operation_records.iter().any(|record| {
            matches!(
                record,
                LaneRecord::StepAttempt(attempt) if attempt.result_entry_id == newest.id()
            )
        });
        let previous_own_entry = own_entries
            .len()
            .checked_sub(2)
            .map(|index| own_entries[index]);
        let produced_by_deferred_fetch = operation_records.iter().any(|record| {
            matches!(
                record,
                LaneRecord::Usage(usage)
                    if matches!(&usage.cause, UsageCause::DeferredFetch { entry_id, .. }
                        if entry_id == newest.id())
            )
        }) || previous_own_entry.is_some_and(|entry| {
            matches!(
                entry,
                Entry::Message(message)
                    if matches!(&message.message, AgentMessage::Assistant(assistant)
                        if assistant.stop_reason == StopReason::Deferred)
            )
        });
        if produced_by_step || produced_by_deferred_fetch {
            terminal_failure = Some(TerminalFailureState {
                entry_id: newest.id().to_owned(),
                source: if produced_by_step {
                    FailureSource::Step
                } else {
                    FailureSource::DeferredFetch
                },
                message: (**assistant).clone(),
            });
        }
    }

    let tool_batch = derive_tool_batch(
        &started.id,
        &operation_records,
        &own_entries,
        &entries_by_id,
        &deferred_write_ids,
    );

    Ok(LaneReductionResult {
        lane_state: LaneState {
            lane: input.lane.clone(),
            leaf_id: input.leaf_id.clone(),
            operation: Some(OperationState {
                id: started.id.clone(),
                kind: intent_kind(&started.intent),
                intent: started.intent.clone(),
                aborting,
                step,
                tool_batch,
                missing_initial_messages,
                pending_steer,
                pending_follow_up,
                pending_writes,
                deferred,
                overflow_recovery_used,
                newest_own,
                targets,
            }),
            pending_next_run,
        },
        effective_configuration,
        terminal_failure,
    })
}

/// The provisioned targets of one pending queue, oldest first.
fn queue_targets(
    records: &[&QueueEnqueuedRecord],
    queue: QueueKind,
    run_id: Option<&str>,
) -> Vec<ProvisionedEntry> {
    records
        .iter()
        .filter(|enqueued| enqueued.queue == queue && enqueued.run_id.as_deref() == run_id)
        .map(|enqueued| enqueued.target.clone())
        .collect()
}

fn derive_effective_configuration(input: &LaneReductionInput) -> EffectiveLaneConfiguration {
    let mut configuration = input.defaults.clone();
    let mut by_id: HashMap<&str, &Entry> = HashMap::new();
    for entry in &input.configuration_entries {
        by_id.insert(entry.id(), entry);
    }
    for entry in &input.own_entries {
        by_id.insert(entry.id(), entry);
    }
    let mut ordered: Vec<&Entry> = by_id.into_values().collect();
    ordered.sort_by_key(|entry| entry.seq());

    for entry in ordered {
        match entry {
            Entry::ModelChange(change) => {
                configuration.model = Some(SessionModel {
                    provider: change.provider.clone(),
                    model_id: change.model_id.clone(),
                })
            }
            Entry::ThinkingLevelChange(change) => {
                configuration.thinking_level = change.thinking_level.clone()
            }
            Entry::ActiveToolsChange(change) => {
                configuration.active_tool_names = Some(change.active_tool_names.clone())
            }
            Entry::Message(message) => {
                if let AgentMessage::Assistant(assistant) = &message.message {
                    configuration.model = Some(SessionModel {
                        provider: assistant.provider.as_str().to_owned(),
                        model_id: assistant.model.clone(),
                    })
                }
            }
            _ => {}
        }
    }
    configuration
}

fn derive_newest_own(entry: Option<&Entry>) -> Option<NewestOwn> {
    let entry = entry?;
    let Entry::Message(message) = entry else {
        return Some(NewestOwn {
            entry_id: entry.id().to_owned(),
            entry_type: entry.entry_type().to_owned(),
            role: None,
            stop_reason: None,
        });
    };
    let AgentMessage::Assistant(assistant) = &message.message else {
        return Some(NewestOwn {
            entry_id: message.id.clone(),
            entry_type: entry.entry_type().to_owned(),
            role: Some(message.message.role().to_owned()),
            stop_reason: None,
        });
    };
    Some(NewestOwn {
        entry_id: message.id.clone(),
        entry_type: entry.entry_type().to_owned(),
        role: Some(message.message.role().to_owned()),
        stop_reason: Some(assistant.stop_reason),
    })
}

fn derive_tool_batch(
    operation_id: &str,
    records: &[&LaneRecord],
    own_entries: &[&Entry],
    entries_by_id: &HashMap<&str, &Entry>,
    deferred_write_ids: &HashSet<&str>,
) -> Option<ToolBatchState> {
    let assistant_entry = own_entries.iter().rev().find(|entry| {
        matches!(
            entry,
            Entry::Message(message)
                if matches!(&message.message, AgentMessage::Assistant(assistant)
                    if assistant.content.iter().any(|content| content.tool_call().is_some()))
        )
    })?;
    let Entry::Message(assistant_message) = assistant_entry else {
        return None;
    };
    let AgentMessage::Assistant(assistant) = &assistant_message.message else {
        return None;
    };

    let tool_calls: Vec<&ToolCall> = assistant
        .content
        .iter()
        .filter_map(|content| content.tool_call())
        .collect();
    let mut starts: HashMap<usize, &ToolStartedRecord> = HashMap::new();
    for record in records {
        if let LaneRecord::ToolStarted(started) = record
            && started.run_id == operation_id
            && started.assistant_entry_id == assistant_message.id
        {
            starts.insert(started.tool_index, started);
        }
    }

    let calls: Vec<ToolCallState> = tool_calls
        .iter()
        .enumerate()
        .map(|(tool_index, tool_call)| {
            let started = starts.get(&tool_index).copied();
            let started_result = started
                .and_then(|started| entries_by_id.get(started.result_entry_id.as_str()).copied());
            let blocked_result = own_entries
                .iter()
                .find(|entry| {
                    entry.seq() > assistant_message.seq
                        && !deferred_write_ids.contains(entry.id())
                        && matches!(
                            entry,
                            Entry::Message(message)
                                if matches!(&message.message, AgentMessage::ToolResult(result)
                                    if result.tool_call_id == tool_call.id)
                        )
                })
                .copied();
            let result = started_result.or(blocked_result);
            ToolCallState {
                tool_index,
                tool_call: (*tool_call).clone(),
                started: started.cloned(),
                result_exists: result.is_some(),
                terminate: match result {
                    Some(Entry::Message(message)) if message.terminate => Some(true),
                    _ => None,
                },
            }
        })
        .collect();

    Some(ToolBatchState {
        assistant_entry_id: assistant_message.id.clone(),
        truncated: assistant.stop_reason == StopReason::Length,
        unresolved: calls.iter().any(|call| !call.result_exists),
        calls,
    })
}

// ---------------------------------------------------------------------------
// Wire-name helpers
// ---------------------------------------------------------------------------

fn step_kind_str(step: StepKind) -> &'static str {
    match step {
        StepKind::Assistant => "assistant",
        StepKind::Compaction => "compaction",
        StepKind::BranchSummary => "branch_summary",
    }
}

fn queue_kind_str(queue: QueueKind) -> &'static str {
    match queue {
        QueueKind::Steer => "steer",
        QueueKind::FollowUp => "followUp",
        QueueKind::NextRun => "nextRun",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use step_harness_providers::types::{
        Api, ProviderId, TextContent, ToolResultMessage, Usage, UserContent, UserContentBlock,
        UserMessage,
    };

    // -- entry helpers ------------------------------------------------------

    fn assistant_message(stop_reason: StopReason, tool_calls: Vec<ToolCall>) -> AssistantMessage {
        let mut content: Vec<step_harness_providers::types::AssistantContent> = tool_calls
            .into_iter()
            .map(step_harness_providers::types::AssistantContent::ToolCall)
            .collect();
        if content.is_empty() {
            content.push(step_harness_providers::types::AssistantContent::Text(
                TextContent {
                    text: "hello".to_owned(),
                    text_signature: None,
                },
            ));
        }
        AssistantMessage {
            content,
            api: Api::AnthropicMessages,
            provider: ProviderId::new("anthropic"),
            model: "claude-test".to_owned(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: Usage::default(),
            stop_reason,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            timestamp: 0,
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
        .into_entry(parent_id, seq, 1_000 + seq)
    }

    fn user_entry(id: &str, seq: u64) -> Entry {
        message_entry(
            id,
            seq,
            None,
            AgentMessage::User(UserMessage {
                content: UserContent::Blocks(vec![UserContentBlock::Text(TextContent {
                    text: "hi".to_owned(),
                    text_signature: None,
                })]),
                timestamp: 0,
            }),
        )
    }

    fn assistant_entry(id: &str, seq: u64, stop_reason: StopReason) -> Entry {
        message_entry(
            id,
            seq,
            None,
            AgentMessage::assistant(assistant_message(stop_reason, Vec::new())),
        )
    }

    fn assistant_tool_calls_entry(id: &str, seq: u64, calls: Vec<ToolCall>) -> Entry {
        message_entry(
            id,
            seq,
            None,
            AgentMessage::assistant(assistant_message(StopReason::ToolUse, calls)),
        )
    }

    fn tool_result_entry(id: &str, seq: u64, tool_call: &ToolCall) -> Entry {
        message_entry(
            id,
            seq,
            None,
            AgentMessage::ToolResult(Box::new(ToolResultMessage {
                tool_call_id: tool_call.id.clone(),
                tool_name: tool_call.name.clone(),
                content: vec![UserContentBlock::Text(TextContent {
                    text: "done".to_owned(),
                    text_signature: None,
                })],
                details: None,
                usage: None,
                added_tool_names: None,
                is_error: false,
                timestamp: 0,
            })),
        )
    }

    fn tool_call(id: &str, name: &str) -> ToolCall {
        ToolCall {
            id: id.to_owned(),
            name: name.to_owned(),
            arguments: serde_json::Map::new(),
            thought_signature: None,
            namespace: None,
        }
    }

    fn provisioned_message(id: &str) -> ProvisionedEntry {
        ProvisionedEntry::Message {
            id: id.to_owned(),
            message: AgentMessage::User(UserMessage {
                content: UserContent::Blocks(vec![UserContentBlock::Text(TextContent {
                    text: "hi".to_owned(),
                    text_signature: None,
                })]),
                timestamp: 0,
            }),
            terminate: false,
        }
    }

    // -- record helpers -----------------------------------------------------

    fn operation_started(id: &str, seq: u64, intent: OperationIntent) -> OperationStartedRecord {
        OperationStartedRecord {
            id: id.to_owned(),
            seq,
            lane: "main".to_owned(),
            timestamp: 0,
            source_leaf_id: None,
            intent,
        }
    }

    fn run_intent(initial_messages: Vec<ProvisionedEntry>) -> OperationIntent {
        OperationIntent::Run {
            original_prompt: Vec::new(),
            initial_messages,
            system_prompt_override: None,
            resume_data: None,
        }
    }

    fn step_attempt(
        id: &str,
        seq: u64,
        run_id: &str,
        step: StepKind,
        attempt: u32,
        result_entry_id: &str,
    ) -> StepAttemptRecord {
        StepAttemptRecord {
            id: id.to_owned(),
            seq,
            lane: "main".to_owned(),
            timestamp: 0,
            run_id: run_id.to_owned(),
            step,
            attempt,
            result_entry_id: result_entry_id.to_owned(),
            compaction_reason: None,
        }
    }

    fn tool_started(
        id: &str,
        seq: u64,
        run_id: &str,
        assistant_entry_id: &str,
        tool_index: usize,
        call: &ToolCall,
        result_entry_id: &str,
    ) -> ToolStartedRecord {
        ToolStartedRecord {
            id: id.to_owned(),
            seq,
            lane: "main".to_owned(),
            timestamp: 0,
            run_id: run_id.to_owned(),
            assistant_entry_id: assistant_entry_id.to_owned(),
            tool_index,
            tool_call_id: call.id.clone(),
            tool_name: call.name.clone(),
            effective_args: serde_json::Map::new(),
            result_entry_id: result_entry_id.to_owned(),
            replay: crate::session::types::ReplayPolicy::Never,
        }
    }

    fn queue_enqueued(
        id: &str,
        seq: u64,
        queue: QueueKind,
        run_id: Option<&str>,
        target: ProvisionedEntry,
    ) -> LaneRecord {
        LaneRecord::QueueEnqueued(QueueEnqueuedRecord {
            id: id.to_owned(),
            seq,
            lane: "main".to_owned(),
            timestamp: 0,
            queue,
            run_id: run_id.map(str::to_owned),
            target,
        })
    }

    fn input(
        open_operations: Vec<OperationStartedRecord>,
        records: Vec<LaneRecord>,
        entries: Vec<Entry>,
    ) -> LaneReductionInput {
        LaneReductionInput {
            lane: "main".to_owned(),
            open_operations,
            records,
            entries,
            leaf_id: None,
            own_entries: Vec::new(),
            configuration_entries: Vec::new(),
            defaults: EffectiveLaneConfiguration::default(),
        }
    }

    fn slice(
        open_operations: Vec<OperationStartedRecord>,
        records: Vec<LaneRecord>,
        entries: Vec<Entry>,
    ) -> RecordLogSlice {
        RecordLogSlice {
            lane: "main".to_owned(),
            open_operations,
            records,
            entries,
        }
    }

    // -- validation tests ---------------------------------------------------

    #[test]
    fn rejects_two_open_operations() {
        let error = validate_record_log(&slice(
            vec![
                operation_started("op1", 1, run_intent(vec![])),
                operation_started("op2", 2, run_intent(vec![])),
            ],
            Vec::new(),
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(
            error.reason,
            RecordLogCorruptionReason::MultipleOpenOperations
        );
    }

    #[test]
    fn rejects_records_after_finish() {
        let mut records = vec![LaneRecord::OperationStarted(operation_started(
            "op1",
            1,
            run_intent(vec![]),
        ))];
        // operation_finished must come from the finished record shape.
        records.push(LaneRecord::OperationFinished(
            crate::session::types::OperationFinishedRecord {
                id: "fin".to_owned(),
                seq: 3,
                lane: "main".to_owned(),
                timestamp: 0,
                run_id: "op1".to_owned(),
                outcome: crate::session::types::OperationOutcome::Completed,
                error: None,
            },
        ));
        records.push(LaneRecord::StepAttempt(step_attempt(
            "s1",
            4,
            "op1",
            StepKind::Assistant,
            1,
            "a1",
        )));
        let error = validate_record_log(&slice(
            Vec::new(),
            records,
            vec![assistant_entry("a1", 2, StopReason::Stop)],
        ))
        .unwrap_err();
        assert_eq!(error.reason, RecordLogCorruptionReason::RecordAfterFinish);
    }

    #[test]
    fn rejects_unknown_operation_and_non_consecutive_attempt() {
        let error = validate_record_log(&slice(
            Vec::new(),
            vec![LaneRecord::StepAttempt(step_attempt(
                "s1",
                1,
                "missing",
                StepKind::Assistant,
                1,
                "a1",
            ))],
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(error.reason, RecordLogCorruptionReason::UnknownOperation);

        let started = operation_started("op1", 1, run_intent(vec![]));
        let error = validate_record_log(&slice(
            vec![started.clone()],
            vec![
                LaneRecord::OperationStarted(started),
                LaneRecord::StepAttempt(step_attempt("s1", 2, "op1", StepKind::Assistant, 2, "a1")),
            ],
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(
            error.reason,
            RecordLogCorruptionReason::NonConsecutiveAttempt
        );
    }

    #[test]
    fn rejects_invalid_compaction_reasons() {
        let started = operation_started("op1", 1, run_intent(vec![]));
        let mut attempt = step_attempt("s1", 2, "op1", StepKind::Compaction, 1, "c1");
        let error = validate_record_log(&slice(
            vec![started.clone()],
            vec![
                LaneRecord::OperationStarted(started.clone()),
                LaneRecord::StepAttempt(attempt.clone()),
            ],
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(
            error.reason,
            RecordLogCorruptionReason::InvalidCompactionReason
        );

        attempt.step = StepKind::Assistant;
        attempt.compaction_reason = Some(CompactionReason::Manual);
        let error = validate_record_log(&slice(
            vec![started.clone()],
            vec![
                LaneRecord::OperationStarted(started),
                LaneRecord::StepAttempt(attempt),
            ],
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(
            error.reason,
            RecordLogCorruptionReason::InvalidCompactionReason
        );
    }

    #[test]
    fn rejects_duplicate_tool_invocations() {
        let call = tool_call("call-1", "bash");
        let assistant = assistant_tool_calls_entry("a1", 1, vec![call.clone()]);
        let result = tool_result_entry("r1", 2, &call);
        let started = operation_started("op1", 3, run_intent(vec![]));
        let records = vec![
            LaneRecord::OperationStarted(started.clone()),
            LaneRecord::ToolStarted(tool_started("t1", 4, "op1", "a1", 0, &call, "r1")),
            LaneRecord::ToolStarted(tool_started("t2", 5, "op1", "a1", 0, &call, "r1")),
        ];
        let error = validate_record_log(&slice(vec![started], records, vec![assistant, result]))
            .unwrap_err();
        assert_eq!(
            error.reason,
            RecordLogCorruptionReason::DuplicateToolInvocation
        );
    }

    #[test]
    fn rejects_provisioned_entry_mismatch() {
        let target = provisioned_message("m1");
        let stored = message_entry(
            "m1",
            1,
            None,
            AgentMessage::User(UserMessage {
                content: UserContent::Blocks(vec![UserContentBlock::Text(TextContent {
                    text: "different".to_owned(),
                    text_signature: None,
                })]),
                timestamp: 0,
            }),
        );
        let started = operation_started("op1", 2, run_intent(vec![target]));
        let error = validate_record_log(&slice(
            vec![started],
            vec![LaneRecord::OperationStarted(operation_started(
                "op1",
                2,
                run_intent(vec![provisioned_message("m1")]),
            ))],
            vec![stored],
        ))
        .unwrap_err();
        assert_eq!(
            error.reason,
            RecordLogCorruptionReason::ProvisionedEntryMismatch
        );
    }

    #[test]
    fn rejects_missing_deferred_handle() {
        let entry = message_entry(
            "a1",
            1,
            None,
            AgentMessage::assistant(assistant_message(StopReason::Deferred, Vec::new())),
        );
        let error = validate_record_log(&slice(Vec::new(), Vec::new(), vec![entry])).unwrap_err();
        assert_eq!(
            error.reason,
            RecordLogCorruptionReason::InvalidDeferredHandle
        );
    }

    // -- reduction tests ----------------------------------------------------

    #[test]
    fn reduces_an_idle_lane_with_pending_next_run() {
        let queued = provisioned_message("m1");
        let records = vec![
            LaneRecord::OperationStarted(operation_started(
                "op1",
                1,
                run_intent(vec![queued.clone()]),
            )),
            LaneRecord::OperationFinished(crate::session::types::OperationFinishedRecord {
                id: "fin".to_owned(),
                seq: 2,
                lane: "main".to_owned(),
                timestamp: 0,
                run_id: "op1".to_owned(),
                outcome: crate::session::types::OperationOutcome::Completed,
                error: None,
            }),
            queue_enqueued("q1", 3, QueueKind::NextRun, None, provisioned_message("m2")),
        ];
        let mut configuration_entries = vec![
            crate::session::types::ProvisionedEntry::ThinkingLevelChange {
                id: "t1".to_owned(),
                thinking_level: "high".to_owned(),
            }
            .into_entry(None, 4, 0),
        ];
        // A model change entry.
        configuration_entries.push(Entry::ModelChange(
            crate::session::types::ModelChangeEntry {
                id: "mc1".to_owned(),
                seq: 5,
                parent_id: None,
                timestamp: 0,
                provider: "anthropic".to_owned(),
                model_id: "claude-test".to_owned(),
            },
        ));
        let mut reduction = input(Vec::new(), records, Vec::new());
        reduction.leaf_id = Some("m1".to_owned());
        reduction.configuration_entries = configuration_entries;
        reduction.defaults = EffectiveLaneConfiguration {
            model: None,
            thinking_level: "off".to_owned(),
            active_tool_names: Some(vec!["bash".to_owned()]),
        };

        let result = reduce_lane_state(&reduction).unwrap();
        assert!(result.lane_state.operation.is_none());
        assert_eq!(result.lane_state.leaf_id.as_deref(), Some("m1"));
        // The captured initial message is not pending; only the later enqueue is.
        assert_eq!(result.lane_state.pending_next_run.len(), 1);
        assert_eq!(result.lane_state.pending_next_run[0].id(), "m2");
        assert_eq!(result.effective_configuration.thinking_level, "high");
        assert_eq!(
            result.effective_configuration.model,
            Some(SessionModel {
                provider: "anthropic".to_owned(),
                model_id: "claude-test".to_owned()
            })
        );
        assert_eq!(
            result.effective_configuration.active_tool_names,
            Some(vec!["bash".to_owned()])
        );
    }

    #[test]
    fn reduces_an_open_operation_and_its_tool_batch() {
        let call = tool_call("call-1", "bash");
        let assistant = assistant_tool_calls_entry("a1", 1, vec![call.clone()]);
        let result_entry = tool_result_entry("r1", 2, &call);
        let started = operation_started("op1", 3, run_intent(vec![]));
        let records = vec![
            LaneRecord::OperationStarted(started.clone()),
            LaneRecord::StepAttempt(step_attempt("s1", 4, "op1", StepKind::Assistant, 1, "a1")),
            LaneRecord::ToolStarted(tool_started("t1", 5, "op1", "a1", 0, &call, "r1")),
        ];
        let mut reduction = input(
            vec![started],
            records,
            vec![assistant.clone(), result_entry.clone()],
        );
        reduction.own_entries = vec![assistant, result_entry];

        let result = reduce_lane_state(&reduction).unwrap();
        let operation = result.lane_state.operation.expect("open operation");
        assert_eq!(operation.kind, OperationKind::Run);
        assert!(!operation.aborting);
        // The step's result entry exists, so no step is in flight.
        assert!(operation.step.is_none());
        let batch = operation.tool_batch.expect("tool batch");
        assert_eq!(batch.assistant_entry_id, "a1");
        assert_eq!(batch.calls.len(), 1);
        assert!(batch.calls[0].result_exists);
        assert!(batch.calls[0].started.is_some());
        assert!(!batch.truncated);
        assert!(!batch.unresolved);
        assert_eq!(batch.calls[0].tool_call.id, "call-1");
    }

    #[test]
    fn reduces_a_truncated_unresolved_tool_batch() {
        let call = tool_call("call-1", "bash");
        let mut assistant_entry = assistant_tool_calls_entry("a1", 1, vec![call.clone()]);
        if let Entry::Message(message) = &mut assistant_entry
            && let AgentMessage::Assistant(assistant) = &mut message.message
        {
            assistant.stop_reason = StopReason::Length;
        }
        let started = operation_started("op1", 2, run_intent(vec![]));
        let mut reduction = input(vec![started], Vec::new(), vec![assistant_entry.clone()]);
        reduction.own_entries = vec![assistant_entry];

        let result = reduce_lane_state(&reduction).unwrap();
        let batch = result
            .lane_state
            .operation
            .expect("open operation")
            .tool_batch
            .expect("tool batch");
        assert!(batch.truncated);
        assert!(batch.unresolved);
        assert!(!batch.calls[0].result_exists);
        assert!(batch.calls[0].started.is_none());
    }

    #[test]
    fn detects_a_terminal_failure_produced_by_a_step() {
        let failed = assistant_entry("a1", 1, StopReason::Error);
        let started = operation_started("op1", 2, run_intent(vec![]));
        let records = vec![
            LaneRecord::OperationStarted(started.clone()),
            LaneRecord::StepAttempt(step_attempt("s1", 3, "op1", StepKind::Assistant, 1, "a1")),
        ];
        let mut reduction = input(vec![started], records, vec![failed.clone()]);
        reduction.own_entries = vec![failed];

        let result = reduce_lane_state(&reduction).unwrap();
        let failure = result.terminal_failure.expect("terminal failure");
        assert_eq!(failure.entry_id, "a1");
        assert_eq!(failure.source, FailureSource::Step);
        assert_eq!(failure.message.stop_reason, StopReason::Error);
    }

    #[test]
    fn reports_pending_steer_and_follow_up_until_abort() {
        let started = operation_started("op1", 1, run_intent(vec![]));
        let steer = queue_enqueued(
            "q1",
            2,
            QueueKind::Steer,
            Some("op1"),
            provisioned_message("m1"),
        );
        let follow_up = queue_enqueued(
            "q2",
            3,
            QueueKind::FollowUp,
            Some("op1"),
            provisioned_message("m2"),
        );
        let mut reduction = input(
            vec![started.clone()],
            vec![
                LaneRecord::OperationStarted(started.clone()),
                steer,
                follow_up,
            ],
            Vec::new(),
        );
        reduction.own_entries = vec![user_entry("u1", 4)];

        let result = reduce_lane_state(&reduction).unwrap();
        let operation = result.lane_state.operation.expect("open operation");
        assert_eq!(operation.pending_steer.len(), 1);
        assert_eq!(operation.pending_follow_up.len(), 1);
        assert_eq!(operation.newest_own.expect("newest own").entry_id, "u1");

        // Once an abort is requested, both queues read as drained.
        let records = vec![
            LaneRecord::OperationStarted(started.clone()),
            queue_enqueued(
                "q1",
                2,
                QueueKind::Steer,
                Some("op1"),
                provisioned_message("m1"),
            ),
            LaneRecord::AbortRequested(crate::session::types::AbortRequestedRecord {
                id: "ab".to_owned(),
                seq: 3,
                lane: "main".to_owned(),
                timestamp: 0,
                run_id: "op1".to_owned(),
            }),
        ];
        let mut aborted = input(vec![started], records, Vec::new());
        aborted.own_entries = vec![user_entry("u1", 4)];
        let result = reduce_lane_state(&aborted).unwrap();
        let operation = result.lane_state.operation.expect("open operation");
        assert!(operation.aborting);
        assert!(operation.pending_steer.is_empty());
        assert!(operation.pending_follow_up.is_empty());
    }
}
