//! Session-level events and the approval channel contract.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use step_harness_agent_core::types::AgentEvent;

/// A settled execution, including model turns, tool calls and approval waits.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExecutionTiming {
    pub first_message_timestamp: u64,
    pub last_message_timestamp: u64,
    pub duration_ms: u64,
    /// Exclusive wall time; parallel tool batches count once, excluding waits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_duration_ms: Option<u64>,
    /// Foreground conversation and compaction requests, including streaming.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_duration_ms: Option<u64>,
    /// Tool confirmations and plan reviews.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval_wait_duration_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_wait_duration_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub other_duration_ms: Option<u64>,
}

#[cfg(test)]
mod timing_tests {
    use super::ExecutionTiming;

    #[test]
    fn legacy_timings_round_trip_without_invented_breakdowns() {
        let value = serde_json::json!({
            "firstMessageTimestamp": 1, "lastMessageTimestamp": 2, "durationMs": 1000,
        });
        let timing: ExecutionTiming = serde_json::from_value(value.clone()).unwrap();
        assert!(timing.tool_duration_ms.is_none());
        assert!(timing.api_duration_ms.is_none());
        assert!(timing.approval_wait_duration_ms.is_none());
        assert!(timing.retry_wait_duration_ms.is_none());
        assert!(timing.other_duration_ms.is_none());
        assert_eq!(serde_json::to_value(timing).unwrap(), value);
    }
}

/// Why a compaction ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum CompactionReason {
    Manual,
    Threshold,
    Overflow,
}

/// A confirmation request surfaced to the embedder's UI.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ApprovalRequest {
    /// Correlation id; the UI echoes it back with its answer.
    pub request_id: String,
    /// Session that owns this request, independent of the selected view.
    pub session_id: String,
    /// Time the request began waiting, in Unix milliseconds.
    pub created_at: u64,
    pub tool_name: String,
    /// Compact `key=value` rendering of the tool arguments.
    pub args_summary: String,
    /// Why the call needs approval.
    pub reason: String,
    /// True when a dangerous-command rule matched.
    pub hazardous: bool,
}

/// The embedder's approval UI.
///
/// Implementations must resolve every request: the agent run blocks on the
/// returned decision. Returning `false` denies the tool call.
#[async_trait::async_trait]
pub trait ApprovalChannel: Send + Sync {
    async fn confirm(&self, request: &ApprovalRequest) -> bool;
}

/// Events a session publishes to its embedder.
///
/// The taxonomy follows the TypeScript `AgentSessionEvent` union, minus the
/// extension and terminal-only variants. `agent` carries the underlying agent
/// event verbatim so the UI can render streaming output directly.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum SessionEvent {
    /// The active transcript was rewound after recalling the last user message.
    MessagesRecalled {
        messages: Vec<step_harness_agent_core::types::AgentMessage>,
        execution_timings: Vec<ExecutionTiming>,
    },
    ExecutionCompleted {
        timing: ExecutionTiming,
    },
    /// Agent lifecycle event (streaming, tool execution).
    Agent {
        event: AgentEvent,
    },
    /// The run finished: retries, compaction, and queue drains are done.
    Settled,
    /// A provider-error retry was scheduled.
    AutoRetryStart {
        attempt: u32,
        max_attempts: u32,
        delay_ms: u64,
        error_message: String,
    },
    /// A retry sequence finished.
    AutoRetryEnd {
        success: bool,
        attempt: u32,
        final_error: Option<String>,
    },
    /// Compaction started.
    CompactionStart {
        reason: CompactionReason,
    },
    /// Compaction finished (successfully or with an error).
    CompactionEnd {
        reason: CompactionReason,
        error_message: Option<String>,
    },
    /// Steering/follow-up queue contents changed.
    QueueUpdate {
        steering: Vec<String>,
        follow_up: Vec<String>,
    },
    /// The session's display name changed.
    SessionInfoChanged {
        name: Option<String>,
    },
    /// The session's project assignment changed.
    ProjectChanged {
        project: Option<String>,
    },
    /// A durable entry was appended to the session store.
    EntryAppended {
        entry_id: String,
    },
    /// The active model or thinking level changed.
    ModelChanged {
        provider: String,
        model_id: String,
        thinking_level: String,
    },
    /// Plan mode was entered or left.
    PlanModeChanged {
        active: bool,
        plan_file_path: Option<String>,
        plan_source: Option<String>,
    },
    /// A recoverable session error for the UI.
    Error {
        message: String,
    },
}

/// Receiver handle for [`SessionEvent`] streams.
pub type SessionEventRx = tokio::sync::broadcast::Receiver<SessionEvent>;

/// An approval channel that always allows (tests and non-interactive mode).
pub struct AutoApprove;

#[async_trait::async_trait]
impl ApprovalChannel for AutoApprove {
    async fn confirm(&self, _request: &ApprovalRequest) -> bool {
        true
    }
}

/// An approval channel that always denies (tests).
pub struct AutoDeny;

#[async_trait::async_trait]
impl ApprovalChannel for AutoDeny {
    async fn confirm(&self, _request: &ApprovalRequest) -> bool {
        false
    }
}

/// A channel backed by a closure (tests and simple embedders).
pub struct FnApproval<F>(pub F)
where
    F: Fn(&ApprovalRequest) -> bool + Send + Sync;

#[async_trait::async_trait]
impl<F> ApprovalChannel for FnApproval<F>
where
    F: Fn(&ApprovalRequest) -> bool + Send + Sync,
{
    async fn confirm(&self, request: &ApprovalRequest) -> bool {
        (self.0)(request)
    }
}

/// Convenience alias for a shared approval channel.
pub type SharedApprovalChannel = Arc<dyn ApprovalChannel>;
