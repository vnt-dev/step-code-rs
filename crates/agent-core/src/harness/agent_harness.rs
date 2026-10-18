//! The agent harness contract: lanes, durable operations, outcomes, and the
//! scaffolded [`AgentHarness`] runtime.
//!
//! Ported from `@step-harness/agent-core` `src\harness\agent-harness.ts`. Only
//! the contract layer is ported: the TypeScript source is itself scaffolding in
//! which every execution-engine method rejects with `HarnessNotImplemented`,
//! and the Rust port keeps that shape. Configuration management and `close` are
//! implemented; everything else returns an error value instead of rejecting a
//! promise.
//!
//! # Notable differences from the TypeScript source
//!
//! - Each TypeScript `TaggedError` tag is one struct plus one variant of the
//!   [`HarnessRejected`] union; `matchError` becomes an exhaustive `match`.
//! - Per-call rejection unions ([`RunRejected`], [`QueueRejected`], ...) carry an
//!   extra [`HarnessNotImplemented`] variant. TypeScript's `unavailable()`
//!   rejects with it even though it is not part of those unions, because
//!   rejected promises are not type-checked; Rust return values are.
//! - The `Result` monad is `std::result::Result`.
//! - `Hooks.on`/`Events.on` return an unsubscribe handle instead of a callback
//!   function, and registration failures are values.
//! - [`HarnessTool`] wraps an [`AgentTool`] instead of intersecting it, and fixes
//!   the tool details type to [`Json`](crate::types::Json) (the TypeScript
//!   default is `any`).
//! - [`AgentHarness`] is generic over the session storage metadata and the
//!   telemetry context, because [`Session`] and [`TelemetryContext`] are.
//! - Async option callbacks (`systemPrompt`, `toolContext`,
//!   `toProviderMessages`) become trait objects or
//!   [`ToolContextSource`](crate::harness::types::ToolContextSource).
//!
//! Like the TypeScript scaffold, [`AgentHarness`] retains only the configuration
//! its getters expose. Engine-only options — `models`, `toolContext`,
//! `systemPrompt`, `drive`, `toolExecution`, `toProviderMessages`, and
//! `entryProjectors` — are accepted by [`AgentHarnessOptions`] and dropped until
//! the execution engine is ported.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use serde::{Deserialize, Serialize};
use step_harness_providers::types::{
    AssistantMessage, DeferredHandle, ImageContent, Model, ModelThinkingLevel, Usage,
};
use step_harness_providers::{Models, RetryPolicy};

use crate::PromptInput;
use crate::compaction::CompactionSettings;
use crate::harness::reducer::OperationKind;
use crate::harness::types::{
    AgentHarnessResources, AgentHarnessStreamOptions, AgentHarnessStreamOptionsPatch,
    ToolContextSource,
};
use crate::session::facade::Session;
use crate::session::types::{
    BranchSummaryEntry, CompactionEntry, Entry, FactName, JsonValue, OperationOutcome,
    ProvisionedEntry, QueueKind, RecordQuery, ReplayPolicy, SessionError, StepKind,
};
use crate::types::{AgentMessage, AgentTool, ConvertToLlm, Json, QueueMode, ToolExecutionMode};
use step_harness_telemetry::TelemetryContext;
use step_harness_telemetry::noop::NoopTelemetryContext;

// ---------------------------------------------------------------------------
// Tagged errors
// ---------------------------------------------------------------------------

/// Another operation already owns the lane.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct LaneBusy {
    pub lane: String,
    pub operation_id: String,
    pub operation_kind: OperationKind,
    pub message: String,
}

/// A resume needs tools or models the harness does not have.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct MissingIdentities {
    pub lane: String,
    pub tools: Vec<String>,
    pub models: Vec<String>,
    pub message: String,
}

/// The lane has no run to queue against.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct NoActiveRun {
    pub lane: String,
    pub message: String,
}

/// The lane has no open operation at all.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct NoActiveOperation {
    pub lane: String,
    pub message: String,
}

/// The lane has nothing suspended to resume.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct NothingToResume {
    pub lane: String,
    pub message: String,
}

/// A prompt or queued message is not a valid agent message.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct InvalidMessage {
    pub lane: String,
    pub reason: String,
    pub message: String,
}

/// The named skill is not registered.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct UnknownSkill {
    pub name: String,
    pub message: String,
}

/// The named prompt template is not registered.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct UnknownTemplate {
    pub name: String,
    pub message: String,
}

/// The navigation target entry does not exist.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct UnknownTarget {
    pub target_id: String,
    pub message: String,
}

/// The queued entry is unknown, already consumed, or already cleared.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct UnknownQueueItem {
    pub lane: String,
    pub entry_id: String,
    pub message: String,
}

/// The lane already exists.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct LaneExists {
    pub lane: String,
    pub message: String,
}

/// The lane name is not usable.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct InvalidLane {
    pub lane: String,
    pub reason: String,
    pub message: String,
}

/// The lane has nothing worth compacting.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct NothingToCompact {
    pub lane: String,
    pub message: String,
}

/// The harness was closed while the operation was active.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct Closed {
    pub message: String,
}

impl Closed {
    /// The rejection used when a call arrives after [`AgentHarness::close`].
    pub fn after_close() -> Self {
        Closed {
            message: "AgentHarness was closed".to_owned(),
        }
    }
}

/// An unexpected engine failure with its underlying cause.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct HarnessFault {
    pub message: String,
    #[source]
    pub cause: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl HarnessFault {
    pub fn new(
        message: impl Into<String>,
        cause: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        HarnessFault {
            message: message.into(),
            cause: Some(Box::new(cause)),
        }
    }
}

/// The harness was closed while the operation was active.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("AgentHarness was closed while the operation was active")]
pub struct HarnessClosed;

/// The requested operation belongs to the execution engine, which is not
/// implemented in this contract layer.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("AgentHarness.{operation} is not implemented yet")]
pub struct HarnessNotImplemented {
    pub operation: String,
}

impl HarnessNotImplemented {
    pub fn new(operation: impl Into<String>) -> Self {
        HarnessNotImplemented {
            operation: operation.into(),
        }
    }
}

/// Error payload of a failed operation, matching the durable record shape.
pub type OperationError = crate::session::types::OperationError;

// ---------------------------------------------------------------------------
// Rejection unions
// ---------------------------------------------------------------------------

/// Error type of a harness call that can express a scaffold refusal.
///
/// Every rejection union implements this so [`AgentHarness`] can refuse a call
/// without knowing which union the caller sees.
pub trait UnavailableError: Sized {
    /// The error for a call the contract layer cannot serve.
    fn unavailable(operation: &'static str, closed: bool) -> Self;
}

/// Generates one rejection union: the given tags plus the scaffold-only
/// [`HarnessNotImplemented`] variant, with `From` conversions into
/// [`HarnessRejected`].
macro_rules! rejected_union {
    (
        umbrella
        $(#[$enum_meta:meta])*
        $name:ident {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident($ty:ident)
            ),* $(,)?
        }
    ) => {
        $(#[$enum_meta])*
        #[derive(Debug, thiserror::Error)]
        pub enum $name {
            $(
                $(#[$variant_meta])*
                #[error(transparent)]
                $variant($ty),
            )*
            /// The requested operation belongs to the execution engine, which
            /// this contract layer does not implement.
            #[error(transparent)]
            NotImplemented(HarnessNotImplemented),
        }

        $(
            impl From<$ty> for $name {
                fn from(error: $ty) -> Self {
                    $name::$variant(error)
                }
            }
        )*

        impl From<HarnessNotImplemented> for $name {
            fn from(error: HarnessNotImplemented) -> Self {
                $name::NotImplemented(error)
            }
        }

        impl UnavailableError for $name {
            fn unavailable(operation: &'static str, closed: bool) -> Self {
                if closed {
                    $name::Closed(Closed::after_close())
                } else {
                    $name::NotImplemented(HarnessNotImplemented::new(operation))
                }
            }
        }
    };
    (
        $(#[$enum_meta:meta])*
        $name:ident {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident($ty:ident)
            ),* $(,)?
        }
    ) => {
        $(#[$enum_meta])*
        #[derive(Debug, thiserror::Error)]
        pub enum $name {
            $(
                $(#[$variant_meta])*
                #[error(transparent)]
                $variant($ty),
            )*
            /// The requested operation belongs to the execution engine, which
            /// this contract layer does not implement.
            #[error(transparent)]
            NotImplemented(HarnessNotImplemented),
        }

        $(
            impl From<$ty> for $name {
                fn from(error: $ty) -> Self {
                    $name::$variant(error)
                }
            }
        )*

        impl From<HarnessNotImplemented> for $name {
            fn from(error: HarnessNotImplemented) -> Self {
                $name::NotImplemented(error)
            }
        }

        impl UnavailableError for $name {
            fn unavailable(operation: &'static str, closed: bool) -> Self {
                if closed {
                    $name::Closed(Closed::after_close())
                } else {
                    $name::NotImplemented(HarnessNotImplemented::new(operation))
                }
            }
        }

        impl From<$name> for HarnessRejected {
            fn from(error: $name) -> Self {
                match error {
                    $(
                        $(#[$variant_meta])*
                        $name::$variant(inner) => HarnessRejected::$variant(inner),
                    )*
                    $name::NotImplemented(inner) => HarnessRejected::NotImplemented(inner),
                }
            }
        }
    };
}

rejected_union! {
    umbrella
    /// Every rejection the harness can return, in tag order.
    HarnessRejected {
        LaneBusy(LaneBusy),
        MissingIdentities(MissingIdentities),
        NoActiveRun(NoActiveRun),
        NoActiveOperation(NoActiveOperation),
        NothingToResume(NothingToResume),
        InvalidMessage(InvalidMessage),
        UnknownSkill(UnknownSkill),
        UnknownTemplate(UnknownTemplate),
        UnknownTarget(UnknownTarget),
        UnknownQueueItem(UnknownQueueItem),
        LaneExists(LaneExists),
        InvalidLane(InvalidLane),
        NothingToCompact(NothingToCompact),
        Closed(Closed),
    }
}

rejected_union! {
    /// Rejections of `prompt`, `skill`, and `promptFromTemplate`.
    RunRejected {
        LaneBusy(LaneBusy),
        InvalidMessage(InvalidMessage),
        UnknownSkill(UnknownSkill),
        UnknownTemplate(UnknownTemplate),
        Closed(Closed),
    }
}

rejected_union! {
    /// Rejections of `compact`.
    CompactionRejected {
        LaneBusy(LaneBusy),
        NothingToCompact(NothingToCompact),
        Closed(Closed),
    }
}

rejected_union! {
    /// Rejections of `navigateTree`.
    NavigationRejected {
        LaneBusy(LaneBusy),
        UnknownTarget(UnknownTarget),
        Closed(Closed),
    }
}

rejected_union! {
    /// Rejections of `resume`.
    ResumeRejected {
        LaneBusy(LaneBusy),
        NothingToResume(NothingToResume),
        MissingIdentities(MissingIdentities),
        Closed(Closed),
    }
}

rejected_union! {
    /// Rejections of `steer`, `followUp`, and `nextRun`.
    QueueRejected {
        NoActiveRun(NoActiveRun),
        InvalidMessage(InvalidMessage),
        Closed(Closed),
    }
}

rejected_union! {
    /// Rejections of `cancelQueued`.
    CancelQueuedRejected {
        UnknownQueueItem(UnknownQueueItem),
        Closed(Closed),
    }
}

rejected_union! {
    /// Rejections of `abort`.
    AbortRejected {
        NoActiveOperation(NoActiveOperation),
        Closed(Closed),
    }
}

rejected_union! {
    /// Rejections of `recordUsage`. The TypeScript union is `Closed` alone; the
    /// scaffold refusal variant is added for the same reason as everywhere else.
    RecordUsageRejected {
        Closed(Closed),
    }
}

rejected_union! {
    /// Rejections of `createLane`.
    CreateLaneRejected {
        LaneExists(LaneExists),
        InvalidLane(InvalidLane),
        UnknownTarget(UnknownTarget),
        Closed(Closed),
    }
}

// ---------------------------------------------------------------------------
// Outcomes and results
// ---------------------------------------------------------------------------

/// How a run invocation ended.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum RunOutcome {
    Completed {
        leaf_id: String,
        final_entry_id: String,
        final_message: AssistantMessage,
    },
    Aborted {
        leaf_id: String,
        final_entry_id: String,
        final_message: AssistantMessage,
    },
    Failed {
        leaf_id: String,
        error: OperationError,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        final_entry_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        final_message: Option<AssistantMessage>,
    },
    Suspended {
        leaf_id: String,
        final_entry_id: String,
        deferred: DeferredHandle,
    },
}

/// How a compaction invocation ended.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
#[allow(clippy::large_enum_variant)] // the `Completed` variant carries the whole compaction entry, mirroring the TypeScript shape.
pub enum CompactionOutcome {
    Completed {
        leaf_id: String,
        entry: CompactionEntry,
    },
    Declined {
        leaf_id: String,
    },
    Aborted {
        leaf_id: String,
    },
    Failed {
        leaf_id: String,
        error: OperationError,
    },
}

/// How a navigation invocation ended.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
#[allow(clippy::large_enum_variant)] // the `Completed` variant carries the whole branch-summary entry, mirroring the TypeScript shape.
pub enum NavigationOutcome {
    Completed {
        new_leaf_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        summary_entry: Option<BranchSummaryEntry>,
    },
    Declined {
        leaf_id: Option<String>,
    },
    Aborted {
        leaf_id: Option<String>,
    },
    Failed {
        leaf_id: Option<String>,
        error: OperationError,
    },
}

/// Which operation a resumed invocation continued.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
#[allow(clippy::large_enum_variant)] // each variant carries a whole outcome enum, mirroring the TypeScript shape.
pub enum ResumeOutcome {
    Run {
        run_id: String,
        outcome: RunOutcome,
    },
    Compaction {
        run_id: String,
        outcome: CompactionOutcome,
    },
    Navigation {
        run_id: String,
        outcome: NavigationOutcome,
    },
}

/// A successful run: its run id plus the outcome.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunSuccess {
    pub run_id: String,
    pub outcome: RunOutcome,
}

/// A successful compaction: its run id plus the outcome.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionSuccess {
    pub run_id: String,
    pub outcome: CompactionOutcome,
}

/// A successful navigation: its run id plus the outcome.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NavigationSuccess {
    pub run_id: String,
    pub outcome: NavigationOutcome,
}

/// A successful abort: the drained queues.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AbortSuccess {
    pub run_id: String,
    pub steer: Vec<AgentMessage>,
    pub follow_up: Vec<AgentMessage>,
}

/// What happened to a cancelled queue item.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CancelQueuedOutcome {
    Cancelled,
    AlreadyConsumed,
    AlreadyCleared,
}

/// Result of `prompt`, `skill`, and `promptFromTemplate`.
pub type RunResult = Result<RunSuccess, RunRejected>;
/// Result of `compact`.
pub type CompactionResult = Result<CompactionSuccess, CompactionRejected>;
/// Result of `navigateTree`.
pub type NavigationResult = Result<NavigationSuccess, NavigationRejected>;
/// Result of `steer`, `followUp`, and `nextRun`: the enqueued entry id.
pub type QueueResult = Result<String, QueueRejected>;
/// Result of `cancelQueued`.
pub type CancelQueuedResult = Result<CancelQueuedOutcome, CancelQueuedRejected>;
/// Result of `recordUsage`.
pub type RecordUsageResult = Result<(), RecordUsageRejected>;
/// Result of `abort`.
pub type AbortResult = Result<AbortSuccess, AbortRejected>;
/// Result of `resume`.
pub type ResumeResult = Result<ResumeOutcome, ResumeRejected>;
/// Result of `createLane`.
pub type CreateLaneResult<M> = Result<Arc<dyn AgentLane<M>>, CreateLaneRejected>;

/// Everything [`AgentHarness::create`] can fail with.
#[derive(Debug, thiserror::Error)]
pub enum HarnessCreateError {
    #[error(transparent)]
    Rejected(#[from] HarnessRejected),
    #[error(transparent)]
    Session(#[from] SessionError),
}

// ---------------------------------------------------------------------------
// Options and supporting types
// ---------------------------------------------------------------------------

/// Options for navigating the session tree.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NavigateOptions {
    pub summarize: Option<bool>,
    pub custom_instructions: Option<String>,
    pub label: Option<String>,
}

/// Options for a manual compaction.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactOptions {
    pub custom_instructions: Option<String>,
}

/// Options for recording usage against an entry.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecordUsageOptions {
    pub entry_id: Option<String>,
    pub details: Option<JsonValue>,
}

/// Why an operation is suspended rather than finished.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SuspensionReason {
    Crash,
    Deferred,
}

/// An operation left suspended by a previous process or by a deferred response.
///
/// Not comparable: [`DeferredHandle`] does not implement `PartialEq`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SuspendedOperation {
    pub lane: String,
    pub kind: OperationKind,
    pub id: String,
    pub started_at: u64,
    pub reason: SuspensionReason,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<Vec<AgentMessage>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deferred: Option<DeferredHandle>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aborting: Option<AbortingState>,
    pub missing: MissingRequirements,
}

/// Queues drained by an abort that is still being reconciled.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AbortingState {
    pub steer: Vec<AgentMessage>,
    pub follow_up: Vec<AgentMessage>,
}

/// Tools and models a suspended operation needs but the harness lacks.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MissingRequirements {
    pub tools: Vec<String>,
    pub models: Vec<String>,
}

/// Status of a lane's open operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LaneOperationStatus {
    Running,
    Suspended,
    Aborting,
}

/// Summary of a lane's open operation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LaneOperationInfo {
    pub id: String,
    pub kind: OperationKind,
    pub status: LaneOperationStatus,
}

/// A lane and its current leaf.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LaneInfo {
    pub name: String,
    pub leaf_id: Option<String>,
    pub operation: Option<LaneOperationInfo>,
}

/// A queued message waiting for a drain point.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QueuedItem {
    pub entry_id: String,
    pub message: AgentMessage,
}

/// A deferred write waiting to be applied.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PendingWrite {
    pub id: String,
    pub entry: ProvisionedEntry,
}

/// The three lane queues.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LaneQueues {
    pub steer: Vec<QueuedItem>,
    pub follow_up: Vec<QueuedItem>,
    pub next_run: Vec<QueuedItem>,
}

/// Snapshot of one lane for [`AgentHarness::watch`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LaneSnapshot {
    pub lane: String,
    pub transcript: Vec<Entry>,
    pub leaf_id: Option<String>,
    pub operation: Option<LaneOperationInfo>,
    pub queues: LaneQueues,
    pub pending_writes: Vec<PendingWrite>,
    pub faulted: bool,
}

/// A lane entry of a [`SessionSnapshot`].
///
/// Not comparable: it carries a [`SuspendedOperation`].
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LaneSummary {
    pub name: String,
    pub leaf_id: Option<String>,
    pub operation: Option<LaneOperationInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suspended: Option<SuspendedOperation>,
}

/// Snapshot of the whole session for [`AgentHarness::watch_session`].
///
/// Not comparable: it carries [`LaneSummary`] values.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionSnapshot {
    pub lanes: Vec<LaneSummary>,
    pub faulted: bool,
}

/// A harness tool: an [`AgentTool`] plus its declared replay policy.
///
/// The TypeScript source intersects `AgentTool` with `{ replay? }`; Rust wraps
/// it. The tool details type is fixed to [`Json`](crate::types::Json), matching
/// the TypeScript default of `any`.
#[derive(Clone, Debug)]
pub struct HarnessTool {
    pub tool: AgentTool,
    pub replay: Option<ReplayPolicy>,
}

impl HarnessTool {
    pub fn new(tool: AgentTool) -> Self {
        HarnessTool { tool, replay: None }
    }

    pub fn with_replay(tool: AgentTool, replay: ReplayPolicy) -> Self {
        HarnessTool {
            tool,
            replay: Some(replay),
        }
    }

    pub fn name(&self) -> &str {
        &self.tool.name
    }
}

/// Skills and prompt templates available to the harness.
pub type Resources = AgentHarnessResources;

/// Curated provider request options owned by the harness.
pub type StreamOptions = AgentHarnessStreamOptions;

/// Per-request stream option patch.
pub type StreamOptionsPatch = AgentHarnessStreamOptionsPatch;

/// Whether the harness drives runs automatically or waits for `executeAction`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriveMode {
    #[default]
    Automatic,
    Manual,
}

/// Static system prompt or an async provider of one.
pub enum SystemPromptSource {
    Static(String),
    Provider(Arc<dyn SystemPromptProvider>),
}

/// Async provider of the system prompt.
#[async_trait::async_trait]
pub trait SystemPromptProvider: Send + Sync {
    async fn system_prompt(&self) -> String;
}

/// Projects one session entry into context messages.
#[async_trait::async_trait]
pub trait EntryProjector: Send + Sync {
    async fn project(&self, entry: &Entry) -> Vec<AgentMessage>;
}

/// Handler invoked for one hook registration.
#[async_trait::async_trait]
pub trait HookHandler: Send + Sync {
    /// Handle one hook invocation. The event is opaque JSON, mirroring the
    /// TypeScript `unknown` event, and the return value is whatever the handler
    /// produced.
    async fn handle(&self, event: Json) -> Json;
}

/// Listener invoked for one delivered harness event.
#[async_trait::async_trait]
pub trait EventListener: Send + Sync {
    async fn on_event(&self, event: Json);
}

/// Handle that removes a hook or event subscription.
pub type Unsubscribe = Arc<dyn Fn() + Send + Sync>;

/// Registration options for [`Hooks::on`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HookRegistrationOptions {
    pub id: Option<String>,
}

/// Hook registry.
pub trait Hooks: Send + Sync {
    /// Register `handler` for `name`; the returned handle unsubscribes.
    fn on(
        &self,
        name: HookName,
        handler: Arc<dyn HookHandler>,
        options: Option<HookRegistrationOptions>,
    ) -> Result<Unsubscribe, HarnessRejected>;
}

/// Passive event listener registry.
pub trait Events: Send + Sync {
    /// Register `listener` for `event_type`; the returned handle unsubscribes.
    fn on(
        &self,
        event_type: &str,
        listener: Arc<dyn EventListener>,
    ) -> Result<Unsubscribe, HarnessRejected>;
}

/// A live subscription to a lane or session snapshot.
pub struct WatchHandle<TSnapshot> {
    pub snapshot: TSnapshot,
    /// Begin delivering events to `listener`.
    pub start: Arc<dyn Fn(Arc<dyn EventListener>) + Send + Sync>,
    /// Stop delivering events and release the subscription.
    pub unsubscribe: Arc<dyn Fn() + Send + Sync>,
}

/// Options for constructing an [`AgentHarness`].
pub struct AgentHarnessOptions<
    M: Clone + Send + Sync + 'static,
    C: TelemetryContext = NoopTelemetryContext,
> {
    pub session: Session<M>,
    pub models: Models,
    pub model: Model,
    pub thinking_level: Option<ModelThinkingLevel>,
    pub active_tool_names: Option<Vec<String>>,
    pub tools: Vec<HarnessTool>,
    pub tool_context: Option<ToolContextSource<Json>>,
    pub system_prompt: Option<SystemPromptSource>,
    pub resources: Option<Resources>,
    pub stream_options: Option<StreamOptions>,
    pub retry: Option<RetryPolicy>,
    pub compaction: Option<CompactionSettings>,
    pub steering_mode: Option<QueueMode>,
    pub follow_up_mode: Option<QueueMode>,
    pub tool_execution: Option<ToolExecutionMode>,
    pub drive: Option<DriveMode>,
    pub to_provider_messages: Option<Arc<dyn ConvertToLlm>>,
    pub entry_projectors: HashMap<String, Arc<dyn EntryProjector>>,
    pub context: Option<Arc<C>>,
}

// ---------------------------------------------------------------------------
// Hook and action vocabularies
// ---------------------------------------------------------------------------

/// Hook names, in TypeScript declaration order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HookName {
    BeforeRun,
    BeforeResume,
    BeforeRunEnd,
    TransformContext,
    BeforeRequest,
    BeforePayload,
    AfterResponse,
    BeforeTool,
    AfterTool,
    BeforeCompaction,
    BeforeNavigation,
}

impl HookName {
    /// Wire name of this hook.
    pub fn as_str(&self) -> &'static str {
        match self {
            HookName::BeforeRun => "before_run",
            HookName::BeforeResume => "before_resume",
            HookName::BeforeRunEnd => "before_run_end",
            HookName::TransformContext => "transform_context",
            HookName::BeforeRequest => "before_request",
            HookName::BeforePayload => "before_payload",
            HookName::AfterResponse => "after_response",
            HookName::BeforeTool => "before_tool",
            HookName::AfterTool => "after_tool",
            HookName::BeforeCompaction => "before_compaction",
            HookName::BeforeNavigation => "before_navigation",
        }
    }

    /// Every hook name, in declaration order.
    pub const ALL: [HookName; 11] = [
        HookName::BeforeRun,
        HookName::BeforeResume,
        HookName::BeforeRunEnd,
        HookName::TransformContext,
        HookName::BeforeRequest,
        HookName::BeforePayload,
        HookName::AfterResponse,
        HookName::BeforeTool,
        HookName::AfterTool,
        HookName::BeforeCompaction,
        HookName::BeforeNavigation,
    ];
}

/// Outcome of `try_finish_run`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TryFinishRunOutcome {
    Completed,
    Failed,
}

/// The next unit of harness work, as reported by `peekAction`/`executeAction`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ActionInfo {
    AppendEntry {
        entry_type: String,
        entry_id: String,
    },
    AppendRecord {
        record_type: String,
    },
    MoveLane {
        to: Option<String>,
    },
    SetFact {
        fact: FactName,
    },
    TryFinishRun {
        outcome: TryFinishRunOutcome,
    },
    FinishOperation {
        outcome: OperationOutcome,
    },
    CommitFollowUp,
    ConsumeQueueItem {
        queue: QueueKind,
        entry_id: String,
    },
    ApplyPendingWrite {
        entry_id: String,
    },
    StreamAssistant {
        step: StepKind,
        attempt: u32,
    },
    ExecuteTool {
        tool_call_id: String,
        tool_name: String,
    },
    FetchDeferred {
        provider: String,
        id: String,
    },
    CancelDeferred {
        provider: String,
        id: String,
    },
    Hook {
        name: HookName,
    },
    Sleep {
        delay_ms: u64,
    },
}

/// Callback invoked by [`AgentLane::run_when_idle`] once the lane is idle.
#[async_trait::async_trait]
pub trait IdleCallback: Send + Sync {
    async fn run(&self);
}

// ---------------------------------------------------------------------------
// The lane contract
// ---------------------------------------------------------------------------

/// One lane of a session: the durable unit the harness operates on.
///
/// `AgentHarness` implements this for its own `main` lane.
#[async_trait::async_trait]
pub trait AgentLane<M: Clone + Send + Sync + 'static>: Send + Sync {
    /// Lane name.
    fn name(&self) -> &str;

    /// The session this lane reads and writes.
    fn session(&self) -> &Session<M>;

    /// The lane's current leaf entry id.
    async fn get_leaf_id(&self) -> Result<Option<String>, SessionError>;

    /// Start a run from text or prepared messages.
    async fn prompt(&self, input: PromptInput, images: Option<Vec<ImageContent>>) -> RunResult;

    /// Start a run from a registered skill.
    async fn skill(&self, name: &str, additional_instructions: Option<&str>) -> RunResult;

    /// Start a run from a registered prompt template.
    async fn prompt_from_template(&self, name: &str, args: Option<&[String]>) -> RunResult;

    /// Compact the lane's context.
    async fn compact(&self, options: Option<&CompactOptions>) -> CompactionResult;

    /// Navigate the session tree, optionally summarizing the abandoned branch.
    async fn navigate_tree(
        &self,
        target_id: Option<&str>,
        options: Option<&NavigateOptions>,
    ) -> NavigationResult;

    /// Resume the lane's suspended operation.
    async fn resume(&self) -> ResumeResult;

    /// Abort the lane's open operation and drain its queues.
    async fn abort(&self) -> AbortResult;

    /// Queue a message to inject mid-run.
    async fn steer(&self, input: PromptInput, images: Option<Vec<ImageContent>>) -> QueueResult;

    /// Queue a message to process after the run would otherwise stop.
    async fn follow_up(&self, input: PromptInput, images: Option<Vec<ImageContent>>)
    -> QueueResult;

    /// Queue a message for the next run.
    async fn next_run(&self, input: PromptInput, images: Option<Vec<ImageContent>>) -> QueueResult;

    /// Cancel a queued message.
    async fn cancel_queued(&self, entry_id: &str) -> CancelQueuedResult;

    /// Record usage against an entry.
    async fn record_usage(
        &self,
        usage: &Usage,
        options: Option<&RecordUsageOptions>,
    ) -> RecordUsageResult;

    /// Resolve once the lane is idle.
    async fn wait_for_idle(&self) -> Result<(), HarnessRejected>;

    /// Run `callback` once the lane is idle.
    async fn run_when_idle(&self, callback: Arc<dyn IdleCallback>) -> Result<(), HarnessRejected>;

    /// The next action without executing it.
    async fn peek_action(&self) -> Result<Option<ActionInfo>, HarnessRejected>;

    /// Execute and return the next action.
    async fn execute_action(&self) -> Result<Option<ActionInfo>, HarnessRejected>;

    /// Drive the lane until it is idle.
    async fn run_to_completion(&self) -> Result<(), HarnessRejected>;

    /// The model future turns use.
    async fn get_model(&self) -> Model;

    /// Replace the model future turns use.
    async fn set_model(&self, model: Model);

    /// The thinking level future turns use.
    async fn get_thinking_level(&self) -> ModelThinkingLevel;

    /// Replace the thinking level future turns use.
    async fn set_thinking_level(&self, level: ModelThinkingLevel);

    /// The active tool names.
    async fn get_active_tools(&self) -> Vec<String>;

    /// Replace the active tool names.
    async fn set_active_tools(&self, names: Vec<String>);

    /// Subscribe to lane snapshots.
    async fn watch(&self) -> Result<WatchHandle<LaneSnapshot>, HarnessRejected>;
}

// ---------------------------------------------------------------------------
// The harness
// ---------------------------------------------------------------------------

/// Registry that refuses every subscription: the contract layer has no hooks or
/// events to deliver yet.
struct UnavailableRegistry {
    operation: &'static str,
    closed: Arc<AtomicBool>,
}

impl UnavailableRegistry {
    fn refusal(&self) -> HarnessRejected {
        if self.closed.load(Ordering::SeqCst) {
            HarnessRejected::Closed(Closed::after_close())
        } else {
            HarnessRejected::NotImplemented(HarnessNotImplemented::new(self.operation))
        }
    }
}

impl Hooks for UnavailableRegistry {
    fn on(
        &self,
        _name: HookName,
        _handler: Arc<dyn HookHandler>,
        _options: Option<HookRegistrationOptions>,
    ) -> Result<Unsubscribe, HarnessRejected> {
        Err(self.refusal())
    }
}

impl Events for UnavailableRegistry {
    fn on(
        &self,
        _event_type: &str,
        _listener: Arc<dyn EventListener>,
    ) -> Result<Unsubscribe, HarnessRejected> {
        Err(self.refusal())
    }
}

/// Mutable harness configuration behind one lock.
#[derive(Clone, Debug)]
struct Configuration {
    model: Model,
    thinking_level: ModelThinkingLevel,
    active_tool_names: Vec<String>,
    tools: Vec<HarnessTool>,
    resources: Resources,
    stream_options: StreamOptions,
    retry_policy: RetryPolicy,
    compaction_settings: CompactionSettings,
    steering_mode: QueueMode,
    follow_up_mode: QueueMode,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().expect("harness mutex poisoned")
}

/// Stateful agent harness: the durable-session front end for one process.
///
/// The execution engine is not part of this contract layer, so every method
/// that would run work returns [`HarnessRejected::NotImplemented`] (or
/// [`HarnessRejected::Closed`] after [`AgentHarness::close`]). Configuration
/// management and `close` are real.
pub struct AgentHarness<
    M: Clone + Send + Sync + 'static,
    C: TelemetryContext = NoopTelemetryContext,
> {
    name: String,
    session: Session<M>,
    hooks: UnavailableRegistry,
    events: UnavailableRegistry,
    configuration: Mutex<Configuration>,
    closed: Arc<AtomicBool>,
    telemetry: Option<Arc<C>>,
}

impl<M: Clone + Send + Sync + 'static, C: TelemetryContext> AgentHarness<M, C> {
    /// Whether this contract layer includes an execution engine. Check this
    /// before presenting run/resume/compaction actions to users.
    pub const fn supports_execution(&self) -> bool {
        false
    }

    /// Restore or create a harness over `options.session`.
    ///
    /// Restoring an existing record log is not implemented yet: a session with
    /// any records is refused, mirroring the TypeScript `create.restore` guard.
    pub async fn create(
        options: AgentHarnessOptions<M, C>,
    ) -> Result<(Arc<Self>, Vec<SuspendedOperation>), HarnessCreateError> {
        let records = options
            .session
            .find_records(&RecordQuery {
                limit: Some(1),
                ..RecordQuery::default()
            })
            .await?;
        if !records.is_empty() {
            return Err(HarnessRejected::NotImplemented(HarnessNotImplemented::new(
                "create.restore",
            ))
            .into());
        }
        Ok((Arc::new(Self::new(options)), Vec::new()))
    }

    fn new(options: AgentHarnessOptions<M, C>) -> Self {
        let closed = Arc::new(AtomicBool::new(false));
        let hooks = UnavailableRegistry {
            operation: "hooks.on",
            closed: Arc::clone(&closed),
        };
        let events = UnavailableRegistry {
            operation: "events.on",
            closed: Arc::clone(&closed),
        };
        let active_tool_names = options.active_tool_names.clone().or_else(|| {
            (!options.tools.is_empty()).then(|| {
                options
                    .tools
                    .iter()
                    .map(|tool| tool.tool.name.clone())
                    .collect()
            })
        });
        AgentHarness {
            name: "main".to_owned(),
            session: options.session,
            hooks,
            events,
            configuration: Mutex::new(Configuration {
                model: options.model,
                thinking_level: options.thinking_level.unwrap_or(ModelThinkingLevel::Off),
                active_tool_names: active_tool_names.unwrap_or_default(),
                tools: options.tools,
                resources: options.resources.unwrap_or_default(),
                stream_options: options.stream_options.unwrap_or_default(),
                retry_policy: options.retry.unwrap_or(RetryPolicy {
                    enabled: false,
                    max_retries: 0,
                    base_delay_ms: 1000,
                }),
                compaction_settings: options
                    .compaction
                    .unwrap_or(crate::compaction::DEFAULT_COMPACTION_SETTINGS),
                steering_mode: options.steering_mode.unwrap_or_default(),
                follow_up_mode: options.follow_up_mode.unwrap_or_default(),
            }),
            closed,
            telemetry: options.context,
        }
    }

    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    /// The error for a call this contract layer cannot serve.
    fn unavailable<T, E: UnavailableError>(&self, operation: &'static str) -> Result<T, E> {
        Err(E::unavailable(operation, self.is_closed()))
    }

    /// Hook registry. Every registration is refused until the execution engine
    /// exists.
    pub fn hooks(&self) -> &dyn Hooks {
        &self.hooks
    }

    /// Event listener registry. Every subscription is refused until the
    /// execution engine exists.
    pub fn events(&self) -> &dyn Events {
        &self.events
    }

    /// The telemetry context this harness records against, when configured.
    pub fn telemetry_context(&self) -> Option<&C> {
        self.telemetry.as_deref()
    }

    /// The lane handle for `name`.
    pub async fn lane(
        &self,
        _name: &str,
    ) -> Result<Option<Arc<dyn AgentLane<M>>>, HarnessRejected> {
        self.unavailable("lane")
    }

    /// Create a new lane at `at` (the current leaf when `None`).
    pub async fn create_lane(&self, _name: &str, _at: Option<&str>) -> CreateLaneResult<M> {
        self.unavailable("createLane")
    }

    /// Every lane in the session.
    pub async fn lanes(&self) -> Result<Vec<LaneInfo>, HarnessRejected> {
        self.unavailable("lanes")
    }

    /// The registered tools.
    pub fn get_tools(&self) -> Vec<HarnessTool> {
        lock(&self.configuration).tools.clone()
    }

    /// Replace the tools; `active_names` defaults to every tool name.
    pub fn set_tools(&self, tools: Vec<HarnessTool>, active_names: Option<Vec<String>>) {
        let mut configuration = lock(&self.configuration);
        configuration.active_tool_names = active_names
            .unwrap_or_else(|| tools.iter().map(|tool| tool.tool.name.clone()).collect());
        configuration.tools = tools;
    }

    /// The registered skills and prompt templates.
    pub fn get_resources(&self) -> Resources {
        lock(&self.configuration).resources.clone()
    }

    /// Replace the registered skills and prompt templates.
    pub fn set_resources(&self, resources: Resources) {
        lock(&self.configuration).resources = resources;
    }

    /// The curated provider request options.
    pub fn get_stream_options(&self) -> StreamOptions {
        lock(&self.configuration).stream_options.clone()
    }

    /// Replace the curated provider request options.
    pub fn set_stream_options(&self, options: StreamOptions) {
        lock(&self.configuration).stream_options = options;
    }

    /// The retry policy.
    pub fn get_retry_policy(&self) -> RetryPolicy {
        lock(&self.configuration).retry_policy
    }

    /// Replace the retry policy.
    pub fn set_retry_policy(&self, policy: RetryPolicy) {
        lock(&self.configuration).retry_policy = policy;
    }

    /// The compaction settings.
    pub fn get_compaction_settings(&self) -> CompactionSettings {
        lock(&self.configuration).compaction_settings
    }

    /// Replace the compaction settings.
    pub fn set_compaction_settings(&self, settings: CompactionSettings) {
        lock(&self.configuration).compaction_settings = settings;
    }

    /// The steering queue mode.
    pub fn get_steering_mode(&self) -> QueueMode {
        lock(&self.configuration).steering_mode
    }

    /// Replace the steering queue mode.
    pub fn set_steering_mode(&self, mode: QueueMode) {
        lock(&self.configuration).steering_mode = mode;
    }

    /// The follow-up queue mode.
    pub fn get_follow_up_mode(&self) -> QueueMode {
        lock(&self.configuration).follow_up_mode
    }

    /// Replace the follow-up queue mode.
    pub fn set_follow_up_mode(&self, mode: QueueMode) {
        lock(&self.configuration).follow_up_mode = mode;
    }

    /// Subscribe to session snapshots.
    pub async fn watch_session(&self) -> Result<WatchHandle<SessionSnapshot>, HarnessRejected> {
        self.unavailable("watchSession")
    }

    /// Refuse further work. Idempotent.
    pub fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
    }
}

#[async_trait::async_trait]
impl<M: Clone + Send + Sync + 'static, C: TelemetryContext> AgentLane<M> for AgentHarness<M, C> {
    fn name(&self) -> &str {
        &self.name
    }

    fn session(&self) -> &Session<M> {
        &self.session
    }

    async fn get_leaf_id(&self) -> Result<Option<String>, SessionError> {
        self.session.get_leaf_id().await
    }

    async fn prompt(&self, _input: PromptInput, _images: Option<Vec<ImageContent>>) -> RunResult {
        self.unavailable("prompt")
    }

    async fn skill(&self, _name: &str, _additional_instructions: Option<&str>) -> RunResult {
        self.unavailable("skill")
    }

    async fn prompt_from_template(&self, _name: &str, _args: Option<&[String]>) -> RunResult {
        self.unavailable("promptFromTemplate")
    }

    async fn compact(&self, _options: Option<&CompactOptions>) -> CompactionResult {
        self.unavailable("compact")
    }

    async fn navigate_tree(
        &self,
        _target_id: Option<&str>,
        _options: Option<&NavigateOptions>,
    ) -> NavigationResult {
        self.unavailable("navigateTree")
    }

    async fn resume(&self) -> ResumeResult {
        self.unavailable("resume")
    }

    async fn abort(&self) -> AbortResult {
        self.unavailable("abort")
    }

    async fn steer(&self, _input: PromptInput, _images: Option<Vec<ImageContent>>) -> QueueResult {
        self.unavailable("steer")
    }

    async fn follow_up(
        &self,
        _input: PromptInput,
        _images: Option<Vec<ImageContent>>,
    ) -> QueueResult {
        self.unavailable("followUp")
    }

    async fn next_run(
        &self,
        _input: PromptInput,
        _images: Option<Vec<ImageContent>>,
    ) -> QueueResult {
        self.unavailable("nextRun")
    }

    async fn cancel_queued(&self, _entry_id: &str) -> CancelQueuedResult {
        self.unavailable("cancelQueued")
    }

    async fn record_usage(
        &self,
        _usage: &Usage,
        _options: Option<&RecordUsageOptions>,
    ) -> RecordUsageResult {
        self.unavailable("recordUsage")
    }

    async fn wait_for_idle(&self) -> Result<(), HarnessRejected> {
        self.unavailable("waitForIdle")
    }

    async fn run_when_idle(&self, _callback: Arc<dyn IdleCallback>) -> Result<(), HarnessRejected> {
        self.unavailable("runWhenIdle")
    }

    async fn peek_action(&self) -> Result<Option<ActionInfo>, HarnessRejected> {
        self.unavailable("peekAction")
    }

    async fn execute_action(&self) -> Result<Option<ActionInfo>, HarnessRejected> {
        self.unavailable("executeAction")
    }

    async fn run_to_completion(&self) -> Result<(), HarnessRejected> {
        self.unavailable("runToCompletion")
    }

    async fn get_model(&self) -> Model {
        lock(&self.configuration).model.clone()
    }

    async fn set_model(&self, model: Model) {
        lock(&self.configuration).model = model;
    }

    async fn get_thinking_level(&self) -> ModelThinkingLevel {
        lock(&self.configuration).thinking_level
    }

    async fn set_thinking_level(&self, level: ModelThinkingLevel) {
        lock(&self.configuration).thinking_level = level;
    }

    async fn get_active_tools(&self) -> Vec<String> {
        lock(&self.configuration).active_tool_names.clone()
    }

    async fn set_active_tools(&self, names: Vec<String>) {
        lock(&self.configuration).active_tool_names = names;
    }

    async fn watch(&self) -> Result<WatchHandle<LaneSnapshot>, HarnessRejected> {
        self.unavailable("watch")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::types::AbortSignal;
    use crate::session::memory::InMemorySessionStorage;
    use crate::session::types::SessionMetadata;
    use crate::types::{AgentToolResult, ToolError, ToolExecutor, ToolUpdateCallback, tool_error};
    use step_harness_providers::CreateModelsOptions;
    use step_harness_providers::types::{Api, ProviderId};

    type TestHarness = AgentHarness<SessionMetadata, NoopTelemetryContext>;

    fn test_model() -> Model {
        Model {
            id: "test-model".to_owned(),
            name: "Test".to_owned(),
            api: Api::AnthropicMessages,
            provider: ProviderId::new("anthropic"),
            base_url: String::new(),
            reasoning: false,
            thinking_level_map: None,
            input: Vec::new(),
            cost: Default::default(),
            context_window: 0,
            max_tokens: 0,
            sampling_params: None,
            headers: None,
            compat: None,
        }
    }

    struct NoopToolExecutor;

    #[async_trait::async_trait]
    impl ToolExecutor for NoopToolExecutor {
        async fn execute(
            &self,
            _tool_call_id: &str,
            _params: &serde_json::Map<String, Json>,
            _signal: Option<&AbortSignal>,
            _on_update: Option<&ToolUpdateCallback>,
        ) -> Result<AgentToolResult, ToolError> {
            Err(tool_error("noop"))
        }
    }

    fn test_tool(name: &str) -> HarnessTool {
        HarnessTool::new(AgentTool {
            name: name.to_owned(),
            label: name.to_owned(),
            description: "test tool".to_owned(),
            parameters: serde_json::json!({"type": "object"}),
            execution_mode: None,
            prepare_arguments: None,
            executor: Arc::new(NoopToolExecutor),
        })
    }

    struct NoopHookHandler;

    #[async_trait::async_trait]
    impl HookHandler for NoopHookHandler {
        async fn handle(&self, _event: Json) -> Json {
            Json::Null
        }
    }

    struct NoopEventListener;

    #[async_trait::async_trait]
    impl EventListener for NoopEventListener {
        async fn on_event(&self, _event: Json) {}
    }

    async fn test_harness(tools: Vec<HarnessTool>) -> Arc<TestHarness> {
        let storage = Arc::new(InMemorySessionStorage::new(SessionMetadata {
            id: "session-1".to_owned(),
            created_at: 0,
            parent_session_id: None,
        }));
        let options: AgentHarnessOptions<SessionMetadata, NoopTelemetryContext> =
            AgentHarnessOptions {
                session: Session::new(storage, None),
                models: Models::new(CreateModelsOptions::default()),
                model: test_model(),
                thinking_level: None,
                active_tool_names: None,
                tools,
                tool_context: None,
                system_prompt: None,
                resources: None,
                stream_options: None,
                retry: None,
                compaction: None,
                steering_mode: None,
                follow_up_mode: None,
                tool_execution: None,
                drive: None,
                to_provider_messages: None,
                entry_projectors: HashMap::new(),
                context: None,
            };
        let (harness, suspended) = AgentHarness::create(options).await.expect("harness");
        assert!(suspended.is_empty());
        harness
    }

    #[test]
    fn every_tag_has_a_rejection_variant() {
        let errors: Vec<HarnessRejected> = vec![
            LaneBusy {
                lane: "main".to_owned(),
                operation_id: "op1".to_owned(),
                operation_kind: OperationKind::Run,
                message: "busy".to_owned(),
            }
            .into(),
            MissingIdentities {
                lane: "main".to_owned(),
                tools: vec!["bash".to_owned()],
                models: Vec::new(),
                message: "missing".to_owned(),
            }
            .into(),
            NoActiveRun {
                lane: "main".to_owned(),
                message: "no run".to_owned(),
            }
            .into(),
            NoActiveOperation {
                lane: "main".to_owned(),
                message: "no operation".to_owned(),
            }
            .into(),
            NothingToResume {
                lane: "main".to_owned(),
                message: "nothing".to_owned(),
            }
            .into(),
            InvalidMessage {
                lane: "main".to_owned(),
                reason: "bad".to_owned(),
                message: "invalid".to_owned(),
            }
            .into(),
            UnknownSkill {
                name: "skill".to_owned(),
                message: "unknown".to_owned(),
            }
            .into(),
            UnknownTemplate {
                name: "template".to_owned(),
                message: "unknown".to_owned(),
            }
            .into(),
            UnknownTarget {
                target_id: "entry".to_owned(),
                message: "unknown".to_owned(),
            }
            .into(),
            UnknownQueueItem {
                lane: "main".to_owned(),
                entry_id: "entry".to_owned(),
                message: "unknown".to_owned(),
            }
            .into(),
            LaneExists {
                lane: "main".to_owned(),
                message: "exists".to_owned(),
            }
            .into(),
            InvalidLane {
                lane: "bad name".to_owned(),
                reason: "empty".to_owned(),
                message: "invalid".to_owned(),
            }
            .into(),
            NothingToCompact {
                lane: "main".to_owned(),
                message: "nothing".to_owned(),
            }
            .into(),
            Closed {
                message: "closed".to_owned(),
            }
            .into(),
            HarnessNotImplemented::new("prompt").into(),
        ];

        // Exhaustive match: one arm per tag, the Rust counterpart of `matchError`.
        let mut tags: Vec<String> = Vec::new();
        for error in &errors {
            tags.push(match error {
                HarnessRejected::LaneBusy(inner) => format!("lane_busy:{}", inner.lane),
                HarnessRejected::MissingIdentities(inner) => {
                    format!("missing_identities:{}", inner.tools.join(","))
                }
                HarnessRejected::NoActiveRun(inner) => format!("no_active_run:{}", inner.lane),
                HarnessRejected::NoActiveOperation(inner) => {
                    format!("no_active_operation:{}", inner.lane)
                }
                HarnessRejected::NothingToResume(inner) => {
                    format!("nothing_to_resume:{}", inner.lane)
                }
                HarnessRejected::InvalidMessage(inner) => {
                    format!("invalid_message:{}", inner.reason)
                }
                HarnessRejected::UnknownSkill(inner) => format!("unknown_skill:{}", inner.name),
                HarnessRejected::UnknownTemplate(inner) => {
                    format!("unknown_template:{}", inner.name)
                }
                HarnessRejected::UnknownTarget(inner) => {
                    format!("unknown_target:{}", inner.target_id)
                }
                HarnessRejected::UnknownQueueItem(inner) => {
                    format!("unknown_queue_item:{}", inner.entry_id)
                }
                HarnessRejected::LaneExists(inner) => format!("lane_exists:{}", inner.lane),
                HarnessRejected::InvalidLane(inner) => format!("invalid_lane:{}", inner.reason),
                HarnessRejected::NothingToCompact(inner) => {
                    format!("nothing_to_compact:{}", inner.lane)
                }
                HarnessRejected::Closed(inner) => format!("closed:{}", inner.message),
                HarnessRejected::NotImplemented(inner) => {
                    format!("not_implemented:{}", inner.operation)
                }
            });
        }
        assert_eq!(tags.len(), 15);
        assert_eq!(tags[0], "lane_busy:main");
        assert_eq!(tags[1], "missing_identities:bash");
        assert_eq!(tags[14], "not_implemented:prompt");
        assert_eq!(
            errors[14].to_string(),
            "AgentHarness.prompt is not implemented yet"
        );
        assert_eq!(errors[13].to_string(), "closed");

        // Each per-call union accepts its own tags plus the scaffold refusal.
        let run: RunRejected = Closed::after_close().into();
        assert!(matches!(run, RunRejected::Closed(_)));
        let queue: QueueRejected = HarnessNotImplemented::new("steer").into();
        assert!(matches!(queue, QueueRejected::NotImplemented(_)));
        let usage: RecordUsageRejected = Closed::after_close().into();
        assert!(matches!(usage, RecordUsageRejected::Closed(_)));
        let abort: AbortRejected = NoActiveOperation {
            lane: "main".to_owned(),
            message: "none".to_owned(),
        }
        .into();
        assert!(matches!(abort, AbortRejected::NoActiveOperation(_)));
    }

    #[tokio::test]
    async fn a_closed_harness_refuses_calls() {
        let harness = test_harness(vec![test_tool("bash")]).await;
        assert_eq!(harness.name(), "main");
        assert_eq!(harness.get_leaf_id().await.unwrap(), None);

        let error = harness
            .prompt(PromptInput::Text("hi".to_owned()), None)
            .await
            .unwrap_err();
        assert!(matches!(error, RunRejected::NotImplemented(_)));
        assert!(
            harness
                .hooks()
                .on(HookName::BeforeRun, Arc::new(NoopHookHandler), None)
                .is_err()
        );
        assert!(
            harness
                .events()
                .on("run_start", Arc::new(NoopEventListener))
                .is_err()
        );
        assert!(harness.lanes().await.is_err());

        harness.close();
        let error = harness
            .prompt(PromptInput::Text("hi".to_owned()), None)
            .await
            .unwrap_err();
        assert!(matches!(error, RunRejected::Closed(_)));
        let error = harness.wait_for_idle().await.unwrap_err();
        assert!(matches!(error, HarnessRejected::Closed(_)));
        // Configuration stays readable after close.
        assert_eq!(harness.get_active_tools().await, vec!["bash".to_owned()]);
    }

    #[tokio::test]
    async fn configuration_round_trips() {
        let harness = test_harness(vec![test_tool("bash"), test_tool("read")]).await;
        // Tool names are the default active set.
        assert_eq!(
            harness.get_active_tools().await,
            vec!["bash".to_owned(), "read".to_owned()]
        );

        harness.set_active_tools(vec!["read".to_owned()]).await;
        assert_eq!(harness.get_active_tools().await, vec!["read".to_owned()]);

        harness.set_model(test_model()).await;
        assert_eq!(harness.get_model().await.id, "test-model");
        harness.set_thinking_level(ModelThinkingLevel::High).await;
        assert_eq!(harness.get_thinking_level().await, ModelThinkingLevel::High);

        harness.set_tools(vec![test_tool("bash")], None);
        assert_eq!(harness.get_tools().len(), 1);
        assert_eq!(harness.get_active_tools().await, vec!["bash".to_owned()]);

        harness.set_resources(Resources::default());
        assert_eq!(harness.get_resources().skills, None);
        harness.set_stream_options(StreamOptions::default());
        assert_eq!(harness.get_stream_options().timeout_ms, None);
        harness.set_retry_policy(RetryPolicy {
            enabled: true,
            max_retries: 3,
            base_delay_ms: 10,
        });
        assert_eq!(harness.get_retry_policy().max_retries, 3);
        harness.set_compaction_settings(CompactionSettings {
            enabled: false,
            reserve_tokens: 1,
            keep_recent_tokens: 2,
        });
        assert_eq!(harness.get_compaction_settings().reserve_tokens, 1);
        harness.set_steering_mode(QueueMode::All);
        assert_eq!(harness.get_steering_mode(), QueueMode::All);
        harness.set_follow_up_mode(QueueMode::All);
        assert_eq!(harness.get_follow_up_mode(), QueueMode::All);
    }
}
