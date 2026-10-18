//! Durable sessions: an append-only mutation log with lanes, records, and
//! pluggable storage backends.
//!
//! Ported from `@step-harness/agent-core` `src\harness\session\`. Every durable
//! write is a [`SessionMutation`] carrying a globally consecutive sequence
//! number; [`SessionState`](state::SessionState) is the single state authority
//! and [`JsonlSessionStorage`](jsonl::JsonlSessionStorage) persists the log as
//! one JSON object per line.

pub mod context;
pub mod facade;
pub mod jsonl;
pub mod memory;
pub mod state;
pub mod types;

pub use context::{
    ContextEntryTransform, CustomEntryContextMessageProjector, SessionContext,
    SessionContextBuildOptions, SessionModel, build_context_entries, build_session_context,
    default_context_entry_transform, session_entry_to_context_messages,
};
pub use facade::{LaneView, Session, SessionRepo, SessionStorage};
pub use jsonl::{
    JsonlDecodeError, JsonlDecodeErrorKind, JsonlSessionCreateOptions, JsonlSessionListOptions,
    JsonlSessionMetadata, JsonlSessionRepo, JsonlSessionRepoOptions, JsonlSessionStorage,
    JsonlV4Header,
};
pub use memory::{InMemorySessionRepo, InMemorySessionStorage};
pub use state::{FactKind, SessionMutation, SessionState};
pub use types::{
    ActiveToolsEntry, BranchBounds, BranchEntryQuery, BranchSummaryEntry, CompactionEntry,
    CompactionReason, CustomEntry, Entry, EntryCursor, EntryOrder, EntryQuery, ForkOptions,
    ForkPosition, IdGenerator, JsonValue, LanePointer, LaneRecord, LogItem, LogOptions,
    MessageEntry, ModelChangeEntry, NewRecord, OperationError, OperationFinishedRecord,
    OperationIntent, OperationOutcome, OperationStartedRecord, ProvisionedEntry, QueueKind,
    RecordQuery, ReplayPolicy, SessionCreateOptions, SessionError, SessionErrorCode,
    SessionMetadata, SessionStats, SessionStopReason, StepAttemptRecord, StepKind,
    ThinkingLevelEntry, ToolStartedRecord, UsageCause, UsageRecord, UuidV7Generator,
    assert_json_serializable, assert_valid_limit, assert_valid_operation_kind,
};
