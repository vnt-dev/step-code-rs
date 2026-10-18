//! `step-harness-coding-agent`: the session-level agent runtime.
//!
//! This crate is the Rust migration of the `@step-harness/coding-agent`
//! package's non-CLI layers. It sits on top of [`step_harness_agent_core`]
//! (the `Agent`, agent loop, tools, sessions, and compaction ported from
//! `@step-harness/agent-core`) and adds everything a hosted UI needs:
//!
//! - [`paths`] — the on-disk layout of the new data directory.
//! - [`settings`] — the settings schema, file loading, and merge precedence.
//! - [`auth`] — the file-backed `auth.json` credential store.
//! - [`models`] — `models.json` configuration, catalog resolution, and the
//!   per-session model/thinking-level controller.
//! - [`permissions`] — approval policy and shell command analysis.
//! - [`plan`] — plan mode: the research-then-propose flow, its state
//!   persistence, and the plan review contract.
//! - [`resources`] — context files (`AGENTS.md`), skills, and prompt templates.
//! - [`session`] — the [`session::AgentSession`] orchestrator: the turn
//!   pipeline, event stream, persistence, and session lifecycle.
//!
//! # Notable differences from the TypeScript source
//!
//! - No CLI, print mode, RPC mode, terminal UI, or extension system: the
//!   embedder (a Tauri desktop app) provides the UI and calls this crate
//!   directly.
//! - The data directory is new (`~/.step-rs/`); legacy formats are not read.

pub mod auth;
pub mod models;
pub mod paths;
pub mod permissions;
pub mod plan;
pub mod remote_mcp;
pub mod resources;
pub mod search_web;
pub mod session;
pub mod settings;
pub mod system_prompt;

pub use auth::{AuthStore, FileCredentialStore};
pub use models::{ModelCatalog, ModelController, ModelSwitchError, ModelsConfigFile};
pub use paths::AppPaths;
pub use permissions::{ApprovalMode, Decision, PermissionPolicy};
pub use plan::{
    PlanReviewChannel, PlanReviewOutcome, PlanReviewRequest, PlanSource, PlanState,
    SharedPlanReviewChannel,
};
pub use session::{
    AgentSession, ApprovalChannel, ApprovalRequest, CompactionReason, SessionEvent, SessionEventRx,
    SessionFailure, SessionRuntime, SessionServices, SessionStats, SharedApprovalChannel,
};
