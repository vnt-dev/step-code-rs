//! The agent session orchestrator.
//!
//! Ported functionally from `@step-harness/coding-agent`
//! `src/core/agent-session.ts`, `src/core/agent-session-services.ts`, and the
//! session layer of `src/core/session-manager.ts`.
//!
//! A session owns the [`Agent`], its durable entry store, the per-session
//! model/thinking state, the approval policy, and an event stream for the
//! embedder. The turn pipeline mirrors the TypeScript order: validate, persist,
//! run, then loop over retry / auto-compaction / queue drains until settled.

pub mod events;
mod naming;
pub mod runtime;
pub mod stats;
pub mod tools;

pub use crate::plan::{
    AutoApprovePlanReview, AutoDismissPlanReview, PlanReviewChannel, PlanReviewOutcome,
    PlanReviewRequest, PlanSource, PlanState, SharedPlanReviewChannel,
};
pub use events::{
    ApprovalChannel, ApprovalRequest, AutoApprove, AutoDeny, CompactionReason, ExecutionTiming,
    FnApproval, SessionEvent, SessionEventRx, SharedApprovalChannel,
};
pub use runtime::SessionRuntime;
pub use stats::{SessionStats, StatsTracker};
pub use tools::build_default_tools;

use std::sync::{Arc, Mutex, RwLock, Weak, atomic::AtomicBool};

use step_harness_agent_core::agent::{Agent, AgentListener, PromptInput};
use step_harness_agent_core::compaction::{
    CompactionSettings, SimpleCompleter, calculate_context_tokens, compact, prepare_compaction,
    should_compact,
};
use step_harness_agent_core::session::build_session_context;
use step_harness_agent_core::session::jsonl::{
    JsonlSessionCreateOptions, JsonlSessionMetadata, JsonlSessionRepo, JsonlSessionRepoOptions,
};
use step_harness_agent_core::session::{
    BranchEntryQuery, Entry, EntryOrder, IdGenerator, ProvisionedEntry, Session,
    SessionContextBuildOptions, SessionError, SessionModel, UuidV7Generator,
};
use step_harness_agent_core::types::{
    AgentEvent, AgentMessage, AgentOptions, AgentState, AgentTool, AgentToolResult, BeforeToolCall,
    BeforeToolCallContext, BeforeToolCallResult, InitialAgentState, Json, StreamFn, ToolError,
    ToolExecutionMode, ToolExecutor, ToolUpdateCallback, text_content,
};
use step_harness_providers::auth::CredentialStore;
use step_harness_providers::models::{Models, Provider};
use step_harness_providers::step_provider::{STEP_PROVIDER_ID, StepProviderOptions, step_provider};
use step_harness_providers::types::{
    AssistantMessage, Context, ImageContent, Model, ModelThinkingLevel, SimpleStreamOptions,
    StopReason, TextContent, UserContent, UserContentBlock, UserMessage,
};
use step_harness_providers::utils::overflow::{is_context_overflow, is_recoverable_length};
use step_harness_providers::utils::retry::{RetryPolicy, is_retryable_assistant_error};
use tokio_util::sync::CancellationToken;

use crate::models::{ModelCatalog, ModelController};
use crate::paths::AppPaths;
use crate::permissions::{ApprovalMode, Decision, PermissionPolicy, ToolOverrides};
use crate::plan::{
    ENTER_PLAN_MODE_TOOL, EXIT_PLAN_MODE_TOOL, PLAN_ENTRY_TYPE, PlanMode, filter_tools_by_name,
    is_mutating_tool, plan_mode_tool_names, targets_plan_file,
};
use crate::resources::ResourceSet;
use crate::settings::{Settings, SettingsManager};
use crate::system_prompt::{SystemPromptOptions, build_system_prompt};

/// Errors surfaced by session operations.
#[derive(Debug, thiserror::Error)]
pub enum SessionFailure {
    #[error("no model selected: add a provider in models.json or log in")]
    NoModelSelected,
    #[error("provider \"{0}\" has no credentials configured")]
    ProviderNotConfigured(String),
    #[error("{0}")]
    Agent(String),
    #[error("session storage failed: {0}")]
    Storage(String),
    #[error("{0}")]
    Other(String),
}

impl From<SessionError> for SessionFailure {
    fn from(error: SessionError) -> Self {
        SessionFailure::Storage(error.to_string())
    }
}

/// Result alias for session operations.
pub type SessionResult<T> = Result<T, SessionFailure>;

/// Everything bound to one working directory.
pub struct SessionServices {
    pub request_observer: Option<Arc<dyn step_harness_providers::request_metrics::RequestObserver>>,
    pub cwd: String,
    pub paths: AppPaths,
    pub settings: SettingsManager,
    pub catalog: Arc<ModelCatalog>,
    pub credentials: Arc<dyn CredentialStore>,
    pub resources: RwLock<ResourceSet>,
    pub project_trusted: bool,
    /// Optional model-completion override for session names (e.g. embedded hosts).
    /// None uses the session's selected model through the catalog.
    pub name_completer: Option<Arc<dyn SimpleCompleter>>,
}

impl SessionServices {
    /// Load settings, build the model catalog, and load resources.
    pub async fn create(
        cwd: impl Into<String>,
        paths: AppPaths,
        env: &Arc<dyn step_harness_agent_core::harness::types::ExecutionEnv>,
        project_trusted: bool,
    ) -> (Self, Vec<String>) {
        let cwd = cwd.into();
        let _ = paths.ensure_dirs();
        let settings =
            SettingsManager::create(std::path::Path::new(&cwd), paths.clone(), project_trusted);
        let (credentials, models_store) = crate::models::catalog::default_stores(&paths);
        let (catalog, diagnostics) =
            ModelCatalog::build(&paths, credentials.clone(), models_store).await;
        let resources = ResourceSet::load(env, &cwd, &paths, &settings.get()).await;
        let services = SessionServices {
            request_observer: None,
            cwd,
            paths,
            settings,
            catalog: Arc::new(catalog),
            credentials,
            resources: RwLock::new(resources),
            project_trusted,
            name_completer: None,
        };
        (services, diagnostics)
    }

    /// Reload settings and resources from disk.
    pub async fn reload(
        &self,
        env: &Arc<dyn step_harness_agent_core::harness::types::ExecutionEnv>,
    ) {
        self.settings.reload();
        let resources = ResourceSet::load(env, &self.cwd, &self.paths, &self.settings.get()).await;
        *self
            .resources
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = resources;
    }

    /// The settings snapshot.
    pub fn settings(&self) -> Settings {
        self.settings.get()
    }
}

/// Mutable per-run bookkeeping.
#[derive(Default)]
struct RunState {
    retry_attempt: u32,
    overflow_recovery_attempted: bool,
}

/// Shared state behind an [`AgentSession`].
struct SessionShared {
    request_execution: Arc<RwLock<Option<String>>>,
    request_run_lock: tokio::sync::Mutex<()>,
    services: Arc<SessionServices>,
    agent: Agent,
    session: Session<JsonlSessionMetadata>,
    metadata: JsonlSessionMetadata,
    model: ModelController,
    permissions: RwLock<PermissionPolicy>,
    approval: SharedApprovalChannel,
    /// Plan-mode state and the embedder's plan review UI.
    plan: PlanMode,
    plan_review: SharedPlanReviewChannel,
    /// The unfiltered tool set plan mode swaps against.
    all_tools: Vec<AgentTool>,
    stats: StatsTracker,
    events: tokio::sync::broadcast::Sender<SessionEvent>,
    run_state: Mutex<RunState>,
    models_handle: Arc<Models>,
    /// Serializes automatic and manual renames; manual changes invalidate pending names.
    name_revision: tokio::sync::Mutex<u64>,
    name_generation_active: AtomicBool,
}

/// Streams assistant events through the model catalog.
struct CatalogStreamFn {
    models: Arc<Models>,
    observer: Option<Arc<dyn step_harness_providers::request_metrics::RequestObserver>>,
    session_id: String,
    execution: Arc<RwLock<Option<String>>>,
}

#[async_trait::async_trait]
impl StreamFn for CatalogStreamFn {
    async fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
    ) -> step_harness_providers::event::AssistantMessageEventStream {
        let mut options = options.clone();
        if let Some(observer) = &self.observer {
            options.base.base.request_metrics = Some(step_harness_providers::request_metrics::RequestTracker::new(observer.clone(), step_harness_providers::request_metrics::RequestContext {
                session_id: self.session_id.clone(), execution_id: self.execution.read().unwrap_or_else(|p| p.into_inner()).clone(), purpose: "conversation".into(),
            }, model));
        }
        self.models.stream_simple(model, context, Some(&options))
    }
}

/// Runs one-shot assistant requests for compaction summaries.
struct CatalogCompleter {
    models: Arc<Models>,
    observer: Option<Arc<dyn step_harness_providers::request_metrics::RequestObserver>>,
    request_context: step_harness_providers::request_metrics::RequestContext,
}

#[async_trait::async_trait]
impl SimpleCompleter for CatalogCompleter {
    async fn complete_simple(
        &self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
    ) -> AssistantMessage {
        let mut options = options.clone();
        if let Some(observer) = &self.observer { options.base.base.request_metrics = Some(step_harness_providers::request_metrics::RequestTracker::new(observer.clone(), self.request_context.clone(), model)); }
        let _timing = step_harness_providers::execution_timing::enter(step_harness_providers::execution_timing::Phase::Api);
        self.models
            .complete_simple(model, context, Some(&options))
            .await
    }
}

/// Persists finished messages and forwards agent events to the embedder.
///
/// Holds a weak reference: the agent owns its listeners, so a strong reference
/// would keep the session (and its storage lease) alive forever.
struct SessionListener {
    shared: Weak<SessionShared>,
}

#[async_trait::async_trait]
impl AgentListener for SessionListener {
    async fn on_event(&self, event: &AgentEvent, _signal: &CancellationToken) {
        let Some(shared) = self.shared.upgrade() else {
            return;
        };
        if let AgentEvent::MessageEnd { message } = event {
            match shared.session.append_message((**message).clone()).await {
                Ok(entry_id) => {
                    let _ = shared.events.send(SessionEvent::EntryAppended { entry_id });
                    if message.as_user().is_some()
                        && let Some(model) = shared.model.current_model(&shared.services.catalog)
                        && let Some(request) = naming::prepare_name_request(&shared, message).await
                    {
                        naming::generate_name(&shared, model, request);
                    }
                }
                Err(error) => {
                    let _ = shared.events.send(SessionEvent::Error {
                        message: format!("failed to persist message: {error}"),
                    });
                }
            }
            if let Some(assistant) = message.as_assistant() {
                if let Some(model) = shared
                    .services
                    .catalog
                    .find(assistant.provider.as_str(), &assistant.model)
                {
                    shared.stats.record(&model, &assistant.usage);
                }
                if assistant.stop_reason != StopReason::Error {
                    let mut state = shared
                        .run_state
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    state.retry_attempt = 0;
                    if assistant.stop_reason != StopReason::Length {
                        state.overflow_recovery_attempted = false;
                    }
                }
            }
        }
        let _ = shared.events.send(SessionEvent::Agent {
            event: event.clone(),
        });
    }
}

/// Gates tool calls through the approval policy.
///
/// The session is installed after construction (the hook is a field on
/// [`Agent`], which is built before the shared state exists), so the hook
/// holds a weak reference upgraded on each call.
struct ApprovalHook {
    shared: Mutex<Option<Weak<SessionShared>>>,
}

impl ApprovalHook {
    fn new() -> Arc<Self> {
        Arc::new(ApprovalHook {
            shared: Mutex::new(None),
        })
    }

    fn install(&self, shared: &Arc<SessionShared>) {
        *self
            .shared
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(Arc::downgrade(shared));
    }
}

#[async_trait::async_trait]
impl BeforeToolCall for ApprovalHook {
    async fn before_tool_call(
        &self,
        context: &BeforeToolCallContext,
        signal: Option<&step_harness_agent_core::harness::types::AbortSignal>,
    ) -> Option<BeforeToolCallResult> {
        let shared = self
            .shared
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .and_then(Weak::upgrade)?;
        // Plan mode limits file mutations to the session's plan file; shell
        // tools keep their normal permissions.
        if shared.plan.is_active() && is_mutating_tool(&context.tool_call.name) {
            let plan_path = shared
                .plan
                .plan_file_path(&shared.metadata.id, &shared.services.cwd);
            let target = context.args.get("path").and_then(|value| value.as_str());
            if !target
                .is_some_and(|target| targets_plan_file(target, &plan_path, &shared.services.cwd))
            {
                return Some(BeforeToolCallResult {
                    block: true,
                    reason: Some(format!(
                        "Plan mode blocked this file mutation. Only the plan file may be written: {}\nTarget: {}",
                        plan_path.display(),
                        target.unwrap_or("(no path provided)")
                    )),
                    terminate: Some(false),
                });
            }
        }
        let policy = shared
            .permissions
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        let decision = policy.decide(
            &context.tool_call.name,
            &serde_json::Value::Object(context.args.clone()),
        );
        match decision {
            Decision::Allow => None,
            Decision::Deny { reason, terminate } => Some(BeforeToolCallResult {
                block: true,
                reason: Some(reason),
                terminate: Some(terminate),
            }),
            Decision::Confirm { reason, hazardous } => {
                let request = events::ApprovalRequest {
                    request_id: format!("{}:{}", shared.metadata.id, context.tool_call.id),
                    session_id: shared.metadata.id.clone(),
                    created_at: step_harness_providers::utils::now_ms(),
                    tool_name: context.tool_call.name.clone(),
                    args_summary: summarize_args(&context.args),
                    reason,
                    hazardous,
                };
                let approved = {
                    let _timing = step_harness_providers::execution_timing::enter(step_harness_providers::execution_timing::Phase::Approval);
                    match signal {
                        Some(signal) => tokio::select! {
                            biased;
                            _ = signal.cancelled() => false,
                            approved = shared.approval.confirm(&request) => approved,
                        },
                        None => shared.approval.confirm(&request).await,
                    }
                };
                if approved {
                    None
                } else {
                    Some(BeforeToolCallResult {
                        block: true,
                        reason: Some(format!("Tool call denied: {}", context.tool_call.name)),
                        terminate: Some(false),
                    })
                }
            }
        }
    }
}

/// Result text for a dismissed review: stay in plan mode and stop the batch.
const STAY_IN_PLAN_MODE_RESULT: &str = "The user dismissed the review without approving the plan or leaving notes. Staying in plan mode. \
     Stop here and wait for the user; do not call exit_plan_mode again until they ask for another review.";

/// Weak cell the plan tools upgrade on each call, mirroring [`ApprovalHook`].
///
/// Clones share one cell, so installing the session once arms both tools.
#[derive(Clone, Default)]
struct PlanToolCell(Arc<Mutex<Option<Weak<SessionShared>>>>);

impl PlanToolCell {
    fn install(&self, shared: &Arc<SessionShared>) {
        *self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(Arc::downgrade(shared));
    }

    fn upgrade(&self) -> Option<Arc<SessionShared>> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .and_then(Weak::upgrade)
    }
}

/// Executor for `enter_plan_mode`.
struct EnterPlanModeExecutor {
    cell: PlanToolCell,
}

/// Executor for `exit_plan_mode`.
struct ExitPlanModeExecutor {
    cell: PlanToolCell,
}

/// The `enter_plan_mode` tool: research first, write the proposal to the
/// session plan file, then submit it with `exit_plan_mode`.
fn create_enter_plan_mode_tool(cell: PlanToolCell) -> AgentTool {
    AgentTool {
        name: ENTER_PLAN_MODE_TOOL.to_owned(),
        label: "enter plan mode".to_owned(),
        description: "Enter plan mode to research and propose an implementation approach before coding, not to track todos. Returns the proposal file path; use write or edit to record the approach, trade-offs, steps, and validation for review. File-editing tools are limited to that file; commands still use normal permissions.".to_owned(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {}
        }),
        execution_mode: None,
        prepare_arguments: None,
        executor: Arc::new(EnterPlanModeExecutor { cell }),
    }
}

/// The `exit_plan_mode` tool: submit the written proposal for review.
fn create_exit_plan_mode_tool(cell: PlanToolCell) -> AgentTool {
    AgentTool {
        name: EXIT_PLAN_MODE_TOOL.to_owned(),
        label: "exit plan mode".to_owned(),
        description: "Submit the written proposal for review. Requires a readable, nonempty regular plan file. User approval exits plan mode; staying, requesting refinements, or cancelling keeps it active.".to_owned(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {}
        }),
        execution_mode: None,
        prepare_arguments: None,
        executor: Arc::new(ExitPlanModeExecutor { cell }),
    }
}

/// Transcript details for a plan review, mirroring the source's
/// `PlanReviewDetails`.
fn plan_review_details(request: &PlanReviewRequest, outcome: &str) -> Json {
    serde_json::json!({
        "planFilePath": request.plan_file_path,
        "planContents": request.plan_contents,
        "outcome": outcome,
    })
}

#[async_trait::async_trait]
impl ToolExecutor<Json> for EnterPlanModeExecutor {
    async fn execute(
        &self,
        _tool_call_id: &str,
        _params: &serde_json::Map<String, Json>,
        _signal: Option<&step_harness_agent_core::harness::types::AbortSignal>,
        _on_update: Option<&ToolUpdateCallback<Json>>,
    ) -> Result<AgentToolResult<Json>, ToolError> {
        let Some(shared) = self.cell.upgrade() else {
            return Ok(AgentToolResult::error("Plan mode is unavailable."));
        };
        if shared.plan.is_active() {
            let path = shared
                .plan
                .plan_file_path(&shared.metadata.id, &shared.services.cwd);
            return Ok(AgentToolResult::text(
                format!(
                    "Already in plan mode. Write your proposal to {} using write or edit, then call exit_plan_mode when it is ready.",
                    path.display()
                ),
                Json::Null,
            ));
        }
        let snapshot = enter_plan_mode(&shared, PlanSource::Agent).await;
        let plan_file_path = snapshot.plan_file_path.unwrap_or_default();
        Ok(AgentToolResult::text(
            format!(
                "Plan mode active. Explore the repository, then write your proposal to {plan_file_path} using write. Call exit_plan_mode when it is ready for review."
            ),
            Json::Null,
        ))
    }
}

#[async_trait::async_trait]
impl ToolExecutor<Json> for ExitPlanModeExecutor {
    async fn execute(
        &self,
        tool_call_id: &str,
        _params: &serde_json::Map<String, Json>,
        signal: Option<&step_harness_agent_core::harness::types::AbortSignal>,
        _on_update: Option<&ToolUpdateCallback<Json>>,
    ) -> Result<AgentToolResult<Json>, ToolError> {
        let Some(shared) = self.cell.upgrade() else {
            return Ok(AgentToolResult::error("Plan mode is unavailable."));
        };
        if !shared.plan.is_active() {
            return Ok(AgentToolResult::text(
                "Not in plan mode.".to_owned(),
                Json::Null,
            ));
        }
        let plan_path = shared
            .plan
            .plan_file_path(&shared.metadata.id, &shared.services.cwd);
        let plan_file_path = plan_path.to_string_lossy().to_string();
        // The plan is shown in full to the reviewer: a review that hides the
        // end of the proposal asks the user to approve something they have
        // not read.
        let plan_contents = match default_env().read_text_file(&plan_file_path, signal).await {
            Ok(contents) if !contents.trim().is_empty() => contents,
            Ok(_) => {
                return Ok(AgentToolResult::text(
                    format!(
                        "The plan file at {plan_file_path} is empty or whitespace-only. Write the proposal with write, then call exit_plan_mode again."
                    ),
                    Json::Null,
                ));
            }
            Err(error) => {
                return Ok(AgentToolResult::text(
                    format!(
                        "Cannot review the plan at {plan_file_path}: {error}. Provide a readable, nonempty file containing your proposal with write, then call exit_plan_mode again."
                    ),
                    Json::Null,
                ));
            }
        };
        let request = PlanReviewRequest {
            request_id: format!("{}:{}", shared.metadata.id, tool_call_id),
            session_id: shared.metadata.id.clone(),
            created_at: step_harness_providers::utils::now_ms(),
            plan_file_path: plan_file_path.clone(),
            plan_contents,
        };
        let outcome = {
            let _timing = step_harness_providers::execution_timing::enter(step_harness_providers::execution_timing::Phase::Approval);
            match signal {
                Some(signal) => tokio::select! {
                    biased;
                    _ = signal.cancelled() => PlanReviewOutcome::Dismissed,
                    outcome = shared.plan_review.review(&request) => outcome,
                },
                None => shared.plan_review.review(&request).await,
            }
        };
        match outcome {
            PlanReviewOutcome::Approved => {
                exit_plan_mode(&shared).await;
                Ok(AgentToolResult::text(
                    format!(
                        "The user approved the plan. Plan mode exited and file mutations are re-enabled. Execute the plan now, using {plan_file_path} as the reference; do not silently diverge from it."
                    ),
                    plan_review_details(&request, "approved"),
                ))
            }
            PlanReviewOutcome::Feedback(notes) => {
                // Steer, don't follow up: the notes must land before the very
                // next response, not after the agent would have stopped.
                shared.agent.steer(AgentMessage::user(
                    format!(
                        "Plan refinement requested. Update {plan_file_path} based on:\n\n{notes}"
                    ),
                    step_harness_providers::utils::now_ms(),
                ));
                Ok(AgentToolResult::text(
                    "The user requested refinements. Their notes follow immediately as the next user message; do not ask what to change. Staying in plan mode: apply the notes to the plan file and call exit_plan_mode again.".to_owned(),
                    plan_review_details(&request, "feedback"),
                ))
            }
            PlanReviewOutcome::Dismissed => Ok(AgentToolResult {
                content: vec![text_content(STAY_IN_PLAN_MODE_RESULT)],
                details: plan_review_details(&request, "dismissed"),
                usage: None,
                added_tool_names: None,
                terminate: Some(true),
            }),
        }
    }
}

/// Enter plan mode: swap the tool set, persist the state, and notify the UI.
async fn enter_plan_mode(shared: &SessionShared, source: PlanSource) -> PlanState {
    let active: Vec<String> = shared
        .agent
        .state()
        .tools
        .iter()
        .map(|tool| tool.name.clone())
        .collect();
    let snapshot = shared
        .plan
        .enter(source, &shared.metadata.id, &shared.services.cwd, &active);
    apply_plan_tools(shared, &snapshot);
    persist_plan_state(shared, &snapshot).await;
    emit_plan_mode_changed(shared, &snapshot);
    snapshot
}

/// Leave plan mode: restore the baseline tool set, persist, and notify.
async fn exit_plan_mode(shared: &SessionShared) -> PlanState {
    let snapshot = shared.plan.exit();
    apply_plan_tools(shared, &snapshot);
    persist_plan_state(shared, &snapshot).await;
    emit_plan_mode_changed(shared, &snapshot);
    snapshot
}

/// Swap the agent's tool set between the plan-mode and baseline sets.
fn apply_plan_tools(shared: &SessionShared, snapshot: &PlanState) {
    let baseline = snapshot.tools_before_plan_mode.clone().unwrap_or_else(|| {
        shared
            .all_tools
            .iter()
            .map(|tool| tool.name.clone())
            .collect()
    });
    let allowed = if snapshot.enabled {
        plan_mode_tool_names(&baseline)
    } else {
        baseline
    };
    shared.agent.set_tools(filter_tools_by_name(
        shared.all_tools.clone(),
        &allowed,
        |tool| tool.name.as_str(),
    ));
    rebuild_system_prompt(shared);
}

/// Persist the plan-mode snapshot as a custom session entry.
async fn persist_plan_state(shared: &SessionShared, snapshot: &PlanState) {
    let entry = ProvisionedEntry::Custom {
        id: next_entry_id(),
        custom_type: PLAN_ENTRY_TYPE.to_owned(),
        data: Some(serde_json::to_value(snapshot).unwrap_or(Json::Null)),
    };
    if let Err(error) = shared.session.append_entry(entry, "main").await {
        let _ = shared.events.send(SessionEvent::Error {
            message: format!("failed to persist plan mode state: {error}"),
        });
    }
}

/// Announce a plan-mode transition to the embedder.
fn emit_plan_mode_changed(shared: &SessionShared, snapshot: &PlanState) {
    let _ = shared.events.send(SessionEvent::PlanModeChanged {
        active: snapshot.enabled,
        plan_file_path: snapshot.plan_file_path.clone(),
        plan_source: snapshot
            .plan_source
            .map(|source| source.as_str().to_owned()),
    });
}

/// Recompute the system prompt from the current tools and resources.
fn rebuild_system_prompt(shared: &SessionShared) {
    let state = shared.agent.state();
    let resources = shared
        .services
        .resources
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone();
    let prompt = build_system_prompt(SystemPromptOptions {
        cwd: &shared.services.cwd,
        paths: &shared.services.paths,
        tools: &state.tools,
        resources: &resources,
        append_system_prompt: None,
    });
    shared.agent.set_system_prompt(prompt);
}

/// Compact `key=value` rendering of tool arguments for confirmation prompts.
fn summarize_args(args: &serde_json::Map<String, serde_json::Value>) -> String {
    let mut pairs: Vec<String> = args
        .iter()
        .map(|(key, value)| {
            let text = match value {
                serde_json::Value::String(text) => text.clone(),
                other => other.to_string(),
            };
            format!("{key}={text}")
        })
        .collect();
    pairs.sort();
    let collapsed = pairs
        .join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if collapsed.chars().count() > 240 {
        let clipped: String = collapsed.chars().take(237).collect();
        format!("{clipped}...")
    } else {
        collapsed
    }
}

/// Branch query for the chronological transcript path.
fn chronological_branch_query() -> BranchEntryQuery {
    BranchEntryQuery {
        order: Some(EntryOrder::OldestFirst),
        ..Default::default()
    }
}

/// The model the session last recorded, from its newest model-change entry.
fn last_recorded_model(entries: &[Entry]) -> Option<SessionModel> {
    entries.iter().rev().find_map(|entry| match entry {
        Entry::ModelChange(change) => Some(SessionModel {
            provider: change.provider.clone(),
            model_id: change.model_id.clone(),
        }),
        _ => None,
    })
}

/// The thinking level the session last recorded, when it persisted one.
fn last_recorded_thinking_level(entries: &[Entry]) -> Option<ModelThinkingLevel> {
    entries.iter().rev().find_map(|entry| match entry {
        Entry::ThinkingLevelChange(change) => parse_thinking_level(&change.thinking_level),
        _ => None,
    })
}

/// Custom-entry type carrying the session's approval mode.
const APPROVAL_MODE_ENTRY: &str = "approval_mode";

/// The approval mode the session last recorded, when it persisted one.
///
/// Sessions recorded before the mode was persisted have no such entry and
/// fall back to the settings preset.
fn last_recorded_approval_mode(entries: &[Entry]) -> Option<ApprovalMode> {
    entries.iter().rev().find_map(|entry| match entry {
        Entry::Custom(entry) if entry.custom_type == APPROVAL_MODE_ENTRY => entry
            .data
            .as_ref()
            .and_then(|data| data.get("mode"))
            .and_then(|mode| mode.as_str())
            .and_then(ApprovalMode::from_label),
        _ => None,
    })
}

fn parse_thinking_level(label: &str) -> Option<ModelThinkingLevel> {
    ModelThinkingLevel::ORDERED
        .into_iter()
        .find(|level| level.as_str() == label)
}

/// Id for a configuration entry; UUIDv7 keeps ids unique within the log.
fn next_entry_id() -> String {
    UuidV7Generator.next()
}

/// One interactive agent session.
pub struct AgentSession {
    shared: Arc<SessionShared>,
}

impl AgentSession {
    /// Open a brand-new session for a working directory.
    pub async fn create(
        services: Arc<SessionServices>,
        approval: SharedApprovalChannel,
        plan_review: SharedPlanReviewChannel,
    ) -> SessionResult<Self> {
        let repo = JsonlSessionRepo::new(
            default_env(),
            JsonlSessionRepoOptions {
                sessions_root: services.paths.sessions_dir().to_string_lossy().to_string(),
            },
        );
        let create_options = JsonlSessionCreateOptions {
            id: None,
            parent_session_id: None,
            cwd: services.cwd.clone(),
            metadata: None,
        };
        let session = repo.create(&create_options).await?;
        let metadata = session.get_metadata().await;
        Self::assemble(services, session, metadata, approval, plan_review).await
    }

    /// Open an existing session file.
    pub async fn open(
        services: Arc<SessionServices>,
        metadata: JsonlSessionMetadata,
        approval: SharedApprovalChannel,
        plan_review: SharedPlanReviewChannel,
    ) -> SessionResult<Self> {
        let repo = JsonlSessionRepo::new(
            default_env(),
            JsonlSessionRepoOptions {
                sessions_root: services.paths.sessions_dir().to_string_lossy().to_string(),
            },
        );
        let session = repo.open(&metadata).await?;
        Self::assemble(services, session, metadata, approval, plan_review).await
    }

    /// Wire the agent, tools, listener, and approval hook together.
    async fn assemble(
        services: Arc<SessionServices>,
        session: Session<JsonlSessionMetadata>,
        metadata: JsonlSessionMetadata,
        approval: SharedApprovalChannel,
        plan_review: SharedPlanReviewChannel,
    ) -> SessionResult<Self> {
        let settings = services.settings.get();
        let plan_cell = PlanToolCell::default();
        let mut tools = tools::build_default_tools_with_search(
            default_env(),
            &settings,
            crate::search_web::SearchWebToolOptions {
                auth_path: Some(services.paths.auth_file()),
                ..Default::default()
            },
        );
        tools.push(create_enter_plan_mode_tool(plan_cell.clone()));
        tools.push(create_exit_plan_mode_tool(plan_cell.clone()));
        let all_tools = tools.clone();
        let (events, _rx) = tokio::sync::broadcast::channel(1024);
        let models_handle = services.catalog.models_handle();

        // Restore the transcript from the durable store.
        let entries = session
            .find_entries_on_branch(&chronological_branch_query())
            .await?;
        let restored = build_session_context(&entries, &SessionContextBuildOptions::default());
        // Restore an in-progress plan before the first turn: the agent starts
        // with the tool set the plan-mode restriction produces.
        let plan = PlanMode::new();
        let restored_plan = plan.restore(&entries, &metadata.id, &services.cwd);
        let tools = if restored_plan.enabled {
            let baseline = restored_plan
                .tools_before_plan_mode
                .clone()
                .unwrap_or_else(|| all_tools.iter().map(|tool| tool.name.clone()).collect());
            filter_tools_by_name(tools, &plan_mode_tool_names(&baseline), |tool| {
                tool.name.as_str()
            })
        } else {
            tools
        };
        // A model or thinking level this session recorded wins over the
        // settings default; the last assistant message stands in for sessions
        // recorded before configuration changes were persisted.
        let stored_model = last_recorded_model(&entries)
            .or_else(|| restored.model.clone())
            .and_then(|stored| services.catalog.find(&stored.provider, &stored.model_id));
        let model = match &stored_model {
            Some(found) => ModelController::from_model(found, &settings),
            None => match ModelController::initial(&services.catalog, &settings) {
                Ok(controller) => controller,
                Err(error) => {
                    let _ = events.send(SessionEvent::Error {
                        message: error.to_string(),
                    });
                    ModelController::unset()
                }
            },
        };
        if let Some(level) = last_recorded_thinking_level(&entries) {
            model.set_thinking_level(level, &services.catalog);
        }
        // The approval mode follows the session the same way: the mode this
        // session recorded wins over the settings preset.
        let approval_mode = last_recorded_approval_mode(&entries)
            .or_else(|| {
                settings
                    .permissions
                    .as_ref()
                    .and_then(|permissions| permissions.preset)
                    .map(ApprovalMode::from)
            })
            .unwrap_or(ApprovalMode::Ask);

        let resources = services
            .resources
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        let system_prompt = build_system_prompt(SystemPromptOptions {
            cwd: &services.cwd,
            paths: &services.paths,
            tools: &tools,
            resources: &resources,
            append_system_prompt: None,
        });

        let initial_state = InitialAgentState {
            system_prompt: Some(system_prompt),
            model: model.current_model(&services.catalog),
            thinking_level: Some(model.thinking_level()),
            tools: Some(tools),
            messages: Some(restored.messages),
        };

        let approval_hook = ApprovalHook::new();
        let request_execution = Arc::new(RwLock::new(None));
        let agent = Agent::new(AgentOptions {
            initial_state: Some(initial_state),
            convert_to_llm: None,
            transform_context: None,
            stream_fn: Arc::new(CatalogStreamFn {
                models: models_handle.clone(),
                observer: services.request_observer.clone(),
                session_id: metadata.id.clone(),
                execution: request_execution.clone(),
            }),
            get_api_key: None,
            on_payload: None,
            on_response: None,
            before_tool_call: Some(approval_hook.clone()),
            after_tool_call: None,
            transform_tool_result: None,
            should_stop_after_turn: None,
            prepare_next_turn: None,
            prepare_next_turn_with_context: None,
            steering_mode: settings.steering_mode.unwrap_or_default().into(),
            follow_up_mode: settings.follow_up_mode.unwrap_or_default().into(),
            session_id: Some(metadata.id.clone()),
            thinking_budgets: None,
            transport: step_harness_providers::types::Transport::Auto,
            max_retry_delay_ms: None,
            tool_execution: ToolExecutionMode::Parallel,
        });

        let shared = Arc::new(SessionShared {
            request_execution,
            request_run_lock: tokio::sync::Mutex::new(()),
            services,
            agent,
            session,
            metadata,
            model,
            permissions: RwLock::new(PermissionPolicy::new(
                approval_mode,
                ToolOverrides::default(),
                settings
                    .permissions
                    .as_ref()
                    .and_then(|permissions| permissions.non_interactive_approval)
                    .unwrap_or(false),
            )),
            approval,
            plan,
            plan_review,
            all_tools,
            stats: StatsTracker::new(),
            events,
            run_state: Mutex::new(RunState::default()),
            models_handle,
            name_revision: tokio::sync::Mutex::new(0),
            name_generation_active: AtomicBool::new(false),
        });

        approval_hook.install(&shared);
        plan_cell.install(&shared);
        let listener: Arc<dyn AgentListener> = Arc::new(SessionListener {
            shared: Arc::downgrade(&shared),
        });
        shared.agent.subscribe(listener);

        Ok(AgentSession { shared })
    }

    /// Subscribe to session events.
    pub fn subscribe(&self) -> SessionEventRx {
        self.shared.events.subscribe()
    }

    /// The session's durable metadata.
    pub fn metadata(&self) -> &JsonlSessionMetadata {
        &self.shared.metadata
    }

    /// The current agent state snapshot.
    pub fn state(&self) -> AgentState {
        self.shared.agent.state()
    }

    /// Session usage statistics.
    pub fn stats(&self) -> SessionStats {
        self.shared.stats.snapshot()
    }

    /// The active model, when one is selected.
    pub fn current_model(&self) -> Option<Model> {
        self.shared
            .model
            .current_model(&self.shared.services.catalog)
    }

    /// The active provider id.
    pub fn provider_id(&self) -> String {
        self.shared.model.provider_id()
    }

    /// The active model id.
    pub fn model_id(&self) -> String {
        self.shared.model.model_id()
    }

    /// The active reasoning level.
    pub fn thinking_level(&self) -> ModelThinkingLevel {
        self.shared.model.thinking_level()
    }

    /// The effective approval mode.
    pub fn approval_mode(&self) -> ApprovalMode {
        self.shared
            .permissions
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .mode()
    }

    /// The current plan-mode snapshot.
    pub fn plan_state(&self) -> PlanState {
        self.shared.plan.snapshot()
    }

    /// Whether plan mode is active.
    pub fn is_plan_mode_active(&self) -> bool {
        self.shared.plan.is_active()
    }

    /// The session's plan file path.
    pub fn plan_file_path(&self) -> std::path::PathBuf {
        self.shared
            .plan
            .plan_file_path(&self.shared.metadata.id, &self.shared.services.cwd)
    }

    /// Enter plan mode on the user's behalf (the `/plan` command).
    pub async fn enter_plan_mode(&self, source: PlanSource) -> PlanState {
        enter_plan_mode(&self.shared, source).await
    }

    /// Leave plan mode on the user's behalf (the `/plan` command).
    pub async fn exit_plan_mode(&self) -> PlanState {
        exit_plan_mode(&self.shared).await
    }

    /// Toggle plan mode, returning the resulting snapshot.
    pub async fn toggle_plan_mode(&self) -> PlanState {
        if self.shared.plan.is_active() {
            self.exit_plan_mode().await
        } else {
            self.enter_plan_mode(PlanSource::User).await
        }
    }

    /// Models the session may select, scoped by settings globs.
    pub fn available_models(&self) -> Vec<Model> {
        let patterns = self
            .shared
            .services
            .settings
            .get()
            .enabled_models
            .unwrap_or_default();
        self.shared.services.catalog.scoped_models(&patterns)
    }

    /// Switch the active model.
    pub async fn set_model(&self, provider: &str, model_id: &str) -> SessionResult<()> {
        self.shared
            .model
            .set_model(provider, model_id, &self.shared.services.catalog)
            .map_err(|error| SessionFailure::Other(error.to_string()))?;
        self.record_configuration_change(ProvisionedEntry::ModelChange {
            id: next_entry_id(),
            provider: provider.to_owned(),
            model_id: model_id.to_owned(),
        })
        .await;
        self.apply_runtime_state();
        let _ = self.shared.events.send(SessionEvent::ModelChanged {
            provider: provider.to_owned(),
            model_id: model_id.to_owned(),
            thinking_level: format!("{:?}", self.thinking_level()).to_lowercase(),
        });
        Ok(())
    }

    /// Set the reasoning level (clamped to the model's supported set).
    pub async fn set_thinking_level(&self, level: ModelThinkingLevel) {
        self.shared
            .model
            .set_thinking_level(level, &self.shared.services.catalog);
        let thinking_level = self.thinking_level();
        self.record_configuration_change(ProvisionedEntry::ThinkingLevelChange {
            id: next_entry_id(),
            thinking_level: thinking_level.as_str().to_owned(),
        })
        .await;
        self.apply_runtime_state();
        let _ = self.shared.events.send(SessionEvent::ModelChanged {
            provider: self.provider_id(),
            model_id: self.model_id(),
            thinking_level: format!("{:?}", thinking_level).to_lowercase(),
        });
    }

    /// Persist a configuration change so reopening the session restores it.
    ///
    /// The in-memory switch already happened, so a failed write only costs the
    /// restore on the next open and must not fail the caller's request.
    async fn record_configuration_change(&self, entry: ProvisionedEntry) {
        if let Err(error) = self.shared.session.append_entry(entry, "main").await {
            let _ = self.shared.events.send(SessionEvent::Error {
                message: format!("failed to persist session configuration: {error}"),
            });
        }
    }

    /// Push the controller's model/level onto the agent and refresh the prompt.
    fn apply_runtime_state(&self) {
        if let Some(model) = self.current_model() {
            self.shared.agent.set_model(model);
        }
        self.shared.agent.set_thinking_level(self.thinking_level());
        self.rebuild_system_prompt();
    }

    /// Recompute the system prompt from current tools and resources.
    fn rebuild_system_prompt(&self) {
        rebuild_system_prompt(&self.shared);
    }

    /// Submit a user prompt and run to settlement.
    pub async fn prompt(
        &self,
        text: impl Into<String>,
        images: Option<Vec<ImageContent>>,
    ) -> SessionResult<()> {
        let model = self
            .current_model()
            .ok_or(SessionFailure::NoModelSelected)?;
        if !self
            .shared
            .services
            .catalog
            .is_provider_configured(model.provider.as_str())
        {
            return Err(SessionFailure::ProviderNotConfigured(
                model.provider.to_string(),
            ));
        }

        let mut content: Vec<UserContentBlock> = vec![UserContentBlock::Text(TextContent {
            text: text.into(),
            text_signature: None,
        })];
        if let Some(images) = images {
            content.extend(images.into_iter().map(UserContentBlock::Image));
        }
        let message = AgentMessage::User(UserMessage {
            content: UserContent::Blocks(content),
            timestamp: step_harness_providers::utils::now_ms(),
        });
        // The agent loop emits message_start/message_end for prompt messages,
        // and the listener persists every message_end — so the user message is
        // stored exactly once, through the same path as every other message.
        self.run_turn(PromptInput::Message(message)).await
    }

    /// Queue a steering message (delivered after the current turn's tools).
    pub fn steer(&self, text: impl Into<String>) {
        self.steer_with_images(text, Vec::new());
    }

    /// Queue a steering message with image attachments.
    pub fn steer_with_images(&self, text: impl Into<String>, images: Vec<ImageContent>) {
        self.shared
            .agent
            .steer(Self::queued_user_message(text.into(), images));
        self.emit_queue_update();
    }

    /// Queue a follow-up message (delivered when the agent would stop).
    pub fn follow_up(&self, text: impl Into<String>) {
        self.follow_up_with_images(text, Vec::new());
    }

    /// Queue a follow-up message with image attachments.
    pub fn follow_up_with_images(&self, text: impl Into<String>, images: Vec<ImageContent>) {
        self.shared
            .agent
            .follow_up(Self::queued_user_message(text.into(), images));
        self.emit_queue_update();
    }

    fn queued_user_message(text: String, images: Vec<ImageContent>) -> AgentMessage {
        if images.is_empty() {
            return AgentMessage::user(text, step_harness_providers::utils::now_ms());
        }
        let mut content = Vec::new();
        if !text.trim().is_empty() {
            content.push(UserContentBlock::Text(TextContent {
                text,
                text_signature: None,
            }));
        }
        content.extend(images.into_iter().map(UserContentBlock::Image));
        AgentMessage::User(UserMessage {
            content: UserContent::Blocks(content),
            timestamp: step_harness_providers::utils::now_ms(),
        })
    }

    /// Clear both queues.
    pub fn clear_queue(&self) {
        self.shared.agent.clear_all_queues();
        self.emit_queue_update();
    }

    fn emit_queue_update(&self) {
        let _ = self.shared.events.send(SessionEvent::QueueUpdate {
            steering: Vec::new(),
            follow_up: Vec::new(),
        });
    }

    /// Abort the active run.
    pub fn abort(&self) {
        self.shared.agent.abort();
    }

    /// True while a run is active.
    pub fn is_running(&self) -> bool {
        self.shared.agent.signal().is_some()
    }
    pub fn request_execution_id(&self) -> Option<String> { self.shared.request_execution.read().unwrap_or_else(|p| p.into_inner()).clone() }

    /// Run one prompt to settlement: the agent loop, then the post-run loop
    /// over retries, auto-compaction, and queued messages.
    async fn run_turn(&self, input: PromptInput) -> SessionResult<()> {
        // A rejected concurrent prompt must never replace an in-flight execution id.
        let _request_run = self.shared.request_run_lock.try_lock().map_err(|_| SessionFailure::Agent("Agent is already running".into()))?;
        let execution_id = step_harness_providers::request_metrics::new_id();
        *self.shared.request_execution.write().unwrap_or_else(|p| p.into_inner()) = Some(execution_id.clone());
        let execution = step_harness_providers::request_metrics::ExecutionTracker::new(
            self.shared.services.request_observer.clone(), execution_id, self.metadata().id.clone(),
        );
        let clock = step_harness_providers::execution_timing::ExecutionClock::new();
        clock.scope(self.run_timed_turn(input, execution, clock.clone())).await
    }

    async fn run_timed_turn(
        &self, input: PromptInput,
        mut execution: step_harness_providers::request_metrics::ExecutionTracker,
        clock: Arc<step_harness_providers::execution_timing::ExecutionClock>,
    ) -> SessionResult<()> {
        let first_message_timestamp = match &input {
            PromptInput::Message(message) => message.timestamp(),
            _ => step_harness_providers::utils::now_ms(),
        };
        let (previous_message_count, previous_timestamp) = {
            let state = self.shared.agent.state();
            (state.messages.len(), state.messages.last().map(AgentMessage::timestamp))
        };
        let result: SessionResult<()> = async {
            self.shared
                .agent
                .prompt(input, None)
                .await
                .map_err(|error| { execution.finish("failed"); SessionFailure::Agent(error.to_string()) })?;

            loop {
                if self
                    .shared
                    .agent
                    .signal()
                    .map(|signal| signal.is_cancelled())
                    .unwrap_or(false)
                {
                    break;
                }
                let Some(last) = self.last_assistant_message() else {
                    break;
                };
                if self.prepare_retry(&last).await.map_err(|error| { execution.finish("failed"); error })? {
                    self.continue_run().await;
                    continue;
                }
                if self.check_compaction(&last).await.map_err(|error| { execution.finish("failed"); error })? {
                    self.continue_run().await;
                    continue;
                }
                if self.shared.agent.has_queued_messages() {
                    self.continue_run().await;
                    continue;
                }
                break;
            }

            Ok(())
        }.await;

        let status = if result.is_err() { "failed" } else if self.shared.agent.signal().is_some_and(|signal| signal.is_cancelled()) {
            "cancelled"
        } else { match self.last_assistant_message().map(|message| message.stop_reason) {
            Some(StopReason::Error | StopReason::Pending) | None => "failed",
            Some(StopReason::Aborted) => "cancelled",
            _ => "success",
        }};
        execution.finish(status);
        let (duration_ms, parts) = clock.finish();
        let state = self.shared.agent.state();
        if let Some(last) = state.messages.last().filter(|last| result.is_ok() || state.messages.len() != previous_message_count || Some(last.timestamp()) != previous_timestamp) {
            let timing = ExecutionTiming {
                first_message_timestamp,
                last_message_timestamp: last.timestamp(),
                duration_ms,
                other_duration_ms: Some(parts[0]),
                tool_duration_ms: Some(parts[1]),
                api_duration_ms: Some(parts[2]),
                retry_wait_duration_ms: Some(parts[3]),
                approval_wait_duration_ms: Some(parts[4]),
            };
            if let Err(error) = self
                .shared
                .session
                .append_custom_entry(
                    "execution_timing",
                    Some(serde_json::to_value(&timing).expect("execution timing serializes")),
                )
                .await
            {
                log::warn!("failed to persist execution timing: {error}");
            }
            let _ = self
                .shared
                .events
                .send(SessionEvent::ExecutionCompleted { timing });
        }
        let _ = self.shared.events.send(SessionEvent::Settled);
        result
    }

    async fn continue_run(&self) {
        if let Err(error) = self.shared.agent.continue_run().await {
            let _ = self.shared.events.send(SessionEvent::Error {
                message: error.to_string(),
            });
        }
    }

    fn last_assistant_message(&self) -> Option<AssistantMessage> {
        self.shared
            .agent
            .state()
            .messages
            .iter()
            .rev()
            .find_map(|message| message.as_assistant().cloned())
    }

    /// Retry a retryable provider error with exponential backoff.
    async fn prepare_retry(&self, message: &AssistantMessage) -> SessionResult<bool> {
        let settings = self.shared.services.settings.get();
        let retry = settings.retry.unwrap_or_default();
        if !retry.enabled.unwrap_or(true) {
            return Ok(false);
        }
        if message.stop_reason != StopReason::Error {
            return Ok(false);
        }
        let context_window = self
            .current_model()
            .map(|model| model.context_window)
            .unwrap_or(0);
        if is_context_overflow(message, Some(context_window)) {
            // Compaction owns overflow recovery.
            return Ok(false);
        }
        if !is_retryable_assistant_error(message) {
            return Ok(false);
        }

        let max_retries = retry
            .max_retries
            .unwrap_or(crate::settings::DEFAULT_MAX_RETRIES);
        let base_delay_ms = retry
            .base_delay_ms
            .unwrap_or(crate::settings::DEFAULT_RETRY_BASE_DELAY_MS);
        let attempt = {
            let mut state = self
                .shared
                .run_state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state.retry_attempt += 1;
            if state.retry_attempt > max_retries {
                state.retry_attempt -= 1;
                return Ok(false);
            }
            state.retry_attempt
        };

        let delay_ms = base_delay_ms.saturating_mul(1u64 << (attempt - 1).min(20));
        let error_message = message.error_message.clone().unwrap_or_default();
        let _ = self.shared.events.send(SessionEvent::AutoRetryStart {
            attempt,
            max_attempts: max_retries,
            delay_ms,
            error_message: error_message.clone(),
        });

        let signal = self.shared.agent.signal();
        let aborted = {
            let _timing = step_harness_providers::execution_timing::enter(step_harness_providers::execution_timing::Phase::Retry);
            match signal {
                Some(signal) => tokio::select! {
                    _ = tokio::time::sleep(std::time::Duration::from_millis(delay_ms)) => false,
                    _ = signal.cancelled() => true,
                },
                None => {
                    tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                    false
                }
            }
        };
        if aborted {
            let _ = self.shared.events.send(SessionEvent::AutoRetryEnd {
                success: false,
                attempt,
                final_error: Some("Retry cancelled".to_owned()),
            });
            self.shared
                .run_state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .retry_attempt = 0;
            return Ok(false);
        }

        // The failed assistant message stays in session history but must not
        // be the transcript tail the continuation continues from.
        self.strip_trailing_assistant_message();
        Ok(true)
    }

    /// Drop the trailing assistant message from the live transcript.
    fn strip_trailing_assistant_message(&self) {
        let mut messages = self.shared.agent.messages();
        if matches!(messages.last(), Some(AgentMessage::Assistant(_))) {
            messages.pop();
            self.shared.agent.set_messages(messages);
        }
    }

    /// Recall the last user message and its following turns from the active branch.
    /// The expected timestamp prevents a stale UI action from recalling another prompt.
    pub async fn recall_last_user_message(&self, expected_timestamp: u64) -> SessionResult<()> {
        let _run = self.shared.request_run_lock.try_lock().map_err(|_| {
            SessionFailure::Other("请先停止当前执行，再撤回消息。".into())
        })?;
        let entries = self.entries().await?;
        let index = entries
            .iter()
            .rposition(|entry| {
                matches!(entry, Entry::Message(entry) if entry.message.as_user().is_some())
            })
            .ok_or_else(|| SessionFailure::Other("没有可以撤回的用户消息。".into()))?;
        if entries[index].message().unwrap().timestamp() != expected_timestamp {
            return Err(SessionFailure::Other("会话消息已变化，请刷新后重试。".into()));
        }

        // Rewind the durable lane so reopening and future model requests see
        // exactly the same history. Keep user-selected runtime configuration.
        let previous_leaf = self.shared.session.get_leaf_id().await?;
        self.shared
            .session
            .move_lane("main", entries[index].parent_id())
            .await?;
        for entry in &entries[index + 1..] {
            let configuration = match entry {
                Entry::ModelChange(entry) => Some(ProvisionedEntry::ModelChange {
                    id: next_entry_id(),
                    provider: entry.provider.clone(),
                    model_id: entry.model_id.clone(),
                }),
                Entry::ThinkingLevelChange(entry) => Some(ProvisionedEntry::ThinkingLevelChange {
                    id: next_entry_id(),
                    thinking_level: entry.thinking_level.clone(),
                }),
                Entry::ActiveToolsChange(entry) => Some(ProvisionedEntry::ActiveToolsChange {
                    id: next_entry_id(),
                    active_tool_names: entry.active_tool_names.clone(),
                }),
                Entry::Custom(entry)
                    if entry.custom_type == APPROVAL_MODE_ENTRY
                        || entry.custom_type == PLAN_ENTRY_TYPE =>
                {
                    Some(ProvisionedEntry::Custom {
                        id: next_entry_id(),
                        custom_type: entry.custom_type.clone(),
                        data: entry.data.clone(),
                    })
                }
                _ => None,
            };
            if let Some(configuration) = configuration
                && let Err(error) = self.shared.session.append_entry(configuration, "main").await
            {
                self.shared.session.move_lane("main", previous_leaf.as_deref()).await?;
                return Err(error.into());
            }
        }
        self.rebuild_transcript_from_store().await?;
        self.clear_queue();
        let _ = self.shared.events.send(SessionEvent::MessagesRecalled {
            messages: self.shared.agent.messages(),
            execution_timings: self.execution_timings().await?,
        });
        Ok(())
    }

    /// Auto-compaction decision after a completed turn.
    async fn check_compaction(&self, message: &AssistantMessage) -> SessionResult<bool> {
        let settings = self.shared.services.settings.get();
        let file_settings = settings.compaction.unwrap_or_default();
        if !file_settings.enabled.unwrap_or(true) {
            return Ok(false);
        }
        let Some(model) = self.current_model() else {
            return Ok(false);
        };
        let same_model =
            model.provider.as_str() == message.provider.as_str() && model.id == message.model;

        let overflow = same_model && is_context_overflow(message, Some(model.context_window));
        let recoverable_length = same_model && is_recoverable_length(message, model.max_tokens);

        if overflow || recoverable_length {
            let will_retry = message.stop_reason != StopReason::Stop;
            if !will_retry {
                return self
                    .run_auto_compaction(CompactionReason::Overflow, false)
                    .await;
            }
            let already_attempted = self
                .shared
                .run_state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .overflow_recovery_attempted;
            if already_attempted {
                let _ = self.shared.events.send(SessionEvent::CompactionEnd {
                    reason: CompactionReason::Overflow,
                    error_message: Some("overflow recovery already attempted".to_owned()),
                });
                return Ok(false);
            }
            self.shared
                .run_state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .overflow_recovery_attempted = true;
            self.strip_trailing_assistant_message();
            return self
                .run_auto_compaction(CompactionReason::Overflow, true)
                .await;
        }

        let usage_tokens = calculate_context_tokens(&message.usage);
        let context_tokens = if message.stop_reason == StopReason::Error || usage_tokens == 0 {
            step_harness_agent_core::compaction::estimate_context_tokens(
                &self.shared.agent.state().messages,
            )
            .tokens
        } else {
            usage_tokens
        };
        let reserve = file_settings
            .reserve_tokens
            .unwrap_or(crate::settings::DEFAULT_COMPACTION_RESERVE_TOKENS);
        let keep_recent = file_settings
            .keep_recent_tokens
            .unwrap_or(crate::settings::DEFAULT_COMPACTION_KEEP_RECENT_TOKENS);
        let compaction_settings = CompactionSettings {
            enabled: true,
            reserve_tokens: reserve,
            keep_recent_tokens: keep_recent,
        };
        if should_compact(context_tokens, model.context_window, &compaction_settings) {
            return self
                .run_auto_compaction(CompactionReason::Threshold, false)
                .await;
        }
        Ok(false)
    }

    /// Summarize the session and rebuild the transcript from the store.
    async fn run_auto_compaction(
        &self,
        reason: CompactionReason,
        will_retry: bool,
    ) -> SessionResult<bool> {
        let entries = self
            .shared
            .session
            .find_entries_on_branch(&chronological_branch_query())
            .await?;
        let settings = self.shared.services.settings.get();
        let file_settings = settings.compaction.unwrap_or_default();
        let compaction_settings = CompactionSettings {
            enabled: true,
            reserve_tokens: file_settings
                .reserve_tokens
                .unwrap_or(crate::settings::DEFAULT_COMPACTION_RESERVE_TOKENS),
            keep_recent_tokens: file_settings
                .keep_recent_tokens
                .unwrap_or(crate::settings::DEFAULT_COMPACTION_KEEP_RECENT_TOKENS),
        };
        let Some(preparation) =
            prepare_compaction(&entries, &compaction_settings).map_err(|error| {
                SessionFailure::Other(format!("compaction preparation failed: {error}"))
            })?
        else {
            return Ok(false);
        };

        let Some(model) = self.current_model() else {
            return Ok(false);
        };
        let _ = self
            .shared
            .events
            .send(SessionEvent::CompactionStart { reason });

        let completer = CatalogCompleter {
            models: self.shared.models_handle.clone(),
            observer: self.shared.services.request_observer.clone(),
            request_context: step_harness_providers::request_metrics::RequestContext { session_id: self.shared.metadata.id.clone(), execution_id: self.request_execution_id(), purpose: "compaction".into() },
        };
        let signal = self.shared.agent.signal();
        let result = compact(
            &preparation,
            &completer,
            &model,
            None,
            signal.as_ref(),
            Some(self.thinking_level()),
            Some(&RetryPolicy {
                enabled: true,
                max_retries: 2,
                base_delay_ms: 1_000,
            }),
            None,
        )
        .await;

        match result {
            Ok(compacted) => {
                let entry = self
                    .shared
                    .session
                    .append_entry(
                        ProvisionedEntry::Compaction {
                            id: format!("cmp-{}", step_harness_providers::utils::now_ms()),
                            summary: compacted.summary,
                            retained_tail: compacted.retained_tail,
                            tokens_before: compacted.tokens_before,
                            details: compacted.details.map(|details| {
                                serde_json::to_value(details).unwrap_or(serde_json::Value::Null)
                            }),
                            usage: compacted.usage,
                        },
                        "main",
                    )
                    .await?;
                let _ = self.shared.events.send(SessionEvent::EntryAppended {
                    entry_id: entry.id().to_owned(),
                });
                self.rebuild_transcript_from_store().await?;
                let _ = self.shared.events.send(SessionEvent::CompactionEnd {
                    reason,
                    error_message: None,
                });
                Ok(will_retry)
            }
            Err(error) => {
                let _ = self.shared.events.send(SessionEvent::CompactionEnd {
                    reason,
                    error_message: Some(error.to_string()),
                });
                Ok(false)
            }
        }
    }

    /// Rebuild the agent transcript from the durable session store.
    async fn rebuild_transcript_from_store(&self) -> SessionResult<()> {
        let entries = self
            .shared
            .session
            .find_entries_on_branch(&chronological_branch_query())
            .await?;
        let context = build_session_context(&entries, &SessionContextBuildOptions::default());
        self.shared.agent.set_messages(context.messages);
        Ok(())
    }

    /// Manual compaction triggered from the UI.
    pub async fn compact_now(&self) -> SessionResult<bool> {
        // Compaction must not write a summary of history being recalled or
        // changed by a run. Automatic compaction already holds this lock.
        let _run = self.shared.request_run_lock.try_lock().map_err(|_| {
            SessionFailure::Other("会话正在执行其他操作，请稍后再压缩。".into())
        })?;
        self.run_auto_compaction(CompactionReason::Manual, false)
            .await
    }

    /// Rename the session.
    pub async fn set_name(&self, name: Option<&str>) -> SessionResult<()> {
        let mut revision = self.shared.name_revision.lock().await;
        // Preserve an explicit clear across reopening, so failure recovery
        // cannot silently regenerate a title the user intentionally removed.
        if name.is_none_or(|name| name.trim().is_empty()) {
            self.shared
                .session
                .append_custom_entry(naming::MANUAL_NAME_ENTRY, None)
                .await?;
        }
        self.shared.session.set_name(name).await?;
        *revision = revision.wrapping_add(1);
        let _ = self.shared.events.send(SessionEvent::SessionInfoChanged {
            name: name.map(ToOwned::to_owned),
        });
        Ok(())
    }

    /// The session's display name.
    pub async fn name(&self) -> Option<String> {
        self.shared.session.get_name().await
    }

    /// Assign the session to a project (`None` reverts to the cwd default).
    pub async fn set_project(&self, project: Option<&str>) -> SessionResult<()> {
        self.shared.session.set_project(project).await?;
        let _ = self.shared.events.send(SessionEvent::ProjectChanged {
            project: project.map(ToOwned::to_owned),
        });
        Ok(())
    }

    /// The session's project assignment.
    pub async fn project(&self) -> Option<String> {
        self.shared.session.get_project().await
    }

    /// Refresh model availability (optionally over the network).
    pub async fn refresh_models(&self, allow_network: bool) {
        let token = CancellationToken::new();
        self.shared
            .services
            .catalog
            .refresh(allow_network, &token)
            .await;
        if let Some(model) = self.current_model() {
            self.shared.agent.set_model(model);
        }
    }

    /// Reload settings and resources from disk.
    pub async fn reload(&self) {
        self.shared.services.reload(&default_env()).await;
        self.rebuild_system_prompt();
    }

    /// Persist a settings patch.
    pub fn update_settings(&self, patch: serde_json::Value) -> SessionResult<()> {
        self.shared
            .services
            .settings
            .update(patch)
            .map_err(SessionFailure::Other)
    }

    /// The current effective settings.
    pub fn settings(&self) -> Settings {
        self.shared.services.settings.get()
    }

    /// Set the approval mode at runtime, recording it with the session so a
    /// reopen restores it instead of the settings preset.
    pub async fn set_approval_mode(&self, mode: ApprovalMode) {
        let changed = {
            let mut policy = self
                .shared
                .permissions
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if policy.mode() == mode {
                false
            } else {
                policy.set_mode(mode);
                true
            }
        };
        // Reselecting the current mode must not grow the log, or every send
        // would pin the session to a mode the user never chose.
        if !changed {
            return;
        }
        self.record_configuration_change(ProvisionedEntry::Custom {
            id: next_entry_id(),
            custom_type: APPROVAL_MODE_ENTRY.to_owned(),
            data: Some(serde_json::json!({ "mode": mode.as_str() })),
        })
        .await;
    }

    /// Cycle the approval mode (ask → bypass → autopilot → ask).
    pub async fn cycle_approval_mode(&self) -> ApprovalMode {
        let next = match self.approval_mode() {
            ApprovalMode::Ask => ApprovalMode::Bypass,
            ApprovalMode::Bypass => ApprovalMode::Autopilot,
            ApprovalMode::Autopilot => ApprovalMode::Ask,
        };
        self.set_approval_mode(next).await;
        next
    }

    /// The durable session handle (for embedders listing entries).
    pub fn session(&self) -> &Session<JsonlSessionMetadata> {
        &self.shared.session
    }

    /// The chronological entry path of this session.
    pub async fn entries(&self) -> SessionResult<Vec<step_harness_agent_core::session::Entry>> {
        Ok(self
            .shared
            .session
            .find_entries_on_branch(&chronological_branch_query())
            .await?)
    }

    /// Completed execution durations restored from the durable transcript.
    pub async fn execution_timings(&self) -> SessionResult<Vec<ExecutionTiming>> {
        let entries = self.entries().await?;
        let mut timings: Vec<ExecutionTiming> = entries
            .iter()
            .filter_map(|entry| match entry {
                Entry::Custom(entry) if entry.custom_type == "execution_timing" => entry
                    .data
                    .clone()
                    .and_then(|value| serde_json::from_value(value).ok()),
                _ => None,
            })
            .collect();
        // An unfinished execution may already contain supplemental user messages.
        // Do not infer completed groups from those messages while it is running.
        if self.is_running() {
            timings.sort_by_key(|timing| timing.first_message_timestamp);
            return Ok(timings);
        }
        // Older sessions stored message-end timestamps but no execution records.
        // Restore their finished user/assistant groups without inventing timings.
        let mut first: Option<u64> = None;
        let mut last: Option<(u64, u64)> = None;
        for entry in &entries {
            if let Entry::Message(entry) = entry {
                if entry.message.as_user().is_some() {
                    if let (Some(start), Some((end, finished))) = (first, last)
                        && !timings.iter().any(|timing| {
                            timing.first_message_timestamp <= start
                                && timing.last_message_timestamp >= end
                        })
                    {
                        timings.push(ExecutionTiming {
                            first_message_timestamp: start,
                            last_message_timestamp: end,
                            duration_ms: finished.saturating_sub(start),
                            tool_duration_ms: None,
                            api_duration_ms: None,
                            approval_wait_duration_ms: None,
                            retry_wait_duration_ms: None,
                            other_duration_ms: None,
                        });
                    }
                    first = Some(entry.message.timestamp());
                    last = None;
                } else if entry.message.as_assistant().is_some()
                    || entry.message.as_tool_result().is_some()
                {
                    last = Some((entry.message.timestamp(), entry.timestamp));
                }
            }
        }
        if let (Some(start), Some((end, finished))) = (first, last)
            && !timings.iter().any(|timing| {
                timing.first_message_timestamp <= start && timing.last_message_timestamp >= end
            })
        {
            timings.push(ExecutionTiming {
                first_message_timestamp: start,
                last_message_timestamp: end,
                duration_ms: finished.saturating_sub(start),
                tool_duration_ms: None,
                api_duration_ms: None,
                approval_wait_duration_ms: None,
                retry_wait_duration_ms: None,
                other_duration_ms: None,
            });
        }
        timings.sort_by_key(|timing| timing.first_message_timestamp);
        Ok(timings)
    }
}

/// The default Node-backed execution environment.
pub fn default_env() -> Arc<dyn step_harness_agent_core::harness::types::ExecutionEnv> {
    Arc::new(step_harness_agent_core::harness::env::NodeExecutionEnv::new(Default::default()))
}

/// Build a fresh Step provider for login flows.
pub fn build_step_provider() -> Result<Arc<Provider>, String> {
    step_provider(&StepProviderOptions::default()).map_err(|error| error.to_string())
}

/// The built-in Step provider id.
pub const STEP_PROVIDER: &str = STEP_PROVIDER_ID;

/// Queue-mode re-export for embedders.
pub use step_harness_agent_core::types::QueueMode as SessionQueueMode;

/// Entry re-export for embedders rendering the session tree.
pub use step_harness_agent_core::session::Entry as SessionEntry;

/// Metadata re-export for embedders listing sessions.
pub use step_harness_agent_core::session::jsonl::JsonlSessionMetadata as SessionMetadata;

/// Settings manager re-export for embedders.
pub use crate::settings::SettingsManager as SessionSettingsManager;

/// Provider context re-export for tool implementors.
pub use step_harness_providers::types::Context as ProviderContext;

/// Usage re-export for embedders reading assistant messages.
pub use step_harness_providers::types::Usage as SessionUsage;

/// True when the agent loop rejected a request before it started.
pub fn is_already_running(error: &str) -> bool {
    error.contains("already processing")
}
