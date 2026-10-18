//! Bridge between the Tauri shell and the coding-agent runtime.
//!
//! The command surface mirrors the TypeScript RPC command taxonomy
//! (`prompt/steer/follow_up/abort/new_session/compact/set_model/...`), and
//! session events are forwarded to the web UI as Tauri events. Tool-call
//! confirmations travel as an `approval-request` event plus a
//! `respond_approval` command; plan reviews travel the same way through
//! `plan-review-request` and `respond_plan_review`.

use std::collections::HashMap;
use std::sync::Arc;

use base64::Engine;
use serde::{Deserialize, Serialize};
use step_harness_agent_core::types::AgentMessage;
use step_harness_providers::types::{ImageContent, Model, ModelThinkingLevel};
use tauri::{AppHandle, Emitter, Manager, State};
use tokio::sync::oneshot;

use step_harness_coding_agent::permissions::ApprovalMode;
use step_harness_coding_agent::plan::{PlanReviewChannel, PlanReviewOutcome, PlanReviewRequest, PlanSource, PlanState};
use step_harness_coding_agent::session::{
    AgentSession, ApprovalChannel, ApprovalRequest, SessionRuntime, SessionServices,
};
use step_harness_coding_agent::{AppPaths, ModelCatalog};

/// Event name for forwarded session events.
pub const EVENT_SESSION: &str = "session-event";
/// Event name for tool-call confirmation requests.
pub const EVENT_APPROVAL: &str = "approval-request";
/// Event name for plan review requests.
pub const EVENT_PLAN_REVIEW: &str = "plan-review-request";

/// Cancellation and browser URL shared by an in-progress login.
type LoginAttempt = (
    tokio_util::sync::CancellationToken,
    Arc<std::sync::Mutex<Option<String>>>,
);

/// Application state: the runtime, approval channel, and forwarder handles.
pub struct AppState {
    pub requests: Arc<crate::request_store::RequestStore>,
    forwarders: std::sync::Mutex<std::collections::HashSet<String>>,
    pub runtime: Arc<SessionRuntime>,
    pub approvals: Arc<TauriApprovalChannel>,
    pub plan_reviews: Arc<TauriPlanReviewChannel>,
    pub configuration_lock: tokio::sync::Mutex<()>,
    pub login: std::sync::Mutex<Option<LoginAttempt>>,
}

/// Removes resolved/cancelled requests even when their waiting future is dropped.
struct PendingRequestGuard<'a, R, O> {
    pending: &'a std::sync::Mutex<HashMap<String, (R, oneshot::Sender<O>)>>,
    app: &'a AppHandle,
    request_id: &'a str,
    session_id: &'a str,
    kind: &'static str,
}

impl<R, O> Drop for PendingRequestGuard<'_, R, O> {
    fn drop(&mut self) {
        self.pending.lock().unwrap_or_else(|p| p.into_inner()).remove(self.request_id);
        let _ = self.app.emit(EVENT_SESSION, serde_json::json!({
            "sessionId": self.session_id, "type": "requestResolved",
            "requestId": self.request_id, "kind": self.kind
        }));
    }
}

/// Approval channel that raises a UI dialog and waits for the answer.
pub struct TauriApprovalChannel {
    app: AppHandle,
    pending: std::sync::Mutex<HashMap<String, (ApprovalRequest, oneshot::Sender<bool>)>>,
}

impl TauriApprovalChannel {
    fn new(app: AppHandle) -> Arc<Self> {
        Arc::new(TauriApprovalChannel {
            app,
            pending: std::sync::Mutex::new(HashMap::new()),
        })
    }

    fn requests(&self, session_id: &str) -> Vec<ApprovalRequest> {
        self.pending.lock().unwrap_or_else(|p| p.into_inner()).values()
            .filter(|(request, _)| request.session_id == session_id)
            .map(|(request, _)| request.clone()).collect()
    }

    /// Resolve a pending request; returns false when the id is unknown.
    fn resolve(&self, request_id: &str, approved: bool) -> bool {
        let sender = self
            .pending
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(request_id);
        match sender {
            Some((_, sender)) => sender.send(approved).is_ok(),
            None => false,
        }
    }
}

#[async_trait::async_trait]
impl ApprovalChannel for TauriApprovalChannel {
    async fn confirm(&self, request: &ApprovalRequest) -> bool {
        let (sender, receiver) = oneshot::channel();
        self.pending
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(request.request_id.clone(), (request.clone(), sender));
        let _pending = PendingRequestGuard { pending: &self.pending, app: &self.app,
            request_id: &request.request_id, session_id: &request.session_id, kind: "approval" };
        if let Err(error) = self.app.emit(EVENT_APPROVAL, request) {
            self.pending
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(&request.request_id);
            eprintln!("approval emit failed: {error}");
            return false;
        }
        // Wait for a decision indefinitely. Aborting the run drops this future
        // and the pending-request guard removes the UI request.
        receiver.await.unwrap_or(false)
    }
}

/// Plan review channel that raises the review UI and waits for the decision.
pub struct TauriPlanReviewChannel {
    app: AppHandle,
    pending: std::sync::Mutex<HashMap<String, (PlanReviewRequest, oneshot::Sender<PlanReviewOutcome>)>>,
}

impl TauriPlanReviewChannel {
    fn new(app: AppHandle) -> Arc<Self> {
        Arc::new(TauriPlanReviewChannel {
            app,
            pending: std::sync::Mutex::new(HashMap::new()),
        })
    }

    fn requests(&self, session_id: &str) -> Vec<PlanReviewRequest> {
        self.pending.lock().unwrap_or_else(|p| p.into_inner()).values()
            .filter(|(request, _)| request.session_id == session_id)
            .map(|(request, _)| request.clone()).collect()
    }

    /// Resolve a pending review; returns false when the id is unknown.
    fn resolve(&self, request_id: &str, outcome: PlanReviewOutcome) -> bool {
        let sender = self
            .pending
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(request_id);
        match sender {
            Some((_, sender)) => sender.send(outcome).is_ok(),
            None => false,
        }
    }
}

#[async_trait::async_trait]
impl PlanReviewChannel for TauriPlanReviewChannel {
    async fn review(&self, request: &PlanReviewRequest) -> PlanReviewOutcome {
        let (sender, receiver) = oneshot::channel();
        self.pending
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(request.request_id.clone(), (request.clone(), sender));
        let _pending = PendingRequestGuard { pending: &self.pending, app: &self.app,
            request_id: &request.request_id, session_id: &request.session_id, kind: "planReview" };
        if let Err(error) = self.app.emit(EVENT_PLAN_REVIEW, request) {
            self.pending
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(&request.request_id);
            eprintln!("plan review emit failed: {error}");
            return PlanReviewOutcome::Dismissed;
        }
        // Like tool approvals, reviews stay pending until a decision or cancellation.
        receiver.await.unwrap_or(PlanReviewOutcome::Dismissed)
    }
}

/// A model as presented to the UI.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    pub provider: String,
    pub id: String,
    pub name: String,
    pub reasoning: bool,
    pub supported_thinking_levels: Vec<ModelThinkingLevel>,
    pub context_window: u64,
    pub configured: bool,
    pub supports_images: bool,
}

impl ModelInfo {
    fn from_model(model: &Model, catalog: &ModelCatalog) -> Self {
        ModelInfo {
            provider: model.provider.to_string(),
            id: model.id.clone(),
            name: model.name.clone(),
            reasoning: model.reasoning,
            supported_thinking_levels:
                step_harness_providers::models::get_supported_thinking_levels(model),
            context_window: model.context_window,
            configured: catalog.is_provider_configured(model.provider.as_str()),
            supports_images: model.supports_images(),
        }
    }
}

/// A session as presented to the UI.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionSummary {
    pub id: String,
    pub path: String,
    pub cwd: String,
    pub created_at: u64,
    pub modified_at: u64,
    pub parent_session_id: Option<String>,
    pub name: Option<String>,
    /// The session's project assignment; `None` follows the cwd default.
    pub project: Option<String>,
}

impl SessionSummary {
    async fn from_session(session: &AgentSession) -> Self {
        let metadata = session.metadata();
        SessionSummary {
            id: metadata.id.clone(),
            path: metadata.path.clone(),
            cwd: metadata.cwd.clone(),
            created_at: metadata.created_at,
            modified_at: metadata.modified_at,
            parent_session_id: metadata.parent_session_id.clone(),
            name: session.name().await,
            project: session.project().await,
        }
    }
}

/// Everything the UI needs to render its first frame.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Bootstrap {
    pub cwd: String,
    pub data_dir: String,
    pub settings: serde_json::Value,
    pub models: Vec<ModelInfo>,
    pub current_model: Option<ModelInfo>,
    pub thinking_level: String,
    pub approval_mode: String,
    pub plan_active: bool,
    pub plan_file_path: Option<String>,
    pub sessions: Vec<SessionSummary>,
    pub projects: Vec<String>,
    pub removed_projects: Vec<String>,
    pub active_session: Option<SessionSummary>,
    pub running: bool,
    pub step_authenticated: bool,
    pub transcript: Vec<AgentMessage>,
    pub stats: Option<serde_json::Value>,
    pub approvals: Vec<ApprovalRequest>,
    pub plan_reviews: Vec<PlanReviewRequest>,
    pub execution_timings: Vec<step_harness_coding_agent::session::ExecutionTiming>,
}

fn thinking_level_label(level: ModelThinkingLevel) -> &'static str {
    match level {
        ModelThinkingLevel::Off => "off",
        ModelThinkingLevel::Minimal => "minimal",
        ModelThinkingLevel::Low => "low",
        ModelThinkingLevel::Medium => "medium",
        ModelThinkingLevel::High => "high",
        ModelThinkingLevel::Xhigh => "xhigh",
        ModelThinkingLevel::Max => "max",
    }
}

/// The execution environment used by every tool.
fn env() -> Arc<dyn step_harness_agent_core::harness::types::ExecutionEnv> {
    step_harness_coding_agent::session::default_env()
}

/// Register managed state and forward session events to the window.
pub fn setup(app: &mut tauri::App) -> tauri::Result<()> {
    let handle = app.handle().clone();
    let cwd = std::env::current_dir()
        .map(|dir| dir.to_string_lossy().to_string())
        .unwrap_or_else(|_| ".".to_owned());
    let paths = AppPaths::discover();
    let notify_handle = handle.clone();
    let requests = crate::request_store::RequestStore::open(paths.root().join("requests.sqlite3"), move |session_id| { let _ = notify_handle.emit("request-record-updated", serde_json::json!({"sessionId": session_id})); });
    let approvals = TauriApprovalChannel::new(handle.clone());
    let plan_reviews = TauriPlanReviewChannel::new(handle.clone());
    let runtime = Arc::new(SessionRuntime::new(
        cwd,
        paths,
        approvals.clone() as Arc<dyn ApprovalChannel>,
        plan_reviews.clone() as Arc<dyn PlanReviewChannel>,
    ).with_request_observer(requests.clone()));
    app.manage(AppState {
        requests,
        runtime,
        forwarders: std::sync::Mutex::new(std::collections::HashSet::new()),
        approvals: approvals.clone(),
        plan_reviews: plan_reviews.clone(),
        configuration_lock: tokio::sync::Mutex::new(()),
        login: std::sync::Mutex::new(None),
    });

    // Bootstrap loads services without allocating an empty session.
    Ok(())
}

/// Forward one session's events to the window until the receiver lags out.
fn forward_events(handle: AppHandle, session: Arc<AgentSession>) {
    let session_id = session.metadata().id.clone();
    if !handle.state::<AppState>().forwarders.lock().unwrap_or_else(|p| p.into_inner()).insert(session_id.clone()) {
        return;
    }
    let mut events = session.subscribe();
    drop(session);
    tauri::async_runtime::spawn(async move {
        loop {
            match events.recv().await {
                Ok(event) => {
                    let mut payload = session_event_payload(&event);
                    payload["sessionId"] = serde_json::json!(session_id);
                    if handle.emit(EVENT_SESSION, payload).is_err() { break; }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                    eprintln!("session event stream lagged, skipped {skipped} events");
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
        handle.state::<AppState>().forwarders.lock().unwrap_or_else(|p| p.into_inner()).remove(&session_id);
    });
}

/// Resolve commands against their originating session, never a later selection.
async fn command_session(state: &AppState, session_id: Option<&str>) -> Result<Arc<AgentSession>, String> {
    match session_id {
        Some(id) => state.runtime.session(id).await,
        None => state.runtime.active().await,
    }.ok_or_else(|| "会话不存在或尚未打开。".to_owned())
}

/// Bootstrap payload for the initial render.
#[tauri::command]
pub async fn get_bootstrap(state: State<'_, AppState>, app: tauri::AppHandle, session_id: Option<String>) -> Result<Bootstrap, String> {
    let env = env();
    let active = match session_id.as_deref() {
        Some(id) => Some(command_session(&state, Some(id)).await?),
        None => match state.runtime.active().await {
            Some(session) => Some(session),
            // A restart reopens the session the previous run left active, so
            // the UI comes back to the conversation the user was in.
            None => match state.runtime.restore_last_active(&env).await {
                Some(session) => {
                    forward_events(app, session.clone());
                    Some(session)
                }
                None => None,
            },
        },
    };
    let cwd = active.as_ref().map(|session| session.metadata().cwd.as_str()).unwrap_or(state.runtime.cwd());
    let services: Arc<SessionServices> = state
        .runtime
        .services_for_cwd(&env, cwd)
        .await
        .map_err(|error| error.to_string())?;
    let catalog: &ModelCatalog = &services.catalog;
    let models: Vec<ModelInfo> = catalog
        .snapshot()
        .available_models
        .iter()
        .map(|model| ModelInfo::from_model(model, catalog))
        .collect();

    let sessions = session_summaries(&state).await?;

    let active_summary = match &active {
        Some(session) => Some(SessionSummary::from_session(session).await),
        None => None,
    };
    let (current_model, thinking_level, approval_mode, plan_state, running) = match &active {
        Some(session) => (
            session
                .current_model()
                .map(|model| ModelInfo::from_model(&model, catalog)),
            thinking_level_label(session.thinking_level()).to_owned(),
            session.approval_mode().as_str().to_owned(),
            session.plan_state(),
            session.is_running(),
        ),
        None => (
            None,
            thinking_level_label(ModelThinkingLevel::Off).to_owned(),
            ApprovalMode::Ask.as_str().to_owned(),
            PlanState::default(),
            false,
        ),
    };

    let project_registry = read_project_registry(&state)?;
    let id = active.as_ref().map(|session| session.metadata().id.as_str()).unwrap_or("");
    let approvals = state.approvals.requests(id);
    let plan_reviews = state.plan_reviews.requests(id);
    let transcript = active.as_ref().map(|session| session.state().messages).unwrap_or_default();
    let stats = match &active {
        Some(session) => Some(stats_payload(session, &state.requests).await?),
        None => None,
    };
    let execution_timings = match &active {
        Some(session) => session.execution_timings().await.map_err(|error| error.to_string())?,
        None => Vec::new(),
    };
    Ok(Bootstrap {
        cwd: services.cwd.clone(),
        data_dir: state.runtime.paths().root().to_string_lossy().to_string(),
        step_authenticated: services
            .credentials
            .read("step", &tokio_util::sync::CancellationToken::new())
            .await
            .map_err(|e| e.to_string())?
            .is_some(),
        settings: serde_json::to_value(services.settings()).unwrap_or(serde_json::Value::Null),
        models,
        current_model,
        thinking_level,
        approval_mode,
        plan_active: plan_state.enabled,
        plan_file_path: plan_state.plan_file_path,
        sessions,
        projects: project_registry.projects,
        removed_projects: project_registry.removed,
        active_session: active_summary,
        running,
        transcript, stats, approvals, plan_reviews, execution_timings,
    })
}

/// Create a new session and make it active.
#[tauri::command]
pub async fn new_session(
    state: State<'_, AppState>,
    app: tauri::AppHandle,
    cwd: Option<String>,
) -> Result<SessionSummary, String> {
    let env = env();
    let session = match cwd {
        Some(cwd) => {
            let cwd = canonical_project_path(&cwd)?;
            state.runtime.new_session_in(&env, &cwd).await
        }
        None => state.runtime.new_session(&env).await,
    }.map_err(|error| error.to_string())?;
    forward_events(app, session.clone());
    Ok(SessionSummary::from_session(&session).await)
}

fn canonical_project_path(path: &str) -> Result<String, String> {
    let path = std::fs::canonicalize(path).map_err(|e| format!("无法打开项目文件夹：{e}"))?;
    if !path.is_dir() { return Err("请选择文件夹作为项目".into()); }
    Ok(path.to_string_lossy().trim_start_matches(r"\\?\").to_owned())
}

#[derive(Default, Serialize, Deserialize)]
struct ProjectRegistry {
    projects: Vec<String>,
    #[serde(default)]
    removed: Vec<String>,
}

fn project_path_key(path: &str) -> String {
    let normalized = path.trim_start_matches(r"\\?\").replace('\\', "/").trim_end_matches('/').to_owned();
    if cfg!(windows) { normalized.to_lowercase() } else { normalized }
}

fn read_project_registry(state: &AppState) -> Result<ProjectRegistry, String> {
    let path = state.runtime.paths().root().join("projects.json");
    match std::fs::read(path) {
        Ok(data) => parse_project_registry(&data),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(ProjectRegistry::default()),
        Err(e) => Err(format!("无法读取项目列表：{e}")),
    }
}

fn parse_project_registry(data: &[u8]) -> Result<ProjectRegistry, String> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Saved { Legacy(Vec<String>), Registry(ProjectRegistry) }
    match serde_json::from_slice(data).map_err(|e| format!("无法读取项目列表：{e}"))? {
        Saved::Legacy(projects) => Ok(ProjectRegistry { projects, removed: Vec::new() }),
        Saved::Registry(registry) => Ok(registry),
    }
}

fn save_project_registry(state: &AppState, registry: &ProjectRegistry) -> Result<(), String> {
    let root = state.runtime.paths().root();
    std::fs::create_dir_all(root).map_err(|e| e.to_string())?;
    let data = serde_json::to_vec_pretty(registry).map_err(|e| e.to_string())?;
    let staging = root.join("projects.json.tmp");
    std::fs::write(&staging, data).map_err(|e| format!("无法保存项目：{e}"))?;
    std::fs::rename(staging, root.join("projects.json")).map_err(|e| format!("无法保存项目：{e}"))
}

/// Hide a project without touching its directory, sessions or drafts.
#[tauri::command]
pub async fn remove_project(state: State<'_, AppState>, path: String) -> Result<(), String> {
    let _guard = state.configuration_lock.lock().await;
    let mut registry = read_project_registry(&state)?;
    let key = project_path_key(&path);
    if key.is_empty() { return Err("项目路径不能为空".into()); }
    registry.projects.retain(|project| project_path_key(project) != key);
    if !registry.removed.iter().any(|project| project_path_key(project) == key) {
        registry.removed.push(path);
    }
    save_project_registry(&state, &registry)
}

/// The native picker only registers a folder; cancellation leaves state untouched.
#[tauri::command]
pub async fn add_project(state: State<'_, AppState>) -> Result<Option<String>, String> {
    let selected = tauri::async_runtime::spawn_blocking(|| {
        rfd::FileDialog::new().set_title("选择项目文件夹").pick_folder()
    }).await.map_err(|e| e.to_string())?;
    let Some(selected) = selected else { return Ok(None); };
    let path = canonical_project_path(&selected.to_string_lossy())?;
    let _guard = state.configuration_lock.lock().await;
    let mut registry = read_project_registry(&state)?;
    let key = project_path_key(&path);
    registry.removed.retain(|project| project_path_key(project) != key);
    if !registry.projects.iter().any(|project| project_path_key(project) == key) {
        registry.projects.push(path.clone());
    }
    save_project_registry(&state, &registry)?;
    Ok(Some(path))
}

/// Open an existing session.
#[tauri::command]
pub async fn open_session(
    state: State<'_, AppState>,
    app: tauri::AppHandle,
    session_id: String,
) -> Result<SessionSummary, String> {
    // The active session already holds its file's lease, so reopening it would
    // fail; the live session is the same one.
    if let Some(active) = state
        .runtime
        .active()
        .await
        .filter(|session| session.metadata().id == session_id)
    {
        return Ok(SessionSummary::from_session(&active).await);
    }
    let env = env();
    let sessions = state
        .runtime
        .list_sessions(&env)
        .await
        .map_err(|error| error.to_string())?;
    let metadata = sessions
        .into_iter()
        .find(|metadata| metadata.id == session_id)
        .ok_or_else(|| format!("session not found: {session_id}"))?;
    let session = state
        .runtime
        .open_session(&env, metadata)
        .await
        .map_err(|error| error.to_string())?;
    forward_events(app, session.clone());
    Ok(SessionSummary::from_session(&session).await)
}

/// Delete a session.
#[tauri::command]
pub async fn delete_session(state: State<'_, AppState>, session_id: String) -> Result<(), String> {
    let env = env();
    let sessions = state
        .runtime
        .list_sessions(&env)
        .await
        .map_err(|error| error.to_string())?;
    let metadata = sessions
        .into_iter()
        .find(|metadata| metadata.id == session_id)
        .ok_or_else(|| format!("session not found: {session_id}"))?;
    state
        .runtime
        .delete_session(&env, &metadata)
        .await
        .map_err(|error| error.to_string())
}

/// List sessions for the working directory.
#[tauri::command]
pub async fn list_sessions(state: State<'_, AppState>) -> Result<Vec<SessionSummary>, String> {
    session_summaries(&state).await
}

/// The active session's transcript.
#[tauri::command]
pub async fn get_transcript(state: State<'_, AppState>, session_id: Option<String>) -> Result<Vec<AgentMessage>, String> {
    let session = command_session(&state, session_id.as_deref()).await?;
    Ok(session.state().messages)
}

/// Recall the latest user prompt and all following messages in its session.
#[tauri::command]
pub async fn recall_last_user_message(state: State<'_, AppState>, session_id: String, timestamp: u64) -> Result<(), String> {
    let session = command_session(&state, Some(&session_id)).await?;
    session.recall_last_user_message(timestamp).await.map_err(|error| error.to_string())
}

/// Validate attachments before acknowledging a send or queuing a message.
fn validate_prompt_images(supports_images: bool, text: &str, images: &[ImageContent]) -> Result<(), String> {
    if text.trim().is_empty() && images.is_empty() {
        return Err("请输入文字或添加图片。".to_owned());
    }
    if images.len() > 8 {
        return Err("最多可添加 8 张图片。".to_owned());
    }
    if !images.is_empty() && !supports_images {
        return Err("当前模型不支持图片，请切换支持图片的模型。".to_owned());
    }
    for image in images {
        if !["image/png", "image/jpeg", "image/gif", "image/webp"].contains(&image.mime_type.as_str()) || image.data.len() > 13_981_016 {
            return Err("仅支持不超过 10 MB 的 PNG、JPEG、GIF 或 WebP 图片。".to_owned());
        }
        let bytes = base64::engine::general_purpose::STANDARD.decode(&image.data)
            .map_err(|_| "图片数据无效。".to_owned())?;
        if bytes.is_empty() || bytes.len() > 10 * 1024 * 1024 ||
            step_harness_agent_core::harness::tools::detect_supported_image_mime_type(&bytes) != Some(image.mime_type.as_str()) {
            return Err("图片内容或格式无效。".to_owned());
        }
    }
    Ok(())
}

/// Model-visible metadata for an explicitly selectable skill.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SkillSummary {
    name: String,
    description: String,
    file_path: String,
}

/// Discover skills for a draft project without creating a session.
#[tauri::command]
pub async fn list_skills(state: State<'_, AppState>, cwd: String) -> Result<Vec<SkillSummary>, String> {
    let resources = load_skill_resources(&state, &cwd).await?;
    let mut seen = std::collections::HashSet::new();
    Ok(resources.skills.into_iter().filter(|skill| seen.insert(skill.file_path.clone())).map(|skill| SkillSummary {
        name: skill.name, description: skill.description, file_path: skill.file_path,
    }).collect())
}

async fn load_skill_resources(state: &AppState, cwd: &str) -> Result<step_harness_coding_agent::resources::ResourceSet, String> {
    let env = step_harness_coding_agent::session::tools::env_for_cwd(cwd);
    let services = state.runtime.services_for_cwd(&env, cwd).await.map_err(|error| error.to_string())?;
    Ok(step_harness_coding_agent::resources::ResourceSet::load(
        &env, cwd, state.runtime.paths(), &services.settings(),
    ).await)
}

async fn expand_selected_skills(state: &AppState, session: &AgentSession, text: String, skills: Option<Vec<String>>) -> Result<String, String> {
    let paths = skills.unwrap_or_default();
    if paths.is_empty() { return Ok(text); }
    // Re-read instructions at send time; only discovered skills in this project are accepted.
    load_skill_resources(state, &session.metadata().cwd).await?.prompt_with_skills(&text, &paths)
}

/// Submit a prompt. The turn runs in the background; results arrive as session events.
#[tauri::command]
pub async fn prompt(
    state: State<'_, AppState>,
    session_id: Option<String>,
    app: tauri::AppHandle,
    text: String,
    images: Option<Vec<ImageContent>>,
    skills: Option<Vec<String>>,
) -> Result<(), String> {
    let session = command_session(&state, session_id.as_deref()).await?;
    let text = expand_selected_skills(&state, &session, text, skills).await?;
    let images = images.unwrap_or_default();
    validate_prompt_images(session.current_model().is_some_and(|model| model.supports_images()), &text, &images)?;
    if session.is_running() {
        return Err("a turn is already running; use steer or follow_up".to_owned());
    }
    tauri::async_runtime::spawn(async move {
        if let Err(error) = session.prompt(text, Some(images)).await {
            let _ = app.emit(
                EVENT_SESSION,
                serde_json::json!({ "sessionId": session.metadata().id, "type": "error", "message": error.to_string() }),
            );
        }
    });
    Ok(())
}

/// Abort the active turn.
#[tauri::command]
pub async fn abort(state: State<'_, AppState>, session_id: Option<String>) -> Result<(), String> {
    let session = command_session(&state, session_id.as_deref()).await?;
    session.abort();
    for request in state.approvals.requests(&session.metadata().id) {
        state.approvals.resolve(&request.request_id, false);
    }
    for request in state.plan_reviews.requests(&session.metadata().id) {
        state.plan_reviews.resolve(&request.request_id, PlanReviewOutcome::Dismissed);
    }
    Ok(())
}

/// Queue a steering message.
#[tauri::command]
pub async fn steer(state: State<'_, AppState>, session_id: Option<String>, text: String, images: Option<Vec<ImageContent>>, skills: Option<Vec<String>>) -> Result<(), String> {
    let session = command_session(&state, session_id.as_deref()).await?;
    let text = expand_selected_skills(&state, &session, text, skills).await?;
    let images = images.unwrap_or_default();
    validate_prompt_images(session.current_model().is_some_and(|model| model.supports_images()), &text, &images)?;
    session.steer_with_images(text, images);
    Ok(())
}

/// Queue a follow-up message.
#[tauri::command]
pub async fn follow_up(state: State<'_, AppState>, session_id: Option<String>, text: String, images: Option<Vec<ImageContent>>, skills: Option<Vec<String>>) -> Result<(), String> {
    let session = command_session(&state, session_id.as_deref()).await?;
    let text = expand_selected_skills(&state, &session, text, skills).await?;
    let images = images.unwrap_or_default();
    validate_prompt_images(session.current_model().is_some_and(|model| model.supports_images()), &text, &images)?;
    session.follow_up_with_images(text, images);
    Ok(())
}

/// Switch the active model.
#[tauri::command]
pub async fn set_model(
    state: State<'_, AppState>,
    session_id: Option<String>,
    provider: String,
    model_id: String,
) -> Result<(), String> {
    let session = command_session(&state, session_id.as_deref()).await?;
    session
        .set_model(&provider, &model_id)
        .await
        .map_err(|error| error.to_string())
}

/// Set the reasoning level.
#[tauri::command]
pub async fn set_thinking_level(state: State<'_, AppState>, session_id: Option<String>, level: String) -> Result<(), String> {
    let session = command_session(&state, session_id.as_deref()).await?;
    let level = match level.to_ascii_lowercase().as_str() {
        "off" => ModelThinkingLevel::Off,
        "minimal" => ModelThinkingLevel::Minimal,
        "low" => ModelThinkingLevel::Low,
        "medium" => ModelThinkingLevel::Medium,
        "high" => ModelThinkingLevel::High,
        "xhigh" => ModelThinkingLevel::Xhigh,
        "max" => ModelThinkingLevel::Max,
        other => return Err(format!("unknown thinking level: {other}")),
    };
    session.set_thinking_level(level).await;
    Ok(())
}

/// Set the approval mode.
#[tauri::command]
pub async fn set_approval_mode(state: State<'_, AppState>, session_id: Option<String>, mode: String) -> Result<(), String> {
    let session = command_session(&state, session_id.as_deref()).await?;
    let mode = match mode.to_ascii_lowercase().as_str() {
        "ask" | "confirm" => ApprovalMode::Ask,
        "read-only" | "readonly" | "strict" => ApprovalMode::Ask,
        "bypass" | "auto" => ApprovalMode::Bypass,
        "autopilot" => ApprovalMode::Autopilot,
        other => return Err(format!("unknown approval mode: {other}")),
    };
    session.set_approval_mode(mode).await;
    Ok(())
}

/// Answer a tool-call confirmation.
#[tauri::command]
pub async fn respond_approval(
    state: State<'_, AppState>,
    request_id: String,
    approved: bool,
) -> Result<(), String> {
    if state.approvals.resolve(&request_id, approved) {
        Ok(())
    } else {
        Err(format!("unknown approval request: {request_id}"))
    }
}

/// Enter plan mode on the active session; returns the plan file path.
#[tauri::command]
pub async fn enter_plan_mode(state: State<'_, AppState>, session_id: Option<String>) -> Result<String, String> {
    let session = command_session(&state, session_id.as_deref()).await?;
    let snapshot = session.enter_plan_mode(PlanSource::User).await;
    Ok(snapshot.plan_file_path.unwrap_or_default())
}

/// Leave plan mode on the active session.
#[tauri::command]
pub async fn exit_plan_mode(state: State<'_, AppState>, session_id: Option<String>) -> Result<(), String> {
    let session = command_session(&state, session_id.as_deref()).await?;
    session.exit_plan_mode().await;
    Ok(())
}

/// Toggle plan mode; returns whether plan mode is now active.
#[tauri::command]
pub async fn toggle_plan_mode(state: State<'_, AppState>, session_id: Option<String>) -> Result<bool, String> {
    let session = command_session(&state, session_id.as_deref()).await?;
    Ok(session.toggle_plan_mode().await.enabled)
}

/// Answer a plan review.
#[tauri::command]
pub async fn respond_plan_review(
    state: State<'_, AppState>,
    request_id: String,
    outcome: String,
    feedback: Option<String>,
) -> Result<(), String> {
    let outcome = match outcome.to_ascii_lowercase().as_str() {
        "approved" => PlanReviewOutcome::Approved,
        "feedback" => match feedback.filter(|text| !text.trim().is_empty()) {
            Some(notes) => PlanReviewOutcome::Feedback(notes),
            None => PlanReviewOutcome::Dismissed,
        },
        "dismissed" => PlanReviewOutcome::Dismissed,
        other => return Err(format!("unknown plan review outcome: {other}")),
    };
    if state.plan_reviews.resolve(&request_id, outcome) {
        Ok(())
    } else {
        Err(format!("unknown plan review request: {request_id}"))
    }
}

/// Run compaction now.
#[tauri::command]
pub async fn compact(state: State<'_, AppState>, session_id: Option<String>) -> Result<(), String> {
    let session = command_session(&state, session_id.as_deref()).await?;
    session
        .compact_now()
        .await
        .map(|_| ())
        .map_err(|error| error.to_string())
}

/// Patch settings (persisted to the project or global file).
#[tauri::command]
pub async fn update_settings(
    state: State<'_, AppState>,
    session_id: Option<String>,
    patch: serde_json::Value,
) -> Result<(), String> {
    let session = command_session(&state, session_id.as_deref()).await?;
    session
        .update_settings(patch)
        .map_err(|error| error.to_string())
}

/// Rename the active session.
#[tauri::command]
pub async fn rename_session(
    state: State<'_, AppState>,
    session_id: String,
    name: String,
) -> Result<(), String> {
    let name = name.trim().to_owned();
    if let Some(session) = state.runtime.session(&session_id).await
    {
        return session
            .set_name(if name.is_empty() { None } else { Some(&name) })
            .await
            .map_err(|e| e.to_string());
    }
    let metadata = state
        .runtime
        .list_sessions(&env())
        .await
        .map_err(|e| e.to_string())?
        .into_iter()
        .find(|m| m.id == session_id)
        .ok_or("会话不存在。")?;
    let repo = session_repo(&state);
    repo.open(&metadata)
        .await
        .map_err(|e| e.to_string())?
        .set_name(if name.is_empty() { None } else { Some(&name) })
        .await
        .map_err(|e| e.to_string())
}

/// Assign a session to a project. An empty name clears the assignment so the
/// session falls back to its working-directory default.
#[tauri::command]
pub async fn set_session_project(
    state: State<'_, AppState>,
    session_id: String,
    project: String,
) -> Result<(), String> {
    let project = project.trim().to_owned();
    let assignment = if project.is_empty() { None } else { Some(project) };
    if let Some(session) = state.runtime.session(&session_id).await
    {
        return session
            .set_project(assignment.as_deref())
            .await
            .map_err(|e| e.to_string());
    }
    let metadata = state
        .runtime
        .list_sessions(&env())
        .await
        .map_err(|e| e.to_string())?
        .into_iter()
        .find(|m| m.id == session_id)
        .ok_or("会话不存在。")?;
    session_repo(&state)
        .open(&metadata)
        .await
        .map_err(|e| e.to_string())?
        .set_project(assignment.as_deref())
        .await
        .map_err(|e| e.to_string())
}

fn session_repo(state: &AppState) -> step_harness_agent_core::session::jsonl::JsonlSessionRepo {
    step_harness_agent_core::session::jsonl::JsonlSessionRepo::new(
        env(),
        step_harness_agent_core::session::jsonl::JsonlSessionRepoOptions {
            sessions_root: state
                .runtime
                .paths()
                .sessions_dir()
                .to_string_lossy()
                .into_owned(),
        },
    )
}
async fn session_summaries(state: &AppState) -> Result<Vec<SessionSummary>, String> {
    let named = state
        .runtime
        .list_sessions_with_names(&env())
        .await
        .map_err(|e| e.to_string())?;
    Ok(named
        .into_iter()
        .map(|(item, name, project)| SessionSummary {
            id: item.id,
            path: item.path,
            cwd: item.cwd,
            created_at: item.created_at,
            modified_at: item.modified_at,
            parent_session_id: item.parent_session_id,
            name,
            project,
        })
        .collect())
}

/// Refresh model availability, optionally over the network.
#[tauri::command]
pub async fn refresh_models(state: State<'_, AppState>, allow_network: bool) -> Result<(), String> {
    let session = state
        .runtime
        .active()
        .await
        .ok_or_else(|| "no active session".to_owned())?;
    session.refresh_models(allow_network).await;
    Ok(())
}

/// Reload settings and resources from disk.
#[tauri::command]
pub async fn reload(state: State<'_, AppState>) -> Result<(), String> {
    state
        .runtime
        .reload(&env())
        .await
        .map_err(|error| error.to_string())
}

/// Session stats for the status bar.
#[tauri::command]
pub async fn get_usage_statistics(state: State<'_, AppState>) -> Result<crate::usage::UsageSnapshot, String> {
    state.requests.usage().await
}

#[tauri::command]
pub async fn get_model_health(state: State<'_, AppState>, day_boundaries: Vec<u64>, per_call: Option<bool>) -> Result<Vec<crate::request_store::HealthPoint>, String> {
    if per_call.unwrap_or(false) { state.requests.health_calls().await } else { state.requests.health(day_boundaries).await }
}

#[tauri::command]
pub async fn list_request_records(state: State<'_, AppState>, filter: crate::request_store::RecordFilter) -> Result<serde_json::Value, String> {
    let page = state.requests.list(filter).await?;
    // Header-only checks do not open or recover old transcript files.
    let sessions = state.runtime.list_sessions(&env()).await.map_err(|e|e.to_string())?;
    let mut names = HashMap::new();
    for session in state.runtime.loaded_sessions().await { names.insert(session.metadata().id.clone(), session.name().await); }
    let mut result = serde_json::to_value(page).map_err(|e|e.to_string())?;
    if let Some(records) = result["records"].as_array_mut() {
        for record in records {
            let id = record["context"]["sessionId"].as_str().unwrap_or_default();
            let session = sessions.iter().find(|s|s.id == id);
            record["sessionName"] = serde_json::json!(names.get(id).cloned().flatten());
            record["sessionDeleted"] = serde_json::json!(session.is_none());
        }
    }
    result["storageError"] = serde_json::json!(state.requests.error());
    Ok(result)
}

/// Session stats for the status bar.
#[tauri::command]
pub async fn get_stats(state: State<'_, AppState>, session_id: Option<String>) -> Result<serde_json::Value, String> {
    let session = command_session(&state, session_id.as_deref()).await?;
    stats_payload(&session, &state.requests).await
}

async fn stats_payload(session: &AgentSession, requests: &crate::request_store::RequestStore) -> Result<serde_json::Value, String> {
    let stats = session.stats();
    // Context occupancy belongs to the active transcript; cumulative usage comes from SQLite.
    let mut context_tokens = 0;
    for entry in session.entries().await.map_err(|error| error.to_string())? {
        if let step_harness_agent_core::session::Entry::Message(entry) = entry
            && let AgentMessage::Assistant(message) = entry.message
        {
            let reported = step_harness_agent_core::compaction::calculate_context_tokens(&message.usage);
            if reported > 0 {
                context_tokens = reported;
            }
        }
    }
    let (performance, performance_error) = match requests.performance(session.metadata().id.clone(), session.request_execution_id()).await {
        Ok(performance) => (performance, requests.error()),
        Err(error) => (crate::request_store::PerformanceSummary::default(), Some(error)),
    };
    Ok(serde_json::json!({ "contextTokens": context_tokens, "inputTokens": performance.session_usage.input, "outputTokens": performance.session_usage.output, "cacheReadTokens": performance.session_usage.cache_read, "cacheWriteTokens": performance.session_usage.cache_write, "cost": stats.cost, "turns": performance.session_rounds, "performance": performance, "performanceError": performance_error }))
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderView {
    id: String,
    name: String,
    base_url: String,
    api: String,
    models: Vec<step_harness_coding_agent::models::config::ModelEntry>,
    has_api_key: bool,
    configured: bool,
}

#[tauri::command]
pub async fn get_providers(state: State<'_, AppState>) -> Result<Vec<ProviderView>, String> {
    let services = state
        .runtime
        .services(&env())
        .await
        .map_err(|e| e.to_string())?;
    let config = step_harness_coding_agent::models::config::ModelsConfigFile::load(
        &services.paths.models_file(),
    )?;
    let stored = services
        .credentials
        .list(&tokio_util::sync::CancellationToken::new())
        .await
        .map_err(|e| e.to_string())?;
    Ok(config
        .providers
        .into_iter()
        .map(|(id, entry)| ProviderView {
            configured: services.catalog.is_provider_configured(&id),
            id: id.clone(),
            name: entry.name.unwrap_or_else(|| id.clone()),
            base_url: entry.base_url.unwrap_or_default(),
            api: entry.api.unwrap_or_else(|| "openai-completions".into()),
            models: entry.models.unwrap_or_default(),
            has_api_key: entry.api_key.is_some_and(|key| !key.is_empty())
                || stored.iter().any(|credential| credential.provider_id == id),
        })
        .collect())
}

async fn check_idle(state: &AppState, action: &str) -> Result<(), String> {
    if state.runtime.loaded_sessions().await.iter().any(|s| s.is_running()) {
        return Err(format!("请先停止正在运行的任务，再{action}。"));
    }
    Ok(())
}

async fn rebuild_catalog(state: &AppState) -> Result<(), String> {
    let services = state
        .runtime
        .services(&env())
        .await
        .map_err(|e| e.to_string())?;
    let (_, store) = step_harness_coding_agent::models::catalog::default_stores(&services.paths);
    let (next, diagnostics) =
        ModelCatalog::build(&services.paths, services.credentials.clone(), store).await;
    if !diagnostics.is_empty() {
        return Err(diagnostics.join("\n"));
    }
    let current = services.catalog.models();
    current.clear_providers();
    for provider in next.models().get_providers() {
        current.set_provider(provider);
    }
    services
        .catalog
        .refresh(true, &tokio_util::sync::CancellationToken::new())
        .await;
    if let Some(session) = state.runtime.active().await {
        let available = services.catalog.snapshot().available_models;
        if let Some(model) = session
            .current_model()
            .and_then(|m| {
                available
                    .iter()
                    .find(|n| n.provider == m.provider && n.id == m.id)
                    .cloned()
            })
            .or_else(|| available.first().cloned())
        {
            session
                .set_model(model.provider.as_str(), &model.id)
                .await
                .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

fn validate_model_entries(models: &[step_harness_coding_agent::models::config::ModelEntry]) -> Result<(), String> {
    let mut ids = std::collections::HashSet::new();
    for model in models {
        if model.id.trim().is_empty() || model.id != model.id.trim() || !ids.insert(&model.id) {
            return Err("模型 ID 不能为空、包含首尾空格或重复。".into());
        }
        for count in [model.context_window, model.max_tokens].into_iter().flatten() {
            if count == 0 || count > 9_007_199_254_740_991 {
                return Err("Token 数必须为正整数，且不能超过安全整数范围。".into());
            }
        }
        if model.max_tokens.zip(model.context_window).is_some_and(|(max, context)| max > context) {
            return Err("最大输出 Token 不能超过上下文窗口。".into());
        }
    }
    Ok(())
}

#[tauri::command]
pub async fn save_provider(
    state: State<'_, AppState>,
    id: String,
    name: String,
    base_url: String,
    api: String,
    models: Vec<step_harness_coding_agent::models::config::ModelEntry>,
    api_key: Option<String>,
) -> Result<(), String> {
    let _guard = state.configuration_lock.lock().await;
    check_idle(&state, "修改供应商配置").await?;
    if id == "step"
        || id.is_empty()
        || !id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err("供应商 ID 只能包含字母、数字、连字符或下划线，且不能使用 step。".into());
    }
    validate_api_url(&base_url)?;
    if ![
        "openai-completions",
        "openai-responses",
        "anthropic-messages",
    ]
    .contains(&api.as_str())
    {
        return Err("不支持的 API 协议。".into());
    }
    validate_model_entries(&models)?;
    use step_harness_coding_agent::models::config::ModelsConfigFile;
    let path = state.runtime.paths().models_file();
    let mut config = ModelsConfigFile::load(&path)?;
    let entry = config.providers.entry(id.clone()).or_default();
    entry.name = Some(name);
    entry.base_url = Some(base_url);
    entry.api = Some(api);
    entry.models = Some(models);
    if api_key.is_some() {
        entry.api_key = None;
    }
    config.save(&path)?;
    if let Some(key) = api_key {
        let services = state
            .runtime
            .services(&env())
            .await
            .map_err(|e| e.to_string())?;
        let token = tokio_util::sync::CancellationToken::new();
        if key.trim().is_empty() {
            services
                .credentials
                .delete(&id, &token)
                .await
                .map_err(|e| e.to_string())?;
        } else {
            let credential = step_harness_providers::auth::Credential::ApiKey(
                step_harness_providers::auth::ApiKeyCredential {
                    key: Some(key.trim().into()),
                    env: None,
                },
            );
            services
                .credentials
                .modify(
                    &id,
                    step_harness_providers::auth::modifier(move |_| {
                        let credential = credential.clone();
                        Box::pin(async move { Ok(Some(credential)) })
                    }),
                    &token,
                )
                .await
                .map_err(|e| e.to_string())?;
        }
    }
    rebuild_catalog(&state).await
}

#[tauri::command]
pub async fn delete_provider(state: State<'_, AppState>, id: String) -> Result<(), String> {
    let _guard = state.configuration_lock.lock().await;
    check_idle(&state, "删除供应商").await?;
    if id == "step" {
        return Err("Step 是内置供应商，请使用退出登录。".into());
    }
    let path = state.runtime.paths().models_file();
    let mut config = step_harness_coding_agent::models::config::ModelsConfigFile::load(&path)?;
    config.providers.remove(&id);
    config.save(&path)?;
    let services = state
        .runtime
        .services(&env())
        .await
        .map_err(|e| e.to_string())?;
    services
        .credentials
        .delete(&id, &tokio_util::sync::CancellationToken::new())
        .await
        .map_err(|e| e.to_string())?;
    rebuild_catalog(&state).await
}

/// Open an auth URL with the system's default browser.
///
/// `explorer.exe` must not be used for this on Windows: it parses the
/// argument as a filesystem path, so a URL either does nothing or opens a
/// File Explorer window instead of the browser.
fn open_auth_browser(url: &str) -> Result<(), String> {
    if !url.starts_with("https://") {
        return Err("登录地址必须使用 HTTPS。".into());
    }
    open::that(url).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_registry_reads_legacy_lists_and_keeps_removals_across_reload() {
        let legacy = parse_project_registry(br#"["C:/work/app"]"#).unwrap();
        assert_eq!(legacy.projects, vec!["C:/work/app"]);
        assert!(legacy.removed.is_empty());
        let registry = ProjectRegistry { projects: vec!["D:/other".into()], removed: vec!["C:/work/app".into()] };
        let saved = serde_json::to_vec(&registry).unwrap();
        let restored = parse_project_registry(&saved).unwrap();
        assert_eq!(restored.projects, registry.projects);
        assert_eq!(restored.removed, registry.removed);
        assert!(parse_project_registry(b"corrupted").is_err());
    }

    #[test]
    fn project_removal_path_matching_does_not_require_existing_folders() {
        assert_eq!(project_path_key(r"\\?\C:\work\app\"), project_path_key("C:/work/app"));
        if cfg!(windows) {
            assert_eq!(project_path_key("c:/WORK/app/"), project_path_key("C:/work/app"));
        }
        assert_ne!(project_path_key("C:/work/app"), project_path_key("D:/work/app"));
    }

    #[test]
    fn rejects_non_https_auth_urls() {
        // ShellExecute honors file:// URLs, so the scheme guard is the only
        // thing keeping a tampered auth URL from opening local files.
        assert!(open_auth_browser("http://insecure.example/authorize").is_err());
        assert!(open_auth_browser("file:///C:/Windows/System32/calc.exe").is_err());
        assert!(open_auth_browser("step-code://callback").is_err());
    }
}

struct BrowserLogin {
    app: AppHandle,
    token: tokio_util::sync::CancellationToken,
    url: Arc<std::sync::Mutex<Option<String>>>,
}
#[async_trait::async_trait]
impl step_harness_providers::auth::AuthInteraction for BrowserLogin {
    async fn prompt(
        &self,
        _: step_harness_providers::auth::AuthPrompt,
    ) -> Result<String, step_harness_providers::auth::AuthPromptCancelled> {
        Err(step_harness_providers::auth::AuthPromptCancelled)
    }
    fn signal(&self) -> tokio_util::sync::CancellationToken {
        self.token.clone()
    }
    fn notify(&self, event: step_harness_providers::auth::AuthEvent) {
        if let step_harness_providers::auth::AuthEvent::AuthUrl { url, .. } = &event {
            *self.url.lock().unwrap() = Some(url.clone());
            if let Err(error) = open_auth_browser(url) {
                let _ = self.app.emit("step-login-event", serde_json::json!({"type": "progress", "message": format!("浏览器打开失败，请点击重新打开：{error}")}));
            }
        }
        let _ = self.app.emit("step-login-event", event);
    }
}
#[tauri::command]
pub async fn login_step(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    let token = tokio_util::sync::CancellationToken::new();
    let url = Arc::new(std::sync::Mutex::new(None));
    {
        let mut login = state.login.lock().unwrap();
        if login.is_some() {
            return Err("登录正在进行。".into());
        }
        *login = Some((token.clone(), url.clone()));
    }
    let interaction = BrowserLogin {
        app,
        token: token.clone(),
        url,
    };
    let result = async {
        let services = state.runtime.services(&env()).await.map_err(|e| e.to_string())?;
        tokio::select! {
            _ = token.cancelled() => return Err("登录已取消。".into()),
            result = services.catalog.models().login("step", step_harness_providers::auth::AuthType::Oauth, &interaction) => { result.map_err(|e| e.to_string())?; }
        }
        services.catalog.refresh(true, &token).await;
        if let Some(session) = state.runtime.active().await
            && session.current_model().is_none()
            && let Some(model) = services.catalog.snapshot().available_models.first()
        {
            session.set_model(model.provider.as_str(), &model.id).await.map_err(|e| e.to_string())?;
        }
        Ok(())
    }.await;
    *state.login.lock().unwrap() = None;
    result
}
#[tauri::command]
pub fn cancel_step_login(state: State<'_, AppState>) {
    if let Some((token, _)) = state.login.lock().unwrap().as_ref() {
        token.cancel();
    }
}
#[tauri::command]
pub fn reopen_step_login(state: State<'_, AppState>) -> Result<(), String> {
    let login = state.login.lock().unwrap();
    let url = login
        .as_ref()
        .and_then(|(_, url)| url.lock().unwrap().clone())
        .ok_or("登录地址尚未就绪。")?;
    open_auth_browser(&url)
}
#[tauri::command]
pub async fn logout_step(state: State<'_, AppState>) -> Result<(), String> {
    let _guard = state.configuration_lock.lock().await;
    check_idle(&state, "退出 Step 登录").await?;
    if state.login.lock().unwrap().is_some() {
        return Err("请先取消当前登录。".into());
    }
    let services = state
        .runtime
        .services(&env())
        .await
        .map_err(|e| e.to_string())?;
    services
        .catalog
        .models()
        .logout("step", None)
        .await
        .map_err(|e| e.to_string())?;
    services
        .catalog
        .refresh_snapshot(&tokio_util::sync::CancellationToken::new())
        .await;
    Ok(())
}

fn session_event_payload(
    event: &step_harness_coding_agent::session::SessionEvent,
) -> serde_json::Value {
    let mut payload = serde_json::to_value(event).expect("session event is serializable");
    if let Some(object) = payload.as_object_mut() {
        for (rust, wire) in [
            ("execution_timings", "executionTimings"),
            ("max_attempts", "maxAttempts"),
            ("delay_ms", "delayMs"),
            ("error_message", "errorMessage"),
            ("final_error", "finalError"),
            ("follow_up", "followUp"),
            ("entry_id", "entryId"),
            ("model_id", "modelId"),
            ("thinking_level", "thinkingLevel"),
            ("plan_file_path", "planFilePath"),
            ("plan_source", "planSource"),
            ("plan_active", "planActive"),
        ] {
            if let Some(value) = object.remove(rust) {
                object.insert(wire.into(), value);
            }
        }
    }
    if let Some(object) = payload
        .get_mut("event")
        .and_then(serde_json::Value::as_object_mut)
    {
        for (rust, wire) in [
            ("tool_call_id", "toolCallId"),
            ("tool_name", "toolName"),
            ("is_error", "isError"),
            ("partial_result", "partialResult"),
            ("tool_results", "toolResults"),
            ("assistant_message_event", "assistantMessageEvent"),
        ] {
            if let Some(value) = object.remove(rust) {
                object.insert(wire.into(), value);
            }
        }
    }
    payload
}

fn validate_api_url(value: &str) -> Result<(), String> {
    let url = tauri::Url::parse(value).map_err(|_| "请输入有效的 API 地址。".to_owned())?;
    if !["http", "https"].contains(&url.scheme())
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
        || url.query().is_some()
    {
        return Err("API 地址必须是 HTTP 或 HTTPS 地址，不能包含账户信息、查询参数或片段。".into());
    }
    Ok(())
}

#[cfg(test)]
mod desktop_tests {
    use super::*;
    #[test]
    fn tool_event_keys_are_normalized_without_changing_tool_arguments() {
        let event = step_harness_coding_agent::session::SessionEvent::Agent {
            event: step_harness_agent_core::types::AgentEvent::ToolExecutionStart {
                tool_call_id: "call-1".into(),
                tool_name: "read".into(),
                args: serde_json::json!({ "file_path": "src/main.rs" }),
            },
        };
        let value = session_event_payload(&event);
        assert_eq!(value["event"]["toolCallId"], "call-1");
        assert_eq!(value["event"]["toolName"], "read");
        assert_eq!(value["event"]["args"]["file_path"], "src/main.rs");
    }
    #[tokio::test]
    async fn stored_provider_key_is_usable_after_catalog_rebuild() {
        use step_harness_coding_agent::models::config::{
            ModelEntry, ModelsConfigFile, ProviderEntry,
        };
        use step_harness_providers::auth::{ApiKeyCredential, Credential, modifier};
        let directory = tempfile::tempdir().unwrap();
        let paths = AppPaths::new(directory.path());
        let mut config = ModelsConfigFile::default();
        config.providers.insert(
            "desktop-test".into(),
            ProviderEntry {
                base_url: Some("https://example.test/v1".into()),
                models: Some(vec![ModelEntry {
                    id: "test-model".into(),
                    context_window: Some(32768),
                    ..Default::default()
                }]),
                ..Default::default()
            },
        );
        config.save(&paths.models_file()).unwrap();
        let (credentials, store) =
            step_harness_coding_agent::models::catalog::default_stores(&paths);
        let token = tokio_util::sync::CancellationToken::new();
        use step_harness_providers::auth::CredentialStore;
        credentials
            .modify(
                "desktop-test",
                modifier(|_| {
                    Box::pin(async {
                        Ok(Some(Credential::ApiKey(ApiKeyCredential {
                            key: Some("test-secret".into()),
                            env: None,
                        })))
                    })
                }),
                &token,
            )
            .await
            .unwrap();
        let (catalog, diagnostics) = ModelCatalog::build(&paths, credentials.clone(), store).await;
        assert!(diagnostics.is_empty());
        assert!(catalog.is_provider_configured("desktop-test"));
        assert!(
            catalog
                .snapshot()
                .available_models
                .iter()
                .any(|model| model.id == "test-model" && model.context_window == 32768)
        );
        assert!(
            !std::fs::read_to_string(paths.models_file())
                .unwrap()
                .contains("test-secret")
        );
        credentials.delete("desktop-test", &token).await.unwrap();
        catalog.refresh_snapshot(&token).await;
        assert!(!catalog.is_provider_configured("desktop-test"));
    }
    #[test]
    fn accepts_cloud_and_local_api_endpoints() {
        assert!(validate_api_url("https://api.example.com/v1").is_ok());
        assert!(validate_api_url("http://127.0.0.1:11434/v1").is_ok());
    }
    #[test]
    fn rejects_unsafe_or_malformed_endpoints() {
        for url in [
            "https://",
            "file:///tmp/key",
            "javascript:alert(1)",
            "https://user:password@example.com",
            "https://example.com#fragment",
            "https://example.com?api_key=secret",
        ] {
            assert!(validate_api_url(url).is_err(), "{url}");
        }
    }
    #[test]
    fn session_and_model_views_match_frontend_field_names() {
        let session = SessionSummary {
            id: "id".into(),
            path: "path".into(),
            cwd: "cwd".into(),
            created_at: 1,
            modified_at: 2,
            parent_session_id: None,
            name: Some("Renamed".into()),
            project: Some("step-code".into()),
        };
        let value = serde_json::to_value(session).unwrap();
        assert_eq!(value["createdAt"], 1);
        assert_eq!(value["modifiedAt"], 2);
        assert_eq!(value["name"], "Renamed");
        assert_eq!(value["project"], "step-code");
        assert!(value.get("created_at").is_none());
        let model = ModelInfo {
            provider: "step".into(),
            id: "model".into(),
            name: "Model".into(),
            reasoning: true,
            supported_thinking_levels: vec![
                ModelThinkingLevel::Low,
                ModelThinkingLevel::Medium,
                ModelThinkingLevel::High,
            ],
            context_window: 128000,
            configured: true,
            supports_images: true,
        };
        let value = serde_json::to_value(model).unwrap();
        assert_eq!(value["contextWindow"], 128000);
        assert_eq!(value["supportedThinkingLevels"], serde_json::json!(["low", "medium", "high"]));
    }
    #[test]
    fn model_forms_validate_and_persist_full_metadata() {
        use step_harness_coding_agent::models::config::{ModelEntry, ModelsConfigFile, ProviderEntry};
        let model: ModelEntry = serde_json::from_value(serde_json::json!({
            "id": "custom", "name": "Custom model", "contextWindow": 1500000,
            "maxTokens": 32000, "reasoning": true, "input": ["text", "image"],
            "thinkingLevelMap": { "off": null, "minimal": null, "low": "low", "medium": "medium", "high": "deep", "xhigh": null, "max": null },
            "headers": { "X-Custom": "preserved" }
        })).unwrap();
        assert!(validate_model_entries(&[]).is_ok());
        assert!(validate_model_entries(std::slice::from_ref(&model)).is_ok());
        assert!(validate_model_entries(&[model.clone(), model.clone()]).is_err());
        let mut invalid = model.clone();
        invalid.max_tokens = Some(1500001);
        assert!(validate_model_entries(&[invalid]).is_err());
        let mut invalid = model.clone();
        invalid.context_window = Some(0);
        assert!(validate_model_entries(&[invalid]).is_err());
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("models.json");
        let mut config = ModelsConfigFile::default();
        config.providers.insert("custom-provider".into(), ProviderEntry {
            models: Some(vec![model.clone()]), ..Default::default()
        });
        config.save(&path).unwrap();
        let loaded = ModelsConfigFile::load(&path).unwrap();
        assert_eq!(loaded.providers["custom-provider"].models.as_ref().unwrap()[0], model);
        let runtime = loaded.models_for("custom-provider");
        assert_eq!(runtime[0].context_window, 1500000);
        assert_eq!(runtime[0].max_tokens, 32000);
        assert_eq!(step_harness_providers::models::get_supported_thinking_levels(&runtime[0]), vec![ModelThinkingLevel::Low, ModelThinkingLevel::Medium, ModelThinkingLevel::High]);
    }
    #[test]
    fn forwards_model_and_retry_events_using_frontend_keys() {
        use step_harness_coding_agent::session::SessionEvent;
        let model = session_event_payload(&SessionEvent::ModelChanged {
            provider: "step".into(),
            model_id: "new-model".into(),
            thinking_level: "high".into(),
        });
        assert_eq!(model["modelId"], "new-model");
        assert_eq!(model["thinkingLevel"], "high");
        let retry = session_event_payload(&SessionEvent::AutoRetryStart {
            attempt: 1,
            max_attempts: 3,
            delay_ms: 100,
            error_message: "offline".into(),
        });
        assert_eq!(retry["maxAttempts"], 3);
        assert_eq!(retry["errorMessage"], "offline");
        let recalled = session_event_payload(&SessionEvent::MessagesRecalled {
            messages: vec![], execution_timings: vec![],
        });
        assert_eq!(recalled["type"], "messagesRecalled");
        assert_eq!(recalled["executionTimings"], serde_json::json!([]));
        assert!(recalled.get("execution_timings").is_none());
    }
}


#[cfg(test)]
mod image_input_tests {
    use super::*;

    #[test]
    fn validates_image_only_input_and_rejects_invalid_attachments() {
        let image = ImageContent {
            mime_type: "image/png".to_owned(),
            data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX1sAAAAASUVORK5CYII=".to_owned(),
        };
        assert!(validate_prompt_images(true, "", std::slice::from_ref(&image)).is_ok());
        assert!(validate_prompt_images(false, "", std::slice::from_ref(&image)).is_err());
        assert!(validate_prompt_images(true, " ", &[]).is_err());
        assert!(validate_prompt_images(false, "text", &[]).is_ok());
        assert!(validate_prompt_images(true, "", &vec![image.clone(); 9]).is_err());
        let mut invalid = image.clone();
        invalid.mime_type = "image/jpeg".to_owned();
        assert!(validate_prompt_images(true, "", &[invalid]).is_err());
        let mut invalid = image.clone();
        invalid.data = "not base64!".to_owned();
        assert!(validate_prompt_images(true, "", &[invalid]).is_err());
        let mut invalid = image;
        invalid.data = "A".repeat(13_981_020);
        assert!(validate_prompt_images(true, "", &[invalid]).is_err());
    }
}
