//! Multi-session runtime: creating, opening, listing, and switching sessions.
//!
//! Ported functionally from `@step-harness/coding-agent`
//! `src/core/agent-session-runtime.ts`. The TypeScript runtime serializes
//! session selection through a FIFO tail. This runtime serializes selection
//! with a mutex and retains loaded sessions so their turns, approval waits
//! and storage leases survive switching to another session.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use step_harness_agent_core::harness::types::ExecutionEnv;
use step_harness_agent_core::session::jsonl::{
    JsonlSessionMetadata, JsonlSessionRepo, JsonlSessionRepoOptions,
};
use tokio::sync::Mutex;

use crate::paths::AppPaths;
use crate::plan::SharedPlanReviewChannel;
use crate::session::{AgentSession, SessionResult, SessionServices, SharedApprovalChannel};

/// The session a run left active, remembered for the next start.
#[derive(Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LastActiveSession {
    #[serde(default)]
    session_id: String,
}

/// Owns loaded sessions and selects the session currently shown by the UI.
pub struct SessionRuntime {
    request_observer: Option<Arc<dyn step_harness_providers::request_metrics::RequestObserver>>,
    cwd: String,
    paths: AppPaths,
    approval: SharedApprovalChannel,
    plan_review: SharedPlanReviewChannel,
    services: tokio::sync::RwLock<Option<Arc<SessionServices>>>,
    active: tokio::sync::RwLock<Option<Arc<AgentSession>>>,
    sessions: tokio::sync::RwLock<HashMap<String, Arc<AgentSession>>>,
    /// Serializes session replacement (new/open/switch/fork).
    switch_lock: Mutex<()>,
}

impl SessionRuntime {
    /// Create a runtime for a working directory. The services are built
    /// lazily on first use so construction cannot fail.
    pub fn new(
        cwd: impl Into<String>,
        paths: AppPaths,
        approval: SharedApprovalChannel,
        plan_review: SharedPlanReviewChannel,
    ) -> Self {
        SessionRuntime {
            request_observer: None,
            cwd: cwd.into(),
            paths,
            approval,
            plan_review,
            services: tokio::sync::RwLock::new(None),
            active: tokio::sync::RwLock::new(None),
            sessions: tokio::sync::RwLock::new(HashMap::new()),
            switch_lock: Mutex::new(()),
        }
    }

    /// The working directory this runtime is bound to.
    pub fn cwd(&self) -> &str {
        &self.cwd
    }
    pub fn with_request_observer(mut self, observer: Arc<dyn step_harness_providers::request_metrics::RequestObserver>) -> Self { self.request_observer = Some(observer); self }

    /// The data directory paths in use.
    pub fn paths(&self) -> &AppPaths {
        &self.paths
    }

    /// Services for the active project's directory, building them on first use.
    pub async fn services(
        &self,
        env: &Arc<dyn ExecutionEnv>,
    ) -> SessionResult<Arc<SessionServices>> {
        let cwd = self
            .active()
            .await
            .map(|s| s.metadata().cwd.clone())
            .unwrap_or_else(|| self.cwd.clone());
        self.services_for_cwd(env, &cwd).await
    }

    pub async fn services_for_cwd(
        &self,
        env: &Arc<dyn ExecutionEnv>,
        cwd: &str,
    ) -> SessionResult<Arc<SessionServices>> {
        if let Some(services) = self.services.read().await.clone()
            && services.cwd == cwd
        {
            return Ok(services);
        }
        let mut slot = self.services.write().await;
        if let Some(services) = slot.clone()
            && services.cwd == cwd
        {
            return Ok(services);
        }
        // A fresh working directory is trusted by default: the desktop app has
        // no CLI trust prompt, and project config only loads from an explicit
        // `.step-rs` directory the user created.
        let (mut services, _diagnostics) =
            SessionServices::create(cwd, self.paths.clone(), env, true).await;
        services.request_observer = self.request_observer.clone();
        let services = Arc::new(services);
        *slot = Some(services.clone());
        Ok(services)
    }

    /// The active session, if any.
    pub async fn active(&self) -> Option<Arc<AgentSession>> {
        self.active.read().await.clone()
    }

    /// A loaded session, including sessions running in the background.
    pub async fn session(&self, id: &str) -> Option<Arc<AgentSession>> {
        self.sessions.read().await.get(id).cloned()
    }

    /// All loaded sessions, for application-wide configuration checks.
    pub async fn loaded_sessions(&self) -> Vec<Arc<AgentSession>> {
        self.sessions.read().await.values().cloned().collect()
    }

    /// Create a new session and make it active.
    pub async fn new_session(
        &self,
        env: &Arc<dyn ExecutionEnv>,
    ) -> SessionResult<Arc<AgentSession>> {
        let cwd = self
            .active()
            .await
            .map(|s| s.metadata().cwd.clone())
            .unwrap_or_else(|| self.cwd.clone());
        self.new_session_in(env, &cwd).await
    }

    /// Create a session using the selected project's settings, resources and tools.
    pub async fn new_session_in(
        &self,
        env: &Arc<dyn ExecutionEnv>,
        cwd: &str,
    ) -> SessionResult<Arc<AgentSession>> {
        let _guard = self.switch_lock.lock().await;
        let services = self.services_for_cwd(env, cwd).await?;
        let session = Arc::new(
            AgentSession::create(services, self.approval.clone(), self.plan_review.clone()).await?,
        );
        self.sessions
            .write()
            .await
            .insert(session.metadata().id.clone(), session.clone());
        *self.active.write().await = Some(session.clone());
        self.remember_active_session(Some(&session.metadata().id));
        Ok(session)
    }

    /// Open an existing session and make it active.
    pub async fn open_session(
        &self,
        env: &Arc<dyn ExecutionEnv>,
        metadata: JsonlSessionMetadata,
    ) -> SessionResult<Arc<AgentSession>> {
        let _guard = self.switch_lock.lock().await;
        if let Some(session) = self.session(&metadata.id).await {
            *self.active.write().await = Some(session.clone());
            self.remember_active_session(Some(&metadata.id));
            return Ok(session);
        }
        let services = self.services_for_cwd(env, &metadata.cwd).await?;
        let session = Arc::new(
            AgentSession::open(
                services,
                metadata,
                self.approval.clone(),
                self.plan_review.clone(),
            )
            .await?,
        );
        self.sessions
            .write()
            .await
            .insert(session.metadata().id.clone(), session.clone());
        *self.active.write().await = Some(session.clone());
        self.remember_active_session(Some(&session.metadata().id));
        Ok(session)
    }

    /// Reopen the session the previous run left active, when it still exists.
    ///
    /// Best effort: a session that was deleted, or one whose file another run
    /// still holds, leaves the runtime without an active session exactly as a
    /// first run would, so a failed restore never blocks startup.
    pub async fn restore_last_active(
        &self,
        env: &Arc<dyn ExecutionEnv>,
    ) -> Option<Arc<AgentSession>> {
        let session_id = self.last_active_session_id()?;
        if let Some(session) = self.session(&session_id).await {
            *self.active.write().await = Some(session.clone());
            return Some(session);
        }
        let Some(metadata) = self
            .list_sessions(env)
            .await
            .ok()?
            .into_iter()
            .find(|metadata| metadata.id == session_id)
        else {
            // The session is gone; forget it instead of rescanning every start.
            self.remember_active_session(None);
            return None;
        };
        match self.open_session(env, metadata).await {
            Ok(session) => Some(session),
            Err(error) => {
                // The file still exists, so keep the id: a lease held by
                // another run is temporary, a deleted session is not.
                log::warn!("last active session {session_id} could not be reopened: {error}");
                None
            }
        }
    }

    /// The session the previous run left active, when it recorded one.
    fn last_active_session_id(&self) -> Option<String> {
        let saved: LastActiveSession =
            serde_json::from_slice(&std::fs::read(self.last_active_file()).ok()?).ok()?;
        (!saved.session_id.is_empty()).then_some(saved.session_id)
    }

    /// File remembering the session to reopen on the next start.
    fn last_active_file(&self) -> PathBuf {
        self.paths.root().join("last-session.json")
    }

    /// Remember the active session so the next start reopens it.
    ///
    /// A failed write only costs the restore on the next start, so it must not
    /// fail the caller's request.
    fn remember_active_session(&self, session_id: Option<&str>) {
        let root = self.paths.root();
        let path = self.last_active_file();
        let result = match session_id {
            Some(session_id) => serde_json::to_vec_pretty(&LastActiveSession {
                session_id: session_id.to_owned(),
            })
            .map_err(|error| error.to_string())
            .and_then(|data| {
                std::fs::create_dir_all(root).map_err(|error| error.to_string())?;
                let staging = root.join("last-session.json.tmp");
                std::fs::write(&staging, data).map_err(|error| error.to_string())?;
                std::fs::rename(&staging, &path).map_err(|error| error.to_string())
            }),
            None => match std::fs::remove_file(&path) {
                Ok(()) => Ok(()),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
                Err(error) => Err(error.to_string()),
            },
        };
        if let Err(error) = result {
            log::warn!("failed to remember the active session: {error}");
        }
    }

    /// List sessions across all project directories, newest first.
    pub async fn list_sessions(
        &self,
        env: &Arc<dyn ExecutionEnv>,
    ) -> SessionResult<Vec<JsonlSessionMetadata>> {
        let mut sessions = self
            .repo(env)
            .list(&step_harness_agent_core::session::jsonl::JsonlSessionListOptions { cwd: None })
            .await?;
        sessions.sort_by_key(|metadata| std::cmp::Reverse(metadata.modified_at));
        Ok(sessions)
    }

    /// List sessions across all project directories with their display names
    /// and project assignments, newest first.
    ///
    /// Loaded sessions own exclusive leases on their files, so their
    /// names and projects are read from live sessions: reopening those files
    /// would collide with the lease. A session locked by another process is
    /// still listed, just unnamed and unassigned until it can be opened.
    pub async fn list_sessions_with_names(
        &self,
        env: &Arc<dyn ExecutionEnv>,
    ) -> SessionResult<Vec<(JsonlSessionMetadata, Option<String>, Option<String>)>> {
        let loaded = self.sessions.read().await.clone();
        let repo = self.repo(env);
        let mut named = Vec::new();
        for item in self.list_sessions(env).await? {
            let (name, project) = match loaded.get(&item.id) {
                Some(session) => (session.name().await, session.project().await),
                _ => match repo.open(&item).await {
                    Ok(session) => (session.get_name().await, session.get_project().await),
                    Err(_) => (None, None),
                },
            };
            named.push((item, name, project));
        }
        Ok(named)
    }

    /// A repository bound to this runtime's sessions directory.
    fn repo(&self, env: &Arc<dyn ExecutionEnv>) -> JsonlSessionRepo {
        JsonlSessionRepo::new(
            env.clone(),
            JsonlSessionRepoOptions {
                sessions_root: self.paths.sessions_dir().to_string_lossy().to_string(),
            },
        )
    }

    /// Delete a session file. Deleting the active session detaches it.
    pub async fn delete_session(
        &self,
        env: &Arc<dyn ExecutionEnv>,
        metadata: &JsonlSessionMetadata,
    ) -> SessionResult<()> {
        let _guard = self.switch_lock.lock().await;
        if self
            .session(&metadata.id)
            .await
            .is_some_and(|session| session.is_running())
        {
            return Err(crate::session::SessionFailure::Other(
                "请先停止该会话的任务，再删除会话。".into(),
            ));
        }
        // Release the loaded session's lease before deleting its file.
        let is_active = self
            .active
            .read()
            .await
            .as_ref()
            .is_some_and(|session| session.metadata().path == metadata.path);
        if is_active {
            *self.active.write().await = None;
            self.remember_active_session(None);
        }
        self.sessions.write().await.remove(&metadata.id);
        self.repo(env).delete(metadata).await?;
        Ok(())
    }

    /// Reload settings and resources for the active session.
    pub async fn reload(&self, env: &Arc<dyn ExecutionEnv>) -> SessionResult<()> {
        if let Some(session) = self.active().await {
            session.reload().await;
        } else {
            let services = self.services(env).await?;
            services.reload(env).await;
        }
        Ok(())
    }
}
