//! Plan mode: the research-then-propose flow.
//!
//! Ported functionally from `@step-harness/coding-agent`
//! `src/features/step-plan.ts` and `src/features/plan-mode-tools.ts`. The
//! TypeScript source builds the feature on the extension system; this port
//! keeps the same behavior with a plain state holder the session owns:
//!
//! - `enter_plan_mode` / `exit_plan_mode` are ordinary session tools that
//!   guard themselves on the current state.
//! - While plan mode is active, file-editing tools may only target the
//!   session's plan file; shell tools keep their normal permissions.
//! - `exit_plan_mode` submits the plan file for review through the
//!   embedder's [`PlanReviewChannel`]; approval re-enables mutations.
//! - State is persisted as a `step-plan` custom entry so reopening a session
//!   restores an in-progress plan.

use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use step_harness_agent_core::session::Entry;

/// Custom-entry type carrying plan-mode state in the session store.
pub const PLAN_ENTRY_TYPE: &str = "step-plan";
/// Environment variable overriding the plan directory.
pub const PLAN_DIR_ENV: &str = "STEP_CODING_AGENT_PLAN_DIR";
/// Default plan directory, relative to the working directory.
pub const DEFAULT_PLAN_DIR: &str = ".stepcode/plans";
/// The model-facing tool that enters plan mode.
pub const ENTER_PLAN_MODE_TOOL: &str = "enter_plan_mode";
/// The model-facing tool that submits the plan for review.
pub const EXIT_PLAN_MODE_TOOL: &str = "exit_plan_mode";
/// Tools that write files; plan mode limits them to the plan file.
pub const MUTATING_TOOLS: &[&str] = &["write", "edit"];

/// Who put the session into plan mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PlanSource {
    /// The user, through the `/plan` command.
    User,
    /// The model, through the `enter_plan_mode` tool.
    Agent,
}

impl PlanSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            PlanSource::User => "user",
            PlanSource::Agent => "agent",
        }
    }
}

/// The persisted plan-mode snapshot.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanState {
    pub enabled: bool,
    /// Tool set in force before plan mode, restored on exit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools_before_plan_mode: Option<Vec<String>>,
    /// Plan file path for this session, when one was announced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_file_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_source: Option<PlanSource>,
}

/// A plan submitted for review by `exit_plan_mode`.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanReviewRequest {
    /// Correlation id; the UI echoes it back with its decision.
    pub request_id: String,
    /// Session that owns this request, independent of the selected view.
    pub session_id: String,
    /// Time the request began waiting, in Unix milliseconds.
    pub created_at: u64,
    pub plan_file_path: String,
    pub plan_contents: String,
}

/// The user's decision on a submitted plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanReviewOutcome {
    /// Execute the plan.
    Approved,
    /// Stay in plan mode and apply the notes as a steering message.
    Feedback(String),
    /// Stay in plan mode without notes.
    Dismissed,
}

/// The embedder's plan review UI.
///
/// Implementations must resolve every request: the agent run blocks on the
/// returned outcome.
#[async_trait::async_trait]
pub trait PlanReviewChannel: Send + Sync {
    async fn review(&self, request: &PlanReviewRequest) -> PlanReviewOutcome;
}

/// Convenience alias for a shared review channel.
pub type SharedPlanReviewChannel = Arc<dyn PlanReviewChannel>;

/// Approves every plan without interaction.
///
/// The headless/RPC equivalent of an interactive approval: the caller must
/// gate execution externally.
pub struct AutoApprovePlanReview;

#[async_trait::async_trait]
impl PlanReviewChannel for AutoApprovePlanReview {
    async fn review(&self, _request: &PlanReviewRequest) -> PlanReviewOutcome {
        PlanReviewOutcome::Approved
    }
}

/// Dismisses every plan without interaction (tests).
pub struct AutoDismissPlanReview;

#[async_trait::async_trait]
impl PlanReviewChannel for AutoDismissPlanReview {
    async fn review(&self, _request: &PlanReviewRequest) -> PlanReviewOutcome {
        PlanReviewOutcome::Dismissed
    }
}

/// A channel backed by a closure (tests and simple embedders).
pub struct FnPlanReview<F>(pub F)
where
    F: Fn(&PlanReviewRequest) -> PlanReviewOutcome + Send + Sync;

#[async_trait::async_trait]
impl<F> PlanReviewChannel for FnPlanReview<F>
where
    F: Fn(&PlanReviewRequest) -> PlanReviewOutcome + Send + Sync,
{
    async fn review(&self, request: &PlanReviewRequest) -> PlanReviewOutcome {
        (self.0)(request)
    }
}

/// Whether a tool writes files.
pub fn is_mutating_tool(tool_name: &str) -> bool {
    MUTATING_TOOLS.contains(&tool_name)
}

/// The per-session plan file path.
///
/// `STEP_CODING_AGENT_PLAN_DIR` overrides the directory; a relative override
/// resolves against the working directory.
pub fn plan_file_path(session_id: &str, cwd: &str) -> PathBuf {
    let override_dir = std::env::var(PLAN_DIR_ENV)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty());
    plan_file_path_in(session_id, override_dir.as_deref(), cwd)
}

/// The per-session plan file path under an explicit directory override.
fn plan_file_path_in(session_id: &str, override_dir: Option<&str>, cwd: &str) -> PathBuf {
    let base = Path::new(cwd);
    let directory = match override_dir {
        Some(directory) => lexical_absolute(Path::new(directory), base),
        None => base.join(DEFAULT_PLAN_DIR),
    };
    directory.join(format!("session-{session_id}.md"))
}

/// Resolve `.`/`..` lexically against `base`, without touching the filesystem.
pub fn lexical_absolute(path: &Path, base: &Path) -> PathBuf {
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    };
    lexically_normalized(&joined)
}

/// Drop `.` segments and apply `..` segments, without touching the filesystem.
fn lexically_normalized(path: &Path) -> PathBuf {
    let mut resolved = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                resolved.pop();
            }
            other => resolved.push(other.as_os_str()),
        }
    }
    resolved
}

/// Whether two paths denote the same file, comparing normalized forms.
pub fn same_file_path(left: &Path, right: &Path) -> bool {
    let normalize = |path: &Path| {
        let text = lexically_normalized(path)
            .to_string_lossy()
            .replace('\\', "/");
        if cfg!(windows) {
            text.to_lowercase()
        } else {
            text
        }
    };
    normalize(left) == normalize(right)
}

/// Whether `target` (a raw tool argument) addresses `plan_path`.
pub fn targets_plan_file(target: &str, plan_path: &Path, cwd: &str) -> bool {
    let base = Path::new(cwd);
    same_file_path(
        &lexical_absolute(Path::new(target), base),
        &lexical_absolute(plan_path, base),
    )
}

/// The tool set plan mode runs with: the active set without file-editing
/// tools, re-added for the plan file, plus the plan-mode tools themselves.
pub fn plan_mode_tool_names(active: &[String]) -> Vec<String> {
    let mut names: Vec<String> = active
        .iter()
        .filter(|name| !is_mutating_tool(name))
        .cloned()
        .collect();
    for required in [MUTATING_TOOLS, &[ENTER_PLAN_MODE_TOOL, EXIT_PLAN_MODE_TOOL]].concat() {
        if !names.iter().any(|name| name == required) {
            names.push(required.to_owned());
        }
    }
    names
}

/// Keep only the tools whose names are in `allowed`.
pub fn filter_tools_by_name<T>(
    tools: Vec<T>,
    allowed: &[String],
    name: impl Fn(&T) -> &str,
) -> Vec<T> {
    tools
        .into_iter()
        .filter(|tool| allowed.iter().any(|candidate| candidate == name(tool)))
        .collect()
}

/// The plan-mode snapshot recorded in the session store, newest entry first.
pub fn plan_state_from_entries(entries: &[Entry]) -> Option<PlanState> {
    entries.iter().rev().find_map(|entry| match entry {
        Entry::Custom(custom) if custom.custom_type == PLAN_ENTRY_TYPE => custom
            .data
            .as_ref()
            .and_then(|data| serde_json::from_value::<PlanState>(data.clone()).ok()),
        _ => None,
    })
}

/// Per-session plan-mode state.
#[derive(Debug, Default)]
pub struct PlanMode {
    state: std::sync::RwLock<PlanState>,
}

impl PlanMode {
    pub fn new() -> Self {
        PlanMode::default()
    }

    /// Adopt the state recorded in the session store, returning the snapshot.
    pub fn restore(&self, entries: &[Entry], session_id: &str, cwd: &str) -> PlanState {
        let mut restored = plan_state_from_entries(entries).unwrap_or_default();
        if restored.enabled && restored.plan_file_path.is_none() {
            restored.plan_file_path = Some(
                plan_file_path(session_id, cwd)
                    .to_string_lossy()
                    .to_string(),
            );
        }
        *self
            .state
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = restored.clone();
        restored
    }

    /// The current snapshot.
    pub fn snapshot(&self) -> PlanState {
        self.state
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub fn is_active(&self) -> bool {
        self.snapshot().enabled
    }

    /// The plan file path: the announced one, or the per-session default.
    pub fn plan_file_path(&self, session_id: &str, cwd: &str) -> PathBuf {
        let snapshot = self.snapshot();
        match snapshot.plan_file_path {
            Some(path) if snapshot.enabled => PathBuf::from(path),
            _ => plan_file_path(session_id, cwd),
        }
    }

    /// Enter plan mode; `active` is the tool set currently in force.
    pub fn enter(
        &self,
        source: PlanSource,
        session_id: &str,
        cwd: &str,
        active: &[String],
    ) -> PlanState {
        let mut state = self
            .state
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let snapshot = PlanState {
            enabled: true,
            tools_before_plan_mode: Some(
                state
                    .tools_before_plan_mode
                    .clone()
                    .unwrap_or_else(|| active.to_vec()),
            ),
            // A previously announced path is reused, matching the source.
            plan_file_path: Some(state.plan_file_path.clone().unwrap_or_else(|| {
                plan_file_path(session_id, cwd)
                    .to_string_lossy()
                    .to_string()
            })),
            plan_source: Some(source),
        };
        *state = snapshot.clone();
        snapshot
    }

    /// Leave plan mode; the tool set to restore travels in the snapshot.
    pub fn exit(&self) -> PlanState {
        let mut state = self
            .state
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let snapshot = PlanState {
            enabled: false,
            plan_source: None,
            ..state.clone()
        };
        *state = snapshot.clone();
        snapshot
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use step_harness_agent_core::session::CustomEntry;

    fn custom_plan_entry(data: serde_json::Value) -> Entry {
        Entry::Custom(CustomEntry {
            id: "p1".to_owned(),
            seq: 1,
            parent_id: None,
            timestamp: 1,
            custom_type: PLAN_ENTRY_TYPE.to_owned(),
            data: Some(data),
        })
    }

    #[test]
    fn derives_the_per_session_plan_path() {
        let cwd = if cfg!(windows) { "E:\\repo" } else { "/repo" };
        let path = plan_file_path_in("session-1", None, cwd);
        assert_eq!(
            path,
            Path::new(cwd)
                .join(DEFAULT_PLAN_DIR)
                .join("session-session-1.md")
        );
    }

    #[test]
    fn a_relative_directory_override_resolves_against_the_cwd() {
        let path = plan_file_path_in("s1", Some("custom-plans"), "/repo");
        assert_eq!(path, Path::new("/repo/custom-plans/session-s1.md"));
    }

    #[test]
    fn an_absolute_directory_override_is_kept() {
        let absolute = if cfg!(windows) {
            "C:\\plans"
        } else {
            "/tmp/plans"
        };
        let path = plan_file_path_in("s1", Some(absolute), "/repo");
        assert_eq!(path, Path::new(absolute).join("session-s1.md"));
    }

    #[test]
    fn reads_the_newest_plan_state_from_entries() {
        let entries = vec![
            custom_plan_entry(json!({"enabled": true, "planFilePath": "/repo/a.md"})),
            custom_plan_entry(json!({"enabled": false})),
        ];
        let state = plan_state_from_entries(&entries).expect("plan state");
        assert!(!state.enabled);
    }

    #[test]
    fn ignores_foreign_custom_entries() {
        let entry = Entry::Custom(CustomEntry {
            id: "x".to_owned(),
            seq: 1,
            parent_id: None,
            timestamp: 1,
            custom_type: "other".to_owned(),
            data: Some(json!({"enabled": true})),
        });
        assert!(plan_state_from_entries(&[entry]).is_none());
    }

    #[test]
    fn plan_mode_tool_set_drops_and_re_adds_mutations() {
        let active: Vec<String> = ["read", "bash", "write", "edit", "ls"]
            .iter()
            .map(|name| (*name).to_owned())
            .collect();
        let names = plan_mode_tool_names(&active);
        assert!(names.contains(&"write".to_owned()));
        assert!(names.contains(&"edit".to_owned()));
        assert!(names.contains(&ENTER_PLAN_MODE_TOOL.to_owned()));
        assert!(names.contains(&EXIT_PLAN_MODE_TOOL.to_owned()));
        assert!(names.contains(&"read".to_owned()));
        // The baseline order is kept and the plan tools are appended once.
        assert_eq!(names.len(), active.len() + 2);
        assert_eq!(names[0], "read");
    }

    #[test]
    fn enter_records_the_baseline_and_path() {
        let plan = PlanMode::new();
        let snapshot = plan.enter(PlanSource::Agent, "s1", "/repo", &["read".to_owned()]);
        assert!(snapshot.enabled);
        assert_eq!(snapshot.plan_source, Some(PlanSource::Agent));
        assert_eq!(
            snapshot.tools_before_plan_mode,
            Some(vec!["read".to_owned()])
        );
        assert!(snapshot.plan_file_path.is_some());
        assert!(plan.is_active());
        // A second enter keeps the original baseline and path.
        let again = plan.enter(
            PlanSource::User,
            "s1",
            "/repo",
            &["read".to_owned(), "bash".to_owned()],
        );
        assert_eq!(again.tools_before_plan_mode, Some(vec!["read".to_owned()]));
        assert_eq!(again.plan_file_path, snapshot.plan_file_path);
    }

    #[test]
    fn exit_clears_the_flag_but_keeps_the_restore_list() {
        let plan = PlanMode::new();
        plan.enter(PlanSource::User, "s1", "/repo", &["read".to_owned()]);
        let snapshot = plan.exit();
        assert!(!snapshot.enabled);
        assert!(snapshot.plan_source.is_none());
        assert_eq!(
            snapshot.tools_before_plan_mode,
            Some(vec!["read".to_owned()])
        );
        assert!(!plan.is_active());
    }

    #[test]
    fn restore_adopts_an_enabled_snapshot() {
        let plan = PlanMode::new();
        let entries = vec![custom_plan_entry(json!({
            "enabled": true,
            "toolsBeforePlanMode": ["read", "bash"],
            "planFilePath": "/repo/.stepcode/plans/session-s1.md",
            "planSource": "agent"
        }))];
        let restored = plan.restore(&entries, "s1", "/repo");
        assert!(restored.enabled);
        assert_eq!(restored.plan_source, Some(PlanSource::Agent));
        assert!(plan.is_active());
        assert_eq!(
            plan.plan_file_path("s1", "/repo"),
            Path::new("/repo/.stepcode/plans/session-s1.md")
        );
    }

    #[test]
    fn restore_derives_a_missing_plan_path() {
        let plan = PlanMode::new();
        let restored = plan.restore(
            &[custom_plan_entry(json!({"enabled": true}))],
            "s1",
            "/repo",
        );
        assert!(restored.enabled);
        assert_eq!(
            restored.plan_file_path.as_deref().map(PathBuf::from),
            Some(
                Path::new("/repo")
                    .join(DEFAULT_PLAN_DIR)
                    .join("session-s1.md")
            )
        );
    }

    #[test]
    fn restore_of_an_inactive_snapshot_keeps_plan_mode_off() {
        let plan = PlanMode::new();
        let restored = plan.restore(
            &[custom_plan_entry(json!({"enabled": false}))],
            "s1",
            "/repo",
        );
        assert!(!restored.enabled);
        assert!(!plan.is_active());
    }

    #[test]
    fn restore_without_entries_stays_off() {
        let plan = PlanMode::new();
        let restored = plan.restore(&[], "s1", "/repo");
        assert!(!restored.enabled);
        assert!(restored.plan_file_path.is_none());
    }

    #[test]
    fn same_file_path_normalizes_separators_and_dots() {
        assert!(same_file_path(
            Path::new("/repo/./plans/a.md"),
            Path::new("/repo/plans/../plans/a.md")
        ));
        assert!(!same_file_path(
            Path::new("/repo/a.md"),
            Path::new("/repo/b.md")
        ));
    }

    #[test]
    fn targets_plan_file_resolves_relative_paths() {
        let plan = Path::new("/repo/.stepcode/plans/session-s1.md");
        assert!(targets_plan_file(
            ".stepcode/plans/session-s1.md",
            plan,
            "/repo"
        ));
        assert!(targets_plan_file(
            "/repo/.stepcode/plans/session-s1.md",
            plan,
            "/other"
        ));
        assert!(!targets_plan_file("src/main.rs", plan, "/repo"));
    }

    #[test]
    fn filter_keeps_allowed_tools_in_order() {
        let tools = vec!["read", "bash", "write"];
        let allowed = vec!["read".to_owned(), "write".to_owned()];
        let kept = filter_tools_by_name(tools.clone(), &allowed, |tool| *tool);
        assert_eq!(kept, vec!["read", "write"]);
    }
}
