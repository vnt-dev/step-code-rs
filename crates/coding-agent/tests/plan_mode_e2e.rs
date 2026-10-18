//! End-to-end tests for plan mode: the `/plan` entry, the plan-file write
//! gate, the `exit_plan_mode` review, and restore across a reopen.

use std::path::Path;
mod common;
use std::sync::Arc;

use serde_json::json;
use step_harness_agent_core::harness::env::NodeExecutionEnv;
use step_harness_agent_core::harness::types::ExecutionEnv;
use step_harness_coding_agent::plan::{AutoApprovePlanReview, PlanReviewChannel, PlanSource};
use step_harness_coding_agent::session::{
    ApprovalChannel, ApprovalRequest, FnApproval, SessionEvent,
};
use step_harness_coding_agent::{AgentSession, AppPaths, SessionServices};
use step_harness_providers::compat::register_faux_provider;
use step_harness_providers::providers::faux::{
    FauxAssistantMessageOptions, FauxContentBlock, FauxModelDefinition, FauxResponseStep,
    RegisterFauxProviderOptions, faux_assistant_message,
};
use step_harness_providers::types::ToolCall;

/// Register a scripted faux provider under a per-test api id.
fn faux_registration(name: &str) -> step_harness_providers::compat::FauxProviderRegistration {
    register_faux_provider(&RegisterFauxProviderOptions {
        api: Some(format!("faux-plan-{name}")),
        provider: Some(format!("faux-{name}")),
        models: Some(vec![FauxModelDefinition {
            id: "faux-1".to_owned(),
            name: Some("Faux One".to_owned()),
            reasoning: Some(false),
            input: None,
            cost: None,
            context_window: Some(128_000),
            max_tokens: Some(4_096),
        }]),
        ..Default::default()
    })
}

/// A data directory isolated per test.
fn isolated_paths(name: &str) -> (AppPaths, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let paths = AppPaths::new(dir.path().join(name));
    paths.ensure_dirs().unwrap();
    (paths, dir)
}

/// Write a models.json that points at the registered faux api.
fn write_faux_models(paths: &AppPaths, api: &str, provider: &str) {
    let config = json!({
        "providers": {
            provider: {
                "name": "Faux",
                "baseUrl": "https://faux.invalid",
                "apiKey": "test-key",
                "api": api,
                "models": [{
                    "id": "faux-1",
                    "name": "Faux One",
                    "contextWindow": 128000,
                    "maxTokens": 4096
                }]
            }
        }
    });
    std::fs::write(
        paths.models_file(),
        serde_json::to_string_pretty(&config).unwrap(),
    )
    .unwrap();
}

/// Session services rooted at a temporary working directory, so the plan file
/// and any stray writes land inside it.
async fn services_in(
    name: &str,
    cwd: &Path,
) -> (
    Arc<dyn ExecutionEnv>,
    SessionServices,
    step_harness_providers::compat::FauxProviderRegistration,
) {
    let registration = faux_registration(name);
    let (paths, _paths_dir) = isolated_paths(name);
    std::mem::forget(_paths_dir);
    write_faux_models(&paths, registration.api(), registration.provider_id());
    let env: Arc<dyn ExecutionEnv> = Arc::new(NodeExecutionEnv::new(Default::default()));
    let (mut services, diagnostics) =
        SessionServices::create(cwd.to_string_lossy().to_string(), paths, &env, true).await;
    services.name_completer = Some(Arc::new(common::SkipNameCompleter));
    assert!(
        diagnostics.is_empty(),
        "unexpected diagnostics: {diagnostics:?}"
    );
    (env, services, registration)
}

fn approve_all() -> Arc<dyn ApprovalChannel> {
    Arc::new(FnApproval(|_: &ApprovalRequest| true))
}

fn approve_all_plans() -> Arc<dyn PlanReviewChannel> {
    Arc::new(AutoApprovePlanReview)
}

/// A scripted tool-call turn.
fn tool_call(id: &str, name: &str, arguments: serde_json::Value) -> FauxResponseStep {
    faux_assistant_message(
        vec![FauxContentBlock::ToolCall(ToolCall {
            id: id.to_owned(),
            name: name.to_owned(),
            arguments: serde_json::from_value(arguments).unwrap(),
            namespace: None,
            thought_signature: None,
        })],
        FauxAssistantMessageOptions::default(),
    )
    .into()
}

/// A scripted plain-text turn.
fn text(message: &str) -> FauxResponseStep {
    faux_assistant_message(message, FauxAssistantMessageOptions::default()).into()
}

/// Every tool-result text in the transcript.
fn tool_result_texts(session: &AgentSession) -> Vec<String> {
    session
        .state()
        .messages
        .iter()
        .filter_map(|message| message.as_tool_result())
        .flat_map(|result| {
            result
                .content
                .iter()
                .filter_map(|block| block.text())
                .map(|text| text.text.clone())
                .collect::<Vec<_>>()
        })
        .collect()
}

fn tool_names(session: &AgentSession) -> Vec<String> {
    session
        .state()
        .tools
        .iter()
        .map(|tool| tool.name.clone())
        .collect()
}

#[tokio::test]
async fn the_plan_command_enters_and_leaves_plan_mode() {
    let cwd = tempfile::tempdir().unwrap();
    let (_env, services, registration) = services_in("toggle", cwd.path()).await;
    registration.set_responses(vec![text("nothing to do")]);

    let session = AgentSession::create(Arc::new(services), approve_all(), approve_all_plans())
        .await
        .unwrap();
    let mut events = session.subscribe();

    let snapshot = session.enter_plan_mode(PlanSource::User).await;
    assert!(snapshot.enabled);
    assert_eq!(snapshot.plan_source, Some(PlanSource::User));

    // The plan file lives under the working directory, per session.
    let plan_path = session.plan_file_path();
    assert!(plan_path.starts_with(cwd.path()), "{plan_path:?}");
    assert_eq!(
        plan_path.file_name().and_then(|name| name.to_str()),
        Some(format!("session-{}.md", session.metadata().id).as_str())
    );

    // File-editing tools stay available (gated to the plan file) alongside
    // the plan-mode tools.
    let names = tool_names(&session);
    assert!(names.contains(&"write".to_owned()));
    assert!(names.contains(&"enter_plan_mode".to_owned()));
    assert!(names.contains(&"exit_plan_mode".to_owned()));

    let mut saw_enter = false;
    while let Ok(event) = events.try_recv() {
        if let SessionEvent::PlanModeChanged { active: true, .. } = event {
            saw_enter = true;
        }
    }
    assert!(saw_enter, "expected a plan-mode-changed event");

    let off = session.exit_plan_mode().await;
    assert!(!off.enabled);
    assert!(off.plan_source.is_none());
    assert!(!session.is_plan_mode_active());
    // The baseline tool set is back.
    assert_eq!(
        tool_names(&session).len(),
        off.tools_before_plan_mode.unwrap().len()
    );
}

#[tokio::test]
async fn plan_mode_blocks_writes_outside_the_plan_file() {
    let cwd = tempfile::tempdir().unwrap();
    let (_env, services, registration) = services_in("gate", cwd.path()).await;
    registration.set_responses(vec![
        tool_call(
            "call-write",
            "write",
            json!({"path": "outside.txt", "content": "x"}),
        ),
        text("blocked"),
    ]);

    let session = AgentSession::create(Arc::new(services), approve_all(), approve_all_plans())
        .await
        .unwrap();
    session.enter_plan_mode(PlanSource::User).await;
    session.prompt("change a file", None).await.unwrap();

    let results = tool_result_texts(&session);
    assert!(
        results
            .iter()
            .any(|text| text.contains("Plan mode blocked")),
        "expected the write to be blocked, got {results:?}"
    );
    assert!(
        !cwd.path().join("outside.txt").exists(),
        "no file may be written outside the plan file"
    );
    assert!(
        session.is_plan_mode_active(),
        "the block keeps plan mode on"
    );
}

#[tokio::test]
async fn plan_mode_allows_writing_the_plan_file() {
    let cwd = tempfile::tempdir().unwrap();
    let (_env, services, registration) = services_in("write-plan", cwd.path()).await;

    let session = AgentSession::create(Arc::new(services), approve_all(), approve_all_plans())
        .await
        .unwrap();
    session.enter_plan_mode(PlanSource::User).await;
    let plan_path = session.plan_file_path();

    registration.set_responses(vec![
        tool_call(
            "call-write",
            "write",
            json!({"path": plan_path.to_string_lossy(), "content": "# Plan\nstep 1"}),
        ),
        text("plan written"),
    ]);
    session.prompt("draft the plan", None).await.unwrap();

    let contents = std::fs::read_to_string(&plan_path).expect("the plan file is written");
    assert!(contents.contains("step 1"));
}

#[tokio::test]
async fn exit_plan_mode_reviews_the_plan_and_reenables_mutations() {
    let cwd = tempfile::tempdir().unwrap();
    let (_env, services, registration) = services_in("review", cwd.path()).await;

    let session = AgentSession::create(Arc::new(services), approve_all(), approve_all_plans())
        .await
        .unwrap();
    session.enter_plan_mode(PlanSource::User).await;
    let plan_path = session.plan_file_path().to_string_lossy().to_string();

    registration.set_responses(vec![
        tool_call(
            "call-write",
            "write",
            json!({"path": plan_path, "content": "# Plan"}),
        ),
        tool_call("call-exit", "exit_plan_mode", json!({})),
        tool_call(
            "call-write-after",
            "write",
            json!({"path": cwd.path().join("after.txt").to_string_lossy(), "content": "ok"}),
        ),
        text("executing"),
    ]);
    session.prompt("plan then execute", None).await.unwrap();

    // The review approved the plan, so the follow-up write went through.
    assert!(!session.is_plan_mode_active());
    assert_eq!(
        std::fs::read_to_string(cwd.path().join("after.txt")).unwrap(),
        "ok"
    );
    let results = tool_result_texts(&session);
    assert!(
        results
            .iter()
            .any(|text| text.contains("The user approved the plan")),
        "expected the approval result, got {results:?}"
    );
}

#[tokio::test]
async fn exit_plan_mode_without_a_plan_file_stays_in_plan_mode() {
    let cwd = tempfile::tempdir().unwrap();
    let (_env, services, registration) = services_in("no-plan", cwd.path()).await;
    registration.set_responses(vec![
        tool_call("call-exit", "exit_plan_mode", json!({})),
        text("nothing to review"),
    ]);

    let session = AgentSession::create(Arc::new(services), approve_all(), approve_all_plans())
        .await
        .unwrap();
    session.enter_plan_mode(PlanSource::User).await;
    session.prompt("submit", None).await.unwrap();

    assert!(session.is_plan_mode_active());
    let results = tool_result_texts(&session);
    assert!(
        results
            .iter()
            .any(|text| text.contains("Cannot review the plan")),
        "expected the missing-plan-file result, got {results:?}"
    );
}

#[tokio::test]
async fn a_dismissed_review_keeps_plan_mode_active() {
    let cwd = tempfile::tempdir().unwrap();
    let (_env, services, registration) = services_in("dismiss", cwd.path()).await;

    let session = AgentSession::create(
        Arc::new(services),
        approve_all(),
        Arc::new(step_harness_coding_agent::plan::AutoDismissPlanReview),
    )
    .await
    .unwrap();
    session.enter_plan_mode(PlanSource::User).await;
    let plan_path = session.plan_file_path().to_string_lossy().to_string();

    registration.set_responses(vec![
        tool_call(
            "call-write",
            "write",
            json!({"path": plan_path, "content": "# Plan"}),
        ),
        tool_call("call-exit", "exit_plan_mode", json!({})),
        text("waiting"),
    ]);
    session.prompt("submit for review", None).await.unwrap();

    assert!(session.is_plan_mode_active());
    let results = tool_result_texts(&session);
    assert!(
        results
            .iter()
            .any(|text| text.contains("Staying in plan mode")),
        "expected the stay-in-plan-mode result, got {results:?}"
    );
    // A dismissed review terminates the batch, so no further tool ran.
    assert!(!cwd.path().join("after.txt").exists());
}

#[tokio::test]
async fn an_in_progress_plan_survives_reopening_the_session() {
    let cwd = tempfile::tempdir().unwrap();
    let (paths, _paths_dir) = isolated_paths("restore");
    let registration = faux_registration("restore");
    write_faux_models(&paths, registration.api(), registration.provider_id());
    let env: Arc<dyn ExecutionEnv> = Arc::new(NodeExecutionEnv::new(Default::default()));
    let (services, diagnostics) = SessionServices::create(
        cwd.path().to_string_lossy().to_string(),
        paths.clone(),
        &env,
        true,
    )
    .await;
    assert!(
        diagnostics.is_empty(),
        "unexpected diagnostics: {diagnostics:?}"
    );

    registration.set_responses(vec![text("planning")]);
    let session = AgentSession::create(Arc::new(services), approve_all(), approve_all_plans())
        .await
        .unwrap();
    session.enter_plan_mode(PlanSource::User).await;
    let metadata = session.metadata().clone();
    drop(session);

    // Reopen from the same store: the plan is still in force.
    let (services, diagnostics) =
        SessionServices::create(cwd.path().to_string_lossy().to_string(), paths, &env, true).await;
    assert!(
        diagnostics.is_empty(),
        "unexpected diagnostics: {diagnostics:?}"
    );
    let reopened = AgentSession::open(
        Arc::new(services),
        metadata,
        approve_all(),
        approve_all_plans(),
    )
    .await
    .unwrap();
    assert!(reopened.is_plan_mode_active());
    assert_eq!(reopened.plan_state().plan_source, Some(PlanSource::User));

    // The restored session still blocks writes outside the plan file.
    registration.set_responses(vec![
        tool_call(
            "call-write",
            "write",
            json!({"path": "outside.txt", "content": "x"}),
        ),
        text("blocked"),
    ]);
    reopened.prompt("change a file", None).await.unwrap();
    assert!(!cwd.path().join("outside.txt").exists());
    let results = tool_result_texts(&reopened);
    assert!(
        results
            .iter()
            .any(|text| text.contains("Plan mode blocked")),
        "expected the restored gate to block, got {results:?}"
    );
}


struct TimedPlanReview;
#[async_trait::async_trait]
impl PlanReviewChannel for TimedPlanReview {
    async fn review(&self, _: &step_harness_coding_agent::plan::PlanReviewRequest) -> step_harness_coding_agent::plan::PlanReviewOutcome {
        tokio::time::sleep(std::time::Duration::from_millis(40)).await;
        step_harness_coding_agent::plan::PlanReviewOutcome::Approved
    }
}

#[tokio::test]
async fn plan_review_wait_is_excluded_from_tool_time() {
    let cwd = tempfile::tempdir().unwrap();
    let (_env, services, registration) = services_in("timed-review", cwd.path()).await;
    let session = AgentSession::create(Arc::new(services), approve_all(), Arc::new(TimedPlanReview)).await.unwrap();
    session.enter_plan_mode(PlanSource::User).await;
    let plan_path = session.plan_file_path();
    std::fs::create_dir_all(plan_path.parent().unwrap()).unwrap();
    std::fs::write(&plan_path, "# Plan\nExecute the task").unwrap();
    registration.set_responses(vec![tool_call("timed-exit", "exit_plan_mode", json!({})), text("approved")]);
    session.prompt("review plan", None).await.unwrap();
    let timing = session.execution_timings().await.unwrap().remove(0);
    let approval = timing.approval_wait_duration_ms.unwrap();
    assert!(approval >= 35, "{timing:?}");
    assert_eq!(timing.duration_ms, timing.tool_duration_ms.unwrap() + timing.api_duration_ms.unwrap()
        + approval + timing.retry_wait_duration_ms.unwrap() + timing.other_duration_ms.unwrap());
    assert!(!session.is_plan_mode_active());
}
