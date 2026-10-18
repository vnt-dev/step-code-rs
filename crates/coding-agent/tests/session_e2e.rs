//! End-to-end tests for the session orchestrator: a scripted provider drives
//! real turns through persistence, events, stats, approvals, and compaction.

use std::sync::Arc;
mod common;

use serde_json::json;
use step_harness_agent_core::harness::env::NodeExecutionEnv;
use step_harness_agent_core::harness::types::ExecutionEnv;
use step_harness_coding_agent::permissions::ApprovalMode;
use step_harness_coding_agent::plan::{AutoApprovePlanReview, PlanReviewChannel};
use step_harness_coding_agent::session::{
    ApprovalChannel, ApprovalRequest, AutoDeny, FnApproval, SessionEvent, SessionRuntime,
    SessionServices,
};
use step_harness_coding_agent::{AgentSession, AppPaths};
use step_harness_providers::compat::register_faux_provider;
use step_harness_providers::providers::faux::{
    FauxAssistantMessageOptions, FauxModelDefinition, RegisterFauxProviderOptions,
    faux_assistant_message,
};
use step_harness_providers::types::{ModelThinkingLevel, StopReason};
use tokio_util::sync::CancellationToken;

/// Register a scripted faux provider under a per-test api id.
///
/// The api registry is process-global, so each test needs its own api id to
/// avoid clobbering a concurrently running sibling.
fn faux_registration(name: &str) -> step_harness_providers::compat::FauxProviderRegistration {
    register_faux_provider(&RegisterFauxProviderOptions {
        api: Some(format!("faux-e2e-{name}")),
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

/// Register a faux provider exposing two models, so a restored session's model
/// is distinguishable from the settings default.
fn faux_registration_two_models(
    name: &str,
) -> step_harness_providers::compat::FauxProviderRegistration {
    register_faux_provider(&RegisterFauxProviderOptions {
        api: Some(format!("faux-e2e-{name}")),
        provider: Some(format!("faux-{name}")),
        models: Some(vec![
            FauxModelDefinition {
                id: "faux-1".to_owned(),
                name: Some("Faux One".to_owned()),
                reasoning: Some(false),
                input: None,
                cost: None,
                context_window: Some(128_000),
                max_tokens: Some(4_096),
            },
            FauxModelDefinition {
                id: "faux-2".to_owned(),
                name: Some("Faux Two".to_owned()),
                reasoning: Some(true),
                input: None,
                cost: None,
                context_window: Some(128_000),
                max_tokens: Some(4_096),
            },
        ]),
        ..Default::default()
    })
}

/// A data directory isolated per test.
fn isolated_paths(name: &str) -> AppPaths {
    let dir = tempfile::tempdir().unwrap();
    let paths = AppPaths::new(dir.path().join(name));
    paths.ensure_dirs().unwrap();
    std::mem::forget(dir);
    paths
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

/// Build session services over an isolated data directory.
async fn services(
    name: &str,
) -> (
    AppPaths,
    Arc<dyn ExecutionEnv>,
    SessionServices,
    step_harness_providers::compat::FauxProviderRegistration,
) {
    let registration = faux_registration(name);
    let paths = isolated_paths(name);
    write_faux_models(&paths, registration.api(), registration.provider_id());
    let env: Arc<dyn ExecutionEnv> = Arc::new(NodeExecutionEnv::new(Default::default()));
    let (mut services, diagnostics) = SessionServices::create(".", paths.clone(), &env, true).await;
    services.name_completer = Some(Arc::new(common::SkipNameCompleter));
    assert!(
        diagnostics.is_empty(),
        "unexpected diagnostics: {diagnostics:?}"
    );
    (paths, env, services, registration)
}

fn approve_all() -> Arc<dyn ApprovalChannel> {
    Arc::new(FnApproval(|_: &ApprovalRequest| true))
}

fn approve_all_plans() -> Arc<dyn PlanReviewChannel> {
    Arc::new(AutoApprovePlanReview)
}

struct NameCompleter {
    result: String,
    responses: std::sync::Mutex<std::collections::VecDeque<step_harness_providers::types::AssistantMessage>>,
    calls: std::sync::Mutex<Vec<(String, step_harness_providers::types::Context)>>,
    started: tokio::sync::Notify,
    gate: tokio::sync::Semaphore,
}

impl NameCompleter {
    fn new(result: &str, blocked: bool) -> Arc<Self> {
        Arc::new(Self {
            result: result.to_owned(),
            responses: Default::default(),
            calls: Default::default(),
            started: Default::default(),
            gate: tokio::sync::Semaphore::new(usize::from(!blocked)),
        })
    }
}

#[async_trait::async_trait]
impl step_harness_agent_core::compaction::SimpleCompleter for NameCompleter {
    async fn complete_simple(
        &self,
        model: &step_harness_providers::types::Model,
        context: &step_harness_providers::types::Context,
        options: &step_harness_providers::types::SimpleStreamOptions,
    ) -> step_harness_providers::types::AssistantMessage {
        assert!(matches!(options.base.max_tokens, Some(1024 | 2048)));
        self.calls
            .lock()
            .unwrap()
            .push((model.id.clone(), context.clone()));
        self.started.notify_one();
        let _permit = self.gate.acquire().await.unwrap();
        self.responses.lock().unwrap().pop_front().unwrap_or_else(||
            faux_assistant_message(self.result.as_str(), FauxAssistantMessageOptions::default()))
    }
}

async fn wait_for_name_completion(completer: &Arc<NameCompleter>) {
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while Arc::strong_count(completer) > 2 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("background naming settles");
}

#[tokio::test]
async fn first_message_generates_and_persists_a_short_name_once() {
    let (_paths, _env, mut services, registration) = services("auto-name").await;
    let completer = NameCompleter::new("“修复网页搜索”\n解释不应保存", false);
    services.name_completer = Some(completer.clone());
    let services = Arc::new(services);
    registration.set_responses(vec![
        faux_assistant_message("done", Default::default()).into(),
    ]);
    let session = AgentSession::create(services.clone(), approve_all(), approve_all_plans())
        .await
        .unwrap();
    let mut events = session.subscribe();
    session.prompt("修复网页搜索工具", None).await.unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            if let SessionEvent::SessionInfoChanged { name } = events.recv().await.unwrap() {
                assert_eq!(name.as_deref(), Some("修复网页搜索"));
                break;
            }
        }
    })
    .await
    .unwrap();
    assert_eq!(session.name().await.as_deref(), Some("修复网页搜索"));
    {
        let calls = completer.calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "faux-1");
        assert!(calls[0].1.tools.is_none());
        assert_eq!(calls[0].1.messages.len(), 1);
    }
    // Metadata completion must not appear as a normal assistant reply.
    assert_eq!(
        session
            .entries()
            .await
            .unwrap()
            .iter()
            .filter(|entry| matches!(entry, step_harness_agent_core::session::Entry::Message(_)))
            .count(),
        2
    );
    wait_for_name_completion(&completer).await;
    let metadata = session.metadata().clone();
    drop(session);
    let reopened = AgentSession::open(services.clone(), metadata, approve_all(), approve_all_plans())
        .await
        .unwrap();
    assert_eq!(reopened.name().await.as_deref(), Some("修复网页搜索"));
    reopened.set_name(None).await.unwrap();
    registration.set_responses(vec![
        faux_assistant_message("second", Default::default()).into(),
    ]);
    reopened.prompt("第二条消息", None).await.unwrap();
    assert_eq!(completer.calls.lock().unwrap().len(), 1);
    assert_eq!(reopened.name().await, None);
    let metadata = reopened.metadata().clone();
    drop(reopened);
    let reopened = AgentSession::open(services, metadata, approve_all(), approve_all_plans()).await.unwrap();
    registration.set_responses(vec![faux_assistant_message("third", Default::default()).into()]);
    reopened.prompt("手动清除后重开", None).await.unwrap();
    assert_eq!(completer.calls.lock().unwrap().len(), 1);
    assert_eq!(reopened.name().await, None);
}

#[tokio::test]
async fn truncated_title_retries_with_more_tokens_and_saves_the_result() {
    let (_paths, _env, mut services, registration) = services("title-token-retry").await;
    let completer = NameCompleter::new("广州天气查询", false);
    let mut truncated = faux_assistant_message("", Default::default());
    truncated.stop_reason = StopReason::Length;
    completer.responses.lock().unwrap().push_back(truncated);
    services.name_completer = Some(completer.clone());
    registration.set_responses(vec![faux_assistant_message("done", Default::default()).into()]);
    let session = AgentSession::create(Arc::new(services), approve_all(), approve_all_plans()).await.unwrap();
    session.prompt("查询广州天气", None).await.unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(2), completer.started.notified()).await.unwrap();
    wait_for_name_completion(&completer).await;
    assert_eq!(completer.calls.lock().unwrap().len(), 2);
    assert_eq!(session.name().await.as_deref(), Some("广州天气查询"));
}

#[tokio::test]
async fn failed_title_can_retry_after_reopening_using_the_original_first_message() {
    let (_paths, _env, mut services, registration) = services("title-reopen-retry").await;
    let completer = NameCompleter::new("原始请求标题", false);
    for _ in 0..2 {
        let mut failure = faux_assistant_message("", Default::default());
        failure.stop_reason = StopReason::Error;
        completer.responses.lock().unwrap().push_back(failure);
    }
    services.name_completer = Some(completer.clone());
    let services = Arc::new(services);
    registration.set_responses(vec![faux_assistant_message("first reply", Default::default()).into()]);
    let session = AgentSession::create(services.clone(), approve_all(), approve_all_plans()).await.unwrap();
    session.prompt("最初的用户请求", None).await.unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(2), completer.started.notified()).await.unwrap();
    wait_for_name_completion(&completer).await;
    assert_eq!(session.name().await, None);
    let metadata = session.metadata().clone();
    drop(session);
    let reopened = AgentSession::open(services, metadata, approve_all(), approve_all_plans()).await.unwrap();
    registration.set_responses(vec![faux_assistant_message("next reply", Default::default()).into()]);
    reopened.prompt("这是第二条，不应成为标题主题", None).await.unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(2), completer.started.notified()).await.unwrap();
    wait_for_name_completion(&completer).await;
    assert_eq!(reopened.name().await.as_deref(), Some("原始请求标题"));
    let calls = completer.calls.lock().unwrap();
    assert_eq!(calls.len(), 3);
    let step_harness_providers::types::Message::User(first) = &calls[2].1.messages[0] else {panic!("user context")};
    assert!(serde_json::to_string(&first.content).unwrap().contains("最初的用户请求"));
}

#[tokio::test]
async fn existing_names_skip_model_naming() {
    let (_paths, _env, mut services, registration) = services("named-first-message").await;
    let completer = NameCompleter::new("自动名称", false);
    services.name_completer = Some(completer.clone());
    registration.set_responses(vec![
        faux_assistant_message("done", Default::default()).into(),
    ]);
    let session = AgentSession::create(Arc::new(services), approve_all(), approve_all_plans())
        .await
        .unwrap();
    session.set_name(Some("手动名称")).await.unwrap();
    session.prompt("first message", None).await.unwrap();
    assert!(completer.calls.lock().unwrap().is_empty());
    assert_eq!(session.name().await.as_deref(), Some("手动名称"));
}

#[tokio::test]
async fn manual_rename_or_clear_wins_over_a_pending_generated_name() {
    for (index, manual) in [Some("手动名称"), None].into_iter().enumerate() {
        let (_paths, _env, mut services, registration) =
            services(&format!("rename-race-{index}")).await;
        let completer = NameCompleter::new("自动名称", true);
        services.name_completer = Some(completer.clone());
        registration.set_responses(vec![
            faux_assistant_message("done", Default::default()).into(),
        ]);
        let session = AgentSession::create(Arc::new(services), approve_all(), approve_all_plans())
            .await
            .unwrap();
        session.prompt("first message", None).await.unwrap();
        tokio::time::timeout(
            std::time::Duration::from_secs(2),
            completer.started.notified(),
        )
        .await
        .unwrap();
        session.set_name(manual).await.unwrap();
        completer.gate.add_permits(1);
        wait_for_name_completion(&completer).await;
        assert_eq!(session.name().await.as_deref(), manual);
    }
}

#[tokio::test]
async fn empty_model_name_keeps_the_session_unnamed_without_affecting_the_reply() {
    let (_paths, _env, mut services, registration) = services("empty-name").await;
    let completer = NameCompleter::new(" ", false);
    services.name_completer = Some(completer.clone());
    registration.set_responses(vec![
        faux_assistant_message("normal reply", Default::default()).into(),
    ]);
    let session = AgentSession::create(Arc::new(services), approve_all(), approve_all_plans())
        .await
        .unwrap();
    session.prompt("first message", None).await.unwrap();
    tokio::time::timeout(
        std::time::Duration::from_secs(2),
        completer.started.notified(),
    )
    .await
    .unwrap();
    wait_for_name_completion(&completer).await;
    assert_eq!(session.name().await, None);
    assert!(
        session
            .state()
            .messages
            .last()
            .unwrap()
            .as_assistant()
            .is_some()
    );
}

#[tokio::test]
async fn search_web_reaches_the_model_in_new_and_reopened_sessions() {
    use step_harness_providers::providers::faux::FauxResponseStep;

    let (_paths, _env, services, registration) = services("search-web-visible").await;
    let services = Arc::new(services);
    let response = FauxResponseStep::Factory(Arc::new(|context, _, _, _| {
        Box::pin(async move {
            let tools = context.tools.as_ref().expect("model receives tools");
            let search = tools
                .iter()
                .find(|tool| tool.name == "search_web")
                .expect("model must receive search_web");
            assert_eq!(search.parameters["properties"]["query"]["type"], "string");
            assert!(search.description.contains("Sources:"));
            assert!(
                context
                    .system_prompt
                    .as_deref()
                    .unwrap()
                    .contains("search_web")
            );
            Ok(faux_assistant_message(
                "search available",
                FauxAssistantMessageOptions::default(),
            ))
        })
    }));
    registration.set_responses(vec![response.clone()]);
    let session = AgentSession::create(services.clone(), approve_all(), approve_all_plans())
        .await
        .unwrap();
    session.prompt("check tools", None).await.unwrap();
    let metadata = session.metadata().clone();
    drop(session);

    registration.set_responses(vec![response]);
    let reopened = AgentSession::open(services, metadata, approve_all(), approve_all_plans())
        .await
        .unwrap();
    reopened
        .prompt("check tools after reopening", None)
        .await
        .unwrap();
}

#[tokio::test]
async fn prompt_runs_a_turn_persists_and_streams_events() {
    let (_paths, _env, services, registration) = services("turn").await;
    registration.set_responses(vec![
        faux_assistant_message("hello from faux", FauxAssistantMessageOptions::default()).into(),
    ]);

    let session = AgentSession::create(Arc::new(services), approve_all(), approve_all_plans())
        .await
        .unwrap();

    let mut events = session.subscribe();
    session.prompt("hi there", None).await.unwrap();

    // The transcript holds the user prompt and the scripted reply.
    let messages = session.state().messages;
    assert_eq!(messages.len(), 2);
    assert!(messages[0].as_user().is_some());
    let assistant = messages[1].as_assistant().expect("assistant reply");
    assert_eq!(assistant.stop_reason, StopReason::Stop);

    // A settled event closed the run.
    let mut saw_stream = false;
    let mut saw_settled = false;
    while let Ok(event) = events.try_recv() {
        match event {
            SessionEvent::Agent { .. } => saw_stream = true,
            SessionEvent::Settled => saw_settled = true,
            _ => {}
        }
    }
    assert!(saw_stream, "expected agent events");
    assert!(saw_settled, "expected settled event");

    // Stats recorded the scripted usage.
    assert!(session.stats().turns >= 1);

    // The durable store holds both messages.
    let entries = session.entries().await.unwrap();
    assert_eq!(
        entries.iter().filter(|entry| matches!(entry, step_harness_agent_core::session::Entry::Message(_))).count(),
        2,
    );
    let timings = session.execution_timings().await.unwrap();
    assert_eq!(timings.len(), 1);
    assert_eq!(timings[0].first_message_timestamp, messages[0].timestamp());
    assert_eq!(timings[0].last_message_timestamp, messages[1].timestamp());
    assert!(entries.iter().any(|entry| matches!(entry, step_harness_agent_core::session::Entry::Custom(custom) if custom.custom_type == "execution_timing")));
}

#[tokio::test]
async fn recall_removes_the_last_prompt_and_tools_across_reopening() {
    use step_harness_providers::providers::faux::FauxContentBlock;
    use step_harness_providers::types::ToolCall;
    let (_paths, _env, services, registration) = services("recall-persist").await;
    registration.set_responses(vec![
        faux_assistant_message("retained answer", Default::default()).into(),
        faux_assistant_message(vec![FauxContentBlock::ToolCall(ToolCall {
            id: "recalled-tool".into(), name: "read".into(),
            arguments: serde_json::from_value(json!({ "path": "nonexistent-recall-fixture.txt" })).unwrap(),
            namespace: None, thought_signature: None,
        })], Default::default()).into(),
        faux_assistant_message("removed answer", Default::default()).into(),
    ]);
    let services = Arc::new(services);
    let session = AgentSession::create(services.clone(), approve_all(), approve_all_plans()).await.unwrap();
    session.prompt("keep this", None).await.unwrap();
    let retained = session.state().messages;
    tokio::time::sleep(std::time::Duration::from_millis(2)).await;
    session.prompt("recall this", None).await.unwrap();
    assert!(session.state().messages.iter().any(|message| message.as_tool_result().is_some()));
    let timestamp = session.state().messages[retained.len()].timestamp();
    session.set_approval_mode(ApprovalMode::Bypass).await;
    session.set_thinking_level(ModelThinkingLevel::Off).await;
    session.enter_plan_mode(step_harness_coding_agent::plan::PlanSource::User).await;
    let metadata = session.metadata().clone();
    let mut events = session.subscribe();
    session.recall_last_user_message(timestamp).await.unwrap();
    assert_eq!(serde_json::to_value(session.state().messages).unwrap(), serde_json::to_value(&retained).unwrap());
    assert_eq!(session.execution_timings().await.unwrap().len(), 1);
    assert!(std::iter::from_fn(|| events.try_recv().ok()).any(|event| matches!(event, SessionEvent::MessagesRecalled { .. })));
    drop(session);

    let reopened = AgentSession::open(services, metadata, approve_all(), approve_all_plans()).await.unwrap();
    assert_eq!(serde_json::to_value(reopened.state().messages).unwrap(), serde_json::to_value(&retained).unwrap());
    assert_eq!(reopened.execution_timings().await.unwrap().len(), 1);
    assert_eq!(reopened.approval_mode(), ApprovalMode::Bypass);
    assert!(reopened.plan_state().enabled);
    // The next model request receives only retained history plus its new prompt.
    registration.set_responses(vec![step_harness_providers::providers::faux::FauxResponseStep::Factory(Arc::new(move |context, _, _, _| {
        Box::pin(async move {
            assert_eq!(context.messages.len(), 3);
            assert!(!serde_json::to_string(&context).unwrap().contains("recall this"));
            Ok(faux_assistant_message("replacement answer", Default::default()))
        })
    }))]);
    reopened.prompt("replacement prompt", None).await.unwrap();
}

#[tokio::test]
async fn recall_first_prompt_is_durable_and_stale_requests_do_not_remove_more_history() {
    let (_paths, _env, services, registration) = services("recall-first").await;
    registration.set_responses(vec![faux_assistant_message("answer", Default::default()).into()]);
    let services = Arc::new(services);
    let session = AgentSession::create(services.clone(), approve_all(), approve_all_plans()).await.unwrap();
    assert!(session.recall_last_user_message(1).await.is_err());
    session.prompt("first prompt", None).await.unwrap();
    let timestamp = session.state().messages[0].timestamp();
    let leaf = session.session().get_leaf_id().await.unwrap();
    assert!(session.recall_last_user_message(timestamp + 1).await.is_err());
    assert_eq!(session.session().get_leaf_id().await.unwrap(), leaf);
    session.recall_last_user_message(timestamp).await.unwrap();
    assert!(session.state().messages.is_empty());
    assert!(session.execution_timings().await.unwrap().is_empty());
    assert!(session.recall_last_user_message(timestamp).await.is_err());
    let metadata = session.metadata().clone();
    drop(session);
    let reopened = AgentSession::open(services, metadata, approve_all(), approve_all_plans()).await.unwrap();
    assert!(reopened.state().messages.is_empty());
    assert!(reopened.execution_timings().await.unwrap().is_empty());
}

#[tokio::test]
async fn approval_denial_blocks_the_tool_call() {
    let denied = Arc::new(AutoDeny) as Arc<dyn ApprovalChannel>;
    let (_paths, _env, services, registration) = services("deny").await;

    // First response asks to run a command; the denial becomes the tool result.
    registration.set_responses(vec![
        step_harness_providers::providers::faux::faux_assistant_message(
            vec![
                step_harness_providers::providers::faux::FauxContentBlock::ToolCall(
                    step_harness_providers::types::ToolCall {
                        id: "call-1".to_owned(),
                        name: "bash".to_owned(),
                        arguments: serde_json::from_value(json!({"command": "echo hi"})).unwrap(),
                        namespace: None,
                        thought_signature: None,
                    },
                ),
            ],
            FauxAssistantMessageOptions::default(),
        )
        .into(),
        faux_assistant_message("done", FauxAssistantMessageOptions::default()).into(),
    ]);

    let session = AgentSession::create(Arc::new(services), denied, approve_all_plans())
        .await
        .unwrap();
    session.prompt("run something", None).await.unwrap();

    // The blocked tool result is visible to the model as an error.
    let has_blocked_result = session
        .state()
        .messages
        .iter()
        .filter_map(|message| message.as_tool_result())
        .any(|result| {
            result
                .content
                .iter()
                .filter_map(|block| block.text())
                .any(|text| text.text.contains("denied"))
        });
    assert!(has_blocked_result, "tool call should have been denied");
}

#[tokio::test]
async fn approval_modes_cycle_through_three_presets() {
    let (_paths, _env, services, registration) = services("approval-modes").await;
    registration.set_responses(vec![
        faux_assistant_message("no tools needed", FauxAssistantMessageOptions::default()).into(),
    ]);

    let session = AgentSession::create(Arc::new(services), approve_all(), approve_all_plans())
        .await
        .unwrap();
    session.set_approval_mode(ApprovalMode::Ask).await;
    assert_eq!(session.cycle_approval_mode().await, ApprovalMode::Bypass);
    assert_eq!(session.cycle_approval_mode().await, ApprovalMode::Autopilot);
    assert_eq!(session.cycle_approval_mode().await, ApprovalMode::Ask);
    session.set_approval_mode(ApprovalMode::Autopilot).await;
    assert_eq!(session.approval_mode(), ApprovalMode::Autopilot);
    session.prompt("just chat", None).await.unwrap();
    assert_eq!(session.state().messages.len(), 2);
}

#[tokio::test]
async fn reopening_a_session_restores_the_approval_mode() {
    let registration = faux_registration("approval-restore");
    let paths = isolated_paths("approval-restore");
    write_faux_models(&paths, registration.api(), registration.provider_id());
    // A preset that differs from both the enum default and the mode recorded
    // below, so a settings fallback cannot pass for a restore.
    std::fs::write(
        paths.settings_file(),
        serde_json::to_string_pretty(&json!({ "permissions": { "preset": "bypass" } })).unwrap(),
    )
    .unwrap();
    let env: Arc<dyn ExecutionEnv> = Arc::new(NodeExecutionEnv::new(Default::default()));
    let (services, diagnostics) = SessionServices::create(".", paths.clone(), &env, true).await;
    assert!(
        diagnostics.is_empty(),
        "unexpected diagnostics: {diagnostics:?}"
    );

    let session = AgentSession::create(Arc::new(services), approve_all(), approve_all_plans())
        .await
        .unwrap();
    // A session that never chose a mode still follows the settings preset.
    assert_eq!(session.approval_mode(), ApprovalMode::Bypass);
    session.set_approval_mode(ApprovalMode::Autopilot).await;
    let metadata = session.metadata().clone();
    drop(session);

    let (services, diagnostics) = SessionServices::create(".", paths, &env, true).await;
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
    assert_eq!(reopened.approval_mode(), ApprovalMode::Autopilot);
}

#[tokio::test]
async fn a_new_runtime_reopens_the_session_the_previous_run_left_active() {
    let registration = faux_registration("restore-active");
    let paths = isolated_paths("restore-active");
    write_faux_models(&paths, registration.api(), registration.provider_id());
    let env: Arc<dyn ExecutionEnv> = Arc::new(NodeExecutionEnv::new(Default::default()));

    let metadata;
    {
        let runtime = SessionRuntime::new(".", paths.clone(), approve_all(), approve_all_plans());
        let session = runtime.new_session(&env).await.unwrap();
        session.set_approval_mode(ApprovalMode::Autopilot).await;
        metadata = session.metadata().clone();
        // Dropping the runtime releases the session file's lease, exactly as
        // quitting the app does.
        drop(session);
        drop(runtime);
    }

    let runtime = SessionRuntime::new(".", paths, approve_all(), approve_all_plans());
    let restored = runtime
        .restore_last_active(&env)
        .await
        .expect("the last active session reopens");
    assert_eq!(restored.metadata().id, metadata.id);
    assert_eq!(restored.approval_mode(), ApprovalMode::Autopilot);
    assert_eq!(
        runtime
            .active()
            .await
            .map(|session| session.metadata().id.clone()),
        Some(metadata.id)
    );
}

#[tokio::test]
async fn restoring_forgets_a_last_session_that_was_deleted() {
    let registration = faux_registration("restore-deleted");
    let paths = isolated_paths("restore-deleted");
    write_faux_models(&paths, registration.api(), registration.provider_id());
    let env: Arc<dyn ExecutionEnv> = Arc::new(NodeExecutionEnv::new(Default::default()));

    let runtime = SessionRuntime::new(".", paths.clone(), approve_all(), approve_all_plans());
    let session = runtime.new_session(&env).await.unwrap();
    let metadata = session.metadata().clone();
    drop(session);
    runtime.delete_session(&env, &metadata).await.unwrap();
    drop(runtime);

    // The next start finds nothing to reopen, and forgets the stale id instead
    // of rescanning for it on every launch.
    let runtime = SessionRuntime::new(".", paths.clone(), approve_all(), approve_all_plans());
    assert!(runtime.restore_last_active(&env).await.is_none());
    assert!(runtime.active().await.is_none());
    assert!(!paths.root().join("last-session.json").exists());
}

#[tokio::test]
async fn sessions_persist_and_reopen() {
    let (paths, env, services, registration) = services("reopen").await;
    registration.set_responses(vec![
        faux_assistant_message("first reply", FauxAssistantMessageOptions::default()).into(),
    ]);

    let session = AgentSession::create(Arc::new(services), approve_all(), approve_all_plans())
        .await
        .unwrap();
    session.prompt("remember this", None).await.unwrap();
    let timing = session.execution_timings().await.unwrap().remove(0);
    let metadata = session.metadata().clone();
    session.set_name(Some("named session")).await.unwrap();
    session.set_project(Some("step-code")).await.unwrap();
    drop(session);

    // Reopen from the same store: transcript, name, and project survive.
    let (services, _diagnostics) = SessionServices::create(".", paths, &env, true).await;
    let reopened = AgentSession::open(
        Arc::new(services),
        metadata,
        approve_all(),
        approve_all_plans(),
    )
    .await
    .unwrap();
    assert_eq!(reopened.name().await.as_deref(), Some("named session"));
    assert_eq!(reopened.project().await.as_deref(), Some("step-code"));
    let restored = reopened.execution_timings().await.unwrap();
    assert_eq!(restored.len(), 1);
    assert_eq!(restored[0].first_message_timestamp, timing.first_message_timestamp);
    assert_eq!(restored[0].last_message_timestamp, timing.last_message_timestamp);
    assert_eq!(restored[0].duration_ms, timing.duration_ms);
    assert_eq!(restored[0].tool_duration_ms, timing.tool_duration_ms);
    assert_eq!(restored[0].api_duration_ms, timing.api_duration_ms);
    assert_eq!(restored[0].approval_wait_duration_ms, timing.approval_wait_duration_ms);
    assert_eq!(restored[0].retry_wait_duration_ms, timing.retry_wait_duration_ms);
    assert_eq!(restored[0].other_duration_ms, timing.other_duration_ms);
    assert_timing_partition(&restored[0]);
    let messages = reopened.state().messages;
    assert_eq!(messages.len(), 2);
    assert!(messages[0].as_user().is_some());
}

#[tokio::test]
async fn listing_named_sessions_does_not_relock_the_active_session() {
    let (paths, env, _services, _registration) = services("list-names").await;
    let runtime = SessionRuntime::new(".", paths, approve_all(), approve_all_plans());
    let active = runtime.new_session(&env).await.unwrap();

    // Startup lists every session while the active one still holds the
    // exclusive lease on its own file: reopening that file would collide with
    // the lease, so the listing must read the name from the live session.
    let named = runtime.list_sessions_with_names(&env).await.unwrap();
    assert_eq!(named.len(), 1);
    assert_eq!(named[0].0.id, active.metadata().id);
    assert_eq!(named[0].1, None);

    active.set_name(Some("listed")).await.unwrap();
    let named = runtime.list_sessions_with_names(&env).await.unwrap();
    assert_eq!(named[0].1.as_deref(), Some("listed"));
    assert_eq!(named[0].2, None);

    active.set_project(Some("step-code")).await.unwrap();
    let named = runtime.list_sessions_with_names(&env).await.unwrap();
    assert_eq!(named[0].2.as_deref(), Some("step-code"));
}

#[tokio::test]
async fn project_sessions_switch_directory_services_and_list_all_projects() {
    let (paths, env, _services, _registration) = services("project-runtime").await;
    let folders = tempfile::tempdir().unwrap();
    let first = folders.path().join("first");
    let second = folders.path().join("second");
    std::fs::create_dir_all(first.join(".step-rs")).unwrap();
    std::fs::create_dir_all(second.join(".step-rs")).unwrap();
    std::fs::write(
        first.join(".step-rs/settings.json"),
        r#"{"hideThinkingBlock":true}"#,
    )
    .unwrap();
    std::fs::write(
        second.join(".step-rs/settings.json"),
        r#"{"hideThinkingBlock":false}"#,
    )
    .unwrap();
    let first = first.to_string_lossy().to_string();
    let second = second.to_string_lossy().to_string();
    let runtime = SessionRuntime::new(&first, paths, approve_all(), approve_all_plans());
    let initial = runtime.new_session(&env).await.unwrap();
    let metadata = initial.metadata().clone();
    initial.set_name(Some("first project")).await.unwrap();
    drop(initial);
    let other = runtime.new_session_in(&env, &second).await.unwrap();
    assert!(other.metadata().cwd.ends_with("second"));
    assert_eq!(
        runtime
            .services(&env)
            .await
            .unwrap()
            .settings()
            .hide_thinking_block,
        Some(false)
    );
    drop(other);
    let next = runtime.new_session(&env).await.unwrap();
    assert!(next.metadata().cwd.ends_with("second"));
    drop(next);
    assert_eq!(runtime.list_sessions(&env).await.unwrap().len(), 3);
    let reopened = runtime.open_session(&env, metadata).await.unwrap();
    assert_eq!(reopened.name().await.as_deref(), Some("first project"));
    assert!(runtime.services(&env).await.unwrap().cwd.ends_with("first"));
    assert_eq!(
        runtime
            .services(&env)
            .await
            .unwrap()
            .settings()
            .hide_thinking_block,
        Some(true)
    );
}

#[tokio::test]
async fn model_switch_updates_the_active_model() {
    let (_paths, _env, services, registration) = services("switch").await;
    registration.set_responses(vec![
        faux_assistant_message("ok", FauxAssistantMessageOptions::default()).into(),
    ]);

    let session = AgentSession::create(Arc::new(services), approve_all(), approve_all_plans())
        .await
        .unwrap();

    let model = session.current_model().expect("faux model selected");
    assert_eq!(model.provider.as_str(), "faux-switch");
    assert_eq!(model.id, "faux-1");

    session.set_model("faux-switch", "faux-1").await.unwrap();
    assert_eq!(session.provider_id(), "faux-switch");
    assert_eq!(session.model_id(), "faux-1");

    // Unknown models are rejected without touching the active model.
    assert!(session.set_model("faux-switch", "nope").await.is_err());
    assert_eq!(session.model_id(), "faux-1");
}

#[tokio::test]
async fn switching_away_and_back_restores_the_sessions_model() {
    let registration = faux_registration_two_models("switch-back");
    let paths = isolated_paths("switch-back");
    let config = json!({
        "providers": {
            registration.provider_id(): {
                "name": "Faux",
                "baseUrl": "https://faux.invalid",
                "apiKey": "test-key",
                "api": registration.api(),
                "models": [
                    { "id": "faux-1", "name": "Faux One", "contextWindow": 128000, "maxTokens": 4096 },
                    { "id": "faux-2", "name": "Faux Two", "reasoning": true, "contextWindow": 128000, "maxTokens": 4096 }
                ]
            }
        }
    });
    std::fs::write(
        paths.models_file(),
        serde_json::to_string_pretty(&config).unwrap(),
    )
    .unwrap();
    let env: Arc<dyn ExecutionEnv> = Arc::new(NodeExecutionEnv::new(Default::default()));
    let (_services, diagnostics) = SessionServices::create(".", paths.clone(), &env, true).await;
    assert!(
        diagnostics.is_empty(),
        "unexpected diagnostics: {diagnostics:?}"
    );
    registration.set_responses(vec![
        faux_assistant_message("reply from faux-2", FauxAssistantMessageOptions::default()).into(),
    ]);

    let runtime = SessionRuntime::new(".", paths, approve_all(), approve_all_plans());
    let first = runtime.new_session(&env).await.unwrap();
    first
        .set_model(registration.provider_id(), "faux-2")
        .await
        .unwrap();
    first.set_thinking_level(ModelThinkingLevel::High).await;
    first.prompt("hello", None).await.unwrap();
    assert_eq!(first.model_id(), "faux-2");
    let metadata = first.metadata().clone();
    drop(first);

    // A second session takes over, as the UI does when the user switches.
    let second = runtime.new_session(&env).await.unwrap();
    assert_ne!(second.metadata().id, metadata.id);
    drop(second);

    // Switching back must restore the model this session was using, not the
    // catalog default the fresh controller would pick.
    let reopened = runtime.open_session(&env, metadata.clone()).await.unwrap();
    assert_eq!(reopened.provider_id(), registration.provider_id());
    assert_eq!(reopened.model_id(), "faux-2");
    assert_eq!(
        reopened.current_model().map(|model| model.id),
        Some("faux-2".to_owned())
    );
    // The restored model is the one the turn runs with, and the recorded
    // thinking level survives the switch too.
    assert_eq!(
        reopened.state().model.id,
        reopened.current_model().expect("model restored").id
    );
    assert_eq!(reopened.thinking_level(), ModelThinkingLevel::High);
    drop(reopened);
}

#[tokio::test]
async fn reopening_a_session_restores_an_unsent_model_switch() {
    let registration = faux_registration_two_models("unsent-switch");
    let paths = isolated_paths("unsent-switch");
    let config = json!({
        "providers": {
            registration.provider_id(): {
                "name": "Faux",
                "baseUrl": "https://faux.invalid",
                "apiKey": "test-key",
                "api": registration.api(),
                "models": [
                    { "id": "faux-1", "name": "Faux One", "contextWindow": 128000, "maxTokens": 4096 },
                    { "id": "faux-2", "name": "Faux Two", "contextWindow": 128000, "maxTokens": 4096 }
                ]
            }
        }
    });
    std::fs::write(
        paths.models_file(),
        serde_json::to_string_pretty(&config).unwrap(),
    )
    .unwrap();
    let env: Arc<dyn ExecutionEnv> = Arc::new(NodeExecutionEnv::new(Default::default()));
    let (services, diagnostics) = SessionServices::create(".", paths.clone(), &env, true).await;
    assert!(
        diagnostics.is_empty(),
        "unexpected diagnostics: {diagnostics:?}"
    );

    // The switch happens before any reply, so the transcript cannot carry it.
    let session = AgentSession::create(Arc::new(services), approve_all(), approve_all_plans())
        .await
        .unwrap();
    session
        .set_model(registration.provider_id(), "faux-2")
        .await
        .unwrap();
    let metadata = session.metadata().clone();
    drop(session);

    let (services, diagnostics) = SessionServices::create(".", paths, &env, true).await;
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
    assert_eq!(reopened.provider_id(), registration.provider_id());
    assert_eq!(reopened.model_id(), "faux-2");
    assert_eq!(
        reopened.current_model().map(|model| model.id),
        Some("faux-2".to_owned())
    );
}

#[tokio::test]
async fn aborting_a_run_emits_no_further_turns() {
    // A factory that blocks until cancelled proves abort settles the run.
    let (_paths, _env, services, registration) = services("abort").await;
    registration.set_responses(vec![
        step_harness_providers::providers::faux::FauxResponseStep::Factory(Arc::new(
            |_context, _options, _state, _model| {
                Box::pin(async {
                    tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                    Err("should have been cancelled".to_owned())
                })
            },
        )),
    ]);

    let session = Arc::new(
        AgentSession::create(Arc::new(services), approve_all(), approve_all_plans())
            .await
            .unwrap(),
    );
    let runner = session.clone();
    let handle = tokio::spawn(async move { runner.prompt("slow", None).await });

    while !session.is_running() {
        tokio::task::yield_now().await;
    }
    session.abort();
    let result = tokio::time::timeout(std::time::Duration::from_secs(5), handle)
        .await
        .expect("run should settle after abort")
        .unwrap();
    assert!(result.is_ok());
    assert!(!session.is_running());
    // No stray cancellation tokens remain.
    let _ = CancellationToken::new();
}

#[tokio::test]
async fn delete_removes_the_file_for_active_and_inactive_sessions() {
    let (paths, env, _services, _registration) = services("delete-runtime").await;
    let runtime = SessionRuntime::new(".", paths.clone(), approve_all(), approve_all_plans());

    // The startup session: created and active. The runtime is the only
    // holder, mirroring the bridge (forward_events drops its clone).
    let active = runtime.new_session(&env).await.unwrap();
    let active_id = active.metadata().id.clone();
    drop(active);
    // A second session in another project, then detach it like the UI would
    // when switching away.
    let other_cwd = tempfile::tempdir().unwrap().path().to_path_buf();
    let inactive = runtime
        .new_session_in(&env, &other_cwd.to_string_lossy())
        .await
        .unwrap();
    let inactive_id = inactive.metadata().id.clone();
    drop(inactive);

    // Switch back so the other project's session is actually inactive.
    let listed = runtime.list_sessions(&env).await.unwrap();
    let active_meta = listed.iter().find(|m| m.id == active_id).unwrap().clone();
    drop(runtime.open_session(&env, active_meta).await.unwrap());

    // Delete the detached (inactive) session first.
    let listed = runtime.list_sessions(&env).await.unwrap();
    let inactive_meta = listed
        .iter()
        .find(|m| m.id == inactive_id)
        .cloned()
        .expect("inactive session listed");
    runtime.delete_session(&env, &inactive_meta).await.unwrap();
    assert_eq!(runtime.active().await.unwrap().metadata().id, active_id);
    assert!(!std::path::Path::new(&inactive_meta.path).exists());

    // Delete the ACTIVE session through the same bridge path: list, find,
    // delete.
    let listed = runtime.list_sessions(&env).await.unwrap();
    let active_meta = listed
        .iter()
        .find(|m| m.id == active_id)
        .cloned()
        .expect("active session listed");
    runtime.delete_session(&env, &active_meta).await.unwrap();
    assert!(runtime.active().await.is_none());
    assert!(!std::path::Path::new(&active_meta.path).exists());

    let remaining = runtime.list_sessions(&env).await.unwrap();
    assert!(
        remaining
            .iter()
            .all(|m| m.id != active_id && m.id != inactive_id),
        "sessions should be gone: {remaining:?}"
    );
}

#[tokio::test]
async fn image_prompts_and_queues_persist_and_reopen() {
    use step_harness_providers::types::{ImageContent, UserContentBlock};
    let (_paths, _env, services, registration) = services("image-input").await;
    registration.set_responses(
        (0..3)
            .map(|_| {
                faux_assistant_message("received", FauxAssistantMessageOptions::default()).into()
            })
            .collect(),
    );
    let services = Arc::new(services);
    let session = AgentSession::create(services.clone(), approve_all(), approve_all_plans())
        .await
        .unwrap();
    let image = ImageContent {
        data: "aGVsbG8=".to_owned(),
        mime_type: "image/png".to_owned(),
    };
    session.steer_with_images("steering", vec![image.clone()]);
    session.follow_up_with_images("", vec![image.clone()]);
    session.prompt("", Some(vec![image.clone()])).await.unwrap();
    let assert_images =
        |messages: Vec<step_harness_agent_core::types::AgentMessage>| {
            let users: Vec<_> = messages
                .iter()
                .filter_map(|message| message.as_user())
                .collect();
            assert_eq!(users.len(), 3);
            for user in users {
                assert!(user.content.blocks().unwrap().iter().any(
                    |block| matches!(block, UserContentBlock::Image(value) if value == &image)
                ));
            }
        };
    assert_images(session.state().messages);
    let metadata = session.metadata().clone();
    drop(session);
    let reopened = AgentSession::open(services, metadata, approve_all(), approve_all_plans())
        .await
        .unwrap();
    assert_images(reopened.state().messages);
}

/// Hold approvals until the test answers each originating session independently.
struct WaitingApproval {
    requests:
        tokio::sync::mpsc::UnboundedSender<(ApprovalRequest, tokio::sync::oneshot::Sender<bool>)>,
}

#[async_trait::async_trait]
impl ApprovalChannel for WaitingApproval {
    async fn confirm(&self, request: &ApprovalRequest) -> bool {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        self.requests.send((request.clone(), sender)).unwrap();
        receiver.await.unwrap_or(false)
    }
}

#[tokio::test]
async fn running_sessions_survive_switching_and_approvals_are_independent() {
    use step_harness_providers::providers::faux::FauxContentBlock;
    use step_harness_providers::types::ToolCall;
    let (paths, env, _services, registration) = services("parallel-runtime").await;
    let tool = || {
        faux_assistant_message(
            vec![FauxContentBlock::ToolCall(ToolCall {
                id: "same-provider-call-id".into(),
                name: "bash".into(),
                arguments: serde_json::from_value(json!({ "command": "echo hello" })).unwrap(),
                namespace: None,
                thought_signature: None,
            })],
            FauxAssistantMessageOptions::default(),
        )
        .into()
    };
    registration.set_responses(vec![
        tool(),
        tool(),
        faux_assistant_message("finished", FauxAssistantMessageOptions::default()).into(),
        faux_assistant_message("finished", FauxAssistantMessageOptions::default()).into(),
    ]);
    let (requests, mut pending) = tokio::sync::mpsc::unbounded_channel();
    let runtime = SessionRuntime::new(
        ".",
        paths,
        Arc::new(WaitingApproval { requests }),
        approve_all_plans(),
    );
    let first = runtime.new_session(&env).await.unwrap();
    first.set_name(Some("first running session")).await.unwrap();
    let first_metadata = first.metadata().clone();
    let mut first_events = first.subscribe();
    let first_run = tokio::spawn({
        let session = first.clone();
        async move { session.prompt("first prompt", None).await }
    });
    let (first_request, first_answer) =
        tokio::time::timeout(std::time::Duration::from_secs(10), pending.recv())
            .await
            .unwrap()
            .unwrap();

    let active_execution = first.request_execution_id();
    assert!(active_execution.is_some());
    let before_recall = first.session().get_leaf_id().await.unwrap();
    assert!(first.recall_last_user_message(first.state().messages[0].timestamp()).await.is_err());
    assert!(first.compact_now().await.is_err());
    assert_eq!(first.session().get_leaf_id().await.unwrap(), before_recall);
    assert!(first.prompt("rejected concurrent prompt", None).await.is_err());
    assert_eq!(first.request_execution_id(), active_execution);

    // Creating another session cannot stop the original run or release its lease.
    let second = runtime.new_session(&env).await.unwrap();
    second
        .set_name(Some("second running session"))
        .await
        .unwrap();
    let second_metadata = second.metadata().clone();
    let second_run = tokio::spawn({
        let session = second.clone();
        async move { session.prompt("second prompt", None).await }
    });
    let (second_request, second_answer) =
        tokio::time::timeout(std::time::Duration::from_secs(10), pending.recv())
            .await
            .unwrap()
            .unwrap();
    assert!(first.is_running() && second.is_running());
    assert_eq!(first_request.session_id, first_metadata.id);
    assert_eq!(second_request.session_id, second_metadata.id);
    assert_ne!(
        first_request.request_id, second_request.request_id,
        "provider tool ids may collide across sessions"
    );

    let reopened = runtime
        .open_session(&env, first_metadata.clone())
        .await
        .unwrap();
    assert!(
        Arc::ptr_eq(&first, &reopened),
        "switching must reuse the live instance"
    );
    drop(reopened);
    let names = runtime.list_sessions_with_names(&env).await.unwrap();
    assert!(
        names
            .iter()
            .any(|(meta, name, _)| meta.id == second_metadata.id
                && name.as_deref() == Some("second running session"))
    );
    assert!(
        runtime
            .delete_session(&env, &second_metadata)
            .await
            .is_err(),
        "running background sessions cannot be deleted"
    );

    // Answering the second request only resumes the second session.
    second_answer.send(false).unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(10), second_run)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(first.is_running());
    assert!(!second.is_running());
    assert_eq!(
        first
            .state()
            .messages
            .iter()
            .filter_map(|m| m.as_user())
            .count(),
        1
    );
    assert_eq!(
        second
            .state()
            .messages
            .iter()
            .filter_map(|m| m.as_user())
            .count(),
        1
    );
    drop(second);
    runtime
        .delete_session(&env, &second_metadata)
        .await
        .unwrap();
    assert!(
        first.is_running(),
        "deleting an idle session must not stop another session"
    );

    first_answer.send(false).unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(10), first_run)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(!first.is_running());
    assert!(
        std::iter::from_fn(|| first_events.try_recv().ok())
            .any(|event| matches!(event, SessionEvent::Settled))
    );
}

#[tokio::test]
async fn abort_while_waiting_for_approval_finishes_without_an_answer() {
    use step_harness_providers::providers::faux::FauxContentBlock;
    use step_harness_providers::types::ToolCall;
    let (_paths, _env, services, registration) = services("abort-approval").await;
    registration.set_responses(vec![
        faux_assistant_message(
            vec![FauxContentBlock::ToolCall(ToolCall {
                id: "waiting-call".into(),
                name: "bash".into(),
                arguments: serde_json::from_value(json!({ "command": "echo hello" })).unwrap(),
                namespace: None,
                thought_signature: None,
            })],
            FauxAssistantMessageOptions::default(),
        )
        .into(),
    ]);
    let (requests, mut pending) = tokio::sync::mpsc::unbounded_channel();
    let session = Arc::new(
        AgentSession::create(
            Arc::new(services),
            Arc::new(WaitingApproval { requests }),
            approve_all_plans(),
        )
        .await
        .unwrap(),
    );
    let run = tokio::spawn({
        let session = session.clone();
        async move { session.prompt("please run", None).await }
    });
    let (_, answer) = tokio::time::timeout(std::time::Duration::from_secs(10), pending.recv())
        .await
        .unwrap()
        .unwrap();
    session.abort();
    tokio::time::timeout(std::time::Duration::from_secs(10), run)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(!session.is_running());
    assert!(
        answer.is_closed(),
        "cancelled approval futures must release their receiver"
    );
    let timings = session.execution_timings().await.unwrap();
    assert_eq!(timings.len(), 1);
    assert_timing_partition(&timings[0]);
}

#[derive(Default)]
struct MetricsCollector(std::sync::Mutex<Vec<step_harness_providers::request_metrics::RequestRecord>>, std::sync::Mutex<Vec<step_harness_providers::request_metrics::ExecutionRecord>>);
impl step_harness_providers::request_metrics::RequestObserver for MetricsCollector {
    fn record(&self,record:step_harness_providers::request_metrics::RequestRecord){self.0.lock().unwrap().push(record);}
    fn execution(&self,record:step_harness_providers::request_metrics::ExecutionRecord){self.1.lock().unwrap().push(record);}
}
struct MetricsProbe { inner:Arc<dyn step_harness_providers::models::ProviderStreams> }
#[async_trait::async_trait]
impl step_harness_providers::models::ProviderStreams for MetricsProbe {
    fn stream(&self,model:&step_harness_providers::types::Model,context:&step_harness_providers::types::Context,options:&step_harness_providers::types::StreamOptions)->step_harness_providers::AssistantMessageEventStream {
        if let Some(metrics)=&options.base.request_metrics {metrics.begin();metrics.finish("success",Some(200),None);}
        self.inner.stream(model,context,options)
    }
    fn stream_simple(&self,model:&step_harness_providers::types::Model,context:&step_harness_providers::types::Context,options:&step_harness_providers::types::SimpleStreamOptions)->step_harness_providers::AssistantMessageEventStream {
        if let Some(metrics)=&options.base.base.request_metrics {metrics.begin();metrics.finish("success",Some(200),None);}
        self.inner.stream_simple(model,context,options)
    }
}
#[tokio::test]
async fn request_context_groups_tool_iterations_and_isolates_parallel_sessions() {
    use step_harness_providers::providers::faux::FauxContentBlock;
    use step_harness_providers::types::ToolCall;
    let registration=faux_registration("metrics-scopes");
    let original=step_harness_providers::compat::get_api_provider(registration.api()).unwrap();
    step_harness_providers::compat::register_api_provider(registration.api(),Arc::new(MetricsProbe{inner:original}),None);
    let paths=isolated_paths("metrics-scopes");write_faux_models(&paths,registration.api(),registration.provider_id());
    let env:Arc<dyn ExecutionEnv>=Arc::new(NodeExecutionEnv::new(Default::default()));
    let (mut services,_)=SessionServices::create(".",paths,&env,true).await;
    services.name_completer=Some(Arc::new(common::SkipNameCompleter));
    let collector=Arc::new(MetricsCollector::default());services.request_observer=Some(collector.clone());let services=Arc::new(services);
    let first=AgentSession::create(services.clone(),Arc::new(AutoDeny),approve_all_plans()).await.unwrap();
    let second=AgentSession::create(services,Arc::new(AutoDeny),approve_all_plans()).await.unwrap();
    first.set_name(Some("first".into())).await.unwrap();second.set_name(Some("second".into())).await.unwrap();
    registration.set_responses(vec![
        faux_assistant_message(vec![FauxContentBlock::ToolCall(ToolCall{id:"tool".into(),name:"bash".into(),arguments:serde_json::from_value(json!({"command":"echo metrics"})).unwrap(),namespace:None,thought_signature:None})],Default::default()).into(),
        faux_assistant_message("after tool",Default::default()).into(),
        faux_assistant_message("first next",Default::default()).into(),
        faux_assistant_message("second",Default::default()).into(),
    ]);
    first.prompt("tool round",None).await.unwrap();let previous=first.request_execution_id().unwrap();
    let (a,b)=tokio::join!(first.prompt("next round",None),second.prompt("parallel",None));a.unwrap();b.unwrap();
    let records=collector.0.lock().unwrap();let requests:Vec<_>=records.iter().filter(|r|r.status=="success").collect();assert_eq!(requests.len(),4);
    let first_requests:Vec<_>=requests.iter().filter(|r|r.context.session_id==first.metadata().id).collect();assert_eq!(first_requests.len(),3);
    assert_eq!(first_requests[0].context.execution_id.as_deref(),Some(previous.as_str()));assert_eq!(first_requests[1].context.execution_id.as_deref(),Some(previous.as_str()));
    assert_ne!(first.request_execution_id(),Some(previous));assert_eq!(first_requests[2].context.execution_id,first.request_execution_id());
    let second_request=requests.iter().find(|r|r.context.session_id==second.metadata().id).unwrap();assert_eq!(second_request.context.execution_id,second.request_execution_id());assert_ne!(first.request_execution_id(),second.request_execution_id());
    assert!(requests.iter().all(|r|r.context.purpose=="conversation"));
    let executions = collector.1.lock().unwrap();
    let completed: Vec<_> = executions.iter().filter(|execution| execution.status == "success").collect();
    assert_eq!(executions.iter().filter(|execution| execution.status == "running").count(), 3);
    assert_eq!(completed.len(), 3);
    for execution in completed {
        assert!(execution.ended_at.unwrap() >= execution.started_at);
        assert!(execution.duration_ms.is_some());
        assert!(requests.iter().any(|request| request.context.execution_id.as_deref() == Some(execution.id.as_str()) && request.context.session_id == execution.session_id));
    }
}


fn assert_timing_partition(timing: &step_harness_coding_agent::session::ExecutionTiming) {
    let parts = [timing.tool_duration_ms, timing.api_duration_ms, timing.approval_wait_duration_ms,
        timing.retry_wait_duration_ms, timing.other_duration_ms];
    assert!(parts.iter().all(Option::is_some), "new execution must have complete timing: {timing:?}");
    assert_eq!(parts.into_iter().map(Option::unwrap).sum::<u64>(), timing.duration_ms);
}

struct DelayedApproval(bool);
#[async_trait::async_trait]
impl ApprovalChannel for DelayedApproval {
    async fn confirm(&self, _: &ApprovalRequest) -> bool {
        tokio::time::sleep(std::time::Duration::from_millis(40)).await;
        self.0
    }
}

#[tokio::test]
async fn execution_timings_separate_api_approval_and_parallel_tools_across_rounds() {
    use step_harness_providers::providers::faux::{FauxContentBlock, FauxResponseStep};
    use step_harness_providers::types::ToolCall;
    for allowed in [false, true] {
        let name = if allowed { "timing-allow" } else { "timing-deny" };
        let (_paths, _env, services, registration) = services(name).await;
        registration.set_responses(vec![
            FauxResponseStep::Factory(Arc::new(|_, _, _, _| Box::pin(async {
                // Simulate an HTTP-layer backoff inside a provider producer task.
                let retry = step_harness_providers::execution_timing::enter(
                    step_harness_providers::execution_timing::Phase::Retry);
                tokio::time::sleep(std::time::Duration::from_millis(30)).await;
                drop(retry);
                tokio::time::sleep(std::time::Duration::from_millis(30)).await;
                Ok(faux_assistant_message((0..2).map(|index| FauxContentBlock::ToolCall(ToolCall {
                    id: format!("timed-{index}"), name: "bash".into(),
                    arguments: serde_json::from_value(json!({"command": "echo timing"})).unwrap(),
                    namespace: None, thought_signature: None,
                })).collect::<Vec<_>>(), Default::default()))
            }))),
            faux_assistant_message("done", Default::default()).into(),
            faux_assistant_message("next round", Default::default()).into(),
        ]);
        let session = AgentSession::create(Arc::new(services), Arc::new(DelayedApproval(allowed)), approve_all_plans()).await.unwrap();
        session.prompt("run tools", None).await.unwrap();
        let timing = session.execution_timings().await.unwrap().remove(0);
        assert_timing_partition(&timing);
        assert!(timing.api_duration_ms.unwrap() >= 25, "{timing:?}");
        assert!(timing.retry_wait_duration_ms.unwrap() >= 25, "provider context must be inherited: {timing:?}");
        assert!(timing.approval_wait_duration_ms.unwrap() >= 70, "{timing:?}");
        session.prompt("next", None).await.unwrap();
        let timings = session.execution_timings().await.unwrap();
        assert_eq!(timings.len(), 2);
        assert_timing_partition(&timings[1]);
        assert_eq!(timings[1].approval_wait_duration_ms, Some(0));
        assert_eq!(timings[1].retry_wait_duration_ms, Some(0));
        assert_eq!(timings[1].tool_duration_ms, Some(0));
    }
}


#[tokio::test]
async fn execution_timings_cover_session_retry_and_terminal_failure() {
    let (_paths, _env, services, registration) = services("timing-session-retry").await;
    services.settings.apply_patch(json!({"retry": {"enabled": true, "maxRetries": 1, "baseDelayMs": 40}}));
    let failed = faux_assistant_message("", FauxAssistantMessageOptions {
        stop_reason: Some(StopReason::Error), error_message: Some("503 temporarily unavailable".into()),
        ..Default::default()
    });
    registration.set_responses(vec![failed.clone().into(), faux_assistant_message("recovered", Default::default()).into()]);
    let session = AgentSession::create(Arc::new(services), approve_all(), approve_all_plans()).await.unwrap();
    session.prompt("retry", None).await.unwrap();
    let first = session.execution_timings().await.unwrap().remove(0);
    assert_timing_partition(&first);
    assert!(first.retry_wait_duration_ms.unwrap() >= 35, "{first:?}");
    registration.set_responses(vec![failed.clone().into(), failed.into()]);
    session.prompt("fail after retry", None).await.unwrap();
    let timings = session.execution_timings().await.unwrap();
    assert_eq!(timings.len(), 2);
    assert_timing_partition(&timings[1]);
    assert!(timings[1].retry_wait_duration_ms.unwrap() >= 35, "{:?}", timings[1]);
}


#[tokio::test]
async fn automatic_compaction_counts_requests_and_backoff_in_the_same_execution() {
    use step_harness_providers::providers::faux::FauxResponseStep;
    let (_paths, _env, services, registration) = services("timing-compaction").await;
    services.settings.apply_patch(json!({"compaction": {"enabled": true, "reserveTokens": 128000, "keepRecentTokens": 0}}));
    registration.set_responses(vec![
        faux_assistant_message("completed response", Default::default()).into(),
        faux_assistant_message("", FauxAssistantMessageOptions {
            stop_reason: Some(StopReason::Error), error_message: Some("503 unavailable".into()),
            ..Default::default()
        }).into(),
        FauxResponseStep::Factory(Arc::new(|_, _, _, _| Box::pin(async {
            tokio::time::sleep(std::time::Duration::from_millis(40)).await;
            Ok(faux_assistant_message("summary of the task", Default::default()))
        }))),
    ]);
    let session = AgentSession::create(Arc::new(services), approve_all(), approve_all_plans()).await.unwrap();
    let mut events = session.subscribe();
    session.prompt("summarize this task after answering", None).await.unwrap();
    let timing = session.execution_timings().await.unwrap().remove(0);
    assert_timing_partition(&timing);
    assert!(timing.api_duration_ms.unwrap() >= 35, "compression request must count as API: {timing:?}");
    assert!(timing.retry_wait_duration_ms.unwrap() >= 900, "compression backoff must be separate: {timing:?}");
    assert_eq!(timing.tool_duration_ms, Some(0));
    assert_eq!(timing.approval_wait_duration_ms, Some(0));
    let mut compacted = false;
    while let Ok(event) = events.try_recv() {
        if matches!(event, SessionEvent::CompactionEnd { error_message: None, .. }) { compacted = true; }
    }
    assert!(compacted);
}
