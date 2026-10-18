//! End-to-end tests for the agent runtime: prompt flow, tool execution, hooks,
//! and queueing.

use std::sync::Arc;
use std::sync::Mutex;

use serde_json::Value as Json;
use step_harness_agent_core::agent::{Agent, AgentError, AgentListener};
use step_harness_agent_core::agent_loop::run_agent_loop;
use step_harness_agent_core::harness::env::NodeExecutionEnv;
use step_harness_agent_core::harness::tools::{
    create_bash_tool_with_env, create_write_tool_with_env,
};
use step_harness_agent_core::types::*;
use step_harness_providers::event::{
    AssistantMessageEvent, AssistantMessageEventStream, DoneReason, assistant_message_event_stream,
};
use step_harness_providers::types::*;
use tokio_util::sync::CancellationToken;

struct PendingListener;

#[async_trait::async_trait]
impl AgentListener for PendingListener {
    async fn on_event(&self, _: &AgentEvent, _: &CancellationToken) {
        std::future::pending::<()>().await;
    }
}

#[tokio::test]
async fn dropping_a_prompt_releases_the_running_slot() {
    let agent = agent_with(Arc::new(NoopStreamFn), vec![]);
    let subscription = agent.subscribe(Arc::new(PendingListener));
    assert!(
        tokio::time::timeout(
            std::time::Duration::from_millis(20),
            agent.prompt_text("first", None)
        )
        .await
        .is_err()
    );
    assert!(!agent.state().is_streaming);
    assert!(agent.signal().is_none());
    agent.reset().unwrap();
    agent.unsubscribe(subscription);
    agent.prompt_text("second", None).await.unwrap();
}

#[tokio::test]
async fn abort_unblocks_an_uncooperative_listener() {
    let agent = Arc::new(agent_with(Arc::new(NoopStreamFn), vec![]));
    agent.subscribe(Arc::new(PendingListener));
    let running = agent.clone();
    let task = tokio::spawn(async move { running.prompt_text("test", None).await });
    while agent.signal().is_none() {
        tokio::task::yield_now().await;
    }
    agent.abort();
    tokio::time::timeout(std::time::Duration::from_secs(1), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(!agent.state().is_streaming);
    assert!(agent.signal().is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_prompt_reservation_is_atomic() {
    let agent = Arc::new(agent_with(Arc::new(NoopStreamFn), vec![]));
    agent.subscribe(Arc::new(PendingListener));
    let barrier = Arc::new(tokio::sync::Barrier::new(17));
    let mut tasks = Vec::new();
    for _ in 0..16 {
        let agent = agent.clone();
        let barrier = barrier.clone();
        tasks.push(tokio::spawn(async move {
            barrier.wait().await;
            tokio::time::timeout(
                std::time::Duration::from_millis(100),
                agent.prompt_text("test", None),
            )
            .await
        }));
    }
    barrier.wait().await;
    let mut started = 0;
    for task in tasks {
        match task.await.unwrap() {
            Err(_) => started += 1,
            Ok(Err(AgentError::AlreadyRunning)) => {}
            result => panic!("unexpected prompt result: {result:?}"),
        }
    }
    assert_eq!(started, 1);
    agent.reset().unwrap();
}

/// A scripted stream function: each call pops the next scripted turn.
struct ScriptedStreamFn {
    turns: Mutex<Vec<AssistantMessage>>,
}

impl ScriptedStreamFn {
    fn new(turns: Vec<AssistantMessage>) -> Arc<Self> {
        Arc::new(ScriptedStreamFn {
            turns: Mutex::new(turns),
        })
    }
}

#[async_trait::async_trait]
impl StreamFn for ScriptedStreamFn {
    async fn stream(
        &self,
        _model: &Model,
        _context: &Context,
        _options: &SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        let message = self
            .turns
            .lock()
            .unwrap()
            .pop()
            .unwrap_or_else(|| error_message("script exhausted"));
        let (mut sender, stream) = assistant_message_event_stream();
        sender
            .push(AssistantMessageEvent::Start {
                partial: message.clone(),
            })
            .await;
        sender
            .push(AssistantMessageEvent::Done {
                reason: DoneReason::Stop,
                message,
            })
            .await;
        stream
    }
}

fn error_message(text: &str) -> AssistantMessage {
    AssistantMessage {
        content: vec![AssistantContent::Text(TextContent {
            text: text.to_owned(),
            text_signature: None,
        })],
        api: Api::OpenaiCompletions,
        provider: ProviderId::new("test"),
        model: "m".to_owned(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason: StopReason::Error,
        deferred: None,
        error_message: Some(text.to_owned()),
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 0,
    }
}

fn model() -> Model {
    Model {
        id: "test-model".to_owned(),
        name: "Test".to_owned(),
        api: Api::OpenaiCompletions,
        provider: ProviderId::new("test"),
        base_url: "http://localhost".to_owned(),
        reasoning: false,
        thinking_level_map: None,
        input: vec![InputModality::Text],
        cost: Default::default(),
        context_window: 128_000,
        max_tokens: 8192,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

fn text_assistant(text: &str) -> AssistantMessage {
    AssistantMessage {
        content: vec![AssistantContent::Text(TextContent {
            text: text.to_owned(),
            text_signature: None,
        })],
        api: Api::OpenaiCompletions,
        provider: ProviderId::new("test"),
        model: "test-model".to_owned(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: Usage {
            input: 10,
            output: 5,
            total_tokens: 15,
            ..Default::default()
        },
        stop_reason: StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 0,
    }
}

fn tool_call_assistant(call_id: &str, name: &str, arguments: Json) -> AssistantMessage {
    AssistantMessage {
        content: vec![AssistantContent::ToolCall(ToolCall {
            id: call_id.to_owned(),
            name: name.to_owned(),
            arguments: arguments.as_object().cloned().unwrap_or_default(),
            thought_signature: None,
            namespace: None,
        })],
        api: Api::OpenaiCompletions,
        provider: ProviderId::new("test"),
        model: "test-model".to_owned(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason: StopReason::ToolUse,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 0,
    }
}

/// Records every event the agent emits.
struct RecordingListener {
    events: Mutex<Vec<String>>,
}

impl RecordingListener {
    fn new() -> Arc<Self> {
        Arc::new(RecordingListener {
            events: Mutex::new(Vec::new()),
        })
    }

    fn kinds(&self) -> Vec<String> {
        self.events.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl AgentListener for RecordingListener {
    async fn on_event(&self, event: &AgentEvent, _signal: &CancellationToken) {
        let kind = match event {
            AgentEvent::AgentStart => "agent_start",
            AgentEvent::AgentEnd { .. } => "agent_end",
            AgentEvent::TurnStart => "turn_start",
            AgentEvent::TurnEnd { .. } => "turn_end",
            AgentEvent::MessageStart { .. } => "message_start",
            AgentEvent::MessageUpdate { .. } => "message_update",
            AgentEvent::MessageEnd { .. } => "message_end",
            AgentEvent::ToolExecutionStart { .. } => "tool_execution_start",
            AgentEvent::ToolExecutionUpdate { .. } => "tool_execution_update",
            AgentEvent::ToolExecutionEnd { .. } => "tool_execution_end",
        };
        self.events.lock().unwrap().push(kind.to_owned());
    }
}

fn agent_with(stream_fn: Arc<dyn StreamFn>, tools: Vec<AgentTool>) -> Agent {
    Agent::new(AgentOptions {
        initial_state: Some(InitialAgentState {
            system_prompt: Some("You are a test agent.".to_owned()),
            model: Some(model()),
            thinking_level: Some(ModelThinkingLevel::Off),
            tools: Some(tools),
            messages: None,
        }),
        convert_to_llm: None,
        transform_context: None,
        stream_fn,
        get_api_key: None,
        on_payload: None,
        on_response: None,
        before_tool_call: None,
        after_tool_call: None,
        transform_tool_result: None,
        should_stop_after_turn: None,
        prepare_next_turn: None,
        prepare_next_turn_with_context: None,
        steering_mode: QueueMode::OneAtATime,
        follow_up_mode: QueueMode::OneAtATime,
        session_id: None,
        thinking_budgets: None,
        transport: Transport::Auto,
        max_retry_delay_ms: None,
        tool_execution: ToolExecutionMode::Parallel,
    })
}

#[tokio::test]
async fn prompt_without_tools_runs_one_turn() {
    let listener = RecordingListener::new();
    let agent = agent_with(
        ScriptedStreamFn::new(vec![text_assistant("hello there")]),
        Vec::new(),
    );
    agent.subscribe(listener.clone());

    agent.prompt_text("hi", None).await.unwrap();
    agent.wait_for_idle().await;

    assert_eq!(
        listener.kinds(),
        vec![
            "agent_start",
            "turn_start",
            "message_start",
            "message_end",
            "message_start",
            "message_end",
            "turn_end",
            "agent_end",
        ]
    );

    let state = agent.state();
    assert_eq!(state.messages.len(), 2);
    assert!(matches!(state.messages[0], AgentMessage::User(_)));
    assert!(matches!(state.messages[1], AgentMessage::Assistant(_)));
    assert!(!state.is_streaming);
}

#[tokio::test]
async fn tool_calls_execute_and_feed_the_next_turn() {
    let env = Arc::new(NodeExecutionEnv::new(Default::default()));
    let dir = std::env::temp_dir().join(format!("agent-core-e2e-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let target = dir.join("out.txt");

    let listener = RecordingListener::new();
    let agent = agent_with(
        ScriptedStreamFn::new(vec![
            // Second turn: the model answers after seeing the tool result.
            text_assistant("done"),
            // First turn: ask the tool to write a file.
            tool_call_assistant(
                "call-1",
                "write",
                serde_json::json!({"path": target.display().to_string(), "content": "written"}),
            ),
        ]),
        vec![create_write_tool_with_env(env)],
    );
    agent.subscribe(listener.clone());

    agent.prompt_text("write the file", None).await.unwrap();
    agent.wait_for_idle().await;

    assert_eq!(std::fs::read_to_string(&target).unwrap(), "written");

    let kinds = listener.kinds();
    // Two turns: prompt → tool call → tool result → final answer.
    assert_eq!(kinds.iter().filter(|kind| *kind == "turn_start").count(), 2);
    assert_eq!(
        kinds
            .iter()
            .filter(|kind| *kind == "tool_execution_start")
            .count(),
        1
    );
    assert_eq!(
        kinds
            .iter()
            .filter(|kind| *kind == "tool_execution_end")
            .count(),
        1
    );

    let state = agent.state();
    // user, assistant (tool call), tool result, assistant (final)
    assert_eq!(state.messages.len(), 4);
    assert!(matches!(state.messages[2], AgentMessage::ToolResult(_)));
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn unknown_tools_become_error_results() {
    let listener = RecordingListener::new();
    let agent = agent_with(
        ScriptedStreamFn::new(vec![
            text_assistant("recovered"),
            tool_call_assistant("call-1", "nope", serde_json::json!({})),
        ]),
        Vec::new(),
    );
    agent.subscribe(listener.clone());

    agent
        .prompt_text("call a missing tool", None)
        .await
        .unwrap();
    agent.wait_for_idle().await;

    let state = agent.state();
    let AgentMessage::ToolResult(result) = &state.messages[2] else {
        panic!("expected a tool result");
    };
    assert!(result.is_error);
    assert!(result.tool_name == "nope");
    // The loop continues after the error result.
    assert_eq!(state.messages.len(), 4);
}

fn malformed_search_turn() -> AssistantMessage {
    let mut message = text_assistant(
        "<tool_call><function=search_web><query>广州天气</parameter></function></tool_call>",
    );
    message
        .content
        .extend(tool_call_assistant("broken", "search_web", serde_json::json!({})).content);
    message.stop_reason = StopReason::Error;
    message.error_message = Some(
        "Invalid tool argument JSON: EOF while parsing an object at line 1 column 1".to_owned(),
    );
    message
}

struct SearchRecorder(Mutex<Vec<String>>);

#[async_trait::async_trait]
impl ToolExecutor for SearchRecorder {
    async fn execute(
        &self,
        _: &str,
        params: &serde_json::Map<String, Json>,
        _: Option<&CancellationToken>,
        _: Option<&ToolUpdateCallback>,
    ) -> Result<AgentToolResult, ToolError> {
        self.0
            .lock()
            .unwrap()
            .push(params["query"].as_str().unwrap().to_owned());
        Ok(AgentToolResult::text("mock search result", Json::Null))
    }
}

fn recorded_search_tool(recorder: Arc<SearchRecorder>) -> AgentTool {
    AgentTool {
        name: "search_web".to_owned(),
        label: "search_web".to_owned(),
        description: "Search".to_owned(),
        parameters: serde_json::json!({"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}),
        execution_mode: None,
        prepare_arguments: None,
        executor: recorder,
    }
}

#[tokio::test]
async fn malformed_search_markup_retries_and_executes_only_valid_arguments() {
    let recorder = Arc::new(SearchRecorder(Mutex::new(Vec::new())));
    let stream = ScriptedStreamFn::new(vec![
        text_assistant("done"),
        tool_call_assistant(
            "valid",
            "search_web",
            serde_json::json!({"query":"广州天气"}),
        ),
        malformed_search_turn(),
    ]);
    let agent = agent_with(stream, vec![recorded_search_tool(recorder.clone())]);
    agent.prompt_text("search weather", None).await.unwrap();
    assert_eq!(*recorder.0.lock().unwrap(), vec!["广州天气"]);
    let state = agent.state();
    // Failed attempts remain in the event history for observability, but the
    // retried turn must finish normally and execute only the validated call.
    assert_eq!(
        state
            .messages
            .last()
            .unwrap()
            .as_assistant()
            .unwrap()
            .stop_reason,
        StopReason::Stop
    );
    assert!(
        state
            .messages
            .iter()
            .any(|message| message.as_tool_result().is_some())
    );
}

#[tokio::test]
async fn malformed_search_markup_stops_at_the_retry_limit_without_execution() {
    let recorder = Arc::new(SearchRecorder(Mutex::new(Vec::new())));
    let stream = ScriptedStreamFn::new(vec![malformed_search_turn(); 4]);
    let agent = agent_with(stream.clone(), vec![recorded_search_tool(recorder.clone())]);
    agent.prompt_text("search weather", None).await.unwrap();
    assert!(recorder.0.lock().unwrap().is_empty());
    assert_eq!(stream.turns.lock().unwrap().len(), 1);
    assert!(
        agent
            .state()
            .messages
            .last()
            .unwrap()
            .as_assistant()
            .is_some_and(|message| message.stop_reason == StopReason::Error)
    );
}

#[tokio::test]
async fn invalid_tool_json_without_markup_retries_and_executes_only_valid_arguments() {
    let recorder = Arc::new(SearchRecorder(Mutex::new(Vec::new())));
    let mut malformed = tool_call_assistant(
        "broken",
        "search_web",
        serde_json::json!({"query":"partial query"}),
    );
    malformed.stop_reason = StopReason::Error;
    malformed.error_message =
        Some("Invalid tool argument JSON: expected `:` at line 1 column 22".to_owned());
    let stream = ScriptedStreamFn::new(vec![
        text_assistant("done"),
        tool_call_assistant(
            "valid",
            "search_web",
            serde_json::json!({"query":"weather"}),
        ),
        malformed,
    ]);
    let agent = agent_with(stream.clone(), vec![recorded_search_tool(recorder.clone())]);
    agent.prompt_text("search weather", None).await.unwrap();
    assert_eq!(*recorder.0.lock().unwrap(), vec!["weather"]);
    assert!(stream.turns.lock().unwrap().is_empty());
    let state = agent.state();
    assert!(state.error_message.is_none());
    assert_eq!(
        state
            .messages
            .last()
            .unwrap()
            .as_assistant()
            .unwrap()
            .stop_reason,
        StopReason::Stop
    );
}

#[tokio::test]
async fn invalid_tool_json_without_content_stops_at_retry_limit() {
    let mut malformed =
        error_message("Invalid tool argument JSON: expected `:` at line 1 column 22");
    malformed.content.clear();
    let stream = ScriptedStreamFn::new(vec![malformed; 4]);
    let recorder = Arc::new(SearchRecorder(Mutex::new(Vec::new())));
    let agent = agent_with(stream.clone(), vec![recorded_search_tool(recorder.clone())]);
    agent.prompt_text("search weather", None).await.unwrap();
    assert_eq!(stream.turns.lock().unwrap().len(), 1);
    assert!(recorder.0.lock().unwrap().is_empty());
    assert_eq!(
        agent.state().error_message.as_deref(),
        Some("Invalid tool argument JSON: expected `:` at line 1 column 22")
    );
}

#[tokio::test]
async fn unrelated_provider_errors_are_not_retried() {
    let stream = ScriptedStreamFn::new(vec![text_assistant("unused"), error_message("HTTP 403")]);
    let agent = agent_with(stream.clone(), Vec::new());
    agent.prompt_text("hello", None).await.unwrap();
    assert_eq!(stream.turns.lock().unwrap().len(), 1);
    assert_eq!(agent.state().error_message.as_deref(), Some("HTTP 403"));
}

#[tokio::test]
async fn aborted_runs_report_an_aborted_assistant_message() {
    let stream_fn = Arc::new(HangingStreamFn);
    let agent = Arc::new(agent_with(stream_fn, Vec::new()));
    let listener = RecordingListener::new();
    agent.subscribe(listener.clone());

    let handle = {
        let agent = agent.clone();
        tokio::spawn(async move {
            agent.prompt_text("hang", None).await.unwrap();
        })
    };
    // Give the run a moment to start, then abort it.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    agent.abort();
    handle.await.unwrap();
    agent.wait_for_idle().await;

    assert!(!agent.state().is_streaming);
    assert!(agent.state().error_message.is_some());
}

/// A stream that hangs until the abort signal fires, then ends with an error —
/// mirroring how real provider streams honor the caller's cancellation token.
struct HangingStreamFn;

#[async_trait::async_trait]
impl StreamFn for HangingStreamFn {
    async fn stream(
        &self,
        model: &Model,
        _context: &Context,
        options: &SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        let signal = options.base.base.signal.clone();
        step_harness_providers::event::lazy_stream(model, move || async move {
            if let Some(signal) = signal {
                signal.cancelled().await;
            }
            Err::<AssistantMessageEventStream, _>("aborted")
        })
    }
}

#[tokio::test]
async fn steering_messages_are_injected_before_the_first_response() {
    let agent = agent_with(
        ScriptedStreamFn::new(vec![text_assistant("answer")]),
        Vec::new(),
    );
    // A steering message queued before the run is drained at loop start and
    // injected ahead of the first assistant response.
    agent.steer(AgentMessage::user("steer me", 1));

    agent.prompt_text("go", None).await.unwrap();
    agent.wait_for_idle().await;

    let messages = agent.messages();
    let roles: Vec<&str> = messages.iter().map(AgentMessage::role).collect();
    assert_eq!(roles, vec!["user", "user", "assistant"]);
    assert!(matches!(messages[1], AgentMessage::User(_)));
}

#[tokio::test]
async fn follow_up_messages_start_another_turn() {
    let agent = agent_with(
        ScriptedStreamFn::new(vec![
            text_assistant("after follow-up"),
            text_assistant("initial"),
        ]),
        Vec::new(),
    );

    agent.prompt_text("go", None).await.unwrap();
    agent.wait_for_idle().await;
    assert_eq!(agent.messages().len(), 2);

    agent.follow_up(AgentMessage::user("and then?", 2));
    agent.continue_run().await.unwrap();
    agent.wait_for_idle().await;
    assert_eq!(agent.messages().len(), 4);
}

#[tokio::test]
async fn truncated_tool_calls_fail_instead_of_executing() {
    // A "length" stop means every tool call may carry truncated arguments.
    let mut truncated =
        tool_call_assistant("call-1", "bash", serde_json::json!({"command": "echo hi"}));
    truncated.stop_reason = StopReason::Length;

    let env = Arc::new(NodeExecutionEnv::new(Default::default()));
    let agent = agent_with(
        ScriptedStreamFn::new(vec![text_assistant("ok"), truncated]),
        vec![create_bash_tool_with_env(env)],
    );

    agent.prompt_text("run something", None).await.unwrap();
    agent.wait_for_idle().await;

    let AgentMessage::ToolResult(result) = &agent.messages()[2] else {
        panic!("expected a tool result");
    };
    assert!(result.is_error);
    let text = match &result.content[0] {
        UserContentBlock::Text(text) => &text.text,
        _ => panic!("expected text"),
    };
    assert!(text.contains("output token limit"), "message was: {text}");
}

#[tokio::test]
async fn concurrent_prompts_are_rejected() {
    let agent = Arc::new(agent_with(Arc::new(HangingStreamFn), Vec::new()));
    let handle = {
        let agent = agent.clone();
        tokio::spawn(async move {
            agent.prompt_text("hang", None).await.unwrap();
        })
    };
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    let error = agent.prompt_text("again", None).await.unwrap_err();
    assert!(matches!(error, AgentError::AlreadyRunning));
    agent.abort();
    handle.await.unwrap();
}

#[tokio::test]
async fn run_agent_loop_returns_the_new_messages() {
    let stream_fn = ScriptedStreamFn::new(vec![text_assistant("hi")]);
    let context = AgentContext {
        system_prompt: "test".to_owned(),
        messages: Vec::new(),
        tools: None,
    };
    let config = AgentLoopConfig {
        model: model(),
        tool_call_leak_retries: None,
        convert_to_llm: Arc::new(DefaultConvertToLlm),
        transform_context: None,
        get_api_key: None,
        should_stop_after_turn: None,
        prepare_next_turn: None,
        get_steering_messages: None,
        get_follow_up_messages: None,
        tool_execution: ToolExecutionMode::Parallel,
        before_tool_call: None,
        after_tool_call: None,
        transform_tool_result: None,
        stream_options: SimpleStreamOptions::default(),
    };
    let messages = run_agent_loop(
        vec![AgentMessage::user("hello", 1)],
        context,
        config,
        None,
        stream_fn.as_ref(),
        &NoopSink,
    )
    .await;
    assert_eq!(messages.len(), 2);
}

struct NoopSink;

#[async_trait::async_trait]
impl AgentEventSink for NoopSink {
    async fn emit(&self, _event: AgentEvent) {}
}
