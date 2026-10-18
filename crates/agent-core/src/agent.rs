//! The stateful [`Agent`]: transcript ownership, lifecycle events, tool
//! execution, and steering/follow-up queues.
//!
//! Ported from `@step-harness/agent-core` `src/agent.ts`.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use futures::future::FutureExt;

use step_harness_providers::types::{
    ImageContent, Model, ModelThinkingLevel, SimpleStreamOptions, TextContent, ThinkingBudgets,
    ToolResultMessage, Transport, UserContentBlock, UserMessage,
};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use crate::agent_loop::{run_agent_loop, run_agent_loop_continue};
use crate::types::{thinking_level_to_reasoning, *};

/// Errors returned by [`Agent`] control methods.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error(
        "Agent is already processing a prompt. Use steer() or followUp() to queue messages, or wait for completion."
    )]
    AlreadyRunning,
    #[error("No messages to continue from")]
    NoMessagesToContinue,
    #[error("Cannot continue from message role: assistant")]
    CannotContinueFromAssistant,
    #[error("Agent is already processing. Wait for completion before resetting.")]
    ResetWhileRunning,
}

/// Listener for agent lifecycle events. Listener futures are awaited in
/// registration order and count toward the current run's settlement.
#[async_trait::async_trait]
pub trait AgentListener: Send + Sync {
    async fn on_event(&self, event: &AgentEvent, signal: &CancellationToken);
}

/// Handle returned by [`Agent::subscribe`]; pass it to [`Agent::unsubscribe`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SubscriptionId(u64);

struct ActiveRun {
    token: CancellationToken,
    done: watch::Receiver<bool>,
}

/// Owns the running slot, including when the invocation future is dropped.
struct RunGuard<'a> {
    agent: &'a Agent,
    token: CancellationToken,
    done: watch::Sender<bool>,
}

impl Drop for RunGuard<'_> {
    fn drop(&mut self) {
        self.token.cancel();
        self.agent.finish_run();
        let _ = self.done.send(true);
    }
}

#[derive(Default)]
struct PendingMessageQueue {
    messages: VecDeque<AgentMessage>,
    mode: QueueMode,
}

impl PendingMessageQueue {
    fn enqueue(&mut self, message: AgentMessage) {
        self.messages.push_back(message);
    }

    fn has_items(&self) -> bool {
        !self.messages.is_empty()
    }

    fn drain(&mut self) -> Vec<AgentMessage> {
        match self.mode {
            QueueMode::All => self.messages.drain(..).collect(),
            QueueMode::OneAtATime => self.messages.pop_front().into_iter().collect(),
        }
    }

    fn clear(&mut self) {
        self.messages.clear();
    }
}

/// Steering queue hook that drains the agent's steering queue, optionally
/// skipping the first poll of a run.
struct SteeringHook {
    queue: Arc<Mutex<PendingMessageQueue>>,
    skip_first_poll: AtomicBool,
}

#[async_trait::async_trait]
impl GetSteeringMessages for SteeringHook {
    async fn get_steering_messages(&self) -> Vec<AgentMessage> {
        if self.skip_first_poll.swap(false, Ordering::SeqCst) {
            return Vec::new();
        }
        lock(&self.queue).drain()
    }
}

struct FollowUpHook {
    queue: Arc<Mutex<PendingMessageQueue>>,
}

#[async_trait::async_trait]
impl GetFollowUpMessages for FollowUpHook {
    async fn get_follow_up_messages(&self) -> Vec<AgentMessage> {
        lock(&self.queue).drain()
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().expect("agent mutex poisoned")
}

/// A stateful wrapper around the low-level agent loop.
///
/// The agent owns the current transcript, emits lifecycle events, executes
/// tools, and exposes queueing APIs for steering and follow-up messages.
pub struct Agent {
    state: Mutex<AgentState>,
    listeners: Mutex<BTreeMap<u64, Arc<dyn AgentListener>>>,
    next_subscription_id: Mutex<u64>,
    steering_queue: Arc<Mutex<PendingMessageQueue>>,
    follow_up_queue: Arc<Mutex<PendingMessageQueue>>,
    active_run: Mutex<Option<ActiveRun>>,

    pub convert_to_llm: Arc<dyn ConvertToLlm>,
    pub transform_context: Option<Arc<dyn TransformContext>>,
    pub stream_fn: Arc<dyn StreamFn>,
    pub get_api_key: Option<Arc<dyn GetApiKey>>,
    pub on_payload: Option<Arc<dyn step_harness_providers::types::PayloadObserver>>,
    pub on_response: Option<Arc<dyn step_harness_providers::types::ResponseObserver>>,
    pub before_tool_call: Option<Arc<dyn BeforeToolCall>>,
    pub after_tool_call: Option<Arc<dyn AfterToolCall>>,
    pub transform_tool_result: Option<Arc<dyn TransformToolResult>>,
    pub should_stop_after_turn: Option<Arc<dyn ShouldStopAfterTurn>>,
    pub prepare_next_turn: Option<Arc<dyn PrepareNextTurnSimple>>,
    pub prepare_next_turn_with_context: Option<Arc<dyn PrepareNextTurn>>,

    session_id: Mutex<Option<String>>,
    thinking_budgets: Mutex<Option<ThinkingBudgets>>,
    transport: Mutex<Transport>,
    max_retry_delay_ms: Mutex<Option<u64>>,
    tool_execution: Mutex<ToolExecutionMode>,
}

impl Agent {
    pub fn new(options: AgentOptions) -> Self {
        let initial = options.initial_state.unwrap_or_default();
        let mut state = AgentState::default();
        if let Some(system_prompt) = initial.system_prompt {
            state.system_prompt = system_prompt;
        }
        if let Some(model) = initial.model {
            state.model = model;
        }
        if let Some(thinking_level) = initial.thinking_level {
            state.thinking_level = thinking_level;
        }
        state.tools = initial.tools.unwrap_or_default();
        state.messages = initial.messages.unwrap_or_default();

        Agent {
            state: Mutex::new(state),
            listeners: Mutex::new(BTreeMap::new()),
            next_subscription_id: Mutex::new(0),
            steering_queue: Arc::new(Mutex::new(PendingMessageQueue {
                messages: VecDeque::new(),
                mode: options.steering_mode,
            })),
            follow_up_queue: Arc::new(Mutex::new(PendingMessageQueue {
                messages: VecDeque::new(),
                mode: options.follow_up_mode,
            })),
            active_run: Mutex::new(None),
            convert_to_llm: options
                .convert_to_llm
                .unwrap_or_else(|| Arc::new(DefaultConvertToLlm)),
            transform_context: options.transform_context,
            stream_fn: options.stream_fn,
            get_api_key: options.get_api_key,
            on_payload: options.on_payload,
            on_response: options.on_response,
            before_tool_call: options.before_tool_call,
            after_tool_call: options.after_tool_call,
            transform_tool_result: options.transform_tool_result,
            should_stop_after_turn: options.should_stop_after_turn,
            prepare_next_turn: options.prepare_next_turn,
            prepare_next_turn_with_context: options.prepare_next_turn_with_context,
            session_id: Mutex::new(options.session_id),
            thinking_budgets: Mutex::new(options.thinking_budgets),
            transport: Mutex::new(options.transport),
            max_retry_delay_ms: Mutex::new(options.max_retry_delay_ms),
            tool_execution: Mutex::new(options.tool_execution),
        }
    }

    /// Subscribe to agent lifecycle events.
    ///
    /// Listener futures are awaited in subscription order and are included in
    /// the current run's settlement. `agent_end` is the final emitted event for
    /// a run, but the agent does not become idle until all awaited listeners
    /// for that event have finished.
    pub fn subscribe(&self, listener: Arc<dyn AgentListener>) -> SubscriptionId {
        let mut next_id = lock(&self.next_subscription_id);
        let id = SubscriptionId(*next_id);
        *next_id += 1;
        lock(&self.listeners).insert(id.0, listener);
        id
    }

    /// Remove a previously registered listener.
    pub fn unsubscribe(&self, id: SubscriptionId) {
        lock(&self.listeners).remove(&id.0);
    }

    /// Snapshot of the current agent state.
    ///
    /// The TypeScript source returns a live view; Rust returns a clone. Use
    /// [`Agent::set_messages`] to replace the transcript.
    pub fn state(&self) -> AgentState {
        lock(&self.state).clone()
    }

    /// Append a message to the transcript, mirroring `agent.state.messages.push(...)`.
    pub fn push_message(&self, message: AgentMessage) {
        lock(&self.state).messages.push(message);
    }

    /// Replace the whole transcript.
    pub fn set_messages(&self, messages: Vec<AgentMessage>) {
        lock(&self.state).messages = messages;
    }

    /// Snapshot of the transcript.
    pub fn messages(&self) -> Vec<AgentMessage> {
        lock(&self.state).messages.clone()
    }

    /// Replace the system prompt used for future turns.
    pub fn set_system_prompt(&self, system_prompt: impl Into<String>) {
        lock(&self.state).system_prompt = system_prompt.into();
    }

    /// Replace the active model used for future turns.
    pub fn set_model(&self, model: Model) {
        lock(&self.state).model = model;
    }

    /// Replace the requested reasoning level for future turns.
    pub fn set_thinking_level(&self, thinking_level: ModelThinkingLevel) {
        lock(&self.state).thinking_level = thinking_level;
    }

    /// Replace the available tools.
    pub fn set_tools(&self, tools: Vec<AgentTool>) {
        lock(&self.state).tools = tools;
    }

    /// Controls how queued steering messages are drained.
    pub fn set_steering_mode(&self, mode: QueueMode) {
        lock(&self.steering_queue).mode = mode;
    }

    pub fn steering_mode(&self) -> QueueMode {
        lock(&self.steering_queue).mode
    }

    /// Controls how queued follow-up messages are drained.
    pub fn set_follow_up_mode(&self, mode: QueueMode) {
        lock(&self.follow_up_queue).mode = mode;
    }

    pub fn follow_up_mode(&self) -> QueueMode {
        lock(&self.follow_up_queue).mode
    }

    /// Queue a message to be injected after the current assistant turn finishes.
    pub fn steer(&self, message: AgentMessage) {
        lock(&self.steering_queue).enqueue(message);
    }

    /// Queue a message to run only after the agent would otherwise stop.
    pub fn follow_up(&self, message: AgentMessage) {
        lock(&self.follow_up_queue).enqueue(message);
    }

    /// Remove all queued steering messages.
    pub fn clear_steering_queue(&self) {
        lock(&self.steering_queue).clear();
    }

    /// Remove all queued follow-up messages.
    pub fn clear_follow_up_queue(&self) {
        lock(&self.follow_up_queue).clear();
    }

    /// Remove all queued steering and follow-up messages.
    pub fn clear_all_queues(&self) {
        self.clear_steering_queue();
        self.clear_follow_up_queue();
    }

    /// Returns true when either queue still contains pending messages.
    pub fn has_queued_messages(&self) -> bool {
        lock(&self.steering_queue).has_items() || lock(&self.follow_up_queue).has_items()
    }

    /// Active abort signal for the current run, if any.
    pub fn signal(&self) -> Option<CancellationToken> {
        lock(&self.active_run).as_ref().map(|run| run.token.clone())
    }

    /// Abort the current run, if one is active.
    pub fn abort(&self) {
        if let Some(run) = lock(&self.active_run).as_ref() {
            run.token.cancel();
        }
    }

    /// Resolve when the current run and all awaited event listeners have finished.
    pub async fn wait_for_idle(&self) {
        let mut done = {
            let run = lock(&self.active_run);
            run.as_ref().map(|run| run.done.clone())
        };
        let Some(done) = done.as_mut() else {
            return;
        };
        while !*done.borrow_and_update() {
            if done.changed().await.is_err() {
                break;
            }
        }
    }

    /// Clear transcript state, runtime state, and queued messages.
    pub fn reset(&self) -> Result<(), AgentError> {
        let active_run = lock(&self.active_run);
        if active_run.is_some() {
            return Err(AgentError::ResetWhileRunning);
        }
        let mut state = lock(&self.state);
        state.messages.clear();
        state.is_streaming = false;
        state.streaming_message = None;
        state.pending_tool_calls.clear();
        state.error_message = None;
        drop(state);
        self.clear_follow_up_queue();
        self.clear_steering_queue();
        Ok(())
    }

    /// Start a new prompt from a prepared input.
    pub async fn prompt(
        &self,
        input: PromptInput,
        images: Option<Vec<ImageContent>>,
    ) -> Result<(), AgentError> {
        let _run = self.begin_run()?;
        let messages = normalize_prompt_input(input, images);
        self.run_prompt_messages(messages, false).await;
        Ok(())
    }

    /// Start a new prompt from text, optionally with images.
    pub async fn prompt_text(
        &self,
        text: impl Into<String>,
        images: Option<Vec<ImageContent>>,
    ) -> Result<(), AgentError> {
        self.prompt(PromptInput::Text(text.into()), images).await
    }

    /// Continue from the current transcript. The last message must be a user or
    /// tool-result message.
    pub async fn continue_run(&self) -> Result<(), AgentError> {
        let _run = self.begin_run()?;

        let last_message = lock(&self.state).messages.last().cloned();
        let Some(last_message) = last_message else {
            return Err(AgentError::NoMessagesToContinue);
        };

        if matches!(last_message, AgentMessage::Assistant(_)) {
            let queued_steering = lock(&self.steering_queue).drain();
            if !queued_steering.is_empty() {
                self.run_prompt_messages(queued_steering, true).await;
                return Ok(());
            }

            let queued_follow_ups = lock(&self.follow_up_queue).drain();
            if !queued_follow_ups.is_empty() {
                self.run_prompt_messages(queued_follow_ups, false).await;
                return Ok(());
            }

            return Err(AgentError::CannotContinueFromAssistant);
        }

        self.run_continuation().await;
        Ok(())
    }

    /// Session identifier forwarded to providers for cache-aware backends.
    pub fn set_session_id(&self, session_id: Option<String>) {
        *lock(&self.session_id) = session_id;
    }

    pub fn session_id(&self) -> Option<String> {
        lock(&self.session_id).clone()
    }

    /// Optional per-level thinking token budgets forwarded to the stream function.
    pub fn set_thinking_budgets(&self, budgets: Option<ThinkingBudgets>) {
        *lock(&self.thinking_budgets) = budgets;
    }

    pub fn thinking_budgets(&self) -> Option<ThinkingBudgets> {
        *lock(&self.thinking_budgets)
    }

    /// Preferred transport forwarded to the stream function.
    pub fn set_transport(&self, transport: Transport) {
        *lock(&self.transport) = transport;
    }

    pub fn transport(&self) -> Transport {
        *lock(&self.transport)
    }

    /// Optional cap for provider-requested retry delays.
    pub fn set_max_retry_delay_ms(&self, max_retry_delay_ms: Option<u64>) {
        *lock(&self.max_retry_delay_ms) = max_retry_delay_ms;
    }

    pub fn max_retry_delay_ms(&self) -> Option<u64> {
        *lock(&self.max_retry_delay_ms)
    }

    /// Tool execution strategy for assistant messages with multiple tool calls.
    pub fn set_tool_execution(&self, mode: ToolExecutionMode) {
        *lock(&self.tool_execution) = mode;
    }

    pub fn tool_execution(&self) -> ToolExecutionMode {
        *lock(&self.tool_execution)
    }

    fn create_context_snapshot(&self) -> AgentContext {
        let state = lock(&self.state);
        AgentContext {
            system_prompt: state.system_prompt.clone(),
            messages: state.messages.clone(),
            tools: Some(state.tools.clone()),
        }
    }

    fn create_loop_config(&self, skip_initial_steering_poll: bool) -> AgentLoopConfig {
        let state = lock(&self.state);
        let stream_options = SimpleStreamOptions {
            reasoning: thinking_level_to_reasoning(state.thinking_level),
            thinking_budgets: *lock(&self.thinking_budgets),
            base: step_harness_providers::types::StreamOptions {
                session_id: lock(&self.session_id).clone(),
                on_payload: self.on_payload.clone(),
                on_response: self.on_response.clone(),
                transport: Some(*lock(&self.transport)),
                base: step_harness_providers::types::ProviderRequestOptions {
                    max_retry_delay_ms: *lock(&self.max_retry_delay_ms),
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };

        let should_stop_after_turn = self.should_stop_after_turn.clone();
        // The context-taking variant wins, mirroring the TypeScript preference
        // for `prepareNextTurnWithContext`.
        let prepare_next_turn: Option<Arc<dyn PrepareNextTurn>> =
            if let Some(hook) = self.prepare_next_turn_with_context.clone() {
                Some(hook)
            } else {
                self.prepare_next_turn.clone().map(|hook| {
                    Arc::new(PrepareNextTurnAdapter { hook }) as Arc<dyn PrepareNextTurn>
                })
            };

        AgentLoopConfig {
            model: state.model.clone(),
            tool_call_leak_retries: None,
            convert_to_llm: self.convert_to_llm.clone(),
            transform_context: self.transform_context.clone(),
            get_api_key: self.get_api_key.clone(),
            should_stop_after_turn,
            prepare_next_turn,
            get_steering_messages: Some(Arc::new(SteeringHook {
                queue: self.steering_queue.clone(),
                skip_first_poll: AtomicBool::new(skip_initial_steering_poll),
            })),
            get_follow_up_messages: Some(Arc::new(FollowUpHook {
                queue: self.follow_up_queue.clone(),
            })),
            tool_execution: *lock(&self.tool_execution),
            before_tool_call: self.before_tool_call.clone(),
            after_tool_call: self.after_tool_call.clone(),
            transform_tool_result: self.transform_tool_result.clone(),
            stream_options,
        }
    }

    async fn run_prompt_messages(
        &self,
        messages: Vec<AgentMessage>,
        skip_initial_steering_poll: bool,
    ) {
        let context = self.create_context_snapshot();
        let config = self.create_loop_config(skip_initial_steering_poll);
        let stream_fn = self.stream_fn.clone();
        self.run_with_lifecycle(|token| async move {
            run_agent_loop(
                messages,
                context,
                config,
                Some(token),
                stream_fn.as_ref(),
                self,
            )
            .await;
        })
        .await;
    }

    async fn run_continuation(&self) {
        let context = self.create_context_snapshot();
        let config = self.create_loop_config(false);
        let stream_fn = self.stream_fn.clone();
        self.run_with_lifecycle(|token| async move {
            run_agent_loop_continue(context, config, Some(token), stream_fn.as_ref(), self)
                .await
                .ok();
        })
        .await;
    }

    /// Run one agent invocation with full lifecycle management.
    ///
    /// Unexpected executor failures are reported to listeners as a synthetic
    /// failure turn, mirroring the TypeScript `handleRunFailure` path, before
    /// the run is finalized.
    async fn run_with_lifecycle<F, Fut>(&self, executor: F)
    where
        F: FnOnce(CancellationToken) -> Fut,
        Fut: std::future::Future<Output = ()>,
    {
        let token = self.signal().expect("invocation owns the running slot");

        {
            let mut state = lock(&self.state);
            state.is_streaming = true;
            state.streaming_message = None;
            state.error_message = None;
        }

        let failed = tokio::select! {
            biased;
            outcome = std::panic::AssertUnwindSafe(executor(token.clone())).catch_unwind() => outcome.is_err(),
            _ = token.cancelled() => true,
        };

        if failed {
            let aborted = token.is_cancelled();
            let failure_message = self.create_failure_message(aborted);
            self.emit(AgentEvent::MessageStart {
                message: Box::new(AgentMessage::assistant(failure_message.clone())),
            })
            .await;
            self.emit(AgentEvent::MessageEnd {
                message: Box::new(AgentMessage::assistant(failure_message.clone())),
            })
            .await;
            self.emit(AgentEvent::TurnEnd {
                message: Box::new(AgentMessage::assistant(failure_message.clone())),
                tool_results: Vec::new(),
            })
            .await;
            self.emit(AgentEvent::AgentEnd {
                messages: vec![AgentMessage::assistant(failure_message)],
            })
            .await;
        }
    }

    fn begin_run(&self) -> Result<RunGuard<'_>, AgentError> {
        let mut active = lock(&self.active_run);
        if active.is_some() {
            return Err(AgentError::AlreadyRunning);
        }
        let token = CancellationToken::new();
        let (done, receiver) = watch::channel(false);
        *active = Some(ActiveRun {
            token: token.clone(),
            done: receiver,
        });
        Ok(RunGuard {
            agent: self,
            token,
            done,
        })
    }

    fn create_failure_message(
        &self,
        aborted: bool,
    ) -> step_harness_providers::types::AssistantMessage {
        let state = lock(&self.state);
        let mut message = step_harness_providers::event::error_result(
            &state.model,
            &state.model.api,
            if aborted {
                "Operation aborted".to_owned()
            } else {
                "The agent run failed unexpectedly".to_owned()
            },
        );
        if aborted {
            message.stop_reason = step_harness_providers::types::StopReason::Aborted;
        }
        message
    }

    fn finish_run(&self) {
        let mut active = lock(&self.active_run);
        {
            let mut state = lock(&self.state);
            state.is_streaming = false;
            state.streaming_message = None;
            state.pending_tool_calls.clear();
        }
        active.take();
    }
}

/// Wraps the signal-only `prepare_next_turn` variant into the context-taking
/// loop hook, discarding the turn context like the TypeScript adapter.
struct PrepareNextTurnAdapter {
    hook: Arc<dyn PrepareNextTurnSimple>,
}

#[async_trait::async_trait]
impl PrepareNextTurn for PrepareNextTurnAdapter {
    async fn prepare_next_turn(
        &self,
        _context: &ShouldStopAfterTurnContext,
        signal: Option<&tokio_util::sync::CancellationToken>,
    ) -> Option<AgentLoopTurnUpdate> {
        self.hook.prepare_next_turn(signal).await
    }
}

/// Input accepted by [`Agent::prompt`].
pub enum PromptInput {
    Text(String),
    Message(AgentMessage),
    Messages(Vec<AgentMessage>),
}

fn normalize_prompt_input(
    input: PromptInput,
    images: Option<Vec<ImageContent>>,
) -> Vec<AgentMessage> {
    match input {
        PromptInput::Messages(messages) => messages,
        PromptInput::Message(message) => vec![message],
        PromptInput::Text(text) => {
            let mut content: Vec<UserContentBlock> = vec![UserContentBlock::Text(TextContent {
                text,
                text_signature: None,
            })];
            if let Some(images) = images {
                content.extend(images.into_iter().map(UserContentBlock::Image));
            }
            vec![AgentMessage::User(UserMessage {
                content: step_harness_providers::types::UserContent::Blocks(content),
                timestamp: step_harness_providers::utils::now_ms(),
            })]
        }
    }
}

#[async_trait::async_trait]
impl AgentEventSink for Agent {
    /// Reduce internal state for a loop event, then await listeners.
    ///
    /// `agent_end` only means no further loop events will be emitted. The run is
    /// considered idle later, after all awaited listeners for `agent_end`
    /// finish and `finish_run()` clears runtime-owned state.
    async fn emit(&self, event: AgentEvent) {
        {
            let mut state = lock(&self.state);
            match &event {
                AgentEvent::MessageStart { message }
                | AgentEvent::MessageUpdate { message, .. } => {
                    state.streaming_message = Some((**message).clone());
                }
                AgentEvent::MessageEnd { message } => {
                    state.streaming_message = None;
                    state.messages.push((**message).clone());
                }
                AgentEvent::ToolExecutionStart { tool_call_id, .. } => {
                    state.pending_tool_calls.insert(tool_call_id.clone());
                }
                AgentEvent::ToolExecutionEnd { tool_call_id, .. } => {
                    state.pending_tool_calls.remove(tool_call_id);
                }
                AgentEvent::TurnEnd { message, .. } => {
                    if let Some(assistant) = message.as_assistant()
                        && let Some(error_message) = &assistant.error_message
                    {
                        state.error_message = Some(error_message.clone());
                    }
                }
                AgentEvent::AgentEnd { .. } => {
                    state.streaming_message = None;
                }
                _ => {}
            }
        }

        let signal = lock(&self.active_run).as_ref().map(|run| run.token.clone());
        let Some(signal) = signal else {
            return;
        };
        let listeners: Vec<Arc<dyn AgentListener>> =
            lock(&self.listeners).values().cloned().collect();
        for listener in listeners {
            tokio::select! {
                biased;
                _ = listener.on_event(&event, &signal) => {}
                _ = signal.cancelled() => {}
            }
        }
    }
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = lock(&self.state);
        f.debug_struct("Agent")
            .field("system_prompt", &state.system_prompt)
            .field("model", &state.model.id)
            .field("thinking_level", &state.thinking_level)
            .field("messages", &state.messages.len())
            .field("is_streaming", &state.is_streaming)
            .finish_non_exhaustive()
    }
}

/// Cache retention re-export for convenience.
pub use step_harness_providers::types::CacheRetention as AgentCacheRetention;

/// Tool result re-export for convenience.
pub type AgentToolResultMessage = ToolResultMessage;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn queues_drain_per_mode() {
        let mut queue = PendingMessageQueue {
            messages: VecDeque::new(),
            mode: QueueMode::OneAtATime,
        };
        queue.enqueue(AgentMessage::user("one", 1));
        queue.enqueue(AgentMessage::user("two", 2));
        assert_eq!(queue.drain().len(), 1);
        assert!(queue.has_items());
        queue.mode = QueueMode::All;
        assert_eq!(queue.drain().len(), 1);
        assert!(!queue.has_items());
    }

    #[tokio::test]
    async fn prompt_rejects_concurrent_runs() {
        let agent = Agent::new(AgentOptions {
            stream_fn: Arc::new(NoopStreamFn),
            ..default_agent_options()
        });
        assert!(agent.prompt_text("hello", None).await.is_ok());
        // The noop stream fails immediately, so the run settles with the prompt
        // message plus the synthesized error assistant message.
        assert!(!agent.state().is_streaming);
        assert_eq!(agent.state().messages.len(), 2);
        assert!(agent.state().error_message.is_some());
    }

    fn default_agent_options() -> AgentOptions {
        AgentOptions {
            initial_state: None,
            convert_to_llm: None,
            transform_context: None,
            stream_fn: Arc::new(NoopStreamFn),
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
        }
    }
}
