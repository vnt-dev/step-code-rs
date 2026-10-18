//! The agent loop: prompt handling, assistant streaming, and tool execution.
//!
//! Ported from `@step-harness/agent-core` `src/agent-loop.ts`. The loop works
//! with [`AgentMessage`] throughout and transforms to provider [`Message`]s
//! only at the LLM call boundary.

use std::sync::Arc;

use futures::FutureExt;
use futures::StreamExt;
use regex::Regex;
use serde_json::{Map, Value as Json};
use step_harness_providers::event::{AssistantMessageEvent, AssistantMessageEventStream};
use step_harness_providers::types::{
    AssistantMessage, Message, StopReason, ToolCall, ToolResultMessage, UserContentBlock,
};
use step_harness_providers::utils::validation::ToolValidationError;
use tokio::sync::mpsc;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use crate::types::{
    AgentContext, AgentEvent, AgentEventSink, AgentLoopConfig, AgentLoopTurnUpdate, AgentMessage,
    AgentTool, AgentToolResult, ShouldStopAfterTurnContext, StreamFn, ToolExecutionMode,
    ToolUpdateCallback, text_content,
};

/// Bounded default for resampling malformed or leaked tool calls.
pub const DEFAULT_TOOL_CALL_LEAK_RETRIES: u32 = 2;

static TOOL_CALL_MARKUP_RE: std::sync::LazyLock<Regex> =
    std::sync::LazyLock::new(|| Regex::new(r"<tool_call>|<function=").expect("valid regex"));

/// Error returned by the continuation entry points.
#[derive(Debug, thiserror::Error)]
pub enum AgentLoopError {
    #[error("Cannot continue: no messages in context")]
    NoMessages,
    #[error("Cannot continue from message role: assistant")]
    CannotContinueFromAssistant,
}

/// A stream of agent events plus the final message list.
pub struct AgentEventStream {
    events: mpsc::Receiver<AgentEvent>,
    task: tokio::task::JoinHandle<()>,
    result: watch::Receiver<Option<Vec<AgentMessage>>>,
}

impl AgentEventStream {
    /// Wait for completion, discarding any unread events to keep the bounded producer moving.
    pub async fn result(&mut self) -> Vec<AgentMessage> {
        loop {
            if let Some(messages) = self.result.borrow().clone() {
                return messages;
            }
            tokio::select! {
                changed = self.result.changed() => {
                    if changed.is_err() { return self.result.borrow().clone().unwrap_or_default(); }
                }
                _ = self.events.recv(), if !self.events.is_closed() || !self.events.is_empty() => {}
            }
        }
    }

    /// Receive the next event, or `None` once the stream is exhausted.
    pub async fn next_event(&mut self) -> Option<AgentEvent> {
        self.events.recv().await
    }
}

impl Drop for AgentEventStream {
    fn drop(&mut self) {
        self.task.abort();
    }
}

impl futures::Stream for AgentEventStream {
    type Item = AgentEvent;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.events.poll_recv(cx)
    }
}

/// Sink that forwards events into an [`AgentEventStream`].
struct ChannelSink {
    events: mpsc::Sender<AgentEvent>,
    result: watch::Sender<Option<Vec<AgentMessage>>>,
}

#[async_trait::async_trait]
impl AgentEventSink for ChannelSink {
    async fn emit(&self, event: AgentEvent) {
        let _ = self.events.send(event).await;
    }
}

fn spawn_event_stream<F, Fut>(run: F) -> AgentEventStream
where
    F: FnOnce(Arc<ChannelSink>) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = Vec<AgentMessage>> + Send,
{
    let (events_tx, events_rx) = mpsc::channel(64);
    let (result_tx, result_rx) = watch::channel(None);
    let sink = Arc::new(ChannelSink {
        events: events_tx,
        result: result_tx,
    });
    let sink_for_task = sink.clone();
    let task = tokio::spawn(async move {
        let messages = std::panic::AssertUnwindSafe(run(sink_for_task))
            .catch_unwind()
            .await
            .unwrap_or_default();
        let _ = sink.result.send(Some(messages));
    });
    AgentEventStream {
        events: events_rx,
        result: result_rx,
        task,
    }
}

/// Start an agent loop with a new prompt message.
///
/// The prompt is added to the context and events are emitted for it.
pub fn agent_loop(
    prompts: Vec<AgentMessage>,
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Option<CancellationToken>,
    stream_fn: Arc<dyn StreamFn>,
) -> AgentEventStream {
    spawn_event_stream(move |sink| async move {
        run_agent_loop(
            prompts,
            context,
            config,
            signal,
            stream_fn.as_ref(),
            sink.as_ref(),
        )
        .await
    })
}

/// Continue an agent loop from the current context without adding a new message.
///
/// The last message in context must convert to a `user` or `toolResult` message.
pub fn agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Option<CancellationToken>,
    stream_fn: Arc<dyn StreamFn>,
) -> Result<AgentEventStream, AgentLoopError> {
    if context.messages.is_empty() {
        return Err(AgentLoopError::NoMessages);
    }
    if matches!(context.messages.last(), Some(AgentMessage::Assistant(_))) {
        return Err(AgentLoopError::CannotContinueFromAssistant);
    }
    Ok(spawn_event_stream(move |sink| async move {
        run_agent_loop_continue(context, config, signal, stream_fn.as_ref(), sink.as_ref())
            .await
            .unwrap_or_default()
    }))
}

/// Run an agent loop with a new prompt, emitting events to `sink`.
pub async fn run_agent_loop(
    prompts: Vec<AgentMessage>,
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Option<CancellationToken>,
    stream_fn: &dyn StreamFn,
    sink: &dyn AgentEventSink,
) -> Vec<AgentMessage> {
    let mut new_messages = prompts.clone();
    let mut current_context = AgentContext {
        messages: [context.messages, prompts.clone()].concat(),
        ..context
    };

    sink.emit(AgentEvent::AgentStart).await;
    sink.emit(AgentEvent::TurnStart).await;
    for prompt in &prompts {
        sink.emit(AgentEvent::MessageStart {
            message: Box::new(prompt.clone()),
        })
        .await;
        sink.emit(AgentEvent::MessageEnd {
            message: Box::new(prompt.clone()),
        })
        .await;
    }

    run_loop(
        &mut current_context,
        &mut new_messages,
        config,
        signal.as_ref(),
        sink,
        stream_fn,
    )
    .await;
    new_messages
}

/// Continue an agent loop from the current context, emitting events to `sink`.
pub async fn run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Option<CancellationToken>,
    stream_fn: &dyn StreamFn,
    sink: &dyn AgentEventSink,
) -> Result<Vec<AgentMessage>, AgentLoopError> {
    if context.messages.is_empty() {
        return Err(AgentLoopError::NoMessages);
    }
    if matches!(context.messages.last(), Some(AgentMessage::Assistant(_))) {
        return Err(AgentLoopError::CannotContinueFromAssistant);
    }

    let mut new_messages = Vec::new();
    let mut current_context = context;

    sink.emit(AgentEvent::AgentStart).await;
    sink.emit(AgentEvent::TurnStart).await;

    run_loop(
        &mut current_context,
        &mut new_messages,
        config,
        signal.as_ref(),
        sink,
        stream_fn,
    )
    .await;
    Ok(new_messages)
}

/// Snapshot of a completed turn passed to `should_stop_after_turn` and
/// `prepare_next_turn`.
#[derive(Clone)]
struct CompletedTurn {
    message: AssistantMessage,
    tool_results: Vec<ToolResultMessage>,
    context: AgentContext,
    new_messages: Vec<AgentMessage>,
}

impl CompletedTurn {
    fn to_hook_context(&self) -> ShouldStopAfterTurnContext {
        ShouldStopAfterTurnContext {
            message: self.message.clone(),
            tool_results: self.tool_results.clone(),
            context: self.context.clone(),
            new_messages: self.new_messages.clone(),
        }
    }
}

/// Main loop logic shared by the prompt and continuation entry points.
async fn run_loop(
    current_context: &mut AgentContext,
    new_messages: &mut Vec<AgentMessage>,
    mut config: AgentLoopConfig,
    signal: Option<&CancellationToken>,
    sink: &dyn AgentEventSink,
    stream_fn: &dyn StreamFn,
) {
    let mut last_completed_turn: Option<CompletedTurn> = None;
    // Check for steering messages at start (the user may have typed while waiting).
    let mut pending_messages = poll_steering(&config).await;

    // Outer loop: continues when queued follow-up messages arrive after the
    // agent would stop.
    loop {
        let mut has_more_tool_calls = true;

        // Inner loop: process tool calls and steering messages.
        while has_more_tool_calls || !pending_messages.is_empty() {
            if let Some(last_turn) = &last_completed_turn {
                if let Some(update) = poll_prepare_next_turn(&config, last_turn, signal).await {
                    if let Some(context) = update.context {
                        *current_context = context;
                    }
                    if let Some(model) = update.model {
                        config.model = model;
                    }
                    if let Some(thinking_level) = update.thinking_level {
                        config.stream_options.reasoning =
                            crate::types::thinking_level_to_reasoning(thinking_level);
                    }
                }
                // Preparation can be long-running (for example, compaction).
                // Pick up steering queued while it ran; only poll again if the
                // earlier poll returned nothing.
                if pending_messages.is_empty() {
                    pending_messages = poll_steering(&config).await;
                }
                sink.emit(AgentEvent::TurnStart).await;
            }

            // Process pending messages (inject before the next assistant response).
            for message in std::mem::take(&mut pending_messages) {
                sink.emit(AgentEvent::MessageStart {
                    message: Box::new(message.clone()),
                })
                .await;
                sink.emit(AgentEvent::MessageEnd {
                    message: Box::new(message.clone()),
                })
                .await;
                current_context.messages.push(message.clone());
                new_messages.push(message);
            }

            // Stream the assistant response.
            let (mut message, mut message_position) =
                stream_assistant_response(current_context, &config, signal, sink, stream_fn).await;

            // Tool calls can contain invalid JSON or leak into plain text when
            // a serving-side parser fails. Neither case should end the run
            // immediately when a fresh response could produce a valid call.
            // Resample the identical context a bounded number of times; the
            // failed attempt is dropped from the request context while its
            // message events above remain for observability.
            let leak_retry_limit = config
                .tool_call_leak_retries
                .unwrap_or(DEFAULT_TOOL_CALL_LEAK_RETRIES);
            for _attempt in 0..leak_retry_limit {
                if !should_retry_tool_call(&message)
                    || signal.is_some_and(CancellationToken::is_cancelled)
                {
                    break;
                }
                // Only resample while the failed turn is still the last context
                // entry; anything appended or removed after it invalidates the
                // retry.
                if message_position + 1 != current_context.messages.len() {
                    break;
                }
                current_context.messages.pop();
                let (next_message, next_position) =
                    stream_assistant_response(current_context, &config, signal, sink, stream_fn)
                        .await;
                message = next_message;
                message_position = next_position;
            }
            new_messages.push(AgentMessage::assistant(message.clone()));

            if message.stop_reason == StopReason::Error
                || message.stop_reason == StopReason::Aborted
            {
                sink.emit(AgentEvent::TurnEnd {
                    message: Box::new(AgentMessage::assistant(message)),
                    tool_results: Vec::new(),
                })
                .await;
                sink.emit(AgentEvent::AgentEnd {
                    messages: new_messages.clone(),
                })
                .await;
                return;
            }

            // Check for tool calls.
            let tool_calls: Vec<ToolCall> = message
                .content
                .iter()
                .filter_map(|block| block.tool_call().cloned())
                .collect();

            let mut tool_results: Vec<ToolResultMessage> = Vec::new();
            has_more_tool_calls = false;
            if !tool_calls.is_empty() {
                let _tool_timing = step_harness_providers::execution_timing::enter(step_harness_providers::execution_timing::Phase::Tool);
                // A "length" stop means the output was cut off by the token
                // limit, so every tool call in the message may carry truncated
                // arguments. Fail them all instead of executing potentially
                // borked calls.
                let executed_tool_batch = if message.stop_reason == StopReason::Length {
                    fail_tool_calls_from_truncated_message(&tool_calls, &config, signal, sink).await
                } else {
                    execute_tool_calls(
                        current_context,
                        &message,
                        &tool_calls,
                        &config,
                        signal,
                        sink,
                    )
                    .await
                };
                tool_results = executed_tool_batch.messages;
                has_more_tool_calls = !executed_tool_batch.terminate;

                for result in &tool_results {
                    current_context
                        .messages
                        .push(AgentMessage::tool_result(result.clone()));
                    new_messages.push(AgentMessage::tool_result(result.clone()));
                }
            }

            sink.emit(AgentEvent::TurnEnd {
                message: Box::new(AgentMessage::assistant(message.clone())),
                tool_results: tool_results.clone(),
            })
            .await;

            last_completed_turn = Some(CompletedTurn {
                message: message.clone(),
                tool_results: tool_results.clone(),
                context: current_context.clone(),
                new_messages: new_messages.clone(),
            });

            if let Some(hook) = &config.should_stop_after_turn
                && hook
                    .should_stop_after_turn(
                        &last_completed_turn.as_ref().unwrap().to_hook_context(),
                        signal,
                    )
                    .await
            {
                sink.emit(AgentEvent::AgentEnd {
                    messages: new_messages.clone(),
                })
                .await;
                return;
            }

            pending_messages = poll_steering(&config).await;
        }

        // The agent would stop here. Check for follow-up messages.
        let follow_up_messages = match &config.get_follow_up_messages {
            Some(hook) => hook.get_follow_up_messages().await,
            None => Vec::new(),
        };
        if !follow_up_messages.is_empty() {
            // Set as pending so the inner loop processes them.
            pending_messages = follow_up_messages;
            continue;
        }

        break;
    }

    sink.emit(AgentEvent::AgentEnd {
        messages: new_messages.clone(),
    })
    .await;
}

async fn poll_steering(config: &AgentLoopConfig) -> Vec<AgentMessage> {
    match &config.get_steering_messages {
        Some(hook) => hook.get_steering_messages().await,
        None => Vec::new(),
    }
}

async fn poll_prepare_next_turn(
    config: &AgentLoopConfig,
    last_turn: &CompletedTurn,
    signal: Option<&CancellationToken>,
) -> Option<AgentLoopTurnUpdate> {
    let hook = config.prepare_next_turn.as_ref()?;
    hook.prepare_next_turn(&last_turn.to_hook_context(), signal)
        .await
}

/// Retry malformed tool arguments even when there is no leaked markup. No
/// calls from an error turn are executed, including partially parsed calls.
/// Successful turns are retried only for leaked markup without a tool call;
/// unrelated errors and aborts still exit.
fn should_retry_tool_call(message: &AssistantMessage) -> bool {
    let argument_parse_error = message.stop_reason == StopReason::Error
        && message
            .error_message
            .as_deref()
            .is_some_and(|error| error.starts_with("Invalid tool argument JSON:"));
    if message.stop_reason == StopReason::Error {
        return argument_parse_error;
    }
    if message.stop_reason == StopReason::Aborted {
        return false;
    }
    let mut saw_markup = false;
    for block in &message.content {
        if block.tool_call().is_some() {
            return false;
        }
        if let Some(text) = block.text()
            && TOOL_CALL_MARKUP_RE.is_match(&text.text)
        {
            saw_markup = true;
        }
    }
    saw_markup
}

/// Stream an assistant response from the LLM.
///
/// The partial message is appended to `context.messages` when streaming starts
/// and replaced by the final message when the stream terminates. Returns the
/// final message together with its position in `context.messages`.
async fn stream_assistant_response(
    context: &mut AgentContext,
    config: &AgentLoopConfig,
    signal: Option<&CancellationToken>,
    sink: &dyn AgentEventSink,
    stream_fn: &dyn StreamFn,
) -> (AssistantMessage, usize) {
    // Apply the context transform if configured (AgentMessage[] → AgentMessage[]).
    let messages = match &config.transform_context {
        Some(transform) => transform.transform(&context.messages, signal).await,
        None => context.messages.clone(),
    };

    // Convert to LLM-compatible messages (AgentMessage[] → Message[]).
    let llm_messages = config.convert_to_llm.convert(&messages).await;

    // Build the LLM context.
    let llm_context = context.to_llm_context(llm_messages);

    // Resolve the API key (important for expiring tokens).
    let resolved_api_key = match &config.get_api_key {
        Some(hook) => hook.get_api_key(config.model.provider.as_str()).await,
        None => None,
    }
    .or_else(|| config.stream_options.base.base.api_key.clone());

    let options = config.stream_options_for_request(resolved_api_key, signal);
    let api_timing = step_harness_providers::execution_timing::enter(step_harness_providers::execution_timing::Phase::Api);
    let mut response: AssistantMessageEventStream = stream_fn
        .stream(&config.model, &llm_context, &options)
        .await;

    let mut added_partial = false;

    while let Some(event) = response.next().await {
        if event.is_terminal() {
            break;
        }
        match &event {
            AssistantMessageEvent::Start { partial } => {
                context
                    .messages
                    .push(AgentMessage::assistant(partial.clone()));
                added_partial = true;
                sink.emit(AgentEvent::MessageStart {
                    message: Box::new(AgentMessage::assistant(partial.clone())),
                })
                .await;
            }
            AssistantMessageEvent::TextStart { partial, .. }
            | AssistantMessageEvent::TextDelta { partial, .. }
            | AssistantMessageEvent::TextEnd { partial, .. }
            | AssistantMessageEvent::ThinkingStart { partial, .. }
            | AssistantMessageEvent::ThinkingDelta { partial, .. }
            | AssistantMessageEvent::ThinkingEnd { partial, .. }
            | AssistantMessageEvent::ToolCallStart { partial, .. }
            | AssistantMessageEvent::ToolCallDelta { partial, .. }
            | AssistantMessageEvent::ToolCallEnd { partial, .. } => {
                if let Some(last) = context.messages.last_mut() {
                    *last = AgentMessage::assistant(partial.clone());
                }
                sink.emit(AgentEvent::MessageUpdate {
                    message: Box::new(AgentMessage::assistant(partial.clone())),
                    assistant_message_event: Box::new(event.clone()),
                })
                .await;
            }
            AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. } => {
                unreachable!("terminal events break the loop above")
            }
        }
    }

    // A producer that exits without a terminal event is encoded as an error
    // assistant message, matching the stream contract.
    let final_message = response.result().await.unwrap_or_else(|error| {
        step_harness_providers::event::error_result(
            &config.model,
            &config.model.api,
            error.to_string(),
        )
    });

    drop(api_timing);

    if added_partial {
        if let Some(last) = context.messages.last_mut() {
            *last = AgentMessage::assistant(final_message.clone());
        }
    } else {
        context
            .messages
            .push(AgentMessage::assistant(final_message.clone()));
        sink.emit(AgentEvent::MessageStart {
            message: Box::new(AgentMessage::assistant(final_message.clone())),
        })
        .await;
    }
    sink.emit(AgentEvent::MessageEnd {
        message: Box::new(AgentMessage::assistant(final_message.clone())),
    })
    .await;
    (final_message, context.messages.len() - 1)
}

/// A batch of finalized tool calls.
struct ExecutedToolCallBatch {
    messages: Vec<ToolResultMessage>,
    terminate: bool,
}

/// A finalized tool call outcome.
#[derive(Clone)]
struct FinalizedToolCallOutcome {
    tool_call: ToolCall,
    result: AgentToolResult,
    is_error: bool,
}

enum ToolCallPreparation {
    Prepared {
        tool_call: ToolCall,
        tool: AgentTool,
        args: Map<String, Json>,
    },
    Immediate {
        result: AgentToolResult,
        is_error: bool,
    },
}

/// Fail all tool calls from an assistant message that was truncated by the
/// output token limit. Streamed tool-call arguments are finalized with a
/// best-effort JSON salvage parser, so a truncated message can yield tool calls
/// whose arguments parse and validate but are silently incomplete. None of them
/// are safe to execute; report each as an error so the model can re-issue them.
async fn fail_tool_calls_from_truncated_message(
    tool_calls: &[ToolCall],
    config: &AgentLoopConfig,
    signal: Option<&CancellationToken>,
    sink: &dyn AgentEventSink,
) -> ExecutedToolCallBatch {
    let mut messages = Vec::with_capacity(tool_calls.len());
    for tool_call in tool_calls {
        sink.emit(AgentEvent::ToolExecutionStart {
            tool_call_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            args: Json::Object(tool_call.arguments.clone()),
        })
        .await;
        let mut finalized = FinalizedToolCallOutcome {
            tool_call: tool_call.clone(),
            result: AgentToolResult::error(format!(
                "Tool call \"{}\" was not executed: the response hit the output token limit, so its arguments may be truncated. Re-issue the tool call with complete arguments.",
                tool_call.name
            )),
            is_error: true,
        };
        emit_tool_execution_end(&mut finalized, config, signal, sink).await;
        let tool_result_message = create_tool_result_message(&finalized);
        emit_tool_result_message(&tool_result_message, sink).await;
        messages.push(tool_result_message);
    }
    ExecutedToolCallBatch {
        messages,
        terminate: false,
    }
}

/// Execute the tool calls of one assistant message, choosing sequential or
/// parallel execution.
async fn execute_tool_calls(
    current_context: &AgentContext,
    assistant_message: &AssistantMessage,
    tool_calls: &[ToolCall],
    config: &AgentLoopConfig,
    signal: Option<&CancellationToken>,
    sink: &dyn AgentEventSink,
) -> ExecutedToolCallBatch {
    let has_sequential_tool_call = tool_calls.iter().any(|tool_call| {
        current_context
            .tools
            .as_ref()
            .and_then(|tools| {
                tools
                    .iter()
                    .find(|tool| tool.name == tool_call.name)
                    .map(|tool| tool.execution_mode == Some(ToolExecutionMode::Sequential))
            })
            .unwrap_or(false)
    });
    if config.tool_execution == ToolExecutionMode::Sequential || has_sequential_tool_call {
        execute_tool_calls_sequential(
            current_context,
            assistant_message,
            tool_calls,
            config,
            signal,
            sink,
        )
        .await
    } else {
        execute_tool_calls_parallel(
            current_context,
            assistant_message,
            tool_calls,
            config,
            signal,
            sink,
        )
        .await
    }
}

async fn execute_tool_calls_sequential(
    current_context: &AgentContext,
    assistant_message: &AssistantMessage,
    tool_calls: &[ToolCall],
    config: &AgentLoopConfig,
    signal: Option<&CancellationToken>,
    sink: &dyn AgentEventSink,
) -> ExecutedToolCallBatch {
    let mut finalized_calls: Vec<FinalizedToolCallOutcome> = Vec::new();
    let mut messages: Vec<ToolResultMessage> = Vec::new();

    for tool_call in tool_calls {
        sink.emit(AgentEvent::ToolExecutionStart {
            tool_call_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            args: Json::Object(tool_call.arguments.clone()),
        })
        .await;

        let preparation = prepare_tool_call(
            current_context,
            assistant_message,
            tool_call,
            config,
            signal,
        )
        .await;
        let mut finalized = match preparation {
            ToolCallPreparation::Immediate { result, is_error } => FinalizedToolCallOutcome {
                tool_call: tool_call.clone(),
                result,
                is_error,
            },
            ToolCallPreparation::Prepared {
                tool_call,
                tool,
                args,
            } => {
                let prepared = PreparedToolCall {
                    tool_call,
                    tool,
                    args,
                };
                let executed = execute_prepared_tool_call(&prepared, signal, sink).await;
                finalize_executed_tool_call(
                    current_context,
                    assistant_message,
                    &prepared,
                    executed,
                    config,
                    signal,
                )
                .await
            }
        };

        emit_tool_execution_end(&mut finalized, config, signal, sink).await;
        let tool_result_message = create_tool_result_message(&finalized);
        emit_tool_result_message(&tool_result_message, sink).await;
        finalized_calls.push(finalized);
        messages.push(tool_result_message);

        if signal.is_some_and(|signal| signal.is_cancelled()) {
            break;
        }
    }

    ExecutedToolCallBatch {
        messages,
        terminate: should_terminate_tool_batch(&finalized_calls),
    }
}

async fn execute_tool_calls_parallel(
    current_context: &AgentContext,
    assistant_message: &AssistantMessage,
    tool_calls: &[ToolCall],
    config: &AgentLoopConfig,
    signal: Option<&CancellationToken>,
    sink: &dyn AgentEventSink,
) -> ExecutedToolCallBatch {
    // Entries are either immediate outcomes or concurrent executions. The
    // concurrent entries are polled together and emit `tool_execution_end` in
    // tool completion order; tool-result messages are emitted afterwards in
    // assistant source order.
    enum Entry<'a> {
        Immediate(Box<FinalizedToolCallOutcome>),
        Concurrent(
            std::pin::Pin<
                Box<dyn std::future::Future<Output = FinalizedToolCallOutcome> + Send + 'a>,
            >,
        ),
    }

    let mut entries: Vec<Entry<'_>> = Vec::new();

    for tool_call in tool_calls {
        sink.emit(AgentEvent::ToolExecutionStart {
            tool_call_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            args: Json::Object(tool_call.arguments.clone()),
        })
        .await;

        let preparation = prepare_tool_call(
            current_context,
            assistant_message,
            tool_call,
            config,
            signal,
        )
        .await;
        match preparation {
            ToolCallPreparation::Immediate { result, is_error } => {
                let mut finalized = FinalizedToolCallOutcome {
                    tool_call: tool_call.clone(),
                    result,
                    is_error,
                };
                emit_tool_execution_end(&mut finalized, config, signal, sink).await;
                entries.push(Entry::Immediate(Box::new(finalized)));
                if signal.is_some_and(|signal| signal.is_cancelled()) {
                    break;
                }
                continue;
            }
            ToolCallPreparation::Prepared {
                tool_call,
                tool,
                args,
            } => {
                let prepared = PreparedToolCall {
                    tool_call,
                    tool,
                    args,
                };
                entries.push(Entry::Concurrent(Box::pin(async move {
                    let executed = execute_prepared_tool_call(&prepared, signal, sink).await;
                    let mut finalized = finalize_executed_tool_call(
                        current_context,
                        assistant_message,
                        &prepared,
                        executed,
                        config,
                        signal,
                    )
                    .await;
                    emit_tool_execution_end(&mut finalized, config, signal, sink).await;
                    finalized
                })));
                if signal.is_some_and(|signal| signal.is_cancelled()) {
                    break;
                }
            }
        }
    }

    let executions: Vec<futures::future::BoxFuture<'_, FinalizedToolCallOutcome>> = entries
        .into_iter()
        .map(|entry| match entry {
            Entry::Immediate(finalized) => Box::pin(async move { *finalized })
                as futures::future::BoxFuture<'_, FinalizedToolCallOutcome>,
            Entry::Concurrent(future) => future,
        })
        .collect();
    let ordered_finalized_calls: Vec<FinalizedToolCallOutcome> = futures::stream::iter(executions)
        .buffered(16)
        .collect()
        .await;

    let mut messages = Vec::with_capacity(ordered_finalized_calls.len());
    for finalized in &ordered_finalized_calls {
        let tool_result_message = create_tool_result_message(finalized);
        emit_tool_result_message(&tool_result_message, sink).await;
        messages.push(tool_result_message);
    }

    ExecutedToolCallBatch {
        messages,
        terminate: should_terminate_tool_batch(&ordered_finalized_calls),
    }
}

fn should_terminate_tool_batch(finalized_calls: &[FinalizedToolCallOutcome]) -> bool {
    !finalized_calls.is_empty()
        && finalized_calls
            .iter()
            .all(|finalized| finalized.result.terminate == Some(true))
}

struct PreparedToolCall {
    tool_call: ToolCall,
    tool: AgentTool,
    args: Map<String, Json>,
}

async fn prepare_tool_call(
    current_context: &AgentContext,
    assistant_message: &AssistantMessage,
    tool_call: &ToolCall,
    config: &AgentLoopConfig,
    signal: Option<&CancellationToken>,
) -> ToolCallPreparation {
    let Some(tool) = current_context
        .tools
        .as_ref()
        .and_then(|tools| tools.iter().find(|tool| tool.name == tool_call.name))
    else {
        return ToolCallPreparation::Immediate {
            result: AgentToolResult::error(format!("Tool {} not found", tool_call.name)),
            is_error: true,
        };
    };

    // Apply the compatibility shim before schema validation.
    let mut prepared_call = tool_call.clone();
    if let Some(prepare) = &tool.prepare_arguments {
        let prepared_arguments = prepare(Json::Object(tool_call.arguments.clone()));
        if let Json::Object(arguments) = prepared_arguments {
            prepared_call.arguments = arguments;
        }
    }

    match validate_tool_arguments(&tool.to_tool(), &prepared_call) {
        Ok(validated_args) => {
            if let Some(hook) = &config.before_tool_call {
                let before_result = hook
                    .before_tool_call(
                        &crate::types::BeforeToolCallContext {
                            assistant_message: assistant_message.clone(),
                            tool_call: tool_call.clone(),
                            args: validated_args.clone(),
                            context: current_context.clone(),
                        },
                        signal,
                    )
                    .await;
                if signal.is_some_and(|signal| signal.is_cancelled()) {
                    return ToolCallPreparation::Immediate {
                        result: AgentToolResult::error("Operation aborted"),
                        is_error: true,
                    };
                }
                if let Some(before_result) = before_result
                    && before_result.block
                {
                    let mut result = AgentToolResult::error(
                        before_result
                            .reason
                            .unwrap_or_else(|| "Tool execution was blocked".to_owned()),
                    );
                    if before_result.terminate == Some(true) {
                        result.terminate = Some(true);
                    }
                    return ToolCallPreparation::Immediate {
                        result,
                        is_error: true,
                    };
                }
            }
            if signal.is_some_and(|signal| signal.is_cancelled()) {
                return ToolCallPreparation::Immediate {
                    result: AgentToolResult::error("Operation aborted"),
                    is_error: true,
                };
            }
            ToolCallPreparation::Prepared {
                tool_call: prepared_call,
                tool: tool.clone(),
                args: validated_args,
            }
        }
        Err(error) => ToolCallPreparation::Immediate {
            result: AgentToolResult::error(error.to_string()),
            is_error: true,
        },
    }
}

struct ExecutedToolCallOutcome {
    result: AgentToolResult,
    is_error: bool,
}

async fn execute_prepared_tool_call(
    prepared: &PreparedToolCall,
    signal: Option<&CancellationToken>,
    sink: &dyn AgentEventSink,
) -> ExecutedToolCallOutcome {
    // Progress snapshots are coalesced when the consumer is slower than the tool.
    let (update_tx, mut update_rx) = watch::channel::<Option<Json>>(None);
    let accepting = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let accepting_for_callback = accepting.clone();
    let on_update: ToolUpdateCallback = Arc::new(move |partial| {
        if accepting_for_callback.load(std::sync::atomic::Ordering::Acquire) {
            update_tx.send_replace(Some(serde_json::to_value(&partial).unwrap_or(Json::Null)));
        }
    });
    let execution = prepared.tool.executor.execute(
        &prepared.tool_call.id,
        &prepared.args,
        signal,
        Some(&on_update),
    );
    tokio::pin!(execution);
    let cancelled = async {
        match signal {
            Some(signal) => signal.cancelled().await,
            None => std::future::pending::<()>().await,
        }
    };
    tokio::pin!(cancelled);
    let outcome = loop {
        tokio::select! {
            biased;
            _ = &mut cancelled => break Err(crate::types::tool_error("Operation aborted")),
            result = &mut execution => break result,
            changed = update_rx.changed() => {
                if changed.is_ok() {
                    let partial = update_rx.borrow_and_update().clone();
                    if let Some(partial_result) = partial {
                        sink.emit(AgentEvent::ToolExecutionUpdate {
                            tool_call_id: prepared.tool_call.id.clone(), tool_name: prepared.tool_call.name.clone(),
                            args: Json::Object(prepared.tool_call.arguments.clone()), partial_result,
                        }).await;
                    }
                }
            }
        }
    };
    accepting.store(false, std::sync::atomic::Ordering::Release);
    if update_rx.has_changed().unwrap_or(false) {
        let partial = update_rx.borrow_and_update().clone();
        if let Some(partial_result) = partial {
            sink.emit(AgentEvent::ToolExecutionUpdate {
                tool_call_id: prepared.tool_call.id.clone(),
                tool_name: prepared.tool_call.name.clone(),
                args: Json::Object(prepared.tool_call.arguments.clone()),
                partial_result,
            })
            .await;
        }
    }

    match outcome {
        Ok(result) => ExecutedToolCallOutcome {
            result,
            is_error: false,
        },
        Err(error) => ExecutedToolCallOutcome {
            result: AgentToolResult::error(error.to_string()),
            is_error: true,
        },
    }
}

async fn finalize_executed_tool_call(
    current_context: &AgentContext,
    assistant_message: &AssistantMessage,
    prepared: &PreparedToolCall,
    executed: ExecutedToolCallOutcome,
    config: &AgentLoopConfig,
    signal: Option<&CancellationToken>,
) -> FinalizedToolCallOutcome {
    let mut result = executed.result;
    let mut is_error = executed.is_error;

    if let Some(hook) = &config.after_tool_call
        && let Some(after_result) = hook
            .after_tool_call(
                &crate::types::AfterToolCallContext {
                    assistant_message: assistant_message.clone(),
                    tool_call: prepared.tool_call.clone(),
                    args: prepared.args.clone(),
                    result: result.clone(),
                    is_error,
                    context: current_context.clone(),
                },
                signal,
            )
            .await
    {
        if let Some(content) = after_result.content {
            result.content = content;
        }
        if let Some(details) = after_result.details {
            result.details = details;
        }
        if let Some(usage) = after_result.usage {
            result.usage = Some(usage);
        }
        if let Some(terminate) = after_result.terminate {
            result.terminate = Some(terminate);
        }
        if let Some(after_is_error) = after_result.is_error {
            is_error = after_is_error;
        }
    }

    FinalizedToolCallOutcome {
        tool_call: prepared.tool_call.clone(),
        result,
        is_error,
    }
}

async fn emit_tool_execution_end(
    finalized: &mut FinalizedToolCallOutcome,
    config: &AgentLoopConfig,
    signal: Option<&CancellationToken>,
    sink: &dyn AgentEventSink,
) {
    if let Some(transform) = &config.transform_tool_result {
        match transform
            .transform_tool_result(finalized.result.content.clone(), signal)
            .await
        {
            Ok(content) => finalized.result.content = content,
            Err(error) => {
                let reason: String = error.to_string().chars().take(512).collect();
                let mut content = vec![text_content(format!(
                    "Tool output could not be prepared: {reason}. The tool may already have run; check its effects before repeating a state-changing call."
                ))];
                content.extend(
                    finalized
                        .result
                        .content
                        .iter()
                        .filter(|part| matches!(part, UserContentBlock::Image(_)))
                        .cloned(),
                );
                finalized.result.content = content;
                finalized.is_error = true;
            }
        }
    }
    sink.emit(AgentEvent::ToolExecutionEnd {
        tool_call_id: finalized.tool_call.id.clone(),
        tool_name: finalized.tool_call.name.clone(),
        result: serde_json::to_value(&finalized.result).unwrap_or(Json::Null),
        is_error: finalized.is_error,
    })
    .await;
}

fn create_tool_result_message(finalized: &FinalizedToolCallOutcome) -> ToolResultMessage {
    ToolResultMessage {
        tool_call_id: finalized.tool_call.id.clone(),
        tool_name: finalized.tool_call.name.clone(),
        // Untyped tools can return results without content; normalize so a
        // missing content array never enters session history or payloads.
        content: finalized.result.content.clone(),
        details: Some(serde_json::to_value(&finalized.result.details).unwrap_or(Json::Null)),
        usage: finalized.result.usage.clone(),
        added_tool_names: finalized
            .result
            .added_tool_names
            .clone()
            .filter(|names| !names.is_empty()),
        is_error: finalized.is_error,
        timestamp: step_harness_providers::utils::now_ms(),
    }
}

async fn emit_tool_result_message(
    tool_result_message: &ToolResultMessage,
    sink: &dyn AgentEventSink,
) {
    sink.emit(AgentEvent::MessageStart {
        message: Box::new(AgentMessage::tool_result(tool_result_message.clone())),
    })
    .await;
    sink.emit(AgentEvent::MessageEnd {
        message: Box::new(AgentMessage::tool_result(tool_result_message.clone())),
    })
    .await;
}

/// Validate tool call arguments against a provider tool.
pub fn validate_tool_arguments(
    tool: &step_harness_providers::types::Tool,
    tool_call: &ToolCall,
) -> Result<Map<String, Json>, ToolValidationError> {
    step_harness_providers::utils::validation::validate_tool_arguments(tool, tool_call)
}

/// Adapter that turns a plain async closure into a [`crate::types::ConvertToLlm`].
pub struct FnConvertToLlm<F>(pub F);

#[async_trait::async_trait]
impl<F, Fut> crate::types::ConvertToLlm for FnConvertToLlm<F>
where
    F: Fn(&[AgentMessage]) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Vec<Message>> + Send,
{
    async fn convert(&self, messages: &[AgentMessage]) -> Vec<Message> {
        (self.0)(messages).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use step_harness_providers::types::{AssistantContent, Model as ProvidersModel, TextContent};

    fn assistant_with_text(text: &str) -> AssistantMessage {
        AssistantMessage {
            content: vec![AssistantContent::Text(TextContent {
                text: text.to_owned(),
                text_signature: None,
            })],
            api: step_harness_providers::types::Api::OpenaiCompletions,
            provider: step_harness_providers::types::ProviderId::new("test"),
            model: "m".to_owned(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: Default::default(),
            stop_reason: StopReason::Stop,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            timestamp: 0,
        }
    }

    #[tokio::test]
    async fn result_drains_bounded_events_without_requiring_a_second_consumer() {
        let mut stream = spawn_event_stream(|sink| async move {
            for _ in 0..1000 {
                sink.emit(AgentEvent::TurnStart).await;
            }
            vec![AgentMessage::user("complete", 0)]
        });
        let result = tokio::time::timeout(std::time::Duration::from_secs(1), stream.result())
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
    }

    #[tokio::test]
    async fn dropping_event_stream_aborts_its_producer() {
        struct CancelGuard(tokio::sync::oneshot::Sender<()>);
        impl Drop for CancelGuard {
            fn drop(&mut self) {
                let sender = std::mem::replace(&mut self.0, tokio::sync::oneshot::channel().0);
                let _ = sender.send(());
            }
        }
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
        let (cancel_tx, cancel_rx) = tokio::sync::oneshot::channel();
        let stream = spawn_event_stream(move |_| async move {
            let _guard = CancelGuard(cancel_tx);
            ready_tx.send(()).unwrap();
            std::future::pending::<()>().await;
            vec![]
        });
        ready_rx.await.unwrap();
        drop(stream);
        tokio::time::timeout(std::time::Duration::from_secs(1), cancel_rx)
            .await
            .unwrap()
            .unwrap();
    }

    #[test]
    fn detects_retryable_tool_call_failures() {
        assert!(should_retry_tool_call(&assistant_with_text(
            "let me call <tool_call> now"
        )));
        assert!(should_retry_tool_call(&assistant_with_text(
            "<function=foo>"
        )));
        assert!(!should_retry_tool_call(&assistant_with_text("hello")));
        let mut with_call = assistant_with_text("x");
        with_call.content.push(AssistantContent::ToolCall(ToolCall {
            id: "1".to_owned(),
            name: "t".to_owned(),
            arguments: Map::new(),
            thought_signature: None,
            namespace: None,
        }));
        assert!(!should_retry_tool_call(&with_call));
        let mut errored = assistant_with_text("<tool_call>");
        errored.stop_reason = StopReason::Error;
        assert!(!should_retry_tool_call(&errored));
        errored.error_message = Some(
            "Invalid tool argument JSON: EOF while parsing an object at line 1 column 1".to_owned(),
        );
        errored.content.push(AssistantContent::ToolCall(ToolCall {
            id: "unfinished".to_owned(),
            name: "search_web".to_owned(),
            arguments: Map::new(),
            thought_signature: None,
            namespace: None,
        }));
        assert!(should_retry_tool_call(&errored));
        // A provider can return only the parse error, without text or a call.
        errored.content.clear();
        assert!(should_retry_tool_call(&errored));
        errored.error_message = Some("HTTP 403".to_owned());
        assert!(!should_retry_tool_call(&errored));
        errored.stop_reason = StopReason::Aborted;
        errored.error_message = Some(
            "Invalid tool argument JSON: expected `:` at line 1 column 22".to_owned(),
        );
        assert!(!should_retry_tool_call(&errored));
    }

    #[test]
    fn continuation_requires_non_assistant_last_message() {
        let model = ProvidersModel {
            id: "m".to_owned(),
            name: "m".to_owned(),
            api: step_harness_providers::types::Api::OpenaiCompletions,
            provider: step_harness_providers::types::ProviderId::new("p"),
            base_url: "http://localhost".to_owned(),
            reasoning: false,
            thinking_level_map: None,
            input: vec![],
            cost: Default::default(),
            context_window: 100,
            max_tokens: 10,
            sampling_params: None,
            headers: None,
            compat: None,
        };
        let config = |model: ProvidersModel| AgentLoopConfig {
            model,
            tool_call_leak_retries: None,
            convert_to_llm: Arc::new(crate::types::DefaultConvertToLlm),
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
            stream_options: Default::default(),
        };
        let stream_fn = Arc::new(crate::types::NoopStreamFn);

        let empty = AgentContext::default();
        assert!(matches!(
            agent_loop_continue(empty, config(model.clone()), None, stream_fn.clone()),
            Err(AgentLoopError::NoMessages)
        ));

        let assistant_last = AgentContext {
            messages: vec![AgentMessage::assistant(assistant_with_text("hi"))],
            ..Default::default()
        };
        assert!(matches!(
            agent_loop_continue(assistant_last, config(model), None, stream_fn),
            Err(AgentLoopError::CannotContinueFromAssistant)
        ));
    }
}
