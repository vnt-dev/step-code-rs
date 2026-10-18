//! The assistant message event stream: a push-style stream handle pair with a
//! final-result future, mirroring the TypeScript `AssistantMessageEventStream`.

use std::pin::Pin;
use std::task::{Context as TaskContext, Poll};

use futures::Stream;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, watch};

use crate::types::{Api, AssistantContent, AssistantMessage, Model, StopReason, ToolCall, Usage};

/// Event protocol for [`AssistantMessageEventStream`].
///
/// Streams emit `Start` before partial updates and terminate with either
/// `Done` (final successful message) or `Error` (final message with stop reason
/// `Error` or `Aborted` plus `error_message`).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AssistantMessageEvent {
    Start {
        partial: AssistantMessage,
    },
    TextStart {
        content_index: usize,
        partial: AssistantMessage,
    },
    TextDelta {
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    TextEnd {
        content_index: usize,
        content: String,
        partial: AssistantMessage,
    },
    ThinkingStart {
        content_index: usize,
        partial: AssistantMessage,
    },
    ThinkingDelta {
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    ThinkingEnd {
        content_index: usize,
        content: String,
        partial: AssistantMessage,
    },
    ToolCallStart {
        content_index: usize,
        partial: AssistantMessage,
    },
    ToolCallDelta {
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    ToolCallEnd {
        content_index: usize,
        tool_call: ToolCall,
        partial: AssistantMessage,
    },
    Done {
        reason: DoneReason,
        message: AssistantMessage,
    },
    Error {
        reason: ErrorReason,
        error: AssistantMessage,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum DoneReason {
    Stop,
    Length,
    ToolUse,
    Deferred,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ErrorReason {
    Aborted,
    Error,
}

/// The accumulated thinking text at a content index, for `ThinkingEnd`.
pub fn thinking_text_at(content: &[AssistantContent], index: usize) -> String {
    content
        .get(index)
        .and_then(AssistantContent::thinking)
        .map(|thinking| thinking.thinking.clone())
        .unwrap_or_default()
}

impl AssistantMessageEvent {
    /// The partial/final message snapshot carried by every event.
    pub fn message(&self) -> &AssistantMessage {
        match self {
            AssistantMessageEvent::Start { partial }
            | AssistantMessageEvent::TextStart { partial, .. }
            | AssistantMessageEvent::TextDelta { partial, .. }
            | AssistantMessageEvent::TextEnd { partial, .. }
            | AssistantMessageEvent::ThinkingStart { partial, .. }
            | AssistantMessageEvent::ThinkingDelta { partial, .. }
            | AssistantMessageEvent::ThinkingEnd { partial, .. }
            | AssistantMessageEvent::ToolCallStart { partial, .. }
            | AssistantMessageEvent::ToolCallDelta { partial, .. }
            | AssistantMessageEvent::ToolCallEnd { partial, .. } => partial,
            AssistantMessageEvent::Done { message, .. } => message,
            AssistantMessageEvent::Error { error, .. } => error,
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. }
        )
    }
}

/// Build an event snapshot without cloning content in incremental mode.
pub(crate) fn snapshot_message(
    message: &AssistantMessage,
    mode: crate::types::EventSnapshotMode,
) -> AssistantMessage {
    AssistantMessage {
        content: match mode {
            crate::types::EventSnapshotMode::Full => message.content.clone(),
            crate::types::EventSnapshotMode::MetadataOnly => Vec::new(),
        },
        api: message.api.clone(),
        provider: message.provider.clone(),
        model: message.model.clone(),
        response_model: message.response_model.clone(),
        response_id: message.response_id.clone(),
        diagnostics: message.diagnostics.clone(),
        usage: message.usage.clone(),
        stop_reason: message.stop_reason,
        deferred: message.deferred.clone(),
        error_message: message.error_message.clone(),
        raw_stop_reason: message.raw_stop_reason.clone(),
        end_turn: message.end_turn,
        timestamp: message.timestamp,
    }
}

/// Consumer side of the event stream. Implements [`Stream`] for iteration, or
/// await [`AssistantMessageEventStream::result`] to obtain the final message
/// (discarding any unconsumed events, like the TypeScript `complete()`).
pub struct AssistantMessageEventStream {
    events: mpsc::Receiver<AssistantMessageEvent>,
    result: watch::Receiver<Option<AssistantMessage>>,
    cancellation: tokio_util::sync::CancellationToken,
}

impl AssistantMessageEventStream {
    /// Wait for the final message. Events that arrive while waiting are
    /// discarded; the producer is never blocked on a full event queue.
    pub async fn result(&mut self) -> Result<AssistantMessage, EventStreamError> {
        loop {
            if let Some(message) = self.result.borrow().clone() {
                return Ok(message);
            }
            tokio::select! {
                biased;
                changed = self.result.changed() => {
                    if changed.is_err() {
                        return self.result.borrow().clone().ok_or(EventStreamError::ProducerDropped);
                    }
                }
                event = self.events.recv() => {
                    if let Some(message) = event.as_ref().and_then(terminal_message) { return Ok(message) }
                    if event.is_none() {
                        return self.result.borrow().clone().ok_or(EventStreamError::ProducerDropped);
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum EventStreamError {
    #[error("Provider stream producer exited before a terminal result")]
    ProducerDropped,
}

impl Drop for AssistantMessageEventStream {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

fn terminal_message(event: &AssistantMessageEvent) -> Option<AssistantMessage> {
    match event {
        AssistantMessageEvent::Done { message, .. } => Some(message.clone()),
        AssistantMessageEvent::Error { error, .. } => Some(error.clone()),
        _ => None,
    }
}

impl Stream for AssistantMessageEventStream {
    type Item = AssistantMessageEvent;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        self.events.poll_recv(cx)
    }
}

/// Producer side of the event stream.
pub struct AssistantMessageEventSender {
    metrics: Option<std::sync::Arc<crate::request_metrics::RequestTracker>>,
    spool: Option<std::sync::Arc<MetricsSpool>>,
    events: mpsc::Sender<AssistantMessageEvent>,
    result: watch::Sender<Option<AssistantMessage>>,
    cancellation: tokio_util::sync::CancellationToken,
    signal: Option<tokio_util::sync::CancellationToken>,
}

impl AssistantMessageEventSender {
    /// Push an event. Terminal events (`Done`/`Error`) also resolve the final
    /// result. Returns `false` when every consumer has dropped.
    pub async fn push(&mut self, event: AssistantMessageEvent) -> bool {
        if let Some(metrics) = &self.metrics { metrics.event(&event); }
        if let Some(result) = terminal_message(&event) {
            let _ = self.result.send_if_modified(|slot| {
                if slot.is_none() {
                    *slot = Some(result);
                    true
                } else {
                    false
                }
            });
        }
        if let Some(spool) = &self.spool { return spool.push(event); }
        // Terminal results are available even if a cancelled/full queue cannot receive them.
        if self
            .signal
            .as_ref()
            .is_some_and(|signal| signal.is_cancelled())
        {
            return event.is_terminal() && self.events.try_send(event).is_ok();
        }
        tokio::select! {
            biased;
            _ = self.cancellation.cancelled() => false,
            _ = async { match &self.signal { Some(signal) => signal.cancelled().await, None => std::future::pending().await } } => false,
            result = self.events.send(event) => result.is_ok(),
        }
    }

    /// End the stream with an explicit final result (no terminal event queued).
    pub fn end(&mut self, result: AssistantMessage) {
        if let Some(metrics) = &self.metrics { metrics.event(&AssistantMessageEvent::Done { reason: DoneReason::Stop, message: result.clone() }); }
        self.result.send_if_modified(|slot| {
            if slot.is_some() {
                return false;
            }
            *slot = Some(result);
            true
        });
    }

    /// Fires when the consumer drops; adapters combine it with caller cancellation.
    pub fn cancellation_token(&self) -> tokio_util::sync::CancellationToken {
        self.cancellation.clone()
    }

    pub fn set_signal(&mut self, signal: tokio_util::sync::CancellationToken) {
        self.signal = Some(signal);
    }

    /// Decouple parsing/measurement from host event-consumption backpressure.
    /// Only instrumented streams use this spool; normal bounded streams are unchanged.
    pub fn set_metrics(&mut self, metrics: Option<std::sync::Arc<crate::request_metrics::RequestTracker>>) {
        self.metrics = metrics;
        if self.metrics.is_some() && self.spool.is_none() {
            let spool = std::sync::Arc::new(MetricsSpool::default());
            let receiver = spool.clone();
            let events = self.events.clone();
            let cancellation = self.cancellation.clone();
            crate::execution_timing::spawn(async move {
                loop {
                    let event = receiver.queue.lock().unwrap_or_else(|p|p.into_inner()).pop_front();
                    if let Some(event) = event {
                        let terminal = event.is_terminal();
                        tokio::select! { biased; _ = cancellation.cancelled() => break, result = events.send(event) => { if result.is_err() { break; } } }
                        if terminal { break; }
                    } else if receiver.closed.load(std::sync::atomic::Ordering::Acquire) { break; }
                    else { tokio::select! { _ = cancellation.cancelled() => break, _ = receiver.notify.notified() => {} } }
                }
                receiver.closed.store(true,std::sync::atomic::Ordering::Release);
                receiver.queue.lock().unwrap_or_else(|p|p.into_inner()).clear();
            });
            self.spool = Some(spool);
        }
    }
    pub fn report_usage(&self, usage: &Usage, raw: &serde_json::Value) {
        if let Some(metrics) = &self.metrics {
            let has = |key: &str| raw.get(key).and_then(serde_json::Value::as_u64).is_some();
            let details = raw.get("prompt_tokens_details").or_else(|| raw.get("input_tokens_details"));
            let detail = |key: &str| details.and_then(|v|v.get(key)).and_then(serde_json::Value::as_u64).is_some();
            metrics.usage(usage, [has("input_tokens") || has("prompt_tokens"), has("cache_read_input_tokens") || detail("cached_tokens"), has("cache_creation_input_tokens") || detail("cache_write_tokens"), has("output_tokens") || has("completion_tokens")]);
        }
    }
    pub fn begin_metrics_frame(&self) { if let Some(metrics) = &self.metrics { metrics.frame(); } }

    pub fn is_closed(&self) -> bool {
        self.events.is_closed()
    }

    /// True once a terminal event or explicit end resolved the result.
    pub fn is_done(&self) -> bool {
        self.result.borrow().is_some()
    }
}

/// Coalescing adjacent deltas keeps full snapshots from accumulating quadratically
/// when a host consumes more slowly than the provider. Start/end events are retained.
#[derive(Default)]
struct MetricsSpool {
    queue: std::sync::Mutex<std::collections::VecDeque<AssistantMessageEvent>>,
    notify: tokio::sync::Notify,
    closed: std::sync::atomic::AtomicBool,
}
impl MetricsSpool {
    fn push(&self, event: AssistantMessageEvent) -> bool {
        if self.closed.load(std::sync::atomic::Ordering::Acquire) { return false; }
        let mut queue = self.queue.lock().unwrap_or_else(|p|p.into_inner());
        let merged = match (queue.back_mut(), &event) {
            (Some(AssistantMessageEvent::TextDelta {content_index:a,delta:old,partial}),AssistantMessageEvent::TextDelta {content_index:b,delta,partial:new}) |
            (Some(AssistantMessageEvent::ThinkingDelta {content_index:a,delta:old,partial}),AssistantMessageEvent::ThinkingDelta {content_index:b,delta,partial:new}) |
            (Some(AssistantMessageEvent::ToolCallDelta {content_index:a,delta:old,partial}),AssistantMessageEvent::ToolCallDelta {content_index:b,delta,partial:new}) if a==b => {old.push_str(delta);*partial=new.clone();true},
            _=>false,
        };
        if !merged { queue.push_back(event); }
        drop(queue); self.notify.notify_one(); true
    }
}
impl Drop for AssistantMessageEventSender {
    fn drop(&mut self) { if let Some(spool)=&self.spool {spool.closed.store(true,std::sync::atomic::Ordering::Release);spool.notify.notify_one();} }
}

/// Create a connected producer/consumer pair.
pub fn assistant_message_event_stream() -> (AssistantMessageEventSender, AssistantMessageEventStream)
{
    const EVENT_CHANNEL_CAPACITY: usize = 32;
    let (event_tx, event_rx) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
    let (result_tx, result_rx) = watch::channel(None);
    let cancellation = tokio_util::sync::CancellationToken::new();
    (
        AssistantMessageEventSender {
            metrics: None,
            spool: None,
            events: event_tx,
            result: result_tx,
            cancellation: cancellation.clone(),
            signal: None,
        },
        AssistantMessageEventStream {
            events: event_rx,
            result: result_rx,
            cancellation,
        },
    )
}

fn setup_error_message(model: &Model, error: impl std::fmt::Display) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason: StopReason::Error,
        deferred: None,
        error_message: Some(error.to_string()),
        raw_stop_reason: None,
        end_turn: None,
        timestamp: crate::utils::now_ms(),
    }
}

/// Return a stream immediately while running async setup (auth resolution,
/// client construction) behind it. Setup failures terminate the stream with an
/// error event carrying the final error message.
pub fn lazy_stream<F, Fut, E>(model: &Model, setup: F) -> AssistantMessageEventStream
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = Result<AssistantMessageEventStream, E>> + Send + 'static,
    E: std::fmt::Display + Send + 'static,
{
    let (mut sender, stream) = assistant_message_event_stream();
    let model = model.clone();
    crate::execution_timing::spawn(async move {
        let cancellation = sender.cancellation_token();
        let setup = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return,
            setup = setup() => setup,
        };
        match setup {
            Ok(mut inner) => {
                loop {
                    let event = tokio::select! {
                        biased;
                        _ = cancellation.cancelled() => return,
                        event = inner.next() => event,
                    };
                    let Some(event) = event else { break };
                    if !sender.push(event).await {
                        return;
                    }
                }
                // Propagate an explicit `end()` result that never traveled as an event.
                if !sender.is_done() {
                    match inner.result().await {
                        Ok(message) => sender.end(message),
                        Err(error) => {
                            let message = setup_error_message(&model, error);
                            sender
                                .push(AssistantMessageEvent::Error {
                                    reason: ErrorReason::Error,
                                    error: message,
                                })
                                .await;
                        }
                    }
                }
            }
            Err(error) => {
                let message = setup_error_message(&model, error);
                sender
                    .push(AssistantMessageEvent::Error {
                        reason: ErrorReason::Error,
                        error: message.clone(),
                    })
                    .await;
                sender.end(message);
            }
        }
    });
    stream
}

use futures::StreamExt;

/// A zero-usage, empty assistant message bound to a model, for error paths.
pub fn error_result(model: &Model, api: &Api, message: impl Into<String>) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason: StopReason::Error,
        deferred: None,
        error_message: Some(message.into()),
        raw_stop_reason: None,
        end_turn: None,
        timestamp: crate::utils::now_ms(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AssistantContent, TextContent};
    use futures::StreamExt;

    fn message(text: &str) -> AssistantMessage {
        AssistantMessage {
            content: vec![AssistantContent::Text(TextContent {
                text: text.to_owned(),
                text_signature: None,
            })],
            api: Api::AnthropicMessages,
            provider: crate::types::ProviderId::new("test"),
            model: "m".to_owned(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            timestamp: 0,
        }
    }

    #[tokio::test]
    async fn iterates_events_then_ends() {
        let (mut tx, mut rx) = assistant_message_event_stream();
        tokio::spawn(async move {
            tx.push(AssistantMessageEvent::Start {
                partial: message(""),
            })
            .await;
            let final_message = message("done");
            tx.push(AssistantMessageEvent::Done {
                reason: DoneReason::Stop,
                message: final_message,
            })
            .await;
        });
        let mut seen = 0;
        while let Some(_event) = rx.next().await {
            seen += 1;
        }
        assert_eq!(seen, 2);
    }

    #[tokio::test]
    async fn result_resolves_from_terminal_event_without_draining() {
        let (mut tx, mut rx) = assistant_message_event_stream();
        let expected = message("final");
        tokio::spawn(async move {
            for _ in 0..2000 {
                tx.push(AssistantMessageEvent::Start {
                    partial: message("x"),
                })
                .await;
            }
            tx.push(AssistantMessageEvent::Error {
                reason: ErrorReason::Error,
                error: expected,
            })
            .await;
        });
        let result = rx.result().await.unwrap();
        assert!(result.error_message.is_none());
        assert_eq!(result.stop_reason, StopReason::Stop);
    }

    #[tokio::test]
    async fn lazy_stream_setup_failure_emits_error_event() {
        let model = crate::types::Model {
            id: "m".to_owned(),
            name: "m".to_owned(),
            api: Api::OpenaiCompletions,
            provider: crate::types::ProviderId::new("p"),
            base_url: "http://localhost".to_owned(),
            reasoning: false,
            thinking_level_map: None,
            input: vec![crate::types::InputModality::Text],
            cost: Default::default(),
            context_window: 1000,
            max_tokens: 100,
            sampling_params: None,
            headers: None,
            compat: None,
        };
        let mut stream = lazy_stream(&model, || async { Err("no key") });
        let mut collected = Vec::new();
        while let Some(event) = stream.next().await {
            collected.push(event);
        }
        assert_eq!(collected.len(), 1);
        assert!(matches!(collected[0], AssistantMessageEvent::Error { .. }));
    }

    #[tokio::test]
    async fn producer_drop_returns_a_typed_error() {
        let (tx, mut rx) = assistant_message_event_stream();
        drop(tx);
        assert_eq!(
            rx.result().await.unwrap_err(),
            EventStreamError::ProducerDropped
        );
    }

    #[tokio::test]
    async fn cancellation_unblocks_a_full_queue_and_publishes_the_final_result() {
        let (mut tx, mut rx) = assistant_message_event_stream();
        let signal = tokio_util::sync::CancellationToken::new();
        tx.set_signal(signal.clone());
        for _ in 0..32 {
            assert!(
                tx.push(AssistantMessageEvent::Start {
                    partial: message("x")
                })
                .await
            );
        }
        let task = tokio::spawn(async move {
            assert!(
                !tx.push(AssistantMessageEvent::Start {
                    partial: message("blocked")
                })
                .await
            );
            tx.push(AssistantMessageEvent::Error {
                reason: ErrorReason::Aborted,
                error: message("final"),
            })
            .await;
        });
        tokio::task::yield_now().await;
        signal.cancel();
        tokio::time::timeout(std::time::Duration::from_secs(1), task)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            rx.result().await.unwrap().content[0].text().unwrap().text,
            "final"
        );
    }

    #[tokio::test]
    async fn dropping_the_consumer_cancels_the_producer() {
        let (tx, rx) = assistant_message_event_stream();
        let token = tx.cancellation_token();
        drop(rx);
        assert!(token.is_cancelled());
    }

    #[test]
    fn incremental_snapshots_preserve_metadata_without_content() {
        let original = message("x");
        let snapshot = snapshot_message(&original, crate::types::EventSnapshotMode::MetadataOnly);
        assert!(snapshot.content.is_empty());
        assert_eq!(snapshot.model, original.model);
        assert!(!original.content.is_empty());
    }
}
