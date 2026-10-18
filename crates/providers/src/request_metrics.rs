//! Host-owned per-transport-attempt metrics, measured before event delivery.
use crate::{
    event::AssistantMessageEvent,
    types::{Model, StopReason, Usage},
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RequestContext {
    pub session_id: String,
    pub execution_id: Option<String>,
    pub purpose: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RequestRecord {
    pub id: String,
    pub logical_id: String,
    pub attempt: u32,
    pub context: RequestContext,
    pub provider: String,
    pub model: String,
    pub response_model: Option<String>,
    pub started_at: u64,
    pub ended_at: Option<u64>,
    /// Total input, including cache read/write; the cache fields are subsets.
    pub input_tokens: Option<u64>,
    pub cache_read_tokens: Option<u64>,
    pub cache_write_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub first_token_ms: Option<f64>,
    pub generation_ms: Option<f64>,
    pub tps: Option<f64>,
    pub status: String,
    pub http_status: Option<u16>,
    pub error_kind: Option<String>,
}

pub trait RequestObserver: Send + Sync {
    /// Must enqueue without blocking the provider's streaming task.
    fn record(&self, record: RequestRecord);
    fn execution(&self, _record: ExecutionRecord) {}
}

#[derive(Clone, Debug)]
pub struct ExecutionRecord {
    pub id: String,
    pub session_id: String,
    pub started_at: u64,
    pub ended_at: Option<u64>,
    pub duration_ms: Option<u64>,
    pub status: String,
}

/// Covers early errors and dropped prompt futures as well as normal settlement.
pub struct ExecutionTracker {
    observer: Option<Arc<dyn RequestObserver>>,
    record: ExecutionRecord,
    started: Instant,
    finished: bool,
}
impl ExecutionTracker {
    pub fn new(observer: Option<Arc<dyn RequestObserver>>, id: String, session_id: String) -> Self {
        let record = ExecutionRecord {
            id,
            session_id,
            started_at: crate::utils::now_ms(),
            ended_at: None,
            duration_ms: None,
            status: "running".into(),
        };
        if let Some(observer) = &observer {
            observer.execution(record.clone());
        }
        Self {
            observer,
            record,
            started: Instant::now(),
            finished: false,
        }
    }
    pub fn finish(&mut self, status: &str) {
        if self.finished {
            return;
        }
        self.finished = true;
        self.record.ended_at = Some(crate::utils::now_ms());
        self.record.duration_ms =
            Some(u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX));
        self.record.status = status.into();
        if let Some(observer) = &self.observer {
            observer.execution(self.record.clone());
        }
    }
}
impl Drop for ExecutionTracker {
    fn drop(&mut self) {
        self.finish("interrupted");
    }
}
pub fn new_id() -> String {
    uuid::Uuid::now_v7().to_string()
}
struct Attempt {
    record: RequestRecord,
    started: Instant,
    first: Option<Instant>,
    last: Option<Instant>,
    output_frames: usize,
}
#[derive(Default)]
struct State {
    sequence: u32,
    active: Option<Attempt>,
    frame_at: Option<Instant>,
}
pub struct RequestTracker {
    observer: Arc<dyn RequestObserver>,
    context: RequestContext,
    logical_id: String,
    provider: String,
    model: String,
    state: Mutex<State>,
}
impl RequestTracker {
    pub fn new(
        observer: Arc<dyn RequestObserver>,
        context: RequestContext,
        model: &Model,
    ) -> Arc<Self> {
        Arc::new(Self {
            observer,
            context,
            logical_id: uuid::Uuid::now_v7().to_string(),
            provider: model.provider.to_string(),
            model: model.id.clone(),
            state: Mutex::new(State::default()),
        })
    }
    pub fn begin(&self) {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        state.sequence += 1;
        state.frame_at = None;
        let record = RequestRecord {
            id: uuid::Uuid::now_v7().to_string(),
            logical_id: self.logical_id.clone(),
            attempt: state.sequence,
            context: self.context.clone(),
            provider: self.provider.clone(),
            model: self.model.clone(),
            response_model: None,
            started_at: crate::utils::now_ms(),
            ended_at: None,
            input_tokens: None,
            cache_read_tokens: None,
            cache_write_tokens: None,
            output_tokens: None,
            first_token_ms: None,
            generation_ms: None,
            tps: None,
            status: "running".into(),
            http_status: None,
            error_kind: None,
        };
        self.observer.record(record.clone());
        state.active = Some(Attempt {
            record,
            started: Instant::now(),
            first: None,
            last: None,
            output_frames: 0,
        });
    }
    pub fn response(&self, status: u16) {
        if let Some(attempt) = self
            .state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .active
            .as_mut()
        {
            attempt.record.http_status = Some(status);
        }
    }
    pub fn usage(&self, usage: &Usage, reported: [bool; 4]) {
        if let Some(attempt) = self
            .state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .active
            .as_mut()
        {
            if reported[0] {
                attempt.record.input_tokens = Some(
                    usage
                        .input
                        .saturating_add(usage.cache_read)
                        .saturating_add(usage.cache_write),
                );
            }
            if reported[1] {
                attempt.record.cache_read_tokens = Some(usage.cache_read);
            }
            if reported[2] {
                attempt.record.cache_write_tokens = Some(usage.cache_write);
            }
            if reported[3] {
                attempt.record.output_tokens = Some(usage.output);
            }
        }
    }
    pub fn event(&self, event: &AssistantMessageEvent) {
        let output = match event {
            AssistantMessageEvent::TextDelta { delta, .. }
            | AssistantMessageEvent::ThinkingDelta { delta, .. }
            | AssistantMessageEvent::ToolCallDelta { delta, .. } => !delta.is_empty(),
            AssistantMessageEvent::TextStart {
                content_index,
                partial,
            }
            | AssistantMessageEvent::ThinkingStart {
                content_index,
                partial,
            }
            | AssistantMessageEvent::ToolCallStart {
                content_index,
                partial,
            } => partial
                .content
                .get(*content_index)
                .is_some_and(|content| match content {
                    crate::types::AssistantContent::Text(text) => !text.text.is_empty(),
                    crate::types::AssistantContent::Thinking(thinking) => {
                        !thinking.thinking.is_empty()
                    }
                    crate::types::AssistantContent::ToolCall(call) => !call.arguments.is_empty(),
                }),
            _ => false,
        };
        if output {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            let now = state.frame_at.unwrap_or_else(Instant::now);
            if let Some(attempt) = state.active.as_mut() {
                attempt.first.get_or_insert(now);
                if attempt.last != Some(now) {
                    attempt.output_frames += 1;
                }
                attempt.last = Some(now);
            }
        }
        if event.is_terminal() {
            let message = event.message();
            if let Some(attempt) = self
                .state
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .active
                .as_mut()
            {
                attempt.record.response_model = message
                    .response_model
                    .clone()
                    .or_else(|| (message.model != self.model).then(|| message.model.clone()));
            }
            self.finish(
                match message.stop_reason {
                    StopReason::Aborted => "cancelled",
                    StopReason::Error | StopReason::Pending => "failed",
                    _ => "success",
                },
                None,
                if message.stop_reason == StopReason::Error {
                    Some(
                        if message.error_message.as_deref().is_some_and(is_timeout) {
                            "timeout"
                        } else {
                            "stream"
                        },
                    )
                } else {
                    None
                },
            );
        }
    }
    pub fn frame(&self) {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .frame_at = Some(Instant::now());
    }
    pub fn finish(&self, status: &str, http_status: Option<u16>, error: Option<&str>) {
        if let Some(mut attempt) = self
            .state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .active
            .take()
        {
            attempt.record.status = status.into();
            attempt.record.ended_at = Some(crate::utils::now_ms());
            if http_status.is_some() {
                attempt.record.http_status = http_status;
            }
            attempt.record.error_kind = error.map(str::to_owned);
            attempt.record.first_token_ms = attempt
                .first
                .map(|first| first.duration_since(attempt.started).as_secs_f64() * 1000.0);
            if attempt.output_frames > 1
                && let (Some(first), Some(last)) = (attempt.first, attempt.last)
            {
                let duration = last.duration_since(first).as_secs_f64() * 1000.0;
                if duration > 0.0 {
                    attempt.record.generation_ms = Some(duration);
                    attempt.record.tps = attempt
                        .record
                        .output_tokens
                        .filter(|n| *n > 0)
                        .map(|n| n as f64 / duration * 1000.0);
                }
            }
            self.observer.record(attempt.record);
        }
    }
}
pub fn is_timeout(message: &str) -> bool {
    let text = message.to_lowercase();
    text.contains("timed out") || text.contains("timeout")
}
impl Drop for RequestTracker {
    fn drop(&mut self) {
        self.finish("interrupted", None, Some("producer_dropped"));
    }
}
