//! Process-local reference adapter. Storage is unbounded; scope contexts to
//! tests or bounded diagnostic sessions. No lock is held across an await.

use crate::{SpanAttributes, SpanOptions, SpanStatus, TelemetryContext, TelemetrySpan};
use parking_lot::Mutex;
use std::future::Future;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct RecordedTelemetryEvent {
    pub name: String,
    pub attributes: SpanAttributes,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecordedTelemetrySpan {
    pub id: u64,
    pub parent_id: Option<u64>,
    pub name: String,
    pub attributes: SpanAttributes,
    pub events: Vec<RecordedTelemetryEvent>,
    pub status: SpanStatus,
    pub settled: bool,
    pub end_sequence: Option<u64>,
}

#[derive(Debug)]
struct SpanData {
    recorded: RecordedTelemetrySpan,
    explicit_status: bool,
}

#[derive(Debug, Default)]
struct State {
    spans: Vec<SpanData>,
    end_sequence: u64,
}

type SharedState = Arc<Mutex<State>>;

#[derive(Debug, Clone, Default)]
pub struct InMemoryTelemetryContext {
    state: SharedState,
}

impl InMemoryTelemetryContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Atomic, detached snapshot in span-admission order. Clones values while
    /// holding the recording lock; use this diagnostic adapter for bounded data.
    pub fn get_spans(&self) -> Vec<RecordedTelemetrySpan> {
        self.state
            .lock()
            .spans
            .iter()
            .map(|span| span.recorded.clone())
            .collect()
    }
}

/// None is an inert handle, requiring no throwaway state or allocation.
#[derive(Debug, Clone, Default)]
pub struct InMemorySpan {
    recording: Option<(SharedState, usize)>,
}

impl InMemorySpan {
    fn update(&self, update: impl FnOnce(&mut SpanData)) {
        if let Some((state, index)) = &self.recording {
            let mut state = state.lock();
            let data = &mut state.spans[*index];
            if !data.recorded.settled {
                update(data);
            }
        }
    }

    fn settle(&self, dropped: bool) {
        if let Some((state, index)) = &self.recording {
            let mut state = state.lock();
            if state.spans[*index].recorded.settled {
                return;
            }
            state.end_sequence += 1;
            let sequence = state.end_sequence;
            let data = &mut state.spans[*index];
            if dropped && !data.explicit_status {
                data.recorded.status = SpanStatus::Cancelled;
            }
            data.recorded.settled = true;
            data.recorded.end_sequence = Some(sequence);
        }
    }
}

impl TelemetrySpan for InMemorySpan {
    fn is_recording(&self) -> bool {
        self.recording
            .as_ref()
            .is_some_and(|(state, index)| !state.lock().spans[*index].recorded.settled)
    }
    fn add_event(&self, name: &str, attributes: SpanAttributes) {
        self.update(|data| {
            data.recorded.events.push(RecordedTelemetryEvent {
                name: name.to_owned(),
                attributes,
            })
        });
    }
    fn set_attributes(&self, attributes: SpanAttributes) {
        self.update(|data| data.recorded.attributes.extend(attributes));
    }
    fn set_status(&self, status: SpanStatus) {
        self.update(|data| {
            data.recorded.status = status;
            data.explicit_status = true;
        });
    }
    fn set_default_status(&self, status: SpanStatus) {
        self.update(|data| {
            if !data.explicit_status {
                data.recorded.status = status;
            }
        });
    }
}

/// Owns settlement independently of any cloned handles held by application code.
struct SpanGuard {
    span: InMemorySpan,
    completed: bool,
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        if !self.completed {
            self.span.settle(true);
        }
    }
}

fn admit(state: &SharedState, parent: Option<usize>, options: SpanOptions) -> InMemorySpan {
    // Checking the parent and inserting its child is one atomic operation.
    let mut locked = state.lock();
    if parent.is_some_and(|index| locked.spans[index].recorded.settled) {
        return InMemorySpan::default();
    }
    let index = locked.spans.len();
    let parent_id = parent.map(|index| locked.spans[index].recorded.id);
    locked.spans.push(SpanData {
        recorded: RecordedTelemetrySpan {
            id: index as u64 + 1,
            parent_id,
            name: options.name,
            attributes: options.attributes,
            events: Vec::new(),
            status: SpanStatus::Ok,
            settled: false,
            end_sequence: None,
        },
        explicit_status: false,
    });
    InMemorySpan {
        recording: Some((Arc::clone(state), index)),
    }
}

async fn run<T, F, Fut>(span: InMemorySpan, callback: F) -> T
where
    F: FnOnce(InMemorySpan) -> Fut,
    Fut: Future<Output = T>,
{
    let mut guard = SpanGuard {
        span: span.clone(),
        completed: false,
    };
    let result = callback(span).await;
    guard.span.settle(false);
    guard.completed = true;
    result
}

impl TelemetryContext for InMemoryTelemetryContext {
    type Span = InMemorySpan;
    async fn start_span<T, F, Fut>(&self, options: SpanOptions, callback: F) -> T
    where
        T: Send,
        F: FnOnce(Self::Span) -> Fut + Send,
        Fut: Future<Output = T> + Send,
    {
        run(admit(&self.state, None, options), callback).await
    }
}

impl TelemetryContext for InMemorySpan {
    type Span = Self;
    async fn start_span<T, F, Fut>(&self, options: SpanOptions, callback: F) -> T
    where
        T: Send,
        F: FnOnce(Self) -> Fut + Send,
        Fut: Future<Output = T> + Send,
    {
        let child = match &self.recording {
            Some((state, index)) => admit(state, Some(*index), options),
            None => Self::default(),
        };
        run(child, callback).await
    }
}
