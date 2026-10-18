//! Inert context and span handles.
use crate::{SpanAttributes, SpanOptions, SpanStatus, TelemetryContext, TelemetrySpan};
use std::future::Future;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NoopTelemetryContext;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NoopSpan;
pub const NOOP_TELEMETRY_CONTEXT: NoopTelemetryContext = NoopTelemetryContext;

impl TelemetryContext for NoopTelemetryContext {
    type Span = NoopSpan;
    async fn start_span<T, F, Fut>(&self, _options: SpanOptions, callback: F) -> T
    where
        T: Send,
        F: FnOnce(NoopSpan) -> Fut + Send,
        Fut: Future<Output = T> + Send,
    {
        callback(NoopSpan).await
    }
}

impl TelemetryContext for NoopSpan {
    type Span = Self;
    async fn start_span<T, F, Fut>(&self, _options: SpanOptions, callback: F) -> T
    where
        T: Send,
        F: FnOnce(Self) -> Fut + Send,
        Fut: Future<Output = T> + Send,
    {
        callback(*self).await
    }
}

impl TelemetrySpan for NoopSpan {
    fn is_recording(&self) -> bool {
        false
    }
    fn add_event(&self, _name: &str, _attributes: SpanAttributes) {}
    fn set_attributes(&self, _attributes: SpanAttributes) {}
    fn set_status(&self, _status: SpanStatus) {}
    fn set_default_status(&self, _status: SpanStatus) {}
}
