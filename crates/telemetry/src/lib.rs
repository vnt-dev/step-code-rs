//! Explicit, backend-neutral telemetry for Rust applications.
//!
//! Use [TelemetryContext::start_span] for arbitrary values and
//! [TelemetryContext::start_result_span] for fallible operations. Callbacks
//! run lazily on the first poll. Adapters end admitted spans on completion or
//! when their future is dropped. Panics are not caught or used as business errors.
//!
//! [schema] contains serializable metadata; [typed] provides compile-time
//! contracts using application-defined attribute structs and event enums.
#![forbid(unsafe_code)]

mod attributes;
pub mod memory;
pub mod noop;
pub mod schema;
pub mod testing;
pub mod typed;

pub use attributes::{AttributeValue, NoAttributes, SpanAttributes};
pub use memory::{
    InMemorySpan, InMemoryTelemetryContext, RecordedTelemetryEvent, RecordedTelemetrySpan,
};
pub use noop::{NOOP_TELEMETRY_CONTEXT, NoopSpan, NoopTelemetryContext};
pub use schema::*;
pub use typed::{SpanDefinition, TelemetryEvent, TypedSpan, TypedTelemetryContext};

use std::fmt::Display;
use std::future::Future;

/// Name and initial attributes. An empty map means no attributes.
#[derive(Debug, Clone, PartialEq)]
pub struct SpanOptions {
    pub name: String,
    pub attributes: SpanAttributes,
}

impl SpanOptions {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            attributes: SpanAttributes::new(),
        }
    }

    pub fn with_attributes(mut self, attributes: SpanAttributes) -> Self {
        self.attributes = attributes;
        self
    }

    pub fn with_attribute(
        mut self,
        name: impl Into<String>,
        value: impl Into<AttributeValue>,
    ) -> Self {
        self.attributes.insert(name.into(), value.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorRecord {
    pub name: String,
    pub message: String,
}

/// Operation status. Cancelled means the admitted operation was dropped
/// before normal completion; it does not classify the cause of that drop.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum SpanStatus {
    #[default]
    Ok,
    Error {
        error: Option<ErrorRecord>,
    },
    Cancelled,
}

impl SpanStatus {
    pub fn error(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Error {
            error: Some(ErrorRecord {
                name: name.into(),
                message: message.into(),
            }),
        }
    }

    pub fn bare_error() -> Self {
        Self::Error { error: None }
    }
}

/// Explicit parent context. Implementations must invoke callbacks once,
/// preserve their values, and settle admitted spans on completion or drop.
/// Recording on a settled span is inert, including recording new children.
///
/// Generic methods use static dispatch. Contexts may borrow application state;
/// only callbacks/futures sent to a runtime need to satisfy its 'static bound.
pub trait TelemetryContext: Send + Sync {
    type Span: TelemetrySpan;

    fn start_span<T, F, Fut>(
        &self,
        options: SpanOptions,
        callback: F,
    ) -> impl Future<Output = T> + Send
    where
        T: Send,
        F: FnOnce(Self::Span) -> Fut + Send,
        Fut: Future<Output = T> + Send;

    /// Preserve Result<T, E> and automatically record Err unless the
    /// callback set an explicit status. Error values need not implement Error,
    /// Clone, or 'static. Their Display text is recorded; redact it if needed.
    fn start_result_span<T, E, F, Fut>(
        &self,
        options: SpanOptions,
        callback: F,
    ) -> impl Future<Output = Result<T, E>> + Send
    where
        T: Send,
        E: Display + Send,
        F: FnOnce(Self::Span) -> Fut + Send,
        Fut: Future<Output = Result<T, E>> + Send,
    {
        self.start_span(options, move |span| async move {
            let result = callback(span.clone()).await;
            if let Err(error) = &result
                && span.is_recording()
            {
                span.set_default_status(SpanStatus::error(
                    std::any::type_name::<E>(),
                    error.to_string(),
                ));
            }
            result
        })
    }
}

/// A cheap, clonable recording handle that is itself an explicit parent.
/// Cloning a handle does not extend the operation's lifetime.
pub trait TelemetrySpan: TelemetryContext<Span = Self> + Clone {
    /// Whether this handle currently accepts data. A hint: concurrent
    /// settlement can make it inert immediately after this method returns.
    fn is_recording(&self) -> bool;
    fn add_event(&self, name: &str, attributes: SpanAttributes);
    fn set_attributes(&self, attributes: SpanAttributes);
    /// Explicit status is last-write-wins and takes precedence over defaults.
    fn set_status(&self, status: SpanStatus);
    /// Automatic status does not overwrite explicit status or settled spans.
    fn set_default_status(&self, status: SpanStatus);
}
