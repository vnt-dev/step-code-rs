//! Compile-time telemetry contracts using Rust structs and enums.
//!
//! Define required attributes as struct fields, optional attributes as Option,
//! and closed values/events as enums. The compiler checks these contracts;
//! serializable schema metadata is independent of them.
//!
//! ```
//! use step_harness_telemetry::*;
//!
//! struct Load;
//! struct LoadInput { account_id: u64 }
//! impl From<LoadInput> for SpanAttributes {
//!     fn from(value: LoadInput) -> Self {
//!         [("account.id".into(), value.account_id.into())].into()
//!     }
//! }
//! impl SpanDefinition for Load {
//!     const NAME: &'static str = "account.load";
//!     type StartAttributes = LoadInput;
//!     type EndAttributes = SpanAttributes;
//!     type Event = std::convert::Infallible; // no events allowed
//! }
//!
//! # async fn example() {
//! let context = InMemoryTelemetryContext::new();
//! let typed = TypedTelemetryContext::new(&context);
//! let value = typed.start_span::<Load, _, _, _>(
//!     LoadInput { account_id: u64::MAX }, |_span| async { 42 }
//! ).await;
//! assert_eq!(value, 42);
//! # }
//! ```
//!
//! Missing or incorrect start attributes are compile-time errors:
//!
//! ```compile_fail
//! use step_harness_telemetry::*;
//! struct Operation;
//! struct Input { id: u64 }
//! impl From<Input> for SpanAttributes {
//!     fn from(value: Input) -> Self { [("id".into(), value.id.into())].into() }
//! }
//! impl SpanDefinition for Operation {
//!     const NAME: &'static str = "operation";
//!     type StartAttributes = Input;
//!     type EndAttributes = SpanAttributes;
//!     type Event = std::convert::Infallible;
//! }
//! let context = NoopTelemetryContext;
//! let typed = TypedTelemetryContext::new(&context);
//! let future = typed.start_span::<Operation, _, _, _>(
//!     SpanAttributes::new(), |_| async {}
//! );
//! ```
//!
//! Incorrect end attributes and undeclared events are also rejected:
//!
//! ```compile_fail
//! use step_harness_telemetry::*;
//! struct Operation;
//! impl SpanDefinition for Operation {
//!     const NAME: &'static str = "operation";
//!     type StartAttributes = NoAttributes;
//!     type EndAttributes = NoAttributes;
//!     type Event = std::convert::Infallible;
//! }
//! let context = NoopTelemetryContext;
//! let typed = TypedTelemetryContext::new(&context);
//! let future = typed.start_span::<Operation, _, _, _>(NoAttributes, |span| async move {
//!     span.set_attributes(SpanAttributes::new());
//! });
//! ```
//!
//! ```compile_fail
//! use step_harness_telemetry::*;
//! struct Operation;
//! impl SpanDefinition for Operation {
//!     const NAME: &'static str = "operation";
//!     type StartAttributes = NoAttributes;
//!     type EndAttributes = NoAttributes;
//!     type Event = std::convert::Infallible;
//! }
//! let context = NoopTelemetryContext;
//! let typed = TypedTelemetryContext::new(&context);
//! let future = typed.start_span::<Operation, _, _, _>(NoAttributes, |span| async move {
//!     span.add_event("unknown.event");
//! });
//! ```

use crate::{SpanAttributes, SpanOptions, SpanStatus, TelemetryContext, TelemetrySpan};
use std::convert::Infallible;
use std::fmt::Display;
use std::future::Future;
use std::marker::PhantomData;

/// Domain marker binding a span name to its allowed start/end/event types.
/// Use newtypes instead of open maps to enforce exact attribute sets.
pub trait SpanDefinition {
    const NAME: &'static str;
    type StartAttributes: Into<SpanAttributes>;
    type EndAttributes: Into<SpanAttributes>;
    type Event: TelemetryEvent;
}

/// Usually implemented by a domain enum whose variants carry event attributes.
pub trait TelemetryEvent {
    fn into_parts(self) -> (&'static str, SpanAttributes);
}

impl TelemetryEvent for Infallible {
    fn into_parts(self) -> (&'static str, SpanAttributes) {
        match self {}
    }
}

/// Borrowed context; supports both root contexts and callback span handles.
/// Multiple schemas are represented by multiple SpanDefinition implementations.
pub struct TypedTelemetryContext<'a, C: TelemetryContext> {
    context: &'a C,
}

impl<'a, C: TelemetryContext> TypedTelemetryContext<'a, C> {
    pub fn new(context: &'a C) -> Self {
        Self { context }
    }

    pub fn start_span<D, T, F, Fut>(
        &self,
        attributes: D::StartAttributes,
        callback: F,
    ) -> impl Future<Output = T> + Send
    where
        D: SpanDefinition,
        T: Send,
        F: FnOnce(TypedSpan<C::Span, D>) -> Fut + Send,
        Fut: Future<Output = T> + Send,
    {
        let options = SpanOptions::new(D::NAME).with_attributes(attributes.into());
        self.context
            .start_span(options, move |span| callback(TypedSpan::new(span)))
    }

    pub fn start_result_span<D, T, E, F, Fut>(
        &self,
        attributes: D::StartAttributes,
        callback: F,
    ) -> impl Future<Output = Result<T, E>> + Send
    where
        D: SpanDefinition,
        T: Send,
        E: Display + Send,
        F: FnOnce(TypedSpan<C::Span, D>) -> Fut + Send,
        Fut: Future<Output = Result<T, E>> + Send,
    {
        let options = SpanOptions::new(D::NAME).with_attributes(attributes.into());
        self.context
            .start_result_span(options, move |span| callback(TypedSpan::new(span)))
    }
}

/// Recording handle exposing only the chosen domain's end attributes/events.
pub struct TypedSpan<S: TelemetrySpan, D: SpanDefinition> {
    span: S,
    definition: PhantomData<fn() -> D>,
}

impl<S: TelemetrySpan, D: SpanDefinition> Clone for TypedSpan<S, D> {
    fn clone(&self) -> Self {
        Self::new(self.span.clone())
    }
}

impl<S: TelemetrySpan, D: SpanDefinition> TypedSpan<S, D> {
    fn new(span: S) -> Self {
        Self {
            span,
            definition: PhantomData,
        }
    }

    pub fn set_attributes(&self, attributes: D::EndAttributes) {
        self.span.set_attributes(attributes.into());
    }

    pub fn add_event(&self, event: D::Event) {
        let (name, attributes) = event.into_parts();
        self.span.add_event(name, attributes);
    }

    pub fn set_status(&self, status: SpanStatus) {
        self.span.set_status(status);
    }

    pub fn children(&self) -> TypedTelemetryContext<'_, S> {
        TypedTelemetryContext::new(&self.span)
    }
}
