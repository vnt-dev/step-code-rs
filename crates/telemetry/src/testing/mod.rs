//! Adapter conformance checks for Rust lifecycle semantics.
//!
//! Each group uses a fresh fixture. Checks cover callback admission, values and
//! Result errors, attributes/events, cancellation, inert settled handles, and
//! genuinely overlapping children. There are no panic payload requirements.

use crate::{
    RecordedTelemetrySpan, SpanAttributes, SpanOptions, SpanStatus, TelemetryContext, TelemetrySpan,
};
use futures_util::{FutureExt, future::join};
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};

pub trait TelemetryAdapterFixture: Send + Sync {
    type Context: TelemetryContext;
    fn context(&self) -> &Self::Context;
    /// Flush exporters if necessary, then return detached snapshots, including
    /// open spans. The suite uses these to verify lifecycle and overlap.
    fn get_spans(&self) -> impl Future<Output = Vec<RecordedTelemetrySpan>> + Send;
}

fn check(condition: bool, message: &str) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(message.to_owned())
    }
}

/// Check an adapter using a factory that produces isolated fixtures.
/// Panics remain the runner's responsibility; expected failures use Result.
pub async fn check_adapter_conformance<Factory, FactoryFuture, Fixture>(
    factory: Factory,
) -> Result<(), String>
where
    Factory: Fn() -> FactoryFuture,
    FactoryFuture: Future<Output = Fixture>,
    Fixture: TelemetryAdapterFixture,
{
    let fixture = factory().await;
    let calls = AtomicUsize::new(0);
    let mut future = Box::pin(
        fixture
            .context()
            .start_span(SpanOptions::new("value"), |_| {
                calls.fetch_add(1, Ordering::Relaxed);
                async { 42 }
            }),
    );
    check(calls.load(Ordering::Relaxed) == 0, "callback must be lazy")?;
    check(future.as_mut().await == 42, "callback value changed")?;
    drop(future);
    check(
        calls.load(Ordering::Relaxed) == 1,
        "callback must run exactly once",
    )?;
    let spans = fixture.get_spans().await;
    check(
        spans.len() == 1 && spans[0].settled && spans[0].status == SpanStatus::Ok,
        "successful span was not settled as Ok",
    )?;

    let fixture = factory().await;
    let result = fixture
        .context()
        .start_result_span(SpanOptions::new("failure"), |_| async {
            Err::<(), _>("expected failure")
        })
        .await;
    check(result == Err("expected failure"), "Result error changed")?;
    let spans = fixture.get_spans().await;
    check(
        spans.len() == 1 && spans[0].settled && matches!(spans[0].status, SpanStatus::Error { .. }),
        "Result::Err must settle with an error status",
    )?;

    let fixture = factory().await;
    let handle = fixture
        .context()
        .start_span(
            SpanOptions::new("recording").with_attribute("count", 1_i64),
            |span| {
                span.set_attributes([("count".into(), 2_i64.into())].into());
                span.add_event("first", SpanAttributes::new());
                span.add_event("second", [("index".into(), 2_u64.into())].into());
                span.set_status(SpanStatus::bare_error());
                span.set_status(SpanStatus::Ok);
                span.set_default_status(SpanStatus::bare_error());
                async move { span }
            },
        )
        .await;
    handle.add_event("late", SpanAttributes::new());
    handle.set_attributes([("count".into(), 3_i64.into())].into());
    handle.set_status(SpanStatus::bare_error());
    handle.set_default_status(SpanStatus::bare_error());
    check(
        handle
            .start_span(SpanOptions::new("late-child"), |_| async { 7 })
            .await
            == 7,
        "late child must still preserve its callback result",
    )?;
    let spans = fixture.get_spans().await;
    check(spans.len() == 1, "late child must not be recorded")?;
    let span = &spans[0];
    check(
        span.attributes.get("count") == Some(&2_i64.into()),
        "attribute merge or late-write handling failed",
    )?;
    check(
        span.events.len() == 2 && span.events[0].name == "first" && span.events[1].name == "second",
        "event ordering or late-event handling failed",
    )?;
    check(
        span.status == SpanStatus::Ok,
        "explicit status precedence failed",
    )?;

    let fixture = factory().await;
    let result = fixture
        .context()
        .start_result_span(SpanOptions::new("explicit"), |span| {
            span.set_status(SpanStatus::Ok);
            async { Err::<(), _>("explicitly classified") }
        })
        .await;
    check(
        result.is_err(),
        "explicit status must not change the result",
    )?;
    let spans = fixture.get_spans().await;
    check(
        spans.len() == 1 && spans[0].status == SpanStatus::Ok,
        "automatic error must not replace explicit status",
    )?;

    let fixture = factory().await;
    let mut future = Box::pin(
        fixture
            .context()
            .start_span(SpanOptions::new("cancelled"), |_| {
                std::future::pending::<()>()
            }),
    );
    check(
        future.as_mut().now_or_never().is_none(),
        "pending callback completed unexpectedly",
    )?;
    drop(future);
    let spans = fixture.get_spans().await;
    check(
        spans.len() == 1
            && spans[0].settled
            && spans[0].status == SpanStatus::Cancelled
            && spans[0].end_sequence.is_some(),
        "dropped future must settle its admitted span",
    )?;

    let fixture = factory().await;
    let (release, wait) = futures_channel::oneshot::channel::<()>();
    let fixture_ref = &fixture;
    let result = fixture
        .context()
        .start_span(SpanOptions::new("parent"), move |parent| async move {
            let first = parent.start_span(SpanOptions::new("first"), |_| async move {
                wait.await.map_err(|_| "child gate was dropped".to_owned())
            });
            let second = parent.start_span(SpanOptions::new("second"), |_| async move {
                let spans = fixture_ref.get_spans().await;
                let overlap = spans
                    .iter()
                    .any(|span| span.name == "first" && !span.settled);
                // Release even when the check fails, so the conformance suite cannot
                // leave the other branch pending due to a failed overlap assertion.
                let _ = release.send(());
                check(overlap, "first child must be open while second child runs")
            });
            let (first, second) = join(first, second).await;
            first?;
            second
        })
        .await;
    result?;
    let spans = fixture.get_spans().await;
    check(
        spans.len() == 3,
        "expected one parent and two overlapping children",
    )?;
    let parent = spans
        .iter()
        .find(|span| span.name == "parent")
        .ok_or("missing parent")?;
    let first = spans
        .iter()
        .find(|span| span.name == "first")
        .ok_or("missing first child")?;
    let second = spans
        .iter()
        .find(|span| span.name == "second")
        .ok_or("missing second child")?;
    check(
        parent.parent_id.is_none()
            && first.parent_id == Some(parent.id)
            && second.parent_id == Some(parent.id),
        "incorrect parent relationships",
    )?;
    check(
        first.settled
            && second.settled
            && parent.settled
            && second.end_sequence.is_some()
            && first.end_sequence.is_some()
            && parent.end_sequence.is_some()
            && second.end_sequence < first.end_sequence
            && first.end_sequence < parent.end_sequence,
        "incorrect overlapping child settlement order",
    )?;
    Ok(())
}
