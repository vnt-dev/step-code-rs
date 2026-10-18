use futures_util::{FutureExt, future::join};
use std::fmt;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use step_harness_telemetry::*;

#[test]
fn integers_remain_exact_and_empty_arrays_keep_their_types() {
    let values = [
        AttributeValue::from(i64::MIN),
        AttributeValue::from(i64::MAX),
        AttributeValue::from(u64::MAX),
        AttributeValue::from(9_007_199_254_740_993_i64),
        AttributeValue::from(1.25_f64),
        AttributeValue::Integers(vec![]),
        AttributeValue::UnsignedIntegers(vec![]),
        AttributeValue::Floats(vec![]),
        AttributeValue::Strings(vec![]),
        AttributeValue::Booleans(vec![]),
    ];
    for value in values {
        let json = serde_json::to_string(&value).unwrap();
        let decoded: AttributeValue = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, value);
    }
    assert_ne!(
        AttributeValue::from(9_007_199_254_740_992_i64),
        AttributeValue::from(9_007_199_254_740_993_i64)
    );
    assert_eq!(AttributeValue::from(42), AttributeValue::Integer(42));
    assert_eq!(
        AttributeValue::from(42_u32),
        AttributeValue::UnsignedInteger(42)
    );
}

#[test]
fn schema_tags_resolve_empty_value_sets_and_nested_examples() {
    let cases = [
        ("string", r#"[]"#),
        ("integer", r#"[]"#),
        ("unsigned_integer", r#"[]"#),
        ("float", r#"[]"#),
        ("boolean", r#"[]"#),
        ("string[]", r#"[[]]"#),
        ("integer[]", r#"[[]]"#),
        ("unsigned_integer[]", r#"[[]]"#),
        ("float[]", r#"[[]]"#),
        ("boolean[]", r#"[[]]"#),
    ];
    for (kind, examples) in cases {
        let allowed = if kind.ends_with("[]") {
            "elementValues"
        } else {
            "values"
        };
        let input = format!(
            r#"{{"type":"{kind}","description":"example","{allowed}":[],"examples":{examples}}}"#
        );
        let definition: TelemetryAttributeDefinition = serde_json::from_str(&input).unwrap();
        let json = serde_json::to_value(&definition).unwrap();
        assert_eq!(
            json,
            serde_json::from_str::<serde_json::Value>(&input).unwrap()
        );
        let decoded: TelemetryAttributeDefinition = serde_json::from_value(json).unwrap();
        assert_eq!(decoded, definition);
    }
    let invalid = [
        r#"{"type":"integer","description":"test","values":[1.5]}"#,
        r#"{"type":"integer","description":"test","values":["1"]}"#,
        r#"{"type":"unsigned_integer","description":"test","values":[-1]}"#,
        r#"{"type":"number","description":"test"}"#,
    ];
    for json in invalid {
        assert!(
            serde_json::from_str::<TelemetryAttributeDefinition>(json).is_err(),
            "{json}"
        );
    }
}

#[test]
fn complete_schema_preserves_integer_boundaries_and_metadata() {
    let json = r#"{
        "version":1,
        "spans":{
            "operation":{
                "description":"Read a resource",
                "parents":{"kind":"root_or_external"},
                "startAttributes":{
                    "id":{"type":"unsigned_integer","description":"ID","values":[18446744073709551615],"required":true,"sensitive":false,"cardinality":"high"}
                },
                "endAttributes":{
                    "count":{"type":"integer","description":"Count","examples":[9223372036854775807]}
                },
                "events":{
                    "result":{"description":"Result","attributes":{
                        "samples":{"type":"integer[]","description":"Samples","elementValues":[],"examples":[[]],"required":false}
                    }}
                },
                "status":{"default":"ok","errorWhen":"Read fails"}
            }
        }
    }"#;
    let schema: TelemetrySchemaDefinition = serde_json::from_str(json).unwrap();
    let encoded = serde_json::to_value(&schema).unwrap();
    assert_eq!(
        encoded,
        serde_json::from_str::<serde_json::Value>(json).unwrap()
    );
    assert_eq!(
        serde_json::from_value::<TelemetrySchemaDefinition>(encoded).unwrap(),
        schema
    );
}

#[tokio::test]
async fn dropping_an_unpolled_future_never_admits_work() {
    let context = InMemoryTelemetryContext::new();
    let calls = AtomicUsize::new(0);
    let future = context.start_span(SpanOptions::new("never-started"), |_| {
        calls.fetch_add(1, Ordering::Relaxed);
        async {}
    });
    drop(future);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert!(context.get_spans().is_empty());
}

#[tokio::test]
async fn cancellation_settles_parent_and_child_and_makes_escaped_handles_inert() {
    let context = InMemoryTelemetryContext::new();
    let escaped = Arc::new(parking_lot::Mutex::new(None));
    let capture = Arc::clone(&escaped);
    let mut future = Box::pin(
        context.start_span(SpanOptions::new("parent"), move |parent| {
            *capture.lock() = Some(parent.clone());
            async move {
                parent
                    .start_span(SpanOptions::new("child"), |_| std::future::pending::<()>())
                    .await;
            }
        }),
    );
    assert!(future.as_mut().now_or_never().is_none());
    assert_eq!(context.get_spans().len(), 2);
    drop(future);
    let spans = context.get_spans();
    assert!(spans.iter().all(|span| span.settled
        && span.status == SpanStatus::Cancelled
        && span.end_sequence.is_some()));
    let parent: InMemorySpan = escaped.lock().take().unwrap();
    parent.add_event("late", SpanAttributes::new());
    let value = parent
        .start_span(SpanOptions::new("late-child"), |_| async { 9 })
        .await;
    assert_eq!(value, 9);
    assert_eq!(context.get_spans(), spans);
}

#[tokio::test]
async fn cancellation_preserves_explicit_status_and_ends_exactly_once() {
    let context = InMemoryTelemetryContext::new();
    let mut future = Box::pin(context.start_span(SpanOptions::new("cancelled"), |span| {
        span.set_status(SpanStatus::bare_error());
        std::future::pending::<()>()
    }));
    assert!(future.as_mut().now_or_never().is_none());
    drop(future);
    context
        .start_span(SpanOptions::new("next"), |_| async {})
        .await;
    let spans = context.get_spans();
    assert!(spans[0].settled);
    assert_eq!(spans[0].status, SpanStatus::bare_error());
    assert_eq!(spans[0].end_sequence, Some(1));
    assert_eq!(spans[1].end_sequence, Some(2));
}

#[tokio::test]
async fn aborting_a_running_task_settles_its_span() {
    let context = InMemoryTelemetryContext::new();
    let shared = context.clone();
    let (started, wait) = tokio::sync::oneshot::channel();
    let task = tokio::spawn(async move {
        shared
            .start_span(SpanOptions::new("aborted"), |_| async {
                started.send(()).unwrap();
                std::future::pending::<()>().await
            })
            .await
    });
    wait.await.unwrap();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    let spans = context.get_spans();
    assert!(spans[0].settled);
    assert_eq!(spans[0].status, SpanStatus::Cancelled);
}

#[derive(Debug)]
struct BusinessError(String);
impl fmt::Display for BusinessError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

#[tokio::test]
async fn result_errors_keep_identity_and_explicit_status_takes_precedence() {
    let context = InMemoryTelemetryContext::new();
    let error = Box::new(BusinessError("failed".into()));
    let address = (&*error) as *const BusinessError;
    let result = context
        .start_result_span(
            SpanOptions::new("error"),
            |_| async move { Err::<(), _>(error) },
        )
        .await;
    let error = result.unwrap_err();
    assert_eq!((&*error) as *const BusinessError, address);
    let spans = context.get_spans();
    assert!(spans[0].settled);
    assert!(
        matches!(&spans[0].status, SpanStatus::Error { error: Some(record) } if record.message == "failed")
    );

    let result = context
        .start_result_span(SpanOptions::new("explicit"), |span| async move {
            span.set_status(SpanStatus::Ok);
            Err::<(), _>("classified by caller")
        })
        .await;
    assert!(result.is_err());
    assert_eq!(context.get_spans()[1].status, SpanStatus::Ok);

    let result = context
        .start_result_span(SpanOptions::new("success"), |_| async { Ok::<_, &str>(42) })
        .await;
    assert_eq!(result, Ok(42));
    assert_eq!(context.get_spans()[2].status, SpanStatus::Ok);
}

#[tokio::test]
async fn snapshots_are_detached_and_recording_merges_attributes() {
    let context = InMemoryTelemetryContext::new();
    let snapshot_context = &context;
    context
        .start_span(
            SpanOptions::new("snapshot").with_attribute("tags", vec!["initial".to_owned()]),
            |span| async move {
                let open = snapshot_context.get_spans();
                assert!(!open[0].settled);
                span.set_attributes([("count".into(), u64::MAX.into())].into());
                span.add_event("event", [("id".into(), i64::MAX.into())].into());
            },
        )
        .await;
    let original = context.get_spans();
    let mut changed = original.clone();
    changed[0].attributes.clear();
    changed[0].events[0].attributes.clear();
    assert_eq!(context.get_spans(), original);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn multiple_threads_record_overlapping_children_without_losing_events() {
    let context = InMemoryTelemetryContext::new();
    let snapshot_context = &context;
    context
        .start_span(SpanOptions::new("parent"), |parent| async move {
            let barrier = Arc::new(tokio::sync::Barrier::new(5));
            let release = Arc::new(tokio::sync::Barrier::new(5));
            let mut tasks = Vec::new();
            for worker in 0..4 {
                let parent = parent.clone();
                let barrier = Arc::clone(&barrier);
                let release = Arc::clone(&release);
                tasks.push(tokio::spawn(async move {
                    parent
                        .start_span(
                            SpanOptions::new(format!("worker-{worker}")),
                            |child| async move {
                                for index in 0..100 {
                                    child
                                        .add_event("item", [("index".into(), index.into())].into());
                                }
                                barrier.wait().await;
                                release.wait().await;
                            },
                        )
                        .await;
                }));
            }
            tokio::time::timeout(std::time::Duration::from_secs(5), barrier.wait())
                .await
                .unwrap();
            let open = snapshot_context.get_spans();
            assert_eq!(open.len(), 5);
            assert!(open.iter().all(|span| !span.settled));
            release.wait().await;
            for task in tasks {
                task.await.unwrap();
            }
        })
        .await;
    let spans = context.get_spans();
    assert!(spans.iter().all(|span| span.settled));
    assert!(
        spans[1..]
            .iter()
            .all(|span| span.parent_id == Some(spans[0].id) && span.events.len() == 100)
    );
    assert_eq!(spans[0].end_sequence, Some(5));
    let sequences: std::collections::BTreeSet<_> = spans
        .iter()
        .map(|span| span.end_sequence.unwrap())
        .collect();
    assert_eq!(sequences.len(), 5);
}

#[tokio::test]
async fn noop_preserves_values_errors_and_nested_callbacks() {
    let result = NOOP_TELEMETRY_CONTEXT
        .start_result_span(SpanOptions::new("root"), |span| async move {
            span.add_event("ignored", SpanAttributes::new());
            span.set_status(SpanStatus::bare_error());
            span.start_result_span(SpanOptions::new("child"), |_| async {
                Err::<(), _>("unchanged")
            })
            .await
        })
        .await;
    assert_eq!(result, Err("unchanged"));
    assert_eq!(
        NOOP_TELEMETRY_CONTEXT
            .start_span(SpanOptions::new("value"), |_| async { 42 })
            .await,
        42
    );
}

struct CountedError(Arc<AtomicUsize>);

impl fmt::Display for CountedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fetch_add(1, Ordering::Relaxed);
        formatter.write_str("should not be inspected by an inert span")
    }
}

#[tokio::test]
async fn inert_spans_do_not_format_result_errors() {
    let formats = Arc::new(AtomicUsize::new(0));
    let error = CountedError(Arc::clone(&formats));
    let result = NOOP_TELEMETRY_CONTEXT
        .start_result_span(
            SpanOptions::new("noop"),
            |_| async move { Err::<(), _>(error) },
        )
        .await;
    assert!(result.is_err());

    let context = InMemoryTelemetryContext::new();
    let parent = context
        .start_span(SpanOptions::new("parent"), |span| async move { span })
        .await;
    assert!(!parent.is_recording());
    let error = CountedError(Arc::clone(&formats));
    let result = parent
        .start_result_span(SpanOptions::new("late-child"), |_| async move {
            Err::<(), _>(error)
        })
        .await;
    assert!(result.is_err());
    assert_eq!(formats.load(Ordering::Relaxed), 0);
    assert_eq!(context.get_spans().len(), 1);
}

#[derive(Clone, Copy)]
enum OperationKind {
    Read,
    Write,
}
struct OperationInput {
    id: u64,
    kind: OperationKind,
}
struct OperationOutput {
    count: u64,
}
enum OperationEvent {
    Result { found: bool },
}
struct Operation;
struct Request;
struct RequestInput {
    provider: String,
}

impl From<OperationInput> for SpanAttributes {
    fn from(input: OperationInput) -> Self {
        let kind = match input.kind {
            OperationKind::Read => "read",
            OperationKind::Write => "write",
        };
        [("id".into(), input.id.into()), ("kind".into(), kind.into())].into()
    }
}
impl From<OperationOutput> for SpanAttributes {
    fn from(output: OperationOutput) -> Self {
        [("count".into(), output.count.into())].into()
    }
}
impl From<RequestInput> for SpanAttributes {
    fn from(input: RequestInput) -> Self {
        [("provider".into(), input.provider.into())].into()
    }
}
impl TelemetryEvent for OperationEvent {
    fn into_parts(self) -> (&'static str, SpanAttributes) {
        match self {
            Self::Result { found } => ("result", [("found".into(), found.into())].into()),
        }
    }
}
impl SpanDefinition for Operation {
    const NAME: &'static str = "operation";
    type StartAttributes = OperationInput;
    type EndAttributes = OperationOutput;
    type Event = OperationEvent;
}
impl SpanDefinition for Request {
    const NAME: &'static str = "request";
    type StartAttributes = RequestInput;
    type EndAttributes = NoAttributes;
    type Event = std::convert::Infallible;
}

#[tokio::test]
async fn typed_contracts_record_domain_attributes_events_and_nested_spans() {
    let context = InMemoryTelemetryContext::new();
    let typed = TypedTelemetryContext::new(&context);
    let result = typed
        .start_result_span::<Operation, _, _, _, _>(
            OperationInput {
                id: u64::MAX,
                kind: OperationKind::Read,
            },
            |operation| async move {
                operation.add_event(OperationEvent::Result { found: true });
                let value = operation
                    .children()
                    .start_span::<Request, _, _, _>(
                        RequestInput {
                            provider: "example".into(),
                        },
                        |_| async { 42 },
                    )
                    .await;
                operation.set_attributes(OperationOutput { count: 1 });
                Ok::<_, &str>(value)
            },
        )
        .await;
    assert_eq!(result, Ok(42));
    let spans = context.get_spans();
    assert_eq!(spans[1].parent_id, Some(spans[0].id));
    assert_eq!(
        spans[0].attributes["id"],
        AttributeValue::UnsignedInteger(u64::MAX)
    );
    assert_eq!(
        spans[0].attributes["count"],
        AttributeValue::UnsignedInteger(1)
    );
    assert_eq!(spans[0].events[0].name, "result");
    let _ = typed
        .start_span::<Operation, _, _, _>(
            OperationInput {
                id: 0,
                kind: OperationKind::Write,
            },
            |_| async {},
        )
        .await;
}

#[tokio::test]
async fn callbacks_can_borrow_application_values() {
    let context = InMemoryTelemetryContext::new();
    let name = String::from("borrowed");
    let result = context
        .start_result_span(SpanOptions::new("borrow"), |_| async {
            Ok::<_, &str>(name.as_str())
        })
        .await;
    assert_eq!(result, Ok("borrowed"));
    let (first, second) = join(
        context.start_span(SpanOptions::new("first"), |_| async { 1 }),
        context.start_span(SpanOptions::new("second"), |_| async { 2 }),
    )
    .await;
    assert_eq!((first, second), (1, 2));
}
