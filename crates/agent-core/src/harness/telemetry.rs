//! Agent-owned telemetry vocabulary: the AI-request schema, the harness schema,
//! and the typed span helpers that record against them.
//!
//! Ported from `@step-harness/agent-core` `src\harness\telemetry.ts`.
//!
//! # Notable differences from the TypeScript source
//!
//! - The TypeScript schema is a compile-time literal (`as const satisfies
//!   TelemetrySchemaDefinition`) whose types derive span names and exact
//!   attribute sets. [`step_harness_telemetry::schema`] models the same data as
//!   runtime values, so the schemas here are [`std::sync::LazyLock`] statics and
//!   span names are plain strings; [`start_ai_span`] and [`start_harness_span`]
//!   assert (in debug builds) that the requested name is declared.
//! - The TypeScript `type: "number"` attributes are ambiguous. They are split by
//!   meaning into `unsigned_integer` (HTTP status, token counts, chunk counts,
//!   elapsed milliseconds, attempt numbers, delays, sequence numbers) and
//!   `float` (cost).
//! - [`ExactTelemetryAttributes`] has no Rust counterpart: attributes are a
//!   [`SpanAttributes`] map, and [`step_harness_telemetry::typed`] provides the
//!   compile-time contract for call sites that declare a
//!   [`SpanDefinition`](step_harness_telemetry::typed::SpanDefinition).

use std::collections::BTreeMap;
use std::future::Future;
use std::sync::LazyLock;

use step_harness_telemetry::schema::{
    TelemetryAttributeCardinality, TelemetryAttributeDefinition, TelemetryAttributeMetadata,
    TelemetryParentDefinition, TelemetrySchemaDefinition, TelemetrySpanDefinition,
    TelemetrySpanStatusDefault, TelemetrySpanStatusDefinition, TelemetryStartAttributeDefinition,
};
use step_harness_telemetry::{SpanAttributes, SpanOptions, TelemetryContext};

/// Hook names declared by `pi.harness.hook`'s `pi.hook.name` attribute.
///
/// Kept private: the schema below is the only consumer, mirroring the
/// TypeScript module-private constant.
const HOOK_NAMES: [&str; 11] = [
    "before_run",
    "before_resume",
    "before_run_end",
    "transform_context",
    "before_request",
    "before_payload",
    "after_response",
    "before_tool",
    "after_tool",
    "before_compaction",
    "before_navigation",
];

/// Delivered harness event types declared by `pi.harness.event_handler`'s
/// `pi.event.type` attribute.
const EVENT_TYPES: [&str; 29] = [
    "run_start",
    "run_resume",
    "run_suspend",
    "run_abort",
    "run_end",
    "fault",
    "handler_error",
    "turn_start",
    "turn_end",
    "retry_scheduled",
    "retry_start",
    "retry_end",
    "message_start",
    "message_update",
    "message_end",
    "tool_start",
    "tool_update",
    "tool_end",
    "entry_added",
    "write_pending",
    "queue_update",
    "fact_update",
    "config_update",
    "compaction_start",
    "compaction_end",
    "navigation_start",
    "navigation_end",
    "lane_created",
    "usage",
];

// ---------------------------------------------------------------------------
// Attribute constructors
// ---------------------------------------------------------------------------

fn metadata(
    description: &str,
    cardinality: Option<TelemetryAttributeCardinality>,
) -> TelemetryAttributeMetadata {
    TelemetryAttributeMetadata {
        description: description.to_owned(),
        sensitive: None,
        cardinality,
    }
}

/// `{ type: "string", description, values?, cardinality? }`.
fn string_attr(
    description: &str,
    values: Option<&[&str]>,
    cardinality: Option<TelemetryAttributeCardinality>,
) -> TelemetryAttributeDefinition {
    TelemetryAttributeDefinition::String {
        metadata: metadata(description, cardinality),
        values: values.map(|values| values.iter().map(|value| (*value).to_owned()).collect()),
        examples: None,
    }
}

/// `{ type: "unsigned_integer", description }`.
fn unsigned_attr(description: &str) -> TelemetryAttributeDefinition {
    TelemetryAttributeDefinition::UnsignedInteger {
        metadata: metadata(description, None),
        values: None,
        examples: None,
    }
}

/// `{ type: "float", description }`.
fn float_attr(description: &str) -> TelemetryAttributeDefinition {
    TelemetryAttributeDefinition::Float {
        metadata: metadata(description, None),
        values: None,
        examples: None,
    }
}

/// `{ type: "boolean", description }`.
fn boolean_attr(description: &str) -> TelemetryAttributeDefinition {
    TelemetryAttributeDefinition::Boolean {
        metadata: metadata(description, None),
        values: None,
        examples: None,
    }
}

/// A free-form string attribute.
fn string(description: &str) -> TelemetryAttributeDefinition {
    string_attr(description, None, None)
}

/// A string attribute restricted to a closed value set.
fn string_values(description: &str, values: &[&str]) -> TelemetryAttributeDefinition {
    string_attr(description, Some(values), None)
}

/// A high-cardinality string attribute (ids and names).
fn string_high(description: &str) -> TelemetryAttributeDefinition {
    string_attr(description, None, Some(TelemetryAttributeCardinality::High))
}

/// A low-cardinality string attribute (error classes and event types).
fn string_low(description: &str) -> TelemetryAttributeDefinition {
    string_attr(description, None, Some(TelemetryAttributeCardinality::Low))
}

/// A low-cardinality string attribute restricted to a closed value set.
fn string_low_values(description: &str, values: &[&str]) -> TelemetryAttributeDefinition {
    string_attr(
        description,
        Some(values),
        Some(TelemetryAttributeCardinality::Low),
    )
}

/// A required start attribute.
fn required(definition: TelemetryAttributeDefinition) -> TelemetryStartAttributeDefinition {
    TelemetryStartAttributeDefinition {
        definition,
        required: true,
    }
}

/// An optional start attribute.
fn optional(definition: TelemetryAttributeDefinition) -> TelemetryStartAttributeDefinition {
    TelemetryStartAttributeDefinition {
        definition,
        required: false,
    }
}

/// A span definition with the shared `default: "ok"` status.
fn span(
    description: &str,
    parents: TelemetryParentDefinition,
    start_attributes: BTreeMap<String, TelemetryStartAttributeDefinition>,
    end_attributes: BTreeMap<String, TelemetryAttributeDefinition>,
    error_when: &str,
) -> TelemetrySpanDefinition {
    TelemetrySpanDefinition {
        description: description.to_owned(),
        parents,
        start_attributes,
        end_attributes,
        events: None,
        status: TelemetrySpanStatusDefinition {
            default: TelemetrySpanStatusDefault::Ok,
            error_when: error_when.to_owned(),
        },
    }
}

/// Start attributes shared by every admitted in-process operation span.
fn operation_start_attributes() -> BTreeMap<String, TelemetryStartAttributeDefinition> {
    [
        (
            "pi.session.id".to_owned(),
            required(string_high("Session id")),
        ),
        (
            "pi.lane.name".to_owned(),
            required(string_high("Lane name")),
        ),
        (
            "pi.operation.id".to_owned(),
            required(string_high("Durable operation id")),
        ),
        (
            "pi.operation.recovery".to_owned(),
            required(boolean_attr("Whether this invocation resumes durable work")),
        ),
    ]
    .into_iter()
    .collect()
}

/// End attributes shared by every admitted in-process operation span.
fn operation_error_attributes() -> BTreeMap<String, TelemetryAttributeDefinition> {
    [
        (
            "pi.error.code".to_owned(),
            string_low("Stable operation error code"),
        ),
        (
            "pi.error.type".to_owned(),
            string_low("Low-cardinality operation error class"),
        ),
    ]
    .into_iter()
    .collect()
}

/// One of the three admitted in-process operation spans (run, compaction,
/// navigation), which differ only in their kind, outcome, and status text.
fn operation_span(
    description: &str,
    kind_description: &str,
    kind: &str,
    outcome_values: &[&str],
    outcome_description: &str,
    error_when: &str,
) -> TelemetrySpanDefinition {
    let mut start_attributes = operation_start_attributes();
    start_attributes.insert(
        "pi.operation.kind".to_owned(),
        required(string_values(kind_description, &[kind])),
    );
    let mut end_attributes = BTreeMap::new();
    end_attributes.insert(
        "pi.operation.outcome".to_owned(),
        string_values(outcome_description, outcome_values),
    );
    end_attributes.extend(operation_error_attributes());
    span(
        description,
        TelemetryParentDefinition::RootOrExternal,
        start_attributes,
        end_attributes,
        error_when,
    )
}

// ---------------------------------------------------------------------------
// Schemas
// ---------------------------------------------------------------------------

/// AI-request telemetry: one logical request to an AI provider.
pub static AI_TELEMETRY_SCHEMA: LazyLock<TelemetrySchemaDefinition> =
    LazyLock::new(|| TelemetrySchemaDefinition {
        version: 1,
        spans: [(
            "pi.ai.request".to_owned(),
            span(
                "One logical request to an AI provider",
                TelemetryParentDefinition::Any,
                [
                    (
                        "pi.ai.operation".to_owned(),
                        required(string_values(
                            "Logical provider operation",
                            &[
                                "stream",
                                "fetch_deferred",
                                "cancel_deferred",
                                "generate_images",
                            ],
                        )),
                    ),
                    (
                        "pi.ai.provider".to_owned(),
                        required(string("Selected provider id")),
                    ),
                    (
                        "pi.ai.model".to_owned(),
                        required(string("Requested model id")),
                    ),
                    ("pi.ai.api".to_owned(), required(string("Provider API id"))),
                    (
                        "pi.ai.streaming".to_owned(),
                        required(boolean_attr("Whether this operation returns a stream")),
                    ),
                    (
                        "pi.ai.deferred".to_owned(),
                        optional(boolean_attr(
                            "Whether the operation requests or participates in deferred execution",
                        )),
                    ),
                ]
                .into_iter()
                .collect(),
                [
                    (
                        "pi.ai.response.model".to_owned(),
                        string("Concrete response model"),
                    ),
                    (
                        "pi.ai.response.id".to_owned(),
                        string_high("Provider response id"),
                    ),
                    (
                        "pi.ai.response.stop_reason".to_owned(),
                        string_values(
                            "Normalized terminal response reason",
                            &["stop", "length", "tool_use", "error", "aborted", "deferred"],
                        ),
                    ),
                    (
                        "pi.ai.http.status_code".to_owned(),
                        unsigned_attr("Final HTTP status"),
                    ),
                    (
                        "pi.ai.usage.input_tokens".to_owned(),
                        unsigned_attr("Reported input tokens"),
                    ),
                    (
                        "pi.ai.usage.output_tokens".to_owned(),
                        unsigned_attr("Reported output tokens"),
                    ),
                    (
                        "pi.ai.usage.cache_read_tokens".to_owned(),
                        unsigned_attr("Reported cache-read tokens"),
                    ),
                    (
                        "pi.ai.usage.cache_write_tokens".to_owned(),
                        unsigned_attr("Reported cache-write tokens"),
                    ),
                    (
                        "pi.ai.usage.reasoning_tokens".to_owned(),
                        unsigned_attr("Reported reasoning tokens"),
                    ),
                    (
                        "pi.ai.usage.total_tokens".to_owned(),
                        unsigned_attr("Reported total tokens"),
                    ),
                    (
                        "pi.ai.usage.cost".to_owned(),
                        float_attr("Reported total cost"),
                    ),
                    (
                        "pi.ai.stream.chunk_count".to_owned(),
                        unsigned_attr("Streamed update chunk count"),
                    ),
                    (
                        "pi.ai.stream.time_to_first_chunk_ms".to_owned(),
                        unsigned_attr("Elapsed milliseconds to first update chunk"),
                    ),
                    (
                        "pi.ai.error.type".to_owned(),
                        string_low("Provider or transport error class"),
                    ),
                ]
                .into_iter()
                .collect(),
                "The operation throws or returns an error result",
            ),
        )]
        .into_iter()
        .collect(),
    });

/// Harness telemetry: admitted operations, their checkpoints, turns, steps,
/// tools, hooks, retries, event listeners, and session writes.
pub static HARNESS_TELEMETRY_SCHEMA: LazyLock<TelemetrySchemaDefinition> =
    LazyLock::new(|| TelemetrySchemaDefinition {
        version: 1,
        spans: [
            (
                "pi.harness.run".to_owned(),
                operation_span(
                    "One admitted in-process run invocation",
                    "Run operation kind",
                    "run",
                    &["completed", "aborted", "failed", "suspended"],
                    "Run invocation outcome",
                    "The run fails or throws",
                ),
            ),
            (
                "pi.harness.compaction".to_owned(),
                operation_span(
                    "One admitted in-process manual compaction invocation",
                    "Compaction operation kind",
                    "compaction",
                    &["completed", "declined", "aborted", "failed"],
                    "Compaction invocation outcome",
                    "The compaction fails or throws",
                ),
            ),
            (
                "pi.harness.navigation".to_owned(),
                operation_span(
                    "One admitted in-process navigation invocation",
                    "Navigation operation kind",
                    "navigation",
                    &["completed", "declined", "aborted", "failed"],
                    "Navigation invocation outcome",
                    "The navigation fails or throws",
                ),
            ),
            (
                "pi.harness.checkpoint".to_owned(),
                span(
                    "One run checkpoint",
                    TelemetryParentDefinition::Spans {
                        spans: vec!["pi.harness.run".to_owned()],
                    },
                    [
                        (
                            "pi.lane.name".to_owned(),
                            required(string_high("Lane name")),
                        ),
                        (
                            "pi.operation.id".to_owned(),
                            required(string_high("Durable operation id")),
                        ),
                        (
                            "pi.checkpoint.kind".to_owned(),
                            required(string_values(
                                "Checkpoint purpose",
                                &["normal", "failure_drain", "abort_reconcile"],
                            )),
                        ),
                    ]
                    .into_iter()
                    .collect(),
                    BTreeMap::new(),
                    "Checkpoint work throws",
                ),
            ),
            (
                "pi.harness.turn".to_owned(),
                span(
                    "One assistant response and its tool batch",
                    TelemetryParentDefinition::Spans {
                        spans: vec!["pi.harness.run".to_owned()],
                    },
                    [
                        (
                            "pi.lane.name".to_owned(),
                            required(string_high("Lane name")),
                        ),
                        (
                            "pi.operation.id".to_owned(),
                            required(string_high("Durable operation id")),
                        ),
                        (
                            "pi.turn.id".to_owned(),
                            required(string_high("Invocation-local turn id")),
                        ),
                    ]
                    .into_iter()
                    .collect(),
                    BTreeMap::new(),
                    "Turn work throws",
                ),
            ),
            (
                "pi.harness.step".to_owned(),
                span(
                    "One durable retry attempt",
                    TelemetryParentDefinition::Spans {
                        spans: vec![
                            "pi.harness.turn".to_owned(),
                            "pi.harness.checkpoint".to_owned(),
                            "pi.harness.compaction".to_owned(),
                            "pi.harness.navigation".to_owned(),
                        ],
                    },
                    [
                        (
                            "pi.lane.name".to_owned(),
                            required(string_high("Lane name")),
                        ),
                        (
                            "pi.operation.id".to_owned(),
                            required(string_high("Durable operation id")),
                        ),
                        (
                            "pi.step.kind".to_owned(),
                            required(string_values(
                                "Retryable step kind",
                                &["assistant", "compaction", "branch_summary"],
                            )),
                        ),
                        (
                            "pi.step.attempt".to_owned(),
                            required(unsigned_attr("One-based durable attempt number")),
                        ),
                        (
                            "pi.compaction.reason".to_owned(),
                            optional(string_values(
                                "Compaction trigger",
                                &["manual", "threshold", "overflow"],
                            )),
                        ),
                    ]
                    .into_iter()
                    .collect(),
                    [(
                        "pi.step.outcome".to_owned(),
                        string_values(
                            "Attempt outcome",
                            &[
                                "succeeded",
                                "retry",
                                "failed",
                                "aborted",
                                "deferred",
                                "overflow",
                            ],
                        ),
                    )]
                    .into_iter()
                    .collect(),
                    "The attempt retries, fails, or throws",
                ),
            ),
            (
                "pi.harness.tool".to_owned(),
                span(
                    "One raw phase-2 tool execution",
                    TelemetryParentDefinition::Spans {
                        spans: vec!["pi.harness.turn".to_owned(), "pi.harness.run".to_owned()],
                    },
                    [
                        (
                            "pi.lane.name".to_owned(),
                            required(string_high("Lane name")),
                        ),
                        (
                            "pi.operation.id".to_owned(),
                            required(string_high("Durable operation id")),
                        ),
                        (
                            "pi.turn.id".to_owned(),
                            optional(string_high("Invocation-local live turn id")),
                        ),
                        ("pi.tool.name".to_owned(), required(string("Tool name"))),
                        (
                            "pi.tool.call_id".to_owned(),
                            required(string_high("Tool call id")),
                        ),
                        (
                            "pi.tool.replay".to_owned(),
                            required(string_values("Declared replay policy", &["never", "safe"])),
                        ),
                        (
                            "pi.tool.recovery".to_owned(),
                            required(boolean_attr("Whether this is recovery execution")),
                        ),
                    ]
                    .into_iter()
                    .collect(),
                    [(
                        "pi.tool.is_error".to_owned(),
                        boolean_attr("Whether raw phase-2 execution returned an error"),
                    )]
                    .into_iter()
                    .collect(),
                    "Raw phase-2 execution returns an error",
                ),
            ),
            (
                "pi.harness.hook".to_owned(),
                span(
                    "One registered hook handler invocation",
                    TelemetryParentDefinition::Any,
                    [
                        (
                            "pi.lane.name".to_owned(),
                            required(string_high("Lane name")),
                        ),
                        (
                            "pi.operation.id".to_owned(),
                            optional(string_high("Durable operation id when accepted")),
                        ),
                        (
                            "pi.hook.name".to_owned(),
                            required(string_values("Hook name", &HOOK_NAMES)),
                        ),
                        (
                            "pi.hook.registration_id".to_owned(),
                            optional(string("Stable hook registration id")),
                        ),
                    ]
                    .into_iter()
                    .collect(),
                    [(
                        "pi.hook.outcome".to_owned(),
                        string_values(
                            "Handler outcome",
                            &["completed", "skipped", "blocked", "failed"],
                        ),
                    )]
                    .into_iter()
                    .collect(),
                    "The handler throws",
                ),
            ),
            (
                "pi.harness.sleep".to_owned(),
                span(
                    "One retry delay",
                    TelemetryParentDefinition::Spans {
                        spans: vec!["pi.harness.step".to_owned(), "pi.harness.run".to_owned()],
                    },
                    [
                        (
                            "pi.operation.id".to_owned(),
                            required(string_high("Durable operation id")),
                        ),
                        (
                            "pi.sleep.delay_ms".to_owned(),
                            required(unsigned_attr("Requested delay in milliseconds")),
                        ),
                    ]
                    .into_iter()
                    .collect(),
                    [(
                        "pi.sleep.outcome".to_owned(),
                        string_values("Delay outcome", &["elapsed", "aborted"]),
                    )]
                    .into_iter()
                    .collect(),
                    "Sleep work throws",
                ),
            ),
            (
                "pi.harness.event_handler".to_owned(),
                span(
                    "One passive event listener invocation",
                    TelemetryParentDefinition::Any,
                    [
                        (
                            "pi.event.type".to_owned(),
                            required(string_low_values(
                                "Delivered harness event type",
                                &EVENT_TYPES,
                            )),
                        ),
                        (
                            "pi.lane.name".to_owned(),
                            optional(string_high("Lane name for lane-scoped events")),
                        ),
                    ]
                    .into_iter()
                    .collect(),
                    BTreeMap::new(),
                    "The listener throws",
                ),
            ),
            (
                "pi.session.write".to_owned(),
                span(
                    "One committed session mutation",
                    TelemetryParentDefinition::Any,
                    [
                        (
                            "pi.lane.name".to_owned(),
                            required(string_high("Lane name")),
                        ),
                        (
                            "pi.operation.id".to_owned(),
                            optional(string_high("Durable operation id when accepted")),
                        ),
                        (
                            "pi.session.mutation".to_owned(),
                            required(string_values(
                                "Session mutation kind",
                                &["entry", "record", "lane", "fact"],
                            )),
                        ),
                        (
                            "pi.session.item_type".to_owned(),
                            optional(string("Entry, record, lane, or fact subtype")),
                        ),
                    ]
                    .into_iter()
                    .collect(),
                    [(
                        "pi.session.seq".to_owned(),
                        unsigned_attr("Committed session sequence when exposed"),
                    )]
                    .into_iter()
                    .collect(),
                    "Storage rejects the mutation",
                ),
            ),
        ]
        .into_iter()
        .collect(),
    });

/// Combined typed span vocabulary for agent-owned AI-request and harness
/// telemetry.
pub static AGENT_TELEMETRY_SCHEMAS: LazyLock<Vec<&'static TelemetrySchemaDefinition>> =
    LazyLock::new(|| vec![&AI_TELEMETRY_SCHEMA, &HARNESS_TELEMETRY_SCHEMA]);

/// Span names declared by [`AI_TELEMETRY_SCHEMA`].
pub const AI_SPAN_NAMES: [&str; 1] = ["pi.ai.request"];

/// Span names declared by [`HARNESS_TELEMETRY_SCHEMA`], in TypeScript
/// declaration order (the schema itself stores them sorted).
pub const HARNESS_SPAN_NAMES: [&str; 11] = [
    "pi.harness.run",
    "pi.harness.compaction",
    "pi.harness.navigation",
    "pi.harness.checkpoint",
    "pi.harness.turn",
    "pi.harness.step",
    "pi.harness.tool",
    "pi.harness.hook",
    "pi.harness.sleep",
    "pi.harness.event_handler",
    "pi.session.write",
];

// ---------------------------------------------------------------------------
// Span helpers
// ---------------------------------------------------------------------------

/// Record one AI-request span.
///
/// The TypeScript source constrains `name` to [`AI_SPAN_NAMES`] and the
/// attribute set to the schema's exact start attributes; here the name is a
/// string checked against the schema in debug builds and the attributes are an
/// open [`SpanAttributes`] map. Call sites that want compile-time guarantees
/// declare a [`SpanDefinition`](step_harness_telemetry::typed::SpanDefinition)
/// and use
/// [`TypedTelemetryContext`](step_harness_telemetry::typed::TypedTelemetryContext).
pub async fn start_ai_span<C, T, F, Fut>(
    telemetry_context: &C,
    name: &str,
    attributes: SpanAttributes,
    callback: F,
) -> T
where
    C: TelemetryContext,
    T: Send,
    F: FnOnce(C::Span) -> Fut + Send,
    Fut: Future<Output = T> + Send,
{
    debug_assert!(
        AI_TELEMETRY_SCHEMA.spans.contains_key(name),
        "unknown AI telemetry span: {name}"
    );
    telemetry_context
        .start_span(SpanOptions::new(name).with_attributes(attributes), callback)
        .await
}

/// Record one harness span. See [`start_ai_span`] for the contract differences
/// from the TypeScript helper of the same name.
pub async fn start_harness_span<C, T, F, Fut>(
    telemetry_context: &C,
    name: &str,
    attributes: SpanAttributes,
    callback: F,
) -> T
where
    C: TelemetryContext,
    T: Send,
    F: FnOnce(C::Span) -> Fut + Send,
    Fut: Future<Output = T> + Send,
{
    debug_assert!(
        HARNESS_TELEMETRY_SCHEMA.spans.contains_key(name),
        "unknown harness telemetry span: {name}"
    );
    telemetry_context
        .start_span(SpanOptions::new(name).with_attributes(attributes), callback)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use step_harness_telemetry::memory::InMemoryTelemetryContext;

    #[test]
    fn schemas_round_trip_through_json() {
        for schema in AGENT_TELEMETRY_SCHEMAS.iter() {
            let encoded = serde_json::to_string(schema).expect("serializable schema");
            let decoded: TelemetrySchemaDefinition =
                serde_json::from_str(&encoded).expect("deserializable schema");
            assert_eq!(&decoded, *schema);
        }
    }

    #[test]
    fn schemas_declare_the_expected_spans() {
        assert_eq!(AI_TELEMETRY_SCHEMA.version, 1);
        assert_eq!(AI_SPAN_NAMES.to_vec(), vec!["pi.ai.request"]);
        assert_eq!(AI_TELEMETRY_SCHEMA.spans.len(), 1);
        let request = &AI_TELEMETRY_SCHEMA.spans["pi.ai.request"];
        assert_eq!(request.start_attributes.len(), 6);
        assert_eq!(request.end_attributes.len(), 14);

        assert_eq!(HARNESS_TELEMETRY_SCHEMA.version, 1);
        assert_eq!(HARNESS_TELEMETRY_SCHEMA.spans.len(), 11);
        for name in HARNESS_SPAN_NAMES {
            assert!(
                HARNESS_TELEMETRY_SCHEMA.spans.contains_key(name),
                "missing harness span {name}"
            );
        }
        let run = &HARNESS_TELEMETRY_SCHEMA.spans["pi.harness.run"];
        assert_eq!(run.start_attributes.len(), 5);
        assert_eq!(run.end_attributes.len(), 3);
        assert_eq!(run.parents, TelemetryParentDefinition::RootOrExternal);
        let step = &HARNESS_TELEMETRY_SCHEMA.spans["pi.harness.step"];
        assert_eq!(
            step.parents,
            TelemetryParentDefinition::Spans {
                spans: vec![
                    "pi.harness.turn".to_owned(),
                    "pi.harness.checkpoint".to_owned(),
                    "pi.harness.compaction".to_owned(),
                    "pi.harness.navigation".to_owned(),
                ],
            }
        );
        assert_eq!(step.start_attributes.len(), 5);
    }

    #[test]
    fn number_attributes_are_split_by_meaning() {
        let request = &AI_TELEMETRY_SCHEMA.spans["pi.ai.request"];
        let unsigned = |name: &str| {
            matches!(
                request.end_attributes[name],
                TelemetryAttributeDefinition::UnsignedInteger { .. }
            )
        };
        assert!(unsigned("pi.ai.http.status_code"));
        assert!(unsigned("pi.ai.usage.total_tokens"));
        assert!(unsigned("pi.ai.stream.time_to_first_chunk_ms"));
        assert!(matches!(
            request.end_attributes["pi.ai.usage.cost"],
            TelemetryAttributeDefinition::Float { .. }
        ));
        let step = &HARNESS_TELEMETRY_SCHEMA.spans["pi.harness.step"];
        assert!(matches!(
            step.start_attributes["pi.step.attempt"].definition,
            TelemetryAttributeDefinition::UnsignedInteger { .. }
        ));
    }

    #[tokio::test]
    async fn span_helpers_record_declared_spans() {
        let context = InMemoryTelemetryContext::new();
        let attributes: SpanAttributes = [
            ("pi.ai.operation".to_owned(), "stream".into()),
            ("pi.ai.streaming".to_owned(), true.into()),
        ]
        .into_iter()
        .collect();
        let value = start_ai_span(&context, "pi.ai.request", attributes, |_span| async {
            42u8
        })
        .await;
        assert_eq!(value, 42);

        let outcome = start_harness_span(
            &context,
            "pi.harness.sleep",
            [("pi.sleep.delay_ms".to_owned(), 250u64.into())]
                .into_iter()
                .collect(),
            |_span| async { "elapsed" },
        )
        .await;
        assert_eq!(outcome, "elapsed");

        let spans = context.get_spans();
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].name, "pi.ai.request");
        assert_eq!(
            spans[0].attributes.get("pi.ai.operation"),
            Some(&"stream".to_owned().into())
        );
        assert_eq!(spans[1].name, "pi.harness.sleep");
        assert!(spans.iter().all(|span| span.settled));
    }
}
