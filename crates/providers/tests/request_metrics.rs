//! Local HTTP fixtures exercise the real adapters, retries and metrics spool.
use serde_json::json;
use std::sync::{Arc, Mutex};
use step_harness_providers::{
    api,
    request_metrics::{RequestContext, RequestObserver, RequestRecord, RequestTracker},
    types::*,
};
#[derive(Default)]
struct Collector(Mutex<Vec<RequestRecord>>);
impl RequestObserver for Collector {
    fn record(&self, record: RequestRecord) {
        self.0.lock().unwrap().push(record);
    }
}
impl Collector {
    fn finished(&self) -> Vec<RequestRecord> {
        self.0
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.status != "running")
            .cloned()
            .collect()
    }
}
fn model(api: Api, url: String) -> Model {
    Model {
        id: "test".into(),
        name: "test".into(),
        api,
        provider: ProviderId::new("test"),
        base_url: url,
        reasoning: true,
        thinking_level_map: None,
        input: vec![InputModality::Text],
        cost: ModelCost::default(),
        context_window: 128000,
        max_tokens: 4096,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}
fn options(model: &Model, collector: Arc<Collector>) -> SimpleStreamOptions {
    let mut options = SimpleStreamOptions::default();
    options.base.base.api_key = Some("secret-test-key".into());
    options.base.base.max_retries = Some(1);
    options.base.base.request_metrics = Some(RequestTracker::new(
        collector,
        RequestContext {
            session_id: "session".into(),
            execution_id: Some("round".into()),
            purpose: "conversation".into(),
        },
        model,
    ));
    options
}
fn frame(value: serde_json::Value) -> String {
    let event = value
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("message");
    format!("event: {event}\ndata: {value}\n\n")
}
fn server(
    chunks: Vec<String>,
    fail_first: bool,
    delay_ms: u64,
) -> (String, std::thread::JoinHandle<()>) {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = std::thread::spawn(move || {
        for attempt in 0..if fail_first { 2 } else { 1 } {
            let (mut stream, _) = listener.accept().unwrap();
            stream.set_nodelay(true).unwrap();
            let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(5)));
            let mut buffer = [0; 32768];
            let _ = stream.read(&mut buffer);
            if fail_first && attempt == 0 {
                let body = "secret-provider-error";
                let _ = write!(
                    stream,
                    "HTTP/1.1 503 Unavailable\r\nContent-Length: {}\r\nRetry-After-Ms: 5\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                continue;
            }
            let length: usize = chunks.iter().map(String::len).sum();
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {length}\r\nConnection: close\r\n\r\n"
            );
            for chunk in &chunks {
                std::thread::sleep(std::time::Duration::from_millis(delay_ms));
                if stream.write_all(chunk.as_bytes()).is_err() {
                    break;
                }
            }
        }
    });
    (format!("http://{addr}"), handle)
}
fn completion_frames() -> Vec<String> {
    vec![
        frame(json!({"choices":[{"index":0,"delta":{"role":"assistant"}}]})),
        frame(json!({"choices":[{"index":0,"delta":{"reasoning_content":"think"}}]})),
        frame(json!({"choices":[{"index":0,"delta":{"content":"reply"}}]})),
        frame(
            json!({"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30,"prompt_tokens_details":{"cached_tokens":4}}}),
        ),
        "data: [DONE]\n\n".into(),
    ]
}
fn stream(
    model: &Model,
    options: &SimpleStreamOptions,
) -> step_harness_providers::AssistantMessageEventStream {
    match model.api {
        Api::AnthropicMessages => {
            api::anthropic_messages::stream_simple(model, &Context::default(), options)
        }
        Api::OpenaiResponses => {
            api::openai_responses::stream_simple(model, &Context::default(), options)
        }
        _ => api::openai_completions::stream_simple(model, &Context::default(), options),
    }
}
#[tokio::test]
async fn actual_http_retries_keep_failures_and_thinking_first_token_without_consumer_delay() {
    let (url, server) = server(completion_frames(), true, 20);
    let model = model(Api::OpenaiCompletions, url);
    let collector = Arc::new(Collector::default());
    let options = options(&model, collector.clone());
    let mut response = stream(&model, &options);
    tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    let output = response.result().await.unwrap();
    assert_eq!(output.stop_reason, StopReason::Stop);
    server.join().unwrap();
    let records = collector.finished();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].status, "failed");
    assert_eq!(records[0].http_status, Some(503));
    assert_eq!(records[0].output_tokens, None);
    assert_eq!(records[0].first_token_ms, None);
    let success = &records[1];
    assert_eq!(success.status, "success");
    assert_eq!(success.attempt, 2);
    assert_eq!(success.logical_id, records[0].logical_id);
    assert_eq!(success.output_tokens, Some(20));
    assert_eq!(success.input_tokens, Some(10));
    assert_eq!(success.cache_read_tokens, Some(4));
    assert!(success.first_token_ms.unwrap() > 10.0);
    assert!(success.generation_ms.unwrap() < 200.0);
    assert!((success.tps.unwrap() - 20000.0 / success.generation_ms.unwrap()).abs() < 0.001);
    let json = serde_json::to_string(&records).unwrap();
    assert!(!json.contains("secret"));
}
#[tokio::test]
async fn responses_and_anthropic_capture_reported_usage_and_first_token() {
    let cases = vec![
        (
            Api::OpenaiResponses,
            vec![
                frame(json!({"type":"response.created","response":{"id":"r"}})),
                frame(
                    json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"m","role":"assistant","content":[]}}),
                ),
                frame(
                    json!({"type":"response.output_text.delta","output_index":0,"item_id":"m","content_index":0,"delta":"a"}),
                ),
                frame(
                    json!({"type":"response.output_text.delta","output_index":0,"item_id":"m","content_index":0,"delta":"b"}),
                ),
                frame(
                    json!({"type":"response.completed","response":{"id":"r","status":"completed","usage":{"input_tokens":10,"output_tokens":20,"total_tokens":30}}}),
                ),
            ],
        ),
        (
            Api::AnthropicMessages,
            vec![
                frame(
                    json!({"type":"message_start","message":{"id":"m","role":"assistant","model":"test","content":[],"usage":{"input_tokens":10,"output_tokens":0}}}),
                ),
                frame(
                    json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
                ),
                frame(
                    json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"a"}}),
                ),
                frame(
                    json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"b"}}),
                ),
                frame(json!({"type":"content_block_stop","index":0})),
                frame(
                    json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":20}}),
                ),
                frame(json!({"type":"message_stop"})),
            ],
        ),
    ];
    for (api, frames) in cases {
        let (url, server) = server(frames, false, 10);
        let model = model(api, url);
        let collector = Arc::new(Collector::default());
        let options = options(&model, collector.clone());
        let output = stream(&model, &options).result().await.unwrap();
        assert_eq!(
            output.stop_reason,
            StopReason::Stop,
            "{:?}: {:?}",
            model.api,
            output.error_message
        );
        server.join().unwrap();
        let records = collector.finished();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].output_tokens, Some(20));
        assert!(records[0].tps.is_some());
        assert!(records[0].first_token_ms.is_some());
    }
}
#[tokio::test]
async fn single_chunk_unknown_usage_truncation_and_cancellation() {
    let cases = vec![
        (
            vec![
                frame(
                    json!({"choices":[{"index":0,"delta":{"content":"a"},"finish_reason":"stop"}]}),
                ),
                "data: [DONE]\n\n".into(),
            ],
            "success",
        ),
        (
            vec![frame(
                json!({"choices":[{"index":0,"delta":{"content":"a"}}]}),
            )],
            "failed",
        ),
    ];
    for (frames, status) in cases {
        let (url, server) = server(frames, false, 10);
        let model = model(Api::OpenaiCompletions, url);
        let collector = Arc::new(Collector::default());
        let options = options(&model, collector.clone());
        let _ = stream(&model, &options).result().await.unwrap();
        server.join().unwrap();
        let records = collector.finished();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].status, status);
        assert_eq!(records[0].output_tokens, None);
        assert_eq!(records[0].tps, None);
    }
    let (url, server) = server(completion_frames(), false, 30);
    let model = model(Api::OpenaiCompletions, url);
    let collector = Arc::new(Collector::default());
    let mut options = options(&model, collector.clone());
    let token = tokio_util::sync::CancellationToken::new();
    options.base.base.signal = Some(token.clone());
    let mut response = stream(&model, &options);
    tokio::time::sleep(std::time::Duration::from_millis(15)).await;
    token.cancel();
    assert_eq!(
        response.result().await.unwrap().stop_reason,
        StopReason::Aborted
    );
    server.join().unwrap();
    let records = collector.finished();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].status, "cancelled");
    assert_eq!(records[0].first_token_ms, None);
}
#[tokio::test]
async fn pure_tool_call_output_counts_toward_generation() {
    let frames = vec![
        frame(
            json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c","type":"function","function":{"name":"read","arguments":"{"}}]}}]}),
        ),
        frame(
            json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]}}]}),
        ),
        frame(
            json!({"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}),
        ),
        "data: [DONE]\n\n".into(),
    ];
    let (url, server) = server(frames, false, 10);
    let model = model(Api::OpenaiCompletions, url);
    let collector = Arc::new(Collector::default());
    let options = options(&model, collector.clone());
    assert_eq!(
        stream(&model, &options).result().await.unwrap().stop_reason,
        StopReason::ToolUse
    );
    server.join().unwrap();
    let records = collector.finished();
    assert!(records[0].first_token_ms.is_some());
    assert!(records[0].tps.is_some());
}

#[tokio::test]
async fn one_frame_with_text_and_thinking_is_not_two_generation_samples() {
    let frames = vec![
        frame(
            json!({"choices":[{"index":0,"delta":{"reasoning_content":"thought","content":"answer"},"finish_reason":"stop"}],"usage":{"completion_tokens":20}}),
        ),
        "data: [DONE]\n\n".into(),
    ];
    let (url, server) = server(frames, false, 10);
    let model = model(Api::OpenaiCompletions, url);
    let collector = Arc::new(Collector::default());
    let options = options(&model, collector.clone());
    assert_eq!(
        stream(&model, &options).result().await.unwrap().stop_reason,
        StopReason::Stop
    );
    server.join().unwrap();
    let records = collector.finished();
    assert_eq!(records[0].output_tokens, Some(20));
    assert_eq!(records[0].input_tokens, None);
    assert_eq!(records[0].cache_read_tokens, None);
    assert_eq!(records[0].generation_ms, None);
    assert_eq!(records[0].tps, None);
    assert!(records[0].first_token_ms.is_some());
}
#[tokio::test]
async fn instrumented_stream_does_not_measure_bounded_consumer_backpressure() {
    let mut frames: Vec<String> = (0..64)
        .map(|_| frame(json!({"choices":[{"index":0,"delta":{"content":"a"}}]})))
        .collect();
    frames.push(frame(json!({"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":64}})));
    frames.push("data: [DONE]\n\n".into());
    let (url, server) = server(frames, false, 2);
    let model = model(Api::OpenaiCompletions, url);
    let collector = Arc::new(Collector::default());
    let options = options(&model, collector.clone());
    let mut response = stream(&model, &options);
    tokio::time::sleep(std::time::Duration::from_millis(600)).await;
    assert_eq!(
        response.result().await.unwrap().stop_reason,
        StopReason::Stop
    );
    server.join().unwrap();
    let records = collector.finished();
    assert_eq!(records.len(), 1);
    assert!(records[0].generation_ms.unwrap() < 450.0);
}
#[tokio::test]
async fn timed_out_attempt_stays_failed_with_no_invented_usage() {
    let (url, server) = server(completion_frames(), false, 40);
    let model = model(Api::OpenaiCompletions, url);
    let collector = Arc::new(Collector::default());
    let mut options = options(&model, collector.clone());
    options.base.base.timeout_ms = Some(5);
    options.base.base.max_retries = Some(0);
    assert_eq!(
        stream(&model, &options).result().await.unwrap().stop_reason,
        StopReason::Error
    );
    server.join().unwrap();
    let records = collector.finished();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].status, "failed");
    assert_eq!(records[0].error_kind.as_deref(), Some("timeout"));
    assert_eq!(records[0].output_tokens, None);
}

#[tokio::test]
async fn prefilled_anthropic_start_content_has_first_token_but_no_generation_interval() {
    let frames = vec![
        frame(
            json!({"type":"message_start","message":{"id":"m","role":"assistant","model":"test","content":[],"usage":{"input_tokens":10,"output_tokens":0}}}),
        ),
        frame(
            json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":"prefilled"}}),
        ),
        frame(json!({"type":"content_block_stop","index":0})),
        frame(
            json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}),
        ),
        frame(json!({"type":"message_stop"})),
    ];
    let (url, server) = server(frames, false, 10);
    let model = model(Api::AnthropicMessages, url);
    let collector = Arc::new(Collector::default());
    let options = options(&model, collector.clone());
    assert_eq!(
        stream(&model, &options).result().await.unwrap().stop_reason,
        StopReason::Stop
    );
    server.join().unwrap();
    let records = collector.finished();
    assert!(records[0].first_token_ms.is_some());
    assert_eq!(records[0].generation_ms, None);
    assert_eq!(records[0].cache_read_tokens, None);
}

#[tokio::test]
async fn buffered_sse_events_in_one_transport_chunk_do_not_invent_tps() {
    let body = completion_frames().join("");
    let (url, server) = server(vec![body], false, 10);
    let model = model(Api::OpenaiCompletions, url);
    let collector = Arc::new(Collector::default());
    let options = options(&model, collector.clone());
    assert_eq!(
        stream(&model, &options).result().await.unwrap().stop_reason,
        StopReason::Stop
    );
    server.join().unwrap();
    let records = collector.finished();
    assert_eq!(records[0].output_tokens, Some(20));
    assert!(records[0].first_token_ms.is_some());
    assert_eq!(records[0].generation_ms, None);
    assert_eq!(records[0].tps, None);
}
