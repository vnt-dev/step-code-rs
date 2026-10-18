//! Proxy stream function for apps that route LLM calls through a server.
//!
//! Ported from `@step-harness/agent-core` `src\proxy.ts`. The server manages
//! auth, forwards requests to providers, and strips the `partial` field from
//! delta events to reduce bandwidth; the partial message is reconstructed
//! client-side from the compact proxy events.

use std::collections::HashMap;

use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;
use step_harness_providers::event::{
    AssistantMessageEvent, AssistantMessageEventSender, AssistantMessageEventStream, DoneReason,
    ErrorReason, assistant_message_event_stream,
};
use step_harness_providers::types::{
    AssistantContent, AssistantMessage, CacheRetention, Context, Model, SimpleStreamOptions,
    StopReason, TextContent, ThinkingBudgets, ThinkingContent, ThinkingLevel, ToolCall, Transport,
    Usage,
};
use step_harness_providers::utils::json_parse::parse_streaming_json;
use step_harness_providers::utils::now_ms;
use tokio_util::sync::CancellationToken;

/// Proxy event types. The server sends these with the partial field stripped.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProxyAssistantMessageEvent {
    Start,
    TextStart {
        content_index: usize,
    },
    TextDelta {
        content_index: usize,
        delta: String,
    },
    TextEnd {
        content_index: usize,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content_signature: Option<String>,
    },
    ThinkingStart {
        content_index: usize,
    },
    ThinkingDelta {
        content_index: usize,
        delta: String,
    },
    ThinkingEnd {
        content_index: usize,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content_signature: Option<String>,
    },
    ToolCallStart {
        content_index: usize,
        id: String,
        tool_name: String,
    },
    ToolCallDelta {
        content_index: usize,
        delta: String,
    },
    ToolCallEnd {
        content_index: usize,
        tool_call: ToolCall,
    },
    Done {
        reason: ProxyDoneReason,
        usage: Usage,
    },
    Error {
        reason: ProxyErrorReason,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error_message: Option<String>,
        usage: Usage,
    },
}

/// Terminal success reasons carried by proxy `done` events.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ProxyDoneReason {
    Stop,
    Length,
    ToolUse,
}

/// Terminal failure reasons carried by proxy `error` events.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ProxyErrorReason {
    Aborted,
    Error,
}

/// Options for [`stream_proxy`].
#[derive(Clone, Debug, Default)]
pub struct ProxyStreamOptions {
    /// Local abort signal for the proxy request.
    pub signal: Option<CancellationToken>,
    /// Auth token for the proxy server.
    pub auth_token: String,
    /// Proxy server URL (e.g. `https://genai.example.com`).
    pub proxy_url: String,

    pub temperature: Option<f64>,
    pub sampling_params: Option<serde_json::Map<String, serde_json::Value>>,
    pub max_tokens: Option<u64>,
    pub reasoning: Option<ThinkingLevel>,
    pub cache_retention: Option<CacheRetention>,
    pub session_id: Option<String>,
    pub headers: Option<HashMap<String, Option<String>>>,
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
    pub transport: Option<Transport>,
    pub thinking_budgets: Option<ThinkingBudgets>,
    pub max_retry_delay_ms: Option<u64>,
}

impl ProxyStreamOptions {
    /// Build proxy options from provider stream options plus the proxy coordinates.
    pub fn from_stream_options(
        options: &SimpleStreamOptions,
        auth_token: impl Into<String>,
        proxy_url: impl Into<String>,
    ) -> Self {
        ProxyStreamOptions {
            signal: options.base.base.signal.clone(),
            auth_token: auth_token.into(),
            proxy_url: proxy_url.into(),
            temperature: options.base.temperature,
            sampling_params: options.base.sampling_params.clone(),
            max_tokens: options.base.max_tokens,
            reasoning: options.reasoning,
            cache_retention: options.base.cache_retention,
            session_id: options.base.session_id.clone(),
            headers: options.base.base.headers.clone(),
            metadata: options.base.metadata.clone(),
            transport: options.base.transport,
            thinking_budgets: options.thinking_budgets,
            max_retry_delay_ms: options.base.base.max_retry_delay_ms,
        }
    }
}

/// Stream function that proxies through a server instead of calling LLM
/// providers directly.
///
/// Use this as the [`crate::types::StreamFn`] when creating an agent that needs
/// to go through a proxy.
pub fn stream_proxy(
    model: &Model,
    context: &Context,
    options: &ProxyStreamOptions,
) -> AssistantMessageEventStream {
    let (mut sender, stream) = assistant_message_event_stream();
    let model = model.clone();
    let context = context.clone();
    let options = options.clone();

    tokio::spawn(async move {
        let cancellation = sender.cancellation_token();
        let mut processor = ProxyEventProcessor::new(&model);

        let outcome = run_proxy(
            &model,
            &context,
            &options,
            &cancellation,
            &mut sender,
            &mut processor,
        )
        .await;

        if let Err(error) = outcome {
            let aborted = options
                .signal
                .as_ref()
                .is_some_and(|signal| signal.is_cancelled());
            let mut partial = processor.partial;
            partial.stop_reason = if aborted {
                StopReason::Aborted
            } else {
                StopReason::Error
            };
            partial.error_message = Some(error);
            sender
                .push(AssistantMessageEvent::Error {
                    reason: if aborted {
                        ErrorReason::Aborted
                    } else {
                        ErrorReason::Error
                    },
                    error: partial,
                })
                .await;
        }
    });

    stream
}

fn initial_partial(model: &Model) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason: StopReason::Pending,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: now_ms(),
    }
}

async fn run_proxy(
    model: &Model,
    context: &Context,
    options: &ProxyStreamOptions,
    cancellation: &CancellationToken,
    sender: &mut AssistantMessageEventSender,
    processor: &mut ProxyEventProcessor,
) -> Result<(), String> {
    let url = format!("{}/api/stream", options.proxy_url.trim_end_matches('/'));
    let body = serde_json::json!({
        "model": model,
        "context": context,
        "options": build_proxy_request_options(options),
    });

    static CLIENT: LazyLock<reqwest::Client> = LazyLock::new(reqwest::Client::new);
    let client = &*CLIENT;
    let response = tokio::select! {
        biased;
        _ = proxy_cancelled(cancellation, options.signal.as_ref()) => return Err("Request aborted by user".to_owned()),
        response = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", options.auth_token))
            .header("Content-Type", "application/json")
            .json(&body)
            .send() => response.map_err(|error| error.to_string())?,
    };

    let status = response.status();
    if !status.is_success() {
        let reason = status.canonical_reason().unwrap_or("");
        let mut error_message = format!("Proxy error: {} {reason}", status.as_u16());
        if let Ok(error_data) = tokio::select! {
            _ = proxy_cancelled(cancellation, options.signal.as_ref()) => return Err("Request aborted by user".to_owned()),
            data = response.json::<serde_json::Value>() => data,
        } && let Some(error) = error_data.get("error").and_then(|value| value.as_str())
        {
            error_message = format!("Proxy error: {error}");
        }
        return Err(error_message);
    }

    let mut limit = SseFrameLimit::default();
    let byte_stream = response.bytes_stream().map(move |chunk| {
        let chunk = chunk.map_err(|error| error.to_string())?;
        limit.accept(&chunk)?;
        Ok::<_, String>(chunk)
    });
    let events = byte_stream.eventsource();
    futures::pin_mut!(events);
    loop {
        let event = tokio::select! {
            biased;
            _ = proxy_cancelled(cancellation, options.signal.as_ref()) => return Err("Request aborted by user".to_owned()),
            event = events.next() => event,
        };
        let Some(event) = event else {
            return Err("Proxy stream ended without a terminal event".to_owned());
        };
        let event = event.map_err(|error| error.to_string())?;
        if event.data.trim().is_empty() {
            continue;
        }
        let proxy_event: ProxyAssistantMessageEvent =
            serde_json::from_str(&event.data).map_err(|error| error.to_string())?;
        let event = processor.process_checked(proxy_event)?;
        let terminal = event.is_terminal();
        let sent = tokio::select! {
            _ = proxy_cancelled(cancellation, options.signal.as_ref()) => return Err("Request aborted by user".to_owned()),
            sent = sender.push(event) => sent,
        };
        if terminal || !sent {
            return Ok(());
        }
    }
}

async fn proxy_cancelled(internal: &CancellationToken, external: Option<&CancellationToken>) {
    match external {
        Some(external) => {
            tokio::select! { _ = internal.cancelled() => {}, _ = external.cancelled() => {} }
        }
        None => internal.cancelled().await,
    }
}

const MAX_PROXY_FRAME_BYTES: usize = 1024 * 1024;
const MAX_PROXY_CONTENT_BLOCKS: usize = 1024;

/// Bound raw SSE framing memory before passing bytes to the library parser.
#[derive(Default)]
struct SseFrameLimit {
    bytes: usize,
    line_bytes: usize,
    previous_cr: bool,
}
impl SseFrameLimit {
    fn accept(&mut self, bytes: &[u8]) -> Result<(), String> {
        for &byte in bytes {
            self.bytes += 1;
            if self.bytes > MAX_PROXY_FRAME_BYTES {
                return Err("Proxy SSE frame exceeds size limit".to_owned());
            }
            if byte == b'\n' && self.previous_cr {
                self.previous_cr = false;
                continue;
            }
            self.previous_cr = byte == b'\r';
            if byte == b'\n' || byte == b'\r' {
                if self.line_bytes == 0 {
                    self.bytes = 0;
                }
                self.line_bytes = 0;
            } else {
                self.line_bytes += 1;
            }
        }
        Ok(())
    }
}

/// The subset of stream options the proxy protocol serializes.
#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct ProxySerializableStreamOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sampling_params: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ThinkingLevel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_retention: Option<CacheRetention>,
    #[serde(skip_serializing_if = "Option::is_none")]
    session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    headers: Option<HashMap<String, Option<String>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    transport: Option<Transport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_budgets: Option<ThinkingBudgets>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_retry_delay_ms: Option<u64>,
}

fn build_proxy_request_options(options: &ProxyStreamOptions) -> ProxySerializableStreamOptions {
    ProxySerializableStreamOptions {
        temperature: options.temperature,
        sampling_params: options.sampling_params.clone(),
        max_tokens: options.max_tokens,
        reasoning: options.reasoning,
        cache_retention: options.cache_retention,
        session_id: options.session_id.clone(),
        headers: options.headers.clone(),
        metadata: options.metadata.clone(),
        transport: options.transport,
        thinking_budgets: options.thinking_budgets,
        max_retry_delay_ms: options.max_retry_delay_ms,
    }
}

/// Reconstructs the partial assistant message from compact proxy events.
///
/// Streamed tool-call arguments accumulate as a raw JSON string per content
/// index and are salvaged with the streaming parser on each delta, matching the
/// TypeScript implementation.
struct ProxyEventProcessor {
    partial: AssistantMessage,
    partial_json: HashMap<usize, String>,
    started: bool,
    received_bytes: usize,
    closed_blocks: std::collections::HashSet<usize>,
}

impl ProxyEventProcessor {
    fn new(model: &Model) -> Self {
        ProxyEventProcessor {
            partial: initial_partial(model),
            partial_json: HashMap::new(),
            started: false,
            received_bytes: 0,
            closed_blocks: Default::default(),
        }
    }

    fn process_checked(
        &mut self,
        event: ProxyAssistantMessageEvent,
    ) -> Result<AssistantMessageEvent, String> {
        if matches!(event, ProxyAssistantMessageEvent::Start) {
            if self.started {
                return Err("Duplicate proxy start event".to_owned());
            }
            self.started = true;
        } else if !self.started && !matches!(event, ProxyAssistantMessageEvent::Error { .. }) {
            return Err("Proxy content arrived before start".to_owned());
        }
        let added_bytes = match &event {
            ProxyAssistantMessageEvent::TextDelta { delta, .. }
            | ProxyAssistantMessageEvent::ThinkingDelta { delta, .. }
            | ProxyAssistantMessageEvent::ToolCallDelta { delta, .. } => delta.len(),
            ProxyAssistantMessageEvent::ToolCallEnd { tool_call, .. } => {
                serde_json::to_string(&tool_call.arguments)
                    .map_err(|error| error.to_string())?
                    .len()
            }
            _ => 0,
        };
        self.received_bytes = self.received_bytes.saturating_add(added_bytes);
        if self.received_bytes > 16 * 1024 * 1024 {
            return Err("Proxy message exceeds size limit".to_owned());
        }
        let index = match &event {
            ProxyAssistantMessageEvent::TextStart { content_index }
            | ProxyAssistantMessageEvent::TextDelta { content_index, .. }
            | ProxyAssistantMessageEvent::TextEnd { content_index, .. }
            | ProxyAssistantMessageEvent::ThinkingStart { content_index }
            | ProxyAssistantMessageEvent::ThinkingDelta { content_index, .. }
            | ProxyAssistantMessageEvent::ThinkingEnd { content_index, .. }
            | ProxyAssistantMessageEvent::ToolCallStart { content_index, .. }
            | ProxyAssistantMessageEvent::ToolCallDelta { content_index, .. }
            | ProxyAssistantMessageEvent::ToolCallEnd { content_index, .. } => Some(*content_index),
            _ => None,
        };
        if let Some(index) = index {
            if self.closed_blocks.contains(&index) {
                return Err("Proxy event targets a closed content block".to_owned());
            }
            if matches!(
                event,
                ProxyAssistantMessageEvent::TextEnd { .. }
                    | ProxyAssistantMessageEvent::ThinkingEnd { .. }
                    | ProxyAssistantMessageEvent::ToolCallEnd { .. }
            ) {
                self.closed_blocks.insert(index);
            }
            if index >= MAX_PROXY_CONTENT_BLOCKS {
                return Err("Proxy content index exceeds limit".to_owned());
            }
            if matches!(
                &event,
                ProxyAssistantMessageEvent::TextStart { .. }
                    | ProxyAssistantMessageEvent::ThinkingStart { .. }
                    | ProxyAssistantMessageEvent::ToolCallStart { .. }
            ) && index != self.partial.content.len()
            {
                return Err("Proxy content blocks must start in order".to_owned());
            }
        }
        self.process(event)
            .ok_or_else(|| "Invalid proxy content event sequence".to_owned())
    }

    /// Process a proxy event and update the partial message.
    fn process(
        &mut self,
        proxy_event: ProxyAssistantMessageEvent,
    ) -> Option<AssistantMessageEvent> {
        match proxy_event {
            ProxyAssistantMessageEvent::Start => {
                // The start event carries its own empty content array rather
                // than aliasing the live one, matching the provider stream
                // contract.
                Some(AssistantMessageEvent::Start {
                    partial: AssistantMessage {
                        content: Vec::new(),
                        ..self.partial.clone()
                    },
                })
            }
            ProxyAssistantMessageEvent::TextStart { content_index } => {
                set_content(
                    &mut self.partial,
                    content_index,
                    AssistantContent::Text(default_text()),
                );
                Some(AssistantMessageEvent::TextStart {
                    content_index,
                    partial: self.partial.clone(),
                })
            }
            ProxyAssistantMessageEvent::TextDelta {
                content_index,
                delta,
            } => {
                let AssistantContent::Text(text) =
                    content_at_mut(&mut self.partial, content_index)?
                else {
                    return None;
                };
                text.text.push_str(&delta);
                Some(AssistantMessageEvent::TextDelta {
                    content_index,
                    delta,
                    partial: self.partial.clone(),
                })
            }
            ProxyAssistantMessageEvent::TextEnd {
                content_index,
                content_signature,
            } => {
                let AssistantContent::Text(text) =
                    content_at_mut(&mut self.partial, content_index)?
                else {
                    return None;
                };
                text.text_signature = content_signature;
                let content_text = text.text.clone();
                Some(AssistantMessageEvent::TextEnd {
                    content_index,
                    content: content_text,
                    partial: self.partial.clone(),
                })
            }
            ProxyAssistantMessageEvent::ThinkingStart { content_index } => {
                set_content(
                    &mut self.partial,
                    content_index,
                    AssistantContent::Thinking(ThinkingContent {
                        thinking: String::new(),
                        thinking_signature: None,
                        redacted: false,
                    }),
                );
                Some(AssistantMessageEvent::ThinkingStart {
                    content_index,
                    partial: self.partial.clone(),
                })
            }
            ProxyAssistantMessageEvent::ThinkingDelta {
                content_index,
                delta,
            } => {
                let AssistantContent::Thinking(thinking) =
                    content_at_mut(&mut self.partial, content_index)?
                else {
                    return None;
                };
                thinking.thinking.push_str(&delta);
                Some(AssistantMessageEvent::ThinkingDelta {
                    content_index,
                    delta,
                    partial: self.partial.clone(),
                })
            }
            ProxyAssistantMessageEvent::ThinkingEnd {
                content_index,
                content_signature,
            } => {
                let AssistantContent::Thinking(thinking) =
                    content_at_mut(&mut self.partial, content_index)?
                else {
                    return None;
                };
                thinking.thinking_signature = content_signature;
                let content_text = thinking.thinking.clone();
                Some(AssistantMessageEvent::ThinkingEnd {
                    content_index,
                    content: content_text,
                    partial: self.partial.clone(),
                })
            }
            ProxyAssistantMessageEvent::ToolCallStart {
                content_index,
                id,
                tool_name,
            } => {
                set_content(
                    &mut self.partial,
                    content_index,
                    AssistantContent::ToolCall(ToolCall {
                        id,
                        name: tool_name,
                        arguments: serde_json::Map::new(),
                        thought_signature: None,
                        namespace: None,
                    }),
                );
                self.partial_json.insert(content_index, String::new());
                Some(AssistantMessageEvent::ToolCallStart {
                    content_index,
                    partial: self.partial.clone(),
                })
            }
            ProxyAssistantMessageEvent::ToolCallDelta {
                content_index,
                delta,
            } => {
                let AssistantContent::ToolCall(tool_call) =
                    content_at_mut(&mut self.partial, content_index)?
                else {
                    return None;
                };
                let partial_json = self.partial_json.entry(content_index).or_default();
                partial_json.push_str(&delta);
                tool_call.arguments = parse_streaming_json(Some(partial_json))
                    .as_object()
                    .cloned()
                    .unwrap_or_default();
                Some(AssistantMessageEvent::ToolCallDelta {
                    content_index,
                    delta,
                    partial: self.partial.clone(),
                })
            }
            ProxyAssistantMessageEvent::ToolCallEnd {
                content_index,
                tool_call,
            } => {
                let AssistantContent::ToolCall(existing) =
                    content_at_mut(&mut self.partial, content_index)?
                else {
                    return None;
                };
                *existing = tool_call;
                self.partial_json.remove(&content_index);
                Some(AssistantMessageEvent::ToolCallEnd {
                    content_index,
                    tool_call: existing.clone(),
                    partial: self.partial.clone(),
                })
            }
            ProxyAssistantMessageEvent::Done { reason, usage } => {
                self.partial.stop_reason = match reason {
                    ProxyDoneReason::Stop => StopReason::Stop,
                    ProxyDoneReason::Length => StopReason::Length,
                    ProxyDoneReason::ToolUse => StopReason::ToolUse,
                };
                self.partial.usage = usage;
                Some(AssistantMessageEvent::Done {
                    reason: match reason {
                        ProxyDoneReason::Stop => DoneReason::Stop,
                        ProxyDoneReason::Length => DoneReason::Length,
                        ProxyDoneReason::ToolUse => DoneReason::ToolUse,
                    },
                    message: self.partial.clone(),
                })
            }
            ProxyAssistantMessageEvent::Error {
                reason,
                error_message,
                usage,
            } => {
                self.partial.stop_reason = match reason {
                    ProxyErrorReason::Aborted => StopReason::Aborted,
                    ProxyErrorReason::Error => StopReason::Error,
                };
                self.partial.error_message = error_message;
                self.partial.usage = usage;
                Some(AssistantMessageEvent::Error {
                    reason: match reason {
                        ProxyErrorReason::Aborted => ErrorReason::Aborted,
                        ProxyErrorReason::Error => ErrorReason::Error,
                    },
                    error: self.partial.clone(),
                })
            }
        }
    }
}

fn default_text() -> TextContent {
    TextContent {
        text: String::new(),
        text_signature: None,
    }
}

fn set_content(partial: &mut AssistantMessage, index: usize, content: AssistantContent) {
    if partial.content.len() <= index {
        partial.content.resize(index + 1, content.clone());
    } else {
        partial.content[index] = content;
    }
}

fn content_at_mut(partial: &mut AssistantMessage, index: usize) -> Option<&mut AssistantContent> {
    partial.content.get_mut(index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use step_harness_providers::types::{Api, InputModality, ProviderId};

    #[test]
    fn rejects_out_of_order_and_oversized_content_indices() {
        let mut processor = ProxyEventProcessor::new(&model());
        assert!(
            processor
                .process_checked(ProxyAssistantMessageEvent::TextStart { content_index: 0 })
                .is_err()
        );
        processor
            .process_checked(ProxyAssistantMessageEvent::Start)
            .unwrap();
        assert!(
            processor
                .process_checked(ProxyAssistantMessageEvent::TextStart {
                    content_index: usize::MAX
                })
                .is_err()
        );
        assert!(processor.partial.content.is_empty());
        assert!(
            processor
                .process_checked(ProxyAssistantMessageEvent::TextStart { content_index: 1 })
                .is_err()
        );
        processor
            .process_checked(ProxyAssistantMessageEvent::TextStart { content_index: 0 })
            .unwrap();
        processor
            .process_checked(ProxyAssistantMessageEvent::TextEnd {
                content_index: 0,
                content_signature: None,
            })
            .unwrap();
        assert!(
            processor
                .process_checked(ProxyAssistantMessageEvent::TextDelta {
                    content_index: 0,
                    delta: "late".to_owned()
                })
                .is_err()
        );
    }

    #[test]
    fn bounds_sse_frames_across_chunks_and_accepts_crlf_boundaries() {
        let mut limit = SseFrameLimit::default();
        for _ in 0..10 {
            limit.accept(b"data:x\r\n\r\n").unwrap();
        }
        limit
            .accept(&vec![b'x'; MAX_PROXY_FRAME_BYTES - 1])
            .unwrap();
        assert!(limit.accept(b"xx").is_err());
    }

    #[tokio::test]
    async fn parses_multiline_sse_split_across_chunks() {
        let chunks = [
            Ok::<_, String>(b"data:{\"type\":\"sta".to_vec()),
            Ok(b"rt\",\r\ndata: \"ignored\": 1}\r\n\r\n".to_vec()),
        ];
        let mut events = futures::stream::iter(chunks).eventsource();
        let event = events.next().await.unwrap().unwrap();
        assert!(matches!(
            serde_json::from_str::<ProxyAssistantMessageEvent>(&event.data).unwrap(),
            ProxyAssistantMessageEvent::Start
        ));
    }

    #[tokio::test]
    async fn external_cancellation_does_not_wait_for_network_data() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let (ready, received) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut bytes = [0; 8192];
            assert!(socket.read(&mut bytes).await.unwrap() > 0);
            socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n").await.unwrap();
            ready.send(()).unwrap();
            std::future::pending::<()>().await;
        });
        let signal = CancellationToken::new();
        let options = ProxyStreamOptions {
            signal: Some(signal.clone()),
            proxy_url: url,
            ..Default::default()
        };
        let mut stream = stream_proxy(
            &model(),
            &Context {
                system_prompt: None,
                messages: vec![],
                tools: None,
            },
            &options,
        );
        received.await.unwrap();
        signal.cancel();
        let result = tokio::time::timeout(std::time::Duration::from_secs(1), stream.result())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(result.stop_reason, StopReason::Aborted);
        server.abort();
    }

    fn model() -> Model {
        Model {
            id: "m".to_owned(),
            name: "m".to_owned(),
            api: Api::OpenaiCompletions,
            provider: ProviderId::new("p"),
            base_url: "http://localhost".to_owned(),
            reasoning: false,
            thinking_level_map: None,
            input: vec![InputModality::Text],
            cost: Default::default(),
            context_window: 100,
            max_tokens: 10,
            sampling_params: None,
            headers: None,
            compat: None,
        }
    }

    #[test]
    fn start_event_has_its_own_empty_content() {
        let mut processor = ProxyEventProcessor::new(&model());
        processor
            .partial
            .content
            .push(AssistantContent::Text(default_text()));
        let event = processor
            .process(ProxyAssistantMessageEvent::Start)
            .unwrap();
        let AssistantMessageEvent::Start { partial: started } = event else {
            panic!("expected start");
        };
        assert!(started.content.is_empty());
        // The live partial keeps its content.
        assert_eq!(processor.partial.content.len(), 1);
    }

    #[test]
    fn text_deltas_accumulate() {
        let mut processor = ProxyEventProcessor::new(&model());
        processor.process(ProxyAssistantMessageEvent::TextStart { content_index: 0 });
        processor.process(ProxyAssistantMessageEvent::TextDelta {
            content_index: 0,
            delta: "hel".to_owned(),
        });
        processor.process(ProxyAssistantMessageEvent::TextDelta {
            content_index: 0,
            delta: "lo".to_owned(),
        });
        let AssistantContent::Text(text) = &processor.partial.content[0] else {
            panic!("expected text");
        };
        assert_eq!(text.text, "hello");
    }

    #[test]
    fn toolcall_events_reconstruct_arguments() {
        let mut processor = ProxyEventProcessor::new(&model());
        processor.process(ProxyAssistantMessageEvent::ToolCallStart {
            content_index: 0,
            id: "call_1".to_owned(),
            tool_name: "bash".to_owned(),
        });
        processor.process(ProxyAssistantMessageEvent::ToolCallDelta {
            content_index: 0,
            delta: "{\"command\": \"l".to_owned(),
        });
        processor.process(ProxyAssistantMessageEvent::ToolCallDelta {
            content_index: 0,
            delta: "s\"}".to_owned(),
        });
        let AssistantContent::ToolCall(tool_call) = &processor.partial.content[0] else {
            panic!("expected tool call");
        };
        assert_eq!(tool_call.id, "call_1");
        assert_eq!(tool_call.arguments["command"], "ls");
    }

    #[test]
    fn done_event_finalizes_the_partial() {
        let mut processor = ProxyEventProcessor::new(&model());
        let event = processor
            .process(ProxyAssistantMessageEvent::Done {
                reason: ProxyDoneReason::Stop,
                usage: Usage::default(),
            })
            .unwrap();
        let AssistantMessageEvent::Done { message, .. } = event else {
            panic!("expected done");
        };
        assert_eq!(message.stop_reason, StopReason::Stop);
    }
}
