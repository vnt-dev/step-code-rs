//! Shared HTTP transport for provider adapters: client construction (with
//! env-aware proxies), POST helpers, and SSE framing over response bodies.

use std::sync::OnceLock;

use eventsource_stream::Eventsource;
use futures::{Stream, StreamExt};
use serde_json::Value;

use crate::error::ProviderHttpError;
use crate::types::ProviderRequestOptions;

/// A parsed SSE event (Anthropic/OpenAI style).
#[derive(Clone, Debug)]
pub struct SseEvent {
    /// Event name; empty when the server sent only `data:` lines.
    pub event: String,
    pub data: String,
}

/// The shared default HTTP client. Env proxies (HTTP_PROXY/HTTPS_PROXY/
/// ALL_PROXY/NO_PROXY) are validated and applied using reqwest proxy rules.
pub fn default_http_client() -> Result<reqwest::Client, ProviderHttpError> {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    if let Some(client) = CLIENT.get() {
        return Ok(client.clone());
    }
    let config = crate::utils::proxy::ProxyConfig::from_env(None).map_err(|error| {
        ProviderHttpError::transport(error.to_string())
            .with_retryable(false)
            .with_source(error)
    })?;
    let client = build_client(&config)?;
    let _ = CLIENT.set(client.clone());
    Ok(CLIENT.get().cloned().unwrap_or(client))
}

/// Injected clients win. Scoped configurations are validated and cached so
/// repeated requests share connections and never silently fall back.
pub fn http_client_for(
    options: &ProviderRequestOptions,
) -> Result<reqwest::Client, ProviderHttpError> {
    if let Some(client) = &options.http_client {
        return Ok(client.clone());
    }
    let Some(env) = options.env.as_ref().filter(|env| {
        env.keys().any(|key| {
            matches!(
                key.to_ascii_lowercase().as_str(),
                "http_proxy" | "https_proxy" | "all_proxy" | "no_proxy"
            )
        })
    }) else {
        return default_http_client();
    };
    static CLIENTS: std::sync::LazyLock<
        moka::sync::Cache<crate::utils::proxy::ProxyConfig, reqwest::Client>,
    > = std::sync::LazyLock::new(|| moka::sync::Cache::new(32));
    let config = crate::utils::proxy::ProxyConfig::from_env(Some(env)).map_err(|error| {
        ProviderHttpError::transport(error.to_string())
            .with_retryable(false)
            .with_source(error)
    })?;
    CLIENTS
        .try_get_with(config.clone(), || build_client(&config))
        .map_err(|error| {
            ProviderHttpError::transport(error.to_string())
                .with_retryable(false)
                .with_source(error)
        })
}

fn build_client(
    config: &crate::utils::proxy::ProxyConfig,
) -> Result<reqwest::Client, ProviderHttpError> {
    config
        .apply(reqwest::Client::builder().user_agent(crate::utils::user_agent::get_user_agent()))
        .map_err(|error| {
            ProviderHttpError::transport(error.to_string())
                .with_retryable(false)
                .with_source(error)
        })?
        .build()
        .map_err(|error| {
            ProviderHttpError::transport("Could not construct HTTP client")
                .with_retryable(false)
                .with_source(error)
        })
}

/// Outcome of a POST whose response body streams (SSE).
pub struct SseResponse {
    pub status: u16,
    pub headers: reqwest::header::HeaderMap,
    pub events: std::pin::Pin<Box<dyn Stream<Item = Result<SseEvent, ProviderHttpError>> + Send>>,
}

/// Log only the origin: credentials can appear in any other URL component.
fn logged_origin(raw: &str) -> String {
    match url::Url::parse(raw) {
        Ok(url) if matches!(url.scheme(), "http" | "https") && url.host_str().is_some() => {
            url.origin().ascii_serialization()
        }
        _ => "[invalid HTTP URL]".to_owned(),
    }
}

/// Build the error for a non-success response, reading and truncating the body.
pub async fn error_from_response(response: reqwest::Response) -> ProviderHttpError {
    let status = response.status().as_u16();
    let headers = response.headers().clone();
    let retry_hint = crate::utils::provider_retry::retry_after_delay_ms(&headers);
    let retryable = crate::error::is_retryable_with_headers(Some(status), &headers);
    // Bound the body before buffering it, rather than after reading an arbitrary response.
    const MAX_ERROR_BODY_BYTES: usize = 16_384;
    let mut bytes = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(chunk) => {
                let remaining = MAX_ERROR_BODY_BYTES - bytes.len();
                bytes.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
                if bytes.len() == MAX_ERROR_BODY_BYTES {
                    break;
                }
            }
            Err(error) => {
                return ProviderHttpError::from_response(
                    status,
                    Some(String::from_utf8_lossy(&bytes).into_owned()),
                    format!("{status} status code; body read failed"),
                )
                .with_retry_after_ms(retry_hint)
                .with_retryable(retryable)
                .with_source(error);
            }
        }
    }
    ProviderHttpError::from_response(
        status,
        Some(String::from_utf8_lossy(&bytes).into_owned()),
        format!("{status} status code"),
    )
    .with_retry_after_ms(retry_hint)
    .with_retryable(retryable)
}

/// POST JSON and hand back the response without consuming the body.
///
/// Only the method, origin and response status are logged at `debug` under
/// the `http` target. Headers, URL paths/parameters and bodies are omitted.
pub async fn post_json(
    client: reqwest::Client,
    url: &str,
    headers: &reqwest::header::HeaderMap,
    payload: &Value,
    timeout_ms: Option<u64>,
) -> Result<reqwest::Response, ProviderHttpError> {
    log::debug!(
        target: "http",
        "POST origin={origin}",
        origin = logged_origin(url),
    );
    let mut request = client.post(url).headers(headers.clone()).json(payload);
    if let Some(timeout_ms) = timeout_ms {
        request = request.timeout(std::time::Duration::from_millis(timeout_ms));
    }
    let result = request.send().await;
    if let Ok(response) = &result {
        log::debug!(
            target: "http",
            "response status={status} origin={origin}",
            status = response.status().as_u16(),
            origin = logged_origin(url),
        );
    }
    result.map_err(|error| {
        let error = error.without_url();
        ProviderHttpError::transport(if error.is_timeout() {
            format!("request timeout: {error}")
        } else {
            format!("request failed: {error}")
        })
        .with_retryable(error.is_timeout() || error.is_connect() || error.is_body())
        .with_source(error)
    })
}

/// POST JSON expecting an SSE response stream. Non-2xx responses become a
/// single-event error stream.
pub(crate) async fn post_json_sse_tracked(
    client: reqwest::Client,
    url: &str,
    headers: &reqwest::header::HeaderMap,
    payload: &Value,
    timeout_ms: Option<u64>,
    metrics: Option<std::sync::Arc<crate::request_metrics::RequestTracker>>,
) -> SseResponse {
    let response = match post_json(client, url, headers, payload, timeout_ms).await {
        Ok(response) => response,
        Err(error) => {
            return SseResponse {
                status: 0,
                headers: Default::default(),
                events: error_event_stream(error),
            };
        }
    };
    let status = response.status().as_u16();
    let headers = response.headers().clone();
    if !(200..300).contains(&status) {
        let error = error_from_response(response).await;
        return SseResponse {
            status,
            headers,
            events: error_event_stream(error),
        };
    }
    // Use transport arrival times, not JSON parsing times. Multiple SSE events
    // buffered in one body chunk cannot provide a generation interval.
    let byte_stream = response.bytes_stream().map(move |result| {
        if result.is_ok()
            && let Some(metrics) = &metrics
        {
            metrics.frame();
        }
        result
    });
    let events = byte_stream.eventsource().map(|result| {
        result
            .map(|event| SseEvent {
                event: event.event,
                data: event.data,
            })
            .map_err(sse_error)
    });
    SseResponse {
        status,
        headers,
        events: Box::pin(events),
    }
}

pub(crate) fn sse_error(
    error: eventsource_stream::EventStreamError<reqwest::Error>,
) -> ProviderHttpError {
    match error {
        eventsource_stream::EventStreamError::Transport(error) if error.is_timeout() => {
            ProviderHttpError::transport("SSE stream timeout").with_source(error)
        }
        error => ProviderHttpError::transport(format!("SSE stream failed: {error}")),
    }
}

fn error_event_stream(
    error: ProviderHttpError,
) -> std::pin::Pin<Box<dyn Stream<Item = Result<SseEvent, ProviderHttpError>> + Send>> {
    let events = futures::stream::iter(vec![Err(error)]);
    Box::pin(events)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::TryStreamExt;

    #[test]
    fn logged_origins_omit_private_url_components() {
        assert_eq!(
            logged_origin(
                "https://user:password@example.test:8443/private-token?key=secret#secret"
            ),
            "https://example.test:8443"
        );
        assert_eq!(logged_origin("not a URL with secret"), "[invalid HTTP URL]");
        assert_eq!(logged_origin("file:private-token"), "[invalid HTTP URL]");
    }

    struct LogSink(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for LogSink {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn post_json_logs_only_debug_metadata() {
        let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        env_logger::Builder::new()
            .filter_level(log::LevelFilter::Off)
            .filter_module("http", log::LevelFilter::Debug)
            .target(env_logger::Target::Pipe(Box::new(LogSink(
                captured.clone(),
            ))))
            .write_style(env_logger::WriteStyle::Never)
            .try_init()
            .ok();
        log::set_max_level(log::LevelFilter::Info);
        let (info_url, info_task) = server("ok", "").await;
        post_json(
            reqwest::Client::new(),
            &info_url,
            &Default::default(),
            &serde_json::json!({ "messages": ["secret-info-message"] }),
            None,
        )
        .await
        .expect("request sent at info level");
        info_task.await.unwrap();
        assert!(captured.lock().unwrap().is_empty());

        log::set_max_level(log::LevelFilter::Debug);
        let (url, task) = server("ok", "").await;
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("authorization", "Bearer secret".parse().unwrap());
        headers.insert("x-api-key", "secret-api-key".parse().unwrap());
        headers.insert("cookie", "session=secret-cookie".parse().unwrap());
        headers.insert("proxy-authorization", "Basic secret-proxy".parse().unwrap());
        headers.insert("x-custom-token", "secret-custom-token".parse().unwrap());

        let response = post_json(
            reqwest::Client::new(),
            &format!("{url}/secret-path?key=secret-query#secret-fragment"),
            &headers,
            &serde_json::json!({ "model": "test-model", "messages": ["secret-message"] }),
            None,
        )
        .await
        .expect("request sent");

        assert_eq!(response.status(), 200);
        let logged = String::from_utf8(captured.lock().unwrap().clone()).expect("utf8 log");
        assert!(logged.contains("DEBUG http] POST origin="), "{logged}");
        assert!(!logged.contains("INFO http]"), "{logged}");
        assert!(!logged.contains("test-model"), "{logged}");
        assert!(!logged.contains("secret"), "{logged}");
        assert!(logged.contains("status=200"), "{logged}");
        task.await.unwrap();
    }

    async fn server(
        body: &'static str,
        headers: &'static str,
    ) -> (String, tokio::task::JoinHandle<()>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0u8; 8192];
            let _ = socket.read(&mut request).await;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n{headers}\r\n{body}",
                body.len()
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        });
        (format!("http://{address}"), task)
    }

    #[tokio::test]
    async fn scoped_proxy_routes_by_protocol_and_obeys_no_proxy() {
        for bypass in [false, true] {
            let (origin, origin_task) = server("origin", "").await;
            let (proxy, proxy_task) = server("proxy", "").await;
            let options = ProviderRequestOptions {
                env: Some(
                    [
                        ("HTTP_PROXY".to_owned(), proxy),
                        ("HTTPS_PROXY".to_owned(), "http://127.0.0.1:1".to_owned()),
                        ("ALL_PROXY".to_owned(), "".to_owned()),
                        (
                            "NO_PROXY".to_owned(),
                            if bypass { "127.0.0.1" } else { "" }.to_owned(),
                        ),
                    ]
                    .into_iter()
                    .collect(),
                ),
                ..Default::default()
            };
            let body = http_client_for(&options)
                .unwrap()
                .get(origin)
                .timeout(std::time::Duration::from_secs(2))
                .send()
                .await
                .unwrap()
                .text()
                .await
                .unwrap();
            assert_eq!(body, if bypass { "origin" } else { "proxy" });
            origin_task.abort();
            proxy_task.abort();
        }
    }

    #[tokio::test]
    async fn invalid_proxy_does_not_fall_back_but_injected_client_wins() {
        let mut options = ProviderRequestOptions {
            env: Some(
                [("HTTPS_PROXY".to_owned(), "http://[secret".to_owned())]
                    .into_iter()
                    .collect(),
            ),
            ..Default::default()
        };
        assert!(http_client_for(&options).is_err());
        options.http_client = Some(reqwest::Client::builder().no_proxy().build().unwrap());
        assert!(http_client_for(&options).is_ok());
    }

    #[tokio::test]
    async fn server_retry_header_survives_error_normalization() {
        let (url, task) = server("error body", "x-should-retry: false\r\nretry-after: 2\r\n").await;
        let response = reqwest::Client::builder()
            .no_proxy()
            .build()
            .unwrap()
            .get(url)
            .send()
            .await
            .unwrap();
        let error = error_from_response(response).await;
        assert!(!error.is_retryable());
        assert_eq!(error.retry_after_ms(), Some(2000));
        task.await.unwrap();
    }

    #[tokio::test]
    async fn default_client_is_shared() {
        // The OnceLock guarantees one cached instance; a second call clones it.
        let client = default_http_client().unwrap();
        assert_eq!(
            client
                .get("http://127.0.0.1:1")
                .build()
                .unwrap()
                .url()
                .as_str(),
            "http://127.0.0.1:1/"
        );
    }

    #[tokio::test]
    async fn sse_parses_event_frames() {
        // Feed a raw SSE body through the eventsource framing layer.
        let chunks: Vec<Result<Vec<u8>, std::io::Error>> = vec![
            Ok(b"event: message_start\ndata: {\"type\":\"message_start\"}\n\n".to_vec()),
            Ok(b"event: content_block_delta\ndata: {\"type\":\"x\"}\n\n".to_vec()),
        ];
        let byte_stream = futures::stream::iter(chunks);
        let mut events = byte_stream.eventsource().map(|result| {
            result
                .map(|event| SseEvent {
                    event: event.event,
                    data: event.data,
                })
                .map_err(|error| {
                    ProviderHttpError::transport(format!("SSE stream failed: {error}"))
                })
        });
        let first = events.try_next().await.unwrap().expect("first event");
        assert_eq!(first.event, "message_start");
        assert_eq!(first.data, "{\"type\":\"message_start\"}");
        let second = events.try_next().await.unwrap().expect("second event");
        assert_eq!(second.event, "content_block_delta");
    }
}
