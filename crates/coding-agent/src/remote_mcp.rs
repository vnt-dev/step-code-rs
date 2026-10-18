//! One-shot Streamable HTTP MCP calls for hosted tools.

use std::{collections::HashSet, time::Duration};

use eventsource_stream::Eventsource;
use futures::StreamExt;
use reqwest::{Client, RequestBuilder, header::HeaderMap};
use serde::Deserialize;
use serde_json::{Value, json};
use step_harness_agent_core::types::{ToolError, tool_error};
use tokio_util::sync::CancellationToken;

pub const DEFAULT_MCP_TIMEOUT: Duration = Duration::from_secs(30);

pub struct RemoteMcpInvocation {
    pub server_name: String,
    pub server_url: String,
    pub tool_name: String,
    pub arguments: serde_json::Map<String, Value>,
    pub headers: HeaderMap,
    pub timeout: Duration,
}

#[derive(Default, Debug, PartialEq)]
pub struct RemoteMcpResult {
    pub is_error: bool,
    pub content: Option<String>,
    pub structured_content: Option<serde_json::Map<String, Value>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ToolCatalog {
    tools: Vec<RemoteTool>,
    next_cursor: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RemoteTool {
    name: String,
    output_schema: Option<Value>,
    execution: Option<TaskExecution>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TaskExecution {
    task_support: Option<String>,
}

struct Transport<'a> {
    client: Client,
    input: &'a RemoteMcpInvocation,
    session_id: Option<reqwest::header::HeaderValue>,
    protocol: Option<String>,
    next_id: u64,
}

impl Transport<'_> {
    fn headers(&self, request: RequestBuilder) -> RequestBuilder {
        let mut request = request
            .headers(self.input.headers.clone())
            .header("Accept", "application/json, text/event-stream");
        if let Some(id) = &self.session_id {
            request = request.header("MCP-Session-Id", id);
        }
        if let Some(protocol) = &self.protocol {
            request = request.header("MCP-Protocol-Version", protocol);
        }
        request
    }

    async fn post(&mut self, message: Value) -> Result<Value, ToolError> {
        let id = message.get("id").and_then(Value::as_u64);
        let response = self
            .headers(self.client.post(&self.input.server_url))
            .json(&message)
            .send()
            .await?
            .error_for_status()?;
        if let Some(session) = response.headers().get("MCP-Session-Id") {
            self.session_id = Some(session.clone());
        }
        if id.is_none() {
            return Ok(Value::Null);
        }
        let is_sse = response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.split(';').next() == Some("text/event-stream"));
        if is_sse {
            let mut events = response.bytes_stream().eventsource();
            while let Some(event) = events.next().await {
                let event = event?;
                if event.data.trim().is_empty() {
                    continue;
                }
                let message: Value = serde_json::from_str(&event.data)?;
                if message.get("id").and_then(Value::as_u64) == id
                    && (message.get("result").is_some() || message.get("error").is_some())
                {
                    return rpc_result(message, id);
                }
                // No client capabilities are advertised. Answer unsolicited requests
                // rather than letting a server wait forever for an unsupported method.
                if message.get("method").is_some() && message.get("id").is_some() {
                    self.headers(self.client.post(&self.input.server_url))
                        .json(&json!({"jsonrpc":"2.0", "id":message["id"],
                            "error":{"code":-32601,"message":"Method not supported"}}))
                        .send()
                        .await?
                        .error_for_status()?;
                }
            }
            Err(tool_error("MCP event stream ended without a response"))
        } else {
            rpc_result(response.json().await?, id)
        }
    }

    async fn request(&mut self, method: &str, params: Value) -> Result<Value, ToolError> {
        let id = self.next_id;
        self.next_id += 1;
        self.post(json!({"jsonrpc":"2.0", "id":id, "method":method, "params":params}))
            .await
    }

    async fn call(&mut self) -> Result<RemoteMcpResult, ToolError> {
        let initialized = self
            .request(
                "initialize",
                json!({
                    "protocolVersion":"2025-11-25", "capabilities":{},
                    "clientInfo":{"name":"stepcode", "version":env!("CARGO_PKG_VERSION")}
                }),
            )
            .await?;
        let protocol = initialized["protocolVersion"]
            .as_str()
            .ok_or_else(|| tool_error("MCP initialize response is missing protocolVersion"))?;
        if !matches!(protocol, "2025-11-25" | "2025-06-18" | "2025-03-26") {
            return Err(tool_error(format!(
                "Unsupported MCP protocol version: {protocol}"
            )));
        }
        self.protocol = Some(protocol.to_owned());
        self.post(json!({"jsonrpc":"2.0", "method":"notifications/initialized"}))
            .await?;

        let mut cursors = HashSet::new();
        let mut cursor = None;
        let mut selected = None;
        loop {
            let params = cursor
                .as_ref()
                .map_or(json!({}), |cursor| json!({"cursor":cursor}));
            let catalog: ToolCatalog =
                serde_json::from_value(self.request("tools/list", params).await?)?;
            for tool in catalog.tools {
                if tool.name == self.input.tool_name && selected.is_none() {
                    selected = Some(tool);
                }
            }
            cursor = catalog.next_cursor;
            match &cursor {
                None => break,
                Some(cursor) if !cursors.insert(cursor.clone()) => {
                    return Err(tool_error(
                        "MCP tools/list returned a repeated pagination cursor",
                    ));
                }
                _ => {}
            }
        }
        let tool = selected.ok_or_else(|| {
            tool_error(format!(
                "Remote MCP server '{}' does not expose tool '{}'",
                self.input.server_name, self.input.tool_name
            ))
        })?;
        if tool
            .execution
            .as_ref()
            .and_then(|execution| execution.task_support.as_deref())
            == Some("required")
        {
            return Err(tool_error(format!(
                "Tool '{}' requires unsupported task-based execution",
                tool.name
            )));
        }
        let validator = tool
            .output_schema
            .as_ref()
            .map(jsonschema::validator_for)
            .transpose()?;
        let raw = self
            .request(
                "tools/call",
                json!({"name":tool.name, "arguments":self.input.arguments}),
            )
            .await?;
        let result = normalize_mcp_result(raw)?;
        if let Some(validator) = validator {
            if let Some(structured) = &result.structured_content {
                validator
                    .validate(&Value::Object(structured.clone()))
                    .map_err(|error| {
                        tool_error(format!(
                            "Structured content does not match the tool's output schema: {error}"
                        ))
                    })?;
            } else if !result.is_error {
                return Err(tool_error(
                    "MCP tool has an output schema but did not return structured content",
                ));
            }
        }
        Ok(result)
    }

    async fn close(&self) {
        if self.session_id.is_some() {
            // Cleanup must never hold up cancellation or mask the original failure.
            let _ = tokio::time::timeout(
                Duration::from_secs(1),
                self.headers(self.client.delete(&self.input.server_url))
                    .send(),
            )
            .await;
        }
    }
}

fn rpc_result(message: Value, id: Option<u64>) -> Result<Value, ToolError> {
    if message["jsonrpc"] != "2.0" || message.get("id").and_then(Value::as_u64) != id {
        return Err(tool_error("Invalid MCP JSON-RPC response"));
    }
    if let Some(error) = message.get("error") {
        return Err(tool_error(format!("MCP RPC error: {error}")));
    }
    message
        .get("result")
        .cloned()
        .ok_or_else(|| tool_error("MCP response is missing result"))
}

pub fn normalize_mcp_result(raw: Value) -> Result<RemoteMcpResult, ToolError> {
    let payload = if raw.get("content").is_some_and(Value::is_array) {
        &raw
    } else {
        raw.get("toolResult").unwrap_or(&Value::Null)
    };
    if let Some(content) = payload.get("content").and_then(Value::as_array) {
        let text = content
            .iter()
            .filter(|item| item["type"] == "text")
            .map(|item| {
                item["text"]
                    .as_str()
                    .ok_or_else(|| tool_error("Invalid MCP text content"))
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n");
        Ok(RemoteMcpResult {
            is_error: payload["isError"].as_bool().unwrap_or(false),
            content: (!text.is_empty()).then_some(text),
            structured_content: payload
                .get("structuredContent")
                .and_then(Value::as_object)
                .cloned(),
        })
    } else {
        Ok(RemoteMcpResult {
            is_error: payload["isError"].as_bool().unwrap_or(false),
            content: (!payload.is_null())
                .then(|| serde_json::to_string_pretty(payload))
                .transpose()?,
            structured_content: payload.as_object().cloned(),
        })
    }
}

/// A cancellation or deadline drops the in-flight HTTP future, then closes the session.
pub async fn invoke_remote_mcp_tool(
    input: &RemoteMcpInvocation,
    signal: Option<&CancellationToken>,
) -> Result<RemoteMcpResult, ToolError> {
    let mut transport = Transport {
        client: Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()?,
        input,
        session_id: None,
        protocol: None,
        next_id: 1,
    };
    let cancelled = async {
        match signal {
            Some(signal) => signal.cancelled().await,
            None => std::future::pending().await,
        }
    };
    let result = tokio::select! {
        biased;
        _ = cancelled => Err(tool_error("Operation aborted")),
        result = tokio::time::timeout(input.timeout, transport.call()) => {
            result.unwrap_or_else(|_| Err(tool_error("MCP request timed out")))
        }
    };
    transport.close().await;
    result.map_err(|error| {
        tool_error(format!(
            "MCP tool {}.{} failed: {error}",
            input.server_name, input.tool_name
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };

    struct Reply {
        body: String,
        sse: bool,
        delay: Duration,
        status: u16,
    }

    impl Reply {
        fn json(id: u64, result: Value) -> Self {
            Self {
                body: json!({"jsonrpc":"2.0", "id":id, "result":result}).to_string(),
                sse: false,
                delay: Duration::ZERO,
                status: 200,
            }
        }
        fn empty() -> Self {
            Self {
                body: String::new(),
                sse: false,
                delay: Duration::ZERO,
                status: 202,
            }
        }
    }

    struct Server {
        url: String,
        requests: Arc<Mutex<Vec<(String, Value)>>>,
        task: tokio::task::JoinHandle<()>,
    }

    impl Drop for Server {
        fn drop(&mut self) {
            self.task.abort();
        }
    }

    async fn server(replies: Vec<Reply>) -> Server {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/mcp", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let saved = requests.clone();
        let task = tokio::spawn(async move {
            for reply in replies {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut bytes = Vec::new();
                let (head, body) = loop {
                    let mut chunk = [0; 4096];
                    let read = stream.read(&mut chunk).await.unwrap();
                    assert!(read > 0);
                    bytes.extend_from_slice(&chunk[..read]);
                    if let Some(end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                        let head = String::from_utf8(bytes[..end].to_vec()).unwrap();
                        let length = head
                            .lines()
                            .find_map(|line| {
                                let (name, value) = line.split_once(':')?;
                                name.eq_ignore_ascii_case("content-length")
                                    .then(|| value.trim().parse::<usize>().unwrap())
                            })
                            .unwrap_or(0);
                        if bytes.len() >= end + 4 + length {
                            let body = if length == 0 {
                                Value::Null
                            } else {
                                serde_json::from_slice(&bytes[end + 4..end + 4 + length]).unwrap()
                            };
                            break (head, body);
                        }
                    }
                };
                saved.lock().unwrap().push((head, body));
                tokio::time::sleep(reply.delay).await;
                let mime = if reply.sse {
                    "text/event-stream"
                } else {
                    "application/json"
                };
                let header = format!(
                    "HTTP/1.1 {} OK\r\nContent-Type: {mime}\r\nContent-Length: {}\r\nMCP-Session-Id: test-session\r\nConnection: close\r\n\r\n",
                    reply.status,
                    reply.body.len()
                );
                let _ = stream.write_all(header.as_bytes()).await;
                let _ = stream.write_all(reply.body.as_bytes()).await;
            }
        });
        Server {
            url,
            requests,
            task,
        }
    }

    fn initialize() -> Reply {
        Reply::json(
            1,
            json!({"protocolVersion":"2025-06-18","capabilities":{"tools":{}},"serverInfo":{"name":"mock","version":"1"}}),
        )
    }

    fn invocation(server: &Server) -> RemoteMcpInvocation {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer test-key".parse().unwrap());
        RemoteMcpInvocation {
            server_name: "stepsearch".to_owned(),
            server_url: server.url.clone(),
            tool_name: "web_search".to_owned(),
            arguments: json!({"query":"rust", "n":10}).as_object().unwrap().clone(),
            headers,
            timeout: Duration::from_secs(3),
        }
    }

    #[tokio::test]
    async fn performs_handshake_pagination_sse_call_validation_and_cleanup() {
        let raw = json!({"jsonrpc":"2.0", "id":4, "result":{"content":[{"type":"text","text":" Search results "}], "structuredContent":{"results":[]}}});
        let reply = Reply {
            body: format!(
                "event: message\r\ndata: {{\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\",\"params\":{{}}}}\r\n\r\nevent: message\r\ndata: {raw}\r\n\r\n"
            ),
            sse: true,
            delay: Duration::ZERO,
            status: 200,
        };
        let server = server(vec![initialize(), Reply::empty(),
            Reply::json(2, json!({"tools":[{"name":"other"}], "nextCursor":"page2"})),
            Reply::json(3, json!({"tools":[{"name":"web_search", "outputSchema":{"type":"object", "properties":{"results":{"type":"array"}}, "required":["results"]}}]})),
            reply, Reply::empty()]).await;
        let result = invoke_remote_mcp_tool(&invocation(&server), None)
            .await
            .unwrap();
        assert_eq!(result.content.as_deref(), Some("Search results"));
        assert_eq!(result.structured_content.unwrap()["results"], json!([]));
        let requests = server.requests.lock().unwrap();
        assert_eq!(requests.len(), 6);
        assert_eq!(requests[0].1["method"], "initialize");
        assert_eq!(requests[1].1["method"], "notifications/initialized");
        assert_eq!(requests[3].1["params"]["cursor"], "page2");
        assert_eq!(
            requests[4].1["params"],
            json!({"name":"web_search", "arguments":{"query":"rust","n":10}})
        );
        assert!(requests[5].0.starts_with("DELETE "));
        for (index, (head, _)) in requests.iter().enumerate() {
            let head = head.to_ascii_lowercase();
            assert!(head.contains("authorization: bearer test-key"));
            if index > 0 {
                assert!(head.contains("mcp-session-id: test-session"));
                assert!(head.contains("mcp-protocol-version: 2025-06-18"));
            }
        }
    }

    #[tokio::test]
    async fn rejects_missing_tools_required_tasks_and_repeated_cursors() {
        for (catalogs, expected) in [
            (vec![json!({"tools":[]})], "does not expose"),
            (
                vec![
                    json!({"tools":[{"name":"web_search", "execution":{"taskSupport":"required"}}]}),
                ],
                "task-based",
            ),
            (
                vec![
                    json!({"tools":[], "nextCursor":"same"}),
                    json!({"tools":[], "nextCursor":"same"}),
                ],
                "repeated pagination",
            ),
        ] {
            let mut replies = vec![initialize(), Reply::empty()];
            replies.extend(
                catalogs
                    .into_iter()
                    .enumerate()
                    .map(|(index, value)| Reply::json(index as u64 + 2, value)),
            );
            replies.push(Reply::empty());
            let server = server(replies).await;
            let error = invoke_remote_mcp_tool(&invocation(&server), None)
                .await
                .unwrap_err()
                .to_string();
            assert!(error.contains(expected), "{error}");
            assert!(
                server
                    .requests
                    .lock()
                    .unwrap()
                    .last()
                    .unwrap()
                    .0
                    .starts_with("DELETE ")
            );
        }
    }

    #[tokio::test]
    async fn validates_output_and_reports_rpc_and_http_failures() {
        for (reply, expected) in [
            (Reply::json(3, json!({"content":[], "structuredContent":{"results":"invalid"}})), "output schema"),
            (Reply::json(3, json!({"content":[]})), "did not return structured content"),
            (Reply {body:json!({"jsonrpc":"2.0","id":3,"error":{"code":-32603,"message":"upstream failure"}}).to_string(), ..Reply::empty()}, "upstream failure"),
            (Reply {status:502, ..Reply::empty()}, "502"),
        ] {
            let server = server(vec![initialize(), Reply::empty(), Reply::json(2, json!({"tools":[{"name":"web_search", "outputSchema":{"type":"object", "properties":{"results":{"type":"array"}}, "required":["results"]}}]})), reply, Reply::empty()]).await;
            let error = invoke_remote_mcp_tool(&invocation(&server), None).await.unwrap_err().to_string();
            assert!(error.contains(expected), "{error}");
        }
    }

    #[tokio::test]
    async fn cancellation_and_timeout_abort_blocked_http() {
        let server = server(vec![Reply {
            delay: Duration::from_secs(60),
            ..initialize()
        }])
        .await;
        let mut input = invocation(&server);
        let token = CancellationToken::new();
        token.cancel();
        let error = invoke_remote_mcp_tool(&input, Some(&token))
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("aborted"));
        assert!(server.requests.lock().unwrap().is_empty());
        input.timeout = Duration::from_millis(30);
        let error = invoke_remote_mcp_tool(&input, None)
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("timed out"));
        input.timeout = Duration::from_secs(3);
        let token = CancellationToken::new();
        let cancel = token.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            cancel.cancel();
        });
        let error = invoke_remote_mcp_tool(&input, Some(&token))
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("aborted"));
    }

    #[test]
    fn normalizes_text_errors_and_legacy_envelopes() {
        let result = normalize_mcp_result(json!({"content":[{"type":"text","text":" one "},{"type":"image","data":"ignored"},{"type":"text","text":"two"}],"isError":true})).unwrap();
        assert!(result.is_error);
        assert_eq!(result.content.as_deref(), Some("one\n\ntwo"));
        let result = normalize_mcp_result(json!({"toolResult":{"results":[]}})).unwrap();
        assert_eq!(result.structured_content.unwrap()["results"], json!([]));
        let result = normalize_mcp_result(
            json!({"toolResult":{"content":[{"type":"text","text":"legacy"}]}}),
        )
        .unwrap();
        assert_eq!(result.content.as_deref(), Some("legacy"));
        assert!(rpc_result(json!({"jsonrpc":"2.0", "id":99, "result":{}}), Some(1)).is_err());
    }
}
