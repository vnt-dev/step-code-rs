//! Step web search, including account routing and compact result rendering.

use std::{collections::HashMap, path::PathBuf, sync::Arc};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use step_harness_agent_core::{
    harness::types::AbortSignal,
    types::{
        AgentTool, AgentToolResult, ToolError, ToolExecutionMode, ToolExecutor, ToolUpdateCallback,
        tool_error,
    },
};

use crate::{
    paths::AppPaths,
    remote_mcp::{
        DEFAULT_MCP_TIMEOUT, RemoteMcpInvocation, RemoteMcpResult, invoke_remote_mcp_tool,
    },
};

pub const SEARCH_WEB_SERVER_NAME: &str = "stepsearch";
pub const SEARCH_WEB_TOOL_NAME: &str = "web_search";
pub const SEARCH_WEB_MAINLAND_URL: &str = "https://api.stepfun.com/v1/mcp/web_search/mcp";
pub const SEARCH_WEB_OVERSEA_URL: &str = "https://api.stepfun.ai/v1/mcp/web_search/mcp";
pub const SEARCH_WEB_PLAN_MAINLAND_URL: &str =
    "https://api.stepfun.com/step_plan/v1/mcp/web_search/mcp";
pub const SEARCH_WEB_PLAN_OVERSEA_URL: &str =
    "https://api.stepfun.ai/step_plan/v1/mcp/web_search/mcp";
pub const SEARCH_WEB_DEFAULT_URL: &str = SEARCH_WEB_MAINLAND_URL;
const RESULT_COUNT: u32 = 10;
const MAX_SNIPPET_CHARS: usize = 400;

/// Credentials intentionally do not implement Debug.
#[derive(Default)]
pub struct SearchWebToolOptions {
    pub url: Option<String>,
    pub api_key: Option<String>,
    pub auth_path: Option<PathBuf>,
    /// None reads the process environment; Some supplies an isolated environment.
    pub env: Option<HashMap<String, String>>,
}

#[derive(Deserialize)]
struct SearchWebInput {
    query: String,
}

#[derive(Serialize)]
struct SearchArguments<'a> {
    query: &'a str,
    n: u32,
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchWebDetails {
    pub query: String,
    pub result_count: usize,
    pub urls: Vec<String>,
}

/// An injectable remote call keeps transport and tool policy independently testable.
#[async_trait::async_trait]
pub trait SearchWebInvoker: Send + Sync {
    async fn invoke(
        &self,
        input: &RemoteMcpInvocation,
        signal: Option<&AbortSignal>,
    ) -> Result<RemoteMcpResult, ToolError>;
}

struct HttpInvoker;

#[async_trait::async_trait]
impl SearchWebInvoker for HttpInvoker {
    async fn invoke(
        &self,
        input: &RemoteMcpInvocation,
        signal: Option<&AbortSignal>,
    ) -> Result<RemoteMcpResult, ToolError> {
        invoke_remote_mcp_tool(input, signal).await
    }
}

fn text(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

fn credential(value: Option<&str>) -> Option<&str> {
    text(value).filter(|value| *value != "<your_api_key>")
}

fn auth_path(options: &SearchWebToolOptions, env: &HashMap<String, String>) -> PathBuf {
    options
        .auth_path
        .clone()
        .or_else(|| text(env.get("STEPCODE_AUTH_PATH").map(String::as_str)).map(PathBuf::from))
        .unwrap_or_else(|| {
            let root = env
                .get(crate::paths::HOME_ENV)
                .filter(|value| !value.is_empty())
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    dirs::home_dir()
                        .unwrap_or_else(|| PathBuf::from("."))
                        .join(crate::paths::APP_DIR_NAME)
                });
            AppPaths::new(root).auth_file()
        })
}

fn stored_auth(path: &std::path::Path) -> Value {
    // Like the source's readStoredCredential, missing or malformed stores do
    // not prevent tool registration. Execution reports the missing credential.
    parse_stored_auth(std::fs::read_to_string(path).ok())
}

fn parse_stored_auth(document: Option<String>) -> Value {
    document
        .and_then(|value| serde_json::from_str::<Value>(value.trim_start_matches('\u{feff}')).ok())
        .and_then(|mut value| value.get_mut("step").map(Value::take))
        .unwrap_or(Value::Null)
}

pub fn resolve_search_web_api_key(options: &SearchWebToolOptions) -> Option<String> {
    let env = options
        .env
        .clone()
        .unwrap_or_else(|| std::env::vars().collect());
    resolve_key(options, &env, &stored_auth(&auth_path(options, &env)))
}

fn resolve_key(
    options: &SearchWebToolOptions,
    env: &HashMap<String, String>,
    stored: &Value,
) -> Option<String> {
    // STEP_API_KEY belongs to the model gateway and must never be sent here.
    credential(options.api_key.as_deref())
        .or_else(|| credential(env.get("STEPCODE_SEARCH_API_KEY").map(String::as_str)))
        .or_else(|| credential(stored["access"].as_str()))
        .or_else(|| credential(stored["key"].as_str()))
        .map(str::to_owned)
}

pub fn resolve_search_web_server_url(
    configured: Option<&str>,
    env: &HashMap<String, String>,
    profile: Option<&str>,
) -> String {
    let default = match text(profile) {
        Some("step" | "step_plan") => SEARCH_WEB_PLAN_MAINLAND_URL,
        Some("step_plan_oversea") => SEARCH_WEB_PLAN_OVERSEA_URL,
        Some("platform_oversea") => SEARCH_WEB_OVERSEA_URL,
        _ => SEARCH_WEB_DEFAULT_URL,
    };
    let value = text(configured)
        .or_else(|| text(env.get("STEPCODE_SEARCH_WEB_MCP_URL").map(String::as_str)))
        .unwrap_or(default);
    let Ok(parsed) = url::Url::parse(value) else {
        return value.to_owned();
    };
    if parsed.path() != "/" && !parsed.path().is_empty() {
        return value.trim_end_matches('/').to_owned();
    }
    let path = if matches!(
        text(profile),
        Some("step" | "step_plan" | "step_plan_oversea")
    ) {
        "/step_plan/v1/mcp/web_search/mcp"
    } else {
        "/v1/mcp/web_search/mcp"
    };
    format!("{}{path}", parsed.origin().ascii_serialization())
}

pub fn build_search_web_description() -> String {
    [
        "Search the web when the answer depends on current or external information: recent news, fresh documentation, live data, or anything outside built-in knowledge.",
        "The tool returns compact structured results with markdown links.",
        "If this tool informs the answer, end the response with a Sources: section containing the relevant result URLs as markdown links.",
    ].join("\n\n")
}

pub fn create_search_web_tool(options: SearchWebToolOptions) -> AgentTool {
    create_search_web_tool_with_invoker(options, Arc::new(HttpInvoker))
}

pub fn create_search_web_tool_with_invoker(
    options: SearchWebToolOptions,
    invoker: Arc<dyn SearchWebInvoker>,
) -> AgentTool {
    AgentTool {
        name: "search_web".to_owned(),
        label: "search_web".to_owned(),
        description: build_search_web_description(),
        parameters: json!({"type":"object", "properties":{"query":{"type":"string", "description":"Search query text"}}, "required":["query"]}),
        execution_mode: Some(ToolExecutionMode::Parallel),
        prepare_arguments: None,
        executor: Arc::new(SearchWebExecutor { options, invoker }),
    }
}

struct SearchWebExecutor {
    options: SearchWebToolOptions,
    invoker: Arc<dyn SearchWebInvoker>,
}

#[async_trait::async_trait]
impl ToolExecutor for SearchWebExecutor {
    async fn execute(
        &self,
        _tool_call_id: &str,
        params: &serde_json::Map<String, Value>,
        signal: Option<&AbortSignal>,
        _on_update: Option<&ToolUpdateCallback>,
    ) -> Result<AgentToolResult, ToolError> {
        let args: SearchWebInput = serde_json::from_value(Value::Object(params.clone()))?;
        let query = args.query.trim();
        if query.is_empty() {
            return Err(tool_error("search_web query must not be empty"));
        }
        // Login/logout can happen after a session was created. Resolve the
        // credential and its billing profile from one current file snapshot.
        let env = self
            .options
            .env
            .clone()
            .unwrap_or_else(|| std::env::vars().collect());
        let stored = parse_stored_auth(
            tokio::fs::read_to_string(auth_path(&self.options, &env))
                .await
                .ok(),
        );
        let server_url = resolve_search_web_server_url(
            self.options.url.as_deref(),
            &env,
            stored["profile"].as_str(),
        );
        let resolved_key = resolve_key(&self.options, &env, &stored);
        let api_key = resolved_key.as_deref().ok_or_else(|| tool_error(
            "search_web requires a credential; configure a Step credential or set STEPCODE_SEARCH_API_KEY"))?;
        let diagnose = |error: String| {
            tool_error(format!("{error} (endpoint {server_url})").replace(api_key, "[redacted]"))
        };
        let mut authorization =
            reqwest::header::HeaderValue::from_str(&format!("Bearer {api_key}"))
                .map_err(|_| diagnose("Invalid search credential header".to_owned()))?;
        authorization.set_sensitive(true);
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(reqwest::header::AUTHORIZATION, authorization);
        let arguments = serde_json::to_value(SearchArguments {
            query,
            n: RESULT_COUNT,
        })?
        .as_object()
        .expect("SearchArguments serializes as an object")
        .clone();
        let input = RemoteMcpInvocation {
            server_name: SEARCH_WEB_SERVER_NAME.to_owned(),
            server_url: server_url.clone(),
            tool_name: SEARCH_WEB_TOOL_NAME.to_owned(),
            arguments,
            headers,
            timeout: DEFAULT_MCP_TIMEOUT,
        };
        let raw = self
            .invoker
            .invoke(&input, signal)
            .await
            .map_err(|error| diagnose(error.to_string()))?;
        if raw.is_error {
            return Err(diagnose(raw.content.unwrap_or_else(|| {
                "MCP tool stepsearch.web_search failed".to_owned()
            })));
        }
        render_search_results(query, raw)
    }
}

fn shorten_text(value: &str) -> String {
    let normalized = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.chars().count() > MAX_SNIPPET_CHARS {
        format!(
            "{}...",
            normalized
                .chars()
                .take(MAX_SNIPPET_CHARS - 3)
                .collect::<String>()
        )
    } else {
        normalized
    }
}

fn render_search_results(query: &str, raw: RemoteMcpResult) -> Result<AgentToolResult, ToolError> {
    let items = raw
        .structured_content
        .as_ref()
        .and_then(|value| value.get("results"))
        .and_then(Value::as_array);
    let mut urls = Vec::new();
    let (content, result_count) = if let Some(items) = items {
        let content = items
            .iter()
            .enumerate()
            .map(|(index, item)| {
                let url = text(item["url"].as_str());
                let title = text(item["title"].as_str());
                // Positions are ranks, so fractional, negative and zero values fall
                // back to the ordinal instead of propagating arbitrary TS numbers.
                let position = item["position"]
                    .as_u64()
                    .filter(|rank| *rank > 0)
                    .map_or_else(|| (index + 1).to_string(), |rank| rank.to_string());
                let snippet = shorten_text(
                    text(item["snippet"].as_str())
                        .or_else(|| text(item["content"].as_str()))
                        .unwrap_or(""),
                );
                let heading = if let Some(url) = url {
                    urls.push(url.to_owned());
                    format!("{position}. [{}]({url})", title.unwrap_or(url))
                } else {
                    format!("{position}. {}", title.unwrap_or("(untitled)"))
                };
                if snippet.is_empty() {
                    heading
                } else {
                    format!("{heading}\n   {snippet}")
                }
            })
            .collect::<Vec<_>>()
            .join("\n\n");
        (
            if items.is_empty() {
                format!("No search results for \"{query}\".")
            } else {
                content
            },
            items.len(),
        )
    } else {
        (
            text(raw.content.as_deref())
                .map(str::to_owned)
                .unwrap_or_else(|| format!("No search results for \"{query}\".")),
            0,
        )
    };
    let details = SearchWebDetails {
        query: query.to_owned(),
        result_count,
        urls,
    };
    Ok(AgentToolResult::text(
        content,
        serde_json::to_value(details)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct Invoker {
        calls: Mutex<Vec<(String, Value, String, bool)>>,
        result: Mutex<Option<Result<RemoteMcpResult, String>>>,
    }

    #[async_trait::async_trait]
    impl SearchWebInvoker for Invoker {
        async fn invoke(
            &self,
            input: &RemoteMcpInvocation,
            signal: Option<&AbortSignal>,
        ) -> Result<RemoteMcpResult, ToolError> {
            self.calls.lock().unwrap().push((
                input.server_url.clone(),
                Value::Object(input.arguments.clone()),
                input.headers["authorization"].to_str().unwrap().to_owned(),
                signal.is_some_and(AbortSignal::is_cancelled),
            ));
            self.result
                .lock()
                .unwrap()
                .take()
                .unwrap()
                .map_err(tool_error)
        }
    }

    fn invoker(result: Result<RemoteMcpResult, String>) -> Arc<Invoker> {
        Arc::new(Invoker {
            calls: Mutex::new(Vec::new()),
            result: Mutex::new(Some(result)),
        })
    }

    async fn execute(
        tool: &AgentTool,
        query: &str,
        signal: Option<&AbortSignal>,
    ) -> Result<AgentToolResult, ToolError> {
        tool.executor
            .execute(
                "call",
                json!({"query":query}).as_object().unwrap(),
                signal,
                None,
            )
            .await
    }

    #[test]
    fn endpoints_follow_billing_profiles_and_overrides() {
        for (profile, expected) in [
            ("step_plan", SEARCH_WEB_PLAN_MAINLAND_URL),
            ("step", SEARCH_WEB_PLAN_MAINLAND_URL),
            ("step_plan_oversea", SEARCH_WEB_PLAN_OVERSEA_URL),
            ("platform_cn", SEARCH_WEB_MAINLAND_URL),
            ("platform_oversea", SEARCH_WEB_OVERSEA_URL),
            ("unknown", SEARCH_WEB_DEFAULT_URL),
        ] {
            assert_eq!(
                resolve_search_web_server_url(None, &HashMap::new(), Some(profile)),
                expected
            );
            let path = url::Url::parse(expected).unwrap().path().to_owned();
            assert_eq!(
                resolve_search_web_server_url(
                    Some("https://proxy.example.com/"),
                    &HashMap::new(),
                    Some(profile)
                ),
                format!("https://proxy.example.com{path}")
            );
            assert_eq!(
                resolve_search_web_server_url(
                    Some("https://proxy.example.com/custom/mcp/"),
                    &HashMap::new(),
                    Some(profile)
                ),
                "https://proxy.example.com/custom/mcp"
            );
        }
        let env = HashMap::from([(
            "STEPCODE_SEARCH_WEB_MCP_URL".to_owned(),
            "https://env.example.com".to_owned(),
        )]);
        assert_eq!(
            resolve_search_web_server_url(Some(" "), &env, None),
            "https://env.example.com/v1/mcp/web_search/mcp"
        );
        assert_eq!(
            resolve_search_web_server_url(Some("invalid-url"), &env, None),
            "invalid-url"
        );
    }

    #[test]
    fn credentials_use_explicit_then_search_env_then_stored_access_or_key() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.json");
        for stored in [
            json!({"step":{"type":"oauth", "access":"stored"}}),
            json!({"step":{"type":"api_key", "key":"stored"}}),
        ] {
            std::fs::write(&path, stored.to_string()).unwrap();
            let mut options = SearchWebToolOptions {
                api_key: Some(" explicit ".to_owned()),
                auth_path: Some(path.clone()),
                env: Some(HashMap::from([
                    ("STEPCODE_SEARCH_API_KEY".to_owned(), "search".to_owned()),
                    ("STEP_API_KEY".to_owned(), "model".to_owned()),
                ])),
                ..Default::default()
            };
            assert_eq!(
                resolve_search_web_api_key(&options).as_deref(),
                Some("explicit")
            );
            options.api_key = Some("<your_api_key>".to_owned());
            assert_eq!(
                resolve_search_web_api_key(&options).as_deref(),
                Some("search")
            );
            options
                .env
                .as_mut()
                .unwrap()
                .remove("STEPCODE_SEARCH_API_KEY");
            assert_eq!(
                resolve_search_web_api_key(&options).as_deref(),
                Some("stored")
            );
        }
        std::fs::write(&path, "invalid").unwrap();
        let options = SearchWebToolOptions {
            auth_path: Some(path),
            env: Some(HashMap::from([(
                "STEP_API_KEY".to_owned(),
                "model".to_owned(),
            )])),
            ..Default::default()
        };
        assert_eq!(resolve_search_web_api_key(&options), None);
        let env = HashMap::from([(
            crate::paths::HOME_ENV.to_owned(),
            dir.path().to_string_lossy().into_owned(),
        )]);
        assert_eq!(
            auth_path(&SearchWebToolOptions::default(), &env),
            dir.path().join("auth.json")
        );
    }

    #[tokio::test]
    async fn invokes_and_compacts_unicode_results_with_integer_ranks() {
        let dir = tempfile::tempdir().unwrap();
        let remote = invoker(Ok(RemoteMcpResult {structured_content: Some(json!({"results":[
            {"position":7, "url":"https://example.com", "title":" Example ", "snippet":"short\n snippet", "content":"full page must not reach model"},
            {"position":1.5, "content":"中".repeat(450)}, null, {"position":-1,"title":"fallback"}
        ]}).as_object().unwrap().clone()), ..Default::default()}));
        let tool = create_search_web_tool_with_invoker(
            SearchWebToolOptions {
                api_key: Some("secret".to_owned()),
                auth_path: Some(dir.path().join("missing-auth.json")),
                env: Some(HashMap::new()),
                ..Default::default()
            },
            remote.clone(),
        );
        assert_eq!(tool.execution_mode, Some(ToolExecutionMode::Parallel));
        let signal = AbortSignal::new();
        signal.cancel();
        let result = execute(&tool, " query ", Some(&signal)).await.unwrap();
        let call = &remote.calls.lock().unwrap()[0];
        assert_eq!(
            call,
            &(
                SEARCH_WEB_DEFAULT_URL.to_owned(),
                json!({"query":"query","n":10}),
                "Bearer secret".to_owned(),
                true
            )
        );
        let serialized = serde_json::to_string(&result).unwrap();
        assert!(serialized.contains("7. [Example](https://example.com)"));
        assert!(serialized.contains("short snippet"));
        assert!(serialized.contains("2. (untitled)"));
        assert!(serialized.contains("4. fallback"));
        assert!(serialized.contains(&format!("{}...", "中".repeat(397))));
        assert!(!serialized.contains("full page must not reach model"));
        assert!(!serialized.contains("secret"));
        assert_eq!(
            result.details,
            json!({"query":"query", "resultCount":4,"urls":["https://example.com"]})
        );
    }

    #[tokio::test]
    async fn persisted_profiles_route_execution() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.json");
        for profile in [
            None,
            Some("step"),
            Some("step_plan"),
            Some("step_plan_oversea"),
            Some("platform_cn"),
            Some("platform_oversea"),
        ] {
            std::fs::write(
                &path,
                json!({"step":{"type":"oauth","access":"key","profile":profile}}).to_string(),
            )
            .unwrap();
            let remote = invoker(Ok(RemoteMcpResult::default()));
            let tool = create_search_web_tool_with_invoker(
                SearchWebToolOptions {
                    auth_path: Some(path.clone()),
                    env: Some(HashMap::new()),
                    ..Default::default()
                },
                remote.clone(),
            );
            execute(&tool, "query", None).await.unwrap();
            assert_eq!(
                remote.calls.lock().unwrap()[0].0,
                resolve_search_web_server_url(None, &HashMap::new(), profile)
            );
        }
    }

    #[tokio::test]
    async fn existing_tools_follow_login_profile_changes_and_logout() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.json");
        let remote = invoker(Ok(RemoteMcpResult::default()));
        let tool = create_search_web_tool_with_invoker(
            SearchWebToolOptions {
                auth_path: Some(path.clone()),
                env: Some(HashMap::new()),
                ..Default::default()
            },
            remote.clone(),
        );
        // The tool exists before login; authentication must be resolved at call time.
        assert!(
            execute(&tool, "query", None)
                .await
                .unwrap_err()
                .to_string()
                .contains("requires a credential")
        );
        for (index, (profile, key, endpoint)) in [
            ("step_plan", "plan-key", SEARCH_WEB_PLAN_MAINLAND_URL),
            (
                "step_plan_oversea",
                "oversea-key",
                SEARCH_WEB_PLAN_OVERSEA_URL,
            ),
            ("platform_cn", "platform-key", SEARCH_WEB_MAINLAND_URL),
        ]
        .into_iter()
        .enumerate()
        {
            std::fs::write(
                &path,
                json!({"step":{"type":"oauth","access":key,"profile":profile}}).to_string(),
            )
            .unwrap();
            *remote.result.lock().unwrap() = Some(Ok(RemoteMcpResult::default()));
            execute(&tool, "query", None).await.unwrap();
            let calls = remote.calls.lock().unwrap();
            assert_eq!(calls[index].0, endpoint);
            assert_eq!(calls[index].2, format!("Bearer {key}"));
        }
        std::fs::write(&path, "{}").unwrap();
        assert!(
            execute(&tool, "query", None)
                .await
                .unwrap_err()
                .to_string()
                .contains("requires a credential")
        );
        assert_eq!(remote.calls.lock().unwrap().len(), 3);
    }

    #[tokio::test]
    async fn validates_before_network_and_redacts_remote_errors() {
        let dir = tempfile::tempdir().unwrap();
        let remote = invoker(Ok(RemoteMcpResult::default()));
        let tool = create_search_web_tool_with_invoker(
            SearchWebToolOptions {
                auth_path: Some(dir.path().join("missing")),
                env: Some(HashMap::new()),
                ..Default::default()
            },
            remote.clone(),
        );
        assert!(
            execute(&tool, " ", None)
                .await
                .unwrap_err()
                .to_string()
                .contains("must not be empty")
        );
        assert!(
            execute(&tool, "query", None)
                .await
                .unwrap_err()
                .to_string()
                .contains("requires a credential")
        );
        assert!(remote.calls.lock().unwrap().is_empty());
        for result in [
            Err("transport secret unavailable".to_owned()),
            Ok(RemoteMcpResult {
                is_error: true,
                content: Some("remote secret error".to_owned()),
                ..Default::default()
            }),
        ] {
            let tool = create_search_web_tool_with_invoker(
                SearchWebToolOptions {
                    api_key: Some("secret".to_owned()),
                    auth_path: Some(dir.path().join("missing-auth.json")),
                    env: Some(HashMap::new()),
                    ..Default::default()
                },
                invoker(result),
            );
            let error = execute(&tool, "query", None).await.unwrap_err().to_string();
            assert!(error.contains(&format!("endpoint {SEARCH_WEB_DEFAULT_URL}")));
            assert!(!error.contains("secret"));
            assert!(error.contains("[redacted]"));
        }
    }

    #[test]
    fn handles_text_fallback_and_empty_results() {
        for raw in [
            RemoteMcpResult::default(),
            RemoteMcpResult {
                structured_content: Some(json!({"results":[]}).as_object().unwrap().clone()),
                ..Default::default()
            },
        ] {
            let result = render_search_results("query", raw).unwrap();
            assert!(
                serde_json::to_string(&result)
                    .unwrap()
                    .contains("No search results")
            );
            assert_eq!(result.details["resultCount"], 0);
        }
        let result = render_search_results(
            "query",
            RemoteMcpResult {
                content: Some(" fallback text ".to_owned()),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            serde_json::to_string(&result)
                .unwrap()
                .contains("fallback text")
        );
    }
}
