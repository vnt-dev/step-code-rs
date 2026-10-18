//! Cloud-polled CLI login for the Step provider: the terminal asks the
//! developer center to start a login flow, shows the user a URL, and then polls
//! for the result.
//!
//! Nothing is pushed to the terminal: the CLI only makes outbound HTTPS
//! requests and never listens on a port, so the flow survives SSH, containers
//! and hosts without a browser.
//!
//! Two values with two different roles, which must never be conflated:
//!
//! - `flow_id`: public identifier. It travels in the browser URL, so it ends up
//!   in history, screenshots and possibly Referer headers. It can never be
//!   redeemed for a credential on its own.
//! - `poll_token`: the only credential on the CLI side. It appears exclusively
//!   in the `Authorization` header and never in a URL, a log or on disk.
//!
//! Rust port of `packages/providers/src/step-provider/login-client.ts`.

use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio_util::sync::CancellationToken;

use crate::utils::{Aborted, race_with_abort};

/// Route prefix served by the developer center (`authBaseUrl`).
pub const STEP_CLI_LOGIN_PATH_PREFIX: &str = "/api/stdhttp/v1/cli-login";

/// Approval path on the developer-center origin; the browser is sent here.
const STEP_CLI_LOGIN_AUTHORIZE_PATH: &str = "/cli-login-remote";

/// A stuck request must leave time for retries within the login deadline.
const REQUEST_TIMEOUT_MS: u64 = 15_000;

/// Both responses are small JSON documents. Cap the body so a malicious or
/// broken endpoint cannot exhaust the CLI's memory.
const MAX_RESPONSE_BYTES: usize = 64 * 1024;

/// `flow_id` is interpolated into a request path; keep it to an opaque token.
const FLOW_ID_MAX_LEN: usize = 128;

/// Bound the account identifier before it reaches local credential storage.
const UID_MAX_LEN: usize = 64;

/// A server may slow the CLI down, but not park it for minutes at a time.
const MAX_POLL_INTERVAL_SEC: u64 = 60;

/// Bound the server-supplied error description kept in a message.
const MAX_ERROR_DETAIL_LENGTH: usize = 240;

/// Terminal metadata for server audit logs. Never used for authentication.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StepCliClientInfo {
    pub name: String,
    pub version: String,
    pub platform: String,
}

/// Audit metadata derived from the running process.
pub fn default_step_cli_client_info(version: impl Into<String>) -> StepCliClientInfo {
    StepCliClientInfo {
        name: "stepcode".to_owned(),
        version: version.into(),
        platform: format!("{}-{}", node_platform(), node_arch()),
    }
}

/// `process.platform` equivalent, for the audit metadata.
fn node_platform() -> &'static str {
    match std::env::consts::OS {
        "macos" => "darwin",
        "windows" => "win32",
        other => other,
    }
}

/// `process.arch` equivalent, for the audit metadata.
fn node_arch() -> &'static str {
    match std::env::consts::ARCH {
        "x86_64" => "x64",
        "aarch64" => "arm64",
        other => other,
    }
}

/// A started login flow: the URL to show the user plus the polling bounds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepCliLoginInit {
    /// Public identifier; safe to print.
    pub flow_id: String,
    /// Always `https:`, always on the developer-center origin.
    pub authorize_url: String,
    /// Server-requested poll interval in seconds, clamped to 60.
    pub poll_interval_sec: u64,
    /// Unix **seconds**; bounds the polling deadline. The wire value may carry
    /// a fraction; it is truncated to whole seconds here (see
    /// [`wait_for_step_cli_login`](super::login_polling::wait_for_step_cli_login),
    /// which converts it back to milliseconds).
    pub expires_at: u64,
}

/// Why a browser flow ended without a credential.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepCliLoginFailureReason {
    /// The user rejected the sign-in in the browser.
    Denied,
    /// The flow failed for an unclassified reason.
    Unknown,
}

/// The current state of a login flow.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StepCliLoginPoll {
    Pending,
    Ready {
        api_key: String,
        /// Bounded account identifier, when the server reported one.
        uid: Option<String>,
    },
    Failed {
        reason: StepCliLoginFailureReason,
    },
}

/// Why a CLI-login exchange failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepCliLoginErrorKind {
    /// Non-2xx HTTP response, or a non-zero `status` field in a 2xx envelope.
    Http,
    /// 2xx response whose shape or field values do not match the protocol.
    Protocol,
    /// The request never produced a response.
    Transport,
}

/// A failed CLI-login exchange, mirroring the TypeScript
/// `StepCliLoginRequestError`.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct StepCliLoginRequestError {
    pub kind: StepCliLoginErrorKind,
    /// HTTP status when the failure carried one.
    pub http_status: Option<u16>,
    pub message: String,
}

impl StepCliLoginRequestError {
    pub fn new(kind: StepCliLoginErrorKind, message: impl Into<String>) -> Self {
        StepCliLoginRequestError {
            kind,
            http_status: None,
            message: message.into(),
        }
    }

    pub fn with_http_status(
        kind: StepCliLoginErrorKind,
        message: impl Into<String>,
        http_status: u16,
    ) -> Self {
        StepCliLoginRequestError {
            kind,
            http_status: Some(http_status),
            message: message.into(),
        }
    }

    /// Retry only on the statuses the protocol defines as transient.
    fn is_retryable(&self) -> bool {
        if self.kind == StepCliLoginErrorKind::Protocol {
            return false;
        }
        match self.http_status {
            Some(status) => status == 408 || status == 429 || status >= 500,
            None => self.kind == StepCliLoginErrorKind::Transport,
        }
    }
}

/// Errors from the CLI-login HTTP layer.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum StepCliLoginError {
    /// The exchange produced a response that failed the protocol.
    #[error("{0}")]
    Request(StepCliLoginRequestError),
    /// The authorization endpoint itself is unusable: a malformed URL, a
    /// non-https scheme, embedded credentials, or an empty poll token.
    #[error("{0}")]
    Endpoint(String),
}

/// Retry only on the statuses the protocol defines as transient. Cancellation,
/// deadlines, browser rejections, and endpoint validation failures are never
/// retried — mirroring the TypeScript classifier, which only ever retried a
/// `StepCliLoginRequestError`.
pub fn is_retryable_step_cli_login_error(error: &StepLoginError) -> bool {
    match error {
        StepLoginError::CliLogin(StepCliLoginError::Request(request)) => request.is_retryable(),
        _ => false,
    }
}

/// Errors raised by the Step cloud login flow.
#[derive(Clone, Debug, thiserror::Error)]
pub enum StepLoginError {
    /// The caller's cancellation token fired.
    #[error("Step sign-in was cancelled")]
    Aborted,
    /// The login deadline elapsed (local timeout or server expiry).
    #[error("{0}")]
    TimedOut(String),
    /// The browser flow ended without a credential (denied or failed).
    #[error("{0}")]
    Rejected(String),
    /// A CLI-login exchange failed.
    #[error("{0}")]
    CliLogin(#[from] StepCliLoginError),
}

impl StepLoginError {
    /// Wrap a failed exchange.
    pub(crate) fn request(error: StepCliLoginRequestError) -> Self {
        StepLoginError::CliLogin(StepCliLoginError::Request(error))
    }

    /// Wrap an unusable endpoint.
    pub(crate) fn endpoint(message: impl Into<String>) -> Self {
        StepLoginError::CliLogin(StepCliLoginError::Endpoint(message.into()))
    }
}

/// HTTP method of a CLI-login exchange.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepCliLoginMethod {
    Get,
    Post,
}

/// One CLI-login exchange, decoupled from the concrete HTTP transport so the
/// flow can be driven by a fake in tests.
pub struct StepCliLoginRequest<'a> {
    pub method: StepCliLoginMethod,
    pub url: String,
    /// Sent in the `Authorization` header only; never in the URL.
    pub poll_token: &'a str,
    /// JSON body for POST exchanges.
    pub body: Option<String>,
}

/// The body outcome of an exchange.
pub enum StepCliLoginBody {
    /// The complete body, never larger than the 64 KiB response cap.
    Text(String),
    /// The body exceeded the size cap.
    TooLarge,
    /// The body read was interrupted mid-stream.
    Interrupted,
}

/// A transport response: the status plus the (capped) body.
pub struct StepCliLoginResponse {
    pub status: u16,
    /// Declared `content-length`, when the server sent one.
    pub content_length: Option<u64>,
    pub body: StepCliLoginBody,
}

/// The request never produced a response.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct StepCliLoginTransportError(String);

impl StepCliLoginTransportError {
    pub fn new(message: impl Into<String>) -> Self {
        StepCliLoginTransportError(message.into())
    }
}

/// HTTP transport for CLI-login exchanges. The production implementation uses
/// the shared reqwest client ([`ReqwestStepCliLoginTransport`]); tests inject a
/// fake.
#[async_trait]
pub trait StepCliLoginTransport: Send + Sync {
    async fn exchange(
        &self,
        request: StepCliLoginRequest<'_>,
    ) -> Result<StepCliLoginResponse, StepCliLoginTransportError>;
}

/// Production transport: one reqwest request per exchange over the supplied
/// client, with the body read under a hard size cap.
pub struct ReqwestStepCliLoginTransport {
    client: reqwest::Client,
}

impl ReqwestStepCliLoginTransport {
    pub fn new(client: reqwest::Client) -> Self {
        ReqwestStepCliLoginTransport { client }
    }

    /// A transport over a dedicated client that refuses redirects. The
    /// TypeScript flow sets `redirect: "error"` per request so a developer
    /// center can never bounce the poll token to another origin; reqwest
    /// exposes its redirect policy per client rather than per request, so the
    /// login flow owns a client with redirects disabled. This is what
    /// [`login_step_oauth`](super::login_step_oauth) uses.
    pub fn refusing_redirects() -> Result<Self, reqwest::Error> {
        let client = reqwest::Client::builder()
            .user_agent(crate::utils::user_agent::get_user_agent())
            .redirect(reqwest::redirect::Policy::none())
            .build()?;
        Ok(ReqwestStepCliLoginTransport::new(client))
    }
}

#[async_trait]
impl StepCliLoginTransport for ReqwestStepCliLoginTransport {
    async fn exchange(
        &self,
        request: StepCliLoginRequest<'_>,
    ) -> Result<StepCliLoginResponse, StepCliLoginTransportError> {
        let mut builder = match request.method {
            StepCliLoginMethod::Get => self.client.get(&request.url),
            StepCliLoginMethod::Post => self.client.post(&request.url),
        };
        builder = builder
            .timeout(Duration::from_millis(REQUEST_TIMEOUT_MS))
            .bearer_auth(request.poll_token.trim())
            .header(reqwest::header::ACCEPT, "application/json");
        if let Some(body) = &request.body {
            builder = builder
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(body.clone());
        }
        let response = builder
            .send()
            .await
            .map_err(|error| StepCliLoginTransportError::new(format!("request failed: {error}")))?;
        let status = response.status().as_u16();
        let content_length = response.content_length();
        let body = read_capped_body(response).await;
        Ok(StepCliLoginResponse {
            status,
            content_length,
            body,
        })
    }
}

/// Read at most [`MAX_RESPONSE_BYTES`], refusing anything larger.
async fn read_capped_body(response: reqwest::Response) -> StepCliLoginBody {
    use futures::StreamExt;

    let mut stream = response.bytes_stream();
    let mut buffer = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(chunk) => chunk,
            Err(_) => return StepCliLoginBody::Interrupted,
        };
        buffer.extend_from_slice(&chunk);
        if buffer.len() > MAX_RESPONSE_BYTES {
            return StepCliLoginBody::TooLarge;
        }
    }
    StepCliLoginBody::Text(String::from_utf8_lossy(&buffer).into_owned())
}

/// Start a login flow and obtain the URL to show the user.
pub async fn init_step_cli_login(
    input: InitStepCliLoginInput<'_>,
) -> Result<StepCliLoginInit, StepLoginError> {
    let base = parse_auth_base_url(input.auth_base_url)?;
    let signal = input.signal.cloned().unwrap_or_else(CancellationToken::new);
    let payload = request(ExchangeInput {
        phase: StepCliLoginPhase::Init,
        url: format!(
            "{}{}/init",
            base.origin().ascii_serialization(),
            STEP_CLI_LOGIN_PATH_PREFIX
        ),
        method: StepCliLoginMethod::Post,
        poll_token: input.poll_token,
        body: Some(
            serde_json::json!({
                "profile": input.profile,
                "origin": base.origin().ascii_serialization(),
                "client": input.client,
            })
            .to_string(),
        ),
        transport: input.transport,
        signal: &signal,
    })
    .await?;

    let flow_id = read_string(&payload, "flow_id").filter(|flow_id| is_valid_flow_id(flow_id));
    let Some(flow_id) = flow_id else {
        return Err(protocol_error(
            StepCliLoginPhase::Init,
            "response did not contain a usable flow_id",
        ));
    };
    // The authorize URL is always built from the developer-center origin, so a
    // server-supplied `authorize_url` can never redirect the browser.
    let authorize_url = authorize_url(&base, &flow_id)?;
    let poll_interval_sec =
        read_finite_number(&payload, "poll_interval_sec").filter(|value| *value >= 1.0);
    let Some(poll_interval_sec) = poll_interval_sec else {
        return Err(protocol_error(
            StepCliLoginPhase::Init,
            "response did not contain a poll_interval_sec of at least 1 second",
        ));
    };
    let expires_at = read_finite_number(&payload, "expires_at").filter(|value| *value > 0.0);
    let Some(expires_at) = expires_at else {
        return Err(protocol_error(
            StepCliLoginPhase::Init,
            "response did not contain an expires_at timestamp",
        ));
    };

    Ok(StepCliLoginInit {
        flow_id,
        authorize_url: authorize_url.to_string(),
        poll_interval_sec: (poll_interval_sec as u64).min(MAX_POLL_INTERVAL_SEC),
        expires_at: expires_at as u64,
    })
}

/// Inputs to [`init_step_cli_login`].
pub struct InitStepCliLoginInput<'a> {
    pub auth_base_url: &'a str,
    pub poll_token: &'a str,
    pub profile: &'a str,
    pub client: StepCliClientInfo,
    pub signal: Option<&'a CancellationToken>,
    pub transport: &'a dyn StepCliLoginTransport,
}

/// Read the current state of a flow.
///
/// A successful poll normally deletes the flow. Persist the returned key and
/// stop polling; the protocol does not promise at-most-once concurrent
/// delivery.
pub async fn poll_step_cli_login(
    input: PollStepCliLoginInput<'_>,
) -> Result<StepCliLoginPoll, StepLoginError> {
    let base = parse_auth_base_url(input.auth_base_url)?;
    if !is_valid_flow_id(input.flow_id) {
        return Err(protocol_error(
            StepCliLoginPhase::Poll,
            "flow id has an unexpected format",
        ));
    }
    let signal = input.signal.cloned().unwrap_or_else(CancellationToken::new);
    let payload = request(ExchangeInput {
        phase: StepCliLoginPhase::Poll,
        // The validated flow-id charset is `encodeURIComponent`-stable, so the
        // id is safe to interpolate into the request path.
        url: format!(
            "{}{}/poll/{}",
            base.origin().ascii_serialization(),
            STEP_CLI_LOGIN_PATH_PREFIX,
            input.flow_id
        ),
        method: StepCliLoginMethod::Get,
        poll_token: input.poll_token,
        body: None,
        transport: input.transport,
        signal: &signal,
    })
    .await?;

    match read_string(&payload, "state").as_deref() {
        Some("pending") => Ok(StepCliLoginPoll::Pending),
        Some("failed") => {
            let reason = match read_string(&payload, "reason").as_deref() {
                Some("denied") => StepCliLoginFailureReason::Denied,
                _ => StepCliLoginFailureReason::Unknown,
            };
            Ok(StepCliLoginPoll::Failed { reason })
        }
        Some("ready") => {
            let credential = payload
                .get("credential")
                .and_then(Value::as_object)
                .ok_or_else(|| {
                    protocol_error(
                        StepCliLoginPhase::Poll,
                        "ready response did not contain a credential object",
                    )
                })?;
            let api_key = read_string(credential, "api_key").ok_or_else(|| {
                protocol_error(
                    StepCliLoginPhase::Poll,
                    "ready response did not contain an api_key",
                )
            })?;
            let uid = read_string(&payload, "uid");
            let uid = uid.as_deref().and_then(bounded_uid);
            Ok(StepCliLoginPoll::Ready {
                api_key,
                uid: uid.map(str::to_owned),
            })
        }
        _ => Err(protocol_error(
            StepCliLoginPhase::Poll,
            "response contained an unknown state",
        )),
    }
}

/// Inputs to [`poll_step_cli_login`].
pub struct PollStepCliLoginInput<'a> {
    pub auth_base_url: &'a str,
    pub poll_token: &'a str,
    pub flow_id: &'a str,
    pub signal: Option<&'a CancellationToken>,
    pub transport: &'a dyn StepCliLoginTransport,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StepCliLoginPhase {
    Init,
    Poll,
}

impl std::fmt::Display for StepCliLoginPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            StepCliLoginPhase::Init => "init",
            StepCliLoginPhase::Poll => "poll",
        })
    }
}

struct ExchangeInput<'a> {
    phase: StepCliLoginPhase,
    url: String,
    method: StepCliLoginMethod,
    poll_token: &'a str,
    body: Option<String>,
    transport: &'a dyn StepCliLoginTransport,
    signal: &'a CancellationToken,
}

/// One protocol exchange, bounded by [`REQUEST_TIMEOUT_MS`] and the caller's
/// signal. Returns the parsed response envelope.
async fn request(input: ExchangeInput<'_>) -> Result<Map<String, Value>, StepLoginError> {
    let phase = input.phase;
    let response = with_step_cli_login_timeout(
        REQUEST_TIMEOUT_MS,
        Some(input.signal),
        || {
            StepLoginError::request(StepCliLoginRequestError::new(
                StepCliLoginErrorKind::Transport,
                "Step CLI login request timed out",
            ))
        },
        |_| async move {
            let poll_token = input.poll_token.trim();
            if poll_token.is_empty() {
                return Err(StepLoginError::endpoint(
                    "Step CLI login poll token must not be empty".to_owned(),
                ));
            }
            input
                .transport
                .exchange(StepCliLoginRequest {
                    method: input.method,
                    url: input.url.clone(),
                    poll_token,
                    body: input.body.clone(),
                })
                .await
                .map_err(|error| {
                    StepLoginError::request(StepCliLoginRequestError::new(
                        StepCliLoginErrorKind::Transport,
                        format!("Step CLI login {phase} request failed: {error}"),
                    ))
                })
        },
    )
    .await?;

    let ok = (200..300).contains(&response.status);
    let declared_too_large = response
        .content_length
        .is_some_and(|len| len > MAX_RESPONSE_BYTES as u64);
    // An error response's body is optional; HTTP status must still decide
    // retries, so a body problem only fails the exchange on a 2xx response.
    let mut text: Option<String> = None;
    let mut body_problem: Option<StepLoginError> = None;
    match &response.body {
        StepCliLoginBody::Text(body) => text = Some(body.clone()),
        StepCliLoginBody::TooLarge => {
            body_problem = Some(protocol_error(phase, "response exceeded the maximum size"));
        }
        StepCliLoginBody::Interrupted => {
            body_problem = Some(StepLoginError::request(StepCliLoginRequestError::new(
                StepCliLoginErrorKind::Transport,
                "Step CLI login response was interrupted",
            )));
        }
    }
    if declared_too_large {
        body_problem = Some(protocol_error(phase, "response exceeded the maximum size"));
    }
    if let Some(problem) = body_problem
        && ok
    {
        return Err(problem);
    }

    let parsed: Option<Value> = match text.as_deref().map(str::trim) {
        None | Some("") => None,
        Some(body) => match serde_json::from_str(body) {
            Ok(value) => Some(value),
            Err(_) if ok => return Err(protocol_error(phase, "response was not valid JSON")),
            Err(_) => None,
        },
    };
    if !ok {
        return Err(StepLoginError::request(
            StepCliLoginRequestError::with_http_status(
                StepCliLoginErrorKind::Http,
                format!(
                    "Step CLI login request failed (HTTP {}){}",
                    response.status,
                    envelope_detail(parsed.as_ref())
                ),
                response.status,
            ),
        ));
    }
    let Some(Value::Object(payload)) = parsed else {
        return Err(protocol_error(phase, "response was not a JSON object"));
    };
    // A non-zero `status` field in a 2xx envelope is an application-level
    // rejection; JS `!==` semantics compare numerically, so `0`/`0.0` pass.
    if payload
        .get("status")
        .is_some_and(|status| status.as_f64() != Some(0.0))
    {
        let detail = envelope_detail(Some(&Value::Object(payload.clone())));
        return Err(StepLoginError::request(
            StepCliLoginRequestError::with_http_status(
                StepCliLoginErrorKind::Http,
                format!("Step CLI login was rejected{detail}"),
                response.status,
            ),
        ));
    }
    Ok(payload)
}

fn parse_auth_base_url(value: &str) -> Result<url::Url, StepLoginError> {
    let url = url::Url::parse(value).map_err(|_| {
        StepLoginError::endpoint("Step OAuth authorization endpoint must be a valid URL".to_owned())
    })?;
    if url.scheme() != "https" {
        return Err(StepLoginError::endpoint(
            "Step CLI login authorization endpoint must use https".to_owned(),
        ));
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(StepLoginError::endpoint(
            "Step OAuth authorization endpoint must not contain credentials".to_owned(),
        ));
    }
    Ok(url)
}

/// Build the approval URL from the developer-center origin only.
fn authorize_url(base: &url::Url, flow_id: &str) -> Result<url::Url, StepLoginError> {
    let mut url =
        url::Url::parse(&format!("{}/", base.origin().ascii_serialization())).map_err(|_| {
            StepLoginError::endpoint(
                "Step OAuth authorization endpoint must be a valid URL".to_owned(),
            )
        })?;
    url.set_path(STEP_CLI_LOGIN_AUTHORIZE_PATH);
    url.query_pairs_mut().append_pair("flow_id", flow_id);
    Ok(url)
}

fn protocol_error(phase: StepCliLoginPhase, detail: &str) -> StepLoginError {
    StepLoginError::CliLogin(StepCliLoginError::Request(StepCliLoginRequestError::new(
        StepCliLoginErrorKind::Protocol,
        format!("Step CLI login {phase} {detail}"),
    )))
}

/// The envelope's `desc` field, stripped of control/format characters and
/// bounded, for inclusion in an error message.
fn envelope_detail(payload: Option<&Value>) -> String {
    let desc = payload
        .and_then(Value::as_object)
        .and_then(|payload| payload.get("desc"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|desc| !desc.is_empty());
    match desc {
        Some(desc) => {
            let cleaned: String = desc
                .chars()
                .filter(|char| !is_control_or_format(*char))
                .take(MAX_ERROR_DETAIL_LENGTH)
                .collect();
            format!(": {cleaned}")
        }
        None => String::new(),
    }
}

/// The Unicode `Cc` (control) and `Cf` (format) classes, matching the
/// TypeScript `[\p{Cc}\p{Cf}]` scrub.
fn is_control_or_format(char: char) -> bool {
    char.is_control()
        || matches!(char as u32,
			0x00AD | 0x0600..=0x0605 | 0x061C | 0x06DD | 0x070F | 0x08E2 | 0x180E
			| 0x200B..=0x200F | 0x202A..=0x202E | 0x2060..=0x2064 | 0x2066..=0x206F
			| 0xFEFF | 0xFFF9..=0xFFFB | 0x110BD | 0x110CD | 0x1D173..=0x1D17A
			| 0xE0001 | 0xE0020..=0xE007F)
}

/// `flow_id` is interpolated into a request path; keep it to an opaque token.
fn is_valid_flow_id(flow_id: &str) -> bool {
    !flow_id.is_empty()
        && flow_id.len() <= FLOW_ID_MAX_LEN
        && flow_id
            .chars()
            .all(|char| char.is_ascii_alphanumeric() || matches!(char, '.' | '_' | '-'))
}

/// Bound the account identifier before it reaches local credential storage.
fn bounded_uid(uid: &str) -> Option<&str> {
    let bounded = !uid.is_empty()
        && uid.len() <= UID_MAX_LEN
        && uid.chars().all(|char| {
            char.is_ascii_alphanumeric() || matches!(char, '_' | '.' | '@' | ':' | '-')
        });
    bounded.then_some(uid)
}

fn read_string(payload: &Map<String, Value>, key: &str) -> Option<String> {
    payload
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn read_finite_number(payload: &Map<String, Value>, key: &str) -> Option<f64> {
    payload
        .get(key)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
}

/// Bound both the per-request timeout and body reads, and clean up
/// timers/listeners on every exit. The caller's signal wins over the deadline:
/// a fired user token reports [`StepLoginError::Aborted`], an elapsed deadline
/// reports `timeout_error`.
pub async fn with_step_cli_login_timeout<T, F, Fut>(
    timeout_ms: u64,
    signal: Option<&CancellationToken>,
    timeout_error: impl FnOnce() -> StepLoginError,
    run: F,
) -> Result<T, StepLoginError>
where
    F: FnOnce(CancellationToken) -> Fut,
    Fut: std::future::Future<Output = Result<T, StepLoginError>> + Send,
{
    let deadline = CancellationToken::new();
    let timer = {
        let deadline = deadline.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(timeout_ms)).await;
            deadline.cancel();
        })
    };
    let bridge = signal.cloned().map(|signal| {
        let deadline = deadline.clone();
        tokio::spawn(async move {
            signal.cancelled().await;
            deadline.cancel();
        })
    });
    let outcome = race_with_abort(run(deadline.clone()), &deadline).await;
    if let Some(bridge) = bridge {
        bridge.abort();
    }
    timer.abort();
    match outcome {
        Ok(result) => result,
        Err(Aborted) => {
            if signal.is_some_and(|signal| signal.is_cancelled()) {
                Err(StepLoginError::Aborted)
            } else {
                Err(timeout_error())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step_provider::test_support::{FakeStep, FakeTransport};

    const AUTH_BASE_URL: &str = "https://auth.example.test";
    fn client() -> StepCliClientInfo {
        StepCliClientInfo {
            name: "stepcode".to_owned(),
            version: "1.2.3".to_owned(),
            platform: "darwin-arm64".to_owned(),
        }
    }

    fn init_body(overrides: Value) -> Value {
        let mut body = serde_json::json!({
            "status": 0,
            "desc": "",
            "flow_id": "flow-1",
            "poll_interval_sec": 2,
            "expires_at": crate::utils::now_ms() / 1000 + 600,
        });
        if let Value::Object(overrides) = overrides {
            for (key, value) in overrides {
                body[key] = value;
            }
        }
        body
    }

    fn cli_error(error: &StepLoginError) -> &StepCliLoginError {
        match error {
            StepLoginError::CliLogin(error) => error,
            other => panic!("expected a CLI login error, got {other:?}"),
        }
    }

    fn request_error(error: &StepLoginError) -> &StepCliLoginRequestError {
        match cli_error(error) {
            StepCliLoginError::Request(request) => request,
            other => panic!("expected a request error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn init_sends_poll_token_in_header_only_and_never_in_the_url() {
        let transport = FakeTransport::json(vec![(init_body(Value::Null), 200)]);
        let init = init_step_cli_login(InitStepCliLoginInput {
            auth_base_url: AUTH_BASE_URL,
            poll_token: "poll-token-value",
            profile: "step_plan",
            client: client(),
            signal: None,
            transport: &transport,
        })
        .await
        .expect("init succeeds");

        assert_eq!(init.flow_id, "flow-1");
        assert_eq!(
            init.authorize_url,
            "https://auth.example.test/cli-login-remote?flow_id=flow-1"
        );
        assert_eq!(init.poll_interval_sec, 2);

        let requests = transport.requests();
        assert_eq!(requests.len(), 1);
        let request = &requests[0];
        assert_eq!(request.method, StepCliLoginMethod::Post);
        assert_eq!(
            request.url,
            "https://auth.example.test/api/stdhttp/v1/cli-login/init"
        );
        assert!(!request.url.contains("poll-token-value"));
        assert_eq!(request.poll_token, "poll-token-value");
        let body: Value =
            serde_json::from_str(request.body.as_deref().expect("body")).expect("body json");
        assert_eq!(
            body,
            serde_json::json!({
                "profile": "step_plan",
                "origin": AUTH_BASE_URL,
                "client": { "name": "stepcode", "version": "1.2.3", "platform": "darwin-arm64" },
            })
        );
    }

    #[tokio::test]
    async fn init_constructs_the_authorization_url_from_the_origin() {
        let transport = FakeTransport::json(vec![(
            init_body(serde_json::json!({"authorize_url": "https://evil.example.test/phishing"})),
            200,
        )]);
        let init = init_step_cli_login(InitStepCliLoginInput {
            auth_base_url: AUTH_BASE_URL,
            poll_token: "token",
            profile: "step_plan",
            client: client(),
            signal: None,
            transport: &transport,
        })
        .await
        .expect("init succeeds");
        assert_eq!(
            init.authorize_url,
            "https://auth.example.test/cli-login-remote?flow_id=flow-1"
        );
    }

    #[tokio::test]
    async fn init_uses_the_overseas_developer_center_and_profile() {
        let auth_base_url = "https://platform.stepfun.ai";
        let transport = FakeTransport::json(vec![(init_body(Value::Null), 200)]);
        let init = init_step_cli_login(InitStepCliLoginInput {
            auth_base_url,
            poll_token: "oversea-poll-token",
            profile: "step_plan_oversea",
            client: client(),
            signal: None,
            transport: &transport,
        })
        .await
        .expect("init succeeds");

        let body: Value =
            serde_json::from_str(transport.requests()[0].body.as_deref().expect("body"))
                .expect("body json");
        assert_eq!(body["profile"], "step_plan_oversea");
        assert_eq!(body["origin"], auth_base_url);
        assert_eq!(
            init.authorize_url,
            "https://platform.stepfun.ai/cli-login-remote?flow_id=flow-1"
        );
    }

    #[tokio::test]
    async fn init_rejects_missing_flow_id_sub_second_interval_and_missing_expiry() {
        for overrides in [
            serde_json::json!({ "flow_id": "" }),
            serde_json::json!({ "flow_id": "has spaces" }),
            serde_json::json!({ "poll_interval_sec": 0.5 }),
            serde_json::json!({ "expires_at": 0 }),
        ] {
            let transport = FakeTransport::json(vec![(init_body(overrides), 200)]);
            let error = init_step_cli_login(InitStepCliLoginInput {
                auth_base_url: AUTH_BASE_URL,
                poll_token: "token",
                profile: "step_plan",
                client: client(),
                signal: None,
                transport: &transport,
            })
            .await
            .expect_err("init must fail");
            assert_eq!(
                request_error(&error).kind,
                StepCliLoginErrorKind::Protocol,
                "expected a protocol error for {error:?}"
            );
        }
    }

    #[tokio::test]
    async fn init_clamps_an_absurd_poll_interval() {
        let transport = FakeTransport::json(vec![(
            init_body(serde_json::json!({ "poll_interval_sec": 86_400 })),
            200,
        )]);
        let init = init_step_cli_login(InitStepCliLoginInput {
            auth_base_url: AUTH_BASE_URL,
            poll_token: "token",
            profile: "step_plan",
            client: client(),
            signal: None,
            transport: &transport,
        })
        .await
        .expect("init succeeds");
        assert_eq!(init.poll_interval_sec, 60);
    }

    #[tokio::test]
    async fn init_reports_the_http_status_before_the_envelope() {
        let transport = FakeTransport::json(vec![(
            serde_json::json!({ "status": 40301, "desc": "no plan" }),
            403,
        )]);
        let error = init_step_cli_login(InitStepCliLoginInput {
            auth_base_url: AUTH_BASE_URL,
            poll_token: "token",
            profile: "step_plan",
            client: client(),
            signal: None,
            transport: &transport,
        })
        .await
        .expect_err("init must fail");
        let StepCliLoginError::Request(request) = cli_error(&error) else {
            panic!("expected a request error");
        };
        assert_eq!(request.kind, StepCliLoginErrorKind::Http);
        assert_eq!(request.http_status, Some(403));
        assert!(
            request.message.contains("no plan"),
            "message: {}",
            request.message
        );
    }

    #[tokio::test]
    async fn init_refuses_a_response_larger_than_the_cap() {
        let transport = FakeTransport::new(vec![FakeStep {
            response: Ok(StepCliLoginResponse {
                status: 200,
                content_length: Some(70 * 1024),
                body: StepCliLoginBody::Text("x".repeat(70 * 1024)),
            }),
            hang: false,
        }]);
        let error = init_step_cli_login(InitStepCliLoginInput {
            auth_base_url: AUTH_BASE_URL,
            poll_token: "token",
            profile: "step_plan",
            client: client(),
            signal: None,
            transport: &transport,
        })
        .await
        .expect_err("init must fail");
        assert_eq!(request_error(&error).kind, StepCliLoginErrorKind::Protocol);
    }

    #[tokio::test]
    async fn init_never_sends_a_poll_token_over_plain_http() {
        let transport = FakeTransport::json(vec![(init_body(Value::Null), 200)]);
        let error = init_step_cli_login(InitStepCliLoginInput {
            auth_base_url: "http://auth.example.test",
            poll_token: "token",
            profile: "step_plan",
            client: client(),
            signal: None,
            transport: &transport,
        })
        .await
        .expect_err("init must fail");
        assert!(error.to_string().contains("https"), "message: {error}");
        assert!(transport.requests().is_empty());
    }

    #[tokio::test]
    async fn init_keeps_http_retry_semantics_when_the_error_body_is_oversized() {
        let transport = FakeTransport::new(vec![FakeStep {
            response: Ok(StepCliLoginResponse {
                status: 503,
                content_length: None,
                body: StepCliLoginBody::TooLarge,
            }),
            hang: false,
        }]);
        let error = init_step_cli_login(InitStepCliLoginInput {
            auth_base_url: AUTH_BASE_URL,
            poll_token: "token",
            profile: "step_plan",
            client: client(),
            signal: None,
            transport: &transport,
        })
        .await
        .expect_err("init must fail");
        let StepCliLoginError::Request(request) = cli_error(&error) else {
            panic!("expected a request error");
        };
        assert_eq!(request.kind, StepCliLoginErrorKind::Http);
        assert_eq!(request.http_status, Some(503));
        assert!(is_retryable_step_cli_login_error(&error));
    }

    // The TypeScript suite also bounds a request that never resolves with fake
    // timers. That port needs virtual time (tokio's `test-util` feature, absent
    // from this crate's dev-dependencies) to stay instant; the bound itself is
    // covered by `with_step_cli_login_timeout_reports_the_deadline_and_the_caller_signal`
    // and the transport-failure mapping by `init_bounds_a_stalled_response_body`.
    #[tokio::test]
    async fn init_bounds_a_stalled_response_body() {
        let transport = FakeTransport::new(vec![FakeStep {
            response: Ok(StepCliLoginResponse {
                status: 200,
                content_length: None,
                body: StepCliLoginBody::Interrupted,
            }),
            hang: false,
        }]);
        let error = init_step_cli_login(InitStepCliLoginInput {
            auth_base_url: AUTH_BASE_URL,
            poll_token: "token",
            profile: "step_plan",
            client: client(),
            signal: None,
            transport: &transport,
        })
        .await
        .expect_err("init must fail");
        assert_eq!(request_error(&error).kind, StepCliLoginErrorKind::Transport);
    }

    #[tokio::test]
    async fn poll_maps_pending_ready_and_failed_states() {
        let read = |body: Value| {
            let transport = FakeTransport::json(vec![(body, 200)]);
            async move {
                poll_step_cli_login(PollStepCliLoginInput {
                    auth_base_url: AUTH_BASE_URL,
                    poll_token: "token",
                    flow_id: "flow-1",
                    signal: None,
                    transport: &transport,
                })
                .await
            }
        };

        assert_eq!(
            read(serde_json::json!({ "status": 0, "state": "pending" }))
                .await
                .expect("pending"),
            StepCliLoginPoll::Pending
        );
        assert_eq!(
            read(serde_json::json!({
                "status": 0, "state": "ready", "uid": "u-1", "profile": "step_plan",
                "credential": { "api_key": "sk-step" },
            }))
            .await
            .expect("ready"),
            StepCliLoginPoll::Ready {
                api_key: "sk-step".to_owned(),
                uid: Some("u-1".to_owned()),
            }
        );
        assert_eq!(
            read(serde_json::json!({ "status": 0, "state": "failed", "reason": "denied" }))
                .await
                .expect("failed"),
            StepCliLoginPoll::Failed {
                reason: StepCliLoginFailureReason::Denied
            }
        );
        assert_eq!(
            read(serde_json::json!({ "status": 0, "state": "failed", "reason": "something-new" }))
                .await
                .expect("failed"),
            StepCliLoginPoll::Failed {
                reason: StepCliLoginFailureReason::Unknown
            }
        );
    }

    #[tokio::test]
    async fn poll_uses_the_origin_scoped_poll_url() {
        let transport = FakeTransport::json(vec![(
            serde_json::json!({ "status": 0, "state": "pending" }),
            200,
        )]);
        poll_step_cli_login(PollStepCliLoginInput {
            auth_base_url: AUTH_BASE_URL,
            poll_token: "token",
            flow_id: "flow-1",
            signal: None,
            transport: &transport,
        })
        .await
        .expect("poll succeeds");
        let requests = transport.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].method, StepCliLoginMethod::Get);
        assert_eq!(
            requests[0].url,
            "https://auth.example.test/api/stdhttp/v1/cli-login/poll/flow-1"
        );
        assert_eq!(requests[0].poll_token, "token");
        assert!(requests[0].body.is_none());
    }

    #[tokio::test]
    async fn poll_treats_a_ready_response_without_an_api_key_as_a_protocol_error() {
        let transport = FakeTransport::json(vec![(
            serde_json::json!({ "status": 0, "state": "ready", "credential": {} }),
            200,
        )]);
        let error = poll_step_cli_login(PollStepCliLoginInput {
            auth_base_url: AUTH_BASE_URL,
            poll_token: "token",
            flow_id: "flow-1",
            signal: None,
            transport: &transport,
        })
        .await
        .expect_err("poll must fail");
        assert_eq!(request_error(&error).kind, StepCliLoginErrorKind::Protocol);
    }

    #[test]
    fn classifies_retryable_and_fatal_failures() {
        let make = |kind: StepCliLoginErrorKind, http_status: Option<u16>| {
            let request = match http_status {
                Some(status) => StepCliLoginRequestError::with_http_status(kind, "boom", status),
                None => StepCliLoginRequestError::new(kind, "boom"),
            };
            StepLoginError::CliLogin(StepCliLoginError::Request(request))
        };

        assert!(is_retryable_step_cli_login_error(&make(
            StepCliLoginErrorKind::Http,
            Some(429)
        )));
        assert!(is_retryable_step_cli_login_error(&make(
            StepCliLoginErrorKind::Http,
            Some(503)
        )));
        assert!(is_retryable_step_cli_login_error(&make(
            StepCliLoginErrorKind::Http,
            Some(408)
        )));
        assert!(is_retryable_step_cli_login_error(&make(
            StepCliLoginErrorKind::Transport,
            None
        )));
        assert!(!is_retryable_step_cli_login_error(&make(
            StepCliLoginErrorKind::Http,
            Some(404)
        )));
        assert!(!is_retryable_step_cli_login_error(&make(
            StepCliLoginErrorKind::Protocol,
            None
        )));
        // Endpoint validation and every non-exchange failure are fatal.
        assert!(!is_retryable_step_cli_login_error(
            &StepLoginError::CliLogin(StepCliLoginError::Endpoint("bad endpoint".to_owned()))
        ));
        assert!(!is_retryable_step_cli_login_error(&StepLoginError::Aborted));
        assert!(!is_retryable_step_cli_login_error(
            &StepLoginError::TimedOut("deadline".to_owned())
        ));
        assert!(!is_retryable_step_cli_login_error(
            &StepLoginError::Rejected("denied".to_owned())
        ));
    }

    #[tokio::test]
    async fn poll_classifies_interrupted_response_bodies_as_retryable_transport_failures() {
        let transport = FakeTransport::new(vec![FakeStep {
            response: Ok(StepCliLoginResponse {
                status: 200,
                content_length: None,
                body: StepCliLoginBody::Interrupted,
            }),
            hang: false,
        }]);
        let error = poll_step_cli_login(PollStepCliLoginInput {
            auth_base_url: AUTH_BASE_URL,
            poll_token: "token",
            flow_id: "flow-1",
            signal: None,
            transport: &transport,
        })
        .await
        .expect_err("poll must fail");
        assert_eq!(request_error(&error).kind, StepCliLoginErrorKind::Transport);
        assert!(is_retryable_step_cli_login_error(&error));
    }

    #[test]
    fn default_client_info_uses_node_style_platform_and_arch() {
        let info = default_step_cli_client_info("9.9.9");
        assert_eq!(info.name, "stepcode");
        assert_eq!(info.version, "9.9.9");
        assert_eq!(
            info.platform,
            format!("{}-{}", node_platform(), node_arch())
        );
    }

    #[test]
    fn flow_id_and_uid_are_bounded() {
        assert!(is_valid_flow_id("flow-1"));
        assert!(is_valid_flow_id("a.b_c-9"));
        assert!(!is_valid_flow_id(""));
        assert!(!is_valid_flow_id("has spaces"));
        assert!(!is_valid_flow_id("path/segment"));
        assert!(!is_valid_flow_id(&"x".repeat(129)));

        assert_eq!(bounded_uid("u-1"), Some("u-1"));
        assert_eq!(
            bounded_uid("user.name@example.com:1"),
            Some("user.name@example.com:1")
        );
        assert_eq!(bounded_uid(""), None);
        assert_eq!(bounded_uid("has spaces"), None);
        assert_eq!(bounded_uid(&"x".repeat(65)), None);
    }

    #[test]
    fn envelope_detail_strips_control_characters_and_is_bounded() {
        let payload = serde_json::json!({ "desc": "  bad\u{200b}thing\u{7}  " });
        assert_eq!(envelope_detail(Some(&payload)), ": badthing");
        let long = serde_json::json!({ "desc": "x".repeat(300) });
        assert_eq!(
            envelope_detail(Some(&long)).chars().count(),
            2 + MAX_ERROR_DETAIL_LENGTH
        );
        assert_eq!(envelope_detail(Some(&serde_json::json!({}))), "");
        assert_eq!(envelope_detail(None), "");
    }

    #[tokio::test]
    async fn with_step_cli_login_timeout_reports_the_deadline_and_the_caller_signal() {
        let timeout = with_step_cli_login_timeout(
            30,
            None,
            || StepLoginError::TimedOut("deadline".to_owned()),
            |_| async {
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                Ok(())
            },
        )
        .await;
        assert_eq!(timeout.unwrap_err().to_string(), "deadline");

        let cancelled = CancellationToken::new();
        let signal = cancelled.clone();
        let aborted = tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            signal.cancel();
        });
        let error = with_step_cli_login_timeout(
            60_000,
            Some(&cancelled),
            || StepLoginError::TimedOut("deadline".to_owned()),
            |_| async {
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                Ok(())
            },
        )
        .await
        .expect_err("must abort");
        assert!(matches!(error, StepLoginError::Aborted));
        aborted.await.expect("canceller task");
    }
}
