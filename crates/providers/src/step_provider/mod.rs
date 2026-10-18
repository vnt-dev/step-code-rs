//! The Step provider: OAuth device-code (cloud-polled CLI) login, dynamic model
//! discovery, and provider registration.
//!
//! Rust port of `packages/providers/src/step-provider/index.ts`. The login
//! transport lives in [`login_client`] and the polling loop in
//! [`login_polling`]; both are re-exported here, mirroring the TypeScript
//! barrel.
//!
//! Wire notes:
//! - `{apiBaseUrl}/v1/models` tags each entry with a `model_type`. Step is a
//!   coding agent, so only chat-capable types are usable — large-language
//!   models and routers; other types (web search, TTS, ASR, realtime voice,
//!   text-to-image) are filtered out. Endpoints that do not yet return
//!   `model_type` are left unfiltered for backward compatibility.
//! - All Step models speak the OpenAI Chat Completions API
//!   (`{baseUrl}/chat/completions`), so both the baseline and the discovered
//!   models use it.
//! - Expiry timestamps: [`OAuthCredential::expires`] is Unix **milliseconds**
//!   (`u64`). The wire values handled here are Unix **seconds**
//!   (`StepCliLoginInit::expires_at`), converted at the boundary.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::api::openai_completions::openai_completions_api;
use crate::api::openai_responses::openai_responses_api;
use crate::auth::types::{
    ApiKeyAuth, ApiKeyCredential, AuthInteraction, AuthResult, Credential, ModelAuth, OAuthAuth,
    OAuthCredential, ProviderAuth,
};
use crate::auth::{AuthContext, AuthEvent, AuthInfoLink};
use crate::error::{ModelsError, ModelsErrorCode};
use crate::models::{
    ApiDispatch, CreateProviderOptions, FetchModelsFn, Provider, ProviderStreams,
    RefreshModelsContext, create_provider,
};
use crate::types::{
    Api, InputModality, Json, Model, ModelCost, ModelThinkingLevel, ProviderHeaders, ProviderId,
    ThinkingLevelMap,
};
use crate::utils::now_ms;

pub mod login_client;
pub mod login_polling;

pub use login_client::{
    ReqwestStepCliLoginTransport, STEP_CLI_LOGIN_PATH_PREFIX, StepCliClientInfo, StepCliLoginBody,
    StepCliLoginError, StepCliLoginErrorKind, StepCliLoginFailureReason, StepCliLoginInit,
    StepCliLoginMethod, StepCliLoginPoll, StepCliLoginRequestError, StepCliLoginResponse,
    StepCliLoginTransport, StepCliLoginTransportError, StepLoginError,
    default_step_cli_client_info, init_step_cli_login, is_retryable_step_cli_login_error,
    poll_step_cli_login, with_step_cli_login_timeout,
};
pub use login_polling::{
    MIN_STEP_POLL_INTERVAL_MS, StepCliLoginReady, WaitForStepCliLoginInput, wait_for_step_cli_login,
};

/// Provider id used by the default Step extension.
pub const STEP_PROVIDER_ID: &str = "step";

/// Environment variables understood by the Step provider.
pub const STEP_API_BASE_URL_ENV: &str = "STEP_BASE_URL";
pub const STEP_AUTH_BASE_URL_ENV: &str = "STEPCODE_DEVCENTER_AUTH_CN_URL";
pub const STEP_API_KEY_ENV: &str = "STEP_API_KEY";
pub const STEP_OAUTH_TIMEOUT_MS_ENV: &str = "STEP_OAUTH_TIMEOUT_MS";

/// Maximum time for a cloud sign-in, including init and polling.
pub const DEFAULT_STEP_CLI_LOGIN_TIMEOUT_MS: u64 = 10 * 60 * 1000;

/// Production defaults; every endpoint can be replaced through the options/env.
///
/// `apiBaseUrl` is the route prefix rather than a versioned operation URL: the
/// OpenAI adapter appends `/chat/completions` to `{baseUrl}/v1`.
pub const STEP_PROVIDER_DEFAULTS_API_BASE_URL: &str = "https://api.stepfun.com/step_plan";
pub const STEP_PROVIDER_DEFAULTS_AUTH_BASE_URL: &str = "https://platform.stepfun.com";

/// Marker used for credentials that contain a non-expiring Step API key.
pub const STEP_STATIC_REFRESH_TOKEN: &str = "step-static-credential";

/// Expiry stored for Step's non-expiring API keys. The TypeScript used
/// `Number.MAX_SAFE_INTEGER` (a millisecond timestamp); the Rust credential
/// field is `u64` milliseconds, so the same sentinel is kept verbatim
/// (≈ year 2255).
pub const STEP_NEVER_EXPIRES_MS: u64 = 9_007_199_254_740_991;

/// Wire dialect for every Step model. All Step profile URLs speak the OpenAI
/// Chat Completions API, so both the built-in baseline and the models
/// discovered from `{base}/v1/models` use it.
const STEP_MODEL_API: Api = Api::OpenaiCompletions;

/// No built-in baseline: the Step catalog is discovered entirely from
/// `{base}/v1/models` after login (see [`fetch_step_models`]). When discovery is
/// unavailable (offline / logged out / request failure) the list stays empty.
pub const STEP_MODELS: &[StepModelConfig] = &[];

/// `model_type` values the Step endpoint may report for chat-capable models.
const STEP_USABLE_MODEL_TYPES: [&str; 2] = ["大语言模型", "路由模型"];

/// Metadata `{base}/v1/models` does not always return; used as fallbacks.
const STEP_DEFAULT_CONTEXT_WINDOW: u64 = 256_000;
const STEP_DEFAULT_REASONING: bool = false;

/// Explicit user overrides win over the shared login profile's endpoints.
const API_BASE_URL_ENV_NAMES: [&str; 6] = [
    "STEP_PROVIDER_API_URL",
    "STEP_API_URL",
    STEP_API_BASE_URL_ENV,
    // Kept for compatibility with the endpoint override used by the legacy
    // StepCode. It may contain the full `/v1/messages` URL.
    "STEPFUN_MESSAGES_ENDPOINT",
    // Set by the shared Step login flow after reading the stored profile. It is
    // intentionally below explicit user overrides so manual endpoint settings win.
    "STEP_LOGIN_PROFILE_API_URL",
    "STEPCODE_STEP_PLAN_API_URL",
];
const AUTH_BASE_URL_ENV_NAMES: [&str; 4] = [
    "STEP_PROVIDER_AUTH_URL",
    "STEP_AUTH_URL",
    // Set by the shared Step login flow from the stored profile, so a
    // provider-side re-login opens the region's own developer center. Below the
    // explicit user overrides, like its API counterpart.
    "STEP_LOGIN_PROFILE_AUTH_URL",
    STEP_AUTH_BASE_URL_ENV,
];

/// Login profile metadata recorded by the backend.
const STEP_DEFAULT_LOGIN_PROFILE: &str = "step_plan";
const STEP_OVERSEA_LOGIN_PROFILE: &str = "step_plan_oversea";

/// Model catalog entry for the Step provider. Structurally identical to the
/// TypeScript `StepModelConfig`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StepModelConfig {
    pub id: String,
    pub name: String,
    /// Wire dialect; defaults to the shared Step dialect when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api: Option<Api>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    pub reasoning: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_level_map: Option<ThinkingLevelMap>,
    pub input: Vec<InputModality>,
    #[serde(default)]
    pub cost: ModelCost,
    pub context_window: u64,
    pub max_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compat: Option<crate::types::ApiCompat>,
}

/// Per-model capabilities parsed from `{base}/v1/models` (any field may be absent).
#[derive(Clone, Debug, Default, PartialEq)]
struct StepModelMeta {
    /// `max_input_tokens` -> context window.
    context_window: Option<u64>,
    /// `enable_reason` -> extended-thinking support.
    reasoning: Option<bool>,
    /// `enable_vision_input` -> image input support.
    vision: Option<bool>,
    /// `reasoning_effort_support_list` -> thinkingLevelMap (present on some endpoints).
    efforts: Option<Vec<String>>,
}

/// Options for the Step provider. Every field is optional; unset fields fall
/// back to the environment and then to the production defaults.
#[derive(Clone, Default)]
pub struct StepProviderOptions {
    pub provider_id: Option<String>,
    pub name: Option<String>,
    pub api_base_url: Option<String>,
    pub auth_base_url: Option<String>,
    pub api_key_env: Option<String>,
    pub timeout_ms: Option<u64>,
    /// Environment used for endpoint/timeout resolution. `None` reads the
    /// process environment (the TypeScript `options.env ?? process.env`).
    pub env: Option<HashMap<String, String>>,
    /// Baseline models. Empty is allowed: the Step catalog is discovered
    /// dynamically from `{base}/v1/models`; there is no built-in baseline.
    pub models: Vec<StepModelConfig>,
    /// HTTP client override (custom proxy, TLS, or instrumentation). The
    /// TypeScript equivalent of the injectable `fetch`.
    pub http_client: Option<reqwest::Client>,
    /// Login profile metadata recorded by the backend.
    pub login_profile: Option<String>,
    /// Terminal metadata for server audit logs. Never used for auth.
    pub client: Option<StepCliClientInfo>,
}

/// Provider settings resolved at registration time, after environment setup.
#[derive(Clone)]
pub struct ResolvedStepProviderOptions {
    pub provider_id: String,
    pub name: String,
    pub api_base_url: String,
    pub auth_base_url: String,
    pub api_key_env: String,
    pub timeout_ms: u64,
    pub env: HashMap<String, String>,
    pub models: Vec<StepModelConfig>,
    pub http_client: Option<reqwest::Client>,
    pub login_profile: String,
    pub client: StepCliClientInfo,
}

impl std::fmt::Debug for StepProviderOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StepProviderOptions")
            .field("provider_id", &self.provider_id)
            .field("timeout_ms", &self.timeout_ms)
            .field(
                "env_keys",
                &self.env.as_ref().map(|env| env.keys().collect::<Vec<_>>()),
            )
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for ResolvedStepProviderOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolvedStepProviderOptions")
            .field("provider_id", &self.provider_id)
            .field("timeout_ms", &self.timeout_ms)
            .field("env_keys", &self.env.keys().collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

/// Resolve provider settings at registration time, after environment setup.
pub fn resolve_step_provider_options(
    options: &StepProviderOptions,
) -> Result<ResolvedStepProviderOptions, ModelsError> {
    let env = options.env.clone().unwrap_or_else(process_env);
    let api_base_url = normalize_step_api_base_url(&validate_http_endpoint(
        &first_set(
            options.api_base_url.as_deref(),
            &API_BASE_URL_ENV_NAMES,
            &env,
        )
        .unwrap_or_else(|| STEP_PROVIDER_DEFAULTS_API_BASE_URL.to_owned()),
        "Step API base URL",
    )?);
    let auth_base_url = validate_http_endpoint(
        &first_set(
            options.auth_base_url.as_deref(),
            &AUTH_BASE_URL_ENV_NAMES,
            &env,
        )
        .unwrap_or_else(|| STEP_PROVIDER_DEFAULTS_AUTH_BASE_URL.to_owned()),
        "Step OAuth authorization URL",
    )?;
    let provider_id = non_empty(
        options.provider_id.as_deref().unwrap_or(STEP_PROVIDER_ID),
        "Step provider id",
    )?;
    let name = non_empty(
        options.name.as_deref().unwrap_or("Step"),
        "Step provider name",
    )?;
    let api_key_env = non_empty(
        options.api_key_env.as_deref().unwrap_or(STEP_API_KEY_ENV),
        "Step API key environment variable",
    )?;
    if !is_valid_env_name(&api_key_env) {
        return Err(ModelsError::new(
            ModelsErrorCode::Provider,
            format!("Invalid Step API key environment variable name: {api_key_env}"),
        ));
    }

    let timeout_ms = match options.timeout_ms {
        Some(timeout_ms) => timeout_ms,
        None => read_positive_integer(env.get(STEP_OAUTH_TIMEOUT_MS_ENV))
            .unwrap_or(DEFAULT_STEP_CLI_LOGIN_TIMEOUT_MS),
    };
    if timeout_ms == 0 {
        return Err(ModelsError::new(
            ModelsErrorCode::Provider,
            "Step login timeout must be finite and greater than zero",
        ));
    }

    let login_profile = match options
        .login_profile
        .as_deref()
        .map(str::trim)
        .filter(|profile| !profile.is_empty())
    {
        Some(profile) => profile.to_owned(),
        None => default_login_profile(&auth_base_url),
    };

    Ok(ResolvedStepProviderOptions {
        provider_id,
        name,
        api_base_url,
        auth_base_url,
        api_key_env,
        timeout_ms,
        env,
        models: options.models.clone(),
        http_client: options.http_client.clone(),
        login_profile,
        client: options
            .client
            .clone()
            .unwrap_or_else(|| default_step_cli_client_info("unknown")),
    })
}

/// A model entry may omit its dialect. Default it to the shared Step wire
/// dialect (OpenAI Chat Completions) so a bare custom entry streams correctly.
pub fn normalize_step_model_config(mut config: StepModelConfig) -> StepModelConfig {
    config.api = config.api.or(Some(STEP_MODEL_API));
    config
}

/// Every Step model uses the active profile's OpenAI endpoint and dialect.
/// Restore both so a stale models.json overlay (e.g. an old proxy host or the
/// legacy Anthropic dialect) cannot misroute a Step model.
pub fn normalize_step_model(model: &Model, openai_base_url: &str) -> Model {
    Model {
        api: STEP_MODEL_API,
        base_url: openai_base_url.to_owned(),
        ..model.clone()
    }
}

/// Build a Step model entry, using discovered metadata where present.
fn step_model_config(
    id: &str,
    name: &str,
    base_url: Option<&str>,
    meta: Option<&StepModelMeta>,
) -> StepModelConfig {
    let context_window = meta
        .and_then(|meta| meta.context_window)
        .unwrap_or(STEP_DEFAULT_CONTEXT_WINDOW);
    // A non-empty effort list implies the model reasons.
    let efforts = meta
        .and_then(|meta| meta.efforts.as_deref())
        .filter(|efforts| !efforts.is_empty());
    let has_efforts = efforts.is_some();
    let reasoning = meta
        .and_then(|meta| meta.reasoning)
        .unwrap_or(if has_efforts {
            true
        } else {
            STEP_DEFAULT_REASONING
        });
    let vision = meta.and_then(|meta| meta.vision).unwrap_or(false);
    StepModelConfig {
        id: id.to_owned(),
        name: name.to_owned(),
        api: Some(STEP_MODEL_API),
        base_url: base_url.map(str::to_owned),
        reasoning,
        thinking_level_map: efforts.map(step_thinking_level_map),
        // Default to text-only unless the endpoint reports otherwise.
        input: if vision {
            vec![InputModality::Text, InputModality::Image]
        } else {
            vec![InputModality::Text]
        },
        // The endpoint reports no pricing.
        cost: ModelCost::default(),
        context_window,
        // The endpoint reports no max-output; default it to the input window.
        max_tokens: context_window,
        headers: None,
        compat: None,
    }
}

/// Convert a Step model config into a runtime model. Step models always speak
/// the OpenAI dialect against the active profile's base, so a stale custom
/// `api`/`base_url` cannot misroute one.
fn step_model_to_model(
    config: &StepModelConfig,
    provider_id: &str,
    openai_base_url: &str,
) -> Model {
    Model {
        id: config.id.clone(),
        name: config.name.clone(),
        api: STEP_MODEL_API,
        provider: ProviderId::new(provider_id),
        base_url: openai_base_url.to_owned(),
        reasoning: config.reasoning,
        thinking_level_map: config.thinking_level_map.clone(),
        input: if config.input.is_empty() {
            vec![InputModality::Text]
        } else {
            config.input.clone()
        },
        cost: config.cost.clone(),
        context_window: config.context_window,
        max_tokens: config.max_tokens,
        sampling_params: None,
        headers: config.headers.clone(),
        compat: config.compat.clone(),
    }
}

/// Cloud login has one path: init, open the supplied URL, poll, store the
/// existing API key.
///
/// The login transport refuses redirects (see
/// [`ReqwestStepCliLoginTransport::refusing_redirects`]), so the poll token can
/// never be bounced off the developer-center origin. An injected
/// `http_client` still applies to model discovery.
pub async fn login_step_oauth(
    interaction: &dyn AuthInteraction,
    options: &StepProviderOptions,
) -> Result<OAuthCredential, ModelsError> {
    let resolved = resolve_step_provider_options(options)?;
    login_step_oauth_resolved(interaction, &resolved).await
}

/// Cloud login against already-resolved settings; the TypeScript
/// `loginStepOAuth` accepts either the raw options or the resolved form.
pub async fn login_step_oauth_resolved(
    interaction: &dyn AuthInteraction,
    resolved: &ResolvedStepProviderOptions,
) -> Result<OAuthCredential, ModelsError> {
    let transport = ReqwestStepCliLoginTransport::refusing_redirects().map_err(|error| {
        ModelsError::with_source(
            ModelsErrorCode::Auth,
            "Could not construct login transport",
            error,
        )
    })?;
    run_step_login(interaction, resolved, &transport)
        .await
        .map_err(Into::into)
}

/// Run the cloud login against an already-resolved configuration and an
/// injected transport. [`login_step_oauth`] is the public entry point; this
/// variant keeps the flow testable without a network.
pub(crate) async fn run_step_login(
    interaction: &dyn AuthInteraction,
    resolved: &ResolvedStepProviderOptions,
    transport: &dyn login_client::StepCliLoginTransport,
) -> Result<OAuthCredential, StepLoginError> {
    let signal = interaction.signal();
    login_client::with_step_cli_login_timeout(
		resolved.timeout_ms,
		Some(&signal),
		|| StepLoginError::TimedOut("Step sign-in timed out. Run `step login` again.".to_owned()),
		|token| async move {
			let poll_token = random_poll_token();
			let init = init_step_cli_login(login_client::InitStepCliLoginInput {
				auth_base_url: &resolved.auth_base_url,
				poll_token: &poll_token,
				profile: &resolved.login_profile,
				client: resolved.client.clone(),
				signal: Some(&token),
				transport,
			})
			.await?;
			if token.is_cancelled() {
				return Err(StepLoginError::Aborted);
			}
			interaction.notify(AuthEvent::Info {
				message: "Complete the sign-in in your browser; this terminal only makes outbound HTTPS requests and never listens on a port.".to_owned(),
				links: vec![AuthInfoLink {
					url: init.authorize_url.clone(),
					label: Some("Open sign-in page".to_owned()),
				}],
			});
			interaction.notify(AuthEvent::AuthUrl {
				url: init.authorize_url.clone(),
				instructions: Some("Open this URL in any browser and approve the sign-in.".to_owned()),
			});
			interaction.notify(AuthEvent::DeviceCode {
				user_code: init.flow_id.clone(),
				verification_uri: init.authorize_url.clone(),
				interval_seconds: Some(init.poll_interval_sec),
				expires_in_seconds: Some(init.expires_at.saturating_sub(now_ms() / 1000)),
			});
			interaction.notify(AuthEvent::Progress {
				message: "Waiting for browser confirmation...".to_owned(),
			});
			let ready = wait_for_step_cli_login(login_polling::WaitForStepCliLoginInput {
				init,
				auth_base_url: &resolved.auth_base_url,
				poll_token: &poll_token,
				signal: Some(&token),
				transport,
			})
			.await?;
			if token.is_cancelled() {
				return Err(StepLoginError::Aborted);
			}
			// Persist the selected profile even if polling omits it or echoes
			// a different value: search billing follows this login's route.
			let mut extra = HashMap::from([(
				"profile".to_owned(), Json::String(resolved.login_profile.clone()),
			)]);
			if let Some(uid) = ready.uid {
				extra.insert("uid".to_owned(), Json::String(uid));
			}
			Ok(OAuthCredential {
				refresh: STEP_STATIC_REFRESH_TOKEN.to_owned(),
				access: ready.api_key,
				expires: STEP_NEVER_EXPIRES_MS,
				extra: Some(extra),
			})
		},
	)
	.await
}

/// The provider interface calls this for expired credentials; current Step keys
/// do not expire locally, so a well-formed credential is simply re-stamped.
pub async fn refresh_step_oauth(
    credential: &OAuthCredential,
) -> Result<OAuthCredential, ModelsError> {
    if credential.refresh != STEP_STATIC_REFRESH_TOKEN {
        return Err(ModelsError::new(
            ModelsErrorCode::Oauth,
            "This Step credential cannot be refreshed. Run `step login` again.",
        ));
    }
    let mut refreshed = credential.clone();
    refreshed.expires = STEP_NEVER_EXPIRES_MS;
    Ok(refreshed)
}

/// The bearer token for Step API requests, taken from the stored credential's
/// access token.
pub fn get_step_oauth_api_key(credential: &OAuthCredential) -> Result<String, ModelsError> {
    let access = credential.access.trim();
    if access.is_empty() {
        return Err(ModelsError::new(
            ModelsErrorCode::Auth,
            "Step OAuth credentials do not contain an access token",
        ));
    }
    Ok(access.to_owned())
}

/// OAuth auth for the Step provider: name, login, refresh, and request-auth
/// derivation.
pub struct StepOAuth {
    resolved: ResolvedStepProviderOptions,
}

impl StepOAuth {
    pub fn new(resolved: ResolvedStepProviderOptions) -> Self {
        StepOAuth { resolved }
    }
}

#[async_trait]
impl OAuthAuth for StepOAuth {
    fn name(&self) -> &str {
        "Step Plan"
    }

    fn is_subscription(&self) -> bool {
        true
    }

    async fn login(
        &self,
        interaction: &dyn AuthInteraction,
    ) -> Result<OAuthCredential, Box<dyn std::error::Error + Send + Sync>> {
        let transport = ReqwestStepCliLoginTransport::refusing_redirects().map_err(|error| {
            ModelsError::with_source(
                ModelsErrorCode::Auth,
                "Could not construct login transport",
                error,
            )
        })?;
        Ok(run_step_login(interaction, &self.resolved, &transport).await?)
    }

    async fn refresh(
        &self,
        credential: &OAuthCredential,
        _token: &CancellationToken,
    ) -> Result<OAuthCredential, Box<dyn std::error::Error + Send + Sync>> {
        Ok(refresh_step_oauth(credential).await?)
    }

    async fn to_auth(
        &self,
        credential: &OAuthCredential,
    ) -> Result<ModelAuth, Box<dyn std::error::Error + Send + Sync>> {
        Ok(ModelAuth {
            api_key: Some(get_step_oauth_api_key(credential)?),
            headers: None,
            base_url: None,
        })
    }
}

/// Ambient api-key auth for the Step provider: a stored key wins, otherwise the
/// configured environment variable is consulted. Never interactive — Step
/// signs in through OAuth.
pub struct StepApiKeyAuth {
    name: String,
    env_var: String,
    /// Provider-scoped values resolved at registration time; these take
    /// precedence over the process environment.
    env: HashMap<String, String>,
}

impl StepApiKeyAuth {
    pub fn new(resolved: &ResolvedStepProviderOptions) -> Self {
        StepApiKeyAuth {
            name: "Step API key".to_owned(),
            env_var: resolved.api_key_env.clone(),
            env: resolved.env.clone(),
        }
    }
}

#[async_trait]
impl ApiKeyAuth for StepApiKeyAuth {
    fn name(&self) -> &str {
        &self.name
    }

    async fn resolve(
        &self,
        ctx: &dyn AuthContext,
        credential: Option<&ApiKeyCredential>,
        token: &CancellationToken,
    ) -> Result<Option<AuthResult>, Box<dyn std::error::Error + Send + Sync>> {
        if token.is_cancelled() {
            return Err("aborted".into());
        }
        if let Some(key) = credential
            .and_then(|credential| credential.key.as_deref())
            .map(str::trim)
            .filter(|key| !key.is_empty())
        {
            return Ok(Some(AuthResult {
                auth: ModelAuth {
                    api_key: Some(key.to_owned()),
                    ..Default::default()
                },
                env: credential.and_then(|credential| credential.env.clone()),
                source: Some("stored credential".to_owned()),
            }));
        }
        if let Some(value) = self
            .env
            .get(&self.env_var)
            .map(String::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            return Ok(Some(AuthResult {
                auth: ModelAuth {
                    api_key: Some(value.to_owned()),
                    ..Default::default()
                },
                env: None,
                source: Some(self.env_var.clone()),
            }));
        }
        let value = ctx.env(&self.env_var).await;
        if token.is_cancelled() {
            return Err("aborted".into());
        }
        Ok(value.map(|value| AuthResult {
            auth: ModelAuth {
                api_key: Some(value),
                ..Default::default()
            },
            env: None,
            source: Some(self.env_var.clone()),
        }))
    }
}

/// Discover the chat-capable models the current credential can use from
/// `{base}/v1/models`, filtered by the endpoint's `model_type` (see
/// `STEP_USABLE_MODEL_TYPES`). Per-model capabilities are taken from the
/// response where present: `max_input_tokens` -> context window, `enable_reason`
/// -> thinking support, `enable_vision_input` -> image input; missing fields use
/// the metadata defaults.
///
/// When discovery is unavailable — offline, unauthenticated, on any
/// transport/parse failure, or when the endpoint returns no usable models — it
/// returns the configured baseline (empty by default), so the Step catalog is
/// simply empty until a successful fetch.
pub async fn fetch_step_models(
    context: &RefreshModelsContext,
    resolved: &ResolvedStepProviderOptions,
) -> Result<Vec<Model>, ModelsError> {
    let openai_base_url = step_open_ai_base_url(&resolved.api_base_url);
    let baseline = || {
        resolved
            .models
            .iter()
            .map(|config| step_model_to_model(config, &resolved.provider_id, &openai_base_url))
            .collect::<Vec<_>>()
    };
    if !context.allow_network || context.signal.is_cancelled() {
        return Ok(baseline());
    }
    let Some(token) = read_step_bearer_token(context.credential.as_ref()) else {
        return Ok(baseline());
    };
    let client = match &resolved.http_client {
        Some(client) => client.clone(),
        None => crate::api::http::default_http_client().map_err(|error| {
            ModelsError::with_source(
                ModelsErrorCode::Provider,
                "Could not construct HTTP client",
                error,
            )
        })?,
    };
    let response = match client
        .get(format!("{openai_base_url}/models"))
        .bearer_auth(&token)
        .header(reqwest::header::ACCEPT, "application/json")
        .send()
        .await
    {
        Ok(response) => response,
        Err(_) => return Ok(baseline()),
    };
    if !response.status().is_success() {
        return Ok(baseline());
    }
    let payload = match response.json::<Json>().await {
        Ok(payload) => payload,
        Err(_) => return Ok(baseline()),
    };
    let configs = parse_step_models_response(&payload, &openai_base_url);
    if configs.is_empty() {
        return Ok(baseline());
    }
    Ok(configs
        .iter()
        .map(|config| step_model_to_model(config, &resolved.provider_id, &openai_base_url))
        .collect())
}

/// Parse a `{base}/v1/models` payload into Step model configs. Pure: no I/O, so
/// the filtering and metadata rules are directly testable.
pub fn parse_step_models_response(payload: &Json, base_url: &str) -> Vec<StepModelConfig> {
    let Some(data) = payload.get("data").and_then(Value::as_array) else {
        return Vec::new();
    };
    let entries: Vec<(String, Option<String>, StepModelMeta)> = data
        .iter()
        .filter_map(|entry| {
            let id = read_json_string(entry, "id")?;
            Some((
                id,
                read_json_string(entry, "model_type"),
                StepModelMeta {
                    context_window: read_positive_number(entry, "max_input_tokens"),
                    reasoning: read_json_bool(entry, "enable_reason"),
                    vision: read_json_bool(entry, "enable_vision_input"),
                    efforts: read_string_array(entry, "reasoning_effort_support_list"),
                },
            ))
        })
        .collect();

    // Keep only large-language models when the endpoint tags model types; if no
    // entry carries a type (older/other endpoints), leave the list unfiltered.
    let typed = entries
        .iter()
        .any(|(_, model_type, _)| model_type.is_some());
    let mut seen = HashSet::new();
    let mut models = Vec::new();
    for (id, model_type, meta) in entries {
        if typed && !model_type.as_deref().is_some_and(is_usable_model_type) {
            continue;
        }
        if !seen.insert(id.clone()) {
            continue;
        }
        models.push(step_model_config(&id, &id, Some(base_url), Some(&meta)));
    }
    models
}

/// Inputs to [`fetch_step_model_efforts`].
pub struct FetchStepModelEffortsInput<'a> {
    /// OpenAI-style base (`.../v1`); the request targets `{baseUrl}/models/{modelId}`.
    pub base_url: &'a str,
    pub model_id: &'a str,
    pub api_key: &'a str,
    pub http_client: Option<reqwest::Client>,
    pub signal: Option<&'a CancellationToken>,
}

/// Fetch a single model's supported reasoning efforts from
/// `{baseUrl}/models/{modelId}` (`reasoning_effort_support_list`). Returns
/// `None` on any transport/parse failure or when the field is absent.
pub async fn fetch_step_model_efforts(
    input: FetchStepModelEffortsInput<'_>,
) -> Option<Vec<String>> {
    let api_key = input.api_key.trim();
    if api_key.is_empty() {
        return None;
    }
    let base = input.base_url.trim_end_matches('/');
    let client = match input.http_client {
        Some(client) => client,
        None => crate::api::http::default_http_client().ok()?,
    };
    let url = format!("{base}/models/{}", encode_path_segment(input.model_id));
    let token = input.signal.cloned().unwrap_or_else(CancellationToken::new);
    let response = match crate::utils::race_with_abort(
        client
            .get(&url)
            .bearer_auth(api_key)
            .header(reqwest::header::ACCEPT, "application/json")
            .send(),
        &token,
    )
    .await
    {
        Ok(Ok(response)) => response,
        _ => return None,
    };
    if !response.status().is_success() {
        return None;
    }
    let payload = response.json::<Json>().await.ok()?;
    let efforts: Vec<String> = payload
        .get("reasoning_effort_support_list")?
        .as_array()?
        .iter()
        .filter_map(Value::as_str)
        .map(str::trim)
        .filter(|effort| !effort.is_empty())
        .map(str::to_owned)
        .collect();
    (!efforts.is_empty()).then_some(efforts)
}

/// Reasoning-effort levels the Step endpoint may report, in ascending order,
/// plus the explicit "off" option.
const STEP_THINKING_LEVELS: [ModelThinkingLevel; 7] = [
    ModelThinkingLevel::Off,
    ModelThinkingLevel::Minimal,
    ModelThinkingLevel::Low,
    ModelThinkingLevel::Medium,
    ModelThinkingLevel::High,
    ModelThinkingLevel::Xhigh,
    ModelThinkingLevel::Max,
];

/// The wire name of a thinking level, matching the TypeScript level strings.
fn thinking_level_key(level: ModelThinkingLevel) -> &'static str {
    match level {
        ModelThinkingLevel::Off => "off",
        ModelThinkingLevel::Minimal => "minimal",
        ModelThinkingLevel::Low => "low",
        ModelThinkingLevel::Medium => "medium",
        ModelThinkingLevel::High => "high",
        ModelThinkingLevel::Xhigh => "xhigh",
        ModelThinkingLevel::Max => "max",
    }
}

/// Convert a `reasoning_effort_support_list` into a `thinkingLevelMap`: listed
/// levels map to their own effort value; every other level (including `off` when
/// the list omits it) is marked `null` (unsupported), so the `/effort` picker
/// shows exactly what the endpoint reports.
pub fn step_thinking_level_map(efforts: &[String]) -> ThinkingLevelMap {
    let supported: HashSet<String> = efforts
        .iter()
        .map(|effort| effort.trim().to_ascii_lowercase())
        .filter(|effort| !effort.is_empty())
        .collect();
    STEP_THINKING_LEVELS
        .iter()
        .map(|level| {
            let key = thinking_level_key(*level);
            (*level, supported.contains(key).then(|| key.to_owned()))
        })
        .collect()
}

/// Build the Step provider: resolved base urls, default headers, both auth
/// methods, the static baseline, api dispatch, and dynamic model discovery.
pub fn step_provider(options: &StepProviderOptions) -> Result<Arc<Provider>, ModelsError> {
    let resolved = resolve_step_provider_options(options)?;
    Ok(step_provider_from_resolved(&resolved))
}

/// Build the Step provider from already-resolved settings.
pub fn step_provider_from_resolved(resolved: &ResolvedStepProviderOptions) -> Arc<Provider> {
    let openai_base_url = step_open_ai_base_url(&resolved.api_base_url);
    let mut api: HashMap<String, Arc<dyn ProviderStreams>> = HashMap::new();
    api.insert(
        Api::OPENAI_COMPLETIONS.to_owned(),
        Arc::new(openai_completions_api()),
    );
    api.insert(
        Api::OPENAI_RESPONSES.to_owned(),
        Arc::new(openai_responses_api()),
    );
    let resolved_for_fetch = resolved.clone();
    let fetch_models: FetchModelsFn = Arc::new(move |context| {
        let resolved = resolved_for_fetch.clone();
        Box::pin(async move { fetch_step_models(&context, &resolved).await })
    });
    create_provider(CreateProviderOptions {
        id: ProviderId::new(&resolved.provider_id),
        name: Some(resolved.name.clone()),
        base_url: Some(openai_base_url.clone()),
        // The Step config declares no default headers: the shared HTTP client
        // supplies the user agent and the adapters set per-request headers.
        headers: step_default_headers(),
        auth: ProviderAuth {
            api_key: Some(Arc::new(StepApiKeyAuth::new(resolved))),
            oauth: Some(Arc::new(StepOAuth::new(resolved.clone()))),
        },
        models: resolved
            .models
            .iter()
            .map(|config| step_model_to_model(config, &resolved.provider_id, &openai_base_url))
            .collect(),
        fetch_models: Some(fetch_models),
        filter_models: None,
        api: ApiDispatch::ByApi(api),
    })
}

/// Provider-level default headers. Step needs none of its own; the map stays
/// empty so a caller-supplied header is never shadowed.
fn step_default_headers() -> Option<ProviderHeaders> {
    Some(HashMap::new())
}

// ---- URL helpers -----------------------------------------------------------

/// Reduce a Step endpoint to its route prefix. Step's configuration (and older
/// models.json files) accepted the versioned operation URL (`/v1/messages`),
/// the `/messages` operation, or the `/v1` parent; strip any of those trailing
/// segments so callers can append the correct suffix for the active dialect
/// (`step_open_ai_base_url` re-appends `/v1`).
pub fn normalize_step_api_base_url(value: &str) -> String {
    let Ok(mut url) = url::Url::parse(value) else {
        // The provider resolver validates its own endpoint. This exported helper
        // is also used while repairing an existing models.json, where malformed
        // custom entries must remain intact for normal validation diagnostics.
        return value.to_owned();
    };
    let mut pathname = trim_trailing_slashes(url.path()).to_owned();
    // Be tolerant of files produced by an intermediate release that already
    // contained one duplicated suffix (for example `/v1/v1/messages`). Repeating
    // the operation keeps the repair idempotent for any number of stale
    // version/operation segments.
    while !pathname.is_empty() {
        let lower_pathname = pathname.to_ascii_lowercase();
        let Some(suffix) = ["/v1/messages", "/messages", "/v1"]
            .into_iter()
            .find(|candidate| lower_pathname.ends_with(candidate))
        else {
            break;
        };
        pathname = trim_trailing_slashes(&pathname[..pathname.len() - suffix.len()]).to_owned();
    }
    url.set_path(if pathname.is_empty() { "/" } else { &pathname });
    trim_trailing_slashes(url.as_ref()).to_owned()
}

/// OpenAI-style base (`.../v1`) for the active profile. `api_base_url` is
/// already normalized (trailing `/v1` stripped), so appending `/v1` yields the
/// OpenAI base for every profile:
///   step_plan   -> <https://api.stepfun.com/step_plan/v1>
///   platform_cn -> <https://api.stepfun.com/v1>
/// The OpenAI adapter appends `/chat/completions`; discovery uses `/models`.
pub fn step_open_ai_base_url(api_base_url: &str) -> String {
    format!("{}/v1", normalize_step_api_base_url(api_base_url))
}

/// Base for per-model detail lookups (`{base}/models/{id}`). The detail endpoint
/// lives at the domain-root `/v1` (e.g. `https://api.stepfun.com/v1`), even for
/// the `step_plan` profile whose chat/list base is `.../step_plan/v1` — the
/// subpath variant 404s. Falls back to the input if it is not a valid URL.
pub fn step_models_detail_base_url(base_url: &str) -> String {
    match url::Url::parse(base_url) {
        Ok(url) => format!("{}/v1", url.origin().ascii_serialization()),
        Err(_) => base_url.to_owned(),
    }
}

fn trim_trailing_slashes(value: &str) -> &str {
    value.trim_end_matches('/')
}

fn validate_http_endpoint(value: &str, label: &str) -> Result<String, ModelsError> {
    let url = url::Url::parse(value).map_err(|_| {
        ModelsError::new(
            ModelsErrorCode::Provider,
            format!("{label} must be a valid URL"),
        )
    })?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(ModelsError::new(
            ModelsErrorCode::Provider,
            format!("{label} must use http or https"),
        ));
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(ModelsError::new(
            ModelsErrorCode::Provider,
            format!("{label} must not contain credentials"),
        ));
    }
    Ok(trim_trailing_slashes(url.as_ref()).to_owned())
}

fn default_login_profile(auth_base_url: &str) -> String {
    let overseas = url::Url::parse(auth_base_url)
        .ok()
        .and_then(|url| {
            url.host_str()
                .map(|host| host.to_ascii_lowercase().ends_with(".ai"))
        })
        .unwrap_or(false);
    if overseas {
        STEP_OVERSEA_LOGIN_PROFILE.to_owned()
    } else {
        STEP_DEFAULT_LOGIN_PROFILE.to_owned()
    }
}

// ---- payload readers -------------------------------------------------------

fn read_json_string(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn read_positive_number(value: &Value, key: &str) -> Option<u64> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .filter(|number| number.is_finite() && *number > 0.0)
        .map(|number| number as u64)
}

fn read_json_bool(value: &Value, key: &str) -> Option<bool> {
    value.get(key).and_then(Value::as_bool)
}

fn read_string_array(value: &Value, key: &str) -> Option<Vec<String>> {
    let items: Vec<String> = value
        .get(key)?
        .as_array()?
        .iter()
        .filter_map(Value::as_str)
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(str::to_owned)
        .collect();
    (!items.is_empty()).then_some(items)
}

fn is_usable_model_type(model_type: &str) -> bool {
    STEP_USABLE_MODEL_TYPES.contains(&model_type)
}

fn read_step_bearer_token(credential: Option<&Credential>) -> Option<String> {
    let credential = credential?;
    match credential {
        Credential::Oauth(oauth) => Some(oauth.access.trim())
            .filter(|token| !token.is_empty())
            .map(str::to_owned),
        Credential::ApiKey(api_key) => api_key
            .key
            .as_deref()
            .map(str::trim)
            .filter(|key| !key.is_empty())
            .map(str::to_owned),
    }
}

/// Percent-encode a path segment the way `encodeURIComponent` does for the
/// characters Step model ids use.
fn encode_path_segment(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        let unreserved = byte.is_ascii_alphanumeric()
            || matches!(
                byte,
                b'-' | b'_' | b'.' | b'!' | b'~' | b'*' | b'\'' | b'(' | b')'
            );
        if unreserved {
            encoded.push(byte as char);
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    encoded
}

// ---- small helpers ---------------------------------------------------------

fn process_env() -> HashMap<String, String> {
    std::env::vars_os()
        .filter_map(|(name, value)| Some((name.into_string().ok()?, value.into_string().ok()?)))
        .collect()
}

fn read_first_env(env: &HashMap<String, String>, names: &[&str]) -> Option<String> {
    names
        .iter()
        .filter_map(|name| env.get(*name))
        .map(|value| value.trim())
        .find(|value| !value.is_empty())
        .map(str::to_owned)
}

fn first_set(
    explicit: Option<&str>,
    names: &[&str],
    env: &HashMap<String, String>,
) -> Option<String> {
    explicit
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .or_else(|| read_first_env(env, names))
}

fn read_positive_integer(value: Option<&String>) -> Option<u64> {
    value
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
}

fn non_empty(value: &str, label: &str) -> Result<String, ModelsError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(ModelsError::new(
            ModelsErrorCode::Provider,
            format!("{label} must not be empty"),
        ));
    }
    Ok(trimmed.to_owned())
}

fn is_valid_env_name(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some(first) if first.is_ascii_alphabetic() || first == '_')
        && chars.all(|char| char.is_ascii_alphanumeric() || char == '_')
}

/// A fresh 256-bit poll token per login, hex-encoded (the TypeScript
/// `randomBytes(32).toString("hex")`).
fn random_poll_token() -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    let mut token = String::with_capacity(64);
    for byte in bytes {
        token.push(HEX[(byte >> 4) as usize] as char);
        token.push(HEX[(byte & 0x0f) as usize] as char);
    }
    token
}

impl From<StepLoginError> for ModelsError {
    fn from(error: StepLoginError) -> Self {
        match error {
            StepLoginError::Aborted => {
                ModelsError::new(ModelsErrorCode::Auth, "Step sign-in was cancelled")
            }
            StepLoginError::TimedOut(message) => ModelsError::new(ModelsErrorCode::Auth, message),
            StepLoginError::Rejected(message) => ModelsError::new(ModelsErrorCode::Oauth, message),
            StepLoginError::CliLogin(login_client::StepCliLoginError::Request(request)) => {
                ModelsError::new(ModelsErrorCode::Oauth, request.message)
            }
            StepLoginError::CliLogin(login_client::StepCliLoginError::Endpoint(message)) => {
                ModelsError::new(ModelsErrorCode::Provider, message)
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    //! Shared fakes for the Step login tests: a scripted transport and a
    //! recording interaction. Network-free by construction.

    use super::login_client::{
        StepCliLoginMethod, StepCliLoginRequest, StepCliLoginResponse, StepCliLoginTransport,
        StepCliLoginTransportError,
    };
    use crate::auth::types::{AuthEvent, AuthInteraction, AuthPrompt, AuthPromptCancelled};
    use async_trait::async_trait;
    use serde_json::Value;
    use std::collections::VecDeque;
    use std::sync::Mutex;
    use tokio_util::sync::CancellationToken;

    /// A JSON body with the given status.
    pub fn json_response(body: Value, status: u16) -> StepCliLoginResponse {
        StepCliLoginResponse {
            status,
            content_length: None,
            body: super::login_client::StepCliLoginBody::Text(body.to_string()),
        }
    }

    /// One request as the transport saw it.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct RecordedRequest {
        pub method: StepCliLoginMethod,
        pub url: String,
        pub poll_token: String,
        pub body: Option<String>,
    }

    /// One scripted exchange; `hang` parks the request forever.
    pub struct FakeStep {
        pub response: Result<StepCliLoginResponse, StepCliLoginTransportError>,
        pub hang: bool,
    }

    /// Scripted transport: replays canned responses in order and records every
    /// request.
    pub struct FakeTransport {
        script: Mutex<VecDeque<FakeStep>>,
        requests: Mutex<Vec<RecordedRequest>>,
    }

    impl FakeTransport {
        pub fn new(steps: Vec<FakeStep>) -> Self {
            FakeTransport {
                script: Mutex::new(steps.into_iter().collect()),
                requests: Mutex::new(Vec::new()),
            }
        }

        pub fn json(steps: Vec<(Value, u16)>) -> Self {
            FakeTransport::new(
                steps
                    .into_iter()
                    .map(|(body, status)| FakeStep {
                        response: Ok(json_response(body, status)),
                        hang: false,
                    })
                    .collect(),
            )
        }

        /// The same canned response for every exchange.
        pub fn repeated(body: Value, status: u16, count: usize) -> Self {
            FakeTransport::json(vec![(body, status); count])
        }

        /// A transport whose first exchange never resolves.
        pub fn hanging() -> Self {
            FakeTransport::new(vec![FakeStep {
                response: Err(StepCliLoginTransportError::new("unused")),
                hang: true,
            }])
        }

        pub fn requests(&self) -> Vec<RecordedRequest> {
            self.requests.lock().expect("fake transport lock").clone()
        }
    }

    #[async_trait]
    impl StepCliLoginTransport for FakeTransport {
        async fn exchange(
            &self,
            request: StepCliLoginRequest<'_>,
        ) -> Result<StepCliLoginResponse, StepCliLoginTransportError> {
            self.requests
                .lock()
                .expect("fake transport lock")
                .push(RecordedRequest {
                    method: request.method,
                    url: request.url.clone(),
                    poll_token: request.poll_token.to_owned(),
                    body: request.body.clone(),
                });
            let Some(step) = self.script.lock().expect("fake transport lock").pop_front() else {
                return Err(StepCliLoginTransportError::new("no scripted response"));
            };
            if step.hang {
                return std::future::pending().await;
            }
            step.response
        }
    }

    /// Per-exchange handler, for tests that must act while a request is in
    /// flight (e.g. cancel the login token).
    type RequestHandler = Box<
        dyn Fn(&RecordedRequest) -> Result<StepCliLoginResponse, StepCliLoginTransportError>
            + Send
            + Sync,
    >;

    pub struct HandlerTransport {
        handler: RequestHandler,
        requests: Mutex<Vec<RecordedRequest>>,
    }

    impl HandlerTransport {
        pub fn new(handler: RequestHandler) -> Self {
            HandlerTransport {
                handler,
                requests: Mutex::new(Vec::new()),
            }
        }

        pub fn requests(&self) -> Vec<RecordedRequest> {
            self.requests.lock().expect("fake transport lock").clone()
        }
    }

    #[async_trait]
    impl StepCliLoginTransport for HandlerTransport {
        async fn exchange(
            &self,
            request: StepCliLoginRequest<'_>,
        ) -> Result<StepCliLoginResponse, StepCliLoginTransportError> {
            let recorded = RecordedRequest {
                method: request.method,
                url: request.url.clone(),
                poll_token: request.poll_token.to_owned(),
                body: request.body.clone(),
            };
            let response = (self.handler)(&recorded);
            self.requests
                .lock()
                .expect("fake transport lock")
                .push(recorded);
            response
        }
    }

    /// An interaction that records every emitted event and never prompts.
    pub struct RecordingInteraction {
        token: CancellationToken,
        events: Mutex<Vec<AuthEvent>>,
    }

    impl RecordingInteraction {
        pub fn new() -> Self {
            RecordingInteraction {
                token: CancellationToken::new(),
                events: Mutex::new(Vec::new()),
            }
        }

        pub fn events(&self) -> Vec<AuthEvent> {
            self.events.lock().expect("interaction lock").clone()
        }
    }

    impl Default for RecordingInteraction {
        fn default() -> Self {
            RecordingInteraction::new()
        }
    }

    #[async_trait]
    impl AuthInteraction for RecordingInteraction {
        async fn prompt(&self, _prompt: AuthPrompt) -> Result<String, AuthPromptCancelled> {
            Err(AuthPromptCancelled)
        }

        fn notify(&self, event: AuthEvent) {
            self.events.lock().expect("interaction lock").push(event);
        }

        fn signal(&self) -> CancellationToken {
            self.token.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{FakeTransport, RecordingInteraction};
    use super::*;
    use crate::auth::types::AuthEvent;
    use serde_json::json;

    const AUTH_BASE_URL: &str = "https://auth.example.test";

    fn client_info() -> StepCliClientInfo {
        StepCliClientInfo {
            name: "stepcode".to_owned(),
            version: "1.2.3".to_owned(),
            platform: "darwin-arm64".to_owned(),
        }
    }

    fn options() -> StepProviderOptions {
        StepProviderOptions {
            auth_base_url: Some(AUTH_BASE_URL.to_owned()),
            client: Some(client_info()),
            env: Some(HashMap::new()),
            ..Default::default()
        }
    }

    fn resolved() -> ResolvedStepProviderOptions {
        resolve_step_provider_options(&options()).expect("options resolve")
    }

    // ---- URL helpers --------------------------------------------------------

    #[test]
    fn normalize_step_api_base_url_strips_version_and_operation_suffixes() {
        assert_eq!(
            normalize_step_api_base_url("https://api.stepfun.com/step_plan"),
            "https://api.stepfun.com/step_plan"
        );
        assert_eq!(
            normalize_step_api_base_url("https://api.stepfun.com/step_plan/"),
            "https://api.stepfun.com/step_plan"
        );
        assert_eq!(
            normalize_step_api_base_url("https://api.stepfun.com/step_plan/v1"),
            "https://api.stepfun.com/step_plan"
        );
        assert_eq!(
            normalize_step_api_base_url("https://api.stepfun.com/v1/messages"),
            "https://api.stepfun.com"
        );
        assert_eq!(
            normalize_step_api_base_url("https://api.stepfun.com/messages"),
            "https://api.stepfun.com"
        );
        // Idempotent, including for a stale duplicated suffix.
        assert_eq!(
            normalize_step_api_base_url("https://api.stepfun.com/v1/v1/messages"),
            "https://api.stepfun.com"
        );
        assert_eq!(
            normalize_step_api_base_url(&normalize_step_api_base_url(
                "https://api.stepfun.com/v1/messages"
            )),
            "https://api.stepfun.com"
        );
        assert_eq!(
            normalize_step_api_base_url("https://api.stepfun.com"),
            "https://api.stepfun.com"
        );
        // Malformed input is left intact for the caller's diagnostics.
        assert_eq!(normalize_step_api_base_url("not a url"), "not a url");
    }

    #[test]
    fn step_open_ai_base_url_reappends_v1() {
        assert_eq!(
            step_open_ai_base_url("https://api.stepfun.com/step_plan"),
            "https://api.stepfun.com/step_plan/v1"
        );
        assert_eq!(
            step_open_ai_base_url("https://api.stepfun.com/step_plan/v1"),
            "https://api.stepfun.com/step_plan/v1"
        );
    }

    #[test]
    fn step_models_detail_base_url_uses_the_domain_root() {
        assert_eq!(
            step_models_detail_base_url("https://api.stepfun.com/step_plan/v1"),
            "https://api.stepfun.com/v1"
        );
        assert_eq!(step_models_detail_base_url("not a url"), "not a url");
    }

    // ---- option resolution --------------------------------------------------

    #[test]
    fn resolves_production_defaults() {
        let resolved = resolve_step_provider_options(&StepProviderOptions {
            env: Some(HashMap::new()),
            ..Default::default()
        })
        .expect("options resolve");
        assert_eq!(resolved.provider_id, STEP_PROVIDER_ID);
        assert_eq!(resolved.name, "Step");
        assert_eq!(resolved.api_base_url, "https://api.stepfun.com/step_plan");
        assert_eq!(resolved.auth_base_url, "https://platform.stepfun.com");
        assert_eq!(resolved.api_key_env, "STEP_API_KEY");
        assert_eq!(resolved.timeout_ms, DEFAULT_STEP_CLI_LOGIN_TIMEOUT_MS);
        assert_eq!(resolved.login_profile, "step_plan");
        assert!(resolved.models.is_empty());
        assert_eq!(resolved.client.version, "unknown");
    }

    #[test]
    fn resolves_overrides_and_env_fallbacks() {
        let mut env = HashMap::new();
        env.insert(
            "STEP_BASE_URL".to_owned(),
            "https://api.example.test/step_plan/v1".to_owned(),
        );
        env.insert(
            "STEP_AUTH_URL".to_owned(),
            "https://auth.example.test".to_owned(),
        );
        env.insert("STEP_OAUTH_TIMEOUT_MS".to_owned(), "1234".to_owned());
        let resolved = resolve_step_provider_options(&StepProviderOptions {
            provider_id: Some("step-custom".to_owned()),
            name: Some(" Step Custom ".to_owned()),
            env: Some(env),
            ..Default::default()
        })
        .expect("options resolve");
        assert_eq!(resolved.provider_id, "step-custom");
        assert_eq!(resolved.name, "Step Custom");
        assert_eq!(resolved.api_base_url, "https://api.example.test/step_plan");
        assert_eq!(resolved.auth_base_url, "https://auth.example.test");
        assert_eq!(resolved.timeout_ms, 1234);
    }

    #[test]
    fn explicit_options_win_over_env() {
        let mut env = HashMap::new();
        env.insert(
            "STEP_BASE_URL".to_owned(),
            "https://env.example.test".to_owned(),
        );
        let resolved = resolve_step_provider_options(&StepProviderOptions {
            api_base_url: Some("https://option.example.test".to_owned()),
            auth_base_url: Some(AUTH_BASE_URL.to_owned()),
            env: Some(env),
            ..Default::default()
        })
        .expect("options resolve");
        assert_eq!(resolved.api_base_url, "https://option.example.test");
    }

    #[test]
    fn overseas_developer_center_selects_the_oversea_profile() {
        let resolved = resolve_step_provider_options(&StepProviderOptions {
            auth_base_url: Some("https://platform.stepfun.ai".to_owned()),
            env: Some(HashMap::new()),
            ..Default::default()
        })
        .expect("options resolve");
        assert_eq!(resolved.login_profile, "step_plan_oversea");
    }

    #[test]
    fn explicit_login_profile_wins() {
        let resolved = resolve_step_provider_options(&StepProviderOptions {
            auth_base_url: Some("https://platform.stepfun.ai".to_owned()),
            login_profile: Some("  step_plan  ".to_owned()),
            env: Some(HashMap::new()),
            ..Default::default()
        })
        .expect("options resolve");
        assert_eq!(resolved.login_profile, "step_plan");
    }

    #[test]
    fn rejects_invalid_endpoints_and_settings() {
        let cases: Vec<(StepProviderOptions, &str)> = vec![
            (
                StepProviderOptions {
                    api_base_url: Some("not a url".to_owned()),
                    env: Some(HashMap::new()),
                    ..Default::default()
                },
                "Step API base URL must be a valid URL",
            ),
            (
                StepProviderOptions {
                    api_base_url: Some("ftp://api.example.test".to_owned()),
                    env: Some(HashMap::new()),
                    ..Default::default()
                },
                "Step API base URL must use http or https",
            ),
            (
                StepProviderOptions {
                    auth_base_url: Some("https://user:pass@auth.example.test".to_owned()),
                    env: Some(HashMap::new()),
                    ..Default::default()
                },
                "Step OAuth authorization URL must not contain credentials",
            ),
            (
                StepProviderOptions {
                    provider_id: Some("   ".to_owned()),
                    env: Some(HashMap::new()),
                    ..Default::default()
                },
                "Step provider id must not be empty",
            ),
            (
                StepProviderOptions {
                    api_key_env: Some("9bad".to_owned()),
                    env: Some(HashMap::new()),
                    ..Default::default()
                },
                "Invalid Step API key environment variable name: 9bad",
            ),
            (
                StepProviderOptions {
                    timeout_ms: Some(0),
                    env: Some(HashMap::new()),
                    ..Default::default()
                },
                "Step login timeout must be finite and greater than zero",
            ),
        ];
        for (options, message) in cases {
            let error = resolve_step_provider_options(&options).expect_err("must reject");
            assert_eq!(error.message(), message);
            assert_eq!(error.code, ModelsErrorCode::Provider);
        }
    }

    #[test]
    fn ignores_an_unparsable_timeout_env_value() {
        let mut env = HashMap::new();
        env.insert(
            "STEP_OAUTH_TIMEOUT_MS".to_owned(),
            "not-a-number".to_owned(),
        );
        let resolved = resolve_step_provider_options(&StepProviderOptions {
            env: Some(env),
            ..Default::default()
        })
        .expect("options resolve");
        assert_eq!(resolved.timeout_ms, DEFAULT_STEP_CLI_LOGIN_TIMEOUT_MS);
    }

    // ---- thinking level map -------------------------------------------------

    #[test]
    fn thinking_level_map_marks_listed_levels_and_disables_the_rest() {
        let map = step_thinking_level_map(&["low".to_owned(), "HIGH".to_owned()]);
        assert_eq!(map.get(&ModelThinkingLevel::Off), Some(&None));
        assert_eq!(map.get(&ModelThinkingLevel::Minimal), Some(&None));
        assert_eq!(
            map.get(&ModelThinkingLevel::Low),
            Some(&Some("low".to_owned()))
        );
        assert_eq!(
            map.get(&ModelThinkingLevel::High),
            Some(&Some("high".to_owned()))
        );
        assert_eq!(map.get(&ModelThinkingLevel::Max), Some(&None));
        assert_eq!(map.len(), 7);
    }

    #[test]
    fn thinking_level_map_of_an_empty_list_disables_everything() {
        let map = step_thinking_level_map(&[]);
        assert!(map.values().all(Option::is_none));
    }

    // ---- model discovery ----------------------------------------------------

    #[test]
    fn parses_models_with_metadata_and_thinking_levels() {
        let payload = json!({
            "data": [
                {
                    "id": "step-3",
                    "model_type": "大语言模型",
                    "max_input_tokens": 128_000,
                    "enable_reason": true,
                    "enable_vision_input": true,
                    "reasoning_effort_support_list": ["minimal", "high"],
                },
                { "id": "step-2", "model_type": "大语言模型" },
            ]
        });
        let models = parse_step_models_response(&payload, "https://api.stepfun.com/step_plan/v1");
        assert_eq!(models.len(), 2);

        let first = &models[0];
        assert_eq!(first.id, "step-3");
        assert_eq!(first.name, "step-3");
        assert_eq!(first.api, Some(Api::OpenaiCompletions));
        assert_eq!(
            first.base_url.as_deref(),
            Some("https://api.stepfun.com/step_plan/v1")
        );
        assert!(first.reasoning);
        assert_eq!(first.input, vec![InputModality::Text, InputModality::Image]);
        assert_eq!(first.context_window, 128_000);
        assert_eq!(first.max_tokens, 128_000);
        let map = first.thinking_level_map.as_ref().expect("thinking map");
        assert_eq!(
            map.get(&ModelThinkingLevel::Minimal),
            Some(&Some("minimal".to_owned()))
        );
        assert_eq!(
            map.get(&ModelThinkingLevel::High),
            Some(&Some("high".to_owned()))
        );
        assert_eq!(map.get(&ModelThinkingLevel::Off), Some(&None));

        let second = &models[1];
        assert!(!second.reasoning);
        assert_eq!(second.input, vec![InputModality::Text]);
        assert_eq!(second.context_window, STEP_DEFAULT_CONTEXT_WINDOW);
        assert_eq!(second.max_tokens, STEP_DEFAULT_CONTEXT_WINDOW);
        assert!(second.thinking_level_map.is_none());
    }

    #[test]
    fn filters_unusable_model_types_only_when_types_are_reported() {
        let typed = json!({
            "data": [
                { "id": "llm", "model_type": "大语言模型" },
                { "id": "router", "model_type": "路由模型" },
                { "id": "tts", "model_type": "语音合成" },
                { "id": "untyped" },
            ]
        });
        let ids: Vec<String> = parse_step_models_response(&typed, "https://api.example.test/v1")
            .into_iter()
            .map(|model| model.id)
            .collect();
        assert_eq!(ids, vec!["llm", "router"]);

        // Older endpoints report no model_type at all: nothing is filtered.
        let untyped = json!({ "data": [{ "id": "tts" }, { "id": "asr" }] });
        let ids: Vec<String> = parse_step_models_response(&untyped, "https://api.example.test/v1")
            .into_iter()
            .map(|model| model.id)
            .collect();
        assert_eq!(ids, vec!["tts", "asr"]);
    }

    #[test]
    fn dedupes_models_and_skips_entries_without_an_id() {
        let payload = json!({
            "data": [
                { "id": "step-3", "max_input_tokens": 1000 },
                { "id": "step-3", "max_input_tokens": 2000 },
                { "id": "   " },
                { "model_type": "大语言模型" },
                "not-an-object",
            ]
        });
        let models = parse_step_models_response(&payload, "https://api.example.test/v1");
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].context_window, 1000);
    }

    #[test]
    fn parses_an_empty_or_malformed_models_payload() {
        assert!(parse_step_models_response(&json!({}), "https://api.example.test/v1").is_empty());
        assert!(
            parse_step_models_response(&json!({ "data": [] }), "https://api.example.test/v1")
                .is_empty()
        );
        assert!(
            parse_step_models_response(&json!("nope"), "https://api.example.test/v1").is_empty()
        );
    }

    // ---- normalization ------------------------------------------------------

    #[test]
    fn normalize_step_model_config_defaults_the_dialect() {
        let config = StepModelConfig {
            id: "m".to_owned(),
            name: "m".to_owned(),
            api: None,
            base_url: Some("https://old.example.test".to_owned()),
            reasoning: false,
            thinking_level_map: None,
            input: vec![InputModality::Text],
            cost: ModelCost::default(),
            context_window: 10,
            max_tokens: 5,
            headers: None,
            compat: None,
        };
        let normalized = normalize_step_model_config(config);
        assert_eq!(normalized.api, Some(Api::OpenaiCompletions));
        assert_eq!(
            normalized.base_url.as_deref(),
            Some("https://old.example.test")
        );
    }

    #[test]
    fn normalize_step_model_forces_the_step_dialect_and_base() {
        let mut model = Model {
            id: "m".to_owned(),
            name: "m".to_owned(),
            api: Api::AnthropicMessages,
            provider: ProviderId::new(STEP_PROVIDER_ID),
            base_url: "https://old.example.test".to_owned(),
            reasoning: false,
            thinking_level_map: None,
            input: vec![InputModality::Text],
            cost: ModelCost::default(),
            context_window: 10,
            max_tokens: 5,
            sampling_params: None,
            headers: None,
            compat: None,
        };
        model = normalize_step_model(&model, "https://api.stepfun.com/step_plan/v1");
        assert_eq!(model.api, Api::OpenaiCompletions);
        assert_eq!(model.base_url, "https://api.stepfun.com/step_plan/v1");
    }

    // ---- oauth --------------------------------------------------------------

    #[tokio::test]
    async fn refresh_re_stamps_a_static_credential_and_rejects_foreign_ones() {
        let credential = OAuthCredential {
            refresh: STEP_STATIC_REFRESH_TOKEN.to_owned(),
            access: "sk-step".to_owned(),
            expires: 1,
            extra: Some(HashMap::from([(
                "uid".to_owned(),
                Json::String("u-1".to_owned()),
            )])),
        };
        let refreshed = refresh_step_oauth(&credential).await.expect("refresh");
        assert_eq!(refreshed.access, "sk-step");
        assert_eq!(refreshed.expires, STEP_NEVER_EXPIRES_MS);
        assert_eq!(refreshed.extra, credential.extra);

        let foreign = OAuthCredential {
            refresh: "rotating-token".to_owned(),
            ..credential
        };
        let error = refresh_step_oauth(&foreign).await.expect_err("must reject");
        assert_eq!(error.code, ModelsErrorCode::Oauth);
        assert_eq!(
            error.message(),
            "This Step credential cannot be refreshed. Run `step login` again."
        );
    }

    #[test]
    fn get_step_oauth_api_key_trims_and_rejects_an_empty_token() {
        let credential = OAuthCredential {
            refresh: STEP_STATIC_REFRESH_TOKEN.to_owned(),
            access: "  sk-step  ".to_owned(),
            expires: STEP_NEVER_EXPIRES_MS,
            extra: None,
        };
        assert_eq!(get_step_oauth_api_key(&credential).expect("key"), "sk-step");

        let empty = OAuthCredential {
            access: "   ".to_owned(),
            ..credential
        };
        let error = get_step_oauth_api_key(&empty).expect_err("must reject");
        assert_eq!(error.code, ModelsErrorCode::Auth);
        assert_eq!(
            error.message(),
            "Step OAuth credentials do not contain an access token"
        );
    }

    #[tokio::test]
    async fn oauth_to_auth_exposes_the_access_token() {
        let oauth = StepOAuth::new(resolved());
        let credential = OAuthCredential {
            refresh: STEP_STATIC_REFRESH_TOKEN.to_owned(),
            access: "sk-step".to_owned(),
            expires: STEP_NEVER_EXPIRES_MS,
            extra: None,
        };
        let auth = oauth.to_auth(&credential).await.expect("to_auth");
        assert_eq!(auth.api_key.as_deref(), Some("sk-step"));
        assert_eq!(oauth.name(), "Step Plan");
        assert!(oauth.is_subscription());
    }

    // ---- cloud login --------------------------------------------------------

    fn init_body(overrides: Value) -> Value {
        let mut body = json!({
            "status": 0,
            "desc": "",
            "flow_id": "flow-1",
            "poll_interval_sec": 1,
            "expires_at": now_ms() / 1000 + 600,
        });
        if let Value::Object(overrides) = overrides {
            for (key, value) in overrides {
                body[key] = value;
            }
        }
        body
    }

    /// The poll token as the transport saw it. The `Bearer` prefix is added by
    /// the reqwest transport, so the boundary carries the raw token.
    fn poll_token(request: &test_support::RecordedRequest) -> &str {
        &request.poll_token
    }

    #[tokio::test]
    async fn login_generates_a_fresh_token_per_login_and_reuses_it_only_for_that_flow() {
        let transport = FakeTransport::json(vec![
            (init_body(Value::Null), 200),
            (
                json!({ "state": "ready", "credential": { "api_key": "key" } }),
                200,
            ),
            (init_body(Value::Null), 200),
            (
                json!({ "state": "ready", "credential": { "api_key": "key" } }),
                200,
            ),
        ]);
        let resolved = resolved();
        let interaction = RecordingInteraction::new();
        run_step_login(&interaction, &resolved, &transport)
            .await
            .expect("first login");
        run_step_login(&interaction, &resolved, &transport)
            .await
            .expect("second login");

        let requests = transport.requests();
        assert_eq!(requests.len(), 4);
        let first = poll_token(&requests[0]);
        let second = poll_token(&requests[1]);
        let third = poll_token(&requests[2]);
        assert_eq!(first, second);
        assert_eq!(third, poll_token(&requests[3]));
        assert_ne!(first, third);
        for token in [first, second, third] {
            assert_eq!(token.len(), 64);
            assert!(token.chars().all(|char| char.is_ascii_hexdigit()));
            for request in &requests {
                assert!(!request.url.contains(token));
                assert!(!request.body.as_deref().unwrap_or_default().contains(token));
            }
        }
    }

    #[tokio::test]
    async fn login_emits_the_browser_events_and_stores_the_polled_key() {
        let transport = FakeTransport::json(vec![
            (init_body(Value::Null), 200),
            (
                json!({ "state": "ready", "uid": "u-9", "credential": { "api_key": "polled-key" } }),
                200,
            ),
        ]);
        let interaction = RecordingInteraction::new();
        let credential = run_step_login(&interaction, &resolved(), &transport)
            .await
            .expect("login succeeds");

        assert_eq!(credential.access, "polled-key");
        assert_eq!(credential.refresh, STEP_STATIC_REFRESH_TOKEN);
        assert_eq!(credential.expires, STEP_NEVER_EXPIRES_MS);
        assert_eq!(
            credential
                .extra
                .as_ref()
                .and_then(|extra| extra.get("profile")),
            Some(&Json::String("step_plan".to_owned()))
        );
        assert_eq!(
            credential.extra.as_ref().and_then(|extra| extra.get("uid")),
            Some(&Json::String("u-9".to_owned()))
        );

        let events = interaction.events();
        let urls: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                AuthEvent::AuthUrl { url, .. } => Some(url.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            urls,
            vec!["https://auth.example.test/cli-login-remote?flow_id=flow-1"]
        );
        assert!(events.iter().any(|event| matches!(
            event,
            AuthEvent::DeviceCode {
                user_code,
                verification_uri,
                interval_seconds: Some(1),
                ..
            } if user_code == "flow-1"
                && verification_uri == "https://auth.example.test/cli-login-remote?flow_id=flow-1"
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            AuthEvent::Progress { message } if message == "Waiting for browser confirmation..."
        )));
        assert!(
            events
                .iter()
                .any(|event| matches!(event, AuthEvent::Info { .. }))
        );
    }

    #[tokio::test]
    async fn login_sends_the_profile_and_client_metadata_to_init() {
        let transport = FakeTransport::json(vec![
            (init_body(Value::Null), 200),
            (
                json!({ "state": "ready", "credential": { "api_key": "key" } }),
                200,
            ),
        ]);
        let resolved = resolve_step_provider_options(&StepProviderOptions {
            auth_base_url: Some("https://platform.stepfun.ai".to_owned()),
            env: Some(HashMap::new()),
            ..Default::default()
        })
        .expect("options resolve");
        assert_eq!(resolved.login_profile, "step_plan_oversea");
        let credential = run_step_login(&RecordingInteraction::new(), &resolved, &transport)
            .await
            .expect("login succeeds");
        assert_eq!(
            credential
                .extra
                .as_ref()
                .and_then(|extra| extra.get("profile")),
            Some(&Json::String("step_plan_oversea".to_owned()))
        );
        let body: Value =
            serde_json::from_str(transport.requests()[0].body.as_deref().expect("init body"))
                .expect("init body json");
        assert_eq!(body["profile"], "step_plan_oversea");
        assert_eq!(body["origin"], "https://platform.stepfun.ai");
        assert_eq!(body["client"]["name"], "stepcode");
        assert_eq!(body["client"]["version"], "unknown");
    }

    #[tokio::test]
    async fn login_reports_unsupported_cloud_http_without_emitting_an_auth_url() {
        for status in [404u16, 405, 501] {
            let transport = FakeTransport::json(vec![(json!({ "desc": "not supported" }), status)]);
            let interaction = RecordingInteraction::new();
            let error = run_step_login(&interaction, &resolved(), &transport)
                .await
                .expect_err("login must fail");
            assert!(
                error.to_string().contains(&format!("HTTP {status}")),
                "{error}"
            );
            assert!(interaction.events().is_empty());
        }
    }

    #[tokio::test]
    async fn login_surfaces_init_protocol_and_transport_failures() {
        let protocol = FakeTransport::json(vec![(
            Value::String("<html>Unsupported endpoint</html>".to_owned()),
            200,
        )]);
        let error = run_step_login(&RecordingInteraction::new(), &resolved(), &protocol)
            .await
            .expect_err("login must fail");
        assert!(
            matches!(error, StepLoginError::CliLogin(login_client::StepCliLoginError::Request(ref request)) if request.kind == StepCliLoginErrorKind::Protocol)
        );

        let transport = FakeTransport::new(vec![test_support::FakeStep {
            response: Err(login_client::StepCliLoginTransportError::new(
                "network unavailable",
            )),
            hang: false,
        }]);
        let error = run_step_login(&RecordingInteraction::new(), &resolved(), &transport)
            .await
            .expect_err("login must fail");
        assert!(
            matches!(error, StepLoginError::CliLogin(login_client::StepCliLoginError::Request(ref request)) if request.kind == StepCliLoginErrorKind::Transport)
        );
    }

    #[tokio::test]
    async fn login_surfaces_a_denied_cloud_flow() {
        let transport = FakeTransport::json(vec![
            (init_body(Value::Null), 200),
            (json!({ "state": "failed", "reason": "denied" }), 200),
        ]);
        let error = run_step_login(&RecordingInteraction::new(), &resolved(), &transport)
            .await
            .expect_err("login must fail");
        assert_eq!(error.to_string(), "Sign-in was denied in the browser.");
    }

    #[tokio::test]
    async fn login_includes_init_in_the_login_deadline() {
        let transport = FakeTransport::hanging();
        let resolved = resolve_step_provider_options(&StepProviderOptions {
            timeout_ms: Some(1_000),
            ..options()
        })
        .expect("options resolve");
        let error = run_step_login(&RecordingInteraction::new(), &resolved, &transport)
            .await
            .expect_err("login must time out");
        assert_eq!(
            error.to_string(),
            "Step sign-in timed out. Run `step login` again."
        );
    }

    #[tokio::test]
    async fn login_does_not_use_echoed_profile_metadata_as_an_authentication_check() {
        let transport = FakeTransport::json(vec![
            (init_body(Value::Null), 200),
            (
                json!({ "state": "ready", "profile": "step_plan_oversea", "credential": { "api_key": "wrong-region" } }),
                200,
            ),
        ]);
        let credential = run_step_login(&RecordingInteraction::new(), &resolved(), &transport)
            .await
            .expect("login succeeds");
        assert_eq!(credential.access, "wrong-region");
        assert_eq!(
            credential
                .extra
                .as_ref()
                .and_then(|extra| extra.get("profile")),
            Some(&Json::String("step_plan".to_owned()))
        );
    }

    #[tokio::test]
    async fn login_maps_failures_to_models_errors() {
        let transport = FakeTransport::json(vec![(json!({ "desc": "no plan" }), 403)]);
        let error: ModelsError =
            run_step_login(&RecordingInteraction::new(), &resolved(), &transport)
                .await
                .expect_err("login must fail")
                .into();
        assert_eq!(error.code, ModelsErrorCode::Oauth);
        assert!(error.message().contains("no plan"), "{}", error.message());
    }

    // ---- provider registration ---------------------------------------------

    #[test]
    fn step_provider_registers_auth_dispatch_and_discovery() {
        let provider = step_provider(&options()).expect("provider builds");
        assert_eq!(provider.id.as_str(), STEP_PROVIDER_ID);
        assert_eq!(provider.name, "Step");
        assert_eq!(
            provider.base_url.as_deref(),
            Some("https://api.stepfun.com/step_plan/v1")
        );
        // No built-in baseline: the catalog is discovered after login.
        assert!(provider.get_models().is_empty());
        assert!(provider.has_refresh());
        let debug = format!("{:?}", provider.auth);
        assert!(debug.contains("Step Plan"), "{debug}");
        assert!(debug.contains("Step API key"), "{debug}");
    }

    #[test]
    fn step_provider_normalizes_baseline_models() {
        let mut options = options();
        options.models = vec![StepModelConfig {
            id: "step-custom".to_owned(),
            name: "Step Custom".to_owned(),
            api: Some(Api::AnthropicMessages),
            base_url: Some("https://stale.example.test".to_owned()),
            reasoning: true,
            thinking_level_map: None,
            input: vec![InputModality::Text],
            cost: ModelCost::default(),
            context_window: 32_000,
            max_tokens: 8_000,
            headers: None,
            compat: None,
        }];
        let provider = step_provider(&options).expect("provider builds");
        let models = provider.get_models();
        assert_eq!(models.len(), 1);
        let model = &models[0];
        assert_eq!(model.api, Api::OpenaiCompletions);
        assert_eq!(model.base_url, "https://api.stepfun.com/step_plan/v1");
        assert_eq!(model.provider.as_str(), STEP_PROVIDER_ID);
        assert_eq!(model.context_window, 32_000);
        assert_eq!(model.max_tokens, 8_000);
    }

    // ---- api-key auth -------------------------------------------------------

    #[tokio::test]
    async fn api_key_auth_prefers_the_stored_key_then_the_scoped_env() {
        use crate::auth::context::DefaultAuthContext;

        let mut env = HashMap::new();
        env.insert("STEP_API_KEY".to_owned(), "scoped-key".to_owned());
        let resolved = resolve_step_provider_options(&StepProviderOptions {
            env: Some(env),
            ..Default::default()
        })
        .expect("options resolve");
        let auth = StepApiKeyAuth::new(&resolved);
        let token = CancellationToken::new();

        let stored = ApiKeyCredential {
            key: Some("stored-key".to_owned()),
            env: None,
        };
        let result = auth
            .resolve(&DefaultAuthContext::new(), Some(&stored), &token)
            .await
            .expect("resolve")
            .expect("configured");
        assert_eq!(result.auth.api_key.as_deref(), Some("stored-key"));
        assert_eq!(result.source.as_deref(), Some("stored credential"));

        let result = auth
            .resolve(&DefaultAuthContext::new(), None, &token)
            .await
            .expect("resolve")
            .expect("configured");
        assert_eq!(result.auth.api_key.as_deref(), Some("scoped-key"));
        assert_eq!(result.source.as_deref(), Some("STEP_API_KEY"));

        // Ambient-only: no interactive login.
        let interaction = RecordingInteraction::new();
        let error = ApiKeyAuth::login(&auth, &interaction)
            .await
            .expect_err("api key login is unsupported");
        assert!(error.to_string().contains("login not supported"), "{error}");
    }

    // ---- encode helper ------------------------------------------------------

    #[test]
    fn encode_path_segment_matches_encode_uri_component() {
        assert_eq!(encode_path_segment("step-3"), "step-3");
        assert_eq!(encode_path_segment("a.b_c~d"), "a.b_c~d");
        assert_eq!(
            encode_path_segment("model/with space"),
            "model%2Fwith%20space"
        );
        assert_eq!(encode_path_segment("模型"), "%E6%A8%A1%E5%9E%8B");
    }
}
