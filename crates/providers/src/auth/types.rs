//! Auth types: credentials, stores, resolution results, and login interaction.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::context::AuthContext;
use crate::types::{Json, ProviderEnv, ProviderHeaders};

/// Request auth for a single model request. If a value cannot be expressed as
/// `api_key`, `headers`, or `base_url`, it is provider config, not auth.
#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ModelAuth {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<ProviderHeaders>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
}

/// Stored api-key credential. `env` holds provider-scoped environment/config
/// values such as Cloudflare account/gateway ids.
#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub struct ApiKeyCredential {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env: Option<ProviderEnv>,
}

/// Stored canonical OAuth credential. Provider-specific extra fields survive
/// round-trips in `extra`.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OAuthCredential {
    pub refresh: String,
    pub access: String,
    /// Expiry as a Unix timestamp in milliseconds.
    pub expires: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<HashMap<String, Json>>,
}

/// One type-tagged credential per provider — the shape of today's auth.json.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Credential {
    ApiKey(ApiKeyCredential),
    Oauth(OAuthCredential),
}

impl Credential {
    pub fn as_api_key(&self) -> Option<&ApiKeyCredential> {
        match self {
            Credential::ApiKey(credential) => Some(credential),
            _ => None,
        }
    }

    pub fn as_oauth(&self) -> Option<&OAuthCredential> {
        match self {
            Credential::Oauth(credential) => Some(credential),
            _ => None,
        }
    }
}

/// Non-secret credential metadata for account/status enumeration.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CredentialInfo {
    pub provider_id: String,
    #[serde(rename = "type")]
    pub kind: CredentialKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CredentialKind {
    ApiKey,
    Oauth,
}

/// Result of resolving auth for a model.
#[derive(Clone, Default)]
pub struct AuthResult {
    pub auth: ModelAuth,
    /// Provider-scoped environment/config values resolved from credentials and
    /// ambient context.
    pub env: Option<ProviderEnv>,
    /// Human-readable label for status UI: "ANTHROPIC_API_KEY", "OAuth", ...
    pub source: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AuthCheck {
    pub source: Option<String>,
    pub kind: Option<CredentialKind>,
}

impl AuthCheck {
    /// A configured check for the given source label.
    pub fn configured(source: impl Into<String>, kind: CredentialKind) -> Self {
        AuthCheck {
            source: Some(source.into()),
            kind: Some(kind),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthType {
    ApiKey,
    Oauth,
}

/// Login prompt shown to the user.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AuthPrompt {
    Text {
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        placeholder: Option<String>,
    },
    Secret {
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        placeholder: Option<String>,
    },
    Select {
        message: String,
        options: Vec<AuthSelectOption>,
    },
    ManualCode {
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        placeholder: Option<String>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthSelectOption {
    pub id: String,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthInfoLink {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

/// Progress events during a login flow.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AuthEvent {
    Info {
        message: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        links: Vec<AuthInfoLink>,
    },
    AuthUrl {
        url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        instructions: Option<String>,
    },
    DeviceCode {
        user_code: String,
        verification_uri: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        interval_seconds: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        expires_in_seconds: Option<u64>,
    },
    Progress {
        message: String,
    },
}

/// Login interaction callbacks serving both api-key and OAuth flows.
///
/// `prompt` returns the entered/selected string (`Select` returns the option
/// id) and errors on cancel. The cancellation token aborts the whole flow.
#[async_trait::async_trait]
pub trait AuthInteraction: Send + Sync {
    async fn prompt(&self, prompt: AuthPrompt) -> Result<String, AuthPromptCancelled>;
    fn notify(&self, event: AuthEvent);
    fn signal(&self) -> tokio_util::sync::CancellationToken;
}

#[derive(Debug, thiserror::Error)]
#[error("prompt cancelled")]
pub struct AuthPromptCancelled;

/// Api-key auth: stored key/provider env plus ambient sources (env vars, AWS
/// profiles, ADC files). Ambient-only providers omit `login`.
#[async_trait::async_trait]
pub trait ApiKeyAuth: Send + Sync {
    /// Display name, e.g. "Anthropic API key".
    fn name(&self) -> &str;

    /// Interactive setup (prompt for key/provider env). Errors when ambient-only.
    async fn login(
        &self,
        interaction: &dyn AuthInteraction,
    ) -> Result<ApiKeyCredential, Box<dyn std::error::Error + Send + Sync>> {
        let _ = interaction;
        Err("login not supported".into())
    }

    /// Optional side-effect-free availability check. Missing means availability
    /// is checked by resolving auth.
    async fn check(
        &self,
        ctx: &dyn AuthContext,
        credential: Option<&ApiKeyCredential>,
        token: &tokio_util::sync::CancellationToken,
    ) -> Result<Option<AuthCheck>, Box<dyn std::error::Error + Send + Sync>> {
        let _ = (ctx, credential, token);
        Ok(None)
    }

    /// Resolve auth from the stored credential and/or ambient sources, merging
    /// per field (`credential.key ?? env("...")`). `None` = not configured.
    async fn resolve(
        &self,
        ctx: &dyn AuthContext,
        credential: Option<&ApiKeyCredential>,
        token: &tokio_util::sync::CancellationToken,
    ) -> Result<Option<AuthResult>, Box<dyn std::error::Error + Send + Sync>>;
}

/// OAuth auth. The `refresh`/`to_auth` split lets the collection own the
/// locked refresh pattern: `refresh` produces a credential, `to_auth` derives
/// request auth from whatever credential ends up stored.
#[async_trait::async_trait]
pub trait OAuthAuth: Send + Sync {
    /// Display name, e.g. "Anthropic (Claude Pro/Max)".
    fn name(&self) -> &str;

    /// Whether access through this auth method is backed by a provider subscription.
    fn is_subscription(&self) -> bool {
        false
    }

    /// Selector label for the OAuth login option.
    fn login_label(&self) -> Option<String> {
        None
    }

    async fn login(
        &self,
        interaction: &dyn AuthInteraction,
    ) -> Result<OAuthCredential, Box<dyn std::error::Error + Send + Sync>>;

    /// Exchange the refresh token. Network call; errors on failure
    /// (invalid_grant etc.). Runs under the store lock.
    async fn refresh(
        &self,
        credential: &OAuthCredential,
        token: &tokio_util::sync::CancellationToken,
    ) -> Result<OAuthCredential, Box<dyn std::error::Error + Send + Sync>>;

    /// Side-effect-free derivation of request auth from a valid credential.
    /// Covers per-credential `base_url` (GitHub Copilot).
    async fn to_auth(
        &self,
        credential: &OAuthCredential,
    ) -> Result<ModelAuth, Box<dyn std::error::Error + Send + Sync>>;
}

/// Provider auth. At least one of `api_key`/`oauth` must be present: even
/// ambient-credential providers and keyless local servers provide `api_key`
/// auth whose `resolve()` reports whether the provider is configured.
#[derive(Clone, Default)]
pub struct ProviderAuth {
    pub api_key: Option<Arc<dyn ApiKeyAuth>>,
    pub oauth: Option<Arc<dyn OAuthAuth>>,
}

impl ProviderAuth {
    pub fn with_api_key(auth: impl ApiKeyAuth + 'static) -> Self {
        ProviderAuth {
            api_key: Some(Arc::new(auth)),
            oauth: None,
        }
    }

    pub fn with_oauth(auth: impl OAuthAuth + 'static) -> Self {
        ProviderAuth {
            api_key: None,
            oauth: Some(Arc::new(auth)),
        }
    }

    pub fn ambient_only() -> Self {
        ProviderAuth::default()
    }
}

impl std::fmt::Debug for ProviderAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderAuth")
            .field("api_key", &self.api_key.as_ref().map(|auth| auth.name()))
            .field("oauth", &self.oauth.as_ref().map(|auth| auth.name()))
            .finish()
    }
}

/// App-owned credential storage, keyed by provider id, one credential per
/// provider. `modify` is the only write path, so every mutation is a
/// serialized read-modify-write; OAuth refreshes run inside it so concurrent
/// requests cannot double-refresh a rotated token.
#[async_trait::async_trait]
pub trait CredentialStore: Send + Sync {
    /// Read the stored credential, possibly expired. Display/status use;
    /// resolved request auth comes from the collection's `get_auth()`.
    async fn read(
        &self,
        provider_id: &str,
        token: &tokio_util::sync::CancellationToken,
    ) -> Result<Option<Credential>, Box<dyn std::error::Error + Send + Sync>>;

    /// List stored credential metadata without exposing secrets.
    async fn list(
        &self,
        token: &tokio_util::sync::CancellationToken,
    ) -> Result<Vec<CredentialInfo>, Box<dyn std::error::Error + Send + Sync>>;

    /// Serialized write — the only write path. `modifier` sees the current
    /// credential because correct writes (refresh, login-during-refresh) depend
    /// on it; return the new credential, or `None` to leave the entry unchanged.
    async fn modify(
        &self,
        provider_id: &str,
        modifier: ModifierFn,
        token: &tokio_util::sync::CancellationToken,
    ) -> Result<Option<Credential>, Box<dyn std::error::Error + Send + Sync>>;

    /// Remove a credential (logout). Serialized against `modify`.
    async fn delete(
        &self,
        provider_id: &str,
        token: &tokio_util::sync::CancellationToken,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}

/// The read-modify-write closure used by [`CredentialStore::modify`]. Return
/// the new credential or `None` to leave the entry unchanged.
pub type ModifierFn = Arc<
    dyn Fn(Option<Credential>) -> futures::future::BoxFuture<'static, ModifyOutcome> + Send + Sync,
>;

/// Convenience alias for a boxed modifier future result.
pub type ModifyOutcome = Result<Option<Credential>, Box<dyn std::error::Error + Send + Sync>>;

/// Wrap a modifier closure into the type-erased [`ModifierFn`].
pub fn modifier<F>(f: F) -> ModifierFn
where
    F: Fn(Option<Credential>) -> futures::future::BoxFuture<'static, ModifyOutcome>
        + Send
        + Sync
        + 'static,
{
    Arc::new(f)
}

impl std::fmt::Debug for ModelAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelAuth")
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field(
                "header_names",
                &self
                    .headers
                    .as_ref()
                    .map(|values| values.keys().collect::<Vec<_>>()),
            )
            .field("base_url", &self.base_url)
            .finish()
    }
}
impl std::fmt::Debug for ApiKeyCredential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApiKeyCredential")
            .field("key", &self.key.as_ref().map(|_| "<redacted>"))
            .field(
                "env_names",
                &self
                    .env
                    .as_ref()
                    .map(|values| values.keys().collect::<Vec<_>>()),
            )
            .finish()
    }
}
impl std::fmt::Debug for OAuthCredential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OAuthCredential")
            .field("refresh", &"<redacted>")
            .field("access", &"<redacted>")
            .field("expires", &self.expires)
            .field(
                "extra_names",
                &self
                    .extra
                    .as_ref()
                    .map(|values| values.keys().collect::<Vec<_>>()),
            )
            .finish()
    }
}
impl std::fmt::Debug for AuthResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthResult")
            .field("auth", &self.auth)
            .field(
                "env_names",
                &self
                    .env
                    .as_ref()
                    .map(|values| values.keys().collect::<Vec<_>>()),
            )
            .field("source", &self.source)
            .finish()
    }
}

#[cfg(test)]
mod debug_tests {
    use super::*;

    #[test]
    fn credentials_and_request_debug_redact_secret_values() {
        let secret = "secret-marker-123";
        let env = HashMap::from([("KEY".to_owned(), secret.to_owned())]);
        let auth = ModelAuth {
            api_key: Some(secret.to_owned()),
            headers: Some(HashMap::from([(
                "authorization".to_owned(),
                Some(secret.to_owned()),
            )])),
            ..Default::default()
        };
        let api_key = ApiKeyCredential {
            key: Some(secret.to_owned()),
            env: Some(env.clone()),
        };
        let oauth = OAuthCredential {
            access: secret.to_owned(),
            refresh: secret.to_owned(),
            expires: 0,
            extra: Some(HashMap::from([(
                "token".to_owned(),
                Json::String(secret.to_owned()),
            )])),
        };
        let resolved = AuthResult {
            auth: auth.clone(),
            env: Some(env.clone()),
            source: None,
        };
        let request = crate::types::ProviderRequestOptions {
            api_key: Some(secret.to_owned()),
            env: Some(env.clone()),
            headers: auth.headers.clone(),
            ..Default::default()
        };
        let step = crate::step_provider::StepProviderOptions {
            env: Some(env),
            ..Default::default()
        };
        for debug in [
            format!("{auth:?}"),
            format!("{api_key:?}"),
            format!("{oauth:?}"),
            format!("{resolved:?}"),
            format!("{request:?}"),
            format!("{step:?}"),
        ] {
            assert!(!debug.contains(secret), "secret leaked: {debug}");
        }
    }
}
