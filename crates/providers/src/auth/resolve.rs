//! Auth resolution shared by model collections. A stored credential owns the
//! provider: ambient/env is consulted only when nothing is stored. No silent
//! env fallback after a failed refresh or for a credential type without a
//! matching handler.

use std::collections::HashMap;
use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::auth::context::AuthContext;
use crate::auth::types::{
    ApiKeyAuth, ApiKeyCredential, AuthResult, Credential, CredentialStore, OAuthAuth,
    OAuthCredential, ProviderAuth,
};
pub use crate::error::{ModelsError, ModelsErrorCode};

use super::types::modifier;

/// Caller-controlled resolution overrides.
#[derive(Clone, Default)]
pub struct AuthResolutionOverrides {
    /// Explicit API key; bypasses stored credentials when the provider has an
    /// api-key auth method.
    pub api_key: Option<String>,
    pub env: Option<HashMap<String, String>>,
    /// Require this much remaining OAuth-token validity; defaults to five minutes.
    pub min_oauth_validity_ms: Option<u64>,
    pub signal: Option<CancellationToken>,
}

const DEFAULT_OAUTH_MINIMUM_VALIDITY_MS: u64 = 5 * 60 * 1000;
const DEFAULT_OAUTH_REFRESH_TIMEOUT_MS: u64 = 15_000;

/// Resolve provider auth: stored credential → ambient (env vars, files).
/// Errors with [`ModelsError`] (codes `auth`/`oauth`) on store or refresh
/// failures; `Ok(None)` means "not configured".
pub async fn resolve_provider_auth(
    provider_id: &str,
    auth: &ProviderAuth,
    credentials: &dyn CredentialStore,
    auth_context: &dyn AuthContext,
    overrides: Option<&AuthResolutionOverrides>,
) -> Result<Option<AuthResult>, ModelsError> {
    let overrides = overrides.cloned().unwrap_or_default();
    let token = overrides
        .signal
        .clone()
        .unwrap_or_else(CancellationToken::new);
    if token.is_cancelled() {
        return Err(ModelsError::new(ModelsErrorCode::Auth, "aborted"));
    }

    let request_auth_context = request_auth_context_boxed(overrides.env.as_ref(), auth_context);

    if let Some(explicit_key) = overrides.api_key.as_deref().filter(|key| !key.is_empty())
        && auth.api_key.is_some()
    {
        let credential = ApiKeyCredential {
            key: Some(explicit_key.to_owned()),
            env: overrides.env.clone(),
        };
        return resolve_api_key(
            request_auth_context.as_ref(),
            auth.api_key.as_ref().expect("checked above"),
            provider_id,
            Some(&credential),
            &token,
        )
        .await;
    }

    let stored = read_credential(credentials, provider_id, &token).await?;
    if let Some(stored) = stored {
        return match &stored {
            Credential::Oauth(oauth_credential) if auth.oauth.is_some() => {
                resolve_stored_oauth(
                    credentials,
                    provider_id,
                    auth.oauth.as_ref().expect("checked above"),
                    oauth_credential,
                    &token,
                    overrides.min_oauth_validity_ms,
                )
                .await
            }
            Credential::ApiKey(api_key_credential) if auth.api_key.is_some() => {
                let mut credential = api_key_credential.clone();
                if let Some(env) = &overrides.env {
                    let merged = credential.env.get_or_insert_with(HashMap::new);
                    for (name, value) in env {
                        merged.insert(name.clone(), value.clone());
                    }
                }
                resolve_api_key(
                    request_auth_context.as_ref(),
                    auth.api_key.as_ref().expect("checked above"),
                    provider_id,
                    Some(&credential),
                    &token,
                )
                .await
            }
            _ => Ok(None),
        };
    }

    // Ambient (env vars, AWS profiles, ADC files).
    match &auth.api_key {
        Some(api_key) => {
            resolve_api_key(
                request_auth_context.as_ref(),
                api_key,
                provider_id,
                None,
                &token,
            )
            .await
        }
        None => Ok(None),
    }
}

fn request_auth_context_boxed<'a>(
    overlay: Option<&HashMap<String, String>>,
    base: &'a dyn AuthContext,
) -> Box<dyn AuthContext + 'a> {
    match overlay {
        Some(overlay) => Box::new(OverlayAuthContext {
            base,
            overlay: overlay.clone(),
        }),
        None => Box::new(PassthroughContext(base)),
    }
}

struct OverlayAuthContext<'a> {
    base: &'a dyn AuthContext,
    overlay: HashMap<String, String>,
}

#[async_trait::async_trait]
impl AuthContext for OverlayAuthContext<'_> {
    async fn env(&self, name: &str) -> Option<String> {
        match self.overlay.get(name) {
            Some(value) => Some(value.clone()),
            None => self.base.env(name).await,
        }
    }

    async fn file_exists(&self, path: &str) -> bool {
        self.base.file_exists(path).await
    }
}

struct PassthroughContext<'a>(&'a dyn AuthContext);

#[async_trait::async_trait]
impl AuthContext for PassthroughContext<'_> {
    async fn env(&self, name: &str) -> Option<String> {
        self.0.env(name).await
    }

    async fn file_exists(&self, path: &str) -> bool {
        self.0.file_exists(path).await
    }
}

/// OAuth resolution with double-checked locking: tokens with less than five
/// minutes remaining lock, re-check expiry under the lock, refresh once
/// globally, and persist the rotated credential before release.
async fn resolve_stored_oauth(
    credentials: &dyn CredentialStore,
    provider_id: &str,
    oauth: &Arc<dyn OAuthAuth>,
    stored: &OAuthCredential,
    signal: &CancellationToken,
    min_validity_ms: Option<u64>,
) -> Result<Option<AuthResult>, ModelsError> {
    let minimum_validity_ms = DEFAULT_OAUTH_MINIMUM_VALIDITY_MS.max(min_validity_ms.unwrap_or(0));
    let expires_soon = move |credential: &OAuthCredential| {
        crate::utils::now_ms().saturating_add(minimum_validity_ms) >= credential.expires
    };
    let mut credential = stored.clone();

    if expires_soon(&credential) {
        // Optimistic check said expired; the authoritative check runs under the lock.
        // The refresh races the caller's signal against a 15s timeout, matching
        // AbortSignal.any([signal, AbortSignal.timeout(15s)]).
        let refresh_signal = signal.clone();
        let oauth_ref = oauth.clone();
        let provider_label = provider_id.to_owned();
        let post = credentials
			.modify(
				provider_id,
				modifier(move |current| {
					let refresh_signal = refresh_signal.clone();
					let oauth_ref = oauth_ref.clone();
					let provider_label = provider_label.clone();
					let expires_soon = expires_soon;
					Box::pin(async move {
						let Some(Credential::Oauth(current)) = current else {
							// logged out meanwhile
							return Ok(None);
						};
						if !expires_soon(&current) {
							// another process/request refreshed
							return Ok(None);
						}
						let never_cancelled = CancellationToken::new();
						let refreshed = tokio::select! {
							biased;
							_ = refresh_signal.cancelled() => Err("aborted".into()),
							_ = tokio::time::sleep(std::time::Duration::from_millis(DEFAULT_OAUTH_REFRESH_TIMEOUT_MS)) => {
								Err("refresh timed out".into())
							}
							result = oauth_ref.refresh(&current, &never_cancelled) => {
								result.map_err(|error| Box::new(ModelsError::with_source(
									ModelsErrorCode::Oauth,
									format!("OAuth refresh failed for {provider_label}"),
									error,
								)) as Box<dyn std::error::Error + Send + Sync>)
							}
						}?;
						Ok(Some(Credential::Oauth(refreshed)))
					})
				}),
				signal,
			)
			.await
			.map_err(|error| {
				ModelsError::with_source(ModelsErrorCode::Auth, format!("Credential store modify failed for {provider_id}"), error)
			})?;

        let Some(Credential::Oauth(refreshed)) = post else {
            // logged out meanwhile
            return Ok(None);
        };
        credential = refreshed;
        // The normal five-minute window triggers a refresh but does not impose a
        // provider contract. Explicit callers require the requested minimum after
        // the refresh.
        if min_validity_ms.is_some() && expires_soon(&credential) {
            return Err(ModelsError::new(
                ModelsErrorCode::Oauth,
                format!("OAuth refresh returned a token that expires too soon for {provider_id}"),
            ));
        }
    }

    match oauth.to_auth(&credential).await {
        Ok(auth) => Ok(Some(AuthResult {
            auth,
            env: None,
            source: Some("OAuth".to_owned()),
        })),
        Err(error) => Err(ModelsError::with_source(
            ModelsErrorCode::Oauth,
            format!("OAuth auth derivation failed for {provider_id}"),
            error,
        )),
    }
}

async fn resolve_api_key(
    auth_context: &dyn AuthContext,
    api_key: &Arc<dyn ApiKeyAuth>,
    provider_id: &str,
    credential: Option<&ApiKeyCredential>,
    signal: &CancellationToken,
) -> Result<Option<AuthResult>, ModelsError> {
    api_key
        .resolve(auth_context, credential, signal)
        .await
        .map_err(|error| {
            ModelsError::with_source(
                ModelsErrorCode::Auth,
                format!("API key auth failed for provider {provider_id}"),
                error,
            )
        })
}

async fn read_credential(
    credentials: &dyn CredentialStore,
    provider_id: &str,
    signal: &CancellationToken,
) -> Result<Option<Credential>, ModelsError> {
    credentials
        .read(provider_id, signal)
        .await
        .map_err(|error| {
            ModelsError::with_source(
                ModelsErrorCode::Auth,
                format!("Credential store read failed for {provider_id}"),
                error,
            )
        })
}

impl std::fmt::Debug for AuthResolutionOverrides {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthResolutionOverrides")
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field(
                "env_names",
                &self.env.as_ref().map(|env| env.keys().collect::<Vec<_>>()),
            )
            .field("min_oauth_validity_ms", &self.min_oauth_validity_ms)
            .finish_non_exhaustive()
    }
}
