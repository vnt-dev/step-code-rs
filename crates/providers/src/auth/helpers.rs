//! Standard auth method builders.

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use super::context::AuthContext;
use super::types::{
    ApiKeyAuth, ApiKeyCredential, AuthCheck, AuthInteraction, AuthPrompt, AuthResult,
    CredentialKind,
};

/// Standard api-key auth: a stored credential key wins, otherwise the first
/// set env var resolves. Includes a `login` that prompts for the key.
/// Providers with non-standard resolution write their own [`ApiKeyAuth`].
pub fn env_api_key_auth(name: impl Into<String>, env_vars: Vec<String>) -> EnvApiKeyAuth {
    EnvApiKeyAuth {
        name: name.into(),
        env_vars,
    }
}

pub struct EnvApiKeyAuth {
    name: String,
    env_vars: Vec<String>,
}

#[async_trait]
impl ApiKeyAuth for EnvApiKeyAuth {
    fn name(&self) -> &str {
        &self.name
    }

    async fn login(
        &self,
        interaction: &dyn AuthInteraction,
    ) -> Result<ApiKeyCredential, Box<dyn std::error::Error + Send + Sync>> {
        if interaction.signal().is_cancelled() {
            return Err("aborted".into());
        }
        let key = interaction
            .prompt(AuthPrompt::Secret {
                message: format!("Enter {}", self.name),
                placeholder: None,
            })
            .await
            .map_err(|_| "aborted")?;
        if interaction.signal().is_cancelled() {
            return Err("aborted".into());
        }
        Ok(ApiKeyCredential {
            key: Some(key),
            env: None,
        })
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
            .filter(|key| !key.is_empty())
        {
            return Ok(Some(AuthResult {
                auth: super::types::ModelAuth {
                    api_key: Some(key.to_owned()),
                    ..Default::default()
                },
                env: credential.and_then(|credential| credential.env.clone()),
                source: Some("stored credential".to_owned()),
            }));
        }
        for env_var in &self.env_vars {
            let value = ctx.env(env_var).await;
            if token.is_cancelled() {
                return Err("aborted".into());
            }
            if let Some(value) = value {
                return Ok(Some(AuthResult {
                    auth: super::types::ModelAuth {
                        api_key: Some(value.clone()),
                        ..Default::default()
                    },
                    env: None,
                    source: Some(env_var.clone()),
                }));
            }
        }
        Ok(None)
    }
}

/// Ambient api-key auth: never interactive, resolves only from env vars,
/// used for keyless/local servers and ambient credential sources.
pub fn ambient_api_key_auth(name: impl Into<String>, env_vars: Vec<String>) -> AmbientApiKeyAuth {
    AmbientApiKeyAuth {
        auth: env_api_key_auth(name, env_vars),
    }
}

pub struct AmbientApiKeyAuth {
    auth: EnvApiKeyAuth,
}

#[async_trait]
impl ApiKeyAuth for AmbientApiKeyAuth {
    fn name(&self) -> &str {
        self.auth.name()
    }

    async fn check(
        &self,
        ctx: &dyn AuthContext,
        credential: Option<&ApiKeyCredential>,
        token: &CancellationToken,
    ) -> Result<Option<AuthCheck>, Box<dyn std::error::Error + Send + Sync>> {
        let resolved = ApiKeyAuth::resolve(&self.auth, ctx, credential, token).await?;
        Ok(resolved.map(|result| AuthCheck {
            source: result.source,
            kind: Some(CredentialKind::ApiKey),
        }))
    }

    async fn resolve(
        &self,
        ctx: &dyn AuthContext,
        credential: Option<&ApiKeyCredential>,
        token: &CancellationToken,
    ) -> Result<Option<AuthResult>, Box<dyn std::error::Error + Send + Sync>> {
        ApiKeyAuth::resolve(&self.auth, ctx, credential, token).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::context::DefaultAuthContext;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn test_context(vars: &[(&str, &str)]) -> DefaultAuthContext {
        let overlay: HashMap<String, String> = vars
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect();
        DefaultAuthContext::with_overlay(overlay)
    }

    #[tokio::test]
    async fn stored_key_wins_over_env() {
        let auth = env_api_key_auth("Test key", vec!["TEST_KEY_A".into(), "TEST_KEY_B".into()]);
        let ctx = test_context(&[("TEST_KEY_B", "from-env")]);
        let credential = ApiKeyCredential {
            key: Some("stored".to_owned()),
            env: None,
        };
        let result = auth
            .resolve(&ctx, Some(&credential), &CancellationToken::new())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(result.auth.api_key.as_deref(), Some("stored"));
        assert_eq!(result.source.as_deref(), Some("stored credential"));
    }

    #[tokio::test]
    async fn first_matching_env_var_resolves() {
        let auth = env_api_key_auth("Test key", vec!["TEST_KEY_A".into(), "TEST_KEY_B".into()]);
        let ctx = test_context(&[("TEST_KEY_B", "from-env")]);
        let result = auth
            .resolve(&ctx, None, &CancellationToken::new())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(result.auth.api_key.as_deref(), Some("from-env"));
        assert_eq!(result.source.as_deref(), Some("TEST_KEY_B"));
    }

    #[tokio::test]
    async fn unconfigured_returns_none() {
        let auth = env_api_key_auth("Test key", vec!["TEST_KEY_A".into()]);
        let ctx = DefaultAuthContext::new();
        assert!(
            auth.resolve(&ctx, None, &CancellationToken::new())
                .await
                .unwrap()
                .is_none()
        );
    }

    struct ScriptedInteraction {
        answer: String,
    }

    #[async_trait]
    impl AuthInteraction for ScriptedInteraction {
        async fn prompt(
            &self,
            _prompt: AuthPrompt,
        ) -> Result<String, crate::auth::types::AuthPromptCancelled> {
            Ok(self.answer.clone())
        }
        fn notify(&self, _event: crate::auth::types::AuthEvent) {}
        fn signal(&self) -> CancellationToken {
            CancellationToken::new()
        }
    }

    #[tokio::test]
    async fn login_prompts_for_key() {
        let auth = env_api_key_auth("Test key", vec![]);
        let interaction = Arc::new(ScriptedInteraction {
            answer: "typed-key".to_owned(),
        });
        let credential = auth.login(interaction.as_ref()).await.unwrap();
        assert_eq!(credential.key.as_deref(), Some("typed-key"));
    }
}
