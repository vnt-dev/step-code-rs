//! The model catalog: builds the provider collection from `models.json` plus
//! the built-in Step provider, and answers "which models exist / are usable".
//!
//! Ported functionally from the provider-composition half of
//! `@step-harness/coding-agent` `src/core/model-runtime.ts` and
//! `src/core/provider-composer.ts`.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use step_harness_providers::auth::{
    ApiKeyAuth, ApiKeyCredential, AuthContext, AuthResult, CredentialStore, ModelAuth, ProviderAuth,
};
use step_harness_providers::compat::{register_builtin_api_providers, registered_api_ids};
use step_harness_providers::models::{
    ApiDispatch, CreateModelsOptions, CreateProviderOptions, Models, ModelsRefreshOptions,
    ProviderStreams, create_provider,
};
use step_harness_providers::models_store::ModelsStore;
use step_harness_providers::step_provider::{StepProviderOptions, step_provider};
use step_harness_providers::types::{Model, ProviderHeaders, ProviderId};
use tokio_util::sync::CancellationToken;

use crate::auth::FileCredentialStore;
use crate::models::config::ModelsConfigFile;
use crate::models::store::FileModelsStore;
use crate::paths::AppPaths;

/// API-key auth backed by a static key from `models.json`.
struct StaticApiKeyAuth {
    key: String,
}

#[async_trait::async_trait]
impl ApiKeyAuth for StaticApiKeyAuth {
    fn name(&self) -> &str {
        "API key"
    }

    async fn resolve(
        &self,
        _ctx: &dyn AuthContext,
        _credential: Option<&ApiKeyCredential>,
        _token: &tokio_util::sync::CancellationToken,
    ) -> Result<Option<AuthResult>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Some(AuthResult {
            auth: ModelAuth {
                api_key: Some(self.key.clone()),
                headers: None,
                base_url: None,
            },
            env: None,
            source: Some("models.json".to_owned()),
        }))
    }
}

/// API-key auth that reads the provider's conventional environment variable.
struct EnvApiKeyAuth {
    env_var: String,
}

#[async_trait::async_trait]
impl ApiKeyAuth for EnvApiKeyAuth {
    fn name(&self) -> &str {
        "API key"
    }

    async fn resolve(
        &self,
        _ctx: &dyn AuthContext,
        credential: Option<&ApiKeyCredential>,
        _token: &tokio_util::sync::CancellationToken,
    ) -> Result<Option<AuthResult>, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(key) = credential
            .and_then(|value| value.key.as_deref())
            .map(str::trim)
            .filter(|key| !key.is_empty())
        {
            return Ok(Some(AuthResult {
                auth: ModelAuth {
                    api_key: Some(key.to_owned()),
                    headers: None,
                    base_url: None,
                },
                env: None,
                source: Some("stored credential".to_owned()),
            }));
        }
        match std::env::var(&self.env_var)
            .ok()
            .filter(|key| !key.trim().is_empty())
        {
            Some(key) => Ok(Some(AuthResult {
                auth: ModelAuth {
                    api_key: Some(key),
                    headers: None,
                    base_url: None,
                },
                env: None,
                source: Some(self.env_var.clone()),
            })),
            None => Ok(None),
        }
    }
}

/// The conventional environment variable name for a provider's API key.
pub fn env_api_key_var(provider_id: &str) -> String {
    format!(
        "{}_API_KEY",
        provider_id
            .to_ascii_uppercase()
            .replace(['-', '.', ' '], "_")
    )
}

/// A snapshot of provider/model availability for status UI.
#[derive(Clone, Debug, Default)]
pub struct CatalogSnapshot {
    /// Every model known to the catalog, authenticated or not.
    pub all_models: Vec<Model>,
    /// Models whose provider has complete auth configuration.
    pub available_models: Vec<Model>,
    /// Provider ids with complete auth configuration.
    pub configured_providers: HashSet<String>,
}

/// Owns the provider collection and answers model queries.
pub struct ModelCatalog {
    models: Arc<Models>,
    snapshot: std::sync::RwLock<CatalogSnapshot>,
}

impl std::fmt::Debug for ModelCatalog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelCatalog")
            .field("providers", &self.models.get_providers().len())
            .finish()
    }
}

impl ModelCatalog {
    /// Build the catalog from `models.json` and the built-in Step provider.
    ///
    /// Returns the catalog plus load diagnostics (unreadable/invalid config).
    pub async fn build(
        paths: &AppPaths,
        credentials: Arc<dyn CredentialStore>,
        models_store: Arc<dyn ModelsStore>,
    ) -> (Self, Vec<String>) {
        register_builtin_api_providers();
        let mut diagnostics = Vec::new();

        let config = match ModelsConfigFile::load(&paths.models_file()) {
            Ok(config) => config,
            Err(message) => {
                diagnostics.push(message);
                ModelsConfigFile::default()
            }
        };

        let models = Models::new(CreateModelsOptions {
            credentials: Some(credentials),
            models_store: Some(models_store),
            auth_context: None,
        });

        // Declared providers dispatch through the api registry, which holds
        // every registered dialect implementation (builtin apis plus anything
        // the embedder registered, such as test doubles).
        // `ApiDispatch` is not `Clone`, so a fresh one is built per provider.
        let build_dispatch = || {
            let mut api_dispatch: HashMap<String, Arc<dyn ProviderStreams>> = HashMap::new();
            for api in registered_api_ids() {
                if let Some(streams) = step_harness_providers::compat::get_api_provider(&api) {
                    api_dispatch.insert(api, streams);
                }
            }
            ApiDispatch::ByApi(api_dispatch)
        };

        for (provider_id, entry) in &config.providers {
            let provider_models = config.models_for(provider_id);
            let base_url = entry
                .base_url
                .clone()
                .unwrap_or_else(|| "https://api.openai.com/v1".to_owned());
            let headers: Option<ProviderHeaders> = entry.headers.as_ref().map(|headers| {
                headers
                    .iter()
                    .map(|(name, value)| (name.clone(), Some(value.clone())))
                    .collect()
            });
            let auth = match &entry.api_key {
                Some(key) if !key.trim().is_empty() => {
                    ProviderAuth::with_api_key(StaticApiKeyAuth { key: key.clone() })
                }
                _ => ProviderAuth::with_api_key(EnvApiKeyAuth {
                    env_var: env_api_key_var(provider_id),
                }),
            };
            let provider = create_provider(CreateProviderOptions {
                id: ProviderId::new(provider_id),
                name: entry.name.clone(),
                base_url: Some(base_url),
                headers,
                auth,
                models: provider_models,
                fetch_models: None,
                filter_models: None,
                api: build_dispatch(),
            });
            models.set_provider(provider);
        }

        // The Step provider ships built in (OAuth device login + STEP_API_KEY).
        let step_options = StepProviderOptions {
            env: None,
            ..Default::default()
        };
        match step_provider(&step_options) {
            Ok(provider) => {
                models.set_provider(provider);
            }
            Err(error) => diagnostics.push(format!("step provider: {error}")),
        }

        let catalog = ModelCatalog {
            models: Arc::new(models),
            snapshot: std::sync::RwLock::new(CatalogSnapshot::default()),
        };
        // Cache-only refresh: restores model catalogs persisted by earlier
        // network refreshes (e.g. Step models fetched at login) so a restart
        // keeps authenticated providers usable without network access.
        diagnostics.append(&mut catalog.refresh(false, &CancellationToken::new()).await);
        (catalog, diagnostics)
    }

    /// A shared handle to the provider collection, for streaming requests.
    pub fn models_handle(&self) -> Arc<Models> {
        self.models.clone()
    }

    /// The provider collection, for streaming requests.
    pub fn models(&self) -> &Models {
        &self.models
    }

    /// The current availability snapshot.
    pub fn snapshot(&self) -> CatalogSnapshot {
        self.snapshot
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Recompute availability (no network).
    pub async fn refresh_snapshot(&self, token: &CancellationToken) {
        let all_models: Vec<Model> = self
            .models
            .get_providers()
            .into_iter()
            .flat_map(|provider| provider.get_models())
            .collect();
        let available_models = self
            .models
            .get_available(None, Some(token.clone()))
            .await
            .unwrap_or_default();
        let mut configured_providers = HashSet::new();
        for provider in self.models.get_providers() {
            if let Ok(Some(_check)) = self
                .models
                .check_auth(provider.id.as_str(), Some(token.clone()))
                .await
            {
                configured_providers.insert(provider.id.to_string());
            }
        }
        let snapshot = CatalogSnapshot {
            all_models,
            available_models,
            configured_providers,
        };
        *self
            .snapshot
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = snapshot;
    }

    /// Refresh remote catalogs when network is allowed, then recompute.
    pub async fn refresh(&self, allow_network: bool, token: &CancellationToken) -> Vec<String> {
        let result = self
            .models
            .refresh(ModelsRefreshOptions {
                allow_network: Some(allow_network),
                providers: None,
                force: false,
                signal: Some(token.clone()),
            })
            .await;
        self.refresh_snapshot(token).await;
        result.errors.into_values().collect()
    }

    /// Find a model by provider and id, authenticated or not.
    pub fn find(&self, provider: &str, id: &str) -> Option<Model> {
        self.snapshot()
            .all_models
            .into_iter()
            .find(|model| model.provider.as_str() == provider && model.id == id)
    }

    /// Whether a provider has complete auth configuration.
    pub fn is_provider_configured(&self, provider: &str) -> bool {
        self.snapshot().configured_providers.contains(provider)
    }

    /// Filter models by settings globs (`provider/id`, bare `id`, wildcards).
    ///
    /// An empty pattern list selects every available model. Patterns may carry
    /// a `:thinking-level` suffix, which is stripped before matching.
    pub fn scoped_models(&self, patterns: &[String]) -> Vec<Model> {
        let snapshot = self.snapshot();
        let matchers: Vec<globset::GlobMatcher> = patterns
            .iter()
            .map(|pattern| strip_thinking_suffix(pattern))
            .filter_map(|pattern| {
                globset::GlobBuilder::new(&pattern)
                    .case_insensitive(true)
                    .literal_separator(false)
                    .build()
                    .ok()
                    .map(|glob| glob.compile_matcher())
            })
            .collect();
        if matchers.is_empty() {
            return snapshot.available_models;
        }
        snapshot
            .available_models
            .into_iter()
            .filter(|model| {
                let reference = format!("{}/{}", model.provider, model.id);
                matchers
                    .iter()
                    .any(|matcher| matcher.is_match(&reference) || matcher.is_match(&model.id))
            })
            .collect()
    }
}

/// Strip a trailing `:thinking-level` suffix from a model pattern.
pub fn strip_thinking_suffix(pattern: &str) -> String {
    match pattern.rsplit_once(':') {
        Some((head, suffix)) => {
            let is_level = matches!(
                suffix.to_ascii_lowercase().as_str(),
                "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max"
            );
            if is_level {
                head.to_owned()
            } else {
                pattern.to_owned()
            }
        }
        None => pattern.to_owned(),
    }
}

/// Convenience constructor wiring the file-backed stores.
pub fn default_stores(paths: &AppPaths) -> (Arc<FileCredentialStore>, Arc<FileModelsStore>) {
    (
        Arc::new(FileCredentialStore::new(paths.auth_file())),
        Arc::new(FileModelsStore::new(paths.models_store_file())),
    )
}

/// Provider ids in a stable order, for status UI.
pub fn provider_ids(catalog: &ModelCatalog) -> Vec<String> {
    let mut ids: Vec<String> = catalog
        .models()
        .get_providers()
        .into_iter()
        .map(|provider| provider.id.to_string())
        .collect();
    ids.sort();
    ids.dedup();
    ids
}

/// A `Provider` reference re-export for callers inspecting the collection.
pub use step_harness_providers::models::Provider as CatalogProvider;

/// Override map type used by callers composing provider entries.
pub type ProviderMap<T> = BTreeMap<String, T>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_var_names_are_conventional() {
        assert_eq!(env_api_key_var("my-openai"), "MY_OPENAI_API_KEY");
        assert_eq!(env_api_key_var("step"), "STEP_API_KEY");
    }

    #[test]
    fn strips_thinking_suffixes() {
        assert_eq!(
            strip_thinking_suffix("anthropic/claude:high"),
            "anthropic/claude"
        );
        assert_eq!(
            strip_thinking_suffix("anthropic/claude"),
            "anthropic/claude"
        );
        // A non-level suffix is part of the pattern.
        assert_eq!(strip_thinking_suffix("a:b"), "a:b");
    }

    #[tokio::test]
    async fn build_restores_persisted_models_without_network() {
        use step_harness_providers::models_store::ModelsStoreEntry;
        let directory = tempfile::tempdir().unwrap();
        let paths = AppPaths::new(directory.path());
        let (credentials, store) = default_stores(&paths);
        let cached: step_harness_providers::types::Model =
            serde_json::from_value(serde_json::json!({
                "id": "step-cached",
                "name": "Step Cached",
                "api": "openai-completions",
                "provider": "step",
                "baseUrl": "https://api.example.test/v1",
                "reasoning": false,
                "input": ["text"],
                "contextWindow": 4096,
                "maxTokens": 1024,
            }))
            .unwrap();
        store
            .write(
                "step",
                ModelsStoreEntry {
                    models: vec![cached],
                    last_modified: None,
                    checked_at: None,
                    etag: None,
                },
                &CancellationToken::new(),
            )
            .await
            .unwrap();
        let (catalog, diagnostics) = ModelCatalog::build(&paths, credentials, store).await;
        assert!(diagnostics.is_empty(), "{diagnostics:?}");
        assert!(
            catalog
                .snapshot()
                .all_models
                .iter()
                .any(|model| model.provider.as_str() == "step" && model.id == "step-cached")
        );
    }
}
