//! The provider runtime: [`Provider`] (the concrete runtime unit) and
//! [`Models`] (the collection resolving auth and dispatching streams).

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::auth::context::{AuthContext, DefaultAuthContext};
use crate::auth::credential_store::InMemoryCredentialStore;
use crate::auth::resolve::{AuthResolutionOverrides, resolve_provider_auth};
use crate::auth::types::{
    AuthCheck, AuthInteraction, AuthResult, AuthType, Credential, CredentialKind, CredentialStore,
};
use crate::error::{ModelsError, ModelsErrorCode};
use crate::event::{AssistantMessageEventStream, lazy_stream};
use crate::models_store::{InMemoryModelsStore, ModelsStore, ModelsStoreEntry};
use crate::types::{
    Api, AssistantMessage, Context, DeferredHandle, Model, ModelThinkingLevel, ProviderEnv,
    ProviderHeaders, ProviderId, SimpleStreamOptions, StreamOptions, Usage,
};
use crate::utils::headers::merge_provider_headers;

/// The uniform stream contract of an API implementation: `stream` and
/// `stream_simple` are required; deferred responses are optional.
#[async_trait]
pub trait ProviderStreams: Send + Sync {
    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: &StreamOptions,
    ) -> AssistantMessageEventStream;
    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
    ) -> AssistantMessageEventStream;

    fn supports_fetch_deferred(&self) -> bool {
        false
    }

    fn fetch_deferred(
        &self,
        _model: &Model,
        _handle: &DeferredHandle,
        _options: &StreamOptions,
    ) -> AssistantMessageEventStream {
        unreachable!("supports_fetch_deferred() must guard fetch_deferred() calls")
    }

    async fn cancel_deferred(
        &self,
        _model: &Model,
        _handle: &DeferredHandle,
        _options: &StreamOptions,
    ) -> Result<(), ModelsError> {
        Err(ModelsError::new(
            ModelsErrorCode::Provider,
            "deferred cancellation unsupported",
        ))
    }
}

/// Publication chosen by a dynamic provider's refresh phase.
pub struct ModelsPublication {
    /// Provider-selected persisted catalog.
    pub persist: Persist,
    /// Synchronous in-memory catalog swap, run only after the persistence
    /// mutation was selected.
    pub update: Option<Box<dyn FnOnce() + Send>>,
}

/// Persistence intent: keep storage unchanged, store a new entry, or delete it.
pub enum Persist {
    Keep,
    Store(ModelsStoreEntry),
    Delete,
}

impl Default for ModelsPublication {
    fn default() -> Self {
        ModelsPublication {
            persist: Persist::Keep,
            update: None,
        }
    }
}

/// Context passed to a provider's model refresh.
pub struct RefreshModelsContext {
    /// Effective configured credential. OAuth credentials are refreshed before
    /// network access.
    pub credential: Option<Credential>,
    /// Immutable provider-scoped catalog snapshot captured before this phase.
    pub stored: Option<ModelsStoreEntry>,
    /// Generation-checked publication. Persistence policy remains
    /// provider-owned; the update runs only after the persistence mutation.
    publish: PublisherFn,
    /// False during offline/cache-only initialization.
    pub allow_network: bool,
    /// Bypass provider freshness checks and fetch immediately when network
    /// access is allowed.
    pub force: bool,
    /// Always present, including when the public refresh caller omits its token.
    pub signal: CancellationToken,
}

pub(crate) type PublisherFn =
    Arc<dyn Fn(ModelsPublication) -> futures::future::BoxFuture<'static, bool> + Send + Sync>;

impl RefreshModelsContext {
    /// Construct a refresh context. `publish` is the generation-checked
    /// publisher the collection installs.
    pub fn new(
        credential: Option<Credential>,
        stored: Option<ModelsStoreEntry>,
        publish: PublisherFn,
        allow_network: bool,
        force: bool,
        signal: CancellationToken,
    ) -> Self {
        RefreshModelsContext {
            credential,
            stored,
            publish,
            allow_network,
            force,
            signal,
        }
    }

    pub async fn publish(&self, publication: ModelsPublication) -> bool {
        (self.publish)(publication).await
    }
}

/// Standard dynamic-catalog fetch input: the raw per-provider fetch hook used
/// by `create_provider`; restore/publish orchestration lives with the
/// collection.
pub type FetchModelsFn = Arc<
    dyn Fn(
            RefreshModelsContext,
        ) -> futures::future::BoxFuture<'static, Result<Vec<Model>, ModelsError>>
        + Send
        + Sync,
>;

/// Optional provider policy for credential-specific model availability.
pub type FilterModelsFn = Arc<dyn Fn(&[Model], Option<&Credential>) -> Vec<Model> + Send + Sync>;

/// API dispatch for a provider: one implementation for all models, or a map
/// keyed by `model.api`.
pub enum ApiDispatch {
    Single(Arc<dyn ProviderStreams>),
    ByApi(HashMap<String, Arc<dyn ProviderStreams>>),
}

impl ApiDispatch {
    fn streams_for(&self, api: &Api) -> Option<Arc<dyn ProviderStreams>> {
        match self {
            ApiDispatch::Single(streams) => Some(streams.clone()),
            ApiDispatch::ByApi(by_api) => by_api.get(api.as_str()).cloned(),
        }
    }

    fn any_supports_deferred(&self) -> bool {
        match self {
            ApiDispatch::Single(streams) => streams.supports_fetch_deferred(),
            ApiDispatch::ByApi(by_api) => by_api
                .values()
                .any(|streams| streams.supports_fetch_deferred()),
        }
    }
}

#[derive(Default)]
struct ProviderState {
    baseline_models: Vec<Model>,
    dynamic_models: Vec<Model>,
}

impl ProviderState {
    fn current_models(&self) -> Vec<Model> {
        let mut merged = self.baseline_models.clone();
        for model in &self.dynamic_models {
            match merged.iter_mut().find(|entry| entry.id == model.id) {
                Some(existing) => *existing = model.clone(),
                None => merged.push(model.clone()),
            }
        }
        merged
    }
}

/// A provider is the concrete runtime unit: id/name/base metadata, auth
/// methods, model listing, and stream behavior.
pub struct Provider {
    pub id: ProviderId,
    pub name: String,
    pub base_url: Option<String>,
    pub headers: Option<ProviderHeaders>,
    /// At least one of api-key/oauth auth; ambient/keyless providers use an
    /// ambient api-key auth whose `resolve` reports whether they are configured.
    pub auth: crate::auth::ProviderAuth,
    state: std::sync::RwLock<ProviderState>,
    api: ApiDispatch,
    fetch_models: Option<FetchModelsFn>,
    filter_models: Option<FilterModelsFn>,
}

impl Provider {
    /// Current known models, sync. Dynamic providers return the list as of the
    /// last refresh (empty before the first).
    pub fn get_models(&self) -> Vec<Model> {
        self.state
            .read()
            .expect("provider state poisoned")
            .current_models()
    }

    /// Replace the dynamic overlay (run from a refresh publication update).
    pub fn set_dynamic_models(&self, models: Vec<Model>) {
        self.state
            .write()
            .expect("provider state poisoned")
            .dynamic_models = models;
    }

    pub fn has_refresh(&self) -> bool {
        self.fetch_models.is_some()
    }

    fn fetch_models_handle(&self) -> Option<FetchModelsFn> {
        self.fetch_models.clone()
    }

    pub fn supports_fetch_deferred(&self) -> bool {
        self.api.any_supports_deferred()
    }

    fn streams_for(&self, model: &Model) -> Option<Arc<dyn ProviderStreams>> {
        self.api.streams_for(&model.api)
    }

    /// Apply the credential-specific availability filter.
    pub fn filter_models(&self, models: Vec<Model>, credential: Option<&Credential>) -> Vec<Model> {
        match &self.filter_models {
            Some(filter) => filter(&models, credential),
            None => models,
        }
    }
}

impl std::fmt::Debug for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Provider")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("models", &self.get_models().len())
            .finish()
    }
}

/// Inputs to [`create_provider`].
pub struct CreateProviderOptions {
    pub id: ProviderId,
    /// Display name. Default: `id`.
    pub name: Option<String>,
    pub base_url: Option<String>,
    pub headers: Option<ProviderHeaders>,
    /// Required — every provider has auth semantics, even ambient/keyless ones.
    pub auth: crate::auth::ProviderAuth,
    /// Static baseline model list (empty for purely dynamic providers).
    pub models: Vec<Model>,
    /// Fetch a dynamic model overlay; `create_provider` restores and publishes
    /// it transactionally.
    pub fetch_models: Option<FetchModelsFn>,
    pub filter_models: Option<FilterModelsFn>,
    /// Single implementation, or map keyed by `model.api` for mixed-API providers.
    pub api: ApiDispatch,
}

/// Build a provider from parts: built-in provider factories and models.json
/// custom providers both go through this. A model whose api has no entry
/// produces a stream error.
pub fn create_provider(input: CreateProviderOptions) -> Arc<Provider> {
    let name = input.name.unwrap_or_else(|| input.id.to_string());
    Arc::new(Provider {
        id: input.id,
        name,
        base_url: input.base_url,
        headers: input.headers,
        auth: input.auth,
        state: std::sync::RwLock::new(ProviderState {
            baseline_models: input.models,
            dynamic_models: Vec::new(),
        }),
        api: input.api,
        fetch_models: input.fetch_models,
        filter_models: input.filter_models,
    })
}

/// Options for [`Models::new`].
pub struct CreateModelsOptions {
    pub credentials: Option<Arc<dyn CredentialStore>>,
    pub models_store: Option<Arc<dyn ModelsStore>>,
    pub auth_context: Option<Arc<dyn AuthContext>>,
}

impl Default for CreateModelsOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl CreateModelsOptions {
    pub fn new() -> Self {
        CreateModelsOptions {
            credentials: None,
            models_store: None,
            auth_context: None,
        }
    }
}

/// Options for [`Models::refresh`].
#[derive(Clone, Debug, Default)]
pub struct ModelsRefreshOptions {
    pub allow_network: Option<bool>,
    /// Restrict refresh to these provider ids. Unknown and static providers are ignored.
    pub providers: Option<Vec<String>>,
    /// Bypass provider freshness checks and fetch immediately when network
    /// access is allowed.
    pub force: bool,
    pub signal: Option<CancellationToken>,
}

#[derive(Clone, Debug, Default)]
pub struct ModelsRefreshResult {
    pub aborted: bool,
    pub errors: HashMap<String, String>,
}

struct RefreshControl {
    generation: u64,
    owner: Arc<()>,
    canceller: CancellationToken,
    publication_lock: Arc<tokio::sync::Mutex<()>>,
}

struct ModelsInner {
    providers: std::sync::RwLock<HashMap<ProviderId, Arc<Provider>>>,
    credentials: Arc<dyn CredentialStore>,
    models_store: Arc<dyn ModelsStore>,
    auth_context: Arc<dyn AuthContext>,
    refresh_state: std::sync::Mutex<HashMap<String, RefreshControl>>,
}

/// Runtime collection of providers plus auth application and stream
/// convenience. A cheap clonable handle over shared state; providers own
/// stream behavior while `Models` resolves auth and delegates each request to
/// the provider that owns the model.
#[derive(Clone)]
pub struct Models {
    inner: Arc<ModelsInner>,
}

/// Reference passed to [`Models::get_auth`].
pub enum ProviderOrModel<'a> {
    ProviderId(&'a str),
    Model(&'a Model),
}

/// Auth resolved and ready to merge into request options.
struct AppliedAuth {
    model: Model,
    api_key: Option<String>,
    headers: Option<ProviderHeaders>,
    env: Option<ProviderEnv>,
}

impl Models {
    pub fn new(options: CreateModelsOptions) -> Self {
        Models {
            inner: Arc::new(ModelsInner {
                providers: std::sync::RwLock::new(HashMap::new()),
                credentials: options
                    .credentials
                    .unwrap_or_else(|| Arc::new(InMemoryCredentialStore::new())),
                models_store: options
                    .models_store
                    .unwrap_or_else(|| Arc::new(InMemoryModelsStore::new())),
                auth_context: options
                    .auth_context
                    .unwrap_or_else(|| Arc::new(DefaultAuthContext::new())),
                refresh_state: std::sync::Mutex::new(HashMap::new()),
            }),
        }
    }

    /// Build and wrap in one call (stream dispatch requires an `Arc`-free clone
    /// handle, which `Models` already is).
    pub fn shared(options: CreateModelsOptions) -> Self {
        Models::new(options)
    }

    /// Upsert/replace by provider id. Provider ids are unique.
    pub fn set_provider(&self, provider: Arc<Provider>) {
        self.supersede_refresh(provider.id.as_str());
        self.inner
            .providers
            .write()
            .expect("providers poisoned")
            .insert(provider.id.clone(), provider);
    }

    pub fn delete_provider(&self, id: &str) {
        self.supersede_refresh(id);
        self.inner
            .providers
            .write()
            .expect("providers poisoned")
            .remove(id);
    }

    pub fn clear_providers(&self) {
        let ids: Vec<String> = {
            let providers = self.inner.providers.read().expect("providers poisoned");
            let mut ids: Vec<String> = providers.keys().map(|id| id.to_string()).collect();
            drop(providers);
            ids.extend(
                self.inner
                    .refresh_state
                    .lock()
                    .expect("refresh state poisoned")
                    .keys()
                    .cloned(),
            );
            ids
        };
        for id in ids {
            self.supersede_refresh(&id);
        }
        self.inner
            .providers
            .write()
            .expect("providers poisoned")
            .clear();
    }

    fn supersede_refresh(&self, provider_id: &str) {
        if let Some(control) = self
            .inner
            .refresh_state
            .lock()
            .expect("refresh state poisoned")
            .get_mut(provider_id)
        {
            control.generation += 1;
            control.publication_lock = Arc::new(tokio::sync::Mutex::new(()));
            control.canceller.cancel();
        }
    }

    fn begin_refresh(&self, provider_id: &str) -> (u64, Arc<()>, CancellationToken) {
        let mut state = self
            .inner
            .refresh_state
            .lock()
            .expect("refresh state poisoned");
        let generation = state
            .get(provider_id)
            .map_or(0, |control| control.generation + 1);
        let owner = Arc::new(());
        let canceller = CancellationToken::new();
        state.insert(
            provider_id.to_owned(),
            RefreshControl {
                generation,
                owner: owner.clone(),
                canceller: canceller.clone(),
                publication_lock: Arc::new(tokio::sync::Mutex::new(())),
            },
        );
        (generation, owner, canceller)
    }

    fn finish_refresh(&self, provider_id: &str, owner: &Arc<()>) {
        let mut state = self
            .inner
            .refresh_state
            .lock()
            .expect("refresh state poisoned");
        if state
            .get(provider_id)
            .is_some_and(|control| Arc::ptr_eq(&control.owner, owner))
        {
            state.remove(provider_id);
        }
    }

    pub fn get_providers(&self) -> Vec<Arc<Provider>> {
        self.inner
            .providers
            .read()
            .expect("providers poisoned")
            .values()
            .cloned()
            .collect()
    }

    pub fn get_provider(&self, id: &str) -> Option<Arc<Provider>> {
        self.inner
            .providers
            .read()
            .expect("providers poisoned")
            .get(id)
            .cloned()
    }

    /// Sync read of last-known models from one provider or all providers.
    pub fn get_models(&self, provider: Option<&str>) -> Vec<Model> {
        let providers = self.inner.providers.read().expect("providers poisoned");
        match provider {
            Some(provider) => providers
                .get(provider)
                .map(|entry| entry.get_models())
                .unwrap_or_default(),
            None => providers
                .values()
                .flat_map(|entry| entry.get_models())
                .collect(),
        }
    }

    /// Sync runtime model lookup against last-known lists.
    pub fn get_model(&self, provider: &str, id: &str) -> Option<Model> {
        self.get_models(Some(provider))
            .into_iter()
            .find(|model| model.id == id)
    }

    /// Refresh selected configured dynamic providers concurrently (all when
    /// `options.providers` is omitted). Provider errors and cancellation are
    /// reported in the result instead of failing the call.
    pub async fn refresh(&self, options: ModelsRefreshOptions) -> ModelsRefreshResult {
        let allow_network = options.allow_network.unwrap_or(true);
        let caller_token = options.signal.clone().unwrap_or_default();
        let mut errors: HashMap<String, String> = HashMap::new();
        if caller_token.is_cancelled() {
            return ModelsRefreshResult {
                aborted: true,
                errors,
            };
        }
        let selected: Option<std::collections::HashSet<String>> = options
            .providers
            .map(|providers| providers.into_iter().collect());
        let refreshable: Vec<Arc<Provider>> = self
            .get_providers()
            .into_iter()
            .filter(|provider| {
                provider.has_refresh()
                    && selected
                        .as_ref()
                        .is_none_or(|set| set.contains(provider.id.as_str()))
            })
            .collect();

        let mut tasks = Vec::new();
        for provider in refreshable {
            let (generation, owner, control_canceller) = self.begin_refresh(provider.id.as_str());
            let models = self.clone();
            let caller_token = caller_token.clone();
            tasks.push(tokio::spawn(async move {
                let combined =
                    crate::utils::combine_abort_tokens(&[&caller_token, &control_canceller]);
                let outcome = models
                    .run_provider_refresh(
                        &provider,
                        &combined.token,
                        allow_network,
                        options.force,
                        generation,
                    )
                    .await;
                (provider.id.to_string(), outcome, owner)
            }));
        }

        let mut aborted = false;
        for task in tasks {
            let Some((provider_id, outcome, owner)) = task.await.ok() else {
                continue;
            };
            match outcome {
                Ok(()) => {}
                Err(error) => {
                    errors.insert(provider_id.clone(), error.message().to_owned());
                }
            }
            if caller_token.is_cancelled() {
                aborted = true;
            }
            self.finish_refresh(&provider_id, &owner);
        }

        ModelsRefreshResult { aborted, errors }
    }

    async fn run_provider_refresh(
        &self,
        provider: &Arc<Provider>,
        signal: &CancellationToken,
        allow_network: bool,
        force: bool,
        generation: u64,
    ) -> Result<(), ModelsError> {
        let stored_credential = match self.read_credential(provider.id.as_str(), signal).await {
            Ok(credential) => credential,
            Err(error) => {
                // Still restore cached state before surfacing the auth error.
                self.run_refresh_phase(provider, None, false, false, generation, signal)
                    .await?;
                return Err(error);
            }
        };

        // Restore cached provider state before auth resolution or network access.
        self.run_refresh_phase(
            provider,
            stored_credential.as_ref(),
            false,
            false,
            generation,
            signal,
        )
        .await?;
        if !allow_network || signal.is_cancelled() {
            return Ok(());
        }

        let credential = self
            .resolve_refresh_credential(provider, stored_credential.as_ref(), signal)
            .await?;
        let Some(credential) = credential else {
            return Ok(());
        };
        self.run_refresh_phase(provider, Some(&credential), true, force, generation, signal)
            .await
    }

    async fn run_refresh_phase(
        &self,
        provider: &Arc<Provider>,
        credential: Option<&Credential>,
        allow_network: bool,
        force: bool,
        generation: u64,
        signal: &CancellationToken,
    ) -> Result<(), ModelsError> {
        let Some(fetch) = provider.fetch_models_handle() else {
            return Ok(());
        };
        let stored = self
            .inner
            .models_store
            .read(provider.id.as_str(), signal)
            .await
            .map_err(|error| {
                ModelsError::with_source(
                    ModelsErrorCode::ModelSource,
                    "models store read failed",
                    error,
                )
            })?;

        let publish = self.make_publisher(provider.clone(), generation, signal.child_token());
        let context = RefreshModelsContext {
            credential: credential.cloned(),
            stored,
            publish,
            allow_network,
            force: allow_network && force,
            signal: signal.clone(),
        };

        // Restore the persisted overlay first: a cache-only phase must still
        // surface previously fetched models.
        if let Some(stored_entry) = &context.stored {
            let restored = stored_entry.models_for_provider(provider.id.as_str());
            let provider_for_update = provider.clone();
            let published = context
                .publish(ModelsPublication {
                    persist: Persist::Keep,
                    update: Some(Box::new(move || {
                        provider_for_update.set_dynamic_models(restored)
                    })),
                })
                .await;
            if !published {
                return Ok(());
            }
        }
        if !context.allow_network || context.signal.is_cancelled() {
            return Ok(());
        }
        let refreshed = fetch(clone_context(&context)).await?;
        if context.signal.is_cancelled() {
            return Ok(());
        }
        let provider_for_update = provider.clone();
        let refreshed_for_update = refreshed.clone();
        context
            .publish(ModelsPublication {
                persist: Persist::Store(ModelsStoreEntry {
                    models: refreshed,
                    last_modified: None,
                    checked_at: Some(crate::utils::now_ms()),
                    etag: None,
                }),
                update: Some(Box::new(move || {
                    provider_for_update.set_dynamic_models(refreshed_for_update)
                })),
            })
            .await;
        Ok(())
    }

    fn make_publisher(
        &self,
        provider: Arc<Provider>,
        generation: u64,
        signal: CancellationToken,
    ) -> PublisherFn {
        let models_store = self.inner.models_store.clone();
        let refresh_state = Arc::downgrade(&self.inner);
        Arc::new(move |publication: ModelsPublication| {
            let refresh_state = refresh_state.clone();
            let models_store = models_store.clone();
            let provider = provider.clone();
            let signal = signal.clone();
            Box::pin(async move {
                let Some(inner) = refresh_state.upgrade() else {
                    return false;
                };
                let publication_lock = inner
                    .refresh_state
                    .lock()
                    .expect("refresh state poisoned")
                    .get(provider.id.as_str())
                    .map(|control| control.publication_lock.clone());
                let Some(publication_lock) = publication_lock else {
                    return false;
                };
                // Serialize publications per provider (the TS publication chain).
                let _guard = publication_lock.lock().await;
                if signal.is_cancelled()
                    || !generation_current(&inner, provider.id.as_str(), generation)
                {
                    return false;
                }
                let persist_token = CancellationToken::new();
                let persisted = match publication.persist {
                    Persist::Keep => true,
                    Persist::Delete => models_store
                        .delete(provider.id.as_str(), &persist_token)
                        .await
                        .is_ok(),
                    Persist::Store(entry) => models_store
                        .write(provider.id.as_str(), entry, &persist_token)
                        .await
                        .is_ok(),
                };
                if !persisted
                    || signal.is_cancelled()
                    || !generation_current(&inner, provider.id.as_str(), generation)
                {
                    return false;
                }
                if let Some(update) = publication.update {
                    update();
                }
                true
            })
        })
    }

    async fn resolve_refresh_credential(
        &self,
        provider: &Arc<Provider>,
        stored: Option<&Credential>,
        signal: &CancellationToken,
    ) -> Result<Option<Credential>, ModelsError> {
        if let Some(Credential::Oauth(oauth_credential)) = stored {
            let Some(oauth) = provider.auth.oauth.clone() else {
                return Ok(None);
            };
            if crate::utils::now_ms() < oauth_credential.expires {
                return Ok(stored.cloned());
            }
            if signal.is_cancelled() {
                return Ok(None);
            }
            let oauth_for_refresh = oauth.clone();
            let refresh_signal = signal.clone();
            let post = self
                .inner
                .credentials
                .modify(
                    provider.id.as_str(),
                    crate::auth::types::modifier(move |current| {
                        let oauth_for_refresh = oauth_for_refresh.clone();
                        let refresh_signal = refresh_signal.clone();
                        Box::pin(async move {
                            let Some(Credential::Oauth(current)) = current else {
                                return Ok(None);
                            };
                            if crate::utils::now_ms() < current.expires {
                                return Ok(None);
                            }
                            let never = CancellationToken::new();
                            let refreshed = tokio::select! {
                                biased;
                                _ = refresh_signal.cancelled() => {
                                    return Ok(None);
                                }
                                result = oauth_for_refresh.refresh(&current, &never) => result,
                            };
                            match refreshed {
                                Ok(refreshed) => Ok(Some(Credential::Oauth(refreshed))),
                                Err(error) => Err(Box::new(ModelsError::with_source(
                                    ModelsErrorCode::Oauth,
                                    "OAuth refresh failed",
                                    error,
                                ))
                                    as Box<dyn std::error::Error + Send + Sync>),
                            }
                        })
                    }),
                    signal,
                )
                .await
                .map_err(|error| {
                    ModelsError::with_source(
                        ModelsErrorCode::Auth,
                        "credential store modify failed",
                        error,
                    )
                })?;
            return Ok(post.filter(|credential| matches!(credential, Credential::Oauth(_))));
        }

        let Some(api_key) = provider.auth.api_key.clone() else {
            return Ok(None);
        };
        let credential = match stored {
            Some(credential @ Credential::ApiKey(_)) => Some(credential.clone()),
            _ => None,
        };
        let result = api_key
            .resolve(
                self.inner.auth_context.as_ref(),
                credential
                    .as_ref()
                    .and_then(|credential| credential.as_api_key()),
                signal,
            )
            .await
            .map_err(|error| {
                ModelsError::with_source(ModelsErrorCode::Auth, "API key auth failed", error)
            })?;
        match result {
            Some(result) => Ok(Some(Credential::ApiKey(
                crate::auth::types::ApiKeyCredential {
                    key: result.auth.api_key,
                    env: result.env,
                },
            ))),
            None => Ok(None),
        }
    }

    async fn read_credential(
        &self,
        provider_id: &str,
        signal: &CancellationToken,
    ) -> Result<Option<Credential>, ModelsError> {
        self.inner
            .credentials
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

    async fn check_provider_auth(
        &self,
        provider: &Arc<Provider>,
        credential: Option<&Credential>,
        signal: &CancellationToken,
    ) -> Result<Option<AuthCheck>, ModelsError> {
        if let Some(Credential::Oauth(_)) = credential {
            return Ok(provider
                .auth
                .oauth
                .as_ref()
                .map(|_| AuthCheck::configured("OAuth", CredentialKind::Oauth)));
        }
        let Some(api_key) = provider.auth.api_key.clone() else {
            return Ok(None);
        };
        let stored_api_key = match credential {
            Some(credential @ Credential::ApiKey(_)) => Some(credential.clone()),
            _ => None,
        }
        .and_then(|credential| match credential {
            Credential::ApiKey(inner) => Some(inner),
            _ => None,
        });
        if let Some(check) = api_key
            .check(
                self.inner.auth_context.as_ref(),
                stored_api_key.as_ref(),
                signal,
            )
            .await
            .map_err(|error| {
                ModelsError::with_source(
                    ModelsErrorCode::Auth,
                    format!("API key auth check failed for provider {}", provider.id),
                    error,
                )
            })?
        {
            return Ok(Some(check));
        }

        let resolution = resolve_provider_auth(
            provider.id.as_str(),
            &provider.auth,
            self.inner.credentials.as_ref(),
            self.inner.auth_context.as_ref(),
            None,
        )
        .await?;
        Ok(resolution.map(|resolution| {
            AuthCheck::configured(
                resolution.source.unwrap_or_default(),
                CredentialKind::ApiKey,
            )
        }))
    }

    /// Check whether a provider has complete auth configuration without
    /// refreshing OAuth.
    pub async fn check_auth(
        &self,
        provider_id: &str,
        signal: Option<CancellationToken>,
    ) -> Result<Option<AuthCheck>, ModelsError> {
        let signal = signal.unwrap_or_default();
        if signal.is_cancelled() {
            return Err(ModelsError::new(ModelsErrorCode::Auth, "aborted"));
        }
        let Some(provider) = self.get_provider(provider_id) else {
            return Ok(None);
        };
        let credential = self.read_credential(provider_id, &signal).await?;
        crate::utils::race_with_abort(
            self.check_provider_auth(&provider, credential.as_ref(), &signal),
            &signal,
        )
        .await
        .map_err(|_| ModelsError::new(ModelsErrorCode::Auth, "aborted"))?
    }

    /// Return models whose providers have complete auth configuration.
    pub async fn get_available(
        &self,
        provider_id: Option<&str>,
        signal: Option<CancellationToken>,
    ) -> Result<Vec<Model>, ModelsError> {
        let signal = signal.unwrap_or_default();
        if signal.is_cancelled() {
            return Err(ModelsError::new(ModelsErrorCode::Auth, "aborted"));
        }
        let providers = match provider_id {
            Some(provider_id) => self
                .get_provider(provider_id)
                .into_iter()
                .collect::<Vec<_>>(),
            None => self.get_providers(),
        };
        let mut available = Vec::new();
        for provider in providers {
            let credential = self.read_credential(provider.id.as_str(), &signal).await?;
            let configured = self
                .check_provider_auth(&provider, credential.as_ref(), &signal)
                .await?;
            if configured.is_none() {
                continue;
            }
            let models = provider.get_models();
            available.extend(provider.filter_models(models, credential.as_ref()));
        }
        Ok(available)
    }

    /// Resolve provider-scoped auth by provider id, or provider auth plus static
    /// model headers when passed a model. Includes a source label for status UI.
    /// `Ok(None)` when the provider is unknown or unconfigured. Errors with
    /// [`ModelsError`] code `oauth` when a token refresh fails (the stored
    /// credential is preserved for retry) and `auth` when api-key resolution or
    /// the credential store fails.
    pub async fn get_auth(
        &self,
        provider_or_model: ProviderOrModel<'_>,
        overrides: Option<&AuthResolutionOverrides>,
    ) -> Result<Option<AuthResult>, ModelsError> {
        let _signal = overrides
            .and_then(|overrides| overrides.signal.clone())
            .unwrap_or_default();
        let (provider_id, model_headers) = match provider_or_model {
            ProviderOrModel::ProviderId(id) => (id.to_owned(), None),
            ProviderOrModel::Model(model) => (model.provider.to_string(), model.headers.clone()),
        };
        let Some(provider) = self.get_provider(&provider_id) else {
            return Ok(None);
        };
        let result = resolve_provider_auth(
            provider_id.as_str(),
            &provider.auth,
            self.inner.credentials.as_ref(),
            self.inner.auth_context.as_ref(),
            overrides,
        )
        .await?;
        let Some(mut result) = result else {
            return Ok(None);
        };
        if let Some(model_headers) = model_headers {
            let provider_headers: ProviderHeaders = model_headers
                .into_iter()
                .map(|(name, value)| (name, Some(value)))
                .collect();
            result.auth.headers =
                merge_provider_headers(result.auth.headers.as_ref(), Some(&provider_headers));
        }
        Ok(Some(result))
    }

    /// Run a provider-owned login flow and persist its returned credential.
    pub async fn login(
        &self,
        provider_id: &str,
        auth_type: AuthType,
        interaction: &dyn AuthInteraction,
    ) -> Result<Credential, ModelsError> {
        let signal = interaction.signal();
        if signal.is_cancelled() {
            return Err(ModelsError::new(ModelsErrorCode::Auth, "aborted"));
        }
        let Some(provider) = self.get_provider(provider_id) else {
            return Err(ModelsError::new(
                ModelsErrorCode::Provider,
                format!("Unknown provider: {provider_id}"),
            ));
        };
        let login_result = match auth_type {
            AuthType::Oauth => provider
                .auth
                .oauth
                .as_ref()
                .ok_or_else(|| {
                    ModelsError::new(
                        ModelsErrorCode::Auth,
                        format!("{} does not support oauth login", provider.name),
                    )
                })?
                .login(interaction)
                .await
                .map(Credential::Oauth),
            AuthType::ApiKey => provider
                .auth
                .api_key
                .as_ref()
                .ok_or_else(|| {
                    ModelsError::new(
                        ModelsErrorCode::Auth,
                        format!("{} does not support api_key login", provider.name),
                    )
                })?
                .login(interaction)
                .await
                .map(Credential::ApiKey),
        };
        let credential = crate::utils::race_with_abort(async { login_result }, &signal)
            .await
            .map_err(|_| ModelsError::new(ModelsErrorCode::Auth, "aborted"))
            .and_then(|result| {
                result.map_err(|error| {
                    ModelsError::with_source(ModelsErrorCode::Auth, "login failed", error)
                })
            })?;

        let stored = credential.clone();
        self.inner
            .credentials
            .modify(
                provider_id,
                crate::auth::types::modifier(move |_| {
                    let stored = stored.clone();
                    Box::pin(async move { Ok(Some(stored)) })
                }),
                &signal,
            )
            .await
            .map_err(|error| {
                ModelsError::with_source(
                    ModelsErrorCode::Auth,
                    format!("Credential store modify failed for {provider_id}"),
                    error,
                )
            })?;
        Ok(credential)
    }

    /// Remove the stored credential for a provider.
    pub async fn logout(
        &self,
        provider_id: &str,
        signal: Option<CancellationToken>,
    ) -> Result<(), ModelsError> {
        let signal = signal.unwrap_or_default();
        if signal.is_cancelled() {
            return Err(ModelsError::new(ModelsErrorCode::Auth, "aborted"));
        }
        self.inner
            .credentials
            .delete(provider_id, &signal)
            .await
            .map_err(|error| {
                ModelsError::with_source(
                    ModelsErrorCode::Auth,
                    format!("Credential store delete failed for {provider_id}"),
                    error,
                )
            })
    }

    fn require_provider(&self, model: &Model) -> Result<Arc<Provider>, ModelsError> {
        self.get_provider(model.provider.as_str()).ok_or_else(|| {
            ModelsError::new(
                ModelsErrorCode::Provider,
                format!("Unknown provider: {}", model.provider),
            )
        })
    }

    /// Resolve auth and merge it into request options. Explicit request options
    /// win per-field; the `transform_headers` hook runs last.
    async fn apply_auth(
        &self,
        model: &Model,
        options: Option<&StreamOptions>,
    ) -> Result<AppliedAuth, ModelsError> {
        self.require_provider(model)?;
        let overrides = AuthResolutionOverrides {
            api_key: options.and_then(|options| options.base.api_key.clone()),
            env: options.and_then(|options| options.base.env.clone()),
            signal: Some(
                options
                    .map(|options| options.base.token())
                    .unwrap_or_default(),
            ),
            min_oauth_validity_ms: None,
        };
        let resolution = self
            .get_auth(ProviderOrModel::Model(model), Some(&overrides))
            .await?;
        let Some(resolution) = resolution else {
            return Err(ModelsError::new(
                ModelsErrorCode::Auth,
                format!("Provider is not configured: {}", model.provider),
            ));
        };
        let auth = resolution.auth;

        let api_key = options
            .and_then(|options| options.base.api_key.clone())
            .or(auth.api_key);
        let mut headers = merge_provider_headers(
            auth.headers.as_ref(),
            options.and_then(|options| options.base.headers.as_ref()),
        );
        if let Some(transformer) = options.and_then(|options| options.transform_headers.clone()) {
            headers = Some(transformer.transform(headers.unwrap_or_default()).await);
        }
        let env = match (
            resolution.env,
            options.and_then(|options| options.base.env.clone()),
        ) {
            (Some(resolved), Some(explicit)) => {
                let mut merged = resolved;
                merged.extend(explicit);
                Some(merged)
            }
            (resolved, explicit) => resolved.or(explicit),
        };
        let mut request_model = model.clone();
        if let Some(base_url) = &auth.base_url {
            request_model.base_url = base_url.clone();
        }

        Ok(AppliedAuth {
            model: request_model,
            api_key,
            headers,
            env,
        })
    }

    fn dispatch_error(provider: &Provider, model: &Model) -> String {
        format!(
            "Provider {} has no API implementation for \"{}\"",
            provider.id, model.api
        )
    }

    pub fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&StreamOptions>,
    ) -> AssistantMessageEventStream {
        let models = self.clone();
        let model = model.clone();
        let model_for_error = model.clone();
        let context = context.clone();
        let options = options.cloned();
        lazy_stream(&model_for_error, move || async move {
            let applied = models
                .apply_auth(&model, options.as_ref())
                .await
                .map_err(|e| e.to_string())?;
            let provider = models
                .require_provider(&applied.model)
                .map_err(|error| error.to_string())?;
            let streams = provider
                .streams_for(&applied.model)
                .ok_or_else(|| Self::dispatch_error(&provider, &applied.model))?;
            let mut request_options = options.unwrap_or_default();
            request_options.base.api_key = applied.api_key;
            request_options.base.headers = applied.headers;
            request_options.base.env = applied.env;
            Ok::<_, String>(streams.stream(&applied.model, &context, &request_options))
        })
    }

    pub async fn complete(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&StreamOptions>,
    ) -> AssistantMessage {
        self.stream(model, context, options)
            .result()
            .await
            .unwrap_or_else(|error| {
                crate::event::error_result(model, &model.api, error.to_string())
            })
    }

    pub fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&SimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        let models = self.clone();
        let model = model.clone();
        let model_for_error = model.clone();
        let context = context.clone();
        let options = options.cloned();
        lazy_stream(&model_for_error, move || async move {
            let applied = models
                .apply_auth(&model, options.as_ref().map(|options| &options.base))
                .await
                .map_err(|error| error.to_string())?;
            let provider = models
                .require_provider(&applied.model)
                .map_err(|error| error.to_string())?;
            let streams = provider
                .streams_for(&applied.model)
                .ok_or_else(|| Self::dispatch_error(&provider, &applied.model))?;
            let mut request_options = options.unwrap_or_default();
            request_options.base.base.api_key = applied.api_key;
            request_options.base.base.headers = applied.headers;
            request_options.base.base.env = applied.env;
            Ok::<_, String>(streams.stream_simple(&applied.model, &context, &request_options))
        })
    }

    pub async fn complete_simple(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&SimpleStreamOptions>,
    ) -> AssistantMessage {
        self.stream_simple(model, context, options)
            .result()
            .await
            .unwrap_or_else(|error| {
                crate::event::error_result(model, &model.api, error.to_string())
            })
    }

    pub fn fetch_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        options: Option<&StreamOptions>,
    ) -> AssistantMessageEventStream {
        let models = self.clone();
        let model = model.clone();
        let model_for_error = model.clone();
        let handle = handle.clone();
        let options = options.cloned();
        lazy_stream(&model_for_error, move || async move {
            let applied = models
                .apply_auth(&model, options.as_ref())
                .await
                .map_err(|error| error.to_string())?;
            let provider = models
                .require_provider(&applied.model)
                .map_err(|error| error.to_string())?;
            let streams = provider
                .streams_for(&applied.model)
                .ok_or_else(|| Self::dispatch_error(&provider, &applied.model))?;
            if !streams.supports_fetch_deferred() {
                return Err(format!(
                    "Provider {} does not support deferred responses for \"{}\"",
                    provider.id, applied.model.api
                ));
            }
            let mut request_options = options.unwrap_or_default();
            request_options.base.api_key = applied.api_key;
            request_options.base.headers = applied.headers;
            request_options.base.env = applied.env;
            Ok::<_, String>(streams.fetch_deferred(&applied.model, &handle, &request_options))
        })
    }

    pub async fn cancel_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        options: Option<&StreamOptions>,
    ) -> Result<(), ModelsError> {
        let provider = self.require_provider(model)?;
        let applied = self.apply_auth(model, options).await?;
        let streams = provider.streams_for(&applied.model).ok_or_else(|| {
            ModelsError::new(
                ModelsErrorCode::Provider,
                Self::dispatch_error(&provider, &applied.model),
            )
        })?;
        if !streams.supports_fetch_deferred() {
            return Err(ModelsError::new(
                ModelsErrorCode::Provider,
                format!(
                    "Provider {} cannot cancel deferred responses",
                    model.provider
                ),
            ));
        }
        let mut request_options = options.cloned().unwrap_or_default();
        request_options.base.api_key = applied.api_key;
        request_options.base.headers = applied.headers;
        request_options.base.env = applied.env;
        streams
            .cancel_deferred(&applied.model, handle, &request_options)
            .await
    }
}

fn generation_current(inner: &ModelsInner, provider_id: &str, generation: u64) -> bool {
    inner
        .refresh_state
        .lock()
        .expect("refresh state poisoned")
        .get(provider_id)
        .is_some_and(|control| control.generation == generation)
}

/// Clone a refresh context handle for a nested fetch call (publish is an Arc).
fn clone_context(context: &RefreshModelsContext) -> RefreshModelsContext {
    RefreshModelsContext {
        credential: context.credential.clone(),
        stored: context.stored.clone(),
        publish: context.publish.clone(),
        allow_network: context.allow_network,
        force: context.force,
        signal: context.signal.clone(),
    }
}

// ---- pure model helpers ---------------------------------------------------

/// Runtime-checked narrowing for dynamically looked-up models.
pub fn has_api(model: &Model, api: &Api) -> bool {
    &model.api == api
}

/// Compute and attach cost to `usage` from the model's rates (and the highest
/// matching request-wide tier). Anthropic charges 2x base input for 1h cache
/// writes.
pub fn calculate_cost(model: &Model, usage: &mut Usage) -> crate::types::UsageCost {
    let input_tokens = usage
        .input
        .saturating_add(usage.cache_read)
        .saturating_add(usage.cache_write);
    let mut rates = model.cost.rates;
    let mut matched_threshold = u64::MAX;
    for tier in &model.cost.tiers {
        if input_tokens > tier.input_tokens_above && tier.input_tokens_above < matched_threshold {
            rates = tier.rates;
            matched_threshold = tier.input_tokens_above;
        }
    }

    let long_write = usage.cache_write_1h.unwrap_or(0);
    let short_write = usage.cache_write.saturating_sub(long_write);
    usage.cost.input = rates.input / 1_000_000.0 * usage.input as f64;
    usage.cost.output = rates.output / 1_000_000.0 * usage.output as f64;
    usage.cost.cache_read = rates.cache_read / 1_000_000.0 * usage.cache_read as f64;
    usage.cost.cache_write = (rates.cache_write * short_write as f64
        + rates.input * 2.0 * long_write as f64)
        / 1_000_000.0;
    usage.cost.total =
        usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write;
    usage.cost.clone()
}

const EXTENDED_THINKING_LEVELS: [ModelThinkingLevel; 7] = [
    ModelThinkingLevel::Off,
    ModelThinkingLevel::Minimal,
    ModelThinkingLevel::Low,
    ModelThinkingLevel::Medium,
    ModelThinkingLevel::High,
    ModelThinkingLevel::Xhigh,
    ModelThinkingLevel::Max,
];

/// Thinking levels the model supports: `off` only when not a reasoning model;
/// `xhigh`/`max` require an explicit mapping entry.
pub fn get_supported_thinking_levels(model: &Model) -> Vec<ModelThinkingLevel> {
    if !model.reasoning {
        return vec![ModelThinkingLevel::Off];
    }
    EXTENDED_THINKING_LEVELS
        .iter()
        .copied()
        .filter(|level| {
            let mapped = model
                .thinking_level_map
                .as_ref()
                .and_then(|map| map.get(level));
            if mapped == Some(&None) {
                // Explicitly unsupported.
                return false;
            }
            if matches!(level, ModelThinkingLevel::Xhigh | ModelThinkingLevel::Max) {
                return mapped.is_some();
            }
            true
        })
        .collect()
}

/// Clamp a requested level to the closest supported one, preferring higher.
pub fn clamp_thinking_level(model: &Model, level: ModelThinkingLevel) -> ModelThinkingLevel {
    let available = get_supported_thinking_levels(model);
    if available.contains(&level) {
        return level;
    }

    let requested_index = EXTENDED_THINKING_LEVELS
        .iter()
        .position(|candidate| *candidate == level)
        .unwrap_or(0);

    for candidate in &EXTENDED_THINKING_LEVELS[requested_index..] {
        if available.contains(candidate) {
            return *candidate;
        }
    }
    for candidate in EXTENDED_THINKING_LEVELS[..requested_index].iter().rev() {
        if available.contains(candidate) {
            return *candidate;
        }
    }
    available
        .first()
        .copied()
        .unwrap_or(ModelThinkingLevel::Off)
}

/// Check if two models are equal by comparing both their id and provider.
pub fn models_are_equal(a: Option<&Model>, b: Option<&Model>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => a.id == b.id && a.provider == b.provider,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ModelCost, ModelCostRates, ModelCostTier, StopReason, ThinkingLevelMap};

    fn model(reasoning: bool, thinking_level_map: Option<ThinkingLevelMap>) -> Model {
        Model {
            id: "m".to_owned(),
            name: "m".to_owned(),
            api: Api::OpenaiCompletions,
            provider: ProviderId::new("test"),
            base_url: "http://localhost".to_owned(),
            reasoning,
            thinking_level_map,
            input: vec![crate::types::InputModality::Text],
            cost: ModelCost::default(),
            context_window: 1000,
            max_tokens: 100,
            sampling_params: None,
            headers: None,
            compat: None,
        }
    }

    #[test]
    fn non_reasoning_model_supports_only_off() {
        assert_eq!(
            get_supported_thinking_levels(&model(false, None)),
            vec![ModelThinkingLevel::Off]
        );
    }

    #[test]
    fn xhigh_and_max_require_explicit_mapping() {
        let levels = get_supported_thinking_levels(&model(true, None));
        assert!(levels.contains(&ModelThinkingLevel::High));
        assert!(!levels.contains(&ModelThinkingLevel::Xhigh));
        assert!(!levels.contains(&ModelThinkingLevel::Max));
    }

    #[test]
    fn clamp_prefers_higher_levels() {
        let clamped = clamp_thinking_level(&model(true, None), ModelThinkingLevel::Max);
        assert_eq!(clamped, ModelThinkingLevel::High);
    }

    #[test]
    fn cost_includes_tier_and_long_cache_write() {
        let mut cost_model = model(false, None);
        cost_model.cost = ModelCost {
            rates: ModelCostRates {
                input: 1.0,
                output: 2.0,
                cache_read: 0.5,
                cache_write: 1.25,
            },
            tiers: vec![ModelCostTier {
                input_tokens_above: 1_000,
                rates: ModelCostRates {
                    input: 0.5,
                    output: 1.0,
                    cache_read: 0.25,
                    cache_write: 1.0,
                },
            }],
        };
        let mut usage = Usage {
            input: 2_000,
            output: 100,
            cache_read: 1_000,
            cache_write: 1_000,
            cache_write_1h: Some(400),
            reasoning: None,
            total_tokens: 3_100,
            cost: Default::default(),
        };
        calculate_cost(&cost_model, &mut usage);
        assert!((usage.cost.input - 0.001).abs() < 1e-12); // 0.5/M * 2000
        assert!((usage.cost.output - 0.0001).abs() < 1e-12); // 1/M * 100
        assert!((usage.cost.cache_read - 0.00025).abs() < 1e-12); // 0.25/M * 1000
        // short: 600 * 1.0 + long: 400 * 0.5*2 = 800 / 1M
        assert!((usage.cost.cache_write - 0.001).abs() < 1e-12);
        assert!(
            (usage.cost.total
                - (usage.cost.input
                    + usage.cost.output
                    + usage.cost.cache_read
                    + usage.cost.cache_write))
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn models_equal_requires_id_and_provider() {
        let a = model(false, None);
        let mut b = model(false, None);
        assert!(models_are_equal(Some(&a), Some(&b)));
        b.provider = ProviderId::new("other");
        assert!(!models_are_equal(Some(&a), Some(&b)));
        assert!(!models_are_equal(None, Some(&b)));
    }

    #[tokio::test]
    async fn stream_reports_unknown_provider_as_error_message() {
        let models = Models::new(CreateModelsOptions::new());
        let context = Context::default();
        let message = models.complete(&model(false, None), &context, None).await;
        assert_eq!(message.stop_reason, StopReason::Error);
        assert!(
            message
                .error_message
                .as_deref()
                .unwrap_or_default()
                .contains("Unknown provider")
        );
    }
}
