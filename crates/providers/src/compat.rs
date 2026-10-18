//! Legacy global API registry compat layer.
//!
//! Rust migration of the TypeScript `compat.ts`: preserves the old global
//! pi-ai API surface — api-dispatch `stream()`/`complete()` with env API key
//! injection, the api-registry, the Cloudflare special case, and faux provider
//! registration.
//!
//! Unlike the TypeScript module (which registers builtin api adapters at module
//! load), registration is explicit: call [`register_builtin_api_providers`] (or
//! [`reset_api_providers`], which calls it) once during startup. This avoids
//! init-order issues in a library crate.
//!
//! The Step-only build ships no builtin providers (`builtinProviders()` is
//! empty), so the `getBuiltinProviderForModel` path always resolves through the
//! api registry; the structure is kept for fidelity.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use crate::api::{anthropic_messages, openai_completions, openai_responses};
use crate::env_api_keys::{AMBIENT_AUTH_MARKER, get_env_api_key};
use crate::event::{AssistantMessageEventStream, lazy_stream};
use crate::models::{CreateModelsOptions, Models, ProviderStreams};
use crate::providers::faux::{
    FauxCore, FauxProviderState, FauxResponseStep, RegisterFauxProviderOptions, create_faux_core,
};
use crate::types::{Api, Context, Model, SimpleStreamOptions, StreamOptions};
use crate::utils::uuid::uuidv7;

/// Cloudflare AI Gateway auth header checked by the builtin-provider path.
const CLOUDFLARE_AUTH_HEADER: &str = "cf-aig-authorization";

/// One registry entry: the api implementation plus its registrant tag.
struct RegisteredApiProvider {
    streams: Arc<dyn ProviderStreams>,
    source_id: Option<String>,
}

fn registry() -> &'static RwLock<HashMap<String, RegisteredApiProvider>> {
    static REGISTRY: OnceLock<RwLock<HashMap<String, RegisteredApiProvider>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// The builtin api adapter instances, for the identity check in
/// [`get_builtin_provider_for_model`].
fn builtin_instances() -> &'static RwLock<HashMap<String, Arc<dyn ProviderStreams>>> {
    static INSTANCES: OnceLock<RwLock<HashMap<String, Arc<dyn ProviderStreams>>>> = OnceLock::new();
    INSTANCES.get_or_init(|| RwLock::new(HashMap::new()))
}

/// The empty builtin provider collection (the Step-only build ships no
/// built-in providers).
fn compat_models() -> &'static Models {
    static MODELS: OnceLock<Models> = OnceLock::new();
    MODELS.get_or_init(|| Models::new(CreateModelsOptions::new()))
}

/// Register an api implementation under its api id, optionally tagged with a
/// source id for later [`unregister_api_providers`].
pub fn register_api_provider(
    api: &str,
    streams: Arc<dyn ProviderStreams>,
    source_id: Option<&str>,
) {
    registry()
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(
            api.to_owned(),
            RegisteredApiProvider {
                streams,
                source_id: source_id.map(|source_id| source_id.to_owned()),
            },
        );
}

/// The api implementation registered for `api`, if any.
pub fn get_api_provider(api: &str) -> Option<Arc<dyn ProviderStreams>> {
    registry()
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(api)
        .map(|entry| entry.streams.clone())
}

/// Every registered api implementation.
pub fn get_api_providers() -> Vec<Arc<dyn ProviderStreams>> {
    registry()
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .values()
        .map(|entry| entry.streams.clone())
        .collect()
}

/// Every registered api id, for callers building api-dispatch maps.
pub fn registered_api_ids() -> Vec<String> {
    registry()
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .keys()
        .cloned()
        .collect()
}

/// Remove every entry registered under `source_id`.
pub fn unregister_api_providers(source_id: &str) {
    registry()
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .retain(|_, entry| entry.source_id.as_deref() != Some(source_id));
}

/// Registers the builtin API implementations into the api-registry without
/// clobbering existing entries: compat may load after a test or extension has
/// already registered an override for a builtin api id.
pub fn register_builtin_api_providers() {
    let builtins: [(&str, Arc<dyn ProviderStreams>); 3] = [
        (
            Api::ANTHROPIC_MESSAGES,
            Arc::new(anthropic_messages::anthropic_messages_api()),
        ),
        (
            Api::OPENAI_COMPLETIONS,
            Arc::new(openai_completions::openai_completions_api()),
        ),
        (
            Api::OPENAI_RESPONSES,
            Arc::new(openai_responses::openai_responses_api()),
        ),
    ];
    for (api, streams) in builtins {
        if get_api_provider(api).is_none() {
            register_api_provider(api, streams.clone(), None);
        }
        if let Some(provider) = get_api_provider(api) {
            builtin_instances()
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .insert(api.to_owned(), provider);
        }
    }
}

/// Clear the registry and re-register the builtin api implementations.
pub fn reset_api_providers() {
    registry()
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clear();
    builtin_instances()
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clear();
    register_builtin_api_providers();
}

/// Register a faux test provider in the api registry and return its
/// registration (scripted-response controls plus `unregister`).
pub fn register_faux_provider(options: &RegisterFauxProviderOptions) -> FauxProviderRegistration {
    let core = Arc::new(create_faux_core(options));
    let source_id = format!("faux-provider-{}", uuidv7());
    register_api_provider(&core.api, core.clone(), Some(&source_id));
    FauxProviderRegistration { core, source_id }
}

/// Registration handle returned by [`register_faux_provider`]: scripted-response
/// controls plus the api/provider/model metadata of the faux provider.
pub struct FauxProviderRegistration {
    /// The faux core backing the registry entry.
    pub core: Arc<FauxCore>,
    source_id: String,
}

impl FauxProviderRegistration {
    pub fn api(&self) -> &str {
        &self.core.api
    }

    /// The provider id models are registered under.
    pub fn provider_id(&self) -> &str {
        &self.core.provider
    }

    pub fn models(&self) -> &[Model] {
        &self.core.models
    }

    pub fn get_model(&self) -> Model {
        self.core.get_model()
    }

    pub fn get_model_by_id(&self, model_id: &str) -> Option<Model> {
        self.core.get_model_by_id(model_id)
    }

    pub fn state(&self) -> Arc<Mutex<FauxProviderState>> {
        self.core.state()
    }

    pub fn set_responses(&self, responses: Vec<FauxResponseStep>) {
        self.core.set_responses(responses);
    }

    pub fn append_responses(&self, responses: Vec<FauxResponseStep>) {
        self.core.append_responses(responses);
    }

    pub fn pending_response_count(&self) -> usize {
        self.core.pending_response_count()
    }

    /// Remove this provider's registry entry.
    pub fn unregister(&self) {
        unregister_api_providers(&self.source_id);
    }
}

fn has_explicit_api_key(api_key: Option<&str>) -> bool {
    api_key.is_some_and(|api_key| !api_key.trim().is_empty())
}

/// Inject the provider's env api key into a copy of the request options unless
/// the request already carries one or the env key is the ambient-auth marker.
fn with_env_api_key(model: &Model, options: Option<&StreamOptions>) -> StreamOptions {
    let mut merged = options.cloned().unwrap_or_default();
    if has_explicit_api_key(merged.base.api_key.as_deref()) {
        return merged;
    }
    let api_key = get_env_api_key(model.provider.as_str(), merged.base.env.as_ref());
    match api_key {
        Some(api_key) if api_key != AMBIENT_AUTH_MARKER => merged.base.api_key = Some(api_key),
        _ => {}
    }
    merged
}

fn has_resolved_cloudflare_auth(options: Option<&StreamOptions>) -> bool {
    if has_explicit_api_key(options.and_then(|options| options.base.api_key.as_deref())) {
        return true;
    }
    options
        .and_then(|options| options.base.headers.as_ref())
        .is_some_and(|headers| {
            headers
                .get(CLOUDFLARE_AUTH_HEADER)
                .is_some_and(|value| value.is_some())
        })
}

/// Resolve the builtin provider for a model: only when the registered api
/// implementation is the builtin instance and the model exists in the builtin
/// catalog. Always `None` in the Step-only build (empty catalog).
fn get_builtin_provider_for_model(model: &Model) -> Option<Arc<dyn ProviderStreams>> {
    let registered = get_api_provider(model.api.as_str())?;
    let builtin = builtin_instances()
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(model.api.as_str())
        .cloned()?;
    if !Arc::ptr_eq(&registered, &builtin) {
        return None;
    }
    let provider = compat_models().get_provider(model.provider.as_str())?;
    if provider
        .get_models()
        .iter()
        .any(|candidate| candidate.api == model.api)
    {
        Some(registered)
    } else {
        None
    }
}

/// A stream that fails immediately with the given error message.
fn error_stream(model: &Model, message: String) -> AssistantMessageEventStream {
    lazy_stream(model, move || async move {
        Err::<AssistantMessageEventStream, _>(message)
    })
}

/// Legacy api-dispatch stream: resolve the registered api provider for
/// `model.api`, inject the env api key, and stream.
pub fn stream(
    model: &Model,
    context: &Context,
    options: Option<&StreamOptions>,
) -> AssistantMessageEventStream {
    if let Some(builtin) = get_builtin_provider_for_model(model) {
        if model.provider.as_str().starts_with("cloudflare-")
            && !has_resolved_cloudflare_auth(options)
        {
            // Cloudflare models without resolved auth fall back to the builtin
            // Models collection, which reports the configuration error.
            return compat_models().stream(model, context, options);
        }
        return builtin.stream(model, context, &with_env_api_key(model, options));
    }
    let Some(provider) = get_api_provider(model.api.as_str()) else {
        return error_stream(
            model,
            format!("No API provider registered for api: {}", model.api),
        );
    };
    provider.stream(model, context, &with_env_api_key(model, options))
}

/// Legacy api-dispatch completion: collect [`stream`] into the final message.
pub async fn complete(
    model: &Model,
    context: &Context,
    options: Option<&StreamOptions>,
) -> crate::types::AssistantMessage {
    stream(model, context, options)
        .result()
        .await
        .unwrap_or_else(|error| crate::event::error_result(model, &model.api, error.to_string()))
}

/// Legacy api-dispatch stream with simple (provider-neutral) options.
pub fn stream_simple(
    model: &Model,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
) -> AssistantMessageEventStream {
    if let Some(builtin) = get_builtin_provider_for_model(model) {
        if model.provider.as_str().starts_with("cloudflare-")
            && !has_resolved_cloudflare_auth(options.map(|options| &options.base))
        {
            return compat_models().stream_simple(model, context, options);
        }
        return builtin.stream_simple(model, context, &with_env_api_key_simple(model, options));
    }
    let Some(provider) = get_api_provider(model.api.as_str()) else {
        return error_stream(
            model,
            format!("No API provider registered for api: {}", model.api),
        );
    };
    provider.stream_simple(model, context, &with_env_api_key_simple(model, options))
}

/// Legacy api-dispatch simple completion: collect [`stream_simple`].
pub async fn complete_simple(
    model: &Model,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
) -> crate::types::AssistantMessage {
    stream_simple(model, context, options)
        .result()
        .await
        .unwrap_or_else(|error| crate::event::error_result(model, &model.api, error.to_string()))
}

/// [`with_env_api_key`] for simple options.
fn with_env_api_key_simple(
    model: &Model,
    options: Option<&SimpleStreamOptions>,
) -> SimpleStreamOptions {
    let mut merged = options.cloned().unwrap_or_default();
    merged.base = with_env_api_key(model, Some(&merged.base));
    merged
}

/// Serializes tests that touch the process-global api registry. An async lock:
/// the guarded section awaits provider streams, and holding a blocking mutex
/// across those awaits would deadlock the registry the streams dispatch on.
#[cfg(test)]
pub(crate) fn registry_test_lock() -> &'static tokio::sync::Mutex<()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{AssistantMessageEvent, DoneReason, assistant_message_event_stream};
    use crate::providers::faux::{
        FauxAssistantMessageOptions, faux_assistant_message, faux_provider,
    };
    use crate::types::{
        AssistantContent, ProviderId, ProviderRequestOptions, StopReason, TextContent, Usage,
    };
    use async_trait::async_trait;

    fn test_model() -> Model {
        Model {
            id: "test-model".to_owned(),
            name: "Test Model".to_owned(),
            api: Api::OpenaiResponses,
            provider: ProviderId::new("custom-openai"),
            base_url: "https://example.test/v1".to_owned(),
            reasoning: false,
            thinking_level_map: None,
            input: vec![crate::types::InputModality::Text],
            cost: Default::default(),
            context_window: 128_000,
            max_tokens: 4096,
            sampling_params: None,
            headers: None,
            compat: None,
        }
    }

    fn test_context() -> Context {
        Context {
            system_prompt: None,
            messages: vec![crate::types::Message::user("hi", 1)],
            tools: None,
        }
    }

    fn test_message(model: &Model) -> crate::types::AssistantMessage {
        crate::types::AssistantMessage {
            content: vec![AssistantContent::Text(TextContent {
                text: "ok".to_owned(),
                text_signature: None,
            })],
            api: model.api.clone(),
            provider: model.provider.clone(),
            model: model.id.clone(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            timestamp: crate::utils::now_ms(),
        }
    }

    /// Stream that immediately emits start + done with the test message.
    fn scripted_stream(model: &Model) -> AssistantMessageEventStream {
        let (mut sender, stream) = assistant_message_event_stream();
        let message = test_message(model);
        tokio::spawn(async move {
            sender
                .push(AssistantMessageEvent::Start {
                    partial: message.clone(),
                })
                .await;
            sender
                .push(AssistantMessageEvent::Done {
                    reason: DoneReason::Stop,
                    message: message.clone(),
                })
                .await;
            sender.end(message);
        });
        stream
    }

    /// Records the api key it was dispatched with, and how often it ran.
    struct CapturingProvider {
        captured_api_key: Arc<Mutex<Option<String>>>,
        call_count: Arc<Mutex<usize>>,
    }

    impl CapturingProvider {
        fn record(&self, api_key: Option<String>) {
            *self
                .captured_api_key
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = api_key;
            *self
                .call_count
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) += 1;
        }
    }

    #[async_trait]
    impl ProviderStreams for CapturingProvider {
        fn stream(
            &self,
            _model: &Model,
            _context: &Context,
            options: &StreamOptions,
        ) -> AssistantMessageEventStream {
            self.record(options.base.api_key.clone());
            scripted_stream(_model)
        }

        fn stream_simple(
            &self,
            _model: &Model,
            _context: &Context,
            options: &SimpleStreamOptions,
        ) -> AssistantMessageEventStream {
            self.record(options.base.base.api_key.clone());
            scripted_stream(_model)
        }
    }

    type Captured = (
        Arc<CapturingProvider>,
        Arc<Mutex<Option<String>>>,
        Arc<Mutex<usize>>,
    );

    fn capturing_provider() -> Captured {
        let captured = Arc::new(Mutex::new(None));
        let calls = Arc::new(Mutex::new(0));
        (
            Arc::new(CapturingProvider {
                captured_api_key: captured.clone(),
                call_count: calls.clone(),
            }),
            captured,
            calls,
        )
    }

    fn assert_call_count(calls: &Mutex<usize>, expected: usize) {
        assert_eq!(
            *calls
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
            expected
        );
    }

    async fn reset() -> tokio::sync::MutexGuard<'static, ()> {
        let guard = registry_test_lock().lock().await;
        reset_api_providers();
        guard
    }

    #[tokio::test]
    async fn dispatches_unknown_providers_through_the_legacy_api_registry() {
        let _serialized = reset().await;
        let (provider, captured, calls) = capturing_provider();
        register_api_provider(Api::OPENAI_RESPONSES, provider, None);

        let model = test_model();
        let options = StreamOptions {
            base: ProviderRequestOptions {
                api_key: Some("request-key".to_owned()),
                ..Default::default()
            },
            ..Default::default()
        };
        complete(&model, &test_context(), Some(&options)).await;

        assert_eq!(
            captured
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .as_deref(),
            Some("request-key")
        );
        assert_call_count(&calls, 1);
    }

    #[tokio::test]
    async fn injects_the_env_api_key_when_options_have_none() {
        let _serialized = reset().await;
        let (provider, captured, calls) = capturing_provider();
        register_api_provider(Api::OPENAI_COMPLETIONS, provider, None);

        let mut model = test_model();
        model.api = Api::OpenaiCompletions;
        model.provider = ProviderId::new("openai");
        let options = StreamOptions {
            base: ProviderRequestOptions {
                env: Some(HashMap::from([(
                    "OPENAI_API_KEY".to_owned(),
                    "env-key".to_owned(),
                )])),
                ..Default::default()
            },
            ..Default::default()
        };
        complete(&model, &test_context(), Some(&options)).await;

        assert_eq!(
            captured
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .as_deref(),
            Some("env-key")
        );
        assert_call_count(&calls, 1);
    }

    #[tokio::test]
    async fn skips_injection_for_ambient_auth_markers() {
        let _serialized = reset().await;
        let (provider, captured, calls) = capturing_provider();
        register_api_provider(Api::OPENAI_COMPLETIONS, provider, None);

        let mut model = test_model();
        model.api = Api::OpenaiCompletions;
        model.provider = ProviderId::new("amazon-bedrock");
        let options = StreamOptions {
            base: ProviderRequestOptions {
                env: Some(HashMap::from([(
                    "AWS_PROFILE".to_owned(),
                    "default".to_owned(),
                )])),
                ..Default::default()
            },
            ..Default::default()
        };
        complete(&model, &test_context(), Some(&options)).await;

        assert!(
            captured
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .is_none()
        );
        assert_call_count(&calls, 1);
    }

    #[tokio::test]
    async fn explicit_request_key_wins_over_env() {
        let _serialized = reset().await;
        let (provider, captured, calls) = capturing_provider();
        register_api_provider(Api::OPENAI_COMPLETIONS, provider, None);

        let mut model = test_model();
        model.api = Api::OpenaiCompletions;
        model.provider = ProviderId::new("openai");
        let options = StreamOptions {
            base: ProviderRequestOptions {
                api_key: Some("request-key".to_owned()),
                env: Some(HashMap::from([(
                    "OPENAI_API_KEY".to_owned(),
                    "env-key".to_owned(),
                )])),
                ..Default::default()
            },
            ..Default::default()
        };
        complete(&model, &test_context(), Some(&options)).await;

        assert_eq!(
            captured
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .as_deref(),
            Some("request-key")
        );
        assert_call_count(&calls, 1);
    }

    #[tokio::test]
    async fn errors_when_no_api_provider_is_registered() {
        let _serialized = reset().await;
        let mut model = test_model();
        model.api = Api::Custom("unregistered-api".into());
        let response = complete(&model, &test_context(), None).await;

        assert_eq!(response.stop_reason, StopReason::Error);
        assert_eq!(
            response.error_message.as_deref(),
            Some("No API provider registered for api: unregistered-api")
        );
    }

    #[tokio::test]
    async fn cloudflare_models_dispatch_through_the_registry() {
        let _serialized = reset().await;
        let (provider, captured, calls) = capturing_provider();
        register_api_provider(Api::OPENAI_COMPLETIONS, provider, None);

        let mut model = test_model();
        model.api = Api::OpenaiCompletions;
        model.provider = ProviderId::new("cloudflare-ai-gateway");
        // The Step-only build ships no builtin providers, so the Cloudflare
        // fallback branch never engages and the registry handles the request.
        let response = complete(&model, &test_context(), None).await;

        assert_eq!(response.stop_reason, StopReason::Stop);
        assert_eq!(
            *calls
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
            1,
            "request should dispatch through the api registry"
        );
        assert!(
            captured
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .is_none()
        );
    }

    #[tokio::test]
    async fn reset_reregisters_builtin_apis_and_drops_custom_entries() {
        let _serialized = reset().await;
        let (provider, _captured, _calls) = capturing_provider();
        register_api_provider("faux:test-reset", provider, None);
        assert!(get_api_provider("faux:test-reset").is_some());

        reset_api_providers();

        assert!(get_api_provider("faux:test-reset").is_none());
        for api in [
            Api::ANTHROPIC_MESSAGES,
            Api::OPENAI_COMPLETIONS,
            Api::OPENAI_RESPONSES,
        ] {
            assert!(
                get_api_provider(api).is_some(),
                "builtin api {api} not registered"
            );
        }
    }

    #[tokio::test]
    async fn unregister_removes_only_the_tagged_source() {
        let _serialized = reset().await;
        let (provider, _captured, _calls) = capturing_provider();
        register_api_provider("faux:test-unregister", provider, Some("source-a"));
        register_api_provider("faux:test-other", capturing_provider().0, Some("source-b"));

        unregister_api_providers("source-a");

        assert!(get_api_provider("faux:test-unregister").is_none());
        assert!(get_api_provider("faux:test-other").is_some());
    }

    #[tokio::test]
    async fn faux_provider_registers_into_the_api_registry() {
        let _serialized = reset().await;
        let handle = faux_provider(&RegisterFauxProviderOptions::default());
        register_api_provider(handle.api(), handle.core.clone(), None);
        handle.set_responses(vec![
            faux_assistant_message("hello", FauxAssistantMessageOptions::default()).into(),
        ]);

        let response = complete(&handle.get_model(), &test_context(), None).await;

        assert_eq!(response.stop_reason, StopReason::Stop);
        assert_eq!(
            serde_json::to_value(&response.content).unwrap(),
            serde_json::json!([{ "type": "text", "text": "hello" }])
        );
        assert_eq!(
            handle
                .state()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .call_count,
            1
        );
    }
}
