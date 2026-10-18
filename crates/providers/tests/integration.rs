//! End-to-end integration: a scripted provider wired through `Models` streams
//! a full assistant turn; the compat api-registry dispatches with env-key
//! injection.

use std::sync::Arc;

use futures::StreamExt;
use step_harness_providers::auth::{DefaultAuthContext, ProviderAuth, env_api_key_auth};
use step_harness_providers::compat;
use step_harness_providers::event::{AssistantMessageEvent, DoneReason};
use step_harness_providers::models::{
    ApiDispatch, CreateModelsOptions, CreateProviderOptions, Models, ProviderStreams,
    create_provider,
};
use step_harness_providers::providers::faux::{
    RegisterFauxProviderOptions, faux_assistant_message, faux_provider, faux_tool_call,
};
use step_harness_providers::types::{
    Api, AssistantContent, Context, InputModality, Message, Model, ModelCost, ProviderId,
    SimpleStreamOptions, StopReason, Tool, UserContent,
};
use step_harness_providers::utils::uuid::uuidv7;

fn faux_model() -> Model {
    Model {
        id: "faux-1".to_owned(),
        name: "Faux 1".to_owned(),
        api: Api::OpenaiCompletions,
        provider: ProviderId::new("faux"),
        base_url: "http://faux.invalid/v1".to_owned(),
        reasoning: true,
        thinking_level_map: None,
        input: vec![InputModality::Text],
        cost: ModelCost::default(),
        context_window: 8192,
        max_tokens: 1024,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// A `Models` collection with one faux-backed provider registered under
/// `faux`. The faux core serves its own scripted api id, so the provider
/// carries the registration's models; auth resolves from an injected
/// environment overlay (the faux provider itself never uses the key).
fn faux_models() -> Models {
    let registration = faux_provider(&RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message("hello from faux", Default::default()).into(),
    ]);

    let provider = create_provider(CreateProviderOptions {
        id: ProviderId::new("faux"),
        name: Some("Faux".to_owned()),
        base_url: Some("http://faux.invalid/v1".to_owned()),
        headers: None,
        auth: ProviderAuth::with_api_key(env_api_key_auth(
            "Faux API key",
            vec!["__INTEGRATION_FAUX_KEY__".to_owned()],
        )),
        models: registration.models().to_vec(),
        fetch_models: None,
        filter_models: None,
        api: ApiDispatch::Single(registration.core.clone()),
    });
    let auth_context = DefaultAuthContext::with_overlay(
        [("__INTEGRATION_FAUX_KEY__".to_owned(), "test-key".to_owned())]
            .into_iter()
            .collect(),
    );
    let models = Models::new(CreateModelsOptions {
        auth_context: Some(Arc::new(auth_context)),
        ..Default::default()
    });
    models.set_provider(provider);
    models
}

fn text_context() -> Context {
    Context {
        system_prompt: Some("Be terse.".to_owned()),
        messages: vec![Message::user("hi", 1)],
        tools: None,
    }
}

#[tokio::test]
async fn models_stream_yields_full_event_sequence_and_final_message() {
    let models = faux_models();
    let mut events = Vec::new();
    let mut stream = models.stream_simple(&faux_model(), &text_context(), None);
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    let final_message = stream.result().await.unwrap();

    let types: Vec<&str> = events
        .iter()
        .map(|event| match event {
            AssistantMessageEvent::Start { .. } => "start",
            AssistantMessageEvent::TextStart { .. } => "text_start",
            AssistantMessageEvent::TextDelta { .. } => "text_delta",
            AssistantMessageEvent::TextEnd { .. } => "text_end",
            AssistantMessageEvent::ThinkingStart { .. } => "thinking_start",
            AssistantMessageEvent::ThinkingDelta { .. } => "thinking_delta",
            AssistantMessageEvent::ThinkingEnd { .. } => "thinking_end",
            AssistantMessageEvent::ToolCallStart { .. } => "toolcall_start",
            AssistantMessageEvent::ToolCallDelta { .. } => "toolcall_delta",
            AssistantMessageEvent::ToolCallEnd { .. } => "toolcall_end",
            AssistantMessageEvent::Done { .. } => "done",
            AssistantMessageEvent::Error { .. } => "error",
        })
        .collect();
    assert_eq!(types.first(), Some(&"start"));
    assert_eq!(types.last(), Some(&"done"));
    assert!(types.contains(&"text_start"));
    assert!(types.contains(&"text_end"));

    assert_eq!(final_message.stop_reason, StopReason::Stop);
    assert!(final_message.usage.input > 0);
    assert_eq!(
        final_message.usage.total_tokens,
        final_message.usage.input + final_message.usage.output
    );
    assert!(final_message.content.iter().any(|block| {
        block
            .text()
            .is_some_and(|text| text.text.contains("hello from faux"))
    }));
}

#[tokio::test]
async fn models_complete_simple_resolves_the_final_message() {
    let models = faux_models();
    let message = models
        .complete_simple(&faux_model(), &text_context(), None)
        .await;
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert!(message.error_message.is_none());
}

#[tokio::test]
async fn unconfigured_provider_streams_an_error_message() {
    let registration = faux_provider(&RegisterFauxProviderOptions::default());
    let provider = create_provider(CreateProviderOptions {
        id: ProviderId::new("faux"),
        name: Some("Faux".to_owned()),
        base_url: None,
        headers: None,
        auth: ProviderAuth::with_api_key(env_api_key_auth(
            "Faux API key",
            vec!["__INTEGRATION_FAUX_KEY_UNSET__".to_owned()],
        )),
        models: registration.models().to_vec(),
        fetch_models: None,
        filter_models: None,
        api: ApiDispatch::Single(registration.core.clone()),
    });
    let models = Models::new(CreateModelsOptions::new());
    models.set_provider(provider);

    let message = models
        .complete_simple(&registration.get_model(), &text_context(), None)
        .await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert!(
        message
            .error_message
            .as_deref()
            .unwrap_or_default()
            .contains("not configured")
    );
}

#[tokio::test]
async fn compat_registry_dispatches_streams_with_env_key_injection() {
    // The faux provider registers itself into the process-global api registry.
    let registration = faux_provider(&RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message("via compat", Default::default()).into(),
    ]);
    let source_id = uuidv7();
    compat::register_api_provider(
        registration.api(),
        registration.core.clone(),
        Some(&source_id),
    );

    // An explicit key wins over the environment.
    let mut options = SimpleStreamOptions::default();
    options.base.base.api_key = Some("explicit".to_owned());
    let mut stream =
        compat::stream_simple(&registration.get_model(), &text_context(), Some(&options));
    let mut saw_done = false;
    while let Some(event) = stream.next().await {
        if matches!(
            event,
            AssistantMessageEvent::Done {
                reason: DoneReason::Stop,
                ..
            }
        ) {
            saw_done = true;
        }
    }
    assert!(saw_done);

    compat::unregister_api_providers(&source_id);
}

#[tokio::test]
async fn tool_calls_and_tool_results_round_trip_through_the_collection() {
    let registration = faux_provider(&RegisterFauxProviderOptions::default());
    let tool_call = faux_tool_call("search", Default::default(), Some("call_1".to_owned()));
    registration.set_responses(vec![
        faux_assistant_message(
            vec![step_harness_providers::providers::faux::FauxContentBlock::ToolCall(tool_call)],
            step_harness_providers::providers::faux::FauxAssistantMessageOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into(),
    ]);

    let provider = create_provider(CreateProviderOptions {
        id: ProviderId::new("faux"),
        name: Some("Faux".to_owned()),
        base_url: None,
        headers: None,
        auth: ProviderAuth::with_api_key(env_api_key_auth(
            "Faux API key",
            vec!["__INTEGRATION_FAUX_KEY__".to_owned()],
        )),
        models: registration.models().to_vec(),
        fetch_models: None,
        filter_models: None,
        api: ApiDispatch::Single(registration.core.clone()),
    });
    let auth_context = DefaultAuthContext::with_overlay(
        [("__INTEGRATION_FAUX_KEY__".to_owned(), "test-key".to_owned())]
            .into_iter()
            .collect(),
    );
    let models = Models::new(CreateModelsOptions {
        auth_context: Some(Arc::new(auth_context)),
        ..Default::default()
    });
    models.set_provider(provider);

    let context = Context {
        system_prompt: None,
        messages: vec![Message::user("search something", 1)],
        tools: Some(vec![Tool {
            name: "search".to_owned(),
            description: "search".to_owned(),
            parameters: serde_json::json!({"type": "object", "properties": {"q": {"type": "string"}}}),
            constrained_sampling: None,
        }]),
    };
    let message = models.complete_simple(&faux_model(), &context, None).await;
    assert_eq!(message.stop_reason, StopReason::ToolUse);
    assert!(
        message.content.iter().any(
            |block| matches!(block, AssistantContent::ToolCall(call) if call.name == "search")
        )
    );
}

#[test]
fn provider_streams_trait_object_dispatch_supports_mixed_apis() {
    // A provider serving two dialects dispatches on model.api; the trait object
    // is what `ApiDispatch::ByApi` stores.
    struct NoopApi;
    impl ProviderStreams for NoopApi {
        fn stream(
            &self,
            _model: &Model,
            _context: &Context,
            _options: &step_harness_providers::types::StreamOptions,
        ) -> step_harness_providers::event::AssistantMessageEventStream {
            unreachable!("not exercised")
        }
        fn stream_simple(
            &self,
            _model: &Model,
            _context: &Context,
            _options: &SimpleStreamOptions,
        ) -> step_harness_providers::event::AssistantMessageEventStream {
            unreachable!("not exercised")
        }
    }
    let provider = create_provider(CreateProviderOptions {
        id: ProviderId::new("mixed"),
        name: None,
        base_url: None,
        headers: None,
        auth: ProviderAuth::ambient_only(),
        models: Vec::new(),
        fetch_models: None,
        filter_models: None,
        api: ApiDispatch::ByApi(
            [(
                Api::OPENAI_COMPLETIONS.to_owned(),
                Arc::new(NoopApi) as Arc<dyn ProviderStreams>,
            )]
            .into_iter()
            .collect(),
        ),
    });
    assert!(provider.get_models().is_empty());
    assert!(!provider.supports_fetch_deferred());
    assert_eq!(provider.id.as_str(), "mixed");
}

#[test]
fn user_content_helper_builds_plain_text() {
    let message = Message::user("plain", 7);
    assert_eq!(
        message.as_user().map(|user| user.content.as_text()),
        Some(Some("plain"))
    );
    assert_eq!(message.timestamp(), 7);
    let _ = UserContent::from("x");
}
