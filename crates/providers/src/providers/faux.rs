//! Faux (fake) test provider: scripted assistant responses for tests.
//!
//! Rust migration of the TypeScript `providers/faux.ts`. The provider streams
//! scripted [`AssistantMessage`] responses (or response factories) as assistant
//! message events, estimates token usage from the serialized context, and
//! supports deferred responses (`fetch_deferred`/`cancel_deferred`).
//!
//! Unlike the single-threaded TypeScript original, all scripted state lives
//! behind [`Mutex`] guards so the provider is `Send + Sync` and can be driven
//! from tokio tasks.

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};

use async_trait::async_trait;
use rand::RngExt;
use tokio_util::sync::CancellationToken;

use crate::auth::context::AuthContext;
use crate::auth::types::{ApiKeyAuth, ApiKeyCredential, AuthResult, ProviderAuth};
use crate::event::{
    AssistantMessageEvent, AssistantMessageEventSender, AssistantMessageEventStream, DoneReason,
    ErrorReason, assistant_message_event_stream, thinking_text_at,
};
use crate::models::{
    ApiDispatch, CreateProviderOptions, Provider, ProviderStreams, create_provider,
};
use crate::types::{
    Api, AssistantContent, AssistantMessage, CacheRetention, Context, DeferredHandle,
    InputModality, Json, Message, Model, ModelCost, ModelCostRates, ProviderId, ProviderResponse,
    SimpleStreamOptions, StopReason, StreamOptions, TextContent, ThinkingContent, ToolCall, Usage,
    UsageCost, UserContent, UserContentBlock,
};
use crate::utils::estimate::estimate_text_tokens;
use crate::utils::uuid::uuidv7;
use crate::utils::{now_ms, sleep_ms};

const DEFAULT_API: &str = "faux";
const DEFAULT_PROVIDER: &str = "faux";
const DEFAULT_MODEL_ID: &str = "faux-1";
const DEFAULT_MODEL_NAME: &str = "Faux Model";
const DEFAULT_BASE_URL: &str = "http://localhost:0";
const DEFAULT_MIN_TOKEN_SIZE: usize = 3;
const DEFAULT_MAX_TOKEN_SIZE: usize = 5;

/// Lock a mutex, recovering from poisoning instead of panicking.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Zero-usage block shared by every synthesized message.
fn default_usage() -> Usage {
    Usage {
        input: 0,
        output: 0,
        cache_read: 0,
        cache_write: 0,
        cache_write_1h: None,
        reasoning: None,
        total_tokens: 0,
        cost: UsageCost::default(),
    }
}

/// Unique id with a prefix (the TypeScript `randomId`).
fn random_id(prefix: &str) -> String {
    format!("{prefix}:{}", uuidv7())
}

/// Model declaration accepted by [`RegisterFauxProviderOptions::models`].
#[derive(Clone, Debug, Default)]
pub struct FauxModelDefinition {
    pub id: String,
    pub name: Option<String>,
    pub reasoning: Option<bool>,
    pub input: Option<Vec<InputModality>>,
    pub cost: Option<ModelCostRates>,
    pub context_window: Option<u64>,
    pub max_tokens: Option<u64>,
}

/// Content blocks a scripted faux assistant message may contain.
#[derive(Clone, Debug)]
pub enum FauxContentBlock {
    Text(TextContent),
    Thinking(ThinkingContent),
    ToolCall(ToolCall),
}

impl From<FauxContentBlock> for AssistantContent {
    fn from(block: FauxContentBlock) -> Self {
        match block {
            FauxContentBlock::Text(content) => AssistantContent::Text(content),
            FauxContentBlock::Thinking(content) => AssistantContent::Thinking(content),
            FauxContentBlock::ToolCall(call) => AssistantContent::ToolCall(call),
        }
    }
}

/// Content argument of [`faux_assistant_message`]: plain text, one block, or a
/// block list.
pub enum FauxContentInput {
    Text(String),
    Block(FauxContentBlock),
    Blocks(Vec<FauxContentBlock>),
}

impl From<&str> for FauxContentInput {
    fn from(value: &str) -> Self {
        FauxContentInput::Text(value.to_owned())
    }
}

impl From<String> for FauxContentInput {
    fn from(value: String) -> Self {
        FauxContentInput::Text(value)
    }
}

impl From<&String> for FauxContentInput {
    fn from(value: &String) -> Self {
        FauxContentInput::Text(value.clone())
    }
}

impl From<FauxContentBlock> for FauxContentInput {
    fn from(value: FauxContentBlock) -> Self {
        FauxContentInput::Block(value)
    }
}

impl From<Vec<FauxContentBlock>> for FauxContentInput {
    fn from(value: Vec<FauxContentBlock>) -> Self {
        FauxContentInput::Blocks(value)
    }
}

fn normalize_faux_assistant_content(content: FauxContentInput) -> Vec<AssistantContent> {
    match content {
        FauxContentInput::Text(text) => vec![AssistantContent::Text(faux_text(text))],
        FauxContentInput::Block(block) => vec![block.into()],
        FauxContentInput::Blocks(blocks) => {
            blocks.into_iter().map(AssistantContent::from).collect()
        }
    }
}

/// A text content block.
pub fn faux_text(text: impl Into<String>) -> TextContent {
    TextContent {
        text: text.into(),
        text_signature: None,
    }
}

/// A thinking content block.
pub fn faux_thinking(thinking: impl Into<String>) -> ThinkingContent {
    ThinkingContent {
        thinking: thinking.into(),
        thinking_signature: None,
        redacted: false,
    }
}

/// A tool call content block with a generated id when none is given.
pub fn faux_tool_call(
    name: impl Into<String>,
    arguments: serde_json::Map<String, Json>,
    id: Option<String>,
) -> ToolCall {
    ToolCall {
        id: id.unwrap_or_else(|| random_id("tool")),
        name: name.into(),
        arguments,
        thought_signature: None,
        namespace: None,
    }
}

/// Optional fields of [`faux_assistant_message`].
#[derive(Clone, Debug, Default)]
pub struct FauxAssistantMessageOptions {
    pub stop_reason: Option<StopReason>,
    pub deferred: Option<DeferredHandle>,
    pub error_message: Option<String>,
    pub response_id: Option<String>,
    pub timestamp: Option<u64>,
}

/// Build a scripted assistant message bound to the faux defaults.
pub fn faux_assistant_message(
    content: impl Into<FauxContentInput>,
    options: FauxAssistantMessageOptions,
) -> AssistantMessage {
    AssistantMessage {
        content: normalize_faux_assistant_content(content.into()),
        api: Api::from(DEFAULT_API),
        provider: ProviderId::new(DEFAULT_PROVIDER),
        model: DEFAULT_MODEL_ID.to_owned(),
        response_model: None,
        response_id: options.response_id,
        diagnostics: None,
        usage: default_usage(),
        stop_reason: options.stop_reason.unwrap_or(StopReason::Stop),
        deferred: options.deferred,
        error_message: options.error_message,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: options.timestamp.unwrap_or_else(now_ms),
    }
}

/// Shared, thread-safe faux provider counters (the TypeScript
/// `FauxProviderState` object).
#[derive(Clone, Debug, Default)]
pub struct FauxProviderState {
    pub call_count: u64,
    pub deferred_fetch_count: u64,
    pub cancelled_deferred: Vec<DeferredHandle>,
}

/// Future returned by a [`FauxResponseFactory`]; borrows the request inputs.
pub type FauxResponseFuture<'a> =
    Pin<Box<dyn Future<Output = Result<AssistantMessage, String>> + Send + 'a>>;

/// Scripted response factory: receives the context, request options, a snapshot
/// of the provider state, and the request model.
pub type FauxResponseFactory = Arc<
    dyn for<'a> Fn(
            &'a Context,
            Option<&'a SimpleStreamOptions>,
            &'a FauxProviderState,
            &'a Model,
        ) -> FauxResponseFuture<'a>
        + Send
        + Sync,
>;

/// One scripted step: a fixed message or a factory. The message variant is
/// boxed to keep the enum small (assistant messages carry content and usage).
#[derive(Clone)]
pub enum FauxResponseStep {
    Message(Box<AssistantMessage>),
    Factory(FauxResponseFactory),
}

impl From<AssistantMessage> for FauxResponseStep {
    fn from(message: AssistantMessage) -> Self {
        FauxResponseStep::Message(Box::new(message))
    }
}

/// Deferred-response pacing options.
#[derive(Clone, Copy, Debug, Default)]
pub struct FauxDeferredOptions {
    /// Number of fetches that return the original handle before the scripted
    /// response becomes ready.
    pub pending_fetches: Option<u32>,
    pub poll_after_ms: Option<u64>,
}

/// Chunk size bounds for streamed deltas.
#[derive(Clone, Copy, Debug, Default)]
pub struct FauxTokenSize {
    pub min: Option<usize>,
    pub max: Option<usize>,
}

/// Options for [`create_faux_core`], [`faux_provider`], and
/// [`crate::compat::register_faux_provider`].
#[derive(Clone, Debug, Default)]
pub struct RegisterFauxProviderOptions {
    pub api: Option<String>,
    pub provider: Option<String>,
    /// Empty or `None` registers the default single model.
    pub models: Option<Vec<FauxModelDefinition>>,
    pub deferred: Option<FauxDeferredOptions>,
    pub tokens_per_second: Option<f64>,
    pub token_size: Option<FauxTokenSize>,
}

/// One in-flight deferred response.
struct DeferredEntry {
    handle: DeferredHandle,
    step: FauxResponseStep,
    context: Context,
    options: Option<SimpleStreamOptions>,
    model: Model,
    pending_fetches: u32,
    cancelled: bool,
    final_message: Option<AssistantMessage>,
}

/// Shared state of one faux provider core.
struct FauxInner {
    api: String,
    provider: String,
    min_token_size: usize,
    max_token_size: usize,
    tokens_per_second: Option<f64>,
    deferred_pending_fetches: u32,
    deferred_poll_after_ms: Option<u64>,
    state: Arc<Mutex<FauxProviderState>>,
    pending_responses: Mutex<VecDeque<FauxResponseStep>>,
    prompt_cache: Mutex<HashMap<String, String>>,
    deferred_responses: Mutex<HashMap<String, DeferredEntry>>,
}

/// The faux provider core: implements [`ProviderStreams`] over scripted state.
pub struct FauxCore {
    /// Registry api id (random when [`RegisterFauxProviderOptions::api`] is unset).
    pub api: String,
    /// Provider id.
    pub provider: String,
    /// Registered models; never empty.
    pub models: Vec<Model>,
    inner: Arc<FauxInner>,
}

/// Api-key auth that is always configured with empty request auth.
struct FauxAuth;

#[async_trait]
impl ApiKeyAuth for FauxAuth {
    fn name(&self) -> &str {
        "Faux"
    }

    async fn resolve(
        &self,
        _ctx: &dyn AuthContext,
        _credential: Option<&ApiKeyCredential>,
        _token: &CancellationToken,
    ) -> Result<Option<AuthResult>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Some(AuthResult::default()))
    }
}

fn build_model(definition: &FauxModelDefinition, api: &str, provider: &str) -> Model {
    Model {
        id: definition.id.clone(),
        name: definition
            .name
            .clone()
            .unwrap_or_else(|| definition.id.clone()),
        api: Api::from(api),
        provider: ProviderId::new(provider),
        base_url: DEFAULT_BASE_URL.to_owned(),
        reasoning: definition.reasoning.unwrap_or(false),
        thinking_level_map: None,
        input: definition
            .input
            .clone()
            .unwrap_or_else(|| vec![InputModality::Text, InputModality::Image]),
        cost: ModelCost {
            rates: definition.cost.unwrap_or_default(),
            tiers: Vec::new(),
        },
        context_window: definition.context_window.unwrap_or(128_000),
        max_tokens: definition.max_tokens.unwrap_or(16_384),
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

/// Build the faux core: scripted-response state plus stream behavior.
pub fn create_faux_core(options: &RegisterFauxProviderOptions) -> FauxCore {
    let api = options
        .api
        .clone()
        .unwrap_or_else(|| random_id(DEFAULT_API));
    let provider = options
        .provider
        .clone()
        .unwrap_or_else(|| DEFAULT_PROVIDER.to_owned());

    let default_min = options
        .token_size
        .and_then(|size| size.min)
        .unwrap_or(DEFAULT_MIN_TOKEN_SIZE);
    let default_max = options
        .token_size
        .and_then(|size| size.max)
        .unwrap_or(DEFAULT_MAX_TOKEN_SIZE);
    let min_token_size = default_min.min(default_max).max(1);
    let max_token_size = default_min.max(default_max).max(min_token_size);

    let definitions: Vec<FauxModelDefinition> = match &options.models {
        Some(definitions) if !definitions.is_empty() => definitions.clone(),
        _ => vec![FauxModelDefinition {
            id: DEFAULT_MODEL_ID.to_owned(),
            name: Some(DEFAULT_MODEL_NAME.to_owned()),
            reasoning: Some(false),
            input: Some(vec![InputModality::Text, InputModality::Image]),
            cost: Some(ModelCostRates::default()),
            context_window: Some(128_000),
            max_tokens: Some(16_384),
        }],
    };
    let models: Vec<Model> = definitions
        .iter()
        .map(|definition| build_model(definition, &api, &provider))
        .collect();

    FauxCore {
        api: api.clone(),
        provider: provider.clone(),
        models,
        inner: Arc::new(FauxInner {
            api,
            provider,
            min_token_size,
            max_token_size,
            tokens_per_second: options.tokens_per_second,
            deferred_pending_fetches: options
                .deferred
                .and_then(|deferred| deferred.pending_fetches)
                .unwrap_or(0),
            deferred_poll_after_ms: options.deferred.and_then(|deferred| deferred.poll_after_ms),
            state: Arc::new(Mutex::new(FauxProviderState::default())),
            pending_responses: Mutex::new(VecDeque::new()),
            prompt_cache: Mutex::new(HashMap::new()),
            deferred_responses: Mutex::new(HashMap::new()),
        }),
    }
}

impl FauxCore {
    /// The shared provider state handle.
    pub fn state(&self) -> Arc<Mutex<FauxProviderState>> {
        self.inner.state.clone()
    }

    /// The first registered model.
    pub fn get_model(&self) -> Model {
        self.models.first().cloned().unwrap_or_else(|| {
            build_model(
                &FauxModelDefinition {
                    id: DEFAULT_MODEL_ID.to_owned(),
                    ..Default::default()
                },
                &self.api,
                &self.provider,
            )
        })
    }

    /// Look up a registered model by id.
    pub fn get_model_by_id(&self, model_id: &str) -> Option<Model> {
        self.models
            .iter()
            .find(|model| model.id == model_id)
            .cloned()
    }

    /// Replace the queued scripted responses.
    pub fn set_responses(&self, responses: Vec<FauxResponseStep>) {
        let mut queue = lock(&self.inner.pending_responses);
        queue.clear();
        queue.extend(responses);
    }

    /// Append scripted responses to the queue.
    pub fn append_responses(&self, responses: Vec<FauxResponseStep>) {
        lock(&self.inner.pending_responses).extend(responses);
    }

    /// Number of queued responses not yet consumed.
    pub fn pending_response_count(&self) -> usize {
        lock(&self.inner.pending_responses).len()
    }

    fn stream_inner(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&SimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        let (mut sender, stream) = assistant_message_event_stream();
        let core = self.inner.clone();
        let api = self.api.clone();
        let provider = self.provider.clone();
        let model = model.clone();
        let model_id = model.id.clone();
        let context = context.clone();
        let options = options.cloned();

        // Consume a queued response and count the call synchronously, matching
        // the TypeScript `shift()` + `callCount++` ordering.
        let step = lock(&core.pending_responses).pop_front();
        lock(&core.state).call_count += 1;

        crate::execution_timing::spawn(async move {
            // Errors thrown while streaming (e.g. a scripted message without a
            // terminal stop reason) become a terminal error event, like the
            // TypeScript `queueMicrotask` try/catch.
            if let Err(error) = run_stream(core, model, context, options, step, &mut sender).await {
                let message = create_error_message(&error, &api, &provider, &model_id);
                sender
                    .push(AssistantMessageEvent::Error {
                        reason: ErrorReason::Error,
                        error: message.clone(),
                    })
                    .await;
                sender.end(message);
            }
        });

        stream
    }
}

#[async_trait]
impl ProviderStreams for FauxCore {
    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: &StreamOptions,
    ) -> AssistantMessageEventStream {
        // The faux core treats both entry points identically; the deferred flag
        // only exists on simple options.
        let simple = SimpleStreamOptions {
            base: options.clone(),
            ..Default::default()
        };
        self.stream_inner(model, context, Some(&simple))
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        self.stream_inner(model, context, Some(options))
    }

    fn supports_fetch_deferred(&self) -> bool {
        true
    }

    fn fetch_deferred(
        &self,
        model: &Model,
        handle: &DeferredHandle,
        options: &StreamOptions,
    ) -> AssistantMessageEventStream {
        let (mut sender, stream) = assistant_message_event_stream();
        let core = self.inner.clone();
        let api = self.api.clone();
        let provider = self.provider.clone();
        let model = model.clone();
        let handle = handle.clone();
        let signal = options.base.signal.clone();
        lock(&core.state).deferred_fetch_count += 1;

        crate::execution_timing::spawn(async move {
            match resolve_deferred(&core, &model, &handle).await {
                Ok(message) => {
                    if let Err(error) =
                        stream_with_deltas(&core, &mut sender, &message, signal.as_ref()).await
                    {
                        let message = create_error_message(&error, &api, &provider, &model.id);
                        sender
                            .push(AssistantMessageEvent::Error {
                                reason: ErrorReason::Error,
                                error: message.clone(),
                            })
                            .await;
                        sender.end(message);
                    }
                }
                Err(error) => {
                    let message = create_error_message(&error, &api, &provider, &model.id);
                    sender
                        .push(AssistantMessageEvent::Error {
                            reason: ErrorReason::Error,
                            error: message.clone(),
                        })
                        .await;
                    sender.end(message);
                }
            }
        });

        stream
    }

    async fn cancel_deferred(
        &self,
        _model: &Model,
        handle: &DeferredHandle,
        _options: &StreamOptions,
    ) -> Result<(), crate::error::ModelsError> {
        lock(&self.inner.state)
            .cancelled_deferred
            .push(handle.clone());
        if let Some(entry) = lock(&self.inner.deferred_responses).get_mut(&handle.id) {
            entry.cancelled = true;
        }
        Ok(())
    }
}

/// Drive one scripted stream: observer, queue exhaustion, deferred handoff, or
/// a resolved scripted message. Errors surface as a terminal error event, like
/// the TypeScript `queueMicrotask` try/catch.
async fn run_stream(
    core: Arc<FauxInner>,
    model: Model,
    context: Context,
    options: Option<SimpleStreamOptions>,
    step: Option<FauxResponseStep>,
    sender: &mut AssistantMessageEventSender,
) -> Result<(), String> {
    if let Some(observer) = options
        .as_ref()
        .and_then(|options| options.base.on_response.clone())
    {
        observer
            .on_response(
                &ProviderResponse {
                    status: 200,
                    headers: HashMap::new(),
                },
                &model,
            )
            .await;
    }

    let Some(step) = step else {
        let message = with_usage_estimate(
            create_error_message(
                "No more faux responses queued",
                &core.api,
                &core.provider,
                &model.id,
            ),
            &context,
            options.as_ref(),
            &core,
        );
        sender
            .push(AssistantMessageEvent::Error {
                reason: ErrorReason::Error,
                error: message.clone(),
            })
            .await;
        sender.end(message);
        return Ok(());
    };

    let signal = options
        .as_ref()
        .and_then(|options| options.base.base.signal.clone());

    if options
        .as_ref()
        .and_then(|options| options.deferred)
        .is_some()
    {
        let handle = DeferredHandle {
            provider: model.provider.to_string(),
            model_id: model.id.clone(),
            api: model.api.to_string(),
            id: random_id("deferred"),
            expires_at: None,
            poll_after_ms: core.deferred_poll_after_ms,
            data: None,
        };
        let entry = DeferredEntry {
            handle: handle.clone(),
            step,
            context: context.clone(),
            options: options.clone(),
            model: model.clone(),
            pending_fetches: core.deferred_pending_fetches,
            cancelled: false,
            final_message: None,
        };
        lock(&core.deferred_responses).insert(handle.id.clone(), entry);
        stream_with_deltas(
            &core,
            sender,
            &create_deferred_message(&model, handle),
            signal.as_ref(),
        )
        .await?;
        return Ok(());
    }

    match resolve_response(&core, &step, &context, options.as_ref(), &model).await {
        Ok(message) => {
            stream_with_deltas(&core, sender, &message, signal.as_ref()).await?;
            Ok(())
        }
        Err(error) => {
            let message = create_error_message(&error, &core.api, &core.provider, &model.id);
            sender
                .push(AssistantMessageEvent::Error {
                    reason: ErrorReason::Error,
                    error: message.clone(),
                })
                .await;
            sender.end(message);
            Ok(())
        }
    }
}

/// Resolve a deferred fetch: still-pending handles replay the deferred message;
/// otherwise the scripted response is resolved once and cached.
async fn resolve_deferred(
    core: &Arc<FauxInner>,
    model: &Model,
    handle: &DeferredHandle,
) -> Result<AssistantMessage, String> {
    enum Outcome {
        Pending(DeferredHandle),
        Resolve(
            Box<(
                FauxResponseStep,
                Context,
                Option<SimpleStreamOptions>,
                Model,
            )>,
        ),
        Final(Box<AssistantMessage>),
    }

    let outcome = {
        let mut entries = lock(&core.deferred_responses);
        let entry = entries
            .get_mut(&handle.id)
            .filter(|entry| {
                entry.handle.provider == handle.provider
                    && entry.handle.model_id == handle.model_id
                    && entry.handle.api == handle.api
            })
            .ok_or_else(|| format!("Unknown faux deferred response: {}", handle.id))?;
        if entry.cancelled {
            return Err(format!(
                "Faux deferred response was cancelled: {}",
                handle.id
            ));
        }
        if entry.pending_fetches > 0 {
            entry.pending_fetches -= 1;
            Outcome::Pending(entry.handle.clone())
        } else if let Some(final_message) = &entry.final_message {
            Outcome::Final(Box::new(final_message.clone()))
        } else {
            Outcome::Resolve(Box::new((
                entry.step.clone(),
                entry.context.clone(),
                entry.options.clone(),
                entry.model.clone(),
            )))
        }
    };

    match outcome {
        Outcome::Pending(handle) => Ok(create_deferred_message(model, handle)),
        Outcome::Final(message) => Ok(*message),
        Outcome::Resolve(resolved) => {
            let (step, context, submission_options, entry_model) = *resolved;
            // Strip submission-only fields, like the TypeScript destructure.
            let mut submission = submission_options.unwrap_or_default();
            submission.deferred = None;
            submission.base.base.signal = None;
            submission.base.on_response = None;
            let final_message = match resolve_response(
                core,
                &step,
                &context,
                Some(&submission),
                &entry_model,
            )
            .await
            {
                Ok(message) => message,
                Err(error) => {
                    create_error_message(&error, &core.api, &core.provider, &entry_model.id)
                }
            };
            if let Some(entry) = lock(&core.deferred_responses).get_mut(&handle.id) {
                entry.final_message = Some(final_message.clone());
            }
            Ok(final_message)
        }
    }
}

/// Resolve one scripted step and attach the usage estimate.
async fn resolve_response(
    core: &Arc<FauxInner>,
    step: &FauxResponseStep,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
    model: &Model,
) -> Result<AssistantMessage, String> {
    let resolved = match step {
        FauxResponseStep::Message(message) => (**message).clone(),
        FauxResponseStep::Factory(factory) => {
            let state = lock(&core.state).clone();
            factory(context, options, &state, model).await?
        }
    };
    Ok(with_usage_estimate(
        clone_message(&resolved, &core.api, &core.provider, &model.id),
        context,
        options,
        core,
    ))
}

/// Rebind a scripted message to the core's api/provider and the request model.
fn clone_message(
    message: &AssistantMessage,
    api: &str,
    provider: &str,
    model_id: &str,
) -> AssistantMessage {
    AssistantMessage {
        api: Api::from(api),
        provider: ProviderId::new(provider),
        model: model_id.to_owned(),
        ..message.clone()
    }
}

/// Terminal error message with empty content and zero usage.
fn create_error_message(
    error: impl std::fmt::Display,
    api: &str,
    provider: &str,
    model_id: &str,
) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: Api::from(api),
        provider: ProviderId::new(provider),
        model: model_id.to_owned(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: default_usage(),
        stop_reason: StopReason::Error,
        deferred: None,
        error_message: Some(error.to_string()),
        raw_stop_reason: None,
        end_turn: None,
        timestamp: now_ms(),
    }
}

/// Empty deferred message carrying the durable handle.
fn create_deferred_message(model: &Model, handle: DeferredHandle) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: default_usage(),
        stop_reason: StopReason::Deferred,
        deferred: Some(handle),
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: now_ms(),
    }
}

/// Push the aborted terminal error derived from a partial message.
async fn push_aborted(sender: &mut AssistantMessageEventSender, partial: &AssistantMessage) {
    let mut aborted = partial.clone();
    aborted.stop_reason = StopReason::Aborted;
    aborted.error_message = Some("Request was aborted".to_owned());
    aborted.timestamp = now_ms();
    sender
        .push(AssistantMessageEvent::Error {
            reason: ErrorReason::Aborted,
            error: aborted.clone(),
        })
        .await;
    sender.end(aborted);
}

/// Stream one message as start/per-block delta events plus a terminal event.
/// Returns `Err` when the scripted message has no terminal stop reason, which
/// the caller turns into a terminal error event.
async fn stream_with_deltas(
    core: &Arc<FauxInner>,
    sender: &mut AssistantMessageEventSender,
    message: &AssistantMessage,
    signal: Option<&CancellationToken>,
) -> Result<(), String> {
    let mut partial = AssistantMessage {
        content: Vec::new(),
        stop_reason: StopReason::Pending,
        ..message.clone()
    };

    let aborted = || signal.is_some_and(|token| token.is_cancelled());
    if aborted() {
        push_aborted(sender, &partial).await;
        return Ok(());
    }

    sender
        .push(AssistantMessageEvent::Start {
            partial: partial.clone(),
        })
        .await;

    for (index, block) in message.content.iter().enumerate() {
        if aborted() {
            push_aborted(sender, &partial).await;
            return Ok(());
        }

        match block {
            AssistantContent::Thinking(thinking) => {
                partial
                    .content
                    .push(AssistantContent::Thinking(ThinkingContent {
                        thinking: String::new(),
                        thinking_signature: None,
                        redacted: false,
                    }));
                sender
                    .push(AssistantMessageEvent::ThinkingStart {
                        content_index: index,
                        partial: partial.clone(),
                    })
                    .await;
                for chunk in split_string_by_token_size(
                    &thinking.thinking,
                    core.min_token_size,
                    core.max_token_size,
                ) {
                    schedule_chunk(&chunk, core.tokens_per_second, signal).await;
                    if aborted() {
                        push_aborted(sender, &partial).await;
                        return Ok(());
                    }
                    if let Some(AssistantContent::Thinking(block)) = partial.content.get_mut(index)
                    {
                        block.thinking.push_str(&chunk);
                    }
                    sender
                        .push(AssistantMessageEvent::ThinkingDelta {
                            content_index: index,
                            delta: chunk,
                            partial: partial.clone(),
                        })
                        .await;
                }
                sender
                    .push(AssistantMessageEvent::ThinkingEnd {
                        content_index: index,
                        content: thinking_text_at(&partial.content, index),
                        partial: partial.clone(),
                    })
                    .await;
            }
            AssistantContent::Text(text) => {
                partial.content.push(AssistantContent::Text(TextContent {
                    text: String::new(),
                    text_signature: None,
                }));
                sender
                    .push(AssistantMessageEvent::TextStart {
                        content_index: index,
                        partial: partial.clone(),
                    })
                    .await;
                for chunk in
                    split_string_by_token_size(&text.text, core.min_token_size, core.max_token_size)
                {
                    schedule_chunk(&chunk, core.tokens_per_second, signal).await;
                    if aborted() {
                        push_aborted(sender, &partial).await;
                        return Ok(());
                    }
                    if let Some(AssistantContent::Text(block)) = partial.content.get_mut(index) {
                        block.text.push_str(&chunk);
                    }
                    sender
                        .push(AssistantMessageEvent::TextDelta {
                            content_index: index,
                            delta: chunk,
                            partial: partial.clone(),
                        })
                        .await;
                }
                sender
                    .push(AssistantMessageEvent::TextEnd {
                        content_index: index,
                        content: text.text.clone(),
                        partial: partial.clone(),
                    })
                    .await;
            }
            AssistantContent::ToolCall(call) => {
                partial.content.push(AssistantContent::ToolCall(ToolCall {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    arguments: serde_json::Map::new(),
                    thought_signature: None,
                    namespace: None,
                }));
                sender
                    .push(AssistantMessageEvent::ToolCallStart {
                        content_index: index,
                        partial: partial.clone(),
                    })
                    .await;
                let arguments = serde_json::to_string(&call.arguments).unwrap_or_default();
                for chunk in
                    split_string_by_token_size(&arguments, core.min_token_size, core.max_token_size)
                {
                    schedule_chunk(&chunk, core.tokens_per_second, signal).await;
                    if aborted() {
                        push_aborted(sender, &partial).await;
                        return Ok(());
                    }
                    sender
                        .push(AssistantMessageEvent::ToolCallDelta {
                            content_index: index,
                            delta: chunk,
                            partial: partial.clone(),
                        })
                        .await;
                }
                if let Some(AssistantContent::ToolCall(block)) = partial.content.get_mut(index) {
                    block.arguments = call.arguments.clone();
                }
                sender
                    .push(AssistantMessageEvent::ToolCallEnd {
                        content_index: index,
                        tool_call: call.clone(),
                        partial: partial.clone(),
                    })
                    .await;
            }
        }
    }

    match message.stop_reason {
        StopReason::Pending => Err("Faux response ended without a stop reason".to_owned()),
        StopReason::Error => {
            sender
                .push(AssistantMessageEvent::Error {
                    reason: ErrorReason::Error,
                    error: message.clone(),
                })
                .await;
            sender.end(message.clone());
            Ok(())
        }
        StopReason::Aborted => {
            sender
                .push(AssistantMessageEvent::Error {
                    reason: ErrorReason::Aborted,
                    error: message.clone(),
                })
                .await;
            sender.end(message.clone());
            Ok(())
        }
        stop_reason => {
            let reason = match stop_reason {
                StopReason::Length => DoneReason::Length,
                StopReason::ToolUse => DoneReason::ToolUse,
                StopReason::Deferred => DoneReason::Deferred,
                _ => DoneReason::Stop,
            };
            sender
                .push(AssistantMessageEvent::Done {
                    reason,
                    message: message.clone(),
                })
                .await;
            sender.end(message.clone());
            Ok(())
        }
    }
}

/// Pace one chunk: a microtask yield when unpaced, otherwise a delay derived
/// from the estimated token count. Cancellation cuts the delay short.
async fn schedule_chunk(
    chunk: &str,
    tokens_per_second: Option<f64>,
    signal: Option<&CancellationToken>,
) {
    match tokens_per_second {
        Some(tokens_per_second) if tokens_per_second > 0.0 => {
            let delay_ms = (estimate_text_tokens(chunk) as f64 / tokens_per_second * 1000.0)
                .round()
                .max(0.0) as u64;
            match signal {
                Some(token) => {
                    let _ = sleep_ms(delay_ms, token).await;
                }
                None => tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await,
            }
        }
        _ => tokio::task::yield_now().await,
    }
}

/// Split text into random-sized chunks of `token_size * 4` chars (at least one).
fn split_string_by_token_size(
    text: &str,
    min_token_size: usize,
    max_token_size: usize,
) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut remaining = text;
    while !remaining.is_empty() {
        let token_size = rand::rng().random_range(min_token_size..=max_token_size);
        let char_size = (token_size * 4).max(1);
        let byte_len = remaining
            .char_indices()
            .nth(char_size)
            .map(|(index, _)| index)
            .unwrap_or(remaining.len());
        chunks.push(remaining[..byte_len].to_owned());
        remaining = &remaining[byte_len..];
    }
    if chunks.is_empty() {
        vec![String::new()]
    } else {
        chunks
    }
}

fn role_name(message: &Message) -> &'static str {
    match message {
        Message::User(_) => "user",
        Message::Assistant(_) => "assistant",
        Message::ToolResult(_) => "toolResult",
    }
}

fn user_content_block_to_text(block: &UserContentBlock) -> String {
    match block {
        UserContentBlock::Text(text) => text.text.clone(),
        UserContentBlock::Image(image) => {
            format!("[image:{}:{}]", image.mime_type, image.data.chars().count())
        }
    }
}

fn user_content_to_text(content: &UserContent) -> String {
    match content {
        UserContent::Text(text) => text.clone(),
        UserContent::Blocks(blocks) => blocks
            .iter()
            .map(user_content_block_to_text)
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn assistant_content_to_text(content: &[AssistantContent]) -> String {
    content
        .iter()
        .map(|block| match block {
            AssistantContent::Text(text) => text.text.clone(),
            AssistantContent::Thinking(thinking) => thinking.thinking.clone(),
            AssistantContent::ToolCall(call) => {
                format!(
                    "{}:{}",
                    call.name,
                    serde_json::to_string(&call.arguments).unwrap_or_default()
                )
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn message_to_text(message: &Message) -> String {
    match message {
        Message::User(message) => user_content_to_text(&message.content),
        Message::Assistant(message) => assistant_content_to_text(&message.content),
        Message::ToolResult(message) => {
            let mut parts = vec![message.tool_name.clone()];
            parts.extend(message.content.iter().map(user_content_block_to_text));
            parts.join("\n")
        }
    }
}

/// Serialize a context to the text the faux usage estimate tokenizes.
fn serialize_context(context: &Context) -> String {
    let mut parts = Vec::new();
    if let Some(system_prompt) = &context.system_prompt {
        parts.push(format!("system:{system_prompt}"));
    }
    for message in &context.messages {
        parts.push(format!(
            "{}:{}",
            role_name(message),
            message_to_text(message)
        ));
    }
    if let Some(tools) = &context.tools
        && !tools.is_empty()
    {
        parts.push(format!(
            "tools:{}",
            serde_json::to_string(tools).unwrap_or_default()
        ));
    }
    parts.join("\n\n")
}

/// Byte length of the common character prefix of two strings.
fn common_prefix_byte_len(a: &str, b: &str) -> usize {
    let mut byte_len = 0;
    for (left, right) in a.chars().zip(b.chars()) {
        if left != right {
            break;
        }
        byte_len += left.len_utf8();
    }
    byte_len
}

/// Attach estimated usage (prompt/output tokens plus the per-session prompt
/// cache simulation) to a message.
fn with_usage_estimate(
    message: AssistantMessage,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
    core: &Arc<FauxInner>,
) -> AssistantMessage {
    let prompt_text = serialize_context(context);
    let prompt_tokens = estimate_text_tokens(&prompt_text);
    let output_tokens = estimate_text_tokens(&assistant_content_to_text(&message.content));
    let mut input = prompt_tokens;
    let mut cache_read = 0;
    let mut cache_write = 0;

    let session_id = options.and_then(|options| options.base.session_id.clone());
    let cache_disabled = matches!(
        options.and_then(|options| options.base.cache_retention),
        Some(CacheRetention::None)
    );

    if let Some(session_id) = session_id
        && !cache_disabled
    {
        let mut cache = lock(&core.prompt_cache);
        match cache.get(&session_id).cloned() {
            Some(previous_prompt) => {
                let cached = common_prefix_byte_len(&previous_prompt, &prompt_text);
                cache_read = estimate_text_tokens(&previous_prompt[..cached]);
                cache_write = estimate_text_tokens(&prompt_text[cached..]);
                input = prompt_tokens.saturating_sub(cache_read);
            }
            None => cache_write = prompt_tokens,
        }
        cache.insert(session_id, prompt_text);
    }

    AssistantMessage {
        usage: Usage {
            input,
            output: output_tokens,
            cache_read,
            cache_write,
            total_tokens: input + output_tokens + cache_read + cache_write,
            cost: UsageCost::default(),
            ..message.usage
        },
        ..message
    }
}

/// Faux provider handle: an explicit `Models` collection plus scripted-response
/// controls (the TypeScript `fauxProvider()`).
pub struct FauxProviderHandle {
    /// The provider to register with a `Models` collection.
    pub provider: Arc<Provider>,
    /// The underlying core (also drives the provider's api dispatch).
    pub core: Arc<FauxCore>,
}

impl FauxProviderHandle {
    pub fn api(&self) -> &str {
        &self.core.api
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
}

/// Build a faux provider on an explicit `Models` collection:
///
/// ```ignore
/// let faux = faux_provider(&RegisterFauxProviderOptions::default());
/// let models = Models::new(CreateModelsOptions::new());
/// models.set_provider(faux.provider.clone());
/// faux.set_responses(vec![faux_assistant_message("hi", Default::default()).into()]);
/// ```
pub fn faux_provider(options: &RegisterFauxProviderOptions) -> FauxProviderHandle {
    let core = Arc::new(create_faux_core(options));
    let provider = create_provider(CreateProviderOptions {
        id: ProviderId::new(&core.provider),
        name: None,
        base_url: None,
        headers: None,
        auth: ProviderAuth::with_api_key(FauxAuth),
        models: core.models.clone(),
        fetch_models: None,
        filter_models: None,
        api: ApiDispatch::Single(core.clone()),
    });
    FauxProviderHandle { provider, core }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compat;
    use crate::types::Tool;
    use futures::StreamExt;

    fn context_text(text: &str) -> Context {
        Context {
            system_prompt: None,
            messages: vec![Message::user(text, 1)],
            tools: None,
        }
    }

    fn message_options() -> FauxAssistantMessageOptions {
        FauxAssistantMessageOptions::default()
    }

    async fn collect_events(mut stream: AssistantMessageEventStream) -> Vec<AssistantMessageEvent> {
        let mut events = Vec::new();
        while let Some(event) = stream.next().await {
            events.push(event);
        }
        events
    }

    fn event_name(event: &AssistantMessageEvent) -> &'static str {
        match event {
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
        }
    }

    fn event_names(events: &[AssistantMessageEvent]) -> Vec<&'static str> {
        events.iter().map(event_name).collect()
    }

    /// Unregisters the faux provider when the test ends, mirroring the
    /// TypeScript `afterEach` cleanup.
    struct RegistrationGuard(compat::FauxProviderRegistration);

    impl std::ops::Deref for RegistrationGuard {
        type Target = compat::FauxProviderRegistration;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl Drop for RegistrationGuard {
        fn drop(&mut self) {
            self.0.unregister();
        }
    }

    /// Compare content blocks as JSON (`AssistantContent` has no `PartialEq`).
    fn content_json(content: &[AssistantContent]) -> Json {
        serde_json::to_value(content).unwrap_or(Json::Null)
    }

    /// The JSON shape of a single text block.
    fn text_json(text: &str) -> Json {
        serde_json::json!([{ "type": "text", "text": text }])
    }

    /// Assert a message carries exactly one text block with `text`.
    fn assert_text(message: &AssistantMessage, text: &str) {
        assert_eq!(content_json(&message.content), text_json(text));
    }

    fn options_with_signal(token: CancellationToken) -> StreamOptions {
        StreamOptions {
            base: crate::types::ProviderRequestOptions {
                signal: Some(token),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn options_with_session(session_id: &str, retention: CacheRetention) -> StreamOptions {
        StreamOptions {
            session_id: Some(session_id.to_owned()),
            cache_retention: Some(retention),
            ..Default::default()
        }
    }

    /// Serialize registry-touching tests: the api registry is process-global.
    /// The guard is async because the guarded section awaits provider streams.
    async fn registry_guard() -> tokio::sync::MutexGuard<'static, ()> {
        compat::registry_test_lock().lock().await
    }

    async fn register(options: &RegisterFauxProviderOptions) -> RegistrationGuard {
        let _serialized = registry_guard().await;
        RegistrationGuard(compat::register_faux_provider(options))
    }

    #[tokio::test]
    async fn registers_a_custom_provider_and_estimates_usage() {
        let _serialized = registry_guard().await;
        let registration = compat::register_faux_provider(&RegisterFauxProviderOptions::default());
        registration.set_responses(vec![
            faux_assistant_message("hello world", message_options()).into(),
        ]);

        let context = Context {
            system_prompt: Some("Be concise.".to_owned()),
            messages: vec![Message::user("hi there", 1)],
            tools: None,
        };

        let response = compat::complete(&registration.get_model(), &context, None).await;
        assert_text(&response, "hello world");
        assert!(response.usage.input > 0);
        assert!(response.usage.output > 0);
        assert_eq!(
            response.usage.total_tokens,
            response.usage.input + response.usage.output
        );
        assert_eq!(lock(&registration.state()).call_count, 1);
    }

    #[tokio::test]
    async fn supports_helper_blocks_for_text_thinking_and_tool_calls() {
        let registration = register(&RegisterFauxProviderOptions::default()).await;
        let mut arguments = serde_json::Map::new();
        arguments.insert("text".to_owned(), Json::String("hi".to_owned()));
        registration.set_responses(vec![
            faux_assistant_message(
                vec![
                    FauxContentBlock::Thinking(faux_thinking("think")),
                    FauxContentBlock::ToolCall(faux_tool_call("echo", arguments.clone(), None)),
                    FauxContentBlock::Text(faux_text("done")),
                ],
                FauxAssistantMessageOptions {
                    stop_reason: Some(StopReason::ToolUse),
                    ..Default::default()
                },
            )
            .into(),
        ]);

        let response = compat::complete(&registration.get_model(), &context_text("hi"), None).await;

        let tool_call_id = response.content[1]
            .tool_call()
            .map(|call| call.id.clone())
            .unwrap_or_default();
        assert_eq!(
            content_json(&response.content),
            serde_json::json!([
                { "type": "thinking", "thinking": "think" },
                { "type": "toolCall", "id": tool_call_id, "name": "echo", "arguments": { "text": "hi" } },
                { "type": "text", "text": "done" },
            ])
        );
        assert_eq!(response.stop_reason, StopReason::ToolUse);
    }

    #[tokio::test]
    async fn supports_multiple_models_with_model_aware_factories() {
        let registration = register(&RegisterFauxProviderOptions {
            models: Some(vec![
                FauxModelDefinition {
                    id: "faux-fast".to_owned(),
                    name: Some("Faux Fast".to_owned()),
                    reasoning: Some(false),
                    ..Default::default()
                },
                FauxModelDefinition {
                    id: "faux-thinker".to_owned(),
                    name: Some("Faux Thinker".to_owned()),
                    reasoning: Some(true),
                    ..Default::default()
                },
            ]),
            ..Default::default()
        })
        .await;
        registration.set_responses(vec![
            FauxResponseStep::Factory(Arc::new(|_context, _options, _state, model| {
                Box::pin(async move {
                    Ok(faux_assistant_message(
                        format!("{}:{}", model.id, model.reasoning),
                        message_options(),
                    ))
                })
            })),
            FauxResponseStep::Factory(Arc::new(|_context, _options, _state, model| {
                Box::pin(async move {
                    Ok(faux_assistant_message(
                        format!("{}:{}", model.id, model.reasoning),
                        message_options(),
                    ))
                })
            })),
        ]);

        assert_eq!(
            registration
                .models()
                .iter()
                .map(|model| model.id.clone())
                .collect::<Vec<_>>(),
            vec!["faux-fast", "faux-thinker"]
        );
        assert_eq!(registration.get_model().id, registration.models()[0].id);
        assert_eq!(
            registration
                .get_model_by_id("faux-fast")
                .map(|model| model.reasoning),
            Some(false)
        );
        assert_eq!(
            registration
                .get_model_by_id("faux-thinker")
                .map(|model| model.reasoning),
            Some(true)
        );

        let fast = compat::complete(
            &registration.get_model_by_id("faux-fast").unwrap(),
            &context_text("hi"),
            None,
        )
        .await;
        let thinker = compat::complete(
            &registration.get_model_by_id("faux-thinker").unwrap(),
            &context_text("hi"),
            None,
        )
        .await;

        assert_text(&fast, "faux-fast:false");
        assert_text(&thinker, "faux-thinker:true");
    }

    #[tokio::test]
    async fn rewrites_api_provider_and_model_on_returned_messages() {
        let registration = register(&RegisterFauxProviderOptions {
            api: Some("faux:test".to_owned()),
            provider: Some("faux-provider".to_owned()),
            models: Some(vec![FauxModelDefinition {
                id: "faux-model".to_owned(),
                ..Default::default()
            }]),
            ..Default::default()
        })
        .await;
        registration.set_responses(vec![
            faux_assistant_message("hello", message_options()).into(),
        ]);

        let response = compat::complete(&registration.get_model(), &context_text("hi"), None).await;

        assert_eq!(response.api.as_str(), "faux:test");
        assert_eq!(response.provider.as_str(), "faux-provider");
        assert_eq!(response.model, "faux-model");
    }

    #[tokio::test]
    async fn consumes_queued_responses_in_order_and_errors_when_exhausted() {
        let registration = register(&RegisterFauxProviderOptions::default()).await;
        registration.set_responses(vec![
            faux_assistant_message("first", message_options()).into(),
            faux_assistant_message("second", message_options()).into(),
        ]);

        let context = context_text("hi");
        let first = compat::complete(&registration.get_model(), &context, None).await;
        let second = compat::complete(&registration.get_model(), &context, None).await;
        let exhausted = compat::complete(&registration.get_model(), &context, None).await;

        assert_text(&first, "first");
        assert_text(&second, "second");
        assert_eq!(exhausted.stop_reason, StopReason::Error);
        assert_eq!(
            exhausted.error_message.as_deref(),
            Some("No more faux responses queued")
        );
        assert_eq!(registration.pending_response_count(), 0);
        assert_eq!(lock(&registration.state()).call_count, 3);
    }

    #[tokio::test]
    async fn can_replace_and_append_queued_responses() {
        let registration = register(&RegisterFauxProviderOptions::default()).await;
        registration.set_responses(vec![
            faux_assistant_message("first", message_options()).into(),
        ]);

        let context = context_text("hi");
        assert_text(
            &compat::complete(&registration.get_model(), &context, None).await,
            "first",
        );
        assert_eq!(registration.pending_response_count(), 0);

        registration.set_responses(vec![
            faux_assistant_message("second", message_options()).into(),
        ]);
        assert_eq!(registration.pending_response_count(), 1);
        assert_text(
            &compat::complete(&registration.get_model(), &context, None).await,
            "second",
        );

        registration.append_responses(vec![
            faux_assistant_message("third", message_options()).into(),
            faux_assistant_message("fourth", message_options()).into(),
        ]);
        assert_eq!(registration.pending_response_count(), 2);
        assert_text(
            &compat::complete(&registration.get_model(), &context, None).await,
            "third",
        );
        assert_text(
            &compat::complete(&registration.get_model(), &context, None).await,
            "fourth",
        );
        assert_eq!(registration.pending_response_count(), 0);
    }

    #[tokio::test]
    async fn supports_async_response_factories() {
        let registration = register(&RegisterFauxProviderOptions::default()).await;
        registration.set_responses(vec![FauxResponseStep::Factory(Arc::new(
            |context, _options, state, _model| {
                Box::pin(async move {
                    Ok(faux_assistant_message(
                        format!("{}:{}", context.messages.len(), state.call_count),
                        message_options(),
                    ))
                })
            },
        ))]);

        let response = compat::complete(&registration.get_model(), &context_text("hi"), None).await;
        assert_text(&response, "1:1");
    }

    #[tokio::test]
    async fn emits_an_error_when_a_response_factory_throws() {
        let registration = register(&RegisterFauxProviderOptions::default()).await;
        registration.set_responses(vec![FauxResponseStep::Factory(Arc::new(
            |_context, _options, _state, _model| Box::pin(async { Err("boom".to_owned()) }),
        ))]);

        let events = collect_events(compat::stream(
            &registration.get_model(),
            &context_text("hi"),
            None,
        ))
        .await;

        assert_eq!(events.len(), 1);
        match &events[0] {
            AssistantMessageEvent::Error { error, .. } => {
                assert_eq!(error.stop_reason, StopReason::Error);
                assert_eq!(error.error_message.as_deref(), Some("boom"));
            }
            other => panic!("expected error event, got {}", event_name(other)),
        }
    }

    #[tokio::test]
    async fn rejects_a_queued_response_without_a_terminal_stop_reason() {
        let registration = register(&RegisterFauxProviderOptions::default()).await;
        registration.set_responses(vec![
            faux_assistant_message(
                "partial",
                FauxAssistantMessageOptions {
                    stop_reason: Some(StopReason::Pending),
                    ..Default::default()
                },
            )
            .into(),
        ]);

        let events = collect_events(compat::stream(
            &registration.get_model(),
            &context_text("hi"),
            None,
        ))
        .await;

        assert!(
            !events
                .iter()
                .any(|event| matches!(event, AssistantMessageEvent::Done { .. }))
        );
        match events.last() {
            Some(AssistantMessageEvent::Error { error, .. }) => {
                assert_eq!(error.stop_reason, StopReason::Error);
                assert_eq!(
                    error.error_message.as_deref(),
                    Some("Faux response ended without a stop reason")
                );
            }
            other => panic!(
                "expected terminal error event, got {}",
                event_name(other.unwrap())
            ),
        }
    }

    #[tokio::test]
    async fn estimates_prompt_and_output_tokens_from_serialized_context() {
        let registration = register(&RegisterFauxProviderOptions::default()).await;
        registration.set_responses(vec![
            faux_assistant_message("done", message_options()).into(),
        ]);

        let tool = Tool {
            name: "echo".to_owned(),
            description: "Echo back text".to_owned(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "text": { "type": "string" } },
                "required": ["text"],
                "additionalProperties": false
            }),
            constrained_sampling: None,
        };
        let context = Context {
            system_prompt: Some("sys".to_owned()),
            messages: vec![
                Message::User(crate::types::UserMessage {
                    content: UserContent::Blocks(vec![
                        UserContentBlock::Text(faux_text("hello")),
                        UserContentBlock::Image(crate::types::ImageContent {
                            data: "abcd".to_owned(),
                            mime_type: "image/png".to_owned(),
                        }),
                    ]),
                    timestamp: 1,
                }),
                Message::Assistant(Box::new(faux_assistant_message("prior", message_options()))),
                Message::ToolResult(Box::new(crate::types::ToolResultMessage {
                    tool_call_id: "tool-1".to_owned(),
                    tool_name: "echo".to_owned(),
                    content: vec![UserContentBlock::Text(faux_text("tool out"))],
                    details: None,
                    usage: None,
                    added_tool_names: None,
                    is_error: false,
                    timestamp: 2,
                })),
            ],
            tools: Some(vec![tool.clone()]),
        };

        let response = compat::complete(&registration.get_model(), &context, None).await;

        let prompt_text = [
            "system:sys".to_owned(),
            "user:hello\n[image:image/png:4]".to_owned(),
            "assistant:prior".to_owned(),
            "toolResult:echo\ntool out".to_owned(),
            format!("tools:{}", serde_json::to_string(&[tool]).unwrap()),
        ]
        .join("\n\n");
        let expected_prompt_tokens = estimate_text_tokens(&prompt_text);
        let expected_output_tokens = estimate_text_tokens("done");

        assert_eq!(response.usage.input, expected_prompt_tokens);
        assert_eq!(response.usage.output, expected_output_tokens);
        assert_eq!(response.usage.cache_read, 0);
        assert_eq!(response.usage.cache_write, 0);
        assert_eq!(
            response.usage.total_tokens,
            expected_prompt_tokens + expected_output_tokens
        );
    }

    #[tokio::test]
    async fn does_not_share_cache_across_sessions_or_requests_without_session_id() {
        let registration = register(&RegisterFauxProviderOptions::default()).await;
        registration.set_responses(vec![
            faux_assistant_message("first", message_options()).into(),
            faux_assistant_message("second", message_options()).into(),
            faux_assistant_message("third", message_options()).into(),
        ]);

        let mut context = context_text("hello");
        let options = options_with_session("session-1", CacheRetention::Short);
        let first = compat::complete(&registration.get_model(), &context, Some(&options)).await;
        assert!(first.usage.cache_write > 0);
        context.messages.push(Message::Assistant(Box::new(first)));
        context.messages.push(Message::user("follow up", 2));

        let options = options_with_session("session-2", CacheRetention::Short);
        let second = compat::complete(&registration.get_model(), &context, Some(&options)).await;
        assert_eq!(second.usage.cache_read, 0);
        assert!(second.usage.cache_write > 0);

        let third = compat::complete(&registration.get_model(), &context, None).await;
        assert_eq!(third.usage.cache_read, 0);
        assert_eq!(third.usage.cache_write, 0);
    }

    #[tokio::test]
    async fn simulates_prompt_caching_per_session_id() {
        let registration = register(&RegisterFauxProviderOptions::default()).await;
        registration.set_responses(vec![
            faux_assistant_message("first", message_options()).into(),
            faux_assistant_message("second", message_options()).into(),
        ]);

        let mut context = Context {
            system_prompt: Some("Be concise.".to_owned()),
            messages: vec![Message::user("hello", 1)],
            tools: None,
        };
        let options = options_with_session("session-1", CacheRetention::Short);
        let first = compat::complete(&registration.get_model(), &context, Some(&options)).await;
        assert_eq!(first.usage.cache_read, 0);
        assert!(first.usage.cache_write > 0);

        context.messages.push(Message::Assistant(Box::new(first)));
        context.messages.push(Message::user("follow up", 2));

        let second = compat::complete(&registration.get_model(), &context, Some(&options)).await;
        assert!(second.usage.cache_read > 0);
        assert!(second.usage.input + second.usage.cache_read > second.usage.input);
    }

    #[tokio::test]
    async fn does_not_simulate_caching_when_cache_retention_is_none() {
        let registration = register(&RegisterFauxProviderOptions::default()).await;
        registration.set_responses(vec![
            faux_assistant_message("first", message_options()).into(),
            faux_assistant_message("second", message_options()).into(),
        ]);

        let mut context = context_text("hello");
        let options = options_with_session("session-1", CacheRetention::None);
        let first = compat::complete(&registration.get_model(), &context, Some(&options)).await;
        context
            .messages
            .push(Message::Assistant(Box::new(faux_assistant_message(
                "first",
                message_options(),
            ))));
        context.messages.push(Message::user("follow up", 2));
        let second = compat::complete(&registration.get_model(), &context, Some(&options)).await;
        assert_eq!(second.usage.cache_read, 0);
        assert_eq!(second.usage.cache_write, 0);
        assert_eq!(first.usage.cache_read, 0);
    }

    #[tokio::test]
    async fn streams_thinking_text_and_partial_tool_call_deltas() {
        let registration = register(&RegisterFauxProviderOptions::default()).await;
        let mut arguments = serde_json::Map::new();
        arguments.insert("text".to_owned(), Json::String("hi".to_owned()));
        arguments.insert("count".to_owned(), Json::from(12));
        registration.set_responses(vec![
            faux_assistant_message(
                vec![
                    FauxContentBlock::Thinking(faux_thinking("thinking text")),
                    FauxContentBlock::Text(faux_text("answer text")),
                    FauxContentBlock::ToolCall(faux_tool_call(
                        "echo",
                        arguments,
                        Some("tool-1".to_owned()),
                    )),
                ],
                FauxAssistantMessageOptions {
                    stop_reason: Some(StopReason::ToolUse),
                    ..Default::default()
                },
            )
            .into(),
        ]);

        let mut events = Vec::new();
        let mut tool_call_deltas = Vec::new();
        let mut stream = compat::stream(&registration.get_model(), &context_text("hi"), None);
        while let Some(event) = stream.next().await {
            events.push(event_name(&event).to_owned());
            if let AssistantMessageEvent::ToolCallDelta { delta, .. } = &event {
                tool_call_deltas.push(delta.clone());
            }
        }

        for expected in [
            "thinking_start",
            "thinking_delta",
            "text_start",
            "text_delta",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_end",
        ] {
            assert!(
                events.iter().any(|event| event == expected),
                "missing {expected} in {events:?}"
            );
        }
        assert!(tool_call_deltas.len() > 1);
        let joined: String = tool_call_deltas.concat();
        let parsed: Json = serde_json::from_str(&joined).expect("tool call deltas form valid json");
        let mut expected_arguments = serde_json::Map::new();
        expected_arguments.insert("text".to_owned(), Json::String("hi".to_owned()));
        expected_arguments.insert("count".to_owned(), Json::from(12));
        assert_eq!(parsed, Json::Object(expected_arguments));
    }

    #[tokio::test]
    async fn streams_an_exact_event_order_for_fixed_size_chunks() {
        let registration = register(&RegisterFauxProviderOptions {
            token_size: Some(FauxTokenSize {
                min: Some(1),
                max: Some(1),
            }),
            ..Default::default()
        })
        .await;
        registration.set_responses(vec![
            faux_assistant_message(
                vec![
                    FauxContentBlock::Thinking(faux_thinking("go")),
                    FauxContentBlock::Text(faux_text("ok")),
                    FauxContentBlock::ToolCall(faux_tool_call(
                        "echo",
                        serde_json::Map::new(),
                        Some("tool-1".to_owned()),
                    )),
                ],
                FauxAssistantMessageOptions {
                    stop_reason: Some(StopReason::ToolUse),
                    ..Default::default()
                },
            )
            .into(),
        ]);

        let events = collect_events(compat::stream(
            &registration.get_model(),
            &context_text("hi"),
            None,
        ))
        .await;

        match &events[0] {
            AssistantMessageEvent::Start { partial } => {
                assert_eq!(partial.stop_reason, StopReason::Pending)
            }
            other => panic!("expected start event, got {}", event_name(other)),
        }
        assert_eq!(
            event_names(&events),
            vec![
                "start",
                "thinking_start",
                "thinking_delta",
                "thinking_end",
                "text_start",
                "text_delta",
                "text_end",
                "toolcall_start",
                "toolcall_delta",
                "toolcall_end",
                "done",
            ]
        );
    }

    #[tokio::test]
    async fn streams_multiple_tool_calls_in_one_message() {
        let registration = register(&RegisterFauxProviderOptions::default()).await;
        registration.set_responses(vec![
            faux_assistant_message(
                vec![
                    FauxContentBlock::ToolCall(faux_tool_call(
                        "echo",
                        serde_json::Map::from_iter([(
                            "text".to_owned(),
                            Json::String("one".to_owned()),
                        )]),
                        Some("tool-1".to_owned()),
                    )),
                    FauxContentBlock::ToolCall(faux_tool_call(
                        "echo",
                        serde_json::Map::from_iter([(
                            "text".to_owned(),
                            Json::String("two".to_owned()),
                        )]),
                        Some("tool-2".to_owned()),
                    )),
                ],
                FauxAssistantMessageOptions {
                    stop_reason: Some(StopReason::ToolUse),
                    ..Default::default()
                },
            )
            .into(),
        ]);

        let events = collect_events(compat::stream(
            &registration.get_model(),
            &context_text("hi"),
            None,
        ))
        .await;

        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, AssistantMessageEvent::ToolCallStart { .. }))
                .count(),
            2
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, AssistantMessageEvent::ToolCallEnd { .. }))
                .count(),
            2
        );
    }

    #[tokio::test]
    async fn streams_an_explicit_assistant_error_message_as_a_terminal_error() {
        let registration = register(&RegisterFauxProviderOptions {
            token_size: Some(FauxTokenSize {
                min: Some(2),
                max: Some(2),
            }),
            ..Default::default()
        })
        .await;
        let mut message = faux_assistant_message("partial", message_options());
        message.stop_reason = StopReason::Error;
        message.error_message = Some("upstream failed".to_owned());
        registration.set_responses(vec![message.into()]);

        let events = collect_events(compat::stream(
            &registration.get_model(),
            &context_text("hi"),
            None,
        ))
        .await;

        assert_eq!(
            event_names(&events),
            vec!["start", "text_start", "text_delta", "text_end", "error"]
        );
        match events.last() {
            Some(AssistantMessageEvent::Error { reason, error }) => {
                assert_eq!(*reason, ErrorReason::Error);
                assert_eq!(error.stop_reason, StopReason::Error);
                assert_eq!(error.error_message.as_deref(), Some("upstream failed"));
            }
            other => panic!(
                "expected terminal error event, got {}",
                event_name(other.unwrap())
            ),
        }
    }

    #[tokio::test]
    async fn streams_an_explicit_assistant_aborted_message_as_a_terminal_error() {
        let registration = register(&RegisterFauxProviderOptions {
            token_size: Some(FauxTokenSize {
                min: Some(2),
                max: Some(2),
            }),
            ..Default::default()
        })
        .await;
        let mut message = faux_assistant_message("partial", message_options());
        message.stop_reason = StopReason::Aborted;
        message.error_message = Some("Request was aborted".to_owned());
        registration.set_responses(vec![message.into()]);

        let events = collect_events(compat::stream(
            &registration.get_model(),
            &context_text("hi"),
            None,
        ))
        .await;

        assert_eq!(
            event_names(&events),
            vec!["start", "text_start", "text_delta", "text_end", "error"]
        );
        match events.last() {
            Some(AssistantMessageEvent::Error { reason, error }) => {
                assert_eq!(*reason, ErrorReason::Aborted);
                assert_eq!(error.stop_reason, StopReason::Aborted);
                assert_eq!(error.error_message.as_deref(), Some("Request was aborted"));
            }
            other => panic!(
                "expected terminal error event, got {}",
                event_name(other.unwrap())
            ),
        }
    }

    #[tokio::test]
    async fn supports_aborting_before_the_first_chunk() {
        let registration = register(&RegisterFauxProviderOptions {
            tokens_per_second: Some(50.0),
            token_size: Some(FauxTokenSize {
                min: Some(3),
                max: Some(3),
            }),
            ..Default::default()
        })
        .await;
        registration.set_responses(vec![
            faux_assistant_message("abcdefghijklmnopqrstuvwxyz", message_options()).into(),
        ]);

        let token = CancellationToken::new();
        token.cancel();
        let options = options_with_signal(token);

        let events = collect_events(compat::stream(
            &registration.get_model(),
            &context_text("hi"),
            Some(&options),
        ))
        .await;

        assert_eq!(events.len(), 1);
        match &events[0] {
            AssistantMessageEvent::Error { reason, error } => {
                assert_eq!(*reason, ErrorReason::Aborted);
                assert_eq!(error.stop_reason, StopReason::Aborted);
            }
            other => panic!("expected error event, got {}", event_name(other)),
        }
    }

    #[tokio::test]
    async fn supports_aborting_mid_text_stream_when_paced() {
        let registration = register(&RegisterFauxProviderOptions {
            tokens_per_second: Some(100.0),
            token_size: Some(FauxTokenSize {
                min: Some(3),
                max: Some(3),
            }),
            ..Default::default()
        })
        .await;
        registration.set_responses(vec![
            faux_assistant_message("abcdefghijklmnopqrstuvwxyz", message_options()).into(),
        ]);

        let token = CancellationToken::new();
        let options = options_with_signal(token.clone());
        let mut events = Vec::new();
        let mut text_delta_count = 0;
        let mut stream = compat::stream(
            &registration.get_model(),
            &context_text("hi"),
            Some(&options),
        );
        while let Some(event) = stream.next().await {
            events.push(event_name(&event));
            if matches!(event, AssistantMessageEvent::TextDelta { .. }) {
                text_delta_count += 1;
                token.cancel();
            }
        }

        assert_eq!(text_delta_count, 1);
        assert!(events.contains(&"text_start"));
        assert!(events.contains(&"text_delta"));
        assert!(events.contains(&"error"));
        assert!(!events.contains(&"text_end"));
    }

    #[tokio::test]
    async fn supports_aborting_mid_thinking_stream_when_paced() {
        let registration = register(&RegisterFauxProviderOptions {
            tokens_per_second: Some(100.0),
            token_size: Some(FauxTokenSize {
                min: Some(3),
                max: Some(3),
            }),
            ..Default::default()
        })
        .await;
        let mut message = faux_assistant_message("ignored", message_options());
        message.content = vec![AssistantContent::Thinking(faux_thinking(
            "abcdefghijklmnopqrstuvwxyz",
        ))];
        registration.set_responses(vec![message.into()]);

        let token = CancellationToken::new();
        let options = options_with_signal(token.clone());
        let mut events = Vec::new();
        let mut thinking_delta_count = 0;
        let mut stream = compat::stream(
            &registration.get_model(),
            &context_text("hi"),
            Some(&options),
        );
        while let Some(event) = stream.next().await {
            events.push(event_name(&event));
            if matches!(event, AssistantMessageEvent::ThinkingDelta { .. }) {
                thinking_delta_count += 1;
                token.cancel();
            }
        }

        assert_eq!(thinking_delta_count, 1);
        assert!(events.contains(&"thinking_start"));
        assert!(events.contains(&"thinking_delta"));
        assert!(events.contains(&"error"));
        assert!(!events.contains(&"thinking_end"));
    }

    #[tokio::test]
    async fn supports_aborting_mid_toolcall_stream_when_paced() {
        let registration = register(&RegisterFauxProviderOptions {
            tokens_per_second: Some(100.0),
            token_size: Some(FauxTokenSize {
                min: Some(3),
                max: Some(3),
            }),
            ..Default::default()
        })
        .await;
        let mut arguments = serde_json::Map::new();
        arguments.insert(
            "text".to_owned(),
            Json::String("abcdefghijklmnopqrstuvwxyz".to_owned()),
        );
        arguments.insert("count".to_owned(), Json::from(123_456_789u64));
        let mut message = faux_assistant_message("done", message_options());
        message.content = vec![AssistantContent::ToolCall(ToolCall {
            id: "tool-1".to_owned(),
            name: "echo".to_owned(),
            arguments,
            thought_signature: None,
            namespace: None,
        })];
        message.stop_reason = StopReason::ToolUse;
        registration.set_responses(vec![message.into()]);

        let token = CancellationToken::new();
        let options = options_with_signal(token.clone());
        let mut events = Vec::new();
        let mut tool_call_delta_count = 0;
        let mut stream = compat::stream(
            &registration.get_model(),
            &context_text("hi"),
            Some(&options),
        );
        while let Some(event) = stream.next().await {
            events.push(event_name(&event));
            if matches!(event, AssistantMessageEvent::ToolCallDelta { .. }) {
                tool_call_delta_count += 1;
                token.cancel();
            }
        }

        assert_eq!(tool_call_delta_count, 1);
        assert!(events.contains(&"toolcall_start"));
        assert!(events.contains(&"toolcall_delta"));
        assert!(events.contains(&"error"));
        assert!(!events.contains(&"toolcall_end"));
    }

    #[tokio::test]
    async fn unregisters_the_provider() {
        let _serialized = registry_guard().await;
        let registration = compat::register_faux_provider(&RegisterFauxProviderOptions::default());
        registration.set_responses(vec![
            faux_assistant_message("hello", message_options()).into(),
        ]);
        registration.unregister();

        let response = compat::complete(&registration.get_model(), &context_text("hi"), None).await;
        assert_eq!(response.stop_reason, StopReason::Error);
        assert_eq!(
            response.error_message.as_deref(),
            Some(format!("No API provider registered for api: {}", registration.api()).as_str())
        );
    }

    // ---- deferred responses ---------------------------------------------

    #[tokio::test]
    async fn streams_a_deferred_handle_then_resolves_after_pending_fetches() {
        let registration = register(&RegisterFauxProviderOptions {
            deferred: Some(FauxDeferredOptions {
                pending_fetches: Some(1),
                poll_after_ms: Some(25),
            }),
            ..Default::default()
        })
        .await;
        registration.set_responses(vec![
            faux_assistant_message("final answer", message_options()).into(),
        ]);

        let options = SimpleStreamOptions {
            deferred: Some(crate::types::DeferredRequest::Enabled(true)),
            ..Default::default()
        };
        let submitted = collect_events(compat::stream_simple(
            &registration.get_model(),
            &context_text("hi"),
            Some(&options),
        ))
        .await;
        let submitted = match submitted.last() {
            Some(AssistantMessageEvent::Done { message, .. }) => message.clone(),
            other => panic!("expected deferred done, got {}", event_name(other.unwrap())),
        };

        assert_eq!(submitted.stop_reason, StopReason::Deferred);
        let handle = submitted
            .deferred
            .expect("deferred message carries a handle");
        assert_eq!(
            handle.provider,
            registration.get_model().provider.to_string()
        );
        assert_eq!(handle.model_id, registration.get_model().id);
        assert_eq!(handle.api, registration.api());
        assert_eq!(handle.poll_after_ms, Some(25));

        // First fetch replays the deferred message; the second resolves.
        let first_fetch = collect_events(registration.core.fetch_deferred(
            &registration.get_model(),
            &handle,
            &StreamOptions::default(),
        ))
        .await;
        assert_eq!(first_fetch.len(), 2);
        match first_fetch.last() {
            Some(AssistantMessageEvent::Done { message, .. }) => {
                assert_eq!(message.stop_reason, StopReason::Deferred);
            }
            other => panic!("expected deferred done, got {}", event_name(other.unwrap())),
        }

        let final_fetch = collect_events(registration.core.fetch_deferred(
            &registration.get_model(),
            &handle,
            &StreamOptions::default(),
        ))
        .await;
        match final_fetch.last() {
            Some(AssistantMessageEvent::Done { message, .. }) => {
                assert_eq!(message.stop_reason, StopReason::Stop);
                assert_text(message, "final answer");
            }
            other => panic!("expected final done, got {}", event_name(other.unwrap())),
        }
        assert_eq!(lock(&registration.state()).deferred_fetch_count, 2);
    }

    #[tokio::test]
    async fn cancels_a_deferred_response() {
        let registration = register(&RegisterFauxProviderOptions::default()).await;
        registration.set_responses(vec![
            faux_assistant_message("final answer", message_options()).into(),
        ]);

        let options = SimpleStreamOptions {
            deferred: Some(crate::types::DeferredRequest::Enabled(true)),
            ..Default::default()
        };
        let stream = compat::stream_simple(
            &registration.get_model(),
            &context_text("hi"),
            Some(&options),
        );
        let submitted = collect_events(stream).await;
        let handle = match submitted.last() {
            Some(AssistantMessageEvent::Done { message, .. }) => {
                message.deferred.clone().expect("deferred handle")
            }
            other => panic!("expected deferred done, got {}", event_name(other.unwrap())),
        };

        registration
            .core
            .cancel_deferred(
                &registration.get_model(),
                &handle,
                &StreamOptions::default(),
            )
            .await
            .expect("cancel");

        let events = collect_events(registration.core.fetch_deferred(
            &registration.get_model(),
            &handle,
            &StreamOptions::default(),
        ))
        .await;
        match events.last() {
            Some(AssistantMessageEvent::Error { error, .. }) => {
                assert_eq!(error.stop_reason, StopReason::Error);
                assert_eq!(
                    error.error_message.as_deref(),
                    Some(format!("Faux deferred response was cancelled: {}", handle.id).as_str())
                );
            }
            other => panic!(
                "expected cancelled error, got {}",
                event_name(other.unwrap())
            ),
        }
        let cancelled = lock(&registration.state()).cancelled_deferred.clone();
        assert_eq!(
            serde_json::to_value(&cancelled).unwrap(),
            serde_json::to_value(vec![handle.clone()]).unwrap()
        );
    }

    #[tokio::test]
    async fn errors_on_an_unknown_deferred_handle() {
        let registration = register(&RegisterFauxProviderOptions::default()).await;
        let handle = DeferredHandle {
            provider: registration.get_model().provider.to_string(),
            model_id: registration.get_model().id.clone(),
            api: registration.api().to_owned(),
            id: "missing".to_owned(),
            expires_at: None,
            poll_after_ms: None,
            data: None,
        };

        let events = collect_events(registration.core.fetch_deferred(
            &registration.get_model(),
            &handle,
            &StreamOptions::default(),
        ))
        .await;
        match events.last() {
            Some(AssistantMessageEvent::Error { error, .. }) => {
                assert_eq!(
                    error.error_message.as_deref(),
                    Some("Unknown faux deferred response: missing")
                );
            }
            other => panic!(
                "expected unknown-handle error, got {}",
                event_name(other.unwrap())
            ),
        }
        assert_eq!(lock(&registration.state()).deferred_fetch_count, 1);
    }
}
