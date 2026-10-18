//! Unified LLM API with automatic model discovery and provider configuration.
//!
//! Rust migration of the TypeScript `@step-harness/providers` package. The
//! public surface mirrors the source package: core types, the assistant
//! message event stream, auth resolution, provider/model collections, API
//! adapters (`anthropic-messages`, `openai-completions`, `openai-responses`),
//! the faux test provider, the Step provider login, and the compat api
//! registry.
//!
//! Not migrated (see README):
//! - `bedrock-provider` (a prebuilt AWS Smithy bundle shipping as JS; the
//!   Step-only build ships no built-in providers).
//! - `telemetry_context` request plumbing (declared but unused inside the
//!   TypeScript package; the workspace telemetry traits are not object-safe).
//! - WebSocket transports (declared in options, unused by current adapters).

pub mod api;
pub mod auth;
pub mod availability;
pub mod compat;
pub mod env_api_keys;
pub mod error;
pub mod event;
pub mod execution_timing;
pub mod request_metrics;
pub mod models;
pub mod models_store;
pub mod provider;
pub mod providers;
pub mod session_resources;
pub mod step_provider;
pub mod types;
pub mod utils;

pub use api::constrained_sampling::{
    create_grammar_tool_input_properties, get_json_schema_tool_parameters, make_strict_json_schema,
    resolve_grammar_constrained_sampling, resolve_json_schema_strict_sampling,
};
pub use api::simple_options::{
    DEFAULT_THINKING_BUDGETS, MIN_ANSWER_TOKENS, adjust_max_tokens_for_thinking,
    build_base_options, clamp_max_tokens_to_context, clamp_reasoning, thinking_budget_for_level,
};
pub use api::transform_messages::transform_messages;
pub use auth::{
    ApiKeyAuth, ApiKeyCredential, AuthCheck, AuthContext, AuthEvent, AuthInteraction, AuthPrompt,
    AuthResolutionOverrides, AuthResult, AuthType, Credential, CredentialInfo, CredentialKind,
    CredentialStore, InMemoryCredentialStore, ModelAuth, OAuthAuth, OAuthCredential, ProviderAuth,
    env_api_key_auth, resolve_provider_auth,
};
pub use availability::{ProviderAvailability, probe_provider_availability};
pub use error::{ModelsError, ModelsErrorCode, ProviderHttpError};
pub use event::{
    AssistantMessageEvent, AssistantMessageEventSender, AssistantMessageEventStream,
    EventStreamError, assistant_message_event_stream, lazy_stream,
};
pub use models::{
    ApiDispatch, CreateModelsOptions, CreateProviderOptions, FetchModelsFn, FilterModelsFn, Models,
    ModelsRefreshOptions, ModelsRefreshResult, Provider, ProviderStreams, RefreshModelsContext,
    calculate_cost, clamp_thinking_level, create_provider, get_supported_thinking_levels, has_api,
    models_are_equal,
};
pub use models_store::{InMemoryModelsStore, ModelsStore, ModelsStoreEntry};
pub use provider::{
    DialectSource, MetadataLookup, ModelApiDialect, ModelDeclaration, ProviderDeclaration,
    ProviderProfile, ProviderRegistry, ResolvedDialect, create_metadata_lookup,
    create_provider_registry, flatten_providers, resolve_model_api_dialect,
};
pub use session_resources::{cleanup_session_resources, register_session_resource_cleanup};
pub use types::{
    AnthropicAllowedFallbackModel, AnthropicMessagesCompat, Api, ApiCompat, AssistantContent,
    AssistantMessage, CacheRetention, ChatTemplateKwargValue, CompatSettings,
    ConstrainedSamplingConfig, Context, DeferredHandle, EventSnapshotMode, GrammarFormat,
    GrammarVariants, ImageContent, InputModality, Json, Message, Model, ModelCost, ModelCostRates,
    ModelCostTier, ModelThinkingLevel, OpenaiCompletionsCompat, OpenaiResponsesCompat, ProviderEnv,
    ProviderHeaders, ProviderId, ProviderRequestOptions, ProviderResponse, SimpleStreamOptions,
    StopReason, StreamOptions, TextContent, ThinkingBudgets, ThinkingContent, ThinkingLevel,
    ThinkingLevelMap, ThinkingTokenBudgetField, Tool, ToolCall, ToolChoice, ToolResultMessage,
    Transport, Usage, UsageCost, UserContent, UserContentBlock, UserMessage,
};
pub use utils::estimate::{estimate_context_tokens, estimate_message_tokens, estimate_text_tokens};
pub use utils::overflow::{is_context_overflow, is_recoverable_length};
pub use utils::retry::{
    RetryCallbacks, RetryPolicy, is_retryable_assistant_error, retry_assistant_call,
};
pub use utils::validation::{
    ToolValidationError, ToolValidationErrorKind, validate_tool_arguments, validate_tool_call,
};

#[cfg(test)]
mod tests {
    #[test]
    fn crate_version_metadata() {
        assert_eq!(env!("CARGO_PKG_NAME"), "step-harness-providers");
    }
}
