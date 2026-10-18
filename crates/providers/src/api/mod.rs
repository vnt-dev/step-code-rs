//! API protocol adapters and shared request-building helpers.

pub mod anthropic_messages;
pub mod constrained_sampling;
pub(crate) mod copilot_headers;
pub(crate) mod http;
pub mod openai_completions;
pub(crate) mod openai_prompt_cache;
pub mod openai_responses;
pub mod openai_responses_shared;
pub mod simple_options;
pub mod transform_messages;
