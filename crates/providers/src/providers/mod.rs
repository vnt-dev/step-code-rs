//! Provider implementations.
//!
//! The Step-only build ships no built-in providers (the generated catalog is
//! empty); the faux test provider lives here.

pub mod faux;

pub use faux::{
    FauxAssistantMessageOptions, FauxContentBlock, FauxContentInput, FauxCore, FauxDeferredOptions,
    FauxModelDefinition, FauxProviderHandle, FauxProviderState, FauxResponseFactory,
    FauxResponseFuture, FauxResponseStep, FauxTokenSize, RegisterFauxProviderOptions,
    create_faux_core, faux_assistant_message, faux_provider, faux_text, faux_thinking,
    faux_tool_call,
};
