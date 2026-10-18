use step_harness_agent_core::compaction::SimpleCompleter;
use step_harness_providers::providers::faux::{
    FauxAssistantMessageOptions, faux_assistant_message,
};
use step_harness_providers::types::{AssistantMessage, Context, Model, SimpleStreamOptions};

/// Keep metadata requests independent of scripted main-turn responses.
pub struct SkipNameCompleter;

#[async_trait::async_trait]
impl SimpleCompleter for SkipNameCompleter {
    async fn complete_simple(
        &self,
        _: &Model,
        _: &Context,
        _: &SimpleStreamOptions,
    ) -> AssistantMessage {
        faux_assistant_message("", FauxAssistantMessageOptions::default())
    }
}
