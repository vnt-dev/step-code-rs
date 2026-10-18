//! Per-session model and thinking-level state.
//!
//! Ported functionally from the model-switching half of
//! `@step-harness/coding-agent` `src/core/model-runtime.ts` and
//! `src/core/agent-session.ts` (`setModel` / `setThinkingLevel`).

use std::sync::Mutex;

use step_harness_providers::types::{Model, ModelThinkingLevel};

use crate::models::catalog::ModelCatalog;
use crate::settings::{Settings, ThinkingLevelSetting};

/// Tracks the active model and reasoning level for one session.
pub struct ModelController {
    state: Mutex<ControllerState>,
}

#[derive(Clone, Debug)]
struct ControllerState {
    provider: String,
    model: String,
    thinking_level: ModelThinkingLevel,
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().expect("model controller mutex poisoned")
}

/// Error from model switches.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct ModelSwitchError(pub String);

impl ModelController {
    /// Choose the initial model from settings, then the first usable model.
    pub fn initial(catalog: &ModelCatalog, settings: &Settings) -> Result<Self, ModelSwitchError> {
        let snapshot = catalog.snapshot();
        let configured = |provider: &str| snapshot.configured_providers.contains(provider);

        // 1) The settings default, when its provider is configured.
        if let (Some(provider), Some(model)) = (
            settings.default_provider.as_deref(),
            settings.default_model.as_deref(),
        ) && configured(provider)
            && let Some(found) = snapshot
                .all_models
                .iter()
                .find(|candidate| candidate.provider.as_str() == provider && candidate.id == model)
        {
            return Ok(Self::from_model(found, settings));
        }

        // 2) The first available model, stable by provider then id.
        if let Some(model) = first_available(&snapshot.available_models) {
            return Ok(Self::from_model(model, settings));
        }

        // 3) Any model of a configured provider (may still need credentials at
        //    request time, but the UI should show something).
        if let Some(model) = snapshot
            .all_models
            .iter()
            .find(|model| configured(model.provider.as_str()))
        {
            return Ok(Self::from_model(model, settings));
        }

        Err(ModelSwitchError(
            "no models available: add a provider in models.json or log in".to_owned(),
        ))
    }

    /// Build a controller pinned to one model, deriving the thinking level.
    pub fn from_model(model: &Model, settings: &Settings) -> Self {
        let thinking_level = resolve_initial_thinking_level(model, settings);
        ModelController {
            state: Mutex::new(ControllerState {
                provider: model.provider.to_string(),
                model: model.id.clone(),
                thinking_level,
            }),
        }
    }

    /// A controller with no model selected (nothing configured yet).
    pub fn unset() -> Self {
        ModelController {
            state: Mutex::new(ControllerState {
                provider: String::new(),
                model: String::new(),
                thinking_level: crate::settings::DEFAULT_THINKING_LEVEL,
            }),
        }
    }

    /// Whether any model is selected.
    pub fn is_set(&self) -> bool {
        let state = lock(&self.state);
        !state.provider.is_empty() && !state.model.is_empty()
    }

    /// The active provider id.
    pub fn provider_id(&self) -> String {
        lock(&self.state).provider.clone()
    }

    /// The active model id.
    pub fn model_id(&self) -> String {
        lock(&self.state).model.clone()
    }

    /// The resolved model, when the catalog still knows it.
    pub fn current_model(&self, catalog: &ModelCatalog) -> Option<Model> {
        let state = lock(&self.state);
        catalog.find(&state.provider, &state.model)
    }

    /// The active reasoning level.
    pub fn thinking_level(&self) -> ModelThinkingLevel {
        lock(&self.state).thinking_level
    }

    /// Switch models; the target provider must be configured.
    pub fn set_model(
        &self,
        provider: &str,
        model_id: &str,
        catalog: &ModelCatalog,
    ) -> Result<(), ModelSwitchError> {
        let target = catalog
            .find(provider, model_id)
            .ok_or_else(|| ModelSwitchError(format!("unknown model: {provider}/{model_id}")))?;
        if !catalog.is_provider_configured(provider) {
            return Err(ModelSwitchError(format!(
                "no credentials configured for provider \"{provider}\""
            )));
        }
        let mut state = lock(&self.state);
        state.provider = provider.to_owned();
        state.model = model_id.to_owned();
        state.thinking_level = clamp_thinking_level(state.thinking_level, &target);
        Ok(())
    }

    /// Set the reasoning level, clamped to what the model supports.
    pub fn set_thinking_level(&self, level: ModelThinkingLevel, catalog: &ModelCatalog) {
        let target = self.current_model(catalog);
        let clamped = match &target {
            Some(model) => clamp_thinking_level(level, model),
            None => level,
        };
        lock(&self.state).thinking_level = clamped;
    }

    /// The reasoning levels the active model supports.
    pub fn supported_levels(&self, catalog: &ModelCatalog) -> Vec<ModelThinkingLevel> {
        match self.current_model(catalog) {
            Some(model) => step_harness_providers::models::get_supported_thinking_levels(&model),
            None => vec![ModelThinkingLevel::Off, ModelThinkingLevel::Medium],
        }
    }
}

/// First available model in a stable order (provider id, then model id).
fn first_available(models: &[Model]) -> Option<&Model> {
    models.iter().min_by(|left, right| {
        let left_key = (left.provider.to_string(), left.id.clone());
        let right_key = (right.provider.to_string(), right.id.clone());
        left_key.cmp(&right_key)
    })
}

/// Thinking-level precedence: per-model setting > global default > medium.
fn resolve_initial_thinking_level(model: &Model, settings: &Settings) -> ModelThinkingLevel {
    let key = format!("{}/{}", model.provider, model.id);
    if let Some(level) = settings
        .model_thinking_levels
        .as_ref()
        .and_then(|levels| levels.get(&key).copied())
    {
        return to_model_level(level);
    }
    settings
        .default_thinking_level
        .map(to_model_level)
        .unwrap_or(crate::settings::DEFAULT_THINKING_LEVEL)
}

fn to_model_level(level: ThinkingLevelSetting) -> ModelThinkingLevel {
    level
}

/// Clamp a requested level into the model's supported set.
fn clamp_thinking_level(level: ModelThinkingLevel, model: &Model) -> ModelThinkingLevel {
    let supported = step_harness_providers::models::get_supported_thinking_levels(model);
    if supported.contains(&level) {
        return level;
    }
    if !model.reasoning {
        return ModelThinkingLevel::Off;
    }
    supported
        .into_iter()
        .find(|candidate| *candidate == ModelThinkingLevel::Medium)
        .unwrap_or(ModelThinkingLevel::Off)
}

#[cfg(test)]
mod tests {
    use super::*;
    use step_harness_providers::types::{Api, ProviderId};

    fn model(id: &str, reasoning: bool) -> Model {
        Model {
            id: id.to_owned(),
            name: id.to_owned(),
            api: Api::OPENAI_COMPLETIONS.into(),
            provider: ProviderId::new("p"),
            base_url: "https://example.invalid".to_owned(),
            reasoning,
            thinking_level_map: None,
            input: Vec::new(),
            cost: Default::default(),
            context_window: 1000,
            max_tokens: 100,
            sampling_params: None,
            headers: None,
            compat: None,
        }
    }

    #[test]
    fn clamps_levels_for_non_reasoning_models() {
        let plain = model("plain", false);
        assert_eq!(
            clamp_thinking_level(ModelThinkingLevel::High, &plain),
            ModelThinkingLevel::Off
        );
    }

    #[test]
    fn resolves_per_model_thinking_override() {
        let settings = Settings {
            model_thinking_levels: Some(
                [("p/override".to_owned(), ModelThinkingLevel::High)]
                    .into_iter()
                    .collect(),
            ),
            ..Default::default()
        };
        let target = model("override", true);
        assert_eq!(
            resolve_initial_thinking_level(&target, &settings),
            ModelThinkingLevel::High
        );
    }
}
