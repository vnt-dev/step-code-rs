//! The user-facing `models.json` configuration.
//!
//! Ported functionally from `@step-harness/coding-agent`
//! `src/core/model-config.ts`. The file declares providers (endpoint, auth,
//! dialect) and their models; it is credential-blind apart from optional
//! inline API keys.

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};
use step_harness_providers::types::{
    Api, ApiCompat, InputModality, Model, ModelCost, ProviderId, ThinkingLevelMap,
};

/// One model declaration inside a provider entry.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelEntry {
    #[serde(default)]
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Wire dialect override for this model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_level_map: Option<ThinkingLevelMap>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<Vec<InputModality>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<ModelCost>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_params: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compat: Option<ApiCompat>,
}

/// One provider declaration.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderEntry {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Inline API key; when absent the provider reads its env vars.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Default dialect for models that omit `api`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub models: Option<Vec<ModelEntry>>,
    /// Per-model field overrides applied on top of `models`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_overrides: Option<BTreeMap<String, ModelEntry>>,
}

/// The `models.json` document.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelsConfigFile {
    #[serde(default)]
    pub providers: BTreeMap<String, ProviderEntry>,
}

impl ModelsConfigFile {
    /// Parse the document; a missing file is an empty configuration.
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        match std::fs::read(path) {
            Ok(bytes) => {
                let text = String::from_utf8(bytes)
                    .map_err(|error| format!("invalid UTF-8 in {path:?}: {error}"))?;
                let text = text.trim_start_matches('\u{feff}');
                serde_json::from_str(text)
                    .map_err(|error| format!("invalid JSON in {path:?}: {error}"))
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(error) => Err(format!("cannot read {path:?}: {error}")),
        }
    }

    /// Save with 2-space indentation (atomic replace).
    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("cannot create {parent:?}: {error}"))?;
        }
        let text = serde_json::to_string_pretty(self)
            .map_err(|error| format!("cannot serialize models config: {error}"))?;
        let temp = path.with_extension(format!("json.tmp.{}", std::process::id()));
        std::fs::write(&temp, text).map_err(|error| format!("cannot write {temp:?}: {error}"))?;
        std::fs::rename(&temp, path).map_err(|error| {
            let _ = std::fs::remove_file(&temp);
            format!("cannot replace {path:?}: {error}")
        })
    }

    /// Flatten into runtime [`Model`] values for one provider, applying
    /// `modelOverrides` on top of `models`.
    pub fn models_for(&self, provider_id: &str) -> Vec<Model> {
        let Some(entry) = self.providers.get(provider_id) else {
            return Vec::new();
        };
        let default_api = entry.api.clone();
        let base_url = entry.base_url.clone().unwrap_or_default();
        let mut models: Vec<Model> = entry
            .models
            .iter()
            .flatten()
            .map(|model| {
                model_entry_to_model(model, provider_id, default_api.as_deref(), &base_url)
            })
            .collect();
        if let Some(overrides) = &entry.model_overrides {
            for (id, override_entry) in overrides {
                if let Some(existing) = models.iter_mut().find(|model| &model.id == id) {
                    apply_override(existing, override_entry);
                }
            }
        }
        models
    }
}

/// Default context window when a model entry omits it.
pub const DEFAULT_CONTEXT_WINDOW: u64 = 200_000;
/// Default max output tokens when a model entry omits it.
pub const DEFAULT_MAX_TOKENS: u64 = 8_192;

fn model_entry_to_model(
    entry: &ModelEntry,
    provider_id: &str,
    default_api: Option<&str>,
    provider_base_url: &str,
) -> Model {
    let api = Api::Custom(std::sync::Arc::from(
        entry
            .api
            .clone()
            .or_else(|| default_api.map(ToOwned::to_owned))
            .unwrap_or_else(|| Api::OPENAI_COMPLETIONS.to_owned()),
    ));
    Model {
        id: entry.id.clone(),
        name: entry.name.clone().unwrap_or_else(|| entry.id.clone()),
        api,
        provider: ProviderId::new(provider_id),
        base_url: entry
            .base_url
            .clone()
            .unwrap_or_else(|| provider_base_url.to_owned()),
        reasoning: entry.reasoning.unwrap_or(false),
        thinking_level_map: entry.thinking_level_map.clone(),
        input: entry
            .input
            .clone()
            .unwrap_or_else(|| vec![InputModality::Text]),
        cost: entry.cost.clone().unwrap_or_default(),
        context_window: entry.context_window.unwrap_or(DEFAULT_CONTEXT_WINDOW),
        max_tokens: entry.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        sampling_params: entry.sampling_params.clone(),
        headers: entry.headers.clone(),
        compat: entry.compat.clone(),
    }
}

/// Apply an override entry's present fields onto an existing model.
fn apply_override(model: &mut Model, override_entry: &ModelEntry) {
    if let Some(name) = &override_entry.name {
        model.name = name.clone();
    }
    if let Some(api) = &override_entry.api {
        model.api = Api::Custom(std::sync::Arc::from(api.clone()));
    }
    if let Some(base_url) = &override_entry.base_url {
        model.base_url = base_url.clone();
    }
    if let Some(reasoning) = override_entry.reasoning {
        model.reasoning = reasoning;
    }
    if let Some(map) = &override_entry.thinking_level_map {
        model.thinking_level_map = Some(map.clone());
    }
    if let Some(input) = &override_entry.input {
        model.input = input.clone();
    }
    if let Some(cost) = &override_entry.cost {
        model.cost = cost.clone();
    }
    if let Some(context_window) = override_entry.context_window {
        model.context_window = context_window;
    }
    if let Some(max_tokens) = override_entry.max_tokens {
        model.max_tokens = max_tokens;
    }
    if let Some(sampling) = &override_entry.sampling_params {
        model.sampling_params = Some(sampling.clone());
    }
    if let Some(headers) = &override_entry.headers {
        model.headers = Some(headers.clone());
    }
    if let Some(compat) = &override_entry.compat {
        model.compat = Some(compat.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_providers_and_models() {
        let json = serde_json::json!({
            "providers": {
                "my-openai": {
                    "baseUrl": "https://api.example.com/v1",
                    "apiKey": "sk-test",
                    "api": "openai-completions",
                    "models": [
                        {"id": "model-a", "name": "Model A", "contextWindow": 64000}
                    ],
                    "modelOverrides": {
                        "model-a": {"maxTokens": 4096}
                    }
                }
            }
        });
        let config: ModelsConfigFile = serde_json::from_value(json).unwrap();
        let models = config.models_for("my-openai");
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "model-a");
        assert_eq!(models[0].name, "Model A");
        assert_eq!(models[0].context_window, 64000);
        assert_eq!(models[0].max_tokens, 4096);
        assert_eq!(models[0].api.as_str(), "openai-completions");
    }

    #[test]
    fn missing_file_is_empty_config() {
        let config =
            ModelsConfigFile::load(std::path::Path::new("/nonexistent/models.json")).unwrap();
        assert!(config.providers.is_empty());
    }
}
