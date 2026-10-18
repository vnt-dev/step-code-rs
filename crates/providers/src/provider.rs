//! Provider identity (profiles) and declarative provider definitions.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::types::{Api, Model, ProviderId};

/// The wire dialect — the only dispatch key for a model request. A provider's
/// identity (`provider` / profile id) is a separate axis and never selects the
/// protocol implementation; only `model.api` does.
pub type ModelApiDialect = Api;

/// Where a model's resolved dialect came from. Telemetry/diagnostics only —
/// never a dispatch input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DialectSource {
    Declared,
    DerivedFromProvider,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolvedDialect {
    pub api: ModelApiDialect,
    pub source: DialectSource,
}

/// A provider's identity and directory: endpoint, auth reference, catalog of
/// model ids, display name, and an optional default dialect for models that
/// omit an explicit `api`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderProfile {
    pub id: ProviderId,
    /// Display name.
    pub label: String,
    /// Where to connect.
    pub base_url: String,
    /// Which env var / credential the key is read from.
    pub auth_ref: String,
    /// Which model ids live under this profile.
    pub catalog: Vec<String>,
    /// This profile's default dialect; individual model entries may override it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_api: Option<ModelApiDialect>,
}

/// Read-only provider-identity registry keyed by profile id.
pub trait ProviderRegistry: Send + Sync {
    fn get(&self, id: &str) -> Option<ProviderProfile>;
    fn has(&self, id: &str) -> bool {
        self.get(id).is_some()
    }
    /// The profile ids this registry can resolve (empty if not enumerable).
    fn ids(&self) -> Vec<ProviderId>;
}

/// Build a registry from a lookup map.
pub fn create_provider_registry(
    profiles: HashMap<String, ProviderProfile>,
) -> impl ProviderRegistry {
    MapProviderRegistry { profiles }
}

struct MapProviderRegistry {
    profiles: HashMap<String, ProviderProfile>,
}

impl ProviderRegistry for MapProviderRegistry {
    fn get(&self, id: &str) -> Option<ProviderProfile> {
        self.profiles.get(id).cloned()
    }

    fn ids(&self) -> Vec<ProviderId> {
        self.profiles.keys().map(ProviderId::new).collect()
    }
}

/// One model under a provider declaration: per-field overrides of the metadata
/// default, with an optional `api` dialect override.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ModelDeclaration {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api: Option<ModelApiDialect>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<Vec<crate::types::InputModality>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<crate::types::ModelCost>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_level_map: Option<crate::types::ThinkingLevelMap>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_params: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compat: Option<crate::types::ApiCompat>,
}

/// A provider's declarative form: default dialect, endpoint, credential
/// reference, and model declarations.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderDeclaration {
    pub id: ProviderId,
    pub label: String,
    pub dialect: ModelApiDialect,
    pub base_url: String,
    pub auth_ref: String,
    pub models: Vec<ModelDeclaration>,
}

/// The pure-metadata half of a model: every [`Model`] field except the
/// identity axis (`api`/`provider`/`base_url`), keyed by model id. Answers
/// "how much does it cost, how long is the context, which thinking levels" —
/// nothing about dispatch.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(from = "Model", into = "Model")]
pub struct ModelMetadata(Model);

impl From<Model> for ModelMetadata {
    fn from(model: Model) -> Self {
        ModelMetadata(model)
    }
}

impl From<ModelMetadata> for Model {
    fn from(metadata: ModelMetadata) -> Self {
        metadata.0
    }
}

impl ModelMetadata {
    pub fn id(&self) -> &str {
        &self.0.id
    }
}

/// Look up pure [`ModelMetadata`] by model id — the `metadata_defaults`
/// source consumed by [`flatten_providers`].
pub trait MetadataLookup: Send + Sync {
    fn get(&self, id: &str) -> Option<ModelMetadata>;
    fn has(&self, id: &str) -> bool {
        self.get(id).is_some()
    }
    fn ids(&self) -> Vec<String>;
}

/// Build a [`MetadataLookup`] from a list of metadata entries (last write wins
/// on id).
pub fn create_metadata_lookup(entries: Vec<ModelMetadata>) -> impl MetadataLookup {
    let mut by_id = HashMap::new();
    for entry in entries {
        by_id.insert(entry.0.id.clone(), entry);
    }
    MapMetadataLookup { by_id }
}

struct MapMetadataLookup {
    by_id: HashMap<String, ModelMetadata>,
}

impl MetadataLookup for MapMetadataLookup {
    fn get(&self, id: &str) -> Option<ModelMetadata> {
        self.by_id.get(id).cloned()
    }

    fn ids(&self) -> Vec<String> {
        self.by_id.keys().cloned().collect()
    }
}

/// Resolve a model's wire dialect, recording whether it was declared
/// explicitly or derived from the provider profile default. Returns `None`
/// when nothing supplies a dialect; the caller raises the configuration error.
pub fn resolve_model_api_dialect(
    model_api: Option<&ModelApiDialect>,
    _provider_id: &str,
    profile: &ProviderProfile,
) -> Option<ResolvedDialect> {
    // 1) Explicit api wins — this is the target state.
    if let Some(api) = model_api {
        return Some(ResolvedDialect {
            api: api.clone(),
            source: DialectSource::Declared,
        });
    }
    // 2) Compatibility path: profile default (there is no legacy
    //    provider-name derivation to consult).
    profile.default_api.clone().map(|api| ResolvedDialect {
        api,
        source: DialectSource::DerivedFromProvider,
    })
}

/// Flatten declarative providers into the ordinary [`Model`] list. Model-level
/// `api` overrides the provider default `dialect`; per-field declaration
/// overrides win over metadata defaults. Errors when a required field is
/// supplied by neither side.
pub fn flatten_providers(
    declarations: &[ProviderDeclaration],
    metadata_defaults: &dyn MetadataLookup,
) -> Result<Vec<Model>, FlattenError> {
    let mut flat = Vec::new();
    for declaration in declarations {
        for model in &declaration.models {
            let api = model
                .api
                .clone()
                .unwrap_or_else(|| declaration.dialect.clone());
            let base = metadata_defaults.get(&model.id);
            let built = Model {
                id: model.id.clone(),
                name: model
                    .name
                    .clone()
                    .or_else(|| base.as_ref().map(|metadata| metadata.0.name.clone()))
                    .ok_or_else(|| FlattenError::missing(&model.id, "name"))?,
                api,
                provider: declaration.id.clone(),
                base_url: declaration.base_url.clone(),
                reasoning: model
                    .reasoning
                    .or(base.as_ref().map(|metadata| metadata.0.reasoning))
                    .ok_or_else(|| FlattenError::missing(&model.id, "reasoning"))?,
                thinking_level_map: model.thinking_level_map.clone().or_else(|| {
                    base.as_ref()
                        .and_then(|metadata| metadata.0.thinking_level_map.clone())
                }),
                input: model
                    .input
                    .clone()
                    .or_else(|| base.as_ref().map(|metadata| metadata.0.input.clone()))
                    .ok_or_else(|| FlattenError::missing(&model.id, "input"))?,
                cost: model
                    .cost
                    .clone()
                    .or_else(|| base.as_ref().map(|metadata| metadata.0.cost.clone()))
                    .ok_or_else(|| FlattenError::missing(&model.id, "cost"))?,
                context_window: model
                    .context_window
                    .or(base.as_ref().map(|metadata| metadata.0.context_window))
                    .ok_or_else(|| FlattenError::missing(&model.id, "contextWindow"))?,
                max_tokens: model
                    .max_tokens
                    .or(base.as_ref().map(|metadata| metadata.0.max_tokens))
                    .ok_or_else(|| FlattenError::missing(&model.id, "maxTokens"))?,
                sampling_params: model.sampling_params.clone().or_else(|| {
                    base.as_ref()
                        .and_then(|metadata| metadata.0.sampling_params.clone())
                }),
                headers: model.headers.clone().or_else(|| {
                    base.as_ref()
                        .and_then(|metadata| metadata.0.headers.clone())
                }),
                compat: model
                    .compat
                    .clone()
                    .or_else(|| base.as_ref().and_then(|metadata| metadata.0.compat.clone())),
            };
            flat.push(built);
        }
    }
    Ok(flat)
}

/// A required metadata field was missing from both the declaration and the
/// metadata default.
#[derive(Debug, thiserror::Error)]
#[error(
    "flatten_providers: model \"{model}\" is missing required field \"{field}\" — no declaration override and no metadata default"
)]
pub struct FlattenError {
    model: String,
    field: &'static str,
}

impl FlattenError {
    fn missing(model: &str, field: &'static str) -> Self {
        FlattenError {
            model: model.to_owned(),
            field,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{InputModality, ModelCost};

    fn declaration() -> ProviderDeclaration {
        ProviderDeclaration {
            id: ProviderId::new("custom"),
            label: "Custom".to_owned(),
            dialect: Api::OpenaiCompletions,
            base_url: "https://api.custom.com/v1".to_owned(),
            auth_ref: "CUSTOM_API_KEY".to_owned(),
            models: vec![ModelDeclaration {
                id: "custom-1".to_owned(),
                ..Default::default()
            }],
        }
    }

    fn metadata() -> impl MetadataLookup {
        create_metadata_lookup(vec![ModelMetadata(Model {
            id: "custom-1".to_owned(),
            name: "Custom One".to_owned(),
            api: Api::OpenaiCompletions,
            provider: ProviderId::new("custom"),
            base_url: String::new(),
            reasoning: false,
            thinking_level_map: None,
            input: vec![InputModality::Text],
            cost: ModelCost::default(),
            context_window: 8192,
            max_tokens: 4096,
            sampling_params: None,
            headers: None,
            compat: None,
        })])
    }

    #[test]
    fn flatten_reproduces_model_shape() {
        let models = flatten_providers(&[declaration()], &metadata()).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "Custom One");
        assert_eq!(models[0].api, Api::OpenaiCompletions);
        assert_eq!(models[0].base_url, "https://api.custom.com/v1");
        assert_eq!(models[0].context_window, 8192);
    }

    #[test]
    fn flatten_errors_on_missing_required_field() {
        let models = flatten_providers(&[declaration()], &create_metadata_lookup(Vec::new()));
        assert!(models.is_err());
    }

    #[test]
    fn model_api_overrides_provider_dialect() {
        let mut with_override = declaration();
        with_override.models[0].api = Some(Api::AnthropicMessages);
        let models = flatten_providers(&[with_override], &metadata()).unwrap();
        assert_eq!(models[0].api, Api::AnthropicMessages);
    }

    #[test]
    fn resolve_dialect_prefers_explicit_api() {
        let mut profile = ProviderProfile {
            id: ProviderId::new("custom"),
            label: "Custom".to_owned(),
            base_url: "https://api.custom.com".to_owned(),
            auth_ref: "CUSTOM_API_KEY".to_owned(),
            catalog: vec!["custom-1".to_owned()],
            default_api: Some(Api::OpenaiResponses),
        };
        let resolved =
            resolve_model_api_dialect(Some(&Api::AnthropicMessages), "custom", &profile).unwrap();
        assert_eq!(resolved.source, DialectSource::Declared);
        profile.default_api = None;
        assert!(resolve_model_api_dialect(None, "custom", &profile).is_none());
    }

    #[test]
    fn compat_roundtrip_through_declaration() {
        let mut declaration = declaration();
        declaration.models[0].compat = Some(crate::types::CompatSettings {
            supports_temperature: Some(false),
            ..Default::default()
        });
        let models = flatten_providers(&[declaration], &metadata()).unwrap();
        let compat = models[0].anthropic_compat().expect("compat present");
        assert_eq!(compat.supports_temperature, Some(false));
        // Anthropic-shaped compat must not be mistaken for an OpenAI one.
        assert!(models[0].openai_completions_compat().is_some());
    }
}
