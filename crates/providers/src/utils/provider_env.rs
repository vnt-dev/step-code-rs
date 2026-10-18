//! Provider-scoped environment resolution.

use crate::types::ProviderEnv;

/// Resolve a provider env value: scoped overrides win over process env.
pub fn get_provider_env_value(name: &str, env: Option<&ProviderEnv>) -> Option<String> {
    if let Some(value) = env.and_then(|values| values.get(name)) {
        return Some(value.clone());
    }
    std::env::var(name).ok().filter(|value| !value.is_empty())
}
