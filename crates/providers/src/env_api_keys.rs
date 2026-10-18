//! Environment-variable API key discovery for known providers.

use crate::types::ProviderEnv;
use crate::utils::provider_env::get_provider_env_value;

pub const ANTHROPIC_AUTH_TOKEN_ENV: &str = "ANTHROPIC_AUTH_TOKEN";
pub const ANTHROPIC_OAUTH_TOKEN_ENV: &str = "ANTHROPIC_OAUTH_TOKEN";
pub const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";

fn api_key_env_vars(provider: &str) -> Option<Vec<&'static str>> {
    if provider == "github-copilot" {
        return Some(vec!["COPILOT_GITHUB_TOKEN"]);
    }

    // ANTHROPIC_AUTH_TOKEN participates in env discovery/status, but
    // get_env_api_key() skips it because requests must pass it as
    // Authorization: Bearer.
    if provider == "anthropic" {
        return Some(vec![
            ANTHROPIC_AUTH_TOKEN_ENV,
            ANTHROPIC_OAUTH_TOKEN_ENV,
            ANTHROPIC_API_KEY_ENV,
        ]);
    }

    let env_var = match provider {
        "ant-ling" => "ANT_LING_API_KEY",
        "qwen-token-plan" => "QWEN_TOKEN_PLAN_API_KEY",
        "qwen-token-plan-cn" => "QWEN_TOKEN_PLAN_CN_API_KEY",
        "qwen-token-plan-individual" => "QWEN_TOKEN_PLAN_API_KEY",
        "openai" => "OPENAI_API_KEY",
        "azure-openai-responses" => "AZURE_OPENAI_API_KEY",
        "nvidia" => "NVIDIA_API_KEY",
        "deepseek" => "DEEPSEEK_API_KEY",
        "google" => "GEMINI_API_KEY",
        "google-vertex" => "GOOGLE_CLOUD_API_KEY",
        "groq" => "GROQ_API_KEY",
        "cerebras" => "CEREBRAS_API_KEY",
        "xai" => "XAI_API_KEY",
        "openrouter" => "OPENROUTER_API_KEY",
        "vercel-ai-gateway" => "AI_GATEWAY_API_KEY",
        "zai" => "ZAI_API_KEY",
        "zai-coding-cn" => "ZAI_CODING_CN_API_KEY",
        "mistral" => "MISTRAL_API_KEY",
        "minimax" => "MINIMAX_API_KEY",
        "minimax-cn" => "MINIMAX_CN_API_KEY",
        "moonshotai" | "moonshotai-cn" => "MOONSHOT_API_KEY",
        "huggingface" => "HF_TOKEN",
        "fireworks" => "FIREWORKS_API_KEY",
        "together" => "TOGETHER_API_KEY",
        "baseten" => "BASETEN_API_KEY",
        "opencode" | "opencode-go" => "OPENCODE_API_KEY",
        "kimi-coding" => "KIMI_API_KEY",
        "cloudflare-workers-ai" | "cloudflare-ai-gateway" => "CLOUDFLARE_API_KEY",
        "xiaomi" => "XIAOMI_API_KEY",
        "xiaomi-token-plan-cn" => "XIAOMI_TOKEN_PLAN_CN_API_KEY",
        "xiaomi-token-plan-ams" => "XIAOMI_TOKEN_PLAN_AMS_API_KEY",
        "xiaomi-token-plan-sgp" => "XIAOMI_TOKEN_PLAN_SGP_API_KEY",
        _ => return None,
    };
    Some(vec![env_var])
}

/// Find configured environment variables that can provide an API key for a
/// provider. Only reports actual API key variables; ambient credential sources
/// (AWS profiles, Google ADC) are excluded.
pub fn find_env_keys(provider: &str, env: Option<&ProviderEnv>) -> Option<Vec<String>> {
    let env_vars = api_key_env_vars(provider)?;
    let found: Vec<String> = env_vars
        .iter()
        .filter(|env_var| get_provider_env_value(env_var, env).is_some())
        .map(|env_var| (*env_var).to_owned())
        .collect();
    if found.is_empty() { None } else { Some(found) }
}

/// Ambient-auth sentinel returned for providers whose credentials come from
/// files/profiles rather than an API key env var.
pub const AMBIENT_AUTH_MARKER: &str = "<authenticated>";

/// Get an API key for a provider from known environment variables, e.g.
/// `OPENAI_API_KEY`. Does not return keys for providers that require OAuth
/// tokens. Ambient-credential providers (Vertex ADC, Bedrock) resolve to
/// [`AMBIENT_AUTH_MARKER`].
pub fn get_env_api_key(provider: &str, env: Option<&ProviderEnv>) -> Option<String> {
    if let Some(env_keys) = find_env_keys(provider, env) {
        let api_key_env = if provider == "anthropic" {
            env_keys
                .iter()
                .find(|key| key.as_str() != ANTHROPIC_AUTH_TOKEN_ENV)
                .cloned()
        } else {
            env_keys.first().cloned()
        };
        if let Some(api_key_env) = api_key_env
            && let Some(value) = get_provider_env_value(&api_key_env, env)
        {
            return Some(value);
        }
    }

    // Vertex AI supports either an explicit API key or Application Default
    // Credentials (via `gcloud auth application-default login`).
    if provider == "google-vertex" {
        let explicit = get_provider_env_value("GOOGLE_APPLICATION_CREDENTIALS", env)
            .is_some_and(|path| std::path::Path::new(&path).exists());
        let has_credentials = explicit
            || crate::auth::context::resolve_tilde(
                "~/.config/gcloud/application_default_credentials.json",
            )
            .is_some_and(|path| path.exists());
        let has_project = get_provider_env_value("GOOGLE_CLOUD_PROJECT", env).is_some()
            || get_provider_env_value("GCLOUD_PROJECT", env).is_some();
        let has_location = get_provider_env_value("GOOGLE_CLOUD_LOCATION", env).is_some();
        if has_credentials && has_project && has_location {
            return Some(AMBIENT_AUTH_MARKER.to_owned());
        }
    }

    if provider == "amazon-bedrock" {
        // Bedrock credential sources: named profile, IAM keys, bearer token,
        // ECS task role URIs, or IRSA web-identity token files.
        let configured = get_provider_env_value("AWS_PROFILE", env).is_some()
            || (get_provider_env_value("AWS_ACCESS_KEY_ID", env).is_some()
                && get_provider_env_value("AWS_SECRET_ACCESS_KEY", env).is_some())
            || get_provider_env_value("AWS_BEARER_TOKEN_BEDROCK", env).is_some()
            || get_provider_env_value("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", env).is_some()
            || get_provider_env_value("AWS_CONTAINER_CREDENTIALS_FULL_URI", env).is_some()
            || get_provider_env_value("AWS_WEB_IDENTITY_TOKEN_FILE", env).is_some();
        if configured {
            return Some(AMBIENT_AUTH_MARKER.to_owned());
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_with(vars: &[(&str, &str)]) -> ProviderEnv {
        vars.iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect()
    }

    #[test]
    fn openai_resolves_from_env() {
        let env = env_with(&[("OPENAI_API_KEY", "sk-test")]);
        assert_eq!(
            get_env_api_key("openai", Some(&env)).as_deref(),
            Some("sk-test")
        );
        assert_eq!(
            find_env_keys("openai", Some(&env)).unwrap(),
            vec!["OPENAI_API_KEY"]
        );
    }

    #[test]
    fn anthropic_skips_auth_token_for_key() {
        let env = env_with(&[
            ("ANTHROPIC_AUTH_TOKEN", "tok"),
            ("ANTHROPIC_API_KEY", "key"),
        ]);
        assert_eq!(
            get_env_api_key("anthropic", Some(&env)).as_deref(),
            Some("key")
        );
        // Auth token alone still counts as configured for discovery.
        let token_only = env_with(&[("ANTHROPIC_AUTH_TOKEN", "tok")]);
        assert_eq!(
            find_env_keys("anthropic", Some(&token_only)).unwrap().len(),
            1
        );
        assert_eq!(get_env_api_key("anthropic", Some(&token_only)), None);
    }

    #[test]
    fn unknown_provider_has_no_env_keys() {
        assert!(find_env_keys("not-a-provider", None).is_none());
        assert!(get_env_api_key("not-a-provider", None).is_none());
    }

    #[test]
    fn bedrock_ambient_marker() {
        let env = env_with(&[("AWS_PROFILE", "default")]);
        assert_eq!(
            get_env_api_key("amazon-bedrock", Some(&env)).as_deref(),
            Some(AMBIENT_AUTH_MARKER)
        );
    }
}
