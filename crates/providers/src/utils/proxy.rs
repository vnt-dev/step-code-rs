//! Provider-scoped proxy configuration using reqwest's protocol and bypass rules.
use crate::types::ProviderEnv;

/// Case-insensitive scoped overrides take priority, including explicit empty values.
pub fn get_proxy_env(key: &str, env: Option<&ProviderEnv>) -> Option<String> {
    let lower = key.to_ascii_lowercase();
    let upper = key.to_ascii_uppercase();
    let scoped = env.and_then(|env| env.get(&lower).or_else(|| env.get(&upper)));
    scoped
        .cloned()
        .or_else(|| {
            std::env::var(&lower)
                .ok()
                .or_else(|| std::env::var(&upper).ok())
        })
        .filter(|value| !value.trim().is_empty())
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct ProxyConfig {
    http: Option<String>,
    https: Option<String>,
    all: Option<String>,
    no_proxy: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum ProxyError {
    #[error("Invalid {0} proxy URL")]
    Invalid(&'static str),
    #[error("{0} proxy must use HTTP or HTTPS")]
    Unsupported(&'static str),
    #[error("Could not configure HTTP proxy")]
    Configure(#[source] reqwest::Error),
}

impl ProxyConfig {
    pub(crate) fn from_env(env: Option<&ProviderEnv>) -> Result<Self, ProxyError> {
        fn proxy(
            key: &'static str,
            env: Option<&ProviderEnv>,
        ) -> Result<Option<String>, ProxyError> {
            let Some(value) = get_proxy_env(key, env) else {
                return Ok(None);
            };
            let value = if value.contains("://") {
                value
            } else {
                format!("http://{value}")
            };
            let url = url::Url::parse(value.trim()).map_err(|_| ProxyError::Invalid(key))?;
            if !matches!(url.scheme(), "http" | "https") {
                return Err(ProxyError::Unsupported(key));
            }
            if url.host_str().is_none() {
                return Err(ProxyError::Invalid(key));
            }
            Ok(Some(url.into()))
        }
        Ok(Self {
            http: proxy("http_proxy", env)?,
            https: proxy("https_proxy", env)?,
            all: proxy("all_proxy", env)?,
            no_proxy: get_proxy_env("no_proxy", env),
        })
    }

    pub(crate) fn apply(
        &self,
        builder: reqwest::ClientBuilder,
    ) -> Result<reqwest::ClientBuilder, ProxyError> {
        let mut builder = builder.no_proxy();
        if self
            .no_proxy
            .as_deref()
            .is_some_and(|value| value.trim() == "*")
        {
            return Ok(builder);
        }
        let bypass = self
            .no_proxy
            .as_deref()
            .and_then(reqwest::NoProxy::from_string);
        for (kind, value) in [
            (0, self.http.as_deref()),
            (1, self.https.as_deref()),
            (2, self.all.as_deref()),
        ] {
            if let Some(value) = value {
                let proxy = match kind {
                    0 => reqwest::Proxy::http(value),
                    1 => reqwest::Proxy::https(value),
                    _ => reqwest::Proxy::all(value),
                }
                .map_err(ProxyError::Configure)?
                .no_proxy(bypass.clone());
                builder = builder.proxy(proxy);
            }
        }
        Ok(builder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn env(values: &[(&str, &str)]) -> ProviderEnv {
        values
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect()
    }
    #[test]
    fn scoped_uppercase_wins_and_empty_disables_proxy() {
        let overrides = env(&[("HTTP_PROXY", "http://scoped:8080"), ("HTTPS_PROXY", "")]);
        assert_eq!(
            get_proxy_env("HTTP_PROXY", Some(&overrides)).as_deref(),
            Some("http://scoped:8080")
        );
        assert_eq!(get_proxy_env("HTTPS_PROXY", Some(&overrides)), None);
    }
    #[test]
    fn validates_proxy_scheme_and_redacts_invalid_urls() {
        assert!(
            ProxyConfig::from_env(Some(&env(&[(
                "https_proxy",
                "socks5://secret@localhost:1080"
            )])))
            .is_err()
        );
        let error = ProxyConfig::from_env(Some(&env(&[("https_proxy", "http://secret@[")])))
            .err()
            .unwrap();
        assert!(!error.to_string().contains("secret"));
    }
    #[test]
    fn bare_proxy_uses_http_and_protocols_stay_separate() {
        let config = ProxyConfig::from_env(Some(&env(&[
            ("HTTP_PROXY", "a:3128"),
            ("HTTPS_PROXY", "http://b:8080"),
            ("ALL_PROXY", ""),
            ("NO_PROXY", "localhost"),
        ])))
        .unwrap();
        assert_eq!(config.http.as_deref(), Some("http://a:3128/"));
        assert_eq!(config.https.as_deref(), Some("http://b:8080/"));
        assert_eq!(config.no_proxy.as_deref(), Some("localhost"));
    }
}
