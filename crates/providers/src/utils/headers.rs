//! Header map helpers.

use crate::types::ProviderHeaders;

#[derive(Debug, thiserror::Error)]
pub enum HeaderError {
    #[error("Invalid HTTP header name")]
    Name(#[source] reqwest::header::InvalidHeaderName),
    #[error("Invalid HTTP header value for {name}")]
    Value {
        name: String,
        #[source]
        source: reqwest::header::InvalidHeaderValue,
    },
    #[error("Conflicting HTTP header names differing only in case: {0}")]
    Duplicate(String),
}

/// Validate names and values once; suppressions never become empty headers.
pub fn provider_headers_to_header_map(
    headers: &ProviderHeaders,
) -> Result<reqwest::header::HeaderMap, HeaderError> {
    let mut map = reqwest::header::HeaderMap::new();
    let mut names = std::collections::HashSet::new();
    for (name, value) in headers {
        let parsed =
            reqwest::header::HeaderName::from_bytes(name.as_bytes()).map_err(HeaderError::Name)?;
        if !names.insert(parsed.clone()) {
            return Err(HeaderError::Duplicate(parsed.to_string()));
        }
        let Some(value) = value else { continue };
        let mut value =
            reqwest::header::HeaderValue::from_str(value).map_err(|source| HeaderError::Value {
                name: parsed.to_string(),
                source,
            })?;
        if matches!(
            parsed.as_str(),
            "authorization"
                | "proxy-authorization"
                | "cookie"
                | "x-api-key"
                | "cf-aig-authorization"
        ) {
            value.set_sensitive(true);
        }
        map.insert(parsed, value);
    }
    Ok(map)
}

/// Convert a `reqwest::HeaderMap` into a plain string record (non-UTF-8
/// values are skipped).
pub fn header_map_to_record(
    headers: &reqwest::header::HeaderMap,
) -> std::collections::HashMap<String, String> {
    headers
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.as_str().to_owned(), value.to_owned()))
        })
        .collect()
}

/// Drop `None` (suppressed) entries and produce a plain string map.
pub fn provider_headers_to_record(
    headers: Option<&ProviderHeaders>,
) -> Option<std::collections::HashMap<String, String>> {
    let headers = headers?;
    let record: std::collections::HashMap<String, String> = headers
        .iter()
        .filter_map(|(name, value)| value.as_ref().map(|value| (name.clone(), value.clone())))
        .collect();
    if record.is_empty() {
        None
    } else {
        Some(record)
    }
}

/// Merge provider header maps: `override` entries replace same-name
/// (case-insensitive) entries in `base`; `None` values are preserved as
/// suppressions.
pub fn merge_provider_headers(
    base: Option<&ProviderHeaders>,
    override_: Option<&ProviderHeaders>,
) -> Option<ProviderHeaders> {
    if base.is_none() && override_.is_none() {
        return None;
    }
    let mut merged: ProviderHeaders = base.cloned().unwrap_or_default();
    for (name, value) in override_.into_iter().flatten() {
        let lower = name.to_ascii_lowercase();
        merged.retain(|existing, _| existing.to_ascii_lowercase() != lower);
        merged.insert(name.clone(), value.clone());
    }
    Some(merged)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suppressions_are_absent_and_secrets_are_sensitive() {
        let headers = [
            ("Authorization".to_owned(), Some("Bearer secret".to_owned())),
            ("x-suppressed".to_owned(), None),
        ]
        .into_iter()
        .collect();
        let converted = provider_headers_to_header_map(&headers).unwrap();
        assert!(!converted.contains_key("x-suppressed"));
        assert!(converted["authorization"].is_sensitive());
    }

    #[test]
    fn invalid_headers_fail_without_leaking_values() {
        let headers = [("authorization".to_owned(), Some("secret\nvalue".to_owned()))]
            .into_iter()
            .collect();
        let error = provider_headers_to_header_map(&headers).unwrap_err();
        assert!(!error.to_string().contains("secret"));
    }
}
