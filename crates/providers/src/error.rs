//! Error types: the [`ModelsError`] collection error and the normalized
//! [`ProviderHttpError`] for provider HTTP failures.

use serde::{Deserialize, Serialize};

/// Max characters of a provider error body kept for display.
pub const MAX_PROVIDER_ERROR_BODY_CHARS: usize = 4000;

/// Error codes mirroring `ModelsErrorCode` in the TypeScript source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelsErrorCode {
    ModelSource,
    ModelValidation,
    Provider,
    Stream,
    Auth,
    Oauth,
}

/// Collection-level error: model sources, providers, auth, and stream setup.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct ModelsError {
    pub code: ModelsErrorCode,
    message: String,
    #[source]
    source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl ModelsError {
    pub fn new(code: ModelsErrorCode, message: impl Into<String>) -> Self {
        ModelsError {
            code,
            message: message.into(),
            source: None,
        }
    }

    /// Keep the underlying reason in the message: callers surface `message` only.
    pub fn with_cause(
        code: ModelsErrorCode,
        message: impl Into<String>,
        cause: impl std::fmt::Display,
    ) -> Self {
        let message = message.into();
        let detail = cause.to_string().trim().to_owned();
        if detail.is_empty() || message.contains(&detail) {
            ModelsError {
                code,
                message,
                source: None,
            }
        } else {
            ModelsError {
                code,
                message: format!("{message}: {detail}"),
                source: None,
            }
        }
    }

    /// Preserve an underlying error for diagnostics and downcasting.
    pub fn with_source(
        code: ModelsErrorCode,
        message: impl Into<String>,
        source: impl Into<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        let source = source.into();
        let mut error = Self::with_cause(code, message, &source);
        error.source = Some(source);
        error
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

/// A provider HTTP error with normalized status/headers/body. Rust adapters
/// build requests directly over `reqwest`, so there are no per-SDK error field
/// shapes to probe (unlike the TypeScript source); the normalization keeps the
/// composed display format.
#[derive(Debug, thiserror::Error)]
#[error("{}", self.display_message())]
pub struct ProviderHttpError {
    /// HTTP status code, when one was received.
    pub status: Option<u16>,
    /// Raw response body, trimmed and truncated to [`MAX_PROVIDER_ERROR_BODY_CHARS`].
    pub body: Option<String>,
    /// Transport-level message when no HTTP response was received.
    pub message: String,
    /// Server-requested retry delay parsed from `retry-after`/`retry-after-ms`.
    retry_after_ms: Option<u64>,
    retryable: Option<bool>,
    #[source]
    source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl ProviderHttpError {
    pub fn transport(message: impl Into<String>) -> Self {
        ProviderHttpError {
            status: None,
            body: None,
            message: message.into(),
            retry_after_ms: None,
            retryable: None,
            source: None,
        }
    }

    pub fn from_response(status: u16, body: Option<String>, message: impl Into<String>) -> Self {
        ProviderHttpError {
            status: Some(status),
            body: body
                .map(|text| truncate_error_text(text.trim(), MAX_PROVIDER_ERROR_BODY_CHARS))
                .filter(|text| !text.is_empty()),
            message: message.into(),
            retry_after_ms: None,
            retryable: None,
            source: None,
        }
    }

    /// Record the server's retry hint (see [`crate::utils::provider_retry::retry_after_delay_ms`]).
    pub fn with_retry_after_ms(mut self, delay_ms: Option<u64>) -> Self {
        self.retry_after_ms = delay_ms;
        self
    }

    pub fn retry_after_ms(&self) -> Option<u64> {
        self.retry_after_ms
    }

    pub fn with_retryable(mut self, retryable: bool) -> Self {
        self.retryable = Some(retryable);
        self
    }

    pub fn is_retryable(&self) -> bool {
        self.retryable
            .unwrap_or_else(|| is_retryable_status(self.status))
    }

    pub fn with_source(
        mut self,
        source: impl Into<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        self.source = Some(source.into());
        self
    }

    pub fn aborted() -> Self {
        Self::transport("request aborted").with_retryable(false)
    }

    /// True when the message already contains the body, so composing should not
    /// duplicate it.
    fn message_carries_body(&self) -> bool {
        match &self.body {
            None => true,
            Some(body) => self.message.contains(body.as_str()),
        }
    }

    fn display_message(&self) -> String {
        format_provider_error(self, None)
    }
}

/// Mirrors the pinned OpenAI/Anthropic SDK retry policy: `x-should-retry`
/// header wins, then retryable status codes; transport errors (no status) retry.
pub fn is_retryable_status(status: Option<u16>) -> bool {
    match status {
        None => true,
        Some(status) => matches!(status, 408 | 409 | 429) || status >= 500,
    }
}

/// `x-should-retry` header variant used by the HTTP layer before a status is known.
pub fn is_retryable_with_headers(
    status: Option<u16>,
    headers: &reqwest::header::HeaderMap,
) -> bool {
    if let Some(should_retry) = headers
        .get("x-should-retry")
        .and_then(|value| value.to_str().ok())
    {
        match should_retry {
            "true" => return true,
            "false" => return false,
            _ => {}
        }
    }
    is_retryable_status(status)
}

/// Compose a display string from a normalized error.
///
/// - no prefix: `"<status>: <body>"`
/// - prefix:    `"<prefix> (<status>): <body>"`
///
/// When the message already carries the body or no body/status exists, the
/// message is preserved.
pub fn format_provider_error(error: &ProviderHttpError, prefix: Option<&str>) -> String {
    if error.message_carries_body() || error.status.is_none() || error.body.is_none() {
        return match (prefix, error.status) {
            (Some(prefix), Some(status)) => format!("{prefix} ({status}): {}", error.message),
            _ => error.message.clone(),
        };
    }
    let body = error.body.as_deref().unwrap_or_default();
    match prefix {
        Some(prefix) => format!("{prefix} ({}): {body}", error.status.unwrap_or_default()),
        None => format!("{}: {body}", error.status.unwrap_or_default()),
    }
}

/// Truncate long error text, reporting how much was cut.
pub fn truncate_error_text(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_owned();
    }
    let truncated: String = text.chars().take(max_chars).collect();
    format!(
        "{truncated}... [truncated {} chars]",
        text.chars().count() - max_chars
    )
}

/// JSON stringify that never fails; non-serializable values fall back to Debug-ish text.
pub fn safe_json_stringify(value: &serde_json::Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncation_reports_char_count() {
        let text = "x".repeat(10);
        let truncated = truncate_error_text(&text, 4);
        assert_eq!(truncated, "xxxx... [truncated 6 chars]".to_owned());
    }

    #[test]
    fn formats_status_and_body_when_message_lacks_body() {
        let error = ProviderHttpError::from_response(
            403,
            Some("forbidden".to_owned()),
            "403 status code".to_owned(),
        );
        assert_eq!(format_provider_error(&error, None), "403: forbidden");
        assert_eq!(
            format_provider_error(&error, Some("OpenAI API error")),
            "OpenAI API error (403): forbidden"
        );
    }

    #[test]
    fn keeps_message_when_it_carries_body() {
        let error = ProviderHttpError::from_response(
            400,
            Some("bad request: too long".to_owned()),
            "400 bad request: too long".to_owned(),
        );
        assert_eq!(
            format_provider_error(&error, None),
            "400 bad request: too long"
        );
    }

    #[test]
    fn transport_errors_have_no_status() {
        let error = ProviderHttpError::transport("connection refused");
        assert_eq!(error.status, None);
        assert_eq!(error.to_string(), "connection refused");
    }

    #[test]
    fn models_error_appends_cause_detail() {
        let error = ModelsError::with_cause(ModelsErrorCode::Auth, "store failed", "disk full");
        assert_eq!(error.message(), "store failed: disk full");
    }
}
