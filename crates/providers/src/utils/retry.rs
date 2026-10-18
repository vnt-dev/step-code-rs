//! Assistant-call retry: bounded attempts with exponential backoff plus the
//! transient-error classifier used to decide whether to restart a turn.

use backon::{BackoffBuilder, ExponentialBuilder};
use std::sync::LazyLock;
use std::time::Duration;

use regex::Regex;

use crate::types::{AssistantMessage, StopReason};

static NON_RETRYABLE_PROVIDER_LIMIT_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
		r"(?i)GoUsageLimitError|FreeUsageLimitError|Monthly usage limit reached|available balance|insufficient_quota|out of budget|quota exceeded|billing",
	)
	.expect("static non-retryable pattern")
});

static RETRYABLE_PROVIDER_ERROR_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
		r"(?i)overloaded|rate.?limit|too many requests|429|500|502|503|504|524|service.?unavailable|server.?error|internal.?error|provider.?returned.?error|exceeded request buffer limit while retrying upstream|network.?error|connection.?error|connection.?refused|connection.?lost|other side closed|fetch failed|getaddrinfo|ENOTFOUND|EAI_AGAIN|upstream.?connect|reset before headers|socket hang up|socket connection was closed|timed? out|timeout|terminated|websocket.?closed|websocket.?error|ended without|stream ended before message_stop|stream ended before a terminal response event|http2 request did not get a response|retry delay|you can retry your request|try your request again|please retry your request|ResourceExhausted",
	)
	.expect("static retryable pattern")
});

/// Retry policy: bounded attempts with exponential backoff
/// (`base_delay_ms * 2^(attempt-1)`, capped at 60 seconds).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetryPolicy {
    pub enabled: bool,
    /// Max retry attempts (0 = no retries). The initial call never counts.
    pub max_retries: u32,
    /// Base delay in ms; exponential delays are capped at 60 seconds.
    pub base_delay_ms: u64,
}

/// Callbacks emitted around each retry attempt.
#[async_trait::async_trait]
pub trait RetryCallbacks: Send + Sync {
    /// Emitted before the backoff sleep of each retry attempt (1-indexed).
    async fn on_retry_scheduled(
        &self,
        _attempt: u32,
        _max_attempts: u32,
        _delay_ms: u64,
        _error_message: &str,
    ) {
    }
    /// Emitted after the backoff sleep, immediately before the retried call.
    async fn on_retry_attempt_start(&self) {}
    /// Emitted once when the loop ends.
    async fn on_retry_finished(&self, _success: bool, _attempt: u32, _final_error: Option<&str>) {}
}

/// Run a single assistant-producing call with bounded retry on transient
/// errors.
///
/// - A successful response is returned immediately. Aborts are terminal and
///   never retried; aborts during the backoff sleep are normalized to an
///   aborted `AssistantMessage`.
/// - A non-retryable error (per [`is_retryable_assistant_error`], including
///   quota/billing exhaustion) is returned immediately.
/// - Otherwise retries up to `max_retries` times with exponential backoff.
///
/// When `policy` is `None` or disabled, the first response is returned
/// unchanged.
pub async fn retry_assistant_call<F, Fut>(
    mut produce: F,
    policy: Option<RetryPolicy>,
    signal: Option<&tokio_util::sync::CancellationToken>,
    callbacks: Option<&(dyn RetryCallbacks + Send + Sync)>,
) -> AssistantMessage
where
    F: FnMut() -> Fut,
    Fut: Future<Output = AssistantMessage>,
{
    let max_attempts = policy
        .filter(|policy| policy.enabled)
        .map_or(0, |policy| policy.max_retries);

    let mut backoff = ExponentialBuilder::default()
        .with_min_delay(Duration::from_millis(
            policy.map_or(0, |p| p.base_delay_ms).min(60_000),
        ))
        .with_max_delay(Duration::from_secs(60))
        .with_factor(2.0)
        .without_max_times()
        .build();
    let mut attempt: u32 = 0;
    let mut last_retry: Option<(u32, String)> = None;
    loop {
        let response = produce().await;

        // Abort: terminal but not successful. Never retry an aborted message.
        if response.stop_reason == StopReason::Aborted {
            if let Some((attempt, _)) = last_retry
                && let Some(callbacks) = callbacks
            {
                callbacks.on_retry_finished(false, attempt, None).await;
            }
            return response;
        }

        // Success: non-error, non-abort responses return as-is.
        if response.stop_reason != StopReason::Error {
            if let Some((attempt, _)) = last_retry
                && let Some(callbacks) = callbacks
            {
                callbacks.on_retry_finished(true, attempt, None).await;
            }
            return response;
        }

        // Non-retryable, or budget exhausted: return the final error message.
        if attempt >= max_attempts || !is_retryable_assistant_error(&response) {
            if let Some((attempt, _)) = last_retry
                && let Some(callbacks) = callbacks
            {
                callbacks
                    .on_retry_finished(false, attempt, response.error_message.as_deref())
                    .await;
            }
            return response;
        }

        attempt += 1;
        let error_message = response
            .error_message
            .clone()
            .unwrap_or_else(|| "Unknown error".to_owned());
        last_retry = Some((attempt, error_message.clone()));
        let delay = backoff.next().unwrap_or(Duration::from_secs(60));
        let delay_ms = u64::try_from(delay.as_millis()).unwrap_or(u64::MAX);
        if let Some(callbacks) = callbacks {
            callbacks
                .on_retry_scheduled(attempt, max_attempts, delay_ms, &error_message)
                .await;
        }

        // Normalize aborts during retry backoff to the same message shape as
        // provider stream aborts, so callers do not care when cancellation hit.
        if crate::utils::sleep_with_optional_abort(delay, signal)
            .await
            .is_err()
        {
            if let Some(callbacks) = callbacks {
                callbacks
                    .on_retry_finished(false, attempt, Some(&error_message))
                    .await;
            }
            let mut aborted = response;
            aborted.stop_reason = StopReason::Aborted;
            aborted.error_message = None;
            return aborted;
        }
        if let Some(callbacks) = callbacks {
            callbacks.on_retry_attempt_start().await;
        }
    }
}

/// Classify whether a failed assistant message looks like a transient provider
/// or transport error. Quota/billing exhaustion is never retryable.
pub fn is_retryable_assistant_error(message: &AssistantMessage) -> bool {
    if message.stop_reason != StopReason::Error {
        return false;
    }
    let Some(error_message) = &message.error_message else {
        return false;
    };
    if NON_RETRYABLE_PROVIDER_LIMIT_PATTERN.is_match(error_message) {
        return false;
    }
    RETRYABLE_PROVIDER_ERROR_PATTERN.is_match(error_message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Api, ProviderId, Usage};

    fn message(stop_reason: StopReason, error: Option<&str>) -> AssistantMessage {
        AssistantMessage {
            content: Vec::new(),
            api: Api::OpenaiCompletions,
            provider: ProviderId::new("test"),
            model: "m".to_owned(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: Usage::default(),
            stop_reason,
            deferred: None,
            error_message: error.map(str::to_owned),
            raw_stop_reason: None,
            end_turn: None,
            timestamp: 0,
        }
    }

    #[test]
    fn classifier_matches_transient_and_rejects_quota() {
        assert!(is_retryable_assistant_error(&message(
            StopReason::Error,
            Some("503 service unavailable")
        )));
        assert!(is_retryable_assistant_error(&message(
            StopReason::Error,
            Some("stream ended before message_stop")
        )));
        assert!(!is_retryable_assistant_error(&message(
            StopReason::Error,
            Some("insufficient_quota: billing hard limit")
        )));
        assert!(!is_retryable_assistant_error(&message(
            StopReason::Error,
            Some("invalid model")
        )));
    }

    #[tokio::test]
    async fn retries_transient_then_succeeds() {
        let attempts = std::cell::Cell::new(0u32);
        let policy = RetryPolicy {
            enabled: true,
            max_retries: 2,
            base_delay_ms: 1,
        };
        let result = retry_assistant_call(
            || {
                let attempt = attempts.get() + 1;
                attempts.set(attempt);
                async move {
                    if attempt < 2 {
                        message(StopReason::Error, Some("503 service unavailable"))
                    } else {
                        message(StopReason::Stop, None)
                    }
                }
            },
            Some(policy),
            None,
            None,
        )
        .await;
        assert_eq!(result.stop_reason, StopReason::Stop);
        assert_eq!(attempts.get(), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn backoff_is_observed_without_a_signal() {
        let start = tokio::time::Instant::now();
        let attempts = std::cell::Cell::new(0);
        let result = retry_assistant_call(
            || {
                let attempt = attempts.get();
                attempts.set(attempt + 1);
                async move {
                    if attempt == 0 {
                        message(StopReason::Error, Some("503"))
                    } else {
                        message(StopReason::Stop, None)
                    }
                }
            },
            Some(RetryPolicy {
                enabled: true,
                max_retries: 1,
                base_delay_ms: 500,
            }),
            None,
            None,
        )
        .await;
        assert_eq!(result.stop_reason, StopReason::Stop);
        assert_eq!(start.elapsed(), Duration::from_millis(500));
    }

    #[tokio::test]
    async fn quota_error_fails_fast() {
        let attempts = std::cell::Cell::new(0u32);
        let policy = RetryPolicy {
            enabled: true,
            max_retries: 5,
            base_delay_ms: 1,
        };
        let result = retry_assistant_call(
            || {
                let attempt = attempts.get() + 1;
                attempts.set(attempt);
                async move { message(StopReason::Error, Some("quota exceeded")) }
            },
            Some(policy),
            None,
            None,
        )
        .await;
        assert_eq!(result.stop_reason, StopReason::Error);
        assert_eq!(attempts.get(), 1);
    }

    #[tokio::test]
    async fn abort_during_backoff_normalizes_to_aborted() {
        let token = tokio_util::sync::CancellationToken::new();
        token.cancel();
        let policy = RetryPolicy {
            enabled: true,
            max_retries: 1,
            base_delay_ms: 5_000,
        };
        let result = retry_assistant_call(
            || async { message(StopReason::Error, Some("503")) },
            Some(policy),
            Some(&token),
            None,
        )
        .await;
        assert_eq!(result.stop_reason, StopReason::Aborted);
        assert!(result.error_message.is_none());
    }
}
