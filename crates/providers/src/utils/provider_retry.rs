//! HTTP-level provider retry mirroring the pinned OpenAI/Anthropic SDK policy
//! with abortable backoff sleeps.

use crate::error::ProviderHttpError;
use backon::{BackoffBuilder, ExponentialBuilder};
use std::time::Duration;

const DEFAULT_MAX_RETRY_DELAY_MS: u64 = 60_000;

/// Options for [`retry_provider_request`].
#[derive(Clone, Debug, Default)]
pub struct ProviderRetryOptions {
    pub max_retries: Option<u32>,
    pub max_retry_delay_ms: Option<u64>,
    pub signal: Option<tokio_util::sync::CancellationToken>,
}

/// Reproduces the retry behavior of the OpenAI and Anthropic SDKs while making
/// the backoff sleep interruptible. Provider-requested delays above
/// `max_retry_delay_ms` fail immediately (60 seconds by default); set it to
/// zero to disable the cap.
pub async fn retry_provider_request<T, F, Fut>(
    request: F,
    options: ProviderRetryOptions,
) -> Result<T, ProviderHttpError>
where F: FnMut() -> Fut, Fut: Future<Output = Result<T, ProviderHttpError>> {
    retry_provider_request_tracked(request, options, None).await
}

pub async fn retry_provider_request_tracked<T, F, Fut>(
    mut request: F,
    options: ProviderRetryOptions,
    tracker: Option<&std::sync::Arc<crate::request_metrics::RequestTracker>>,
) -> Result<T, ProviderHttpError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, ProviderHttpError>>,
{
    let max_retries = options.max_retries.unwrap_or(0);
    let mut retries_remaining = max_retries;
    let mut backoff = ExponentialBuilder::default()
        .with_min_delay(Duration::from_millis(500))
        .with_max_delay(Duration::from_secs(8))
        .with_factor(2.0)
        .without_max_times()
        .build();

    loop {
        if options.signal.as_ref().is_some_and(|s| s.is_cancelled()) { return Err(ProviderHttpError::aborted()); }
        if let Some(tracker) = tracker { tracker.begin(); }
        let outcome = match options.signal.as_ref() {
            Some(signal) => tokio::select! {
                biased;
                _ = signal.cancelled() => Err(ProviderHttpError::aborted()),
                result = request() => result,
            },
            None => request().await,
        };
        match outcome {
            Ok(value) => return Ok(value),
            Err(error) => {
                if let Some(tracker) = tracker {
                    tracker.finish(if options.signal.as_ref().is_some_and(|s| s.is_cancelled()) { "cancelled" } else { "failed" }, error.status, Some(if error.status.is_some() { "http" } else if crate::request_metrics::is_timeout(&error.message) { "timeout" } else { "transport" }));
                }
                if let Some(signal) = &options.signal
                    && signal.is_cancelled()
                {
                    return Err(ProviderHttpError::aborted());
                }
                let retryable = error.is_retryable();
                if retries_remaining == 0 || !retryable {
                    return Err(error);
                }

                retries_remaining -= 1;
                let delay_ms = compute_retry_delay_ms(
                    &error,
                    backoff.next().unwrap_or(Duration::from_secs(8)),
                    options.max_retry_delay_ms,
                )?;
                if crate::utils::sleep_with_optional_abort(
                    Duration::from_millis(delay_ms),
                    options.signal.as_ref(),
                )
                .await
                .is_err()
                {
                    return Err(ProviderHttpError::aborted());
                }
            }
        }
    }
}

fn compute_retry_delay_ms(
    error: &ProviderHttpError,
    backoff: Duration,
    max_retry_delay_ms: Option<u64>,
) -> Result<u64, ProviderHttpError> {
    if let Some(delay_ms) = error.retry_after_ms() {
        return validate_server_retry_delay_ms(delay_ms, max_retry_delay_ms, &error.to_string());
    }
    Ok((backoff.as_millis() as f64 * (1.0 - rand::random::<f64>() * 0.25)) as u64)
}

/// Server-requested delay validation: delays above the cap fail immediately
/// with a descriptive error so higher-level retry logic can surface them.
fn validate_server_retry_delay_ms(
    delay_ms: u64,
    max_retry_delay_ms: Option<u64>,
    provider_message: &str,
) -> Result<u64, ProviderHttpError> {
    let max_delay_ms = max_retry_delay_ms.unwrap_or(DEFAULT_MAX_RETRY_DELAY_MS);
    if max_delay_ms > 0 && delay_ms > max_delay_ms {
        return Err(ProviderHttpError::transport(format!(
            "Server requested {}s retry delay (max: {}s). {provider_message}",
            delay_ms.div_ceil(1000),
            max_delay_ms.div_ceil(1000)
        ))
        .with_retryable(false));
    }
    Ok(delay_ms)
}

/// Parse `retry-after-ms` (milliseconds) or `retry-after` (seconds or HTTP
/// date) into a delay in milliseconds. Exposed for the HTTP layer.
pub fn retry_after_delay_ms(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    if let Some(value) = headers
        .get("retry-after-ms")
        .and_then(|value| value.to_str().ok())
        && let Ok(parsed) = value.trim().parse::<f64>()
        && parsed.is_finite()
        && parsed >= 0.0
    {
        return Some(parsed as u64);
    }

    let value = headers
        .get("retry-after")
        .and_then(|value| value.to_str().ok())?
        .trim();
    if value.is_empty() {
        return None;
    }
    if let Ok(seconds) = value.parse::<f64>()
        && seconds.is_finite()
        && seconds >= 0.0
    {
        return Some((seconds * 1000.0).max(0.0) as u64);
    }
    // HTTP-date form: delay until that instant.
    http_date_timestamp_ms(value)
        .map(|target| target.saturating_sub(crate::utils::now_ms()))
        .filter(|delay| *delay > 0)
}

fn http_date_timestamp_ms(value: &str) -> Option<u64> {
    httpdate::parse_http_date(value)
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .and_then(|duration| u64::try_from(duration.as_millis()).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponential_delay_is_bounded_and_jittered() {
        for index in 0..10u32 {
            let base = Duration::from_millis(500 * (1u64 << index.min(4)));
            let delay =
                compute_retry_delay_ms(&ProviderHttpError::transport("test"), base, None).unwrap();
            assert!((300..=8000).contains(&delay), "delay out of range: {delay}");
        }
    }

    #[tokio::test]
    async fn retries_transport_errors_then_succeeds() {
        let attempts = std::cell::Cell::new(0u32);
        let result = retry_provider_request(
            || {
                let attempt = attempts.get() + 1;
                attempts.set(attempt);
                async move {
                    if attempt < 2 {
                        Err(ProviderHttpError::transport("connection refused"))
                    } else {
                        Ok(42)
                    }
                }
            },
            ProviderRetryOptions {
                max_retries: Some(2),
                ..Default::default()
            },
        )
        .await;
        assert_eq!(result.unwrap(), 42);
        assert_eq!(attempts.get(), 2);
    }

    #[tokio::test]
    async fn non_retryable_status_fails_immediately() {
        let attempts = std::cell::Cell::new(0u32);
        let result: Result<u32, ProviderHttpError> = retry_provider_request(
            || {
                let attempt = attempts.get() + 1;
                attempts.set(attempt);
                async move {
                    Err(ProviderHttpError::from_response(
                        401,
                        None,
                        "unauthorized".to_owned(),
                    ))
                }
            },
            ProviderRetryOptions {
                max_retries: Some(3),
                ..Default::default()
            },
        )
        .await;
        assert!(result.is_err());
        assert_eq!(attempts.get(), 1);
    }

    #[test]
    fn retry_after_seconds_header_parsed() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("retry-after", "2".parse().unwrap());
        assert_eq!(retry_after_delay_ms(&headers), Some(2000));
        headers.insert("retry-after-ms", "150".parse().unwrap());
        assert_eq!(retry_after_delay_ms(&headers), Some(150));
    }

    #[tokio::test(start_paused = true)]
    async fn server_delay_is_observed_without_a_signal() {
        let start = tokio::time::Instant::now();
        let attempts = std::cell::Cell::new(0);
        let result = retry_provider_request(
            || {
                let attempt = attempts.get();
                attempts.set(attempt + 1);
                async move {
                    if attempt == 0 {
                        Err(ProviderHttpError::from_response(429, None, "busy")
                            .with_retry_after_ms(Some(2000)))
                    } else {
                        Ok(())
                    }
                }
            },
            ProviderRetryOptions {
                max_retries: Some(1),
                ..Default::default()
            },
        )
        .await;
        assert!(result.is_ok());
        assert_eq!(start.elapsed(), Duration::from_secs(2));
    }

    #[tokio::test]
    async fn retry_header_can_disable_server_error_retries() {
        let attempts = std::cell::Cell::new(0);
        let result: Result<(), _> = retry_provider_request(
            || {
                attempts.set(attempts.get() + 1);
                async {
                    Err(ProviderHttpError::from_response(503, None, "busy").with_retryable(false))
                }
            },
            ProviderRetryOptions {
                max_retries: Some(3),
                ..Default::default()
            },
        )
        .await;
        assert!(result.is_err());
        assert_eq!(attempts.get(), 1);
    }

    #[tokio::test]
    async fn cancellation_interrupts_a_hung_request() {
        let signal = tokio_util::sync::CancellationToken::new();
        let cancel = signal.clone();
        let request = tokio::spawn(async move {
            retry_provider_request(
                std::future::pending::<Result<(), ProviderHttpError>>,
                ProviderRetryOptions {
                    signal: Some(signal),
                    ..Default::default()
                },
            )
            .await
        });
        tokio::task::yield_now().await;
        cancel.cancel();
        let error = tokio::time::timeout(Duration::from_secs(1), request)
            .await
            .unwrap()
            .unwrap()
            .unwrap_err();
        assert!(!error.is_retryable());
    }

    #[test]
    fn parses_all_http_date_forms_and_rejects_nonfinite_delays() {
        let expected = http_date_timestamp_ms("Sun, 06 Nov 1994 08:49:37 GMT").unwrap();
        assert_eq!(
            http_date_timestamp_ms("Sunday, 06-Nov-94 08:49:37 GMT"),
            Some(expected)
        );
        assert_eq!(
            http_date_timestamp_ms("Sun Nov  6 08:49:37 1994"),
            Some(expected)
        );
        for value in ["NaN", "inf", "-1"] {
            let mut headers = reqwest::header::HeaderMap::new();
            headers.insert("retry-after", value.parse().unwrap());
            assert_eq!(retry_after_delay_ms(&headers), None);
        }
    }
}
