//! Shared utilities: token estimation, overflow detection, retry policies,
//! JSON repair/partial parsing, validation, and small helpers.

pub mod deferred_tools;
pub mod diagnostics;
pub mod estimate;
pub mod hash;
pub mod headers;
pub mod json_parse;
pub mod overflow;
pub mod provider_env;
pub mod provider_retry;
pub(crate) mod proxy;
pub mod retry;
pub mod sanitize;
pub mod text;
pub mod user_agent;
pub mod uuid;
pub mod validation;

use std::time::{SystemTime, UNIX_EPOCH};

/// Current Unix timestamp in milliseconds.
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default()
}

/// Abortable sleep: resolves after `ms` or errors when the token fires first.
pub async fn sleep_ms(ms: u64, token: &tokio_util::sync::CancellationToken) -> Result<(), Aborted> {
    tokio::select! {
        biased;
        _ = token.cancelled() => Err(Aborted),
        _ = tokio::time::sleep(std::time::Duration::from_millis(ms)) => Ok(()),
    }
}

/// Sleep even when cancellation is not configured.
pub(crate) async fn sleep_with_optional_abort(
    duration: std::time::Duration,
    token: Option<&tokio_util::sync::CancellationToken>,
) -> Result<(), Aborted> {
    let _timing = crate::execution_timing::enter(crate::execution_timing::Phase::Retry);
    match token {
        Some(token) => tokio::select! {
            biased;
            _ = token.cancelled() => Err(Aborted),
            _ = tokio::time::sleep(duration) => Ok(()),
        },
        None => {
            tokio::time::sleep(duration).await;
            Ok(())
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("the operation was aborted")]
pub struct Aborted;

/// Stop waiting and drop `operation` when the token fires.
pub async fn race_with_abort<T>(
    operation: impl Future<Output = T> + Send,
    token: &tokio_util::sync::CancellationToken,
) -> Result<T, Aborted> {
    if token.is_cancelled() {
        return Err(Aborted);
    }
    tokio::select! {
        biased;
        _ = token.cancelled() => Err(Aborted),
        result = operation => Ok(result),
    }
}

/// Combine cancellation tokens into one child token that fires when any parent
/// fires. Dropping the guard cancels the child (listener cleanup).
pub fn combine_abort_tokens(tokens: &[&tokio_util::sync::CancellationToken]) -> CombinedAbortToken {
    let child = tokio_util::sync::CancellationToken::new();
    let mut tasks = Vec::new();
    for token in tokens {
        if token.is_cancelled() {
            child.cancel();
        }
        let parent = (*token).clone();
        let child_clone = child.clone();
        tasks.push(tokio::spawn(async move {
            parent.cancelled().await;
            child_clone.cancel();
        }));
    }
    CombinedAbortToken {
        token: child,
        _tasks: tasks,
    }
}

pub struct CombinedAbortToken {
    pub token: tokio_util::sync::CancellationToken,
    _tasks: Vec<tokio::task::JoinHandle<()>>,
}

impl Drop for CombinedAbortToken {
    fn drop(&mut self) {
        self.token.cancel();
        for task in &self._tasks {
            task.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn sleep_resolves_after_duration() {
        let token = tokio_util::sync::CancellationToken::new();
        let started = std::time::Instant::now();
        sleep_ms(20, &token).await.unwrap();
        assert!(started.elapsed() >= std::time::Duration::from_millis(20));
    }

    #[tokio::test]
    async fn sleep_aborts_when_cancelled() {
        let token = tokio_util::sync::CancellationToken::new();
        token.cancel();
        assert!(sleep_ms(10_000, &token).await.is_err());
    }

    #[tokio::test]
    async fn race_returns_operation_result() {
        let token = tokio_util::sync::CancellationToken::new();
        let value = race_with_abort(async { 7 }, &token).await.unwrap();
        assert_eq!(value, 7);
    }

    #[tokio::test]
    async fn race_aborts_when_cancelled_first() {
        let token = tokio_util::sync::CancellationToken::new();
        token.cancel();
        let result = race_with_abort(
            async {
                tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                7
            },
            &token,
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn combined_token_fires_with_any_parent() {
        let parent_a = tokio_util::sync::CancellationToken::new();
        let parent_b = tokio_util::sync::CancellationToken::new();
        let combined = combine_abort_tokens(&[&parent_a, &parent_b]);
        parent_b.cancel();
        combined.token.cancelled().await;
        assert!(combined.token.is_cancelled());
    }
    #[tokio::test]
    async fn combined_token_is_cancelled_immediately_and_on_guard_drop() {
        let parent = tokio_util::sync::CancellationToken::new();
        parent.cancel();
        let guard = combine_abort_tokens(&[&parent]);
        assert!(guard.token.is_cancelled());
        let guard = combine_abort_tokens(&[]);
        let child = guard.token.clone();
        drop(guard);
        assert!(child.is_cancelled());
    }
}
