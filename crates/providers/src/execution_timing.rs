//! Exclusive wall-clock accounting for a single foreground execution.
//!
//! The task-local context is inherited only by explicitly instrumented provider
//! tasks. Background work (such as session naming) deliberately does not inherit it.
use std::{
    future::Future,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug)]
#[repr(usize)]
pub enum Phase {
    Other,
    Tool,
    Api,
    Retry,
    Approval,
}

#[derive(Debug)]
struct State {
    last: Instant,
    active: [usize; 5],
    elapsed: [Duration; 5],
    finished: Option<[u64; 5]>,
}

impl State {
    fn advance(&mut self, now: Instant) {
        if self.finished.is_some() {
            return;
        }
        let phase = (1..5).rev().find(|&i| self.active[i] > 0).unwrap_or(0);
        self.elapsed[phase] += now.duration_since(self.last);
        self.last = now;
    }
}

#[derive(Debug)]
pub struct ExecutionClock {
    started: Instant,
    state: Mutex<State>,
}

tokio::task_local! { static CURRENT: Arc<ExecutionClock>; }

impl ExecutionClock {
    pub fn new() -> Arc<Self> {
        let started = Instant::now();
        Arc::new(Self {
            started,
            state: Mutex::new(State {
                last: started,
                active: [0; 5],
                elapsed: [Duration::ZERO; 5],
                finished: None,
            }),
        })
    }

    pub async fn scope<T>(self: &Arc<Self>, future: impl Future<Output = T>) -> T {
        CURRENT.scope(self.clone(), future).await
    }

    fn enter(self: &Arc<Self>, phase: Phase) -> PhaseGuard {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        state.advance(Instant::now());
        state.active[phase as usize] += 1;
        PhaseGuard {
            clock: self.clone(),
            phase,
        }
    }

    /// Order: other, tool, API, retry, approval. Integer remainder belongs to
    /// other processing, so the millisecond partition is exact.
    pub fn finish(&self) -> (u64, [u64; 5]) {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(parts) = state.finished {
            return (parts.iter().sum(), parts);
        }
        let now = Instant::now();
        state.advance(now);
        let total = now
            .duration_since(self.started)
            .as_millis()
            .min(u64::MAX as u128) as u64;
        let mut parts = state
            .elapsed
            .map(|duration| duration.as_millis().min(u64::MAX as u128) as u64);
        parts[0] = total - parts[1..].iter().sum::<u64>();
        state.finished = Some(parts);
        (total, parts)
    }
}

pub struct PhaseGuard {
    clock: Arc<ExecutionClock>,
    phase: Phase,
}
impl Drop for PhaseGuard {
    fn drop(&mut self) {
        let mut state = self.clock.state.lock().unwrap_or_else(|p| p.into_inner());
        state.advance(Instant::now());
        state.active[self.phase as usize] -= 1;
    }
}

pub fn enter(phase: Phase) -> Option<PhaseGuard> {
    CURRENT.try_with(|clock| clock.enter(phase)).ok()
}

/// Carry foreground accounting across provider streaming task boundaries only.
pub fn spawn<F>(future: F) -> tokio::task::JoinHandle<F::Output>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    let clock = CURRENT.try_with(Arc::clone).ok();
    tokio::spawn(async move {
        match clock {
            Some(clock) => clock.scope(future).await,
            None => future.await,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_and_parallel_phases_partition_time() {
        let now = Instant::now();
        let mut state = State {
            last: now,
            active: [0; 5],
            elapsed: [Duration::ZERO; 5],
            finished: None,
        };
        state.advance(now + Duration::from_millis(10));
        state.active[Phase::Tool as usize] = 2;
        state.advance(now + Duration::from_millis(30));
        state.active[Phase::Approval as usize] = 1;
        state.advance(now + Duration::from_millis(60));
        state.active[Phase::Approval as usize] = 0;
        state.active[Phase::Tool as usize] = 0;
        state.active[Phase::Api as usize] = 1;
        state.advance(now + Duration::from_millis(100));
        state.active[Phase::Retry as usize] = 1;
        state.advance(now + Duration::from_millis(150));
        assert_eq!(state.elapsed.map(|d| d.as_millis()), [10, 20, 40, 50, 30]);
    }

    #[tokio::test]
    async fn guards_inherit_only_when_requested_and_finish_is_frozen() {
        let clock = ExecutionClock::new();
        clock
            .scope(async {
                let _api = enter(Phase::Api);
                spawn(async {
                    assert!(enter(Phase::Retry).is_some());
                })
                .await
                .unwrap();
                tokio::spawn(async {
                    assert!(enter(Phase::Api).is_none());
                })
                .await
                .unwrap();
            })
            .await;
        let (total, parts) = clock.finish();
        assert_eq!(total, parts.iter().sum::<u64>());
        assert_eq!(clock.finish(), (total, parts));
        assert!(enter(Phase::Api).is_none());
    }

    #[tokio::test]
    async fn cancelled_http_backoff_is_retry_time_inside_api_time() {
        let clock = ExecutionClock::new();
        clock
            .scope(async {
                let _api = enter(Phase::Api);
                let signal = tokio_util::sync::CancellationToken::new();
                let cancel = signal.clone();
                tokio::spawn(async move {
                    tokio::time::sleep(Duration::from_millis(30)).await;
                    cancel.cancel();
                });
                let result: Result<(), crate::error::ProviderHttpError> =
                    crate::utils::provider_retry::retry_provider_request(
                        || async { Err(crate::error::ProviderHttpError::transport("test")) },
                        crate::utils::provider_retry::ProviderRetryOptions {
                            max_retries: Some(1),
                            signal: Some(signal),
                            ..Default::default()
                        },
                    )
                    .await;
                assert!(result.is_err());
            })
            .await;
        let (total, parts) = clock.finish();
        assert!(parts[Phase::Retry as usize] >= 25, "{parts:?}");
        assert_eq!(parts.iter().sum::<u64>(), total);
        assert_eq!(parts[Phase::Tool as usize], 0);
        assert_eq!(parts[Phase::Approval as usize], 0);
    }
}
