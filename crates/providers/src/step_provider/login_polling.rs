//! The polling loop that drives the Step CLI-login flow to completion.
//!
//! Rust port of `packages/providers/src/step-provider/login-polling.ts`.

use tokio_util::sync::CancellationToken;

use super::login_client::{
    StepCliLoginFailureReason, StepCliLoginInit, StepCliLoginPoll, StepCliLoginTransport,
    StepLoginError, is_retryable_step_cli_login_error, poll_step_cli_login,
    with_step_cli_login_timeout,
};

/// Never poll faster than this floor, even when the server asks for it.
pub const MIN_STEP_POLL_INTERVAL_MS: u64 = 1_000;

/// Never park the terminal for longer than this between polls.
const MAX_STEP_POLL_INTERVAL_MS: u64 = 60_000;

/// The browser approved the sign-in: the credential to store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepCliLoginReady {
    pub api_key: String,
    /// Bounded account identifier, when the server reported one.
    pub uid: Option<String>,
}

/// Inputs to [`wait_for_step_cli_login`].
pub struct WaitForStepCliLoginInput<'a> {
    pub init: StepCliLoginInit,
    pub auth_base_url: &'a str,
    pub poll_token: &'a str,
    pub signal: Option<&'a CancellationToken>,
    pub transport: &'a dyn StepCliLoginTransport,
}

/// Drive the flow to completion: poll, sleep, repeat — until the server reports
/// a credential, the flow is rejected, the deadline elapses, or the caller
/// cancels. Transient failures are retried; fatal ones surface immediately.
///
/// The outer login owns the local timeout; this loop only bounds itself by the
/// server-reported expiry.
pub async fn wait_for_step_cli_login(
    input: WaitForStepCliLoginInput<'_>,
) -> Result<StepCliLoginReady, StepLoginError> {
    // `expires_at` is Unix seconds on the wire; the deadline is computed in ms.
    let remaining_ms = input
        .init
        .expires_at
        .saturating_mul(1_000)
        .saturating_sub(crate::utils::now_ms());
    let timeout_error = || {
        StepLoginError::TimedOut(
            "Step sign-in timed out before it was approved in the browser.".to_owned(),
        )
    };
    if input.signal.is_some_and(|signal| signal.is_cancelled()) {
        return Err(StepLoginError::Aborted);
    }
    if remaining_ms == 0 {
        return Err(timeout_error());
    }
    let interval_ms = step_poll_interval_ms(input.init.poll_interval_sec);
    with_step_cli_login_timeout(
        remaining_ms,
        input.signal,
        timeout_error,
        |signal| async move {
            loop {
                let outcome = poll_step_cli_login(super::login_client::PollStepCliLoginInput {
                    auth_base_url: input.auth_base_url,
                    poll_token: input.poll_token,
                    flow_id: &input.init.flow_id,
                    signal: Some(&signal),
                    transport: input.transport,
                })
                .await;
                // A result that races with cancellation is discarded. The caller's
                // own token is checked alongside the deadline: the bridge that
                // forwards it into the deadline runs as a task, so the deadline can
                // lag a cancellation that happened while the poll was in flight.
                if signal.is_cancelled() || input.signal.is_some_and(|signal| signal.is_cancelled())
                {
                    return Err(StepLoginError::Aborted);
                }
                match outcome {
                    Ok(StepCliLoginPoll::Ready { api_key, uid }) => {
                        return Ok(StepCliLoginReady { api_key, uid });
                    }
                    Ok(StepCliLoginPoll::Failed { reason }) => {
                        return Err(StepLoginError::Rejected(
                            match reason {
                                StepCliLoginFailureReason::Denied => {
                                    "Sign-in was denied in the browser."
                                }
                                StepCliLoginFailureReason::Unknown => {
                                    "Step sign-in failed in the browser."
                                }
                            }
                            .to_owned(),
                        ));
                    }
                    Ok(StepCliLoginPoll::Pending) => {}
                    Err(error) => {
                        if !is_retryable_step_cli_login_error(&error) {
                            return Err(error);
                        }
                    }
                }
                crate::utils::sleep_ms(interval_ms, &signal)
                    .await
                    .map_err(|_| StepLoginError::Aborted)?;
            }
        },
    )
    .await
}

/// Clamp the server-requested interval into the `[floor, 60s]` window the
/// terminal honors.
fn step_poll_interval_ms(poll_interval_sec: u64) -> u64 {
    poll_interval_sec
        .saturating_mul(1_000)
        .clamp(MIN_STEP_POLL_INTERVAL_MS, MAX_STEP_POLL_INTERVAL_MS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step_provider::test_support::{FakeTransport, HandlerTransport};
    use serde_json::json;

    const AUTH_BASE_URL: &str = "https://auth.example.test";

    fn defaults() -> StepCliLoginInit {
        StepCliLoginInit {
            flow_id: "flow-1".to_owned(),
            authorize_url: "https://auth.example.test/cli-login-remote?flow_id=flow-1".to_owned(),
            poll_interval_sec: 1,
            expires_at: crate::utils::now_ms() / 1000 + 600,
        }
    }

    #[test]
    fn poll_interval_is_clamped_to_the_floor_and_ceiling() {
        assert_eq!(step_poll_interval_ms(0), MIN_STEP_POLL_INTERVAL_MS);
        assert_eq!(step_poll_interval_ms(1), 1_000);
        assert_eq!(step_poll_interval_ms(2), 2_000);
        assert_eq!(step_poll_interval_ms(120), 60_000);
    }

    #[tokio::test]
    async fn keeps_polling_through_transient_failures_until_a_credential_arrives() {
        let transport = FakeTransport::json(vec![
            (json!({ "desc": "busy" }), 429),
            (json!({ "state": "pending" }), 200),
            (
                json!({ "state": "ready", "credential": { "api_key": "key" } }),
                200,
            ),
        ]);
        let ready = wait_for_step_cli_login(WaitForStepCliLoginInput {
            init: defaults(),
            auth_base_url: AUTH_BASE_URL,
            poll_token: "token",
            signal: None,
            transport: &transport,
        })
        .await
        .expect("login succeeds");
        assert_eq!(
            ready,
            StepCliLoginReady {
                api_key: "key".to_owned(),
                uid: None
            }
        );
        assert_eq!(transport.requests().len(), 3);
    }

    #[tokio::test]
    async fn stops_at_the_server_expiry() {
        // Real wall-clock timing (the crate's dev-dependency tokio has no
        // `test-util`, so virtual time is unavailable): a two-second server
        // expiry against the one-second poll floor yields at least two polls
        // before the deadline wins. The exact count is not pinned because the
        // final sleep and the deadline can land in the same timer tick.
        let transport = FakeTransport::repeated(json!({ "state": "pending" }), 200, 8);
        let mut init = defaults();
        init.expires_at = crate::utils::now_ms() / 1000 + 2;
        let error = wait_for_step_cli_login(WaitForStepCliLoginInput {
            init,
            auth_base_url: AUTH_BASE_URL,
            poll_token: "token",
            signal: None,
            transport: &transport,
        })
        .await
        .expect_err("must time out");
        assert!(error.to_string().contains("timed out"), "message: {error}");
        assert!(transport.requests().len() >= 2);
    }

    #[tokio::test]
    async fn surfaces_a_denied_flow_immediately() {
        let transport = FakeTransport::json(vec![(
            json!({ "state": "failed", "reason": "denied" }),
            200,
        )]);
        let error = wait_for_step_cli_login(WaitForStepCliLoginInput {
            init: defaults(),
            auth_base_url: AUTH_BASE_URL,
            poll_token: "token",
            signal: None,
            transport: &transport,
        })
        .await
        .expect_err("must be rejected");
        assert_eq!(error.to_string(), "Sign-in was denied in the browser.");
        assert_eq!(transport.requests().len(), 1);
    }

    #[tokio::test]
    async fn surfaces_an_unclassified_browser_failure() {
        let transport = FakeTransport::json(vec![(
            json!({ "state": "failed", "reason": "something-new" }),
            200,
        )]);
        let error = wait_for_step_cli_login(WaitForStepCliLoginInput {
            init: defaults(),
            auth_base_url: AUTH_BASE_URL,
            poll_token: "token",
            signal: None,
            transport: &transport,
        })
        .await
        .expect_err("must be rejected");
        assert_eq!(error.to_string(), "Step sign-in failed in the browser.");
    }

    #[tokio::test]
    async fn does_not_retry_a_fatal_poll_failure() {
        let transport = FakeTransport::json(vec![(json!({ "desc": "gone" }), 404)]);
        let error = wait_for_step_cli_login(WaitForStepCliLoginInput {
            init: defaults(),
            auth_base_url: AUTH_BASE_URL,
            poll_token: "token",
            signal: None,
            transport: &transport,
        })
        .await
        .expect_err("must fail");
        match error {
            StepLoginError::CliLogin(super::super::login_client::StepCliLoginError::Request(
                request,
            )) => {
                assert_eq!(request.http_status, Some(404));
            }
            other => panic!("expected a CLI login request error, got {other:?}"),
        }
        assert_eq!(transport.requests().len(), 1);
    }

    #[tokio::test]
    async fn cancels_during_the_wait_between_polls() {
        let transport = FakeTransport::repeated(json!({ "state": "pending" }), 200, 4);
        let signal = CancellationToken::new();
        let canceller = {
            let signal = signal.clone();
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                signal.cancel();
            })
        };
        let error = wait_for_step_cli_login(WaitForStepCliLoginInput {
            init: defaults(),
            auth_base_url: AUTH_BASE_URL,
            poll_token: "token",
            signal: Some(&signal),
            transport: &transport,
        })
        .await
        .expect_err("must abort");
        assert!(matches!(error, StepLoginError::Aborted));
        assert_eq!(transport.requests().len(), 1);
        canceller.await.expect("canceller task");
    }

    #[tokio::test]
    async fn expires_even_while_a_poll_is_hung() {
        let transport = FakeTransport::hanging();
        let mut init = defaults();
        init.expires_at = crate::utils::now_ms() / 1000 + 2;
        let error = wait_for_step_cli_login(WaitForStepCliLoginInput {
            init,
            auth_base_url: AUTH_BASE_URL,
            poll_token: "token",
            signal: None,
            transport: &transport,
        })
        .await
        .expect_err("must time out");
        assert!(error.to_string().contains("timed out"), "message: {error}");
        assert_eq!(transport.requests().len(), 1);
    }

    #[tokio::test]
    async fn rejects_a_ready_result_that_races_with_cancellation() {
        let signal = CancellationToken::new();
        let canceller = signal.clone();
        let transport = HandlerTransport::new(Box::new(move |_request| {
            canceller.cancel();
            Ok(crate::step_provider::test_support::json_response(
                json!({ "state": "ready", "credential": { "api_key": "key" } }),
                200,
            ))
        }));
        let error = wait_for_step_cli_login(WaitForStepCliLoginInput {
            init: defaults(),
            auth_base_url: AUTH_BASE_URL,
            poll_token: "token",
            signal: Some(&signal),
            transport: &transport,
        })
        .await
        .expect_err("must abort");
        assert!(matches!(error, StepLoginError::Aborted));
        assert_eq!(transport.requests().len(), 1);
    }

    #[tokio::test]
    async fn rejects_an_already_cancelled_signal_before_polling() {
        let transport = FakeTransport::repeated(json!({ "state": "pending" }), 200, 2);
        let signal = CancellationToken::new();
        signal.cancel();
        let error = wait_for_step_cli_login(WaitForStepCliLoginInput {
            init: defaults(),
            auth_base_url: AUTH_BASE_URL,
            poll_token: "token",
            signal: Some(&signal),
            transport: &transport,
        })
        .await
        .expect_err("must abort");
        assert!(matches!(error, StepLoginError::Aborted));
        assert!(transport.requests().is_empty());
    }

    #[tokio::test]
    async fn rejects_an_already_expired_flow_before_polling() {
        let transport = FakeTransport::repeated(json!({ "state": "pending" }), 200, 2);
        let mut init = defaults();
        init.expires_at = crate::utils::now_ms() / 1000 - 5;
        let error = wait_for_step_cli_login(WaitForStepCliLoginInput {
            init,
            auth_base_url: AUTH_BASE_URL,
            poll_token: "token",
            signal: None,
            transport: &transport,
        })
        .await
        .expect_err("must time out");
        assert!(error.to_string().contains("timed out"), "message: {error}");
        assert!(transport.requests().is_empty());
    }
}
