//! Shared validation of public seconds-based timeouts into native durations.

use std::time::Duration;

use super::types::{ExecutionError, ExecutionErrorCode};

pub(crate) const MAX_TIMEOUT: Duration = Duration::from_millis(2_147_483_647);
pub(crate) const MAX_TIMEOUT_SECONDS: f64 = 2_147_483_647.0 / 1000.0;

pub(crate) fn resolve_timeout(seconds: Option<f64>) -> Result<Option<Duration>, ExecutionError> {
    let Some(seconds) = seconds else {
        return Ok(None);
    };
    let invalid = |message: String| ExecutionError::new(ExecutionErrorCode::Timeout, message);
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err(invalid(
            "Invalid timeout: must be a positive finite number of seconds".to_owned(),
        ));
    }
    // Validate the input before rounding so slightly excessive values cannot
    // sneak through a truncated milliseconds conversion.
    if seconds > MAX_TIMEOUT_SECONDS {
        return Err(invalid(format!(
            "Invalid timeout: maximum is {MAX_TIMEOUT_SECONDS} seconds"
        )));
    }
    let duration = Duration::try_from_secs_f64(seconds)
        .map_err(|error| invalid(format!("Invalid timeout: {error}")))?;
    if duration.is_zero() {
        return Err(invalid(
            "Invalid timeout: must resolve to at least one nanosecond".to_owned(),
        ));
    }
    // The documented maximum is an exact millisecond duration; its f64
    // seconds representation may round a few nanoseconds above that duration.
    Ok(Some(duration.min(MAX_TIMEOUT)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fractional_seconds_preserve_submillisecond_timeouts() {
        assert_eq!(resolve_timeout(None).unwrap(), None);
        assert_eq!(
            resolve_timeout(Some(1.5)).unwrap(),
            Some(Duration::from_millis(1500))
        );
        assert_eq!(
            resolve_timeout(Some(0.0005)).unwrap(),
            Some(Duration::from_micros(500))
        );
        assert_eq!(
            resolve_timeout(Some(1e-9)).unwrap(),
            Some(Duration::from_nanos(1))
        );
    }

    #[test]
    fn rejects_invalid_and_just_over_limit_inputs_before_rounding() {
        assert_eq!(
            resolve_timeout(Some(MAX_TIMEOUT_SECONDS)).unwrap(),
            Some(MAX_TIMEOUT)
        );
        for seconds in [
            0.0,
            -1.0,
            f64::NAN,
            f64::INFINITY,
            1e-20,
            MAX_TIMEOUT_SECONDS + 0.0001,
            f64::from_bits(MAX_TIMEOUT_SECONDS.to_bits() + 1),
        ] {
            assert!(
                resolve_timeout(Some(seconds)).is_err(),
                "accepted {seconds}"
            );
        }
    }
}
