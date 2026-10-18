//! Resolve fractional configuration into integer token budgets.

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[error("{field} must be finite and in (0, 1], received {value}")]
pub struct RatioError {
    pub field: &'static str,
    pub value: f64,
}

pub(crate) fn validate_ratio(field: &'static str, value: f64) -> Result<(), RatioError> {
    if value.is_finite() && value > 0.0 && value <= 1.0 {
        Ok(())
    } else {
        Err(RatioError { field, value })
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Rounding {
    Floor,
    Ceil,
}

/// Use the ratio's shortest round-trip decimal representation. This gives
/// configured decimal fractions (e.g. 0.6) their expected meaning without
/// converting the u64 token count to f64 or losing precision above 2^53.
pub(crate) fn ratio_tokens(
    tokens: u64,
    ratio: f64,
    rounding: Rounding,
    field: &'static str,
) -> Result<u64, RatioError> {
    validate_ratio(field, ratio)?;
    let invalid = || RatioError {
        field,
        value: ratio,
    };
    let decimal = ratio.to_string();
    let (mantissa, exponent) = match decimal.split_once(['e', 'E']) {
        Some((mantissa, exponent)) => (mantissa, exponent.parse::<i32>().map_err(|_| invalid())?),
        None => (decimal.as_str(), 0),
    };
    let fractional_digits = mantissa
        .split_once('.')
        .map_or(0, |(_, digits)| digits.len());
    let scale = i32::try_from(fractional_digits).map_err(|_| invalid())? - exponent;
    let digits = mantissa.replace('.', "");
    // A finite f64 has at most 17 significant decimal digits, so its
    // coefficient multiplied by any u64 fits in u128.
    let coefficient = digits
        .trim_start_matches('0')
        .parse::<u128>()
        .map_err(|_| invalid())?;
    let product = u128::from(tokens)
        .checked_mul(coefficient)
        .ok_or_else(invalid)?;
    let scale = u32::try_from(scale).map_err(|_| invalid())?;
    let result = if scale > 38 {
        // 10^39 is larger than every possible product above.
        u128::from(matches!(rounding, Rounding::Ceil) && product > 0)
    } else {
        let denominator = 10u128.pow(scale);
        match rounding {
            Rounding::Floor => product / denominator,
            Rounding::Ceil => product.div_ceil(denominator),
        }
    };
    u64::try_from(result).map_err(|_| invalid())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimal_budgets_have_explicit_rounding() {
        assert_eq!(ratio_tokens(10, 0.6, Rounding::Floor, "ratio").unwrap(), 6);
        assert_eq!(ratio_tokens(11, 0.6, Rounding::Ceil, "ratio").unwrap(), 7);
        assert_eq!(ratio_tokens(11, 0.75, Rounding::Floor, "ratio").unwrap(), 8);
    }

    #[test]
    fn full_width_counts_keep_integer_precision() {
        let count = (1u64 << 53) + 1;
        assert_eq!(
            ratio_tokens(count, 1.0, Rounding::Floor, "ratio").unwrap(),
            count
        );
        assert_eq!(
            ratio_tokens(u64::MAX, 1.0, Rounding::Ceil, "ratio").unwrap(),
            u64::MAX
        );
        assert_eq!(
            ratio_tokens(u64::MAX, 0.5, Rounding::Floor, "ratio").unwrap(),
            u64::MAX / 2
        );
        assert_eq!(
            ratio_tokens(u64::MAX, 0.5, Rounding::Ceil, "ratio").unwrap(),
            u64::MAX / 2 + 1
        );
    }

    #[test]
    fn very_small_ratios_do_not_overflow() {
        for ratio in [f64::from_bits(1), 1e-100, 1e-38] {
            assert_eq!(
                ratio_tokens(u64::MAX, ratio, Rounding::Floor, "ratio").unwrap(),
                0
            );
            assert_eq!(
                ratio_tokens(u64::MAX, ratio, Rounding::Ceil, "ratio").unwrap(),
                1
            );
            assert_eq!(ratio_tokens(0, ratio, Rounding::Ceil, "ratio").unwrap(), 0);
        }
    }
}
