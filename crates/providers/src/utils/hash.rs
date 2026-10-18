//! Fast deterministic hash to shorten long strings (platform-independent
//! variant of the TypeScript imul-based hash; the algorithm is unchanged).

/// Hash a string to a short base-36 string. Uses the same mixing constants and
/// 32-bit wrapping arithmetic as the TypeScript `shortHash`.
pub fn short_hash(input: &str) -> String {
    let mut h1: u32 = 0xdeadbeef;
    let mut h2: u32 = 0x41c6ce57;
    for unit in input.encode_utf16() {
        h1 = (h1 ^ u32::from(unit)).wrapping_mul(2654435761);
        h2 = (h2 ^ u32::from(unit)).wrapping_mul(1597334677);
    }
    h1 = (h1 ^ (h1 >> 16)).wrapping_mul(2246822507) ^ (h2 ^ (h2 >> 13)).wrapping_mul(3266489909);
    h2 = (h2 ^ (h2 >> 16)).wrapping_mul(2246822507) ^ (h1 ^ (h1 >> 13)).wrapping_mul(3266489909);
    format!("{}{}", radix36(h2), radix36(h1))
}

fn radix36(value: u32) -> String {
    if value == 0 {
        return "0".to_owned();
    }
    let mut digits = Vec::new();
    let mut rest = value;
    while rest > 0 {
        let digit = (rest % 36) as usize;
        digits.push(std::char::from_digit(digit as u32, 36).unwrap_or('0'));
        rest /= 36;
    }
    digits.into_iter().rev().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_short() {
        let a = short_hash("hello");
        let b = short_hash("hello");
        assert_eq!(a, b);
        assert!(a.len() <= 13);
        assert_ne!(a, short_hash("world"));
    }
}
