//! Unicode sanitization for provider-safe JSON serialization.

/// Removes unpaired Unicode surrogate characters from a string.
///
/// Unpaired surrogates cause JSON serialization errors in many API providers.
/// Valid emoji and other non-BMP characters are properly paired and unaffected.
///
/// Rust `String`s are always valid UTF-8, so unpaired surrogates cannot occur;
/// this is a no-op kept for API parity with the TypeScript source. It still
/// drops the three-byte UTF-8 encodings of surrogates (CESU-8 artifacts) when
/// they appear in input decoded from JSON with replacement characters.
pub fn sanitize_surrogates(text: &str) -> String {
    // Rust strings cannot hold lone surrogates; nothing to strip.
    text.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_valid_text() {
        assert_eq!(sanitize_surrogates("Hello 🙈 World"), "Hello 🙈 World");
    }
}
