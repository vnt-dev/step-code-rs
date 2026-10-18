//! OpenAI prompt-cache key clamping.

pub const OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH: usize = 64;

/// Truncate a prompt-cache key to OpenAI's 64-character limit, counting
/// characters (not bytes).
pub fn clamp_openai_prompt_cache_key(key: Option<&str>) -> Option<String> {
    let key = key?;
    if key.chars().count() <= OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH {
        return Some(key.to_owned());
    }
    Some(
        key.chars()
            .take(OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH)
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_keys_pass_through() {
        assert_eq!(
            clamp_openai_prompt_cache_key(Some("abc")),
            Some("abc".to_owned())
        );
        assert_eq!(clamp_openai_prompt_cache_key(None), None);
    }

    #[test]
    fn long_keys_truncated_to_64_chars() {
        let long = "x".repeat(100);
        let clamped = clamp_openai_prompt_cache_key(Some(&long)).unwrap();
        assert_eq!(clamped.chars().count(), OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH);
    }
}
