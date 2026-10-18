//! Context-overflow detection from provider error messages and usage shapes.
//!
//! Patterns match overflow errors returned by known providers (Anthropic,
//! OpenAI/LiteLLM, Google, xAI, Groq, OpenRouter, Together, llama.cpp,
//! LM Studio, GitHub Copilot, MiniMax, Kimi, DS4, Mistral, DashScope/Qwen,
//! Ollama, Cerebras, Bedrock, z.ai, Xiaomi MiMo); see the TypeScript source
//! for example messages per provider.

use std::sync::LazyLock;

use regex::Regex;

use crate::types::{AssistantMessage, StopReason};

static OVERFLOW_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    [
        r"(?i)prompt is too long",                    // Anthropic token overflow
        r"(?i)request_too_large", // Anthropic request byte-size overflow (HTTP 413)
        r"(?i)input is too long for requested model", // Amazon Bedrock
        r"(?i)exceeds the context window", // OpenAI (Completions & Responses)
        r"(?i)exceeds (?:the )?(?:model'?s )?maximum context length(?: of [\d,]+ tokens?|\s*\([\d,]+\))", // LiteLLM
        r"(?i)input token count.*exceeds the maximum", // Google (Gemini)
        r"(?i)maximum prompt length is \d+",           // xAI (Grok)
        r"(?i)reduce the length of the messages",      // Groq
        r"(?i)maximum context length is \d+ tokens",   // OpenRouter (most backends)
        r"(?i)exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?", // OpenRouter/Poolside
        r"(?i)input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)", // Together AI
        r"(?i)exceeds the limit of \d+",           // GitHub Copilot
        r"(?i)exceeds the available context size", // llama.cpp server
        r"(?i)greater than the context length",    // LM Studio
        r"(?i)context window exceeds limit",       // MiniMax
        r"(?i)exceeded model token limit",         // Kimi For Coding
        r"(?i)too large for model with \d+ maximum context length", // Mistral
        r"(?i)prompt has [\d,]+ tokens?, but the configured context size is [\d,]+ tokens?", // DS4 server
        r"(?i)model_context_window_exceeded", // z.ai non-standard finish_reason
        r"(?i)prompt too long; exceeded (?:max )?context length", // Ollama explicit overflow
        r"(?i)range of input length should be", // DashScope / Qwen Token Plan
        r"(?i)context[_ ]length[_ ]exceeded", // Generic fallback
        r"(?i)too many tokens",               // Generic fallback
        r"(?i)token limit exceeded",          // Generic fallback
        r"(?i)^4(?:00|13)\s*(?:status code)?\s*\(no body\)", // Cerebras: 400/413 no body
    ]
    .iter()
    .map(|pattern| Regex::new(pattern).expect("static overflow pattern"))
    .collect()
});

/// Patterns that indicate non-overflow errors (rate limiting, server errors).
/// Messages matching these are excluded from overflow detection even when they
/// also match an overflow pattern (e.g. Bedrock "ThrottlingException: Too many
/// tokens, please wait before trying again.").
static NON_OVERFLOW_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    [
        r"(?i)^(Throttling error|Service unavailable):", // AWS Bedrock non-overflow errors
        r"(?i)rate limit",                               // Generic rate limiting
        r"(?i)too many requests",                        // Generic HTTP 429 style
    ]
    .iter()
    .map(|pattern| Regex::new(pattern).expect("static non-overflow pattern"))
    .collect()
});

/// Check if an assistant message represents a context overflow error.
///
/// Three cases are handled:
/// 1. Error-based overflow: most providers return stop reason `Error` with a
///    recognizable message pattern.
/// 2. Silent overflow: some providers accept overflow requests and return
///    successfully; detected via `usage.input + cache_read > context_window`.
/// 3. Length-stop overflow: Xiaomi MiMo truncates input to the context window
///    and returns `Length` with zero output; detected via input filling ≥99% of
///    the window.
pub fn is_context_overflow(message: &AssistantMessage, context_window: Option<u64>) -> bool {
    // Case 1: error message patterns
    if message.stop_reason == StopReason::Error
        && let Some(error) = &message.error_message
    {
        let is_non_overflow = NON_OVERFLOW_PATTERNS
            .iter()
            .any(|pattern| pattern.is_match(error));
        if !is_non_overflow
            && OVERFLOW_PATTERNS
                .iter()
                .any(|pattern| pattern.is_match(error))
        {
            return true;
        }
    }

    if let Some(context_window) = context_window {
        // Case 2: silent overflow (z.ai style) — successful but usage exceeds context
        if message.stop_reason == StopReason::Stop {
            let input_tokens = message.usage.input.saturating_add(message.usage.cache_read);
            if input_tokens > context_window {
                return true;
            }
        }

        // Case 3: length-stop overflow (Xiaomi MiMo style)
        if message.stop_reason == StopReason::Length && message.usage.output == 0 {
            let input_tokens = message.usage.input.saturating_add(message.usage.cache_read);
            if input_tokens as f64 >= context_window as f64 * 0.99 {
                return true;
            }
        }
    }

    false
}

/// Check whether a length stop ended below the caller's or model's intended
/// output limit: such responses may warrant one bounded compact-and-retry.
/// `desired_max_output` must be the original limit before context clamping.
pub fn is_recoverable_length(message: &AssistantMessage, desired_max_output: u64) -> bool {
    message.stop_reason == StopReason::Length
        && desired_max_output > 0
        && message.usage.output < desired_max_output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Api, ProviderId, Usage};

    fn error_message(text: &str) -> AssistantMessage {
        AssistantMessage {
            content: Vec::new(),
            api: Api::AnthropicMessages,
            provider: ProviderId::new("test"),
            model: "m".to_owned(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: Usage::default(),
            stop_reason: StopReason::Error,
            deferred: None,
            error_message: Some(text.to_owned()),
            raw_stop_reason: None,
            end_turn: None,
            timestamp: 0,
        }
    }

    #[test]
    fn detects_anthropic_overflow() {
        let message = error_message("prompt is too long: 213462 tokens > 200000 maximum");
        assert!(is_context_overflow(&message, None));
    }

    #[test]
    fn detects_litellm_overflow() {
        let message = error_message(
            "Requested token count exceeds the model's maximum context length of 131072 tokens",
        );
        assert!(is_context_overflow(&message, None));
    }

    #[test]
    fn detects_cerebras_no_body() {
        assert!(is_context_overflow(
            &error_message("400 status code (no body)"),
            None
        ));
        assert!(is_context_overflow(&error_message("413 (no body)"), None));
    }

    #[test]
    fn throttling_excluded_from_overflow() {
        assert!(!is_context_overflow(
            &error_message("Throttling error: Too many tokens, please wait before trying again."),
            None
        ));
        assert!(!is_context_overflow(
            &error_message("rate limit exceeded: too many tokens"),
            None
        ));
    }

    #[test]
    fn silent_overflow_detected_via_usage() {
        let mut message = error_message("unused");
        message.stop_reason = StopReason::Stop;
        message.usage.input = 150_000;
        assert!(is_context_overflow(&message, Some(100_000)));
    }

    #[test]
    fn length_stop_overflow_detected() {
        let mut message = error_message("unused");
        message.stop_reason = StopReason::Length;
        message.usage.input = 99_900;
        message.usage.output = 0;
        assert!(is_context_overflow(&message, Some(100_000)));
    }

    #[test]
    fn recoverable_length_true_when_below_limit() {
        let mut message = error_message("unused");
        message.stop_reason = StopReason::Length;
        message.usage.output = 10;
        assert!(is_recoverable_length(&message, 100));
        assert!(!is_recoverable_length(&message, 10));
    }
}
