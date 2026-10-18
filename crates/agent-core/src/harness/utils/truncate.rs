//! Shared truncation utilities for tool outputs.
//!
//! Ported from `@step-harness/agent-core` `src/harness/utils/truncate.ts`.
//! Truncation applies two independent limits — lines and bytes — and whichever
//! is hit first wins. Partial lines are never returned, except for the bash
//! tail-truncation edge case.
//!
//! The TypeScript source counts UTF-16 code units and manually handles
//! surrogate pairs; Rust strings are UTF-8, so byte lengths come from
//! [`str::len`] and the surrogate repair steps disappear.

use serde::{Deserialize, Serialize};

/// Default maximum number of lines.
pub const DEFAULT_MAX_LINES: usize = 2000;
/// Default maximum number of bytes (50KB).
pub const DEFAULT_MAX_BYTES: usize = 50 * 1024;
/// Maximum characters per grep match line.
pub const GREP_MAX_LINE_LENGTH: usize = 500;

/// Which limit was hit by truncation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TruncatedBy {
    Lines,
    Bytes,
}

/// Result of a truncation operation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TruncationResult {
    /// The truncated content.
    pub content: String,
    /// Whether truncation occurred.
    pub truncated: bool,
    /// Which limit was hit; `None` when not truncated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncated_by: Option<TruncatedBy>,
    /// Total number of lines in the original content.
    pub total_lines: usize,
    /// Total number of bytes in the original content.
    pub total_bytes: usize,
    /// Number of complete lines in the truncated output.
    pub output_lines: usize,
    /// Number of bytes in the truncated output.
    pub output_bytes: usize,
    /// Whether the last line was partially truncated (tail edge case only).
    pub last_line_partial: bool,
    /// Whether the first line exceeded the byte limit (head truncation).
    pub first_line_exceeds_limit: bool,
    /// The max lines limit that was applied.
    pub max_lines: usize,
    /// The max bytes limit that was applied.
    pub max_bytes: usize,
}

/// Options for truncation.
#[derive(Clone, Copy, Debug, Default)]
pub struct TruncationOptions {
    /// Maximum number of lines (default: 2000).
    pub max_lines: Option<usize>,
    /// Maximum number of bytes (default: 50KB).
    pub max_bytes: Option<usize>,
}

impl TruncationOptions {
    fn resolved(&self) -> (usize, usize) {
        (
            self.max_lines.unwrap_or(DEFAULT_MAX_LINES),
            self.max_bytes.unwrap_or(DEFAULT_MAX_BYTES),
        )
    }
}

/// Format bytes as a human-readable size.
pub fn format_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

/// Split content into lines for counting: a trailing newline does not produce
/// an extra empty line, and empty content has no lines.
fn split_lines_for_counting(content: &str) -> Vec<&str> {
    if content.is_empty() {
        return Vec::new();
    }
    let mut lines: Vec<&str> = content.split('\n').collect();
    if content.ends_with('\n') {
        lines.pop();
    }
    lines
}

fn not_truncated(
    content: &str,
    total_lines: usize,
    max_lines: usize,
    max_bytes: usize,
) -> TruncationResult {
    TruncationResult {
        content: content.to_owned(),
        truncated: false,
        truncated_by: None,
        total_lines,
        total_bytes: content.len(),
        output_lines: total_lines,
        output_bytes: content.len(),
        last_line_partial: false,
        first_line_exceeds_limit: false,
        max_lines,
        max_bytes,
    }
}

/// Truncate content from the head (keep the first N lines/bytes).
///
/// Suitable for file reads where you want to see the beginning. Never returns
/// partial lines; if the first line alone exceeds the byte limit, returns empty
/// content with `first_line_exceeds_limit` set.
pub fn truncate_head(content: &str, options: &TruncationOptions) -> TruncationResult {
    let (max_lines, max_bytes) = options.resolved();
    let lines = split_lines_for_counting(content);
    let total_lines = lines.len();
    let total_bytes = content.len();

    if total_lines <= max_lines && total_bytes <= max_bytes {
        return not_truncated(content, total_lines, max_lines, max_bytes);
    }

    // Check whether the first line alone exceeds the byte limit.
    let first_line_bytes = lines.first().map(|line| line.len()).unwrap_or(0);
    if first_line_bytes > max_bytes {
        return TruncationResult {
            content: String::new(),
            truncated: true,
            truncated_by: Some(TruncatedBy::Bytes),
            total_lines,
            total_bytes,
            output_lines: 0,
            output_bytes: 0,
            last_line_partial: false,
            first_line_exceeds_limit: true,
            max_lines,
            max_bytes,
        };
    }

    // Collect complete lines that fit.
    let mut output_lines: Vec<&str> = Vec::new();
    let mut output_bytes_count = 0usize;
    let mut truncated_by = TruncatedBy::Lines;

    for (index, line) in lines.iter().enumerate() {
        if index >= max_lines {
            break;
        }
        let line_bytes = line.len() + usize::from(index > 0); // +1 for the newline
        if output_bytes_count + line_bytes > max_bytes {
            truncated_by = TruncatedBy::Bytes;
            break;
        }
        output_lines.push(line);
        output_bytes_count += line_bytes;
    }

    // If we exited due to the line limit.
    if output_lines.len() >= max_lines && output_bytes_count <= max_bytes {
        truncated_by = TruncatedBy::Lines;
    }

    let output_content = output_lines.join("\n");
    TruncationResult {
        output_bytes: output_content.len(),
        content: output_content,
        truncated: true,
        truncated_by: Some(truncated_by),
        total_lines,
        total_bytes,
        output_lines: output_lines.len(),
        last_line_partial: false,
        first_line_exceeds_limit: false,
        max_lines,
        max_bytes,
    }
}

/// Truncate content from the tail (keep the last N lines/bytes).
///
/// Suitable for bash output where you want to see the end (errors, final
/// results). May return a partial first line when the last line of the
/// original content exceeds the byte limit.
pub fn truncate_tail(content: &str, options: &TruncationOptions) -> TruncationResult {
    let (max_lines, max_bytes) = options.resolved();
    let lines = split_lines_for_counting(content);
    let total_lines = lines.len();
    let total_bytes = content.len();

    if total_lines <= max_lines && total_bytes <= max_bytes {
        return not_truncated(content, total_lines, max_lines, max_bytes);
    }

    // Work backwards from the end.
    let mut output_lines: Vec<&str> = Vec::new();
    let mut output_bytes_count = 0usize;
    let mut truncated_by = TruncatedBy::Lines;
    let mut last_line_partial = false;

    for line in lines.iter().rev() {
        if output_lines.len() >= max_lines {
            break;
        }
        let line_bytes = line.len() + usize::from(!output_lines.is_empty()); // +1 for the newline
        if output_bytes_count + line_bytes > max_bytes {
            truncated_by = TruncatedBy::Bytes;
            // Edge case: no lines added yet and this line alone exceeds the
            // limit — keep the end of the line (partial).
            if output_lines.is_empty() {
                let truncated_line = truncate_string_to_bytes_from_end(line, max_bytes);
                output_bytes_count = truncated_line.len();
                output_lines.push(truncated_line);
                last_line_partial = true;
            }
            break;
        }
        output_lines.push(line);
        output_bytes_count += line_bytes;
    }

    // If we exited due to the line limit.
    if output_lines.len() >= max_lines && output_bytes_count <= max_bytes {
        truncated_by = TruncatedBy::Lines;
    }

    output_lines.reverse();
    let output_content = output_lines.join("\n");
    TruncationResult {
        output_bytes: output_content.len(),
        content: output_content,
        truncated: true,
        truncated_by: Some(truncated_by),
        total_lines,
        total_bytes,
        output_lines: output_lines.len(),
        last_line_partial,
        first_line_exceeds_limit: false,
        max_lines,
        max_bytes,
    }
}

/// Truncate a string to fit within a byte limit, keeping the end.
///
/// The cut lands on a UTF-8 character boundary, so no surrogate repair is
/// needed (unlike the UTF-16 TypeScript original).
fn truncate_string_to_bytes_from_end(text: &str, max_bytes: usize) -> &str {
    if max_bytes == 0 {
        return "";
    }
    if text.len() <= max_bytes {
        return text;
    }
    let mut start = text.len();
    let mut output_bytes = 0usize;
    for (index, character) in text.char_indices().rev() {
        let character_bytes = character.len_utf8();
        if output_bytes + character_bytes > max_bytes {
            break;
        }
        output_bytes += character_bytes;
        start = index;
    }
    &text[start..]
}

/// Truncate a single line to a maximum number of characters, adding a
/// `... [truncated]` suffix. Used for grep match lines.
pub fn truncate_line(line: &str, max_chars: usize) -> TruncatedLine {
    let char_count = line.chars().count();
    if char_count <= max_chars {
        return TruncatedLine {
            text: line.to_owned(),
            was_truncated: false,
        };
    }
    let kept: String = line.chars().take(max_chars).collect();
    TruncatedLine {
        text: format!("{kept}... [truncated]"),
        was_truncated: true,
    }
}

/// Result of [`truncate_line`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TruncatedLine {
    pub text: String,
    pub was_truncated: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_truncation_when_within_limits() {
        let result = truncate_head("hello\nworld", &TruncationOptions::default());
        assert!(!result.truncated);
        assert_eq!(result.truncated_by, None);
        assert_eq!(result.total_lines, 2);
        assert_eq!(result.content, "hello\nworld");
    }

    #[test]
    fn trailing_newline_is_not_an_extra_line() {
        let result = truncate_head("a\nb\n", &TruncationOptions::default());
        assert_eq!(result.total_lines, 2);
    }

    #[test]
    fn head_truncates_by_lines() {
        let content: String = (0..2500).map(|i| format!("line{i}\n")).collect();
        let result = truncate_head(&content, &TruncationOptions::default());
        assert!(result.truncated);
        assert_eq!(result.truncated_by, Some(TruncatedBy::Lines));
        assert_eq!(result.output_lines, 2000);
        assert_eq!(result.total_lines, 2500);
    }

    #[test]
    fn head_truncates_by_bytes() {
        // Many small lines: the byte limit is hit before the line limit.
        let content: String = (0..2000)
            .map(|i| format!("{i}{}\n", "x".repeat(50)))
            .collect();
        assert!(content.len() > DEFAULT_MAX_BYTES);
        let result = truncate_head(&content, &TruncationOptions::default());
        assert!(result.truncated);
        assert_eq!(result.truncated_by, Some(TruncatedBy::Bytes));
        assert!(result.output_bytes <= DEFAULT_MAX_BYTES);
        assert!(result.output_lines < result.total_lines);
    }

    #[test]
    fn head_first_line_exceeds_limit() {
        let content = format!("{}\nshort", "y".repeat(60000));
        let result = truncate_head(&content, &TruncationOptions::default());
        assert!(result.first_line_exceeds_limit);
        assert!(result.content.is_empty());
    }

    #[test]
    fn tail_keeps_the_end() {
        let content: String = (0..2500).map(|i| format!("line{i}\n")).collect();
        let result = truncate_tail(&content, &TruncationOptions::default());
        assert!(result.truncated);
        assert_eq!(result.truncated_by, Some(TruncatedBy::Lines));
        assert_eq!(result.output_lines, 2000);
        assert!(result.content.starts_with("line500"));
        assert!(result.content.ends_with("line2499"));
    }

    #[test]
    fn tail_partial_last_line_when_it_alone_exceeds_the_limit() {
        let content = "z".repeat(60000);
        let result = truncate_tail(&content, &TruncationOptions::default());
        assert!(result.truncated);
        assert!(result.last_line_partial);
        assert_eq!(result.output_lines, 1);
        assert!(result.output_bytes <= DEFAULT_MAX_BYTES);
    }

    #[test]
    fn tail_truncation_lands_on_char_boundaries() {
        // Multi-byte characters must never be split.
        let content = "日本語".repeat(30000);
        let result = truncate_tail(&content, &TruncationOptions::default());
        assert!(result.truncated);
        assert!(
            result
                .content
                .chars()
                .all(|c| c == '日' || c == '本' || c == '語')
        );
        assert!(result.output_bytes <= DEFAULT_MAX_BYTES);
    }

    #[test]
    fn format_size_units() {
        assert_eq!(format_size(512), "512B");
        assert_eq!(format_size(2048), "2.0KB");
        assert_eq!(format_size(1024 * 1024 * 3), "3.0MB");
    }

    #[test]
    fn truncate_line_by_characters() {
        let short = truncate_line("abc", 10);
        assert!(!short.was_truncated);
        let long = truncate_line(&"a".repeat(600), GREP_MAX_LINE_LENGTH);
        assert!(long.was_truncated);
        assert!(long.text.ends_with("... [truncated]"));
    }
}
