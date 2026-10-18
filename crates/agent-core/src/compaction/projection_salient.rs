//! Head + tail + salient-line text cutting used by the projection rules.
//!
//! A cut keeps the head and tail of an oversized text plus a bounded number of
//! "salient" middle lines (errors, test output, paths, diff hunks) so the model
//! retains the actionable parts of the omitted region.
//!
//! Ported from `@step-harness/agent-core`
//! `src/harness/compaction/projection-salient.ts`.

use std::collections::HashSet;
use std::sync::LazyLock;

use regex::Regex;

use super::projection_content::{PROJECTION_CUT_MARKER_PREFIX, char_len, normalize_for_hash};

/// Salient-line pattern: error/test/path/diff keywords plus stack-frame shapes
/// (`File "x.py"`, Rust `-->`, and `path.ext:12` / `path.ext(12` references).
///
/// The TypeScript source uses the `/i` flag; here that is the `(?i)` inline
/// prefix. Character counting is by Unicode scalar value (see
/// [`super::projection_content::char_len`]).
static SALIENT_LINE_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)error|fail|test|exit|path|diff|warning|traceback|File "|-->|\S+\.\w+[:(]\d+"#)
        .expect("valid salient line regex")
});

/// Additional salient shapes for code/patch/JSON payloads (large-code rule).
static CODE_SALIENT_LINE_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?i)^diff --git |^@@ |^[+-]{3} |^Index: |^\s*at |SyntaxError|ParseError|Unexpected token|error TS\d+|expected|unterminated|unclosed"#,
    )
    .expect("valid code salient line regex")
});

/// Handle to the shared [`SALIENT_LINE_REGEX`], passed to [`cut_text_with_salient_lines`].
pub(crate) fn salient_line_regex() -> &'static Regex {
    &SALIENT_LINE_REGEX
}

/// Handle to the shared [`CODE_SALIENT_LINE_REGEX`].
pub(crate) fn code_salient_line_regex() -> &'static Regex {
    &CODE_SALIENT_LINE_REGEX
}

/// Minimum characters a cut must save before it is applied.
const MIN_CUT_GAIN_CHARS: usize = 500;
/// Maximum characters for a single preserved salient line.
const MAX_SALIENT_LINE_CHARS: usize = 400;

/// Outcome of one head+tail+salient cut.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CutResult {
    pub text: String,
    pub salient_lines: usize,
}

fn take_head_lines(lines: &[&str], budget: usize) -> Vec<String> {
    let mut collected: Vec<String> = Vec::new();
    let mut used_chars = 0usize;
    for line in lines {
        let line_len = char_len(line);
        if collected.is_empty() && line_len > budget {
            let sliced: String = line.chars().take(budget).collect();
            collected.push(format!("{sliced}…"));
            return collected;
        }
        if used_chars + line_len + 1 > budget && !collected.is_empty() {
            break;
        }
        collected.push((*line).to_owned());
        used_chars += line_len + 1;
        if used_chars >= budget {
            break;
        }
    }
    collected
}

fn take_tail_lines(lines: &[&str], budget: usize) -> Vec<String> {
    let mut collected: Vec<String> = Vec::new();
    let mut used_chars = 0usize;
    for line in lines.iter().rev() {
        let line_len = char_len(line);
        if collected.is_empty() && line_len > budget {
            let sliced: String = line.chars().skip(line_len - budget).collect();
            collected.insert(0, format!("…{sliced}"));
            return collected;
        }
        if used_chars + line_len + 1 > budget && !collected.is_empty() {
            break;
        }
        collected.insert(0, (*line).to_owned());
        used_chars += line_len + 1;
        if used_chars >= budget {
            break;
        }
    }
    collected
}

/// Collect up to `max_salient_lines` deduplicated salient lines from the omitted middle.
fn collect_salient_middle_lines(
    lines: &[&str],
    head_count: usize,
    tail_count: usize,
    max_salient_lines: usize,
    salient_regex: &Regex,
) -> Vec<String> {
    let mut salient_lines: Vec<String> = Vec::new();
    let mut seen_line_keys: HashSet<String> = HashSet::new();
    let end = lines.len() - tail_count;
    let mut i = head_count;
    while i < end && salient_lines.len() < max_salient_lines {
        let line = lines[i];
        let trimmed = line.trim();
        if trimmed.is_empty() {
            i += 1;
            continue;
        }
        // A line is salient when it matches the caller's regex or the shared one.
        if !salient_regex.is_match(line) && !SALIENT_LINE_REGEX.is_match(line) {
            i += 1;
            continue;
        }
        let line_key = normalize_for_hash(trimmed);
        if !seen_line_keys.insert(line_key) {
            i += 1;
            continue;
        }
        if char_len(trimmed) > MAX_SALIENT_LINE_CHARS {
            let sliced: String = trimmed.chars().take(MAX_SALIENT_LINE_CHARS).collect();
            salient_lines.push(format!("{sliced}…"));
        } else {
            salient_lines.push(trimmed.to_owned());
        }
        i += 1;
    }
    salient_lines
}

/// Cut `text` to head + tail plus up to `max_salient_lines` deduplicated salient
/// lines from the omitted middle. Returns `None` when cutting would not save at
/// least `MIN_CUT_GAIN_CHARS`.
pub fn cut_text_with_salient_lines(
    text: &str,
    head_chars: usize,
    tail_chars: usize,
    max_salient_lines: usize,
    salient_regex: &Regex,
) -> Option<CutResult> {
    if char_len(text) <= head_chars + tail_chars + MIN_CUT_GAIN_CHARS {
        return None;
    }

    let lines: Vec<&str> = text.split('\n').collect();
    let head_lines = take_head_lines(&lines, head_chars);
    let tail_lines = take_tail_lines(&lines, tail_chars);
    if head_lines.len() + tail_lines.len() >= lines.len() {
        return None;
    }

    let salient_lines = collect_salient_middle_lines(
        &lines,
        head_lines.len(),
        tail_lines.len(),
        max_salient_lines,
        salient_regex,
    );

    let head_text = head_lines.join("\n");
    let tail_text = tail_lines.join("\n");
    let salient_text = salient_lines.join("\n");
    let retained_chars = char_len(&head_text) + char_len(&salient_text) + char_len(&tail_text);
    let omitted_chars = char_len(text).saturating_sub(retained_chars);
    let marker = format!(
        "{PROJECTION_CUT_MARKER_PREFIX} original_chars={} retained_chars={retained_chars} omitted_chars={omitted_chars}; salient_lines={}]",
        char_len(text),
        salient_lines.len()
    );

    let mut cut_sections: Vec<String> = vec![head_text, marker];
    if !salient_lines.is_empty() {
        cut_sections.push(salient_text);
        cut_sections.push("[...]".to_owned());
    }
    cut_sections.push(tail_text);
    let cut_text = cut_sections.join("\n");

    if char_len(&cut_text) >= char_len(text).saturating_sub(MIN_CUT_GAIN_CHARS) {
        return None;
    }
    Some(CutResult {
        text: cut_text,
        salient_lines: salient_lines.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_is_not_cut() {
        // Below head+tail+MIN_CUT_GAIN_CHARS -> no cut.
        let text = "x".repeat(100);
        assert!(cut_text_with_salient_lines(&text, 800, 800, 20, salient_line_regex()).is_none());
    }

    #[test]
    fn large_text_is_cut_and_keeps_marker() {
        let mut lines: Vec<String> = (0..200)
            .map(|i| format!("filler line number {i}"))
            .collect();
        // sprinkle salient lines in the middle
        lines.insert(100, "ERROR: something failed".to_owned());
        lines.insert(120, "test_case_42 failed".to_owned());
        let text = lines.join("\n");
        let cut = cut_text_with_salient_lines(&text, 200, 200, 20, salient_line_regex())
            .expect("should cut");
        assert!(cut.text.contains(PROJECTION_CUT_MARKER_PREFIX));
        assert!(cut.salient_lines >= 1);
        // The cut must actually save characters.
        assert!(char_len(&cut.text) < char_len(&text));
        // Head and tail content retained.
        assert!(cut.text.contains("filler line number 0"));
    }

    #[test]
    fn salient_lines_are_capped() {
        let mut lines: Vec<String> = Vec::new();
        for i in 0..100 {
            lines.push(format!("error number {i}"));
        }
        let text = lines.join("\n");
        let cut = cut_text_with_salient_lines(&text, 100, 100, 3, salient_line_regex()).unwrap();
        assert!(cut.salient_lines <= 3);
    }

    #[test]
    fn head_and_tail_lines_respect_budget() {
        // Lines longer than the budget trigger the first-line/last-line truncation.
        let lines: Vec<String> = (0..50).map(|_| "a".repeat(200)).collect();
        let head = take_head_lines(&lines.iter().map(|s| s.as_str()).collect::<Vec<_>>(), 120);
        // First line alone exceeds budget -> truncated to budget chars + ellipsis.
        assert_eq!(head.len(), 1);
        assert_eq!(char_len(&head[0]), 121); // 120 + "…"
    }

    #[test]
    fn code_salient_regex_matches_diff_shapes() {
        assert!(code_salient_line_regex().is_match("diff --git a/x b/x"));
        assert!(code_salient_line_regex().is_match("@@ -1,2 +3,4 @@"));
        assert!(code_salient_line_regex().is_match("SyntaxError: bad"));
        assert!(!code_salient_line_regex().is_match("just some text"));
    }
}
