//! Shared diff computation utilities for the edit tool.
//!
//! Ported from `@step-harness/agent-core` `src\harness\tools\edit-diff.ts`.
//! The TypeScript source uses the `diff` (jsdiff) package; Rust uses the
//! `similar` crate, which implements the same Myers line diff.

use similar::TextDiff;
use unicode_normalization::UnicodeNormalization;

/// Line ending style of a file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineEnding {
    CrLf,
    Lf,
}

impl LineEnding {
    pub fn as_str(&self) -> &'static str {
        match self {
            LineEnding::CrLf => "\r\n",
            LineEnding::Lf => "\n",
        }
    }
}

/// Detect the dominant line ending: whichever of CRLF/LF appears first.
pub fn detect_line_ending(content: &str) -> LineEnding {
    let crlf_index = content.find("\r\n");
    let lf_index = content.find('\n');
    match (crlf_index, lf_index) {
        (Some(crlf), Some(lf)) if crlf < lf => LineEnding::CrLf,
        _ => LineEnding::Lf,
    }
}

/// Normalize all line endings to LF.
pub fn normalize_to_lf(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

/// Restore the requested line ending.
pub fn restore_line_endings(text: &str, ending: LineEnding) -> String {
    match ending {
        LineEnding::CrLf => text.replace('\n', "\r\n"),
        LineEnding::Lf => text.to_owned(),
    }
}

/// Normalize text for fuzzy matching:
/// NFKC, trailing whitespace stripped per line, smart quotes and dashes
/// folded to ASCII, and special Unicode spaces folded to regular spaces.
pub fn normalize_for_fuzzy_match(text: &str) -> String {
    let nfkc: String = text.nfkc().collect();
    let trimmed = nfkc
        .split('\n')
        .map(|line| line.trim_end())
        .collect::<Vec<_>>()
        .join("\n");
    trimmed
        .replace(['\u{2018}', '\u{2019}', '\u{201A}', '\u{201B}'], "'")
        .replace(['\u{201C}', '\u{201D}', '\u{201E}', '\u{201F}'], "\"")
        .replace(
            [
                '\u{2010}', '\u{2011}', '\u{2012}', '\u{2013}', '\u{2014}', '\u{2015}', '\u{2212}',
            ],
            "-",
        )
        .replace(
            [
                '\u{00A0}', '\u{2002}', '\u{2003}', '\u{2004}', '\u{2005}', '\u{2006}', '\u{2007}',
                '\u{2008}', '\u{2009}', '\u{200A}', '\u{202F}', '\u{205F}', '\u{3000}',
            ],
            " ",
        )
}

/// Split content into lines, keeping each line's trailing newline.
fn split_lines_with_endings(content: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    let bytes = content.as_bytes();
    for (index, byte) in bytes.iter().enumerate() {
        if *byte == b'\n' {
            lines.push(&content[start..=index]);
            start = index + 1;
        }
    }
    if start < content.len() {
        lines.push(&content[start..]);
    }
    lines
}

struct LineSpan {
    start: usize,
    end: usize,
}

fn get_line_spans(content: &str) -> Vec<LineSpan> {
    let mut offset = 0;
    split_lines_with_endings(content)
        .into_iter()
        .map(|line| {
            let span = LineSpan {
                start: offset,
                end: offset + line.len(),
            };
            offset = span.end;
            span
        })
        .collect()
}

#[derive(Clone, Debug)]
struct TextReplacement {
    match_index: usize,
    match_length: usize,
    new_text: String,
}

fn get_replacement_line_range(
    lines: &[LineSpan],
    replacement: &TextReplacement,
) -> Result<(usize, usize), String> {
    let replacement_start = replacement.match_index;
    let replacement_end = replacement.match_index + replacement.match_length;

    let Some(start_line) = lines
        .iter()
        .position(|line| replacement_start >= line.start && replacement_start < line.end)
    else {
        return Err("Replacement range is outside the base content.".to_owned());
    };

    let mut end_line = start_line;
    while end_line < lines.len() && lines[end_line].end < replacement_end {
        end_line += 1;
    }
    if end_line >= lines.len() {
        return Err("Replacement range is outside the base content.".to_owned());
    }

    Ok((start_line, end_line + 1))
}

fn apply_replacements(content: &str, replacements: &[TextReplacement], offset: usize) -> String {
    let mut result = content.to_owned();
    for replacement in replacements.iter().rev() {
        let match_index = replacement.match_index - offset;
        let end = match_index + replacement.match_length;
        result.replace_range(match_index..end, &replacement.new_text);
    }
    result
}

/// Apply replacements matched against `base_content` to `original_content`
/// while preserving unchanged line blocks from the original.
///
/// Each replacement is widened to the lines it touches, those lines are
/// rewritten from the normalized base, and all other lines are copied back from
/// the original. The replacement ranges drive preservation so duplicate
/// normalized lines cannot be aligned to the wrong occurrence.
fn apply_replacements_preserving_unchanged_lines(
    original_content: &str,
    base_content: &str,
    replacements: &[TextReplacement],
) -> Result<String, String> {
    let original_lines = split_lines_with_endings(original_content);
    let base_lines = get_line_spans(base_content);
    if original_lines.len() != base_lines.len() {
        return Err(
            "Cannot preserve unchanged lines because the base content has a different line count."
                .to_owned(),
        );
    }

    struct Group {
        start_line: usize,
        end_line: usize,
        replacements: Vec<TextReplacement>,
    }

    let mut sorted: Vec<&TextReplacement> = replacements.iter().collect();
    sorted.sort_by_key(|replacement| replacement.match_index);

    let mut groups: Vec<Group> = Vec::new();
    for replacement in sorted {
        let (start_line, end_line) = get_replacement_line_range(&base_lines, replacement)?;
        if let Some(current) = groups.last_mut()
            && start_line < current.end_line
        {
            current.end_line = current.end_line.max(end_line);
            current.replacements.push(replacement.clone());
            continue;
        }
        groups.push(Group {
            start_line,
            end_line,
            replacements: vec![replacement.clone()],
        });
    }

    let mut original_line_index = 0;
    let mut result = String::new();
    for group in groups {
        result.push_str(&original_lines[original_line_index..group.start_line].concat());
        let group_start_offset = base_lines[group.start_line].start;
        let group_end_offset = base_lines[group.end_line - 1].end;
        result.push_str(&apply_replacements(
            &base_content[group_start_offset..group_end_offset],
            &group.replacements,
            group_start_offset,
        ));
        original_line_index = group.end_line;
    }
    result.push_str(&original_lines[original_line_index..].concat());

    Ok(result)
}

/// Result of a fuzzy text search.
#[derive(Clone, Debug)]
pub struct FuzzyMatchResult {
    pub found: bool,
    /// Byte index where the match starts, in `content_for_replacement`.
    pub index: usize,
    /// Byte length of the matched text.
    pub match_length: usize,
    /// Whether fuzzy matching was used (`false` = exact match).
    pub used_fuzzy_match: bool,
    /// The content to use for replacement operations.
    pub content_for_replacement: String,
}

/// A single targeted replacement.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Edit {
    pub old_text: String,
    pub new_text: String,
}

/// Find `old_text` in `content`, trying exact match first, then fuzzy match.
///
/// When fuzzy matching is used, the returned offsets are in normalized space
/// and `content_for_replacement` is the fuzzy-normalized content.
pub fn fuzzy_find_text(content: &str, old_text: &str) -> FuzzyMatchResult {
    if let Some(index) = content.find(old_text) {
        return FuzzyMatchResult {
            found: true,
            index,
            match_length: old_text.len(),
            used_fuzzy_match: false,
            content_for_replacement: content.to_owned(),
        };
    }

    let fuzzy_content = normalize_for_fuzzy_match(content);
    let fuzzy_old_text = normalize_for_fuzzy_match(old_text);
    match fuzzy_content.find(&fuzzy_old_text) {
        Some(index) => FuzzyMatchResult {
            found: true,
            index,
            match_length: fuzzy_old_text.len(),
            used_fuzzy_match: true,
            content_for_replacement: fuzzy_content,
        },
        None => FuzzyMatchResult {
            found: false,
            index: 0,
            match_length: 0,
            used_fuzzy_match: false,
            content_for_replacement: content.to_owned(),
        },
    }
}

/// Strip a UTF-8 BOM if present, returning both the BOM and the remaining text.
pub fn strip_bom(content: &str) -> (&'static str, &str) {
    match content.strip_prefix('\u{FEFF}') {
        Some(text) => ("\u{FEFF}", text),
        None => ("", content),
    }
}

fn count_occurrences(content: &str, old_text: &str) -> usize {
    let fuzzy_content = normalize_for_fuzzy_match(content);
    let fuzzy_old_text = normalize_for_fuzzy_match(old_text);
    fuzzy_content.matches(&fuzzy_old_text).count()
}

fn not_found_error(path: &str, edit_index: usize, total_edits: usize) -> String {
    if total_edits == 1 {
        format!(
            "Could not find the exact text in {path}. The old text must match exactly including all whitespace and newlines."
        )
    } else {
        format!(
            "Could not find edits[{edit_index}] in {path}. The oldText must match exactly including all whitespace and newlines."
        )
    }
}

fn duplicate_error(
    path: &str,
    edit_index: usize,
    total_edits: usize,
    occurrences: usize,
) -> String {
    if total_edits == 1 {
        format!(
            "Found {occurrences} occurrences of the text in {path}. The text must be unique. Please provide more context to make it unique."
        )
    } else {
        format!(
            "Found {occurrences} occurrences of edits[{edit_index}] in {path}. Each oldText must be unique. Please provide more context to make it unique."
        )
    }
}

fn empty_old_text_error(path: &str, edit_index: usize, total_edits: usize) -> String {
    if total_edits == 1 {
        format!("oldText must not be empty in {path}.")
    } else {
        format!("edits[{edit_index}].oldText must not be empty in {path}.")
    }
}

fn no_change_error(path: &str, total_edits: usize) -> String {
    if total_edits == 1 {
        format!(
            "No changes made to {path}. The replacement produced identical content. This might indicate an issue with special characters or the text not existing as expected."
        )
    } else {
        format!("No changes made to {path}. The replacements produced identical content.")
    }
}

/// Result of applying edits.
#[derive(Clone, Debug)]
pub struct AppliedEditsResult {
    pub base_content: String,
    pub new_content: String,
}

/// Apply one or more exact-text replacements to LF-normalized content.
///
/// All edits are matched against the same original content. Replacements are
/// applied in reverse order so offsets remain stable. If any edit needs fuzzy
/// matching, the operation runs in fuzzy-normalized content space and then
/// overlays those line-level changes onto the original content so unchanged
/// line blocks keep their original bytes.
pub fn apply_edits_to_normalized_content(
    normalized_content: &str,
    edits: &[Edit],
    path: &str,
) -> Result<AppliedEditsResult, String> {
    let normalized_edits: Vec<(String, String)> = edits
        .iter()
        .map(|edit| {
            (
                normalize_to_lf(&edit.old_text),
                normalize_to_lf(&edit.new_text),
            )
        })
        .collect();

    for (index, (old_text, _)) in normalized_edits.iter().enumerate() {
        if old_text.is_empty() {
            return Err(empty_old_text_error(path, index, normalized_edits.len()));
        }
    }

    let initial_matches: Vec<FuzzyMatchResult> = normalized_edits
        .iter()
        .map(|(old_text, _)| fuzzy_find_text(normalized_content, old_text))
        .collect();
    let used_fuzzy_match = initial_matches.iter().any(|m| m.used_fuzzy_match);
    let replacement_base_content = if used_fuzzy_match {
        normalize_for_fuzzy_match(normalized_content)
    } else {
        normalized_content.to_owned()
    };

    struct MatchedEdit {
        edit_index: usize,
        replacement: TextReplacement,
    }

    let mut matched_edits: Vec<MatchedEdit> = Vec::new();
    for (index, (old_text, new_text)) in normalized_edits.iter().enumerate() {
        let match_result = fuzzy_find_text(&replacement_base_content, old_text);
        if !match_result.found {
            return Err(not_found_error(path, index, normalized_edits.len()));
        }
        let occurrences = count_occurrences(&replacement_base_content, old_text);
        if occurrences > 1 {
            return Err(duplicate_error(
                path,
                index,
                normalized_edits.len(),
                occurrences,
            ));
        }
        matched_edits.push(MatchedEdit {
            edit_index: index,
            replacement: TextReplacement {
                match_index: match_result.index,
                match_length: match_result.match_length,
                new_text: new_text.clone(),
            },
        });
    }

    matched_edits.sort_by_key(|edit| edit.replacement.match_index);
    for index in 1..matched_edits.len() {
        let previous = &matched_edits[index - 1];
        let current = &matched_edits[index];
        if previous.replacement.match_index + previous.replacement.match_length
            > current.replacement.match_index
        {
            return Err(format!(
                "edits[{}] and edits[{}] overlap in {path}. Merge them into one edit or target disjoint regions.",
                previous.edit_index, current.edit_index
            ));
        }
    }

    let base_content = normalized_content.to_owned();
    let replacements: Vec<TextReplacement> = matched_edits
        .into_iter()
        .map(|edit| edit.replacement)
        .collect();
    let new_content = if used_fuzzy_match {
        apply_replacements_preserving_unchanged_lines(
            normalized_content,
            &replacement_base_content,
            &replacements,
        )?
    } else {
        apply_replacements(&replacement_base_content, &replacements, 0)
    };

    if base_content == new_content {
        return Err(no_change_error(path, normalized_edits.len()));
    }

    Ok(AppliedEditsResult {
        base_content,
        new_content,
    })
}

/// Generate a standard unified patch with file headers only.
pub fn generate_unified_patch(
    path: &str,
    old_content: &str,
    new_content: &str,
    context_lines: usize,
) -> String {
    let diff = TextDiff::from_lines(old_content, new_content);
    diff.unified_diff()
        .context_radius(context_lines)
        .header(path, path)
        .to_string()
}

/// A display-oriented diff with line numbers and context elision.
#[derive(Clone, Debug)]
pub struct DiffStringResult {
    pub diff: String,
    /// First changed line number in the new file, when there is a change.
    pub first_changed_line: Option<usize>,
}

/// Generate a display-oriented diff string with line numbers and context.
pub fn generate_diff_string(
    old_content: &str,
    new_content: &str,
    context_lines: usize,
) -> DiffStringResult {
    // Group consecutive changes by tag, mirroring jsdiff's `diffLines` parts.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Tag {
        Added,
        Removed,
        Equal,
    }

    let text_diff = TextDiff::from_lines(old_content, new_content);
    let mut groups: Vec<(Tag, String)> = Vec::new();
    for change in text_diff.iter_all_changes() {
        let tag = match change.tag() {
            similar::ChangeTag::Delete => Tag::Removed,
            similar::ChangeTag::Insert => Tag::Added,
            similar::ChangeTag::Equal => Tag::Equal,
        };
        match groups.last_mut() {
            Some((last_tag, value)) if *last_tag == tag => value.push_str(change.value()),
            _ => groups.push((tag, change.value().to_owned())),
        }
    }

    let old_lines: Vec<&str> = old_content.split('\n').collect();
    let new_lines: Vec<&str> = new_content.split('\n').collect();
    let max_line_num = old_lines.len().max(new_lines.len());
    let line_num_width = max_line_num.to_string().len();

    let mut output: Vec<String> = Vec::new();
    let mut old_line_num = 1usize;
    let mut new_line_num = 1usize;
    let mut last_was_change = false;
    let mut first_changed_line: Option<usize> = None;

    let pad = |number: usize| -> String { format!("{number:>width$}", width = line_num_width) };

    for index in 0..groups.len() {
        let (tag, value) = &groups[index];
        let mut raw: Vec<&str> = value.split('\n').collect();
        if raw.last() == Some(&"") {
            raw.pop();
        }

        let is_change = *tag != Tag::Equal;
        if is_change {
            if first_changed_line.is_none() {
                first_changed_line = Some(new_line_num);
            }
            for line in raw {
                if *tag == Tag::Added {
                    output.push(format!("+{} {line}", pad(new_line_num)));
                    new_line_num += 1;
                } else {
                    output.push(format!("-{} {line}", pad(old_line_num)));
                    old_line_num += 1;
                }
            }
            last_was_change = true;
        } else {
            let next_part_is_change = index + 1 < groups.len() && groups[index + 1].0 != Tag::Equal;
            let has_leading_change = last_was_change;
            let has_trailing_change = next_part_is_change;
            let marker = format!("{} ...", pad(0));

            if has_leading_change && has_trailing_change {
                if raw.len() <= context_lines * 2 {
                    for line in &raw {
                        output.push(format!(" {} {line}", pad(old_line_num)));
                        old_line_num += 1;
                        new_line_num += 1;
                    }
                } else {
                    for line in &raw[..context_lines] {
                        output.push(format!(" {} {line}", pad(old_line_num)));
                        old_line_num += 1;
                        new_line_num += 1;
                    }
                    let skipped = raw.len() - context_lines * 2;
                    output.push(marker);
                    old_line_num += skipped;
                    new_line_num += skipped;
                    for line in &raw[raw.len() - context_lines..] {
                        output.push(format!(" {} {line}", pad(old_line_num)));
                        old_line_num += 1;
                        new_line_num += 1;
                    }
                }
            } else if has_leading_change {
                let shown = raw.len().min(context_lines);
                for line in &raw[..shown] {
                    output.push(format!(" {} {line}", pad(old_line_num)));
                    old_line_num += 1;
                    new_line_num += 1;
                }
                let skipped = raw.len() - shown;
                if skipped > 0 {
                    output.push(marker);
                    old_line_num += skipped;
                    new_line_num += skipped;
                }
            } else if has_trailing_change {
                let skipped = raw.len().saturating_sub(context_lines);
                if skipped > 0 {
                    output.push(marker);
                    old_line_num += skipped;
                    new_line_num += skipped;
                }
                for line in &raw[skipped..] {
                    output.push(format!(" {} {line}", pad(old_line_num)));
                    old_line_num += 1;
                    new_line_num += 1;
                }
            } else {
                old_line_num += raw.len();
                new_line_num += raw.len();
            }

            last_was_change = false;
        }
    }

    DiffStringResult {
        diff: output.join("\n"),
        first_changed_line,
    }
}

use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_line_endings() {
        assert_eq!(detect_line_ending("a\r\nb\nc"), LineEnding::CrLf);
        assert_eq!(detect_line_ending("a\nb\r\nc"), LineEnding::Lf);
        assert_eq!(detect_line_ending("a\nb"), LineEnding::Lf);
        assert_eq!(detect_line_ending("abc"), LineEnding::Lf);
    }

    #[test]
    fn round_trips_line_endings() {
        let text = "a\nb\nc";
        assert_eq!(restore_line_endings(text, LineEnding::CrLf), "a\r\nb\r\nc");
        assert_eq!(normalize_to_lf("a\r\nb\rc\nd"), "a\nb\nc\nd");
    }

    #[test]
    fn strips_bom() {
        let (bom, text) = strip_bom("\u{FEFF}hello");
        assert_eq!(bom, "\u{FEFF}");
        assert_eq!(text, "hello");
        let (bom, text) = strip_bom("hello");
        assert_eq!(bom, "");
        assert_eq!(text, "hello");
    }

    #[test]
    fn fuzzy_match_normalizes_quotes_and_dashes() {
        let content = "let x = \u{201C}smart\u{201D} \u{2013} dash";
        let result = fuzzy_find_text(content, "let x = \"smart\" - dash");
        assert!(result.found);
        assert!(result.used_fuzzy_match);
    }

    #[test]
    fn exact_match_preferred() {
        let result = fuzzy_find_text("hello world", "world");
        assert!(result.found);
        assert!(!result.used_fuzzy_match);
        assert_eq!(result.index, 6);
    }

    #[test]
    fn applies_single_edit() {
        let result =
            apply_edits_to_normalized_content("hello world", &[edit("world", "rust")], "f.txt")
                .unwrap();
        assert_eq!(result.new_content, "hello rust");
    }

    #[test]
    fn applies_multiple_edits_in_reverse_order() {
        let result = apply_edits_to_normalized_content(
            "aaa bbb ccc",
            &[edit("aaa", "AAA"), edit("ccc", "CCC")],
            "f.txt",
        )
        .unwrap();
        assert_eq!(result.new_content, "AAA bbb CCC");
    }

    #[test]
    fn rejects_empty_old_text() {
        let error =
            apply_edits_to_normalized_content("abc", &[edit("", "x")], "f.txt").unwrap_err();
        assert!(error.contains("must not be empty"));
    }

    #[test]
    fn rejects_missing_text() {
        let error =
            apply_edits_to_normalized_content("abc", &[edit("zzz", "x")], "f.txt").unwrap_err();
        assert!(error.contains("Could not find the exact text"));
    }

    #[test]
    fn rejects_duplicate_text() {
        let error =
            apply_edits_to_normalized_content("abc abc", &[edit("abc", "x")], "f.txt").unwrap_err();
        assert!(error.contains("Found 2 occurrences"));
    }

    #[test]
    fn rejects_overlapping_edits() {
        let error = apply_edits_to_normalized_content(
            "hello world",
            &[edit("hello world", "x"), edit("world", "y")],
            "f.txt",
        )
        .unwrap_err();
        assert!(error.contains("overlap"));
    }

    #[test]
    fn rejects_no_op_edit() {
        let error =
            apply_edits_to_normalized_content("abc", &[edit("abc", "abc")], "f.txt").unwrap_err();
        assert!(error.contains("No changes made"));
    }

    #[test]
    fn fuzzy_edit_preserves_original_bytes_elsewhere() {
        // The trailing whitespace differs; the fuzzy match must still work and
        // untouched lines keep their original bytes.
        let original = "line one  \nline two\n";
        let result =
            apply_edits_to_normalized_content(original, &[edit("line two", "line 2")], "f.txt")
                .unwrap();
        assert_eq!(result.new_content, "line one  \nline 2\n");
    }

    #[test]
    fn unified_patch_has_file_headers_and_hunk() {
        let patch = generate_unified_patch("f.txt", "a\nb\nc\n", "a\nB\nc\n", 4);
        assert!(patch.contains("--- f.txt"), "patch was: {patch}");
        assert!(patch.contains("+++ f.txt"), "patch was: {patch}");
        assert!(patch.contains("@@"), "patch was: {patch}");
        assert!(patch.contains("-b"));
        assert!(patch.contains("+B"));
    }

    #[test]
    fn diff_string_has_line_numbers_and_first_change() {
        let result = generate_diff_string("a\nb\nc\n", "a\nB\nc\n", 4);
        assert_eq!(result.first_changed_line, Some(2));
        assert!(result.diff.contains("-2 b"));
        assert!(result.diff.contains("+2 B"));
        assert!(result.diff.contains(" 1 a"));
        assert!(result.diff.contains(" 3 c"));
    }

    fn edit(old_text: &str, new_text: &str) -> Edit {
        Edit {
            old_text: old_text.to_owned(),
            new_text: new_text.to_owned(),
        }
    }
}
