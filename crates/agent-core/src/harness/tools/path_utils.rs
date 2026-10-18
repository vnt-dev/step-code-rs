//! Tool path resolution helpers.
//!
//! Ported from `@step-harness/agent-core` `src\harness\tools\path-utils.ts`.

use unicode_normalization::UnicodeNormalization;

use crate::harness::types::{AbortSignal, ExecutionEnv, FileError};

/// Unicode spaces that models occasionally emit inside paths.
const UNICODE_SPACES: [char; 9] = [
    '\u{00A0}', '\u{2000}', '\u{2001}', '\u{2002}', '\u{2003}', '\u{2004}', '\u{2005}', '\u{2006}',
    '\u{2007}',
];
const NARROW_NO_BREAK_SPACE: char = '\u{202F}';

/// Normalize a model-provided path: Unicode spaces become regular spaces and a
/// leading `@` (a common model habit) is stripped.
fn normalize_tool_path(path: &str) -> String {
    let normalized: String = path
        .chars()
        .map(|character| {
            if UNICODE_SPACES.contains(&character) {
                ' '
            } else {
                character
            }
        })
        .collect();
    normalized
        .strip_prefix('@')
        .unwrap_or(&normalized)
        .to_owned()
}

/// Resolve a tool path against the environment's working directory.
pub async fn resolve_tool_path(
    env: &dyn ExecutionEnv,
    path: &str,
    signal: Option<&AbortSignal>,
) -> Result<String, FileError> {
    env.absolute_path(&normalize_tool_path(path), signal).await
}

/// Resolve a read path, retrying common filename variants.
///
/// Models often emit macOS screenshot names (`1. PM.png`), NFD-decomposed
/// filenames, or curly quotes; the first variant that exists wins. When none
/// exist, the plainly resolved path is returned so the read reports the
/// original not-found error.
pub async fn resolve_read_tool_path(
    env: &dyn ExecutionEnv,
    path: &str,
    signal: Option<&AbortSignal>,
) -> Result<String, FileError> {
    let resolved = resolve_tool_path(env, path, signal).await?;
    let mut variants = vec![resolved.clone()];

    // macOS screenshot style: "1. PM.png" uses a narrow no-break space.
    variants.push(replace_am_pm_spaces(&resolved));
    // HFS+ decomposed filenames.
    variants.push(resolved.nfd().collect());
    // Curly apostrophes.
    variants.push(resolved.replace('\'', "\u{2019}"));
    variants.push(resolved.nfd().collect::<String>().replace('\'', "\u{2019}"));

    let mut seen = std::collections::HashSet::new();
    for variant in variants {
        if !seen.insert(variant.clone()) {
            continue;
        }
        if env.exists(&variant, signal).await? {
            return Ok(variant);
        }
    }
    Ok(resolved)
}

/// Replace the space before `AM.`/`PM.` (case-insensitive) with a narrow
/// no-break space, matching the TypeScript regex `/ (AM|PM)\./gi`.
fn replace_am_pm_spaces(path: &str) -> String {
    let mut output = String::with_capacity(path.len());
    let mut index = 0;
    while index < path.len() {
        if path.as_bytes()[index] == b' ' {
            let suffix = path.as_bytes().get(index + 1..index + 4);
            if suffix.is_some_and(|suffix| {
                suffix.eq_ignore_ascii_case(b"am.") || suffix.eq_ignore_ascii_case(b"pm.")
            }) {
                output.push(NARROW_NO_BREAK_SPACE);
                index += 1;
                continue;
            }
        }
        let character = path[index..].chars().next().unwrap();
        output.push(character);
        index += character.len_utf8();
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn case_expanding_unicode_does_not_shift_byte_offsets() {
        assert_eq!(replace_am_pm_spaces("İ 1 PM.png"), "İ 1\u{202f}PM.png");
        assert_eq!(replace_am_pm_spaces("图 1 am.png"), "图 1\u{202f}am.png");
    }
}
