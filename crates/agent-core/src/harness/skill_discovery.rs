//! Gitignore-style ignore matching scoped to one directory in a resource tree.
//!
//! Ported from `@step-harness/agent-core` `src\harness\skill-discovery.ts`.
//! Paths are relative to the scan root, using forward slashes. Callers prune
//! ignored directories before reading child rules, as gitignore requires.
//!
//! The TypeScript source delegates rule matching to the `ignore` npm package
//! (backed by ripgrep). Here the Rust `ignore` crate is used directly:
//! `GitignoreBuilder::add_line` replaces `matcher.add`, and
//! `Gitignore::matched` replaces `matcher.test`, with the result's `ignored`
//! and `unignored` flags mapped onto [`Match::Ignore`] and
//! [`Match::Whitelist`]. The parent-chain delegation is implemented here.

use ignore::Match;
use ignore::gitignore::{Gitignore, GitignoreBuilder};
use std::path::Path;
use std::sync::Arc;

/// Ignore rules scoped to one directory in a resource tree.
///
/// Paths are relative to the scan root, using forward slashes. Callers prune
/// ignored directories before reading child rules, as gitignore requires.
///
/// Rules accumulate in insertion order across `add` calls, exactly like the
/// `ignore()` accumulator the TypeScript source wraps. Each matcher's rules
/// apply to paths relative to *its own* directory: [`SkillIgnoreMatcher::ignores`]
/// strips the directory prefix before testing, and delegates to the parent for
/// inherited rules. Parent matchers are shared through [`Arc`] so a whole
/// directory chain can be handed to [`SkillIgnoreMatcher::with_parent`].
pub struct SkillIgnoreMatcher {
    /// Re-anchored rule lines in insertion order.
    lines: std::sync::Mutex<Vec<String>>,
    /// Matcher rebuilt from `lines` whenever rules are added.
    matcher: std::sync::Mutex<Option<Gitignore>>,
    /// Directory prefix this matcher's rules are relative to, with a trailing
    /// slash, or the empty string for the scan root.
    prefix: String,
    /// Optional parent matcher providing inherited rules.
    parent: Option<Arc<SkillIgnoreMatcher>>,
}

impl SkillIgnoreMatcher {
    /// Create a root matcher with no parent.
    pub fn new(directory: impl Into<String>) -> Self {
        SkillIgnoreMatcher {
            lines: std::sync::Mutex::new(Vec::new()),
            matcher: std::sync::Mutex::new(None),
            prefix: directory_prefix(directory.into()),
            parent: None,
        }
    }

    /// Create a matcher inheriting rules from `parent`.
    pub fn with_parent(directory: impl Into<String>, parent: Arc<SkillIgnoreMatcher>) -> Self {
        SkillIgnoreMatcher {
            lines: std::sync::Mutex::new(Vec::new()),
            matcher: std::sync::Mutex::new(None),
            prefix: directory_prefix(directory.into()),
            parent: Some(parent),
        }
    }

    /// Add ignore-rule patterns from one ignore file's contents.
    ///
    /// Preserve whitespace and escaping for the `ignore` crate to interpret. Patterns are stored verbatim: like the
    /// `ignore` npm package the TypeScript source uses, they are matched against
    /// paths relative to *this* matcher's directory (see [`Self::ignores`]),
    /// not against paths relative to the scan root.
    pub fn add(&self, patterns: &str) {
        let mut new_lines: Vec<String> = Vec::new();
        for line in patterns.lines() {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            new_lines.push(line.to_owned());
        }
        if new_lines.is_empty() {
            return;
        }

        let mut lines = self.lines.lock().expect("skill ignore matcher poisoned");
        lines.extend(new_lines);
        let rebuilt = build_matcher(&lines);
        *self.matcher.lock().expect("skill ignore matcher poisoned") = Some(rebuilt);
    }

    /// True when `path` (relative to the scan root, forward slashes) is ignored.
    ///
    /// Parent rules are consulted first, mirroring the TypeScript delegation
    /// `this.parent?.ignores(path)`. The prefix is stripped so this matcher's
    /// own rules are tested against the path relative to its directory, which
    /// is what the unscoped `ignore` matcher expects. A local negation
    /// (whitelist) clears both local and inherited ignores; a local ignore is
    /// OR-ed with the inherited result.
    pub fn ignores(&self, path: &str) -> bool {
        let inherited = self
            .parent
            .as_ref()
            .is_some_and(|parent| parent.ignores(path));
        let Some(local_path) = path.strip_prefix(&self.prefix) else {
            return inherited;
        };
        // A trailing slash marks a directory, which gitignore needs to know
        // because directory-only patterns (`build/`) match directories only.
        // The `ignore` crate rejects trailing slashes in the path it is given,
        // so the marker is turned into the `is_dir` flag instead.
        let is_dir = local_path.ends_with('/');
        let local_path = local_path.trim_end_matches('/');
        if local_path.is_empty() {
            return inherited;
        }
        let matcher = self.matcher.lock().expect("skill ignore matcher poisoned");
        let Some(matcher) = matcher.as_ref() else {
            return inherited;
        };
        match matcher.matched(Path::new(local_path), is_dir) {
            // A local negation clears inherited ignores.
            Match::Whitelist(_) => false,
            Match::Ignore(_) => true,
            Match::None => inherited,
        }
    }

    /// The directory prefix this matcher scopes rules to, with trailing slash.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }
}

/// Directory prefix with a trailing slash, or the empty string for the root.
fn directory_prefix(directory: String) -> String {
    if directory.is_empty() {
        String::new()
    } else {
        format!("{directory}/")
    }
}

/// Build a gitignore matcher from re-anchored rule lines.
fn build_matcher(lines: &[String]) -> Gitignore {
    let mut builder = GitignoreBuilder::new("");
    for line in lines {
        // `add_line` only fails on invalid glob syntax; the ignore crate treats
        // those lines as non-matching, which is the desired behavior.
        let _ = builder.add_line(None, line);
    }
    builder.build().unwrap_or_else(|_| {
        GitignoreBuilder::new("")
            .build()
            .expect("empty gitignore builds")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matcher(dir: &str) -> SkillIgnoreMatcher {
        SkillIgnoreMatcher::new(dir)
    }

    #[test]
    fn root_pattern_ignores_descendants() {
        let matcher = matcher("");
        matcher.add("*.log\n");
        assert!(matcher.ignores("a.log"));
        assert!(matcher.ignores("nested/b.log"));
        assert!(!matcher.ignores("a.txt"));
    }

    #[test]
    fn scoped_pattern_does_not_leak_outside_prefix() {
        let matcher = matcher("nested");
        matcher.add("*.log\n");
        assert!(matcher.ignores("nested/a.log"));
        assert!(!matcher.ignores("a.log"));
    }

    #[test]
    fn negation_whitelists_and_beats_ignore() {
        let matcher = matcher("");
        matcher.add("*.log\n!keep.log\n");
        assert!(matcher.ignores("drop.log"));
        assert!(!matcher.ignores("keep.log"));
    }

    #[test]
    fn parent_chain_is_delegated() {
        let parent = Arc::new(matcher(""));
        parent.add("*.log\n");
        let child = SkillIgnoreMatcher::with_parent("child", parent);
        assert!(child.ignores("child/a.log"));
        assert!(child.ignores("a.log"));
        assert!(!child.ignores("child/a.txt"));
    }

    #[test]
    fn local_negation_beats_parent_ignore() {
        let parent = Arc::new(matcher(""));
        parent.add("*.log\n");
        let child = SkillIgnoreMatcher::with_parent("child", parent);
        child.add("!*.log\n");
        assert!(child.ignores("a.log"));
        assert!(!child.ignores("child/a.log"));
    }

    #[test]
    fn grandparent_chain_is_delegated() {
        let root = Arc::new(matcher(""));
        root.add("*.tmp\n");
        let mid = Arc::new(SkillIgnoreMatcher::with_parent("mid", root));
        let leaf = SkillIgnoreMatcher::with_parent("mid/leaf", mid);
        assert!(leaf.ignores("mid/leaf/x.tmp"));
        assert!(leaf.ignores("mid/x.tmp"));
        assert!(leaf.ignores("x.tmp"));
        assert!(!leaf.ignores("mid/leaf/x.txt"));
    }

    #[test]
    fn comments_and_blank_lines_are_skipped() {
        let matcher = matcher("");
        matcher.add("# a comment\n\n*.log\n");
        assert!(matcher.ignores("a.log"));
        assert!(!matcher.ignores("# a comment"));
    }

    #[test]
    fn directory_only_pattern_needs_trailing_slash() {
        let matcher = matcher("");
        matcher.add("build/\n");
        // The trailing slash marks the path as a directory.
        assert!(matcher.ignores("build/"));
        assert!(!matcher.ignores("build"));
    }

    #[test]
    fn unanchored_pattern_does_not_leak_from_sibling() {
        // A rule in `nested/` only applies inside `nested/`, because the
        // matcher tests paths relative to its own directory.
        let matcher = matcher("nested");
        matcher.add("*.log\n");
        assert!(matcher.ignores("nested/a.log"));
        assert!(!matcher.ignores("a.log"));
    }
}
