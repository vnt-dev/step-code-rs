//! Prompt template discovery, loading, and argument substitution.
//!
//! Ported from `@step-harness/agent-core` `src\harness\prompt-templates.ts`.
//! Directory inputs load direct `.md` children non-recursively; file inputs
//! load explicit `.md` files. Read and parse failures surface as
//! [`PromptTemplateDiagnostic`] warnings.
//!
//! Notable differences from the TypeScript source:
//! - `Result` values are `std::result::Result`.
//! - YAML frontmatter uses a shared typed serde-saphyr parser.
//! - Unlike the skill loader, this parser does not strip a leading BOM; the
//!   TypeScript source's `prompt-templates.ts` variant does not either, so the
//!   behavior is preserved exactly.
//! - `localeCompare` ordering is replaced by byte-order sorting on the
//!   UTF-8-encoded name; see [`sort_file_infos`].

use std::sync::Arc;

use crate::harness::types::{ExecutionEnv, FileErrorCode, FileInfo, FileKind, PromptTemplate};

/// Stable diagnostic codes emitted while loading prompt templates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PromptTemplateDiagnosticCode {
    FileInfoFailed,
    ListFailed,
    ReadFailed,
    ParseFailed,
}

impl PromptTemplateDiagnosticCode {
    /// Wire name of this code, matching the TypeScript string union.
    pub fn as_str(&self) -> &'static str {
        match self {
            PromptTemplateDiagnosticCode::FileInfoFailed => "file_info_failed",
            PromptTemplateDiagnosticCode::ListFailed => "list_failed",
            PromptTemplateDiagnosticCode::ReadFailed => "read_failed",
            PromptTemplateDiagnosticCode::ParseFailed => "parse_failed",
        }
    }
}

/// Warning produced while loading prompt templates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PromptTemplateDiagnostic {
    /// Diagnostic severity. Currently only warnings are emitted.
    pub diagnostic_type: &'static str,
    /// Stable diagnostic code.
    pub code: PromptTemplateDiagnosticCode,
    /// Human-readable diagnostic message.
    pub message: String,
    /// Path associated with the diagnostic.
    pub path: String,
}

impl PromptTemplateDiagnostic {
    fn warning(
        code: PromptTemplateDiagnosticCode,
        message: impl Into<String>,
        path: impl Into<String>,
    ) -> Self {
        PromptTemplateDiagnostic {
            diagnostic_type: "warning",
            code,
            message: message.into(),
            path: path.into(),
        }
    }
}

/// Outcome of loading prompt templates from one or more paths.
#[derive(Clone, Debug, Default)]
pub struct LoadPromptTemplatesResult {
    pub prompt_templates: Vec<PromptTemplate>,
    pub diagnostics: Vec<PromptTemplateDiagnostic>,
}

/// One source-tagged prompt template.
#[derive(Clone, Debug)]
pub struct SourcedPromptTemplate<TSource, TPromptTemplate = PromptTemplate> {
    pub prompt_template: TPromptTemplate,
    pub source: TSource,
}

/// Outcome of loading source-tagged prompt templates.
#[derive(Clone, Debug, Default)]
pub struct LoadSourcedPromptTemplatesResult<TSource, TPromptTemplate = PromptTemplate> {
    pub prompt_templates: Vec<SourcedPromptTemplate<TSource, TPromptTemplate>>,
    pub diagnostics: Vec<(PromptTemplateDiagnostic, TSource)>,
}

/// Load prompt templates from one or more paths.
///
/// Directory inputs load direct `.md` children non-recursively. File inputs load
/// explicit `.md` files. Missing paths and non-markdown files are skipped. Read
/// and parse failures are returned as diagnostics.
pub async fn load_prompt_templates(
    env: &Arc<dyn ExecutionEnv>,
    paths: &[String],
) -> Result<LoadPromptTemplatesResult, crate::harness::types::FileError> {
    let mut prompt_templates = Vec::new();
    let mut diagnostics = Vec::new();
    for path in paths {
        let info = match env.file_info(path, None).await {
            Ok(info) => info,
            Err(error) => {
                if error.code != FileErrorCode::NotFound {
                    diagnostics.push(PromptTemplateDiagnostic::warning(
                        PromptTemplateDiagnosticCode::FileInfoFailed,
                        error.message.clone(),
                        path.clone(),
                    ));
                }
                continue;
            }
        };
        let mut inner = Vec::new();
        let kind = resolve_kind(env, &info, &mut inner).await;
        diagnostics.extend(inner);
        match kind {
            Some(FileKind::Directory) => {
                let result = load_templates_from_dir(env, &info.path).await;
                prompt_templates.extend(result.prompt_templates);
                diagnostics.extend(result.diagnostics);
            }
            Some(FileKind::File) if info.name.to_ascii_lowercase().ends_with(".md") => {
                let result = load_template_from_file(env, &info.path, &info.name).await;
                prompt_templates.extend(result.prompt_templates);
                diagnostics.extend(result.diagnostics);
            }
            _ => {}
        }
    }
    Ok(LoadPromptTemplatesResult {
        prompt_templates,
        diagnostics,
    })
}

/// Load prompt templates from source-tagged paths.
///
/// Source values are preserved exactly and attached to every loaded prompt
/// template and diagnostic. The agent package does not interpret source values;
/// applications define their own provenance shape.
pub async fn load_sourced_prompt_templates<TSource, F>(
    env: &Arc<dyn ExecutionEnv>,
    inputs: &[(String, TSource)],
    map_prompt_template: Option<F>,
) -> Result<
    LoadSourcedPromptTemplatesResult<TSource, PromptTemplate>,
    crate::harness::types::FileError,
>
where
    TSource: Clone,
    F: Fn(PromptTemplate, &TSource) -> PromptTemplate,
{
    let mut prompt_templates = Vec::new();
    let mut diagnostics = Vec::new();
    for (path, source) in inputs {
        let result = load_prompt_templates(env, std::slice::from_ref(path)).await?;
        for template in result.prompt_templates {
            let mapped = match &map_prompt_template {
                Some(map) => map(template.clone(), source),
                None => template,
            };
            prompt_templates.push(SourcedPromptTemplate {
                prompt_template: mapped,
                source: source.clone(),
            });
        }
        for diagnostic in result.diagnostics {
            diagnostics.push((diagnostic, source.clone()));
        }
    }
    Ok(LoadSourcedPromptTemplatesResult {
        prompt_templates,
        diagnostics,
    })
}

/// Load direct `.md` children of one directory, non-recursively.
async fn load_templates_from_dir(
    env: &Arc<dyn ExecutionEnv>,
    dir: &str,
) -> LoadPromptTemplatesResult {
    let mut prompt_templates = Vec::new();
    let mut diagnostics = Vec::new();
    let mut entries = match env.list_dir(dir, None).await {
        Ok(entries) => entries,
        Err(error) => {
            diagnostics.push(PromptTemplateDiagnostic::warning(
                PromptTemplateDiagnosticCode::ListFailed,
                error.message.clone(),
                dir.to_owned(),
            ));
            return LoadPromptTemplatesResult {
                prompt_templates,
                diagnostics,
            };
        }
    };

    sort_file_infos(&mut entries);
    for entry in entries {
        let mut inner = Vec::new();
        let kind = resolve_kind(env, &entry, &mut inner).await;
        diagnostics.extend(inner);
        if kind != Some(FileKind::File) || !entry.name.to_ascii_lowercase().ends_with(".md") {
            continue;
        }
        let result = load_template_from_file(env, &entry.path, &entry.name).await;
        prompt_templates.extend(result.prompt_templates);
        diagnostics.extend(result.diagnostics);
    }
    LoadPromptTemplatesResult {
        prompt_templates,
        diagnostics,
    }
}

/// Load and derive one prompt template file.
async fn load_template_from_file(
    env: &Arc<dyn ExecutionEnv>,
    file_path: &str,
    file_name: &str,
) -> LoadPromptTemplatesResult {
    let mut diagnostics = Vec::new();
    let raw_content = match env.read_text_file(file_path, None).await {
        Ok(content) => content,
        Err(error) => {
            diagnostics.push(PromptTemplateDiagnostic::warning(
                PromptTemplateDiagnosticCode::ReadFailed,
                error.message.clone(),
                file_path.to_owned(),
            ));
            return LoadPromptTemplatesResult {
                prompt_templates: Vec::new(),
                diagnostics,
            };
        }
    };

    let (frontmatter, body) = match parse_frontmatter(&raw_content) {
        Ok(parsed) => parsed,
        Err(message) => {
            diagnostics.push(PromptTemplateDiagnostic::warning(
                PromptTemplateDiagnosticCode::ParseFailed,
                message,
                file_path.to_owned(),
            ));
            return LoadPromptTemplatesResult {
                prompt_templates: Vec::new(),
                diagnostics,
            };
        }
    };

    let first_line = body.lines().find(|line| !line.trim().is_empty());
    let mut description = frontmatter.description.unwrap_or_default();
    if description.is_empty()
        && let Some(first_line) = first_line
    {
        description = truncate_with_ellipsis(first_line, 60);
    }

    LoadPromptTemplatesResult {
        prompt_templates: vec![PromptTemplate {
            name: strip_md_extension(file_name),
            description: Some(description),
            content: body,
        }],
        diagnostics,
    }
}

/// Resolve the kind of a filesystem entry, following symlinks once.
async fn resolve_kind(
    env: &Arc<dyn ExecutionEnv>,
    info: &FileInfo,
    diagnostics: &mut Vec<PromptTemplateDiagnostic>,
) -> Option<FileKind> {
    if info.kind == FileKind::File || info.kind == FileKind::Directory {
        return Some(info.kind);
    }
    let canonical_path = match env.canonical_path(&info.path, None).await {
        Ok(path) => path,
        Err(error) => {
            if error.code != FileErrorCode::NotFound {
                diagnostics.push(PromptTemplateDiagnostic::warning(
                    PromptTemplateDiagnosticCode::FileInfoFailed,
                    error.message.clone(),
                    info.path.clone(),
                ));
            }
            return None;
        }
    };
    let target = match env.file_info(&canonical_path, None).await {
        Ok(info) => info,
        Err(error) => {
            if error.code != FileErrorCode::NotFound {
                diagnostics.push(PromptTemplateDiagnostic::warning(
                    PromptTemplateDiagnosticCode::FileInfoFailed,
                    error.message.clone(),
                    info.path.clone(),
                ));
            }
            return None;
        }
    };
    match target.kind {
        FileKind::File => Some(FileKind::File),
        FileKind::Directory => Some(FileKind::Directory),
        FileKind::Symlink => None,
    }
}

/// Sort file infos by name.
///
/// The TypeScript source uses `String.prototype.localeCompare`; Rust has no
/// equivalent without a collation crate, so names compare by UTF-8 byte order.
/// This matches `localeCompare` for plain ASCII names.
fn sort_file_infos(entries: &mut [FileInfo]) {
    entries.sort_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));
}

/// Strip a trailing `.md` extension, case-insensitively.
///
/// Mirrors the TypeScript `fileName.replace(/\.md$/i, "")`.
fn strip_md_extension(file_name: &str) -> String {
    let lowered = file_name.to_ascii_lowercase();
    match lowered.strip_suffix(".md") {
        Some(_) => file_name[..file_name.len() - 3].to_owned(),
        None => file_name.to_owned(),
    }
}

/// Truncate to `max` characters, appending `...` when truncated.
fn truncate_with_ellipsis(value: &str, max: usize) -> String {
    if value.chars().count() > max {
        format!("{}...", value.chars().take(max).collect::<String>())
    } else {
        value.to_owned()
    }
}

// ---------------------------------------------------------------------------
// Frontmatter
// ---------------------------------------------------------------------------

/// Flat frontmatter fields consumed by the prompt-template loader.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct PromptTemplateFrontmatter {
    pub description: Option<String>,
    /// `argument-hint` key; parsed but not surfaced on [`PromptTemplate`].
    pub argument_hint: Option<String>,
}

/// Parse typed YAML metadata; preserve the existing prompt-template BOM policy.
fn parse_frontmatter(content: &str) -> Result<(PromptTemplateFrontmatter, String), String> {
    super::frontmatter::parse(content, false)
}

// ---------------------------------------------------------------------------
// Argument handling
// ---------------------------------------------------------------------------

/// Parse an argument string using simple shell-style single and double quotes.
///
/// Quotes toggle quoting state; quoted whitespace stays inside the current
/// argument. Empty arguments are dropped, matching the TypeScript state machine.
pub fn parse_command_args(args_string: &str) -> Vec<String> {
    let mut args: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut in_quote: Option<char> = None;

    for char in args_string.chars() {
        if let Some(quote) = in_quote {
            if char == quote {
                in_quote = None;
            } else {
                current.push(char);
            }
        } else if char == '"' || char == '\'' {
            in_quote = Some(char);
        } else if char == ' ' || char == '\t' {
            if !current.is_empty() {
                args.push(std::mem::take(&mut current));
            }
        } else {
            current.push(char);
        }
    }
    if !current.is_empty() {
        args.push(current);
    }
    args
}

/// Substitute prompt template placeholders with command arguments.
///
/// Placeholders are substituted in a fixed order, which is load-bearing:
/// 1. `$1`, `$2`, ... positional arguments;
/// 2. `${@:N}` / `${@:N:L}` argument slices;
/// 3. `$ARGUMENTS` all arguments joined by spaces;
/// 4. `$@` all arguments joined by spaces.
///
/// The order matters because `$ARGUMENTS` and `$@` expand to the joined
/// argument list, which could itself contain text that looks like an earlier
/// placeholder form; expanding them last keeps earlier forms referring to the
/// original arguments only.
pub fn substitute_args(content: &str, args: &[String]) -> String {
    let mut result = content.to_owned();

    // 1. `$N` positional arguments. Out-of-range indices become empty strings.
    result = replace_dollar_indexed(&result, args);

    // 2. `${@:N}` / `${@:N:L}` slices.
    result = replace_dollar_slice(&result, args);

    // 3. and 4. `$ARGUMENTS` then `$@`, both the joined argument list.
    let all_args = args.join(" ");
    result = result.replace("$ARGUMENTS", &all_args);
    result = result.replace("$@", &all_args);

    result
}

/// Replace `$N` placeholders with the Nth argument (1-based).
fn replace_dollar_indexed(content: &str, args: &[String]) -> String {
    let mut result = String::with_capacity(content.len());
    let mut rest = content;
    while let Some(index) = rest.find('$') {
        result.push_str(&rest[..index]);
        rest = &rest[index + 1..];
        // Collect the following ASCII digits.
        let digits_end = rest
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(rest.len());
        if digits_end == 0 {
            // Not a positional placeholder; emit the `$` verbatim.
            result.push('$');
            continue;
        }
        let (digits, tail) = rest.split_at(digits_end);
        // Leading zeros are accepted by `parseInt`; a value that does not fit
        // is treated as out of range. Out-of-range indices become empty.
        if let Some(value) = digits
            .parse::<usize>()
            .ok()
            .filter(|position| *position > 0)
            .and_then(|position| args.get(position - 1))
        {
            result.push_str(value);
        }
        rest = tail;
    }
    result.push_str(rest);
    result
}

/// Replace `${@:N}` and `${@:N:L}` placeholders with argument slices.
///
/// `N` is 1-based and clamped to 0; `L` is the slice length.
fn replace_dollar_slice(content: &str, args: &[String]) -> String {
    let mut result = String::with_capacity(content.len());
    let mut rest = content;
    while let Some(start) = rest.find("${@:") {
        result.push_str(&rest[..start]);
        let after_marker = &rest[start + 4..];
        let Some(close) = after_marker.find('}') else {
            // No closing brace: the placeholder never closes, so it cannot
            // match. Emit the marker and keep scanning the remainder.
            result.push_str("${@:");
            rest = after_marker;
            continue;
        };
        let inner = &after_marker[..close];
        let tail = &after_marker[close + 1..];

        let mut parts = inner.splitn(2, ':');
        let start_digits = parts.next().unwrap_or("");
        let length_digits = parts.next();
        if start_digits.is_empty() || !start_digits.chars().all(|c| c.is_ascii_digit()) {
            // Malformed placeholder: emit it verbatim.
            result.push_str("${@:");
            rest = after_marker;
            continue;
        }

        let start = start_digits
            .parse::<usize>()
            .ok()
            .map(|start| start.saturating_sub(1))
            .unwrap_or(0);
        let slice: String = match length_digits {
            Some(length) if !length.is_empty() && length.chars().all(|c| c.is_ascii_digit()) => {
                match length.parse::<usize>() {
                    Ok(length) => args
                        .iter()
                        .skip(start)
                        .take(length)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(" "),
                    Err(_) => String::new(),
                }
            }
            _ => args
                .iter()
                .skip(start)
                .cloned()
                .collect::<Vec<_>>()
                .join(" "),
        };
        result.push_str(&slice);
        rest = tail;
    }
    result.push_str(rest);
    result
}

/// Format a prompt template invocation with positional arguments.
pub fn format_prompt_template_invocation(template: &PromptTemplate, args: &[String]) -> String {
    substitute_args(&template.content, args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::env::nodejs::NodeExecutionEnv;
    use crate::harness::env::nodejs::NodeExecutionEnvOptions;
    use tempfile::TempDir;

    fn env_for(dir: &TempDir) -> Arc<dyn ExecutionEnv> {
        Arc::new(NodeExecutionEnv::new(NodeExecutionEnvOptions {
            cwd: Some(dir.path().display().to_string()),
            ..Default::default()
        }))
    }

    #[test]
    fn parses_simple_and_quoted_arguments() {
        assert_eq!(parse_command_args(""), Vec::<String>::new());
        assert_eq!(
            parse_command_args("one two three"),
            vec!["one", "two", "three"]
        );
        assert_eq!(
            parse_command_args("\"quoted arg\" 'single quoted' bare"),
            vec!["quoted arg", "single quoted", "bare"]
        );
        // Whitespace runs do not produce empty arguments.
        assert_eq!(parse_command_args("  a \t b  "), vec!["a", "b"]);
        // Quotes inside the other quote style are literal.
        assert_eq!(parse_command_args("\"it's\""), vec!["it's"]);
    }

    #[test]
    fn substitutes_positional_arguments() {
        assert_eq!(
            substitute_args(
                "first=$1 second=$2 third=$3",
                &["a".to_owned(), "b".to_owned()]
            ),
            "first=a second=b third="
        );
        // Unknown indices become empty.
        assert_eq!(substitute_args("$9", &["a".to_owned()]), "");
        // A lone `$` is preserved.
        assert_eq!(substitute_args("cost: $5", &[]), "cost: ");
        assert_eq!(substitute_args("100$", &[]), "100$");
    }

    #[test]
    fn substitutes_argument_slices() {
        let args: Vec<String> = ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
        assert_eq!(substitute_args("${@:2}", &args), "b c d");
        assert_eq!(substitute_args("${@:2:2}", &args), "b c");
        assert_eq!(substitute_args("${@:0:2}", &args), "a b");
        assert_eq!(substitute_args("${@:10}", &args), "");
        assert_eq!(substitute_args("${@:2:10}", &args), "b c d");
    }

    #[test]
    fn substitution_order_is_preserved() {
        let args: Vec<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        // `$ARGUMENTS` expands after `$1`, so the expanded text is not rescanned.
        assert_eq!(substitute_args("$ARGUMENTS", &args), "a b");
        assert_eq!(substitute_args("$@", &args), "a b");
        // `$1` wins over `$@` because positional substitution runs first.
        assert_eq!(substitute_args("$1 $@", &args), "a a b");
        // The slice form runs before `$ARGUMENTS`.
        assert_eq!(substitute_args("${@:1:1}/$ARGUMENTS", &args), "a/a b");
    }

    #[test]
    fn formats_template_invocation() {
        let template = PromptTemplate {
            name: "review".to_owned(),
            description: None,
            content: "Review $ARGUMENTS carefully".to_owned(),
        };
        assert_eq!(
            format_prompt_template_invocation(&template, &["the diff".to_owned()]),
            "Review the diff carefully"
        );
        assert_eq!(
            format_prompt_template_invocation(&template, &[]),
            "Review  carefully"
        );
    }

    #[tokio::test]
    async fn loads_templates_from_directory_non_recursively() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("review.md"),
            "---\ndescription: Review a change\n---\nReview body\n",
        )
        .unwrap();
        std::fs::write(dir.path().join("notes.md"), "Just body text\n").unwrap();
        std::fs::write(dir.path().join("ignore.txt"), "not markdown\n").unwrap();
        std::fs::create_dir_all(dir.path().join("nested")).unwrap();
        std::fs::write(
            dir.path().join("nested").join("deep.md"),
            "---\ndescription: Nested\n---\nNested\n",
        )
        .unwrap();

        let env = env_for(&dir);
        let root = dir.path().display().to_string();
        let result = load_prompt_templates(&env, &[root]).await.unwrap();
        let names: Vec<&str> = result
            .prompt_templates
            .iter()
            .map(|t| t.name.as_str())
            .collect();
        assert_eq!(names, vec!["notes", "review"]);
        assert_eq!(
            result.prompt_templates[0].description.as_deref(),
            Some("Just body text")
        );
        assert_eq!(
            result.prompt_templates[1].description.as_deref(),
            Some("Review a change")
        );
        assert_eq!(result.prompt_templates[1].content, "Review body");
    }

    #[tokio::test]
    async fn loads_explicit_file_and_skips_missing_paths() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("explicit.md");
        std::fs::write(&file, "Body only\n").unwrap();
        let env = env_for(&dir);
        let file_path = file.display().to_string();
        let result = load_prompt_templates(&env, &[file_path]).await.unwrap();
        assert_eq!(result.prompt_templates.len(), 1);
        assert_eq!(result.prompt_templates[0].name, "explicit");
        assert!(result.diagnostics.is_empty());

        let missing = dir.path().join("nope.md").display().to_string();
        let result = load_prompt_templates(&env, &[missing]).await.unwrap();
        assert!(result.prompt_templates.is_empty());
        assert!(result.diagnostics.is_empty());
    }

    #[tokio::test]
    async fn sourced_loader_tags_templates() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.md"), "Body\n").unwrap();
        let env = env_for(&dir);
        let root = dir.path().display().to_string();
        let result = load_sourced_prompt_templates(
            &env,
            &[(root, "builtin".to_owned())],
            None::<fn(PromptTemplate, &String) -> PromptTemplate>,
        )
        .await
        .unwrap();
        assert_eq!(result.prompt_templates.len(), 1);
        assert_eq!(result.prompt_templates[0].source, "builtin");
        assert_eq!(result.prompt_templates[0].prompt_template.name, "a");
    }

    #[test]
    fn derives_description_from_first_body_line() {
        let long_line = "x".repeat(80);
        let content = format!("---\nname: ignored\n---\n{long_line}\n");
        let (frontmatter, body) = parse_frontmatter(&content).unwrap();
        assert_eq!(frontmatter.description, None);
        assert_eq!(body, long_line);

        // Description falls back to the first non-blank line, truncated to 60.
        let (_, body) = parse_frontmatter("---\n---\n\n\nSecond line\n").unwrap();
        assert_eq!(body, "Second line");
        let truncated = truncate_with_ellipsis(&"y".repeat(80), 60);
        assert_eq!(truncated.chars().count(), 63);
        assert!(truncated.ends_with("..."));
    }

    #[test]
    fn frontmatter_does_not_strip_bom() {
        // The prompt-templates parser keeps the BOM, unlike the skills parser.
        let (frontmatter, body) =
            parse_frontmatter("\u{feff}---\ndescription: d\n---\nBody\n").unwrap();
        // With the BOM present the content does not start with `---`, so no
        // frontmatter is recognized and the BOM stays in the body.
        assert_eq!(frontmatter.description, None);
        assert!(body.starts_with('\u{feff}'));
    }
}
