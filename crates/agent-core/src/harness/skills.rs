//! Skill discovery and loading from `SKILL.md` files.
//!
//! Ported from `@step-harness/agent-core` `src\harness\skills.ts`.
//! Skills are discovered by walking directories recursively, loading the first
//! `SKILL.md` in each directory, plus direct root-level `.md` files carrying
//! skill frontmatter. Ignore files (`.gitignore`, `.ignore`, `.fdignore`) prune
//! the walk; invalid skill files surface as warnings rather than failures.
//!
//! Notable differences from the TypeScript source:
//! - `Result` values are `std::result::Result`; failures are [`SkillDiagnostic`]
//!   warnings, not thrown errors.
//! - Recursion uses an explicit work stack instead of async recursion.
//! - `localeCompare` ordering is replaced by byte-order sorting on the
//!   UTF-8-encoded name; see [`sort_file_infos`] for the caveat.
//! - YAML frontmatter uses a shared typed serde-saphyr parser.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use crate::harness::skill_discovery::SkillIgnoreMatcher;
use crate::harness::types::{ExecutionEnv, FileErrorCode, FileInfo, FileKind, Skill};

const MAX_NAME_LENGTH: usize = 64;
const MAX_DESCRIPTION_LENGTH: usize = 1024;
const IGNORE_FILE_NAMES: [&str; 3] = [".gitignore", ".ignore", ".fdignore"];

/// Stable diagnostic codes emitted while loading skills.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkillDiagnosticCode {
    FileInfoFailed,
    ListFailed,
    ReadFailed,
    ParseFailed,
    InvalidMetadata,
}

impl SkillDiagnosticCode {
    /// Wire name of this code, matching the TypeScript string union.
    pub fn as_str(&self) -> &'static str {
        match self {
            SkillDiagnosticCode::FileInfoFailed => "file_info_failed",
            SkillDiagnosticCode::ListFailed => "list_failed",
            SkillDiagnosticCode::ReadFailed => "read_failed",
            SkillDiagnosticCode::ParseFailed => "parse_failed",
            SkillDiagnosticCode::InvalidMetadata => "invalid_metadata",
        }
    }
}

/// Warning produced while loading skills.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkillDiagnostic {
    /// Diagnostic severity. Currently only warnings are emitted.
    pub diagnostic_type: &'static str,
    /// Stable diagnostic code.
    pub code: SkillDiagnosticCode,
    /// Human-readable diagnostic message.
    pub message: String,
    /// Path associated with the diagnostic.
    pub path: String,
}

impl SkillDiagnostic {
    fn warning(
        code: SkillDiagnosticCode,
        message: impl Into<String>,
        path: impl Into<String>,
    ) -> Self {
        SkillDiagnostic {
            diagnostic_type: "warning",
            code,
            message: message.into(),
            path: path.into(),
        }
    }

    fn from_file_error(
        code: SkillDiagnosticCode,
        error: &crate::harness::types::FileError,
    ) -> Self {
        SkillDiagnostic::warning(
            code,
            error.message.clone(),
            error
                .path
                .clone()
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
        )
    }
}

/// Outcome of loading skills from one or more directories.
#[derive(Clone, Debug, Default)]
pub struct LoadSkillsResult {
    pub skills: Vec<Skill>,
    pub diagnostics: Vec<SkillDiagnostic>,
}

/// One source-tagged skill plus its diagnostic list.
#[derive(Clone, Debug)]
pub struct SourcedSkill<TSource, TSkill = Skill> {
    pub skill: TSkill,
    pub source: TSource,
}

/// Outcome of loading source-tagged skills.
#[derive(Clone, Debug, Default)]
pub struct LoadSourcedSkillsResult<TSource, TSkill = Skill> {
    pub skills: Vec<SourcedSkill<TSource, TSkill>>,
    pub diagnostics: Vec<(SkillDiagnostic, TSource)>,
}

/// Format a skill invocation prompt, optionally appending additional user instructions.
pub fn format_skill_invocation(skill: &Skill, additional_instructions: Option<&str>) -> String {
    let skill_block = format!(
        "<skill name=\"{}\" location=\"{}\">\nReferences are relative to {}.\n\n{}\n</skill>",
        skill.name,
        skill.file_path,
        dirname_env_path(&skill.file_path),
        skill.content
    );
    match additional_instructions {
        Some(instructions) => format!("{skill_block}\n\n{instructions}"),
        None => skill_block,
    }
}

/// Load skills from one or more directories.
///
/// Traverses directories recursively, loads `SKILL.md` files, loads direct root
/// `.md` files with skill frontmatter, honors ignore files, and returns
/// diagnostics for invalid declared skill files. Missing input directories are
/// skipped.
pub async fn load_skills(
    env: &Arc<dyn ExecutionEnv>,
    dirs: &[String],
) -> Result<LoadSkillsResult, crate::harness::types::FileError> {
    let mut skills: Vec<Skill> = Vec::new();
    let mut diagnostics: Vec<SkillDiagnostic> = Vec::new();
    for dir in dirs {
        let root_info = match env.file_info(dir, None).await {
            Ok(info) => info,
            Err(error) => {
                if error.code != FileErrorCode::NotFound {
                    diagnostics.push(SkillDiagnostic::from_file_error(
                        SkillDiagnosticCode::FileInfoFailed,
                        &error,
                    ));
                }
                continue;
            }
        };
        let root_dir = root_info.path.clone();
        let mut inner = Vec::new();
        if resolve_kind(env, &root_info, &mut inner).await != Some(FileKind::Directory) {
            diagnostics.extend(inner);
            continue;
        }
        diagnostics.extend(inner);

        let result =
            load_skills_from_dir(env, &root_dir, true, None, &root_dir, &mut HashSet::new()).await;
        skills.extend(result.skills);
        diagnostics.extend(result.diagnostics);
    }
    Ok(LoadSkillsResult {
        skills,
        diagnostics,
    })
}

/// Load skills from source-tagged directories.
///
/// Source values are preserved exactly and attached to every loaded skill and
/// diagnostic. The agent package does not interpret source values;
/// applications define their own provenance shape.
pub async fn load_sourced_skills<TSource, F>(
    env: &Arc<dyn ExecutionEnv>,
    inputs: &[(String, TSource)],
    map_skill: Option<F>,
) -> Result<LoadSourcedSkillsResult<TSource, Skill>, crate::harness::types::FileError>
where
    TSource: Clone,
    F: Fn(Skill, &TSource) -> Skill,
{
    let mut skills: Vec<SourcedSkill<TSource, Skill>> = Vec::new();
    let mut diagnostics: Vec<(SkillDiagnostic, TSource)> = Vec::new();
    for (path, source) in inputs {
        let result = load_skills(env, std::slice::from_ref(path)).await?;
        for skill in result.skills {
            let mapped = match &map_skill {
                Some(map) => map(skill.clone(), source),
                None => skill,
            };
            skills.push(SourcedSkill {
                skill: mapped,
                source: source.clone(),
            });
        }
        for diagnostic in result.diagnostics {
            diagnostics.push((diagnostic, source.clone()));
        }
    }
    Ok(LoadSourcedSkillsResult {
        skills,
        diagnostics,
    })
}

/// One unit of pending traversal work.
///
/// The TypeScript source recurses into directories and loads root `.md` files
/// in the same sorted pass, so the emitted order interleaves both. Modeling the
/// work as one item kind per step lets the explicit stack reproduce that order
/// exactly: items are pushed in reverse so the LIFO drains them in sorted order.
enum WorkItem {
    /// Visit a directory subtree.
    VisitDir {
        dir: String,
        include_root_files: bool,
        parent: Option<Arc<SkillIgnoreMatcher>>,
    },
    /// Load one `.md` file as a skill candidate.
    LoadFile {
        path: String,
        parent_dir_name: String,
    },
}

/// Load skills from one directory subtree.
///
/// Uses an explicit work stack instead of the TypeScript source's async
/// recursion, draining depth-first in the source's sorted order. `visited_dirs`
/// detects symlink cycles via canonical paths.
async fn load_skills_from_dir(
    env: &Arc<dyn ExecutionEnv>,
    root_dir: &str,
    include_root_files: bool,
    parent_matcher: Option<Arc<SkillIgnoreMatcher>>,
    root_dir_for_relative: &str,
    visited_dirs: &mut HashSet<PathBuf>,
) -> LoadSkillsResult {
    let mut skills = Vec::new();
    let mut diagnostics = Vec::new();

    let mut stack: Vec<WorkItem> = vec![WorkItem::VisitDir {
        dir: root_dir.to_owned(),
        include_root_files,
        parent: parent_matcher,
    }];

    while let Some(item) = stack.pop() {
        let (dir, include_root_files, parent) = match item {
            WorkItem::LoadFile {
                path,
                parent_dir_name,
            } => {
                let result = load_skill_from_file(env, &path, &parent_dir_name).await;
                skills.extend(result.skills);
                diagnostics.extend(result.diagnostics);
                continue;
            }
            WorkItem::VisitDir {
                dir,
                include_root_files,
                parent,
            } => (dir, include_root_files, parent),
        };
        let dir_info = match env.file_info(&dir, None).await {
            Ok(info) => info,
            Err(error) => {
                if error.code != FileErrorCode::NotFound {
                    diagnostics.push(SkillDiagnostic::from_file_error(
                        SkillDiagnosticCode::FileInfoFailed,
                        &error,
                    ));
                }
                continue;
            }
        };
        let mut inner = Vec::new();
        if resolve_kind(env, &dir_info, &mut inner).await != Some(FileKind::Directory) {
            diagnostics.extend(inner);
            continue;
        }
        diagnostics.extend(inner);

        let real_dir = match env.canonical_path(&dir, None).await {
            Ok(canonical) => canonical,
            Err(_) => dir_info.path.clone(),
        };
        let canonical = PathBuf::from(&real_dir);
        if !visited_dirs.insert(canonical) {
            continue;
        }

        let matcher = Arc::new(match parent {
            Some(parent) => SkillIgnoreMatcher::with_parent(
                relative_env_path(root_dir_for_relative, &dir),
                parent,
            ),
            None => SkillIgnoreMatcher::new(relative_env_path(root_dir_for_relative, &dir)),
        });
        add_ignore_rules(env, &matcher, &dir, &mut diagnostics).await;
        let mut entries = match env.list_dir(&dir, None).await {
            Ok(entries) => entries,
            Err(error) => {
                diagnostics.push(SkillDiagnostic::warning(
                    SkillDiagnosticCode::ListFailed,
                    error.message.clone(),
                    dir.clone(),
                ));
                continue;
            }
        };

        // First pass: the first SKILL.md in this directory wins and stops the walk.
        let mut found_skill_md = false;
        for entry in &entries {
            if entry.name != "SKILL.md" {
                continue;
            }
            let full_path = entry.path.clone();
            let mut inner = Vec::new();
            let kind = resolve_kind(env, entry, &mut inner).await;
            diagnostics.extend(inner);
            if kind != Some(FileKind::File) {
                continue;
            }
            let rel_path = relative_env_path(root_dir_for_relative, &full_path);
            if matcher.ignores(&rel_path) {
                continue;
            }
            let result = load_skill_from_file(env, &full_path, &dir_info.name).await;
            if !result.skills.is_empty() {
                skills.extend(result.skills);
                found_skill_md = true;
            }
            diagnostics.extend(result.diagnostics);
            break;
        }
        if found_skill_md {
            continue;
        }

        // Second pass: sorted children. Work items are pushed in reverse so
        // the LIFO stack drains them in the source's sorted order, interleaving
        // directory visits and root-file loads exactly as the recursive
        // TypeScript source does.
        sort_file_infos(&mut entries);
        let mut pending: Vec<WorkItem> = Vec::new();
        for entry in entries {
            if entry.name.starts_with('.') || entry.name == "node_modules" {
                continue;
            }
            let full_path = entry.path.clone();
            let mut inner = Vec::new();
            let kind = resolve_kind(env, &entry, &mut inner).await;
            diagnostics.extend(inner);
            let Some(kind) = kind else { continue };

            let rel_path = relative_env_path(root_dir_for_relative, &full_path);
            let ignore_path = if kind == FileKind::Directory {
                format!("{rel_path}/")
            } else {
                rel_path.clone()
            };
            if matcher.ignores(&ignore_path) {
                continue;
            }

            if kind == FileKind::Directory {
                pending.push(WorkItem::VisitDir {
                    dir: full_path,
                    include_root_files: false,
                    parent: Some(matcher.clone()),
                });
                continue;
            }

            if kind != FileKind::File || !include_root_files || !entry.name.ends_with(".md") {
                continue;
            }
            pending.push(WorkItem::LoadFile {
                path: full_path,
                parent_dir_name: dir_info.name.clone(),
            });
        }
        stack.extend(pending.into_iter().rev());
    }

    LoadSkillsResult {
        skills,
        diagnostics,
    }
}

/// Sort file infos by name.
///
/// The TypeScript source uses `String.prototype.localeCompare`, which orders by
/// locale collation rules. Rust has no equivalent without a collation crate, so
/// names are compared by their UTF-8 byte order. This matches `localeCompare`
/// for ASCII names without case-folding differences; names differing only in
/// case or non-ASCII collation may order differently.
fn sort_file_infos(entries: &mut [FileInfo]) {
    entries.sort_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));
}

/// Read one directory's ignore files and add their rules to `matcher`.
async fn add_ignore_rules(
    env: &Arc<dyn ExecutionEnv>,
    matcher: &SkillIgnoreMatcher,
    dir: &str,
    diagnostics: &mut Vec<SkillDiagnostic>,
) {
    for filename in IGNORE_FILE_NAMES {
        let ignore_path = match env
            .join_path(&[dir.to_owned(), filename.to_owned()], None)
            .await
        {
            Ok(path) => path,
            Err(error) => {
                diagnostics.push(SkillDiagnostic::warning(
                    SkillDiagnosticCode::FileInfoFailed,
                    error.message.clone(),
                    dir.to_owned(),
                ));
                continue;
            }
        };
        let info = match env.file_info(&ignore_path, None).await {
            Ok(info) => info,
            Err(error) => {
                if error.code != FileErrorCode::NotFound {
                    diagnostics.push(SkillDiagnostic::warning(
                        SkillDiagnosticCode::FileInfoFailed,
                        error.message.clone(),
                        ignore_path.clone(),
                    ));
                }
                continue;
            }
        };
        if info.kind != FileKind::File {
            continue;
        }
        match env.read_text_file(&ignore_path, None).await {
            Ok(content) => matcher.add(&content),
            Err(error) => diagnostics.push(SkillDiagnostic::warning(
                SkillDiagnosticCode::ReadFailed,
                error.message.clone(),
                ignore_path.clone(),
            )),
        }
    }
}

/// Load and validate one skill file.
async fn load_skill_from_file(
    env: &Arc<dyn ExecutionEnv>,
    file_path: &str,
    parent_dir_name: &str,
) -> LoadSkillsResult {
    let mut diagnostics = Vec::new();
    let is_declared_skill = is_skill_md_path(file_path);
    let raw_content = match env.read_text_file(file_path, None).await {
        Ok(content) => content,
        Err(error) => {
            diagnostics.push(SkillDiagnostic::warning(
                SkillDiagnosticCode::ReadFailed,
                error.message.clone(),
                file_path.to_owned(),
            ));
            return LoadSkillsResult {
                skills: Vec::new(),
                diagnostics,
            };
        }
    };

    let (frontmatter, body) = match parse_skill_frontmatter(&raw_content) {
        Ok(parsed) => parsed,
        Err(message) => {
            if is_declared_skill {
                diagnostics.push(SkillDiagnostic::warning(
                    SkillDiagnosticCode::ParseFailed,
                    message,
                    file_path.to_owned(),
                ));
            }
            return LoadSkillsResult {
                skills: Vec::new(),
                diagnostics,
            };
        }
    };

    let description = frontmatter.description.clone();
    if !is_declared_skill && description.as_deref().unwrap_or("").trim().is_empty() {
        return LoadSkillsResult {
            skills: Vec::new(),
            diagnostics,
        };
    }

    for error in validate_description(description.as_deref()) {
        diagnostics.push(SkillDiagnostic::warning(
            SkillDiagnosticCode::InvalidMetadata,
            error,
            file_path.to_owned(),
        ));
    }

    let name = frontmatter
        .name
        .clone()
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| parent_dir_name.to_owned());
    for error in validate_name(&name, parent_dir_name) {
        diagnostics.push(SkillDiagnostic::warning(
            SkillDiagnosticCode::InvalidMetadata,
            error,
            file_path.to_owned(),
        ));
    }

    let Some(description) = description.filter(|description| !description.trim().is_empty()) else {
        return LoadSkillsResult {
            skills: Vec::new(),
            diagnostics,
        };
    };

    LoadSkillsResult {
        skills: vec![Skill {
            name,
            description,
            content: body,
            file_path: file_path.to_owned(),
            disable_model_invocation: frontmatter.disable_model_invocation.unwrap_or(false),
        }],
        diagnostics,
    }
}

/// True when the addressed file is a `SKILL.md`, matching the TypeScript
/// trailing-separator-insensitive basename check.
fn is_skill_md_path(path: &str) -> bool {
    let trimmed = path.trim_end_matches(['/', '\\']);
    let last_separator = trimmed.rfind(['/', '\\']);
    match last_separator {
        Some(index) => &trimmed[index + 1..] == "SKILL.md",
        None => trimmed == "SKILL.md",
    }
}

fn validate_name(name: &str, parent_dir_name: &str) -> Vec<String> {
    let mut errors = Vec::new();
    if name != parent_dir_name {
        errors.push(format!(
            "name \"{name}\" does not match parent directory \"{parent_dir_name}\""
        ));
    }
    if name.chars().count() > MAX_NAME_LENGTH {
        errors.push(format!(
            "name exceeds {MAX_NAME_LENGTH} characters ({})",
            name.chars().count()
        ));
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        || name.is_empty()
    {
        errors.push(
            "name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)"
                .to_owned(),
        );
    }
    if name.starts_with('-') || name.ends_with('-') {
        errors.push("name must not start or end with a hyphen".to_owned());
    }
    if name.contains("--") {
        errors.push("name must not contain consecutive hyphens".to_owned());
    }
    errors
}

fn validate_description(description: Option<&str>) -> Vec<String> {
    let mut errors = Vec::new();
    match description {
        None => errors.push("description is required".to_owned()),
        Some(description) if description.trim().is_empty() => {
            errors.push("description is required".to_owned())
        }
        Some(description) => {
            if description.chars().count() > MAX_DESCRIPTION_LENGTH {
                errors.push(format!(
                    "description exceeds {MAX_DESCRIPTION_LENGTH} characters ({})",
                    description.chars().count()
                ));
            }
        }
    }
    errors
}

/// Resolve the kind of a filesystem entry, following symlinks once.
///
/// Returns `None` when the kind is neither file nor directory, or when
/// resolution fails.
async fn resolve_kind(
    env: &Arc<dyn ExecutionEnv>,
    info: &FileInfo,
    diagnostics: &mut Vec<SkillDiagnostic>,
) -> Option<FileKind> {
    if info.kind == FileKind::File || info.kind == FileKind::Directory {
        return Some(info.kind);
    }
    let canonical_path = match env.canonical_path(&info.path, None).await {
        Ok(path) => path,
        Err(error) => {
            if error.code != FileErrorCode::NotFound {
                diagnostics.push(SkillDiagnostic::warning(
                    SkillDiagnosticCode::FileInfoFailed,
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
                diagnostics.push(SkillDiagnostic::warning(
                    SkillDiagnosticCode::FileInfoFailed,
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

/// Directory portion of an environment path, matching `dirnameEnvPath`.
fn dirname_env_path(path: &str) -> String {
    let normalized = path.trim_end_matches(['/', '\\']);
    let separator_index = normalized
        .rfind('/')
        .into_iter()
        .chain(normalized.rfind('\\'))
        .max();
    match separator_index {
        // Windows drive root, e.g. `C:\`.
        Some(2) if normalized.as_bytes().get(1) == Some(&b':') => normalized[..3].to_owned(),
        Some(index) if index > 0 => normalized[..index].to_owned(),
        _ => "/".to_owned(),
    }
}

/// Path of `path` relative to `root`, using forward slashes.
fn relative_env_path(root: &str, path: &str) -> String {
    let normalized_root = root.replace('\\', "/");
    let normalized_root = normalized_root.trim_end_matches('/');
    let normalized_path = path.replace('\\', "/");
    let normalized_path = normalized_path.trim_end_matches('/');
    if normalized_path == normalized_root {
        return String::new();
    }
    match normalized_path.strip_prefix(&format!("{normalized_root}/")) {
        Some(rest) => rest.to_owned(),
        None => normalized_path.trim_start_matches('/').to_owned(),
    }
}

// ---------------------------------------------------------------------------
// Frontmatter
// ---------------------------------------------------------------------------

/// Flat frontmatter fields consumed by the skill loader.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct SkillFrontmatter {
    pub name: Option<String>,
    pub description: Option<String>,
    /// `disable-model-invocation` key; `None` when absent.
    pub disable_model_invocation: Option<bool>,
}

/// Parse typed YAML metadata; unknown extension fields are ignored.
fn parse_skill_frontmatter(content: &str) -> Result<(SkillFrontmatter, String), String> {
    super::frontmatter::parse(content, true)
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

    async fn write(dir: &TempDir, relative: &str, content: &str) {
        let path = dir.path().join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, content).unwrap();
    }

    #[tokio::test]
    async fn loads_declared_skill_md() {
        let dir = TempDir::new().unwrap();
        write(
            &dir,
            "pdf/SKILL.md",
            "---\nname: pdf\ndescription: Read PDFs\n---\nBody text\n",
        )
        .await;
        let env = env_for(&dir);
        let root = dir.path().join("pdf").display().to_string();
        let result = load_skills(&env, &[root]).await.unwrap();
        assert_eq!(result.skills.len(), 1);
        let skill = &result.skills[0];
        assert_eq!(skill.name, "pdf");
        assert_eq!(skill.description, "Read PDFs");
        assert_eq!(skill.content, "Body text");
        assert!(result.diagnostics.is_empty());
    }

    #[tokio::test]
    async fn invalid_name_and_description_produce_diagnostics() {
        let dir = TempDir::new().unwrap();
        write(
            &dir,
            "Bad_Name/SKILL.md",
            "---\nname: Bad Name!\n---\nBody\n",
        )
        .await;
        let env = env_for(&dir);
        let root = dir.path().join("Bad_Name").display().to_string();
        let result = load_skills(&env, &[root]).await.unwrap();
        assert!(result.skills.is_empty());
        let codes: Vec<_> = result
            .diagnostics
            .iter()
            .map(|d| (d.code, d.message.clone()))
            .collect();
        // The frontmatter name wins over the directory name, and is invalid.
        assert!(codes.iter().any(|(code, message)| {
            *code == SkillDiagnosticCode::InvalidMetadata
                && message.contains("does not match parent directory")
        }));
        assert!(codes.iter().any(|(code, message)| {
            *code == SkillDiagnosticCode::InvalidMetadata && message.contains("invalid characters")
        }));
        assert!(codes.iter().any(|(code, message)| {
            *code == SkillDiagnosticCode::InvalidMetadata && message == "description is required"
        }));
    }

    #[tokio::test]
    async fn first_skill_md_per_directory_wins_and_root_md_files_load() {
        let dir = TempDir::new().unwrap();
        write(
            &dir,
            "root/one/SKILL.md",
            "---\nname: one\ndescription: First\n---\nOne body\n",
        )
        .await;
        write(
            &dir,
            "root/one/SKILL.md.bak",
            "---\nname: one\ndescription: Duplicate\n---\nDup\n",
        )
        .await;
        write(
            &dir,
            "root/loose.md",
            "---\nname: loose\ndescription: Loose root file\n---\nLoose body\n",
        )
        .await;
        write(&dir, "root/notes.txt", "not a skill\n").await;
        let env = env_for(&dir);
        let root = dir.path().join("root").display().to_string();
        let result = load_skills(&env, &[root]).await.unwrap();
        assert_eq!(result.skills.len(), 2);
        assert!(result.skills.iter().any(|s| s.name == "one"));
        assert!(result.skills.iter().any(|s| s.name == "loose"));
    }

    #[tokio::test]
    async fn ignore_file_prunes_skills() {
        let dir = TempDir::new().unwrap();
        write(&dir, "tree/.gitignore", "ignored/\n").await;
        write(
            &dir,
            "tree/kept/SKILL.md",
            "---\nname: kept\ndescription: Kept\n---\nKept\n",
        )
        .await;
        write(
            &dir,
            "tree/ignored/SKILL.md",
            "---\nname: ignored\ndescription: Ignored\n---\nIgnored\n",
        )
        .await;
        let env = env_for(&dir);
        let root = dir.path().join("tree").display().to_string();
        let result = load_skills(&env, &[root]).await.unwrap();
        assert_eq!(result.skills.len(), 1);
        assert_eq!(result.skills[0].name, "kept");
    }

    #[tokio::test]
    async fn sourced_loader_tags_skills_and_diagnostics() {
        let dir = TempDir::new().unwrap();
        write(
            &dir,
            "pdf/SKILL.md",
            "---\nname: pdf\ndescription: Read PDFs\n---\nBody\n",
        )
        .await;
        let env = env_for(&dir);
        let root = dir.path().join("pdf").display().to_string();
        let result = load_sourced_skills(
            &env,
            &[(root, "builtin".to_owned())],
            None::<fn(Skill, &String) -> Skill>,
        )
        .await
        .unwrap();
        assert_eq!(result.skills.len(), 1);
        assert_eq!(result.skills[0].source, "builtin");
        assert_eq!(result.skills[0].skill.name, "pdf");
    }

    #[test]
    fn formats_skill_invocation_with_and_without_extra_instructions() {
        let skill = Skill {
            name: "pdf".to_owned(),
            description: "Read PDFs".to_owned(),
            content: "Do the thing".to_owned(),
            file_path: "/skills/pdf/SKILL.md".to_owned(),
            disable_model_invocation: false,
        };
        let plain = format_skill_invocation(&skill, None);
        assert!(plain.starts_with("<skill name=\"pdf\" location=\"/skills/pdf/SKILL.md\">"));
        assert!(plain.contains("References are relative to /skills/pdf."));
        assert!(plain.ends_with("</skill>"));
        let with_extra = format_skill_invocation(&skill, Some("Also do X"));
        assert!(with_extra.ends_with("</skill>\n\nAlso do X"));
    }

    #[test]
    fn detects_skill_md_paths() {
        assert!(is_skill_md_path("/a/b/SKILL.md"));
        assert!(is_skill_md_path("/a/b/SKILL.md/"));
        assert!(is_skill_md_path("C:\\skills\\pdf\\SKILL.md"));
        assert!(!is_skill_md_path("/a/b/SKILL.mdx"));
        assert!(!is_skill_md_path("/a/b/OTHER.md"));
    }

    #[test]
    fn relative_paths_use_forward_slashes() {
        assert_eq!(relative_env_path("/root", "/root/a/b"), "a/b");
        assert_eq!(relative_env_path("/root/", "/root"), "");
        assert_eq!(relative_env_path("C:\\root", "C:\\root\\a"), "a");
    }

    #[test]
    fn frontmatter_parsing_handles_bom_crlf_and_fences() {
        let content = "\u{feff}---\r\nname: pdf\r\ndescription: \"Reads: PDFs\"\r\ndisable-model-invocation: true\r\n---\r\nBody line\r\n";
        let (frontmatter, body) = parse_skill_frontmatter(content).unwrap();
        assert_eq!(frontmatter.name.as_deref(), Some("pdf"));
        assert_eq!(frontmatter.description.as_deref(), Some("Reads: PDFs"));
        assert_eq!(frontmatter.disable_model_invocation, Some(true));
        assert_eq!(body, "Body line");

        let (empty, body) = parse_skill_frontmatter("no frontmatter here").unwrap();
        assert_eq!(empty, SkillFrontmatter::default());
        assert_eq!(body, "no frontmatter here");
    }

    #[test]
    fn sorts_entries_by_byte_order() {
        let mut entries = vec![
            FileInfo {
                name: "b".to_owned(),
                path: "/b".to_owned(),
                kind: FileKind::Directory,
                size: 0,
                mtime_ms: 0,
            },
            FileInfo {
                name: "A".to_owned(),
                path: "/A".to_owned(),
                kind: FileKind::Directory,
                size: 0,
                mtime_ms: 0,
            },
        ];
        sort_file_infos(&mut entries);
        // Byte order puts uppercase before lowercase.
        assert_eq!(entries[0].name, "A");
        assert_eq!(entries[1].name, "b");
    }
}
