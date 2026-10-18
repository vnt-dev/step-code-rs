//! Resource loading: context files (`AGENTS.md`), skills, and prompt templates.
//!
//! Ported functionally from `@step-harness/coding-agent`
//! `src/core/resource-loader.ts`. Skills and prompt templates delegate to the
//! agent-core loaders; this module adds the working-directory walk for
//! context files and merges everything into one snapshot.

use std::sync::Arc;

use step_harness_agent_core::harness::prompt_templates::load_prompt_templates;
use step_harness_agent_core::harness::skills::load_skills;
use step_harness_agent_core::harness::system_prompt::format_skills_for_system_prompt;
use step_harness_agent_core::harness::types::{ExecutionEnv, PromptTemplate, Skill};

use crate::paths::AppPaths;
use crate::settings::Settings;

/// Candidate context-file names, in per-directory precedence order.
const CONTEXT_FILE_NAMES: &[&str] = &[
    "AGENTS.override.md",
    "AGENTS.md",
    "AGENTS.MD",
    "CLAUDE.md",
    "CLAUDE.MD",
];

/// A loaded context file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContextFile {
    pub path: String,
    pub content: String,
}

/// Everything the session injects into its system prompt.
#[derive(Clone, Debug, Default)]
pub struct ResourceSet {
    pub skills: Vec<Skill>,
    pub prompt_templates: Vec<PromptTemplate>,
    /// Context files, ordered global first then root-most ancestor to cwd.
    pub context_files: Vec<ContextFile>,
    /// Human-readable load warnings for the UI.
    pub diagnostics: Vec<String>,
}

impl ResourceSet {
    /// Load skills, prompt templates, and context files for a working directory.
    pub async fn load(
        env: &Arc<dyn ExecutionEnv>,
        cwd: &str,
        paths: &AppPaths,
        settings: &Settings,
    ) -> Self {
        let mut diagnostics = Vec::new();

        // Skills: user directory first, then project directory.
        let mut skill_dirs = vec![paths.skills_dir().to_string_lossy().to_string()];
        skill_dirs.extend(
            settings
                .skills
                .iter()
                .flatten()
                .map(|path| expand_tilde(path))
                .filter(|path| !path.trim().is_empty()),
        );
        let project_skills = crate::paths::project_config_dir(cwd).join("skills");
        skill_dirs.push(project_skills.to_string_lossy().to_string());

        let mut skills = Vec::new();
        for dir in &skill_dirs {
            match load_skills(env, std::slice::from_ref(dir)).await {
                Ok(result) => {
                    skills.extend(result.skills);
                    for diagnostic in result.diagnostics {
                        diagnostics.push(diagnostic.message);
                    }
                }
                Err(error) => diagnostics.push(format!("skills: {error}")),
            }
        }

        // Prompt templates: user directory, then project directory.
        let template_dirs = [
            paths.prompts_dir().to_string_lossy().to_string(),
            crate::paths::project_config_dir(cwd)
                .join("prompts")
                .to_string_lossy()
                .to_string(),
        ];
        let mut prompt_templates = Vec::new();
        match load_prompt_templates(env, &template_dirs).await {
            Ok(result) => {
                prompt_templates.extend(result.prompt_templates);
                for diagnostic in result.diagnostics {
                    diagnostics.push(diagnostic.message);
                }
            }
            Err(error) => diagnostics.push(format!("prompt templates: {error}")),
        }

        let context_files = if settings.context_files.unwrap_or(true) {
            load_context_files(env, cwd, paths).await
        } else {
            Vec::new()
        };

        ResourceSet {
            skills,
            prompt_templates,
            context_files,
            diagnostics,
        }
    }

    /// The `<available_skills>` block for the system prompt.
    pub fn skills_prompt_section(&self) -> String {
        format_skills_for_system_prompt(&self.skills)
    }

    /// The `<project_context>` block for the system prompt.
    pub fn context_files_section(&self) -> String {
        if self.context_files.is_empty() {
            return String::new();
        }
        let mut lines = vec!["<project_context>".to_owned()];
        for file in &self.context_files {
            lines.push(format!("<project_instructions path=\"{}\">", file.path));
            lines.push(file.content.trim_end().to_owned());
            lines.push("</project_instructions>".to_owned());
        }
        lines.push("</project_context>".to_owned());
        lines.join("\n")
    }

    /// Find a skill by name.
    pub fn find_skill(&self, name: &str) -> Option<&Skill> {
        self.skills.iter().find(|skill| skill.name == name)
    }

    /// Resolve explicit selections by file identity and include the full instructions.
    pub fn prompt_with_skills(&self, text: &str, paths: &[String]) -> Result<String, String> {
        let mut blocks = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for path in paths {
            if !seen.insert(path) {
                continue;
            }
            let skill = self.skills.iter().find(|skill| &skill.file_path == path)
                .ok_or_else(|| format!("所选技能已不可用，请重新选择：{path}"))?;
            blocks.push(step_harness_agent_core::harness::skills::format_skill_invocation(skill, None));
        }
        if blocks.is_empty() {
            return Ok(text.to_owned());
        }
        if !text.trim().is_empty() {
            blocks.push(text.to_owned());
        }
        Ok(blocks.join("\n\n"))
    }

    /// Find a prompt template by name.
    pub fn find_template(&self, name: &str) -> Option<&PromptTemplate> {
        self.prompt_templates
            .iter()
            .find(|template| template.name.trim_start_matches('/') == name)
    }
}

/// Load the global context file, then walk ancestors from cwd to the root.
async fn load_context_files(
    env: &Arc<dyn ExecutionEnv>,
    cwd: &str,
    paths: &AppPaths,
) -> Vec<ContextFile> {
    let mut files = Vec::new();

    // Global context file from the data directory.
    if let Some(file) = first_context_file_in(env, &paths.root().to_string_lossy()).await {
        files.push(file);
    }

    // Ancestors from cwd upward; ancestors are unshifted so the final order is
    // root-most first.
    let mut ancestors: Vec<String> = Vec::new();
    let mut current = std::path::PathBuf::from(cwd);
    loop {
        ancestors.push(current.to_string_lossy().to_string());
        if !current.pop() {
            break;
        }
    }
    for dir in ancestors.into_iter().rev() {
        if let Some(file) = first_context_file_in(env, &dir).await
            && !files.iter().any(|existing| existing.path == file.path)
        {
            files.push(file);
        }
    }
    files
}

/// The first matching context file in a directory, if any.
async fn first_context_file_in(env: &Arc<dyn ExecutionEnv>, dir: &str) -> Option<ContextFile> {
    for name in CONTEXT_FILE_NAMES {
        let path = std::path::Path::new(dir).join(name);
        let path_string = path.to_string_lossy().to_string();
        match env.read_text_file(&path_string, None).await {
            Ok(content) => {
                return Some(ContextFile {
                    path: path_string,
                    content: content.trim_start_matches('\u{feff}').to_owned(),
                });
            }
            Err(_) => continue,
        }
    }
    None
}

/// Expand a leading `~` to the user's home directory.
pub fn expand_tilde(path: &str) -> String {
    if path == "~" {
        return dirs::home_dir()
            .map(|home| home.to_string_lossy().to_string())
            .unwrap_or_else(|| path.to_owned());
    }
    if let Some(rest) = path.strip_prefix("~/").or_else(|| path.strip_prefix("~\\"))
        && let Some(home) = dirs::home_dir()
    {
        return home.join(rest).to_string_lossy().to_string();
    }
    path.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn selected_skills_load_full_instructions_and_refresh_from_disk() {
        let dir = tempfile::tempdir().unwrap();
        let cwd = dir.path().to_string_lossy().to_string();
        let paths = AppPaths::new(dir.path().join("user"));
        let skill_dir = dir.path().join(".step-rs/skills/example");
        std::fs::create_dir_all(&skill_dir).unwrap();
        let file = skill_dir.join("SKILL.md");
        std::fs::write(&file, "---\nname: example\ndescription: Example skill\n---\nFollow these complete instructions.").unwrap();
        let env = crate::session::tools::env_for_cwd(&cwd);
        let resources = ResourceSet::load(&env, &cwd, &paths, &Settings::default()).await;
        let selected = resources.skills[0].file_path.clone();
        let prompt = resources.prompt_with_skills("My task", &[selected.clone(), selected.clone()]).unwrap();
        assert!(prompt.contains("Follow these complete instructions."));
        assert!(prompt.replace('\\', "/").contains(&format!(
            "References are relative to {}", file.parent().unwrap().to_string_lossy().replace('\\', "/")
        )));
        assert_eq!(prompt.matches("<skill name=").count(), 1);
        assert!(prompt.ends_with("My task"));
        assert_eq!(resources.prompt_with_skills("plain", &[]).unwrap(), "plain");
        assert!(resources.prompt_with_skills("task", &["unregistered/SKILL.md".into()]).is_err());
        assert!(resources.prompt_with_skills("", std::slice::from_ref(&selected)).unwrap().contains("<skill name="));

        std::fs::write(&file, "---\nname: example\ndescription: Example skill\n---\nUpdated instructions.").unwrap();
        let refreshed = ResourceSet::load(&env, &cwd, &paths, &Settings::default()).await;
        assert!(refreshed.prompt_with_skills("task", std::slice::from_ref(&selected)).unwrap().contains("Updated instructions."));
        std::fs::remove_file(&file).unwrap();
        let refreshed = ResourceSet::load(&env, &cwd, &paths, &Settings::default()).await;
        assert!(refreshed.prompt_with_skills("task", &[selected]).is_err());
    }

    #[test]
    fn expands_tilde_paths() {
        let home = dirs::home_dir().unwrap();
        assert_eq!(
            expand_tilde("~/skills"),
            home.join("skills").to_string_lossy()
        );
        assert_eq!(expand_tilde("/abs/path"), "/abs/path");
    }
}
