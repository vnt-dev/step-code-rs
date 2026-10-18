//! System prompt assembly.
//!
//! Ported functionally from `@step-harness/coding-agent`
//! `src/core/system-prompt.ts`. The desktop app has no terminal UI, so
//! theme/editor guidance is gone; the structure follows the source's default
//! path: introduction, tools, guidelines, project context, skills, cwd.

use step_harness_agent_core::types::AgentTool;

use crate::paths::{AppPaths, project_config_dir};
use crate::resources::ResourceSet;

/// Inputs for [`build_system_prompt`].
pub struct SystemPromptOptions<'a> {
    /// Working directory shown to the model.
    pub cwd: &'a str,
    /// Resolved user data root, including any STEP_RS_HOME override.
    pub paths: &'a AppPaths,
    /// Tools offered this turn.
    pub tools: &'a [AgentTool],
    /// Loaded resources (context files and skills).
    pub resources: &'a ResourceSet,
    /// Appended after the built-in sections.
    pub append_system_prompt: Option<&'a str>,
}

/// Build the system prompt for a session.
pub fn build_system_prompt(options: SystemPromptOptions<'_>) -> String {
    let mut sections: Vec<String> = Vec::new();

    sections.push(
        "You are an expert coding assistant operating inside Step Code, a coding agent harness. \
         You help users with software engineering tasks by reading files, editing code, and \
         running commands in their working directory."
            .to_owned(),
    );

    let tool_lines: Vec<String> = options
        .tools
        .iter()
        .map(|tool| format!("- {}: {}", tool.name, tool.description.trim()))
        .collect();
    if tool_lines.is_empty() {
        sections.push("Available tools:\n(none)".to_owned());
    } else {
        sections.push(format!("Available tools:\n{}", tool_lines.join("\n")));
    }

    sections.push(
        "Guidelines:\n\
         - Be concise in your responses.\n\
         - Show file paths clearly when working with files.\n\
         - Prefer dedicated tools over shell commands when one fits the task."
            .to_owned(),
    );

    let context_section = options.resources.context_files_section();
    if !context_section.is_empty() {
        sections.push(context_section);
    }

    // Installation guidance must be visible even before the first skill exists.
    let user_skills = options
        .paths
        .skills_dir()
        .to_string_lossy()
        .replace('\\', "/");
    let project_skills = project_config_dir(options.cwd)
        .join("skills")
        .to_string_lossy()
        .replace('\\', "/");
    sections.push(format!(
        "Skill installation in Step Code:\n\
         - User-level skills directory (normally ~/.step-rs/skills/): {user_skills}\n\
         - Project-level skills directory (<project>/.step-rs/skills/): {project_skills}\n\
         - When installing or creating a skill, use the project directory by default; use the user directory when the user requests a global/user-level installation. Follow an explicitly requested scope.\n\
         - Install each skill as <skills-directory>/<skill-name>/SKILL.md and preserve its supporting files and relative directory structure. SKILL.md must include YAML frontmatter with a name matching its directory and a description.\n\
         - These are Step Code's installation destinations. Do not use ~/.agents/skills/ or ~/.claude/skills/ unless the user explicitly requests installation for those other tools.\n\
         - If an external installer defaults to another agent's directories, configure its destination explicitly if supported, or download/copy the complete skill directory into the appropriate Step Code directory. Do not rely on the installer's success message; verify SKILL.md and supporting files exist at the Step Code destination."
    ));

    let skills_section = options.resources.skills_prompt_section();
    if !skills_section.is_empty() {
        sections.push(skills_section);
    }

    // Present whenever the plan-mode tools are registered, so the model learns
    // the flow before it needs it.
    if options
        .tools
        .iter()
        .any(|tool| tool.name == crate::plan::ENTER_PLAN_MODE_TOOL)
    {
        sections.push(PLANNING_SECTION.to_owned());
    }

    if let Some(append) = options
        .append_system_prompt
        .filter(|text| !text.trim().is_empty())
    {
        sections.push(append.trim().to_owned());
    }

    let cwd = options.cwd.replace('\\', "/");
    sections.push(format!("Current working directory: {cwd}"));

    sections.join("\n\n")
}

/// Plan-mode guidance, ported from the source's `# Planning` section.
const PLANNING_SECTION: &str = "# Planning\n\
     - A plan is a Markdown proposal explaining how and why to do the work: approach, constraints, trade-offs, and validation. It is not a todo checklist.\n\
     - Call enter_plan_mode before work that spans multiple files, changes architecture or public interfaces, or where the request is ambiguous enough that exploration should shape the approach.\n\
     - There is no entry gate: enter_plan_mode takes effect immediately without user approval, so prefer entering plan mode over guessing when scope is unclear.\n\
     - While plan mode is active, file-editing tools can only write to the session plan file announced by enter_plan_mode. This is not a shell sandbox: bash and powershell retain normal permissions; keep commands read-only and non-destructive while planning.\n\
     - Explore the repository first, then write the complete plan to the announced plan file with write: goal, ordered steps with file paths, validation commands, and open questions.\n\
     - Call exit_plan_mode to submit the written proposal for review, not to approve it yourself. Approval exits plan mode; staying, requesting refinements, or cancelling keeps it active. After refinements, update the plan file and submit it again.\n\
     - After the user approves execution, keep the plan file as the reference and follow it step by step; do not silently diverge from the approved plan.\n\
     - Skip plan mode for trivial work: single-file edits, direct questions, or tasks with an obvious short path.";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::ContextFile;
    use std::sync::Arc;

    fn resources() -> ResourceSet {
        ResourceSet {
            context_files: vec![ContextFile {
                path: "/repo/AGENTS.md".to_owned(),
                content: "Use pnpm.".to_owned(),
            }],
            ..Default::default()
        }
    }

    #[test]
    fn includes_context_files_and_cwd() {
        let prompt = build_system_prompt(SystemPromptOptions {
            cwd: "E:\\repo",
            paths: &AppPaths::new("/user/.step-rs"),
            tools: &[],
            resources: &resources(),
            append_system_prompt: None,
        });
        assert!(prompt.contains("expert coding assistant"));
        assert!(prompt.contains("<project_instructions path=\"/repo/AGENTS.md\">"));
        assert!(prompt.contains("Use pnpm."));
        assert!(prompt.contains("Current working directory: E:/repo"));
    }

    #[test]
    fn appends_custom_instructions() {
        let prompt = build_system_prompt(SystemPromptOptions {
            cwd: "/repo",
            paths: &AppPaths::new("/user/.step-rs"),
            tools: &[],
            resources: &ResourceSet::default(),
            append_system_prompt: Some("Always answer in Chinese."),
        });
        assert!(prompt.contains("Always answer in Chinese."));
        assert!(!prompt.contains("<project_context>"));
    }

    #[tokio::test]
    async fn installation_destinations_are_visible_without_skills_and_loadable() {
        let dir = tempfile::tempdir().unwrap();
        let cwd = dir.path().join("project");
        std::fs::create_dir_all(&cwd).unwrap();
        let cwd = cwd.to_string_lossy().to_string();
        // An explicit data root models STEP_RS_HOME without mutating process env.
        let paths = AppPaths::new(dir.path().join("custom-step-home"));
        let env = crate::session::tools::env_for_cwd(&cwd);
        let settings = crate::settings::Settings::default();
        let resources = ResourceSet::load(&env, &cwd, &paths, &settings).await;
        assert!(resources.skills.is_empty());
        let prompt = build_system_prompt(SystemPromptOptions {
            cwd: &cwd,
            paths: &paths,
            tools: &[],
            resources: &resources,
            append_system_prompt: None,
        });
        let destinations = [paths.skills_dir(), project_config_dir(&cwd).join("skills")];
        for (index, destination) in destinations.iter().enumerate() {
            assert!(prompt.contains(&destination.to_string_lossy().replace('\\', "/")));
            let name = format!("example-{index}");
            let skill_dir = destination.join(&name);
            std::fs::create_dir_all(&skill_dir).unwrap();
            std::fs::write(
                skill_dir.join("SKILL.md"),
                format!("---\nname: {name}\ndescription: Example skill\n---\nUse this skill."),
            )
            .unwrap();
        }
        assert!(prompt.contains("Do not use ~/.agents/skills/ or ~/.claude/skills/"));
        let loaded = ResourceSet::load(&env, &cwd, &paths, &settings).await;
        assert!(loaded.diagnostics.is_empty(), "{:?}", loaded.diagnostics);
        assert_eq!(loaded.skills.len(), 2);
        assert!(loaded.find_skill("example-0").is_some());
        assert!(loaded.find_skill("example-1").is_some());
    }

    #[test]
    fn includes_planning_guidance_when_plan_tools_are_registered() {
        struct NoopExecutor;

        #[async_trait::async_trait]
        impl step_harness_agent_core::types::ToolExecutor for NoopExecutor {
            async fn execute(
                &self,
                _tool_call_id: &str,
                _params: &serde_json::Map<String, serde_json::Value>,
                _signal: Option<&step_harness_agent_core::harness::types::AbortSignal>,
                _on_update: Option<&step_harness_agent_core::types::ToolUpdateCallback>,
            ) -> Result<
                step_harness_agent_core::types::AgentToolResult,
                step_harness_agent_core::types::ToolError,
            > {
                unimplemented!()
            }
        }

        let tool = AgentTool {
            name: crate::plan::ENTER_PLAN_MODE_TOOL.to_owned(),
            label: "enter plan mode".to_owned(),
            description: "Enter plan mode.".to_owned(),
            parameters: serde_json::json!({"type": "object"}),
            execution_mode: None,
            prepare_arguments: None,
            executor: Arc::new(NoopExecutor),
        };
        let prompt = build_system_prompt(SystemPromptOptions {
            cwd: "/repo",
            paths: &AppPaths::new("/user/.step-rs"),
            tools: std::slice::from_ref(&tool),
            resources: &ResourceSet::default(),
            append_system_prompt: None,
        });
        assert!(prompt.contains("# Planning"));
        assert!(prompt.contains("enter_plan_mode"));
        // Without the tools the section stays out.
        let plain = build_system_prompt(SystemPromptOptions {
            cwd: "/repo",
            paths: &AppPaths::new("/user/.step-rs"),
            tools: &[],
            resources: &ResourceSet::default(),
            append_system_prompt: None,
        });
        assert!(!plain.contains("# Planning"));
    }
}
