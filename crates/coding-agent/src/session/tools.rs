//! The default tool set for a session.

use std::sync::Arc;

use step_harness_agent_core::harness::env::NodeExecutionEnvOptions;
use step_harness_agent_core::harness::types::ExecutionEnv;
use step_harness_agent_core::types::AgentTool;

use crate::search_web::{SearchWebToolOptions, create_search_web_tool};
use crate::settings::Settings;

/// Build the session's tools, honoring `settings.defaultTools`.
///
/// The default set is the four core tools (`read`, `bash`, `write`, `edit`)
/// plus discovery (`ls`, `find`, `grep`) and `search_web`. On Windows the
/// `powershell` tool is added as a native alternative to `bash`.
pub fn build_default_tools(env: Arc<dyn ExecutionEnv>, settings: &Settings) -> Vec<AgentTool> {
    build_default_tools_with_search(env, settings, SearchWebToolOptions::default())
}

pub(crate) fn build_default_tools_with_search(
    env: Arc<dyn ExecutionEnv>,
    settings: &Settings,
    search_options: SearchWebToolOptions,
) -> Vec<AgentTool> {
    let mut tools = vec![
        step_harness_agent_core::create_read_tool_with_env(env.clone(), None),
        step_harness_agent_core::create_bash_tool_with_env(env.clone()),
        step_harness_agent_core::create_write_tool_with_env(env.clone()),
        step_harness_agent_core::create_edit_tool_with_env(env.clone()),
        step_harness_agent_core::create_ls_tool_with_env(env.clone()),
        step_harness_agent_core::create_find_tool_with_env(env.clone()),
        step_harness_agent_core::create_grep_tool_with_env(env.clone()),
        create_search_web_tool(search_options),
    ];
    if cfg!(windows) {
        tools.push(step_harness_agent_core::create_powershell_tool_with_env(
            env,
        ));
    }

    if let Some(enabled) = &settings.default_tools
        && !enabled.is_empty()
    {
        tools.retain(|tool| enabled.iter().any(|name| name == &tool.name));
    }
    tools
}

/// Build an execution environment rooted at a working directory.
pub fn env_for_cwd(cwd: &str) -> Arc<dyn ExecutionEnv> {
    Arc::new(
        step_harness_agent_core::harness::env::NodeExecutionEnv::new(NodeExecutionEnvOptions {
            cwd: Some(cwd.to_owned()),
            ..Default::default()
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_the_default_tool_set() {
        let env = Arc::new(
            step_harness_agent_core::harness::env::NodeExecutionEnv::new(Default::default()),
        ) as Arc<dyn ExecutionEnv>;
        let tools = build_default_tools(env, &Settings::default());
        let names: Vec<&str> = tools.iter().map(|tool| tool.name.as_str()).collect();
        assert!(names.contains(&"read"));
        assert!(names.contains(&"bash"));
        assert!(names.contains(&"write"));
        assert!(names.contains(&"edit"));
        assert!(names.contains(&"ls"));
        assert!(names.contains(&"find"));
        assert!(names.contains(&"grep"));
        assert!(names.contains(&"search_web"));
    }

    #[test]
    fn default_tools_setting_filters_the_set() {
        let env = Arc::new(
            step_harness_agent_core::harness::env::NodeExecutionEnv::new(Default::default()),
        ) as Arc<dyn ExecutionEnv>;
        let settings = Settings {
            default_tools: Some(vec!["read".to_owned(), "bash".to_owned()]),
            ..Default::default()
        };
        let tools = build_default_tools(env, &settings);
        assert_eq!(tools.len(), 2);
    }
}
