//! Built-in execution tools: `bash`, `read`, `write`, `edit`, `ls`, `find`,
//! `grep`, and `powershell`.
//!
//! Ported from `@step-harness/agent-core` `src\harness\tools\` (bash, read,
//! write, edit) and `@step-harness/coding-agent` `src\core\tools\` (ls, find,
//! grep, powershell). Tools are harness tools (they receive a resolved
//! context); use [`AgentHarnessTool::into_agent_tool`] or the `*_with_env`
//! constructors to bind them to one execution environment for direct use with
//! [`crate::Agent`].

pub mod bash;
pub mod edit;
pub mod edit_diff;
pub mod find;
pub mod grep;
pub mod image;
pub mod ls;
pub mod mutation_queue;
pub mod path_utils;
pub mod powershell;
pub mod read;
pub mod tool_context;
pub mod write;

pub use bash::{
    BASH_UPDATE_THROTTLE, BashExecution, BashPrepare, BashTimeoutError, BashToolDetails,
    BashToolInput, BashToolOptions, bash_parameters_schema, create_bash_tool,
    create_bash_tool_with_env,
};
pub use edit::{
    EditAccessError, EditToolDetails, EditToolInput, create_edit_tool, create_edit_tool_with_env,
    edit_parameters_schema, prepare_edit_arguments,
};
pub use edit_diff::{
    AppliedEditsResult, DiffStringResult, Edit, FuzzyMatchResult, LineEnding,
    apply_edits_to_normalized_content, detect_line_ending, fuzzy_find_text, generate_diff_string,
    generate_unified_patch, normalize_for_fuzzy_match, normalize_to_lf, restore_line_endings,
    strip_bom,
};
pub use find::{
    DEFAULT_FIND_LIMIT, FindToolDetails, FindToolInput, create_find_tool,
    create_find_tool_with_env, find_parameters_schema,
};
pub use grep::{
    DEFAULT_MATCH_LIMIT, GrepToolDetails, GrepToolInput, create_grep_tool,
    create_grep_tool_with_env, grep_parameters_schema,
};
pub use image::{DefaultImageProcessor, detect_supported_image_mime_type, encode_base64};
pub use ls::{
    DEFAULT_ENTRY_LIMIT, LsToolDetails, LsToolInput, create_ls_tool, create_ls_tool_with_env,
    ls_parameters_schema,
};
pub use mutation_queue::with_file_mutation_queue;
pub use path_utils::{resolve_read_tool_path, resolve_tool_path};
pub use powershell::{
    PowerShellToolDetails, PowerShellToolInput, create_powershell_tool,
    create_powershell_tool_with_env, powershell_parameters_schema,
};
pub use read::{
    ReadImageProcessor, ReadImageProcessorOptions, ReadImageProcessorResult, ReadToolDetails,
    ReadToolInput, ReadToolOptions, create_read_tool, create_read_tool_with_env,
    read_parameters_schema,
};
pub use tool_context::{ExecutionToolContext, ToolEnvContext};
pub use write::{
    WriteToolInput, create_write_tool, create_write_tool_with_env, write_parameters_schema,
};
