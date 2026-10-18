//! Backend-neutral contracts shared by the agent runtime and the harness:
//! filesystem and shell capabilities, error codes, skills, and prompt templates.
//!
//! Ported from `@step-harness/agent-core` `src/harness/types.ts`. The TypeScript
//! `Result` monad is replaced by `std::result::Result`, and `AbortSignal` by
//! [`tokio_util::sync::CancellationToken`].

use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::types::{AgentToolResult, ToolUpdateCallback};
use step_harness_providers::types::{CacheRetention, Json, SimpleStreamOptions, Transport};

/// Shared abort signal handle for environment operations.
pub type AbortSignal = CancellationToken;

/// Content written to a file: UTF-8 text or raw bytes.
#[derive(Clone, Debug, PartialEq)]
pub enum FileContent {
    Text(String),
    Binary(Vec<u8>),
}

impl From<String> for FileContent {
    fn from(value: String) -> Self {
        FileContent::Text(value)
    }
}

impl From<&str> for FileContent {
    fn from(value: &str) -> Self {
        FileContent::Text(value.to_owned())
    }
}

impl From<Vec<u8>> for FileContent {
    fn from(value: Vec<u8>) -> Self {
        FileContent::Binary(value)
    }
}

/// Kind of filesystem object as addressed by a [`FileSystem`]. Symlinks are not
/// followed automatically.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum FileKind {
    File,
    Directory,
    Symlink,
}

/// Metadata for one filesystem object in a [`FileSystem`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileInfo {
    /// Basename of [`FileInfo::path`].
    pub name: String,
    /// Absolute, syntactically normalized addressed path. Symlinks are not followed.
    pub path: String,
    /// Object kind. Symlink targets are not followed.
    pub kind: FileKind,
    /// Size in bytes for the addressed filesystem object.
    pub size: u64,
    /// Modification time as milliseconds since the Unix epoch.
    pub mtime_ms: u64,
}

/// Stable, backend-independent file error codes returned by [`FileSystem`] operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileErrorCode {
    Aborted,
    NotFound,
    PermissionDenied,
    NotDirectory,
    IsDirectory,
    Invalid,
    NotSupported,
    Unknown,
}

/// Error returned by [`FileSystem`] file operations.
#[derive(Clone, Debug, thiserror::Error)]
#[error("{message}")]
pub struct FileError {
    pub code: FileErrorCode,
    pub message: String,
    /// Absolute addressed path associated with the failure, when available.
    pub path: Option<PathBuf>,
    #[source]
    pub cause: Option<Arc<dyn std::error::Error + Send + Sync>>,
}

impl FileError {
    pub fn new(code: FileErrorCode, message: impl Into<String>, path: Option<PathBuf>) -> Self {
        FileError {
            code,
            message: message.into(),
            path,
            cause: None,
        }
    }

    pub fn with_cause(
        code: FileErrorCode,
        message: impl Into<String>,
        path: Option<PathBuf>,
        cause: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        FileError {
            code,
            message: message.into(),
            path,
            cause: Some(Arc::new(cause)),
        }
    }

    /// Wire name of this error code, matching the TypeScript string union.
    pub fn code_name(&self) -> &'static str {
        match self.code {
            FileErrorCode::Aborted => "aborted",
            FileErrorCode::NotFound => "not_found",
            FileErrorCode::PermissionDenied => "permission_denied",
            FileErrorCode::NotDirectory => "not_directory",
            FileErrorCode::IsDirectory => "is_directory",
            FileErrorCode::Invalid => "invalid",
            FileErrorCode::NotSupported => "not_supported",
            FileErrorCode::Unknown => "unknown",
        }
    }
}

/// Stable, backend-independent execution error codes returned by [`Shell::exec`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionErrorCode {
    Aborted,
    Timeout,
    ShellUnavailable,
    SpawnError,
    CallbackError,
    Unknown,
}

/// Error returned by [`Shell::exec`].
#[derive(Clone, Debug, thiserror::Error)]
#[error("{message}")]
pub struct ExecutionError {
    pub code: ExecutionErrorCode,
    pub message: String,
    #[source]
    pub cause: Option<Arc<dyn std::error::Error + Send + Sync>>,
}

impl ExecutionError {
    pub fn new(code: ExecutionErrorCode, message: impl Into<String>) -> Self {
        ExecutionError {
            code,
            message: message.into(),
            cause: None,
        }
    }

    pub fn with_cause(
        code: ExecutionErrorCode,
        message: impl Into<String>,
        cause: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        ExecutionError {
            code,
            message: message.into(),
            cause: Some(Arc::new(cause)),
        }
    }
}

/// Stable compaction error codes returned by compaction helpers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionErrorCode {
    Aborted,
    SummarizationFailed,
}

/// Error returned by compaction helpers.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct CompactionError {
    pub code: CompactionErrorCode,
    pub message: String,
    #[source]
    pub cause: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl CompactionError {
    pub fn new(code: CompactionErrorCode, message: impl Into<String>) -> Self {
        CompactionError {
            code,
            message: message.into(),
            cause: None,
        }
    }
}

/// Stable branch-summary error codes returned by branch summarization helpers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BranchSummaryErrorCode {
    Aborted,
    SummarizationFailed,
}

/// Error returned by branch summarization helpers.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct BranchSummaryError {
    pub code: BranchSummaryErrorCode,
    pub message: String,
    #[source]
    pub cause: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl BranchSummaryError {
    pub fn new(code: BranchSummaryErrorCode, message: impl Into<String>) -> Self {
        BranchSummaryError {
            code,
            message: message.into(),
            cause: None,
        }
    }
}

/// Options for [`FileSystem::read_text_lines`].
#[derive(Clone, Debug, Default)]
pub struct ReadTextLinesOptions {
    /// Stop once this many lines have been read.
    pub max_lines: Option<usize>,
    pub signal: Option<AbortSignal>,
}

/// A bounded text page. The backend may scan the whole file to report its line count.
#[derive(Clone, Debug)]
pub struct TextFilePage {
    pub content: String,
    pub total_lines: usize,
    pub selected_lines: usize,
    pub first_line_bytes: usize,
    pub has_more: bool,
    pub byte_truncated: bool,
}

#[derive(Clone, Debug)]
pub struct TextFilePageOptions {
    pub offset: usize,
    pub limit: Option<usize>,
    pub max_bytes: usize,
    pub signal: Option<AbortSignal>,
}

/// Options for [`FileSystem::create_dir`]. Defaults: `recursive: true`.
#[derive(Clone, Debug, Default)]
pub struct CreateDirOptions {
    pub recursive: Option<bool>,
    pub signal: Option<AbortSignal>,
}

/// Options for [`FileSystem::remove`]. Defaults: `recursive: false`, `force: false`.
#[derive(Clone, Debug, Default)]
pub struct RemoveOptions {
    pub recursive: Option<bool>,
    pub force: Option<bool>,
    pub signal: Option<AbortSignal>,
}

/// Options for [`FileSystem::create_temp_file`].
#[derive(Clone, Debug, Default)]
pub struct CreateTempFileOptions {
    pub prefix: Option<String>,
    pub suffix: Option<String>,
    pub signal: Option<AbortSignal>,
}

/// Result of [`Shell::exec`].
#[derive(Clone, Debug)]
pub struct ShellExecResult {
    /// Captured output, empty when the corresponding streaming callback is configured.
    pub stdout: String,
    /// Captured output, empty when the corresponding streaming callback is configured.
    pub stderr: String,
    pub exit_code: i32,
}

/// Options for [`Shell::exec`].
#[derive(Clone, Default)]
pub struct ShellExecOptions {
    /// Working directory. Relative paths resolve against [`FileSystem::cwd`].
    pub cwd: Option<String>,
    /// Environment variables; values override inherited defaults when `inherit_env` is true.
    pub env: Option<std::collections::HashMap<String, String>>,
    /// Whether to inherit the environment's default variables. Defaults to true.
    pub inherit_env: Option<bool>,
    /// Timeout in seconds. Implementations should return a timeout error when exceeded.
    pub timeout: Option<f64>,
    pub signal: Option<AbortSignal>,
    /// Called with stdout chunks as they are produced.
    pub on_stdout: Option<ShellChunkCallback>,
    /// Called with stderr chunks as they are produced.
    pub on_stderr: Option<ShellChunkCallback>,
}

/// Callback invoked with stdout/stderr chunks as they are produced.
pub type ShellChunkCallback = std::sync::Arc<dyn Fn(&str) + Send + Sync>;

pub(crate) struct TextPageCollector {
    options: TextFilePageOptions,
    line: usize,
    line_bytes: usize,
    content: Vec<u8>,
    byte_truncated: bool,
    first_line_bytes: usize,
}
impl TextPageCollector {
    pub(crate) fn new(options: &TextFilePageOptions) -> Self {
        Self {
            options: options.clone(),
            line: 0,
            line_bytes: 0,
            content: Vec::new(),
            byte_truncated: false,
            first_line_bytes: 0,
        }
    }
    pub(crate) fn accept(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            let selected = self.line >= self.options.offset
                && self
                    .options
                    .limit
                    .is_none_or(|limit| self.line - self.options.offset < limit);
            let newline_selected = selected
                && self
                    .options
                    .limit
                    .is_none_or(|limit| self.line - self.options.offset + 1 < limit);
            if selected && (byte != b'\n' || newline_selected) {
                if self.content.len() < self.options.max_bytes.saturating_add(4) {
                    self.content.push(byte);
                } else {
                    self.byte_truncated = true;
                }
            }
            if byte == b'\n' {
                if self.line == self.options.offset {
                    self.first_line_bytes = self.line_bytes;
                }
                self.line += 1;
                self.line_bytes = 0;
            } else {
                self.line_bytes += 1;
            }
        }
    }
    pub(crate) fn finish(mut self) -> Result<TextFilePage, FileError> {
        let total_lines = self.line + 1;
        if self.options.offset >= total_lines {
            return Err(FileError::new(
                FileErrorCode::Invalid,
                format!(
                    "Offset {} is beyond end of file ({total_lines} lines total)",
                    self.options.offset.saturating_add(1)
                ),
                None,
            ));
        }
        if self.line == self.options.offset {
            self.first_line_bytes = self.line_bytes;
        }
        let selected_lines =
            (total_lines - self.options.offset).min(self.options.limit.unwrap_or(usize::MAX));
        Ok(TextFilePage {
            content: String::from_utf8_lossy(&self.content).into_owned(),
            total_lines,
            selected_lines,
            first_line_bytes: self.first_line_bytes,
            has_more: selected_lines < total_lines - self.options.offset,
            byte_truncated: self.byte_truncated || self.content.len() > self.options.max_bytes,
        })
    }
}
fn text_file_page_from_bytes(
    bytes: &[u8],
    options: &TextFilePageOptions,
) -> Result<TextFilePage, FileError> {
    let mut collector = TextPageCollector::new(options);
    collector.accept(bytes);
    collector.finish()
}

/// An exclusive lease released when its last owner is dropped.
pub trait FileLease: Send + Sync {}
impl<T: Send + Sync> FileLease for T {}

/// Filesystem capability used by the harness.
///
/// Paths may be absolute or relative to [`FileSystem::cwd`]. Operation methods
/// must never panic: all failures are encoded in the returned `Result`.
#[async_trait::async_trait]
pub trait FileSystem: Send + Sync {
    /// Current working directory for relative paths.
    fn cwd(&self) -> &str;

    /// Absolute addressed path without requiring existence and without resolving symlinks.
    async fn absolute_path(
        &self,
        path: &str,
        signal: Option<&AbortSignal>,
    ) -> Result<String, FileError>;

    /// Join path segments in the filesystem namespace without requiring existence.
    async fn join_path(
        &self,
        parts: &[String],
        signal: Option<&AbortSignal>,
    ) -> Result<String, FileError>;

    /// Read a UTF-8 text file.
    async fn read_text_file(
        &self,
        path: &str,
        signal: Option<&AbortSignal>,
    ) -> Result<String, FileError>;

    /// Read UTF-8 text lines. Implementations stop once `max_lines` lines are read.
    async fn read_text_lines(
        &self,
        path: &str,
        options: Option<&ReadTextLinesOptions>,
    ) -> Result<Vec<String>, FileError>;

    /// Read a binary file.
    async fn read_binary_file(
        &self,
        path: &str,
        signal: Option<&AbortSignal>,
    ) -> Result<Vec<u8>, FileError>;

    /// Read only a prefix for format detection. Backends should override the fallback.
    async fn read_file_prefix(
        &self,
        path: &str,
        max_bytes: usize,
        signal: Option<&AbortSignal>,
    ) -> Result<Vec<u8>, FileError> {
        let mut bytes = self.read_binary_file(path, signal).await?;
        bytes.truncate(max_bytes);
        Ok(bytes)
    }

    /// Read a bounded text page. Non-local backends may use this compatibility fallback.
    async fn read_text_page(
        &self,
        path: &str,
        options: &TextFilePageOptions,
    ) -> Result<TextFilePage, FileError> {
        let bytes = self.read_binary_file(path, options.signal.as_ref()).await?;
        text_file_page_from_bytes(&bytes, options)
    }

    /// Create or overwrite a file, creating parent directories when supported.
    async fn write_file(
        &self,
        path: &str,
        content: FileContent,
        signal: Option<&AbortSignal>,
    ) -> Result<(), FileError>;

    /// Create or append to a file, creating parent directories when supported.
    async fn append_file(
        &self,
        path: &str,
        content: FileContent,
        signal: Option<&AbortSignal>,
    ) -> Result<(), FileError>;

    /// Acquire an exclusive cross-process lease for a durable file.
    /// Backends without durable storage support must report NotSupported.
    async fn lock_file(&self, _path: &str) -> Result<Arc<dyn FileLease>, FileError> {
        Err(FileError::new(
            FileErrorCode::NotSupported,
            "Exclusive file leases are unsupported",
            None,
        ))
    }

    /// Append and sync all bytes before acknowledging the mutation.
    async fn append_file_durable(
        &self,
        _path: &str,
        _content: FileContent,
    ) -> Result<(), FileError> {
        Err(FileError::new(
            FileErrorCode::NotSupported,
            "Durable appends are unsupported",
            None,
        ))
    }

    /// Publish a complete synced file using a unique same-directory temporary file.
    /// When overwrite is false, fail if the destination already exists.
    async fn write_file_atomic(
        &self,
        _path: &str,
        _content: FileContent,
        _overwrite: bool,
    ) -> Result<(), FileError> {
        Err(FileError::new(
            FileErrorCode::NotSupported,
            "Atomic durable publication is unsupported",
            None,
        ))
    }

    /// Atomically rename a file, replacing the destination when it exists.
    async fn rename_file(
        &self,
        source_path: &str,
        destination_path: &str,
        signal: Option<&AbortSignal>,
    ) -> Result<(), FileError>;

    /// Metadata for the addressed path without following symlinks.
    async fn file_info(
        &self,
        path: &str,
        signal: Option<&AbortSignal>,
    ) -> Result<FileInfo, FileError>;

    /// List direct children of a directory without following symlinks.
    async fn list_dir(
        &self,
        path: &str,
        signal: Option<&AbortSignal>,
    ) -> Result<Vec<FileInfo>, FileError>;

    /// Canonical path for an existing path, resolving symlinks where supported.
    async fn canonical_path(
        &self,
        path: &str,
        signal: Option<&AbortSignal>,
    ) -> Result<String, FileError>;

    /// `false` for missing paths; other errors return a [`FileError`].
    async fn exists(&self, path: &str, signal: Option<&AbortSignal>) -> Result<bool, FileError>;

    /// Create a directory. Defaults to recursive.
    async fn create_dir(
        &self,
        path: &str,
        options: Option<&CreateDirOptions>,
    ) -> Result<(), FileError>;

    /// Remove a file or directory.
    async fn remove(&self, path: &str, options: Option<&RemoveOptions>) -> Result<(), FileError>;

    /// Create a temporary directory and return its absolute path.
    async fn create_temp_dir(
        &self,
        prefix: Option<&str>,
        signal: Option<&AbortSignal>,
    ) -> Result<String, FileError>;

    /// Create a temporary file and return its absolute path.
    async fn create_temp_file(
        &self,
        options: Option<&CreateTempFileOptions>,
    ) -> Result<String, FileError>;

    /// Release filesystem resources. Best-effort; must not panic.
    async fn cleanup(&self);
}

/// Shell execution capability used by the harness.
#[async_trait::async_trait]
pub trait Shell: Send + Sync {
    /// Execute a shell command in [`FileSystem::cwd`] unless `options.cwd` is provided.
    async fn exec(
        &self,
        command: &str,
        options: Option<&ShellExecOptions>,
    ) -> Result<ShellExecResult, ExecutionError>;

    /// Release shell resources. Best-effort; must not panic.
    async fn cleanup(&self);
}

/// Filesystem and process execution environment used by the harness.
///
/// Implement [`FileSystem`] and [`Shell`] for one type to get an
/// `ExecutionEnv` for free via the blanket implementation.
#[async_trait::async_trait]
pub trait ExecutionEnv: FileSystem + Shell {}

#[async_trait::async_trait]
impl<T: FileSystem + Shell> ExecutionEnv for T {}

/// Skill loaded from a `SKILL.md` file or provided by an application.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Skill {
    /// Stable skill name used for lookup and model-visible listings.
    pub name: String,
    /// Short model-visible description of when to use the skill.
    pub description: String,
    /// Full skill instructions.
    pub content: String,
    /// Absolute path to the skill file.
    pub file_path: String,
    /// Exclude from model-visible skill lists while allowing explicit invocation.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub disable_model_invocation: bool,
}

/// Prompt template that can be formatted into a prompt for explicit invocation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptTemplate {
    /// Stable template name used for lookup or command routing.
    pub name: String,
    /// Optional description for command lists or autocomplete.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Template content; placeholders are formatted by `format_prompt_template_invocation`.
    pub content: String,
}

/// Resources made available to explicit invocation methods and system-prompt callbacks.
#[derive(Clone, Debug, Default)]
pub struct AgentHarnessResources {
    pub prompt_templates: Option<Vec<PromptTemplate>>,
    pub skills: Option<Vec<Skill>>,
}

/// Tool definition executed by the harness with an application-defined context.
///
/// The Rust counterpart of the TypeScript `AgentHarnessTool`: an [`AgentTool`]
/// whose `execute` also receives a resolved turn context.
#[derive(Clone)]
pub struct AgentHarnessTool<TContext, TDetails = Json> {
    pub name: String,
    pub label: String,
    pub description: String,
    pub parameters: Json,
    pub execution_mode: Option<crate::types::ToolExecutionMode>,
    pub prepare_arguments: Option<crate::types::PrepareArgumentsFn>,
    pub execute: std::sync::Arc<dyn ContextToolExecutor<TContext, TDetails>>,
}

/// Executor for an [`AgentHarnessTool`], receiving the turn context.
#[async_trait::async_trait]
pub trait ContextToolExecutor<TContext, TDetails = Json>: Send + Sync {
    async fn execute(
        &self,
        tool_call_id: &str,
        params: &serde_json::Map<String, Json>,
        signal: Option<&AbortSignal>,
        on_update: Option<&ToolUpdateCallback<TDetails>>,
        context: &TContext,
    ) -> Result<AgentToolResult<TDetails>, crate::types::ToolError>;
}

impl<TContext, TDetails> AgentHarnessTool<TContext, TDetails> {
    /// Bind this tool to one context, producing an [`crate::types::AgentTool`]
    /// usable directly with the agent loop.
    pub fn into_agent_tool(self, context: TContext) -> crate::types::AgentTool<TDetails>
    where
        TContext: Send + Sync + 'static,
        TDetails: Send + Sync + 'static,
    {
        let executor = self.execute;
        crate::types::AgentTool {
            name: self.name,
            label: self.label,
            description: self.description,
            parameters: self.parameters,
            execution_mode: self.execution_mode,
            prepare_arguments: self.prepare_arguments,
            executor: Arc::new(BoundContextExecutor { executor, context }),
        }
    }
}

struct BoundContextExecutor<TContext, TDetails> {
    executor: Arc<dyn ContextToolExecutor<TContext, TDetails>>,
    context: TContext,
}

#[async_trait::async_trait]
impl<TContext, TDetails> crate::types::ToolExecutor<TDetails>
    for BoundContextExecutor<TContext, TDetails>
where
    TContext: Send + Sync,
    TDetails: Send + Sync + 'static,
{
    async fn execute(
        &self,
        tool_call_id: &str,
        params: &serde_json::Map<String, Json>,
        signal: Option<&AbortSignal>,
        on_update: Option<&ToolUpdateCallback<TDetails>>,
    ) -> Result<AgentToolResult<TDetails>, crate::types::ToolError> {
        self.executor
            .execute(tool_call_id, params, signal, on_update, &self.context)
            .await
    }
}

/// Async provider of a tool context, resolved for each turn snapshot.
#[async_trait::async_trait]
pub trait ToolContextProvider<TContext>: Send + Sync {
    async fn resolve(&self) -> TContext;
}

/// Static tool context or zero-argument provider resolved for each turn snapshot.
pub enum ToolContextSource<TContext> {
    Static(TContext),
    Provider(std::sync::Arc<dyn ToolContextProvider<TContext>>),
}

/// Curated provider request options owned by the harness and snapshotted per turn.
#[derive(Clone, Debug, Default)]
pub struct AgentHarnessStreamOptions {
    /// Preferred transport forwarded to the stream function.
    pub transport: Option<Transport>,
    /// Provider request timeout in milliseconds.
    pub timeout_ms: Option<u64>,
    /// Maximum provider retry attempts.
    pub max_retries: Option<u32>,
    /// Optional cap for provider-requested retry delays.
    pub max_retry_delay_ms: Option<u64>,
    /// Additional request headers merged with auth and lifecycle headers.
    pub headers: Option<std::collections::HashMap<String, String>>,
    /// Provider metadata forwarded with requests.
    pub metadata: Option<serde_json::Map<String, Json>>,
    /// Provider cache retention hint.
    pub cache_retention: Option<CacheRetention>,
}

impl AgentHarnessStreamOptions {
    /// Project these options onto provider [`SimpleStreamOptions`].
    pub fn to_simple_stream_options(&self) -> SimpleStreamOptions {
        let mut options = SimpleStreamOptions::default();
        options.base.transport = self.transport;
        options.base.base.timeout_ms = self.timeout_ms;
        options.base.base.max_retries = self.max_retries;
        options.base.base.max_retry_delay_ms = self.max_retry_delay_ms;
        if let Some(headers) = &self.headers {
            options.base.base.headers = Some(
                headers
                    .iter()
                    .map(|(key, value)| (key.clone(), Some(value.clone())))
                    .collect(),
            );
        }
        options.base.metadata = self.metadata.clone();
        options.base.cache_retention = self.cache_retention;
        options
    }
}

/// Per-request stream option patch returned by provider hooks.
///
/// A `None` map value deletes that key; a `None` field clears the whole map.
#[derive(Clone, Debug, Default)]
pub struct AgentHarnessStreamOptionsPatch {
    pub transport: Option<Transport>,
    pub timeout_ms: Option<u64>,
    pub max_retries: Option<u32>,
    pub max_retry_delay_ms: Option<u64>,
    pub headers: Option<std::collections::HashMap<String, Option<String>>>,
    pub metadata: Option<serde_json::Map<String, Json>>,
    pub cache_retention: Option<CacheRetention>,
}
