//! Node-style execution environment backed by tokio.
//!
//! Ported from `@step-harness/agent-core` `src\harness\env\nodejs.ts`
//! (`NodeExecutionEnv`). File operations map IO errors onto [`FileErrorCode`].
//! process-wrap owns Unix process groups and Windows Job Objects.
//!
//! Notable behavior preserved from the TypeScript source:
//! - the command runs with stdout and stderr piped, and the exit is settled
//!   only after both pipes close (or a 100ms post-exit idle grace elapses);
//! - legacy WSL bash paths transport the command over stdin instead of argv;
//! - a panicking stdout/stderr callback aborts the process and surfaces a
//!   `callback_error`.

use std::collections::HashMap;
use std::io;
use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use process_wrap::tokio::{ChildWrapper, CommandWrap, KillOnDrop};
use tokio::io::{AsyncBufReadExt, AsyncReadExt};
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use crate::harness::types::{
    AbortSignal, CreateDirOptions, CreateTempFileOptions, ExecutionError, ExecutionErrorCode,
    FileContent, FileError, FileErrorCode, FileInfo, FileKind, FileSystem, ReadTextLinesOptions,
    RemoveOptions, Shell, ShellExecOptions, ShellExecResult,
};

use crate::harness::timeout::resolve_timeout;
/// Idle grace after process exit before settling with partial pipe output.
const EXIT_STDIO_GRACE: Duration = Duration::from_millis(100);

/// Options for constructing a [`NodeExecutionEnv`].
#[derive(Clone, Debug, Default)]
pub struct NodeExecutionEnvOptions {
    pub cwd: Option<String>,
    pub shell_path: Option<String>,
    pub shell_env: Option<HashMap<String, String>>,
}

/// Filesystem and shell execution environment for the host platform.
pub struct NodeExecutionEnv {
    cwd: String,
    shell_path: Option<String>,
    shell_env: Option<HashMap<String, String>>,
    active_commands: Mutex<HashMap<uuid::Uuid, CancellationToken>>,
}

impl NodeExecutionEnv {
    pub fn new(options: NodeExecutionEnvOptions) -> Self {
        NodeExecutionEnv {
            cwd: options.cwd.unwrap_or_else(|| {
                std::env::current_dir()
                    .map(|dir| dir.display().to_string())
                    .unwrap_or_default()
            }),
            shell_path: options.shell_path,
            shell_env: options.shell_env,
            active_commands: Mutex::new(HashMap::new()),
        }
    }

    fn resolve_path(&self, path: &str) -> Result<String, FileError> {
        resolve_path(&self.cwd, path)
    }

    fn aborted<T>(
        &self,
        signal: Option<&AbortSignal>,
        path: Option<&str>,
    ) -> Option<Result<T, FileError>> {
        signal.filter(|signal| signal.is_cancelled()).map(|_| {
            Err(FileError::new(
                FileErrorCode::Aborted,
                "aborted",
                path.map(PathBuf::from),
            ))
        })
    }
}

/// Lexically normalize a path: resolve `.` and `..` without touching the
/// filesystem, mirroring Node's `path.resolve`/`path.join`.
fn lexical_normalize(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if matches!(
                    normalized.components().next_back(),
                    Some(Component::Normal(_))
                ) {
                    normalized.pop();
                } else if !normalized.has_root() {
                    normalized.push("..");
                }
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized
}

fn resolve_path(cwd: &str, path: &str) -> Result<String, FileError> {
    let mut normalized = path.to_owned();
    if normalized == "~" {
        normalized = home_dir();
    } else if let Some(rest) = normalized.strip_prefix("~/") {
        normalized = format!("{}/{}", home_dir().trim_end_matches('/'), rest);
    } else if cfg!(windows)
        && let Some(rest) = normalized.strip_prefix("~\\")
    {
        normalized = format!("{}\\{}", home_dir().trim_end_matches(['/', '\\']), rest);
    } else if normalized.starts_with("file:") {
        let file_path = file_url_to_path(&normalized)?;
        return path_string(lexical_normalize(&file_path));
    }

    let candidate = Path::new(&normalized);
    if candidate.is_absolute() {
        path_string(lexical_normalize(candidate))
    } else {
        path_string(lexical_normalize(&Path::new(cwd).join(candidate)))
    }
}

fn home_dir() -> String {
    dirs::home_dir()
        .map(|dir| dir.display().to_string())
        .unwrap_or_default()
}

fn path_string(path: PathBuf) -> Result<String, FileError> {
    path.to_str().map(str::to_owned).ok_or_else(|| {
        FileError::new(
            FileErrorCode::Invalid,
            "Path cannot be represented in the UTF-8 environment protocol",
            Some(path),
        )
    })
}

fn file_url_to_path(url: &str) -> Result<PathBuf, FileError> {
    url::Url::parse(url)
        .ok()
        .and_then(|url| url.to_file_path().ok())
        .ok_or_else(|| {
            FileError::new(
                FileErrorCode::Invalid,
                format!("Invalid file URL: {url}"),
                None,
            )
        })
}

fn file_kind_from_metadata(metadata: &std::fs::Metadata) -> Option<FileKind> {
    let file_type = metadata.file_type();
    if file_type.is_file() {
        Some(FileKind::File)
    } else if file_type.is_dir() {
        Some(FileKind::Directory)
    } else if file_type.is_symlink() {
        Some(FileKind::Symlink)
    } else {
        None
    }
}

fn file_info_from_metadata(path: &str, metadata: std::fs::Metadata) -> Result<FileInfo, FileError> {
    let Some(kind) = file_kind_from_metadata(&metadata) else {
        return Err(FileError::new(
            FileErrorCode::Invalid,
            "Unsupported file type",
            Some(PathBuf::from(path)),
        ));
    };
    let name = Path::new(path)
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_default();
    Ok(FileInfo {
        name,
        path: path.to_owned(),
        kind,
        size: metadata.len(),
        mtime_ms: metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|age| age.as_millis() as u64)
            .unwrap_or(0),
    })
}

/// Map a standard IO error onto a backend-independent file error code.
fn to_file_error(error: io::Error, fallback_path: Option<&str>) -> FileError {
    let code = match error.kind() {
        io::ErrorKind::NotFound => FileErrorCode::NotFound,
        io::ErrorKind::PermissionDenied => FileErrorCode::PermissionDenied,
        io::ErrorKind::Interrupted => FileErrorCode::Aborted,
        _ => match error.raw_os_error() {
            Some(20) => FileErrorCode::NotDirectory,  // ENOTDIR
            Some(21) => FileErrorCode::IsDirectory,   // EISDIR
            Some(22) => FileErrorCode::Invalid,       // EINVAL
            Some(267) => FileErrorCode::NotDirectory, // Windows ERROR_DIRECTORY
            _ => FileErrorCode::Unknown,
        },
    };
    FileError::with_cause(
        code,
        error.to_string(),
        fallback_path.map(PathBuf::from),
        error,
    )
}

/// Canonicalize existing files and the nearest existing ancestor for new files.
/// This keeps leases stable across file creation and symlink aliases.
fn canonical_storage_path(path: &Path) -> io::Result<PathBuf> {
    let mut ancestor = path.to_path_buf();
    let mut suffix = Vec::new();
    loop {
        match std::fs::canonicalize(&ancestor) {
            Ok(mut canonical) => {
                for component in suffix.into_iter().rev() {
                    canonical.push(component);
                }
                return Ok(canonical);
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                let Some(name) = ancestor.file_name().map(std::ffi::OsStr::to_os_string) else {
                    return Err(error);
                };
                suffix.push(name);
                if !ancestor.pop() {
                    return Err(error);
                }
            }
            Err(error) => return Err(error),
        }
    }
}

async fn path_exists(path: &str) -> bool {
    tokio::fs::metadata(path).await.is_ok()
}

/// Holds process ownership and unregisters it even when the execution future is dropped.
struct ProcessGuard<'a> {
    child: Box<dyn ChildWrapper>,
    commands: &'a Mutex<HashMap<uuid::Uuid, CancellationToken>>,
    id: uuid::Uuid,
    reaped: bool,
}

impl Drop for ProcessGuard<'_> {
    fn drop(&mut self) {
        if !self.reaped {
            let _ = self.child.start_kill();
        }
        self.commands
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .remove(&self.id);
    }
}

/// How the command reaches the shell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CommandTransport {
    Argv,
    Stdin,
}

struct ShellConfig {
    shell: String,
    args: Vec<String>,
    command_transport: CommandTransport,
}

fn is_legacy_wsl_bash_path(path: &str) -> bool {
    let normalized = path.replace('/', "\\").to_lowercase();
    normalized == r"c:\windows\system32\bash.exe" || normalized == r"c:\windows\sysnative\bash.exe"
}

fn bash_shell_config(shell: &str) -> ShellConfig {
    if is_legacy_wsl_bash_path(shell) {
        ShellConfig {
            shell: shell.to_owned(),
            args: vec!["-s".to_owned()],
            command_transport: CommandTransport::Stdin,
        }
    } else {
        ShellConfig {
            shell: shell.to_owned(),
            args: vec!["-c".to_owned()],
            command_transport: CommandTransport::Argv,
        }
    }
}

async fn find_bash_on_path() -> Option<String> {
    tokio::task::spawn_blocking(|| which::which("bash").ok())
        .await
        .ok()
        .flatten()
        .map(|path| path.to_string_lossy().into_owned())
}

async fn get_shell_config(custom_shell_path: Option<&str>) -> Result<ShellConfig, ExecutionError> {
    if let Some(custom) = custom_shell_path {
        if path_exists(custom).await {
            return Ok(bash_shell_config(custom));
        }
        return Err(ExecutionError::new(
            ExecutionErrorCode::ShellUnavailable,
            format!("Custom shell path not found: {custom}"),
        ));
    }

    if cfg!(windows) {
        let mut candidates: Vec<String> = Vec::new();
        if let Ok(program_files) = std::env::var("ProgramFiles") {
            candidates.push(format!("{program_files}\\Git\\bin\\bash.exe"));
        }
        if let Ok(program_files_x86) = std::env::var("ProgramFiles(x86)") {
            candidates.push(format!("{program_files_x86}\\Git\\bin\\bash.exe"));
        }
        for candidate in &candidates {
            if path_exists(candidate).await {
                return Ok(bash_shell_config(candidate));
            }
        }
        if let Some(bash) = find_bash_on_path().await {
            return Ok(bash_shell_config(&bash));
        }
        let searched = candidates
            .iter()
            .map(|path| format!("  {path}"))
            .collect::<Vec<_>>()
            .join("\n");
        return Err(ExecutionError::new(
            ExecutionErrorCode::ShellUnavailable,
            format!(
                "No bash shell found. Options:\n  1. Install Git for Windows: https://git-scm.com/download/win\n  2. Add your bash to PATH (Cygwin, MSYS2, etc.)\n  3. Configure an explicit shellPath\n\nSearched Git Bash in:\n{searched}"
            ),
        ));
    }

    if path_exists("/bin/bash").await {
        return Ok(bash_shell_config("/bin/bash"));
    }
    if let Some(bash) = find_bash_on_path().await {
        return Ok(bash_shell_config(&bash));
    }
    Ok(ShellConfig {
        shell: "sh".to_owned(),
        args: vec!["-c".to_owned()],
        command_transport: CommandTransport::Argv,
    })
}

fn shell_env(
    base_env: Option<&HashMap<String, String>>,
    extra_env: Option<&HashMap<String, String>>,
    inherit_env: bool,
) -> HashMap<String, String> {
    let mut env: HashMap<String, String> = HashMap::new();
    if inherit_env {
        env.extend(
            std::env::vars_os().filter_map(|(key, value)| {
                Some((key.into_string().ok()?, value.into_string().ok()?))
            }),
        );
    }
    if let Some(base_env) = base_env {
        env.extend(
            base_env
                .iter()
                .map(|(key, value)| (key.clone(), value.clone())),
        );
    }
    if let Some(extra_env) = extra_env {
        env.extend(
            extra_env
                .iter()
                .map(|(key, value)| (key.clone(), value.clone())),
        );
    }
    env
}

#[async_trait::async_trait]
impl FileSystem for NodeExecutionEnv {
    fn cwd(&self) -> &str {
        &self.cwd
    }

    async fn absolute_path(
        &self,
        path: &str,
        _signal: Option<&AbortSignal>,
    ) -> Result<String, FileError> {
        self.resolve_path(path)
    }

    async fn join_path(
        &self,
        parts: &[String],
        _signal: Option<&AbortSignal>,
    ) -> Result<String, FileError> {
        let mut joined = PathBuf::new();
        for part in parts {
            joined.push(part);
        }
        Ok(lexical_normalize(&joined).display().to_string())
    }

    async fn read_text_file(
        &self,
        path: &str,
        signal: Option<&AbortSignal>,
    ) -> Result<String, FileError> {
        let resolved = self.resolve_path(path)?;
        if let Some(result) = self.aborted(signal, Some(&resolved)) {
            return result;
        }
        match read_to_string_cancellable(&resolved, signal).await {
            Ok(content) => Ok(content),
            Err(error) => Err(to_file_error(error, Some(&resolved))),
        }
    }

    async fn read_text_lines(
        &self,
        path: &str,
        options: Option<&ReadTextLinesOptions>,
    ) -> Result<Vec<String>, FileError> {
        let resolved = self.resolve_path(path)?;
        let signal = options.and_then(|options| options.signal.as_ref());
        if let Some(result) = self.aborted(signal, Some(&resolved)) {
            return result;
        }
        let max_lines = options.and_then(|options| options.max_lines);
        if max_lines == Some(0) {
            return Ok(Vec::new());
        }

        let file = match tokio::fs::File::open(&resolved).await {
            Ok(file) => file,
            Err(error) => return Err(to_file_error(error, Some(&resolved))),
        };
        let mut lines = tokio::io::BufReader::new(file).lines();
        let mut collected = Vec::new();
        loop {
            if signal.is_some_and(|signal| signal.is_cancelled()) {
                return Err(FileError::new(
                    FileErrorCode::Aborted,
                    "aborted",
                    Some(PathBuf::from(&resolved)),
                ));
            }
            let next = match signal {
                Some(signal) => {
                    tokio::select! {
                        biased;
                        _ = signal.cancelled() => {
                            return Err(FileError::new(
                                FileErrorCode::Aborted,
                                "aborted",
                                Some(PathBuf::from(&resolved)),
                            ));
                        }
                        line = lines.next_line() => line,
                    }
                }
                None => lines.next_line().await,
            };
            match next {
                Ok(Some(line)) => {
                    collected.push(line);
                    if max_lines.is_some_and(|max_lines| collected.len() >= max_lines) {
                        break;
                    }
                }
                Ok(None) => break,
                Err(error) => return Err(to_file_error(error, Some(&resolved))),
            }
        }
        if signal.is_some_and(|signal| signal.is_cancelled()) {
            return Err(FileError::new(
                FileErrorCode::Aborted,
                "aborted",
                Some(PathBuf::from(&resolved)),
            ));
        }
        Ok(collected)
    }

    async fn read_binary_file(
        &self,
        path: &str,
        signal: Option<&AbortSignal>,
    ) -> Result<Vec<u8>, FileError> {
        let resolved = self.resolve_path(path)?;
        if let Some(result) = self.aborted(signal, Some(&resolved)) {
            return result;
        }
        match read_cancelable(&resolved, signal).await {
            Ok(bytes) => Ok(bytes),
            Err(error) => Err(to_file_error(error, Some(&resolved))),
        }
    }

    async fn read_file_prefix(
        &self,
        path: &str,
        max_bytes: usize,
        signal: Option<&AbortSignal>,
    ) -> Result<Vec<u8>, FileError> {
        let resolved = self.resolve_path(path)?;
        let read = async {
            let file = tokio::fs::File::open(&resolved).await?;
            let mut bytes = Vec::new();
            file.take(max_bytes as u64).read_to_end(&mut bytes).await?;
            Ok::<_, io::Error>(bytes)
        };
        tokio::select! {
            _ = async { match signal { Some(signal) => signal.cancelled().await, None => std::future::pending::<()>().await } } => {
                Err(FileError::new(FileErrorCode::Aborted, "aborted", Some(PathBuf::from(&resolved))))
            }
            result = read => result.map_err(|error| to_file_error(error, Some(&resolved))),
        }
    }

    async fn read_text_page(
        &self,
        path: &str,
        options: &crate::harness::types::TextFilePageOptions,
    ) -> Result<crate::harness::types::TextFilePage, FileError> {
        let resolved = self.resolve_path(path)?;
        let read = async {
            let mut file = tokio::fs::File::open(&resolved)
                .await
                .map_err(|error| to_file_error(error, Some(&resolved)))?;
            let mut buffer = [0u8; 8192];
            let mut collector = crate::harness::types::TextPageCollector::new(options);
            loop {
                let count = file
                    .read(&mut buffer)
                    .await
                    .map_err(|error| to_file_error(error, Some(&resolved)))?;
                if count == 0 {
                    return collector.finish();
                }
                collector.accept(&buffer[..count]);
            }
        };
        tokio::select! {
            _ = async { match options.signal.as_ref() { Some(signal) => signal.cancelled().await, None => std::future::pending::<()>().await } } => {
                Err(FileError::new(FileErrorCode::Aborted, "aborted", Some(PathBuf::from(&resolved))))
            }
            result = read => result,
        }
    }

    async fn write_file(
        &self,
        path: &str,
        content: FileContent,
        signal: Option<&AbortSignal>,
    ) -> Result<(), FileError> {
        let resolved = self.resolve_path(path)?;
        if let Some(result) = self.aborted(signal, Some(&resolved)) {
            return result;
        }
        if let Some(parent) = Path::new(&resolved).parent()
            && let Err(error) = tokio::fs::create_dir_all(parent).await
        {
            return Err(to_file_error(error, Some(&resolved)));
        }
        if signal.is_some_and(|signal| signal.is_cancelled()) {
            return Err(FileError::new(
                FileErrorCode::Aborted,
                "aborted",
                Some(PathBuf::from(&resolved)),
            ));
        }
        // Follow existing symlinks while publishing complete file contents.
        let destination = match tokio::fs::canonicalize(&resolved).await {
            Ok(path) => path_string(path)?,
            Err(error) if error.kind() == io::ErrorKind::NotFound => resolved,
            Err(error) => return Err(to_file_error(error, Some(&resolved))),
        };
        self.write_file_atomic(&destination, content, true).await
    }

    async fn append_file(
        &self,
        path: &str,
        content: FileContent,
        _signal: Option<&AbortSignal>,
    ) -> Result<(), FileError> {
        let resolved = self.resolve_path(path)?;
        if let Some(parent) = Path::new(&resolved).parent()
            && let Err(error) = tokio::fs::create_dir_all(parent).await
        {
            return Err(to_file_error(error, Some(&resolved)));
        }
        use tokio::io::AsyncWriteExt;
        let mut file = tokio::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&resolved)
            .await
            .map_err(|error| to_file_error(error, Some(&resolved)))?;
        match content {
            FileContent::Text(text) => file
                .write_all(text.as_bytes())
                .await
                .map_err(|error| to_file_error(error, Some(&resolved))),
            FileContent::Binary(bytes) => file
                .write_all(&bytes)
                .await
                .map_err(|error| to_file_error(error, Some(&resolved))),
        }
    }

    async fn lock_file(
        &self,
        path: &str,
    ) -> Result<Arc<dyn crate::harness::types::FileLease>, FileError> {
        let resolved = self.resolve_path(path)?;
        tokio::task::spawn_blocking(move || {
            let storage_path = canonical_storage_path(Path::new(&resolved))
                .map_err(|error| to_file_error(error, Some(&resolved)))?;
            let mut lock_name = storage_path.as_os_str().to_os_string();
            lock_name.push(".lock");
            let lock_path = PathBuf::from(lock_name);
            if let Some(parent) = lock_path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|error| to_file_error(error, Some(&resolved)))?;
            }
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&lock_path)
                .map_err(|error| to_file_error(error, Some(&resolved)))?;
            file.try_lock().map_err(|error| {
                FileError::with_cause(
                    FileErrorCode::PermissionDenied,
                    format!("Session is already open or cannot be locked: {resolved}"),
                    Some(lock_path),
                    error,
                )
            })?;
            Ok(Arc::new(file) as Arc<dyn crate::harness::types::FileLease>)
        })
        .await
        .map_err(|error| {
            FileError::with_cause(FileErrorCode::Unknown, error.to_string(), None, error)
        })?
    }

    async fn append_file_durable(&self, path: &str, content: FileContent) -> Result<(), FileError> {
        let resolved = self.resolve_path(path)?;
        tokio::task::spawn_blocking(move || {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .append(true)
                .open(&resolved)
                .map_err(|error| to_file_error(error, Some(&resolved)))?;
            let bytes = match &content {
                FileContent::Text(text) => text.as_bytes(),
                FileContent::Binary(bytes) => bytes,
            };
            file.write_all(bytes)
                .and_then(|()| file.sync_all())
                .map_err(|error| to_file_error(error, Some(&resolved)))
        })
        .await
        .map_err(|error| {
            FileError::with_cause(FileErrorCode::Unknown, error.to_string(), None, error)
        })?
    }

    async fn write_file_atomic(
        &self,
        path: &str,
        content: FileContent,
        overwrite: bool,
    ) -> Result<(), FileError> {
        let resolved = self.resolve_path(path)?;
        tokio::task::spawn_blocking(move || {
            use std::io::Write;
            let destination = Path::new(&resolved);
            let parent = destination.parent().ok_or_else(|| {
                FileError::new(FileErrorCode::Invalid, "Missing parent directory", None)
            })?;
            std::fs::create_dir_all(parent)
                .map_err(|error| to_file_error(error, Some(&resolved)))?;
            let mut temp = tempfile::NamedTempFile::new_in(parent)
                .map_err(|error| to_file_error(error, Some(&resolved)))?;
            let bytes = match &content {
                FileContent::Text(text) => text.as_bytes(),
                FileContent::Binary(bytes) => bytes,
            };
            if overwrite {
                match std::fs::metadata(destination) {
                    Ok(metadata) => temp
                        .as_file()
                        .set_permissions(metadata.permissions())
                        .map_err(|error| to_file_error(error, Some(&resolved)))?,
                    Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                    Err(error) => return Err(to_file_error(error, Some(&resolved))),
                }
            }
            temp.write_all(bytes)
                .and_then(|()| temp.as_file().sync_all())
                .map_err(|error| to_file_error(error, Some(&resolved)))?;
            let result = if overwrite {
                temp.persist(destination)
            } else {
                temp.persist_noclobber(destination)
            };
            result.map_err(|error| to_file_error(error.error, Some(&resolved)))?;
            #[cfg(unix)]
            std::fs::File::open(parent)
                .and_then(|dir| dir.sync_all())
                .map_err(|error| to_file_error(error, Some(&resolved)))?;
            Ok(())
        })
        .await
        .map_err(|error| {
            FileError::with_cause(FileErrorCode::Unknown, error.to_string(), None, error)
        })?
    }

    async fn rename_file(
        &self,
        source_path: &str,
        destination_path: &str,
        signal: Option<&AbortSignal>,
    ) -> Result<(), FileError> {
        let source = self.resolve_path(source_path)?;
        let destination = self.resolve_path(destination_path)?;
        if let Some(result) = self.aborted(signal, Some(&destination)) {
            return result;
        }
        tokio::fs::rename(&source, &destination)
            .await
            .map_err(|error| to_file_error(error, Some(&source)))
    }

    async fn file_info(
        &self,
        path: &str,
        _signal: Option<&AbortSignal>,
    ) -> Result<FileInfo, FileError> {
        let resolved = self.resolve_path(path)?;
        match tokio::fs::symlink_metadata(&resolved).await {
            Ok(metadata) => file_info_from_metadata(&resolved, metadata),
            Err(error) => Err(to_file_error(error, Some(&resolved))),
        }
    }

    async fn list_dir(
        &self,
        path: &str,
        signal: Option<&AbortSignal>,
    ) -> Result<Vec<FileInfo>, FileError> {
        let resolved = self.resolve_path(path)?;
        if let Some(result) = self.aborted(signal, Some(&resolved)) {
            return result;
        }
        let mut entries = match tokio::fs::read_dir(&resolved).await {
            Ok(entries) => entries,
            Err(error) => return Err(to_file_error(error, Some(&resolved))),
        };
        let mut infos = Vec::new();
        loop {
            if signal.is_some_and(|signal| signal.is_cancelled()) {
                return Err(FileError::new(
                    FileErrorCode::Aborted,
                    "aborted",
                    Some(PathBuf::from(&resolved)),
                ));
            }
            let next = match signal {
                Some(signal) => {
                    tokio::select! {
                        biased;
                        _ = signal.cancelled() => {
                            return Err(FileError::new(
                                FileErrorCode::Aborted,
                                "aborted",
                                Some(PathBuf::from(&resolved)),
                            ));
                        }
                        entry = entries.next_entry() => entry,
                    }
                }
                None => entries.next_entry().await,
            };
            match next {
                Ok(Some(entry)) => {
                    let entry_path =
                        lexical_normalize(&Path::new(&resolved).join(entry.file_name()))
                            .display()
                            .to_string();
                    match tokio::fs::symlink_metadata(&entry_path).await {
                        Ok(metadata) => infos.push(file_info_from_metadata(&entry_path, metadata)?),
                        Err(error) => return Err(to_file_error(error, Some(&entry_path))),
                    }
                }
                Ok(None) => break,
                Err(error) => return Err(to_file_error(error, Some(&resolved))),
            }
        }
        Ok(infos)
    }

    async fn canonical_path(
        &self,
        path: &str,
        _signal: Option<&AbortSignal>,
    ) -> Result<String, FileError> {
        let resolved = self.resolve_path(path)?;
        match tokio::fs::canonicalize(&resolved).await {
            Ok(canonical) => path_string(canonical),
            Err(error) => Err(to_file_error(error, Some(&resolved))),
        }
    }

    async fn exists(&self, path: &str, _signal: Option<&AbortSignal>) -> Result<bool, FileError> {
        match self.file_info(path, None).await {
            Ok(_) => Ok(true),
            Err(error) if error.code == FileErrorCode::NotFound => Ok(false),
            Err(error) => Err(error),
        }
    }

    async fn create_dir(
        &self,
        path: &str,
        options: Option<&CreateDirOptions>,
    ) -> Result<(), FileError> {
        let resolved = self.resolve_path(path)?;
        let recursive = options
            .and_then(|options| options.recursive)
            .unwrap_or(true);
        let result = if recursive {
            tokio::fs::create_dir_all(&resolved).await
        } else {
            tokio::fs::create_dir(&resolved).await
        };
        result.map_err(|error| to_file_error(error, Some(&resolved)))
    }

    async fn remove(&self, path: &str, options: Option<&RemoveOptions>) -> Result<(), FileError> {
        let resolved = self.resolve_path(path)?;
        let recursive = options
            .and_then(|options| options.recursive)
            .unwrap_or(false);
        let force = options.and_then(|options| options.force).unwrap_or(false);
        let result = if recursive {
            tokio::fs::remove_dir_all(&resolved).await
        } else {
            tokio::fs::remove_file(&resolved).await
        };
        match result {
            Ok(()) => Ok(()),
            Err(error) => {
                if force && error.kind() == io::ErrorKind::NotFound {
                    Ok(())
                } else {
                    Err(to_file_error(error, Some(&resolved)))
                }
            }
        }
    }

    async fn create_temp_dir(
        &self,
        prefix: Option<&str>,
        signal: Option<&AbortSignal>,
    ) -> Result<String, FileError> {
        if let Some(result) = self.aborted(signal, None) {
            return result;
        }
        let prefix = prefix.unwrap_or("tmp-").to_owned();
        tokio::task::spawn_blocking(move || {
            let dir = tempfile::Builder::new()
                .prefix(&prefix)
                .tempdir()
                .map_err(|error| to_file_error(error, None))?;
            path_string(dir.keep())
        })
        .await
        .map_err(|error| {
            FileError::with_cause(FileErrorCode::Unknown, error.to_string(), None, error)
        })?
    }

    async fn create_temp_file(
        &self,
        options: Option<&CreateTempFileOptions>,
    ) -> Result<String, FileError> {
        if let Some(result) = self.aborted(options.and_then(|o| o.signal.as_ref()), None) {
            return result;
        }
        let prefix = options.and_then(|o| o.prefix.clone()).unwrap_or_default();
        let suffix = options.and_then(|o| o.suffix.clone()).unwrap_or_default();
        tokio::task::spawn_blocking(move || {
            let file = tempfile::Builder::new()
                .prefix(&prefix)
                .suffix(&suffix)
                .tempfile()
                .map_err(|error| to_file_error(error, None))?;
            let (_, path) = file
                .keep()
                .map_err(|error| to_file_error(error.error, None))?;
            path_string(path)
        })
        .await
        .map_err(|error| {
            FileError::with_cause(FileErrorCode::Unknown, error.to_string(), None, error)
        })?
    }

    async fn cleanup(&self) {
        for token in self
            .active_commands
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .values()
        {
            token.cancel();
        }
    }
}

#[async_trait::async_trait]
impl Shell for NodeExecutionEnv {
    async fn exec(
        &self,
        command: &str,
        options: Option<&ShellExecOptions>,
    ) -> Result<ShellExecResult, ExecutionError> {
        if options
            .and_then(|options| options.signal.as_ref())
            .is_some_and(|signal| signal.is_cancelled())
        {
            return Err(ExecutionError::new(ExecutionErrorCode::Aborted, "aborted"));
        }
        let timeout = options.and_then(|options| options.timeout);
        let timeout_duration = resolve_timeout(timeout)?;
        let signal = options.and_then(|options| options.signal.clone());

        let cwd = match options.and_then(|options| options.cwd.clone()) {
            Some(cwd) => self.resolve_path(&cwd).map_err(|error| {
                ExecutionError::with_cause(
                    ExecutionErrorCode::SpawnError,
                    error.message.clone(),
                    error,
                )
            })?,
            None => self.cwd.clone(),
        };
        let shell_config = get_shell_config(self.shell_path.as_deref()).await?;

        if !path_exists(&cwd).await {
            return Err(ExecutionError::new(
                ExecutionErrorCode::SpawnError,
                format!("Working directory does not exist: {cwd}\nCannot execute bash commands."),
            ));
        }

        let from_stdin = shell_config.command_transport == CommandTransport::Stdin;
        let mut command_builder = Command::new(&shell_config.shell);
        command_builder.args(&shell_config.args);
        if !from_stdin {
            command_builder.arg(command);
        }
        if !options
            .and_then(|options| options.inherit_env)
            .unwrap_or(true)
        {
            command_builder.env_clear();
        }
        command_builder
            .current_dir(&cwd)
            .envs(shell_env(
                self.shell_env.as_ref(),
                options.and_then(|options| options.env.as_ref()),
                options
                    .and_then(|options| options.inherit_env)
                    .unwrap_or(true),
            ))
            .stdin(if from_stdin {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut wrapped = CommandWrap::from(command_builder);
        wrapped.wrap(KillOnDrop);
        #[cfg(unix)]
        {
            wrapped.wrap(process_wrap::tokio::ProcessGroup::leader());
        }
        #[cfg(windows)]
        {
            wrapped.wrap(process_wrap::tokio::CreationFlags(
                windows::Win32::System::Threading::CREATE_NO_WINDOW,
            )); // CREATE_NO_WINDOW
            wrapped.wrap(process_wrap::tokio::JobObject);
        }
        let child = wrapped.spawn().map_err(|error| {
            ExecutionError::with_cause(ExecutionErrorCode::SpawnError, error.to_string(), error)
        })?;
        let abort_token = CancellationToken::new();
        let id = uuid::Uuid::now_v7();
        self.active_commands
            .lock()
            .expect("command registry poisoned")
            .insert(id, abort_token.clone());
        let mut process = ProcessGuard {
            child,
            commands: &self.active_commands,
            id,
            reaped: false,
        };
        let mut stdin = process.child.stdin().take();
        let stdout = process.child.stdout().take();
        let stderr = process.child.stderr().take();
        let stdout_accumulator = Arc::new(Mutex::new(String::new()));
        let stderr_accumulator = Arc::new(Mutex::new(String::new()));
        let last_data = Arc::new(Mutex::new(std::time::Instant::now()));
        let callback_error = Arc::new(Mutex::new(None::<ExecutionError>));
        let read_cancel = CancellationToken::new();
        let mut readers = tokio::task::JoinSet::new();
        for (stream, accumulator, callback) in [
            (
                stdout.map(|pipe| Box::new(pipe) as Box<dyn tokio::io::AsyncRead + Unpin + Send>),
                stdout_accumulator.clone(),
                options.and_then(|o| o.on_stdout.clone()),
            ),
            (
                stderr.map(|pipe| Box::new(pipe) as Box<dyn tokio::io::AsyncRead + Unpin + Send>),
                stderr_accumulator.clone(),
                options.and_then(|o| o.on_stderr.clone()),
            ),
        ] {
            if let Some(stream) = stream {
                readers.spawn(
                    PipeReader {
                        stream,
                        accumulator,
                        callback,
                        callback_error: callback_error.clone(),
                        last_data: last_data.clone(),
                        read_cancel: read_cancel.clone(),
                        abort_token: abort_token.clone(),
                    }
                    .run(),
                );
            }
        }
        let deadline = async {
            match timeout_duration {
                Some(duration) => tokio::time::sleep(duration).await,
                None => std::future::pending::<()>().await,
            }
        };
        let cancelled = async {
            match &signal {
                Some(signal) => signal.cancelled().await,
                None => std::future::pending::<()>().await,
            }
        };
        let mut failure = None;
        let status = tokio::select! {
            biased;
            _ = cancelled => { failure = Some(ExecutionErrorCode::Aborted); None }
            _ = abort_token.cancelled() => { failure = Some(ExecutionErrorCode::Aborted); None }
            _ = deadline => { failure = Some(ExecutionErrorCode::Timeout); None }
            result = async {
                if let Some(stdin) = stdin.as_mut() {
                    use tokio::io::AsyncWriteExt;
                    stdin.write_all(command.as_bytes()).await?;
                }
                drop(stdin.take());
                process.child.wait().await
            } => Some(result),
        };
        let status = match status {
            Some(Ok(status)) => {
                process.reaped = true;
                Some(status)
            }
            Some(Err(error)) => {
                failure = Some(ExecutionErrorCode::Unknown);
                *callback_error.lock().expect("callback error poisoned") =
                    Some(ExecutionError::with_cause(
                        ExecutionErrorCode::Unknown,
                        error.to_string(),
                        error,
                    ));
                let _ = process.child.start_kill();
                process.reaped = process.child.wait().await.is_ok();
                None
            }
            None => {
                let _ = process.child.start_kill();
                let status = process.child.wait().await.ok();
                process.reaped = status.is_some();
                status
            }
        };
        // Drain output while it remains active, with a fixed upper bound after exit.
        let drain_started = std::time::Instant::now();
        while !readers.is_empty() {
            let idle_deadline = (*last_data.lock().expect("last data poisoned")).max(drain_started)
                + EXIT_STDIO_GRACE;
            let deadline = idle_deadline.min(drain_started + Duration::from_secs(1));
            tokio::select! {
                _ = tokio::time::sleep_until(deadline.into()) => {
                    read_cancel.cancel();
                    readers.abort_all();
                    break;
                }
                result = readers.join_next() => {
                    if let Some(Err(error)) = result {
                        *callback_error.lock().expect("callback error poisoned") = Some(ExecutionError::with_cause(
                            ExecutionErrorCode::Unknown, error.to_string(), error,
                        ));
                    }
                }
            }
        }
        if let Some(error) = callback_error
            .lock()
            .expect("callback error poisoned")
            .take()
        {
            return Err(error);
        }
        if let Some(code) = failure {
            return Err(ExecutionError::new(
                code,
                if code == ExecutionErrorCode::Timeout {
                    format!(
                        "Command timed out after {} seconds",
                        timeout.unwrap_or_default()
                    )
                } else {
                    "aborted".to_owned()
                },
            ));
        }
        let status = status.ok_or_else(|| {
            ExecutionError::new(ExecutionErrorCode::Unknown, "Missing process status")
        })?;
        let exit_code = process_exit_code(status);
        let stdout = stdout_accumulator.lock().expect("stdout poisoned").clone();
        let stderr = stderr_accumulator.lock().expect("stderr poisoned").clone();
        Ok(ShellExecResult {
            stdout,
            stderr,
            exit_code,
        })
    }

    async fn cleanup(&self) {
        <Self as FileSystem>::cleanup(self).await;
    }
}

fn process_exit_code(status: std::process::ExitStatus) -> i32 {
    if let Some(code) = status.code() {
        return code;
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        return 128 + status.signal().unwrap_or(1);
    }
    #[cfg(not(unix))]
    {
        1
    }
}

struct PipeReader {
    stream: Box<dyn tokio::io::AsyncRead + Unpin + Send>,
    accumulator: Arc<Mutex<String>>,
    callback: Option<crate::harness::types::ShellChunkCallback>,
    callback_error: Arc<Mutex<Option<ExecutionError>>>,
    last_data: Arc<Mutex<std::time::Instant>>,
    read_cancel: CancellationToken,
    abort_token: CancellationToken,
}

impl PipeReader {
    async fn run(mut self) {
        let mut buffer = [0u8; 8192];
        let mut decoder = encoding_rs::UTF_8.new_decoder_without_bom_handling();
        loop {
            let result = tokio::select! {
                biased;
                _ = self.read_cancel.cancelled() => return,
                _ = self.abort_token.cancelled() => return,
                read = self.stream.read(&mut buffer) => read,
            };
            let read = match result {
                Ok(read) => read,
                Err(error) => {
                    *self.callback_error.lock().expect("callback error poisoned") =
                        Some(ExecutionError::with_cause(
                            ExecutionErrorCode::Unknown,
                            error.to_string(),
                            error,
                        ));
                    self.abort_token.cancel();
                    return;
                }
            };
            let mut chunk =
                String::with_capacity(decoder.max_utf8_buffer_length(read).unwrap_or(32768));
            let (_, consumed, _) = decoder.decode_to_string(&buffer[..read], &mut chunk, read == 0);
            debug_assert_eq!(consumed, read);
            if !chunk.is_empty() {
                *self.last_data.lock().expect("last data poisoned") = std::time::Instant::now();
                // Streaming callers own capture/spooling; avoid duplicating the full output.
                if let Some(callback) = &self.callback {
                    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| callback(&chunk)))
                        .is_err()
                    {
                        *self.callback_error.lock().expect("callback error poisoned") =
                            Some(ExecutionError::new(
                                ExecutionErrorCode::CallbackError,
                                "callback error",
                            ));
                        self.abort_token.cancel();
                        return;
                    }
                } else {
                    let mut output = self.accumulator.lock().expect("accumulator poisoned");
                    if output.len().saturating_add(chunk.len()) > 16 * 1024 * 1024 {
                        *self.callback_error.lock().expect("callback error poisoned") =
                            Some(ExecutionError::new(
                                ExecutionErrorCode::Unknown,
                                "Captured output exceeds 16 MiB; use streaming callbacks",
                            ));
                        self.abort_token.cancel();
                        return;
                    }
                    output.push_str(&chunk);
                }
            }
            if read == 0 {
                return;
            }
        }
    }
}

async fn read_to_string_cancellable(
    path: &str,
    signal: Option<&AbortSignal>,
) -> io::Result<String> {
    match signal {
        Some(signal) => {
            let read = tokio::fs::read_to_string(path);
            tokio::select! {
                biased;
                _ = signal.cancelled() => Err(io::Error::new(io::ErrorKind::Interrupted, "aborted")),
                result = read => result,
            }
        }
        None => tokio::fs::read_to_string(path).await,
    }
}

async fn read_cancelable(path: &str, signal: Option<&AbortSignal>) -> io::Result<Vec<u8>> {
    match signal {
        Some(signal) => {
            let read = tokio::fs::read(path);
            tokio::select! {
                biased;
                _ = signal.cancelled() => Err(io::Error::new(io::ErrorKind::Interrupted, "aborted")),
                result = read => result,
            }
        }
        None => tokio::fs::read(path).await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Expected path text with platform separators applied.
    fn expected(path: &str) -> String {
        if cfg!(windows) {
            path.replace('/', "\\")
        } else {
            path.to_owned()
        }
    }

    #[test]
    fn resolves_relative_and_absolute_paths() {
        assert_eq!(resolve_path("/work", "a/b").unwrap(), expected("/work/a/b"));
        assert_eq!(resolve_path("/work", "/abs/x").unwrap(), expected("/abs/x"));
        assert_eq!(
            resolve_path("/work", "./a/../b").unwrap(),
            expected("/work/b")
        );
        assert_eq!(
            resolve_path("/work", "a//b").unwrap(),
            expected("/work/a/b")
        );
    }

    #[test]
    fn resolves_home_and_file_urls() {
        let home = home_dir();
        assert_eq!(resolve_path("/work", "~").unwrap(), home);
        assert!(resolve_path("/work", "~/x").unwrap().starts_with(&home));
        #[cfg(unix)]
        assert_eq!(
            resolve_path("/work", "file:///tmp/x").unwrap(),
            expected("/tmp/x")
        );
        #[cfg(windows)]
        assert_eq!(
            resolve_path("/work", "file:///C:/tmp/x").unwrap(),
            expected("C:/tmp/x")
        );
    }

    #[test]
    fn detects_legacy_wsl_bash() {
        assert!(is_legacy_wsl_bash_path(r"C:\Windows\System32\bash.exe"));
        assert!(is_legacy_wsl_bash_path(r"c:\windows\sysnative\bash.exe"));
        assert!(!is_legacy_wsl_bash_path(
            r"C:\Program Files\Git\bin\bash.exe"
        ));
        assert!(!is_legacy_wsl_bash_path("/bin/bash"));
    }

    #[test]
    fn merges_shell_environments() {
        let mut base = HashMap::new();
        base.insert("A".to_owned(), "base".to_owned());
        let mut extra = HashMap::new();
        extra.insert("A".to_owned(), "extra".to_owned());
        extra.insert("B".to_owned(), "b".to_owned());

        let inherited = shell_env(Some(&base), Some(&extra), true);
        assert_eq!(inherited.get("A").map(String::as_str), Some("extra"));
        assert_eq!(inherited.get("B").map(String::as_str), Some("b"));

        let isolated = shell_env(Some(&base), Some(&extra), false);
        assert_eq!(isolated.get("A").map(String::as_str), Some("extra"));
        assert_eq!(isolated.get("B").map(String::as_str), Some("b"));
        assert_eq!(isolated.len(), 2);
    }

    #[tokio::test]
    async fn executes_commands_and_captures_output() {
        let env = NodeExecutionEnv::new(NodeExecutionEnvOptions {
            cwd: Some(std::env::temp_dir().display().to_string()),
            ..Default::default()
        });
        let command = "echo hello";
        let result = env.exec(command, None).await;
        match result {
            Ok(result) => {
                assert!(result.stdout.contains("hello"), "stdout: {}", result.stdout);
                assert_eq!(result.exit_code, 0);
            }
            Err(error) if error.code == ExecutionErrorCode::ShellUnavailable => {
                eprintln!("Skipping shell test: {error}")
            }
            Err(error) => panic!("Shell execution failed: {error}"),
        }
    }

    #[tokio::test]
    async fn reports_nonzero_exit_codes() {
        let env = NodeExecutionEnv::new(NodeExecutionEnvOptions::default());
        let command = "exit 3";
        match env.exec(command, None).await {
            Ok(result) => assert_eq!(result.exit_code, 3),
            Err(error) if error.code == ExecutionErrorCode::ShellUnavailable => {
                eprintln!("Skipping shell test: {error}")
            }
            Err(error) => panic!("Shell execution failed: {error}"),
        }
    }

    #[tokio::test]
    async fn aborted_commands_return_aborted_errors() {
        let env = NodeExecutionEnv::new(NodeExecutionEnvOptions::default());
        let signal = CancellationToken::new();
        signal.cancel();
        let command = "echo hi";
        let error = env
            .exec(
                command,
                Some(&ShellExecOptions {
                    signal: Some(signal),
                    ..Default::default()
                }),
            )
            .await
            .unwrap_err();
        assert_eq!(error.code, ExecutionErrorCode::Aborted);
    }

    #[tokio::test]
    async fn decodes_invalid_split_and_incomplete_utf8_without_stalling() {
        let (mut writer, reader) = tokio::io::duplex(16);
        let accumulator = Arc::new(Mutex::new(String::new()));
        let errors = Arc::new(Mutex::new(None));
        let task = tokio::spawn(
            PipeReader {
                stream: Box::new(reader),
                accumulator: accumulator.clone(),
                callback: None,
                callback_error: errors.clone(),
                last_data: Arc::new(Mutex::new(std::time::Instant::now())),
                read_cancel: CancellationToken::new(),
                abort_token: CancellationToken::new(),
            }
            .run(),
        );
        use tokio::io::AsyncWriteExt;
        writer.write_all(&[0xff, b'A', 0xe4]).await.unwrap();
        tokio::task::yield_now().await;
        writer.write_all(&[0xb8, 0xad, b'B', 0xe2]).await.unwrap();
        drop(writer);
        task.await.unwrap();
        assert_eq!(&*accumulator.lock().unwrap(), "\u{fffd}A中B\u{fffd}");
        assert!(errors.lock().unwrap().is_none());
    }

    #[tokio::test]
    async fn streaming_capture_does_not_duplicate_output_in_the_return_value() {
        let (mut writer, reader) = tokio::io::duplex(16);
        let accumulator = Arc::new(Mutex::new(String::new()));
        let capture = Arc::new(Mutex::new(String::new()));
        let callback_capture = capture.clone();
        let task = tokio::spawn(
            PipeReader {
                stream: Box::new(reader),
                accumulator: accumulator.clone(),
                callback: Some(Arc::new(move |text| {
                    callback_capture.lock().unwrap().push_str(text)
                })),
                callback_error: Arc::new(Mutex::new(None)),
                last_data: Arc::new(Mutex::new(std::time::Instant::now())),
                read_cancel: CancellationToken::new(),
                abort_token: CancellationToken::new(),
            }
            .run(),
        );
        use tokio::io::AsyncWriteExt;
        writer.write_all(b"captured").await.unwrap();
        drop(writer);
        task.await.unwrap();
        assert!(accumulator.lock().unwrap().is_empty());
        assert_eq!(&*capture.lock().unwrap(), "captured");
    }

    #[tokio::test]
    async fn timeout_kills_the_process_and_releases_its_registration() {
        let env = NodeExecutionEnv::new(NodeExecutionEnvOptions::default());
        let result = env
            .exec(
                "sleep 10",
                Some(&ShellExecOptions {
                    timeout: Some(0.05),
                    ..Default::default()
                }),
            )
            .await;
        match result {
            Err(error) if error.code == ExecutionErrorCode::ShellUnavailable => {
                eprintln!("Skipping shell test: {error}");
                return;
            }
            Err(error) => assert_eq!(error.code, ExecutionErrorCode::Timeout),
            Ok(_) => panic!("sleep completed before its timeout"),
        }
        assert!(env.active_commands.lock().unwrap().is_empty());
        assert_eq!(env.exec("echo next", None).await.unwrap().exit_code, 0);
    }

    #[tokio::test]
    async fn text_pages_bound_memory_even_for_a_very_long_line() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("large.txt");
        std::fs::write(&path, format!("prefix\n{}\nlast\n", "x".repeat(100000))).unwrap();
        let env = NodeExecutionEnv::new(NodeExecutionEnvOptions::default());
        let page = env
            .read_text_page(
                &path.display().to_string(),
                &crate::harness::types::TextFilePageOptions {
                    offset: 1,
                    limit: Some(1),
                    max_bytes: 32,
                    signal: None,
                },
            )
            .await
            .unwrap();
        assert_eq!(page.total_lines, 4);
        assert_eq!(page.selected_lines, 1);
        assert!(page.has_more);
        assert!(page.byte_truncated);
        assert!(page.content.len() <= 36);
        assert_eq!(page.first_line_bytes, 100000);
    }

    #[test]
    fn lexical_parent_components_do_not_cancel_each_other() {
        assert_eq!(
            lexical_normalize(Path::new("../../a")),
            PathBuf::from("../../a")
        );
        #[cfg(unix)]
        assert_eq!(
            lexical_normalize(Path::new("/../../a")),
            PathBuf::from("/a")
        );
        #[cfg(windows)]
        {
            assert_eq!(
                resolve_path("C:/work", "file://server/share/file").unwrap(),
                r"\\server\share\file"
            );
        }
        assert!(file_url_to_path("file:not-a-path").is_err());
    }

    #[tokio::test]
    async fn file_roundtrip_and_metadata() {
        let env = NodeExecutionEnv::new(NodeExecutionEnvOptions::default());
        let dir = std::env::temp_dir().join(format!("node-env-test-{}", uuid::Uuid::new_v4()));
        let file = dir.join("nested").join("file.txt");
        let file_path = file.display().to_string();

        env.write_file(&file_path, FileContent::Text("hello".to_owned()), None)
            .await
            .unwrap();
        let content = env.read_text_file(&file_path, None).await.unwrap();
        assert_eq!(content, "hello");

        let info = env.file_info(&file_path, None).await.unwrap();
        assert_eq!(info.kind, FileKind::File);
        assert_eq!(info.size, 5);
        assert_eq!(info.name, "file.txt");

        let lines = env
            .read_text_lines(&file_path, Some(&ReadTextLinesOptions::default()))
            .await
            .unwrap();
        assert_eq!(lines, vec!["hello".to_owned()]);

        assert!(env.exists(&file_path, None).await.unwrap());
        assert!(
            !env.exists(&dir.join("missing.txt").display().to_string(), None)
                .await
                .unwrap()
        );

        env.remove(
            &dir.display().to_string(),
            Some(&RemoveOptions {
                recursive: Some(true),
                force: Some(true),
                signal: None,
            }),
        )
        .await
        .unwrap();
        assert!(!env.exists(&file_path, None).await.unwrap());
    }
}
