//! JSONL session repository: cwd-encoded directory layout and lifecycle.
//!
//! Ported from `@step-harness/agent-core` `src\harness\session\jsonl\repo.ts`.

use std::collections::HashSet;
use std::sync::Arc;

use regex::Regex;
use step_harness_providers::utils::now_ms;
use tokio::sync::Mutex;

use crate::harness::types::{ExecutionEnv, FileKind};
use crate::session::facade::{Session, SessionStorage};
use crate::session::jsonl::codec::{metadata_from_header, parse_header};
use crate::session::jsonl::storage::JsonlSessionStorage;
use crate::session::jsonl::types::{
    JsonlSessionCreateOptions, JsonlSessionListOptions, JsonlSessionMetadata,
    JsonlSessionRepoOptions, JsonlV4Header, file_result,
};
use crate::session::types::{
    ForkOptions, SessionError, SessionErrorCode, assert_json_serializable,
};

static SESSION_ID_PATTERN: std::sync::LazyLock<Regex> = std::sync::LazyLock::new(|| {
    Regex::new(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$").expect("valid session id pattern")
});

fn validate_session_id(id: &str) -> Result<(), SessionError> {
    if !SESSION_ID_PATTERN.is_match(id) {
        return Err(SessionError::new(
            SessionErrorCode::InvalidPayload,
            "Session id must be non-empty, contain only alphanumeric characters, '-', '_', and '.', and start and end with an alphanumeric character",
        ));
    }
    Ok(())
}

/// Encode a cwd as the coding-agent-compatible session directory name.
pub fn jsonl_session_directory_name(cwd: &str) -> String {
    let stripped = cwd
        .strip_prefix('/')
        .or_else(|| cwd.strip_prefix('\\'))
        .unwrap_or(cwd);
    let encoded: String = stripped
        .chars()
        .map(|character| match character {
            '/' | '\\' | ':' => '-',
            other => other,
        })
        .collect();
    format!("--{encoded}--")
}

/// Format an epoch-milliseconds timestamp as the session file name timestamp.
fn iso_timestamp(epoch_ms: u64) -> String {
    let date = time::OffsetDateTime::from_unix_timestamp_nanos(i128::from(epoch_ms) * 1_000_000)
        .unwrap_or(time::OffsetDateTime::UNIX_EPOCH);
    format!(
        "{:04}-{:02}-{:02}T{:02}-{:02}-{:02}-{:03}Z",
        date.year(),
        date.month() as u8,
        date.day(),
        date.hour(),
        date.minute(),
        date.second(),
        date.millisecond()
    )
}

/// The file name of a session: `<ISO timestamp>_<id>.jsonl`.
pub fn session_file_name(created_at: u64, id: &str) -> String {
    format!(
        "{}_{id}.jsonl",
        iso_timestamp(created_at).replace([':', '.'], "-")
    )
}

/// List session metadata under a sessions root.
pub async fn list_jsonl_session_metadata(
    fs: &Arc<dyn ExecutionEnv>,
    sessions_root: &str,
    query: &JsonlSessionListOptions,
) -> Result<Vec<JsonlSessionMetadata>, SessionError> {
    let mut metadata = Vec::new();
    for directory in jsonl_session_directories(fs, sessions_root, query.cwd.as_deref()).await? {
        let files = file_result(
            fs.list_dir(&directory, None).await,
            &format!("Failed to list sessions directory {directory}"),
        )?
        .into_iter()
        .filter(|entry| entry.kind != FileKind::Directory && entry.name.ends_with(".jsonl"))
        .collect::<Vec<_>>();
        for file in files {
            let lines = file_result(
                fs.read_text_lines(
                    &file.path,
                    Some(&crate::harness::types::ReadTextLinesOptions {
                        max_lines: Some(1),
                        signal: None,
                    }),
                )
                .await,
                &format!("Failed to read session header {}", file.path),
            )?;
            let Some(first_line) = lines.first() else {
                continue;
            };
            let Ok(header) = parse_header(first_line) else {
                continue;
            };
            metadata.push(metadata_from_header(&header, &file.path, file.mtime_ms));
        }
    }
    metadata.sort_by_key(|entry| std::cmp::Reverse(entry.modified_at));
    Ok(metadata)
}

/// Load a session's storage, verifying the header id matches the metadata.
pub async fn load_jsonl_session_storage(
    fs: &Arc<dyn ExecutionEnv>,
    metadata: &JsonlSessionMetadata,
) -> Result<JsonlSessionStorage, SessionError> {
    if !file_result(
        fs.exists(&metadata.path, None).await,
        &format!("Failed to check session {}", metadata.path),
    )? {
        return Err(SessionError::new(
            SessionErrorCode::NotFound,
            format!("Session not found: {}", metadata.id),
        ));
    }
    let storage = JsonlSessionStorage::load(fs.clone(), &metadata.path).await?;
    let loaded = storage.get_metadata().await;
    if loaded.id != metadata.id {
        return Err(SessionError::new(
            SessionErrorCode::InvalidEntry,
            format!("Session id does not match header: {}", metadata.id),
        ));
    }
    Ok(storage)
}

async fn jsonl_session_directories(
    fs: &Arc<dyn ExecutionEnv>,
    sessions_root: &str,
    cwd: Option<&str>,
) -> Result<Vec<String>, SessionError> {
    let root = file_result(
        fs.absolute_path(sessions_root, None).await,
        &format!("Failed to resolve sessions root {sessions_root}"),
    )?;
    if let Some(cwd) = cwd {
        let resolved_cwd = file_result(
            fs.absolute_path(cwd, None).await,
            &format!("Failed to resolve session cwd {cwd}"),
        )?;
        let directory = file_result(
            fs.join_path(&[root, jsonl_session_directory_name(&resolved_cwd)], None)
                .await,
            &format!("Failed to resolve sessions directory for {resolved_cwd}"),
        )?;
        let exists = file_result(
            fs.exists(&directory, None).await,
            &format!("Failed to check sessions directory {directory}"),
        )?;
        return Ok(if exists { vec![directory] } else { Vec::new() });
    }
    if !file_result(
        fs.exists(&root, None).await,
        &format!("Failed to check sessions directory {root}"),
    )? {
        return Ok(Vec::new());
    }
    Ok(file_result(
        fs.list_dir(&root, None).await,
        &format!("Failed to list sessions directory {root}"),
    )?
    .into_iter()
    .filter(|entry| entry.kind == FileKind::Directory || entry.kind == FileKind::Symlink)
    .map(|entry| entry.path)
    .collect())
}

/// Repository of JSONL-backed sessions.
#[derive(Clone)]
pub struct JsonlSessionRepo {
    fs: Arc<dyn ExecutionEnv>,
    sessions_root_input: String,
    active_create_destinations: Arc<std::sync::Mutex<HashSet<String>>>,
    root: Arc<Mutex<Option<String>>>,
}

struct CreateClaim<'a> {
    destinations: &'a std::sync::Mutex<HashSet<String>>,
    key: String,
}
impl Drop for CreateClaim<'_> {
    fn drop(&mut self) {
        self.destinations
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .remove(&self.key);
    }
}

impl JsonlSessionRepo {
    pub fn new(fs: Arc<dyn ExecutionEnv>, options: JsonlSessionRepoOptions) -> Self {
        JsonlSessionRepo {
            fs,
            sessions_root_input: options.sessions_root,
            active_create_destinations: Arc::new(std::sync::Mutex::new(HashSet::new())),
            root: Arc::new(Mutex::new(None)),
        }
    }

    pub async fn create(
        &self,
        options: &JsonlSessionCreateOptions,
    ) -> Result<Session<JsonlSessionMetadata>, SessionError> {
        let repo = self.clone();
        let options = options.clone();
        tokio::spawn(async move { repo.create_inner(&options).await })
            .await
            .map_err(|error| {
                SessionError::with_cause(
                    SessionErrorCode::Storage,
                    "Session create task failed",
                    error,
                )
            })?
    }

    async fn create_inner(
        &self,
        options: &JsonlSessionCreateOptions,
    ) -> Result<Session<JsonlSessionMetadata>, SessionError> {
        let destination = self.resolve_create_destination(options).await?;
        self.claim_create_destination(&destination, async {
            let (header, path) = self.prepare_create(&destination, options).await?;
            let storage = JsonlSessionStorage::create(self.fs.clone(), &path, &header).await?;
            Ok(Session::new(Arc::new(storage), None))
        })
        .await
    }

    pub async fn open(
        &self,
        metadata: &JsonlSessionMetadata,
    ) -> Result<Session<JsonlSessionMetadata>, SessionError> {
        let storage = self.load_storage(metadata).await?;
        Ok(Session::new(Arc::new(storage), None))
    }

    pub async fn list(
        &self,
        options: &JsonlSessionListOptions,
    ) -> Result<Vec<JsonlSessionMetadata>, SessionError> {
        let root = self.root().await?;
        list_jsonl_session_metadata(&self.fs, &root, options).await
    }

    pub async fn delete(&self, metadata: &JsonlSessionMetadata) -> Result<(), SessionError> {
        let fs = self.fs.clone();
        let path = metadata.path.clone();
        tokio::spawn(async move {
            let _lease = file_result(fs.lock_file(&path).await, "Cannot delete an open session")?;
            file_result(
                fs.remove(&path, None).await,
                &format!("Failed to delete session {path}"),
            )
        })
        .await
        .map_err(|error| {
            SessionError::with_cause(
                SessionErrorCode::Storage,
                "Session delete task failed",
                error,
            )
        })?
    }

    pub async fn fork(
        &self,
        source: &JsonlSessionMetadata,
        options: &ForkOptions,
        create_options: &JsonlSessionCreateOptions,
    ) -> Result<Session<JsonlSessionMetadata>, SessionError> {
        let repo = self.clone();
        let source = source.clone();
        let options = options.clone();
        let create_options = create_options.clone();
        tokio::spawn(async move { repo.fork_inner(&source, &options, &create_options).await })
            .await
            .map_err(|error| {
                SessionError::with_cause(
                    SessionErrorCode::Storage,
                    "Session fork task failed",
                    error,
                )
            })?
    }

    async fn fork_inner(
        &self,
        source: &JsonlSessionMetadata,
        options: &ForkOptions,
        create_options: &JsonlSessionCreateOptions,
    ) -> Result<Session<JsonlSessionMetadata>, SessionError> {
        let source_storage = self.load_storage(source).await?;
        let create_options = JsonlSessionCreateOptions {
            parent_session_id: Some(
                create_options
                    .parent_session_id
                    .clone()
                    .unwrap_or_else(|| source.id.clone()),
            ),
            ..create_options.clone()
        };
        let destination = self.resolve_create_destination(&create_options).await?;
        self.claim_create_destination(&destination, async {
            let (header, path) = self.prepare_create(&destination, &create_options).await?;
            let storage = source_storage.fork(&path, &header, options).await?;
            Ok(Session::new(Arc::new(storage), None))
        })
        .await
    }

    async fn load_storage(
        &self,
        metadata: &JsonlSessionMetadata,
    ) -> Result<JsonlSessionStorage, SessionError> {
        load_jsonl_session_storage(&self.fs, metadata).await
    }

    async fn resolve_create_destination(
        &self,
        options: &JsonlSessionCreateOptions,
    ) -> Result<(String, String), SessionError> {
        let id = options
            .id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::now_v7().to_string());
        validate_session_id(&id)?;
        let cwd = file_result(
            self.fs.absolute_path(&options.cwd, None).await,
            &format!("Failed to resolve session cwd {}", options.cwd),
        )?;
        Ok((id, cwd))
    }

    /// Prevent same-process create/fork races for one logical destination: the
    /// durable file name includes a timestamp, so an existence check alone can
    /// let two concurrent calls publish duplicate sessions.
    async fn claim_create_destination<T>(
        &self,
        destination: &(String, String),
        operation: impl std::future::Future<Output = Result<T, SessionError>>,
    ) -> Result<T, SessionError> {
        let (id, cwd) = destination;
        let key = format!("{cwd}\0{id}");
        {
            let mut active = self
                .active_create_destinations
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if active.contains(&key) {
                return Err(SessionError::new(
                    SessionErrorCode::AlreadyExists,
                    format!("Session already exists: {id}"),
                ));
            }
            active.insert(key.clone());
        }
        let _claim = CreateClaim {
            destinations: &self.active_create_destinations,
            key,
        };
        let directory = self.session_directory(cwd).await?;
        let lock_path = file_result(
            self.fs
                .join_path(&[directory, format!(".create-{id}")], None)
                .await,
            "Failed to resolve session creation lock",
        )?;
        let _lease = file_result(
            self.fs.lock_file(&lock_path).await,
            "Session creation is already in progress",
        )?;
        operation.await
    }

    async fn prepare_create(
        &self,
        destination: &(String, String),
        options: &JsonlSessionCreateOptions,
    ) -> Result<(JsonlV4Header, String), SessionError> {
        let (id, cwd) = destination;
        if self.session_id_exists(id, cwd).await? {
            return Err(SessionError::new(
                SessionErrorCode::AlreadyExists,
                format!("Session already exists: {id}"),
            ));
        }
        let created_at = now_ms();
        let session_directory = self.session_directory(cwd).await?;
        let path = file_result(
            self.fs
                .join_path(
                    &[session_directory.clone(), session_file_name(created_at, id)],
                    None,
                )
                .await,
            &format!("Failed to resolve path for session {id}"),
        )?;
        if let Some(metadata) = &options.metadata {
            assert_json_serializable(&serde_json::Value::Object(metadata.clone()))?;
        }
        let header = JsonlV4Header::new(
            id.clone(),
            created_at,
            cwd.clone(),
            options.parent_session_id.clone(),
            options.metadata.clone(),
        );
        file_result(
            self.fs
                .create_dir(
                    &session_directory,
                    Some(&crate::harness::types::CreateDirOptions {
                        recursive: Some(true),
                        signal: None,
                    }),
                )
                .await,
            "Failed to create sessions directory",
        )?;
        Ok((header, path))
    }

    async fn session_id_exists(&self, id: &str, cwd: &str) -> Result<bool, SessionError> {
        let suffix = format!("_{id}.jsonl");
        let directory = self.session_directory(cwd).await?;
        if !file_result(
            self.fs.exists(&directory, None).await,
            &format!("Failed to check sessions directory {directory}"),
        )? {
            return Ok(false);
        }
        let files = file_result(
            self.fs.list_dir(&directory, None).await,
            &format!("Failed to list sessions directory {directory}"),
        )?;
        Ok(files
            .iter()
            .any(|entry| entry.kind != FileKind::Directory && entry.name.ends_with(&suffix)))
    }

    async fn session_directory(&self, cwd: &str) -> Result<String, SessionError> {
        let root = self.root().await?;
        file_result(
            self.fs
                .join_path(&[root, jsonl_session_directory_name(cwd)], None)
                .await,
            &format!("Failed to resolve sessions directory for {cwd}"),
        )
    }

    async fn root(&self) -> Result<String, SessionError> {
        let mut guard = self.root.lock().await;
        if let Some(root) = guard.as_ref() {
            return Ok(root.clone());
        }
        let root = file_result(
            self.fs.absolute_path(&self.sessions_root_input, None).await,
            &format!(
                "Failed to resolve sessions root {}",
                self.sessions_root_input
            ),
        )?;
        *guard = Some(root.clone());
        Ok(root)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_cwd_directories() {
        assert_eq!(
            jsonl_session_directory_name("/home/user/proj"),
            "--home-user-proj--"
        );
        assert_eq!(
            jsonl_session_directory_name("C:\\work\\proj"),
            "--C--work-proj--"
        );
    }

    #[test]
    fn formats_iso_timestamps() {
        // 2026-10-11T12:00:00.000Z
        assert_eq!(iso_timestamp(1_791_720_000_000), "2026-10-11T12-00-00-000Z");
        assert_eq!(
            session_file_name(1_791_720_000_000, "abc"),
            "2026-10-11T12-00-00-000Z_abc.jsonl"
        );
        // Epoch start.
        assert_eq!(iso_timestamp(0), "1970-01-01T00-00-00-000Z");
    }

    #[test]
    fn validates_session_ids() {
        assert!(validate_session_id("abc-123_x.y").is_ok());
        assert!(validate_session_id("a").is_ok());
        assert!(validate_session_id("").is_err());
        assert!(validate_session_id("-abc").is_err());
        assert!(validate_session_id("abc-").is_err());
        assert!(validate_session_id("a/b").is_err());
    }
}
