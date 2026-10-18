//! JSONL session storage: append-only file persistence with torn-tail repair.
//!
//! Ported from `@step-harness/agent-core` `src\harness\session\jsonl\storage.ts`.
//! Writes are serialized through a tokio mutex: each mutation is appended to
//! the file before it is applied to the in-memory state, so a crash can only
//! lose the final unacknowledged append, which is repaired on load. Exclusive
//! file leases prevent concurrent writers; acknowledged commits have been synced.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use step_harness_providers::utils::now_ms;
use tokio::sync::Mutex;

use crate::harness::types::{ExecutionEnv, FileContent};
use crate::session::facade::SessionStorage;
use crate::session::jsonl::codec::{
    encode_header, encode_mutation, metadata_from_header, parse_header, parse_mutation,
};
use crate::session::jsonl::types::{
    JsonlDecodeError, JsonlSessionMetadata, JsonlV4Header, file_result, invalid_file,
};
use crate::session::state::{SessionMutation, SessionState};
use crate::session::types::*;

/// Durable JSONL session storage.
pub struct JsonlSessionStorage {
    fs: Arc<dyn ExecutionEnv>,
    metadata: JsonlSessionMetadata,
    state: Arc<Mutex<SessionState>>,
    lease: Arc<dyn crate::harness::types::FileLease>,
    write_failed: Arc<AtomicBool>,
}

impl JsonlSessionStorage {
    /// Create a new session file with the given header.
    pub async fn create(
        fs: Arc<dyn ExecutionEnv>,
        path: &str,
        header: &JsonlV4Header,
    ) -> Result<Self, SessionError> {
        let path = path.to_owned();
        let header = header.clone();
        tokio::spawn(async move { Self::create_inner(fs, &path, &header).await })
            .await
            .map_err(|error| {
                SessionError::with_cause(
                    SessionErrorCode::Storage,
                    "Session creation task failed",
                    error,
                )
            })?
    }

    async fn create_inner(
        fs: Arc<dyn ExecutionEnv>,
        path: &str,
        header: &JsonlV4Header,
    ) -> Result<Self, SessionError> {
        let lease = file_result(fs.lock_file(path).await, "Failed to lock session")?;
        file_result(
            fs.write_file_atomic(path, FileContent::Text(encode_header(header)), false)
                .await,
            &format!("Failed to initialize session {path}"),
        )?;
        let file_info = file_result(
            fs.file_info(path, None).await,
            &format!("Failed to read session metadata {path}"),
        )?;
        Ok(JsonlSessionStorage {
            fs,
            metadata: metadata_from_header(header, path, file_info.mtime_ms),
            state: Arc::new(Mutex::new(SessionState::default())),
            lease,
            write_failed: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Load a session file, replaying mutations and repairing a torn tail.
    pub async fn load(fs: Arc<dyn ExecutionEnv>, path: &str) -> Result<Self, SessionError> {
        let path = path.to_owned();
        tokio::spawn(async move { Self::load_inner(fs, &path).await })
            .await
            .map_err(|error| {
                SessionError::with_cause(
                    SessionErrorCode::Storage,
                    "Session recovery task failed",
                    error,
                )
            })?
    }

    async fn load_inner(fs: Arc<dyn ExecutionEnv>, path: &str) -> Result<Self, SessionError> {
        let lease = file_result(fs.lock_file(path).await, "Failed to lock session")?;
        let content = file_result(
            fs.read_text_file(path, None).await,
            &format!("Failed to read session {path}"),
        )?;
        let mut physical_lines: Vec<&str> = content.split('\n').collect();
        if physical_lines.last() == Some(&"") {
            physical_lines.pop();
        }
        if physical_lines.is_empty() || physical_lines[0].is_empty() {
            return Err(invalid_file(
                path,
                1,
                JsonlDecodeError::schema("is missing a header"),
            ));
        }
        let header = match parse_header(physical_lines[0]) {
            Ok(header) => header,
            Err(error) => return Err(invalid_file(path, 1, error)),
        };
        let file_info = file_result(
            fs.file_info(path, None).await,
            &format!("Failed to read session metadata {path}"),
        )?;
        let storage = JsonlSessionStorage {
            fs: fs.clone(),
            metadata: metadata_from_header(&header, path, file_info.mtime_ms),
            state: Arc::new(Mutex::new(SessionState::default())),
            lease,
            write_failed: Arc::new(AtomicBool::new(false)),
        };
        {
            let mut state = storage.state.lock().await;
            for (index, line) in physical_lines.iter().enumerate().skip(1) {
                match parse_mutation(line) {
                    Ok(mutation) => {
                        if let Err(error) = state.apply_mutation(&mutation) {
                            if error.code == SessionErrorCode::InvalidEntry {
                                return Err(invalid_file(path, index + 1, decode_error_of(&error)));
                            }
                            return Err(error);
                        }
                    }
                    Err(error) => {
                        let is_torn_tail = index == physical_lines.len() - 1
                            && error.kind
                                == crate::session::jsonl::types::JsonlDecodeErrorKind::Syntax;
                        if is_torn_tail {
                            // Drop the unacknowledged partial append by atomically
                            // publishing the valid prefix.
                            let valid_prefix = format!("{}\n", physical_lines[..index].join("\n"));
                            publish_file_atomically(
                                fs.as_ref(),
                                path,
                                &valid_prefix,
                                &format!("Failed to stage torn-tail repair {path}"),
                            )
                            .await?;
                            drop(state);
                            return Ok(storage);
                        }
                        return Err(invalid_file(path, index + 1, error));
                    }
                }
            }
        }
        if !content.ends_with('\n') {
            file_result(
                fs.append_file_durable(path, FileContent::Text("\n".to_owned()))
                    .await,
                &format!("Failed to repair unterminated session tail {path}"),
            )?;
        }
        Ok(storage)
    }

    /// Atomically publish a complete fork without exposing a partially initialized file.
    pub async fn fork(
        &self,
        path: &str,
        header: &JsonlV4Header,
        options: &ForkOptions,
    ) -> Result<Self, SessionError> {
        let mutations = {
            let state = self.state.lock().await;
            state.create_fork_mutations(options)?
        };
        let fs = self.fs.clone();
        let path = path.to_owned();
        let header = header.clone();
        tokio::spawn(async move {
            let mut content = encode_header(&header);
            let mut state = SessionState::default();
            for mutation in mutations {
                state.apply_mutation(&mutation)?;
                content.push_str(&encode_mutation(&mutation));
            }
            let lease = file_result(fs.lock_file(&path).await, "Failed to lock fork destination")?;
            file_result(
                fs.write_file_atomic(&path, FileContent::Text(content), false)
                    .await,
                "Failed to publish forked session",
            )?;
            let info = file_result(
                fs.file_info(&path, None).await,
                "Failed to read fork metadata",
            )?;
            Ok(Self {
                fs: fs.clone(),
                metadata: metadata_from_header(&header, &path, info.mtime_ms),
                state: Arc::new(Mutex::new(state)),
                lease,
                write_failed: Arc::new(AtomicBool::new(false)),
            })
        })
        .await
        .map_err(|error| {
            SessionError::with_cause(SessionErrorCode::Storage, "Session fork task failed", error)
        })?
    }

    /// A commit runs in an owned task so dropping its caller cannot interrupt
    /// the append/apply boundary. A failed append requires reopening for recovery.
    async fn commit<T, F>(&self, prepare: F) -> Result<T, SessionError>
    where
        T: Send + 'static,
        F: FnOnce(&SessionState) -> Result<(SessionMutation, T), SessionError> + Send + 'static,
    {
        let mut state = self.state.clone().lock_owned().await;
        let fs = self.fs.clone();
        let path = self.metadata.path.clone();
        let lease = self.lease.clone();
        let write_failed = self.write_failed.clone();
        tokio::spawn(async move {
            let _lease = lease;
            if write_failed.load(Ordering::Acquire) {
                return Err(SessionError::new(
                    SessionErrorCode::Storage,
                    "Session must be reopened after a failed write",
                ));
            }
            let (mutation, result) = prepare(&state)?;
            state.validate_mutation(&mutation)?;
            if let Err(error) = fs
                .append_file_durable(&path, FileContent::Text(encode_mutation(&mutation)))
                .await
            {
                write_failed.store(true, Ordering::Release);
                return Err(SessionError::with_cause(
                    SessionErrorCode::Storage,
                    format!("Failed to append session {path}"),
                    error,
                ));
            }
            state.apply_mutation(&mutation)?;
            Ok(result)
        })
        .await
        .map_err(|error| {
            SessionError::with_cause(
                SessionErrorCode::Storage,
                "Session commit task failed",
                error,
            )
        })?
    }

    /// Wait for all queued writes to settle.
    ///
    /// Every write path holds the state mutex across its file append and state
    /// update, so acquiring the guard here observes a fully drained queue.
    pub async fn drain(&self) {
        let _guard = self.state.lock().await;
    }
}

/// Build a complete temporary file, then atomically rename it over the destination.
async fn publish_file_atomically(
    fs: &dyn ExecutionEnv,
    destination_path: &str,
    content: &str,
    message: &str,
) -> Result<(), SessionError> {
    file_result(
        fs.write_file_atomic(
            destination_path,
            FileContent::Text(content.to_owned()),
            true,
        )
        .await,
        message,
    )
}

/// Wrap a state error into the decode error shape expected by `invalid_file`.
fn decode_error_of(error: &SessionError) -> JsonlDecodeError {
    JsonlDecodeError::schema(error.message.clone())
}

#[async_trait::async_trait]
impl SessionStorage for JsonlSessionStorage {
    type Metadata = JsonlSessionMetadata;

    async fn get_metadata(&self) -> JsonlSessionMetadata {
        self.metadata.clone()
    }

    async fn get_lanes(&self) -> Vec<LanePointer> {
        self.state.lock().await.get_lanes()
    }

    async fn create_lane(&self, lane: &str, at: Option<&str>) -> Result<(), SessionError> {
        let lane = lane.to_owned();
        let at = at.map(str::to_owned);
        self.commit(move |state| {
            state.validate_new_lane(&lane)?;
            state.validate_target(at.as_deref())?;
            Ok((
                SessionMutation::Lane {
                    seq: state.next_sequence(),
                    lane,
                    leaf_id: at,
                },
                (),
            ))
        })
        .await
    }

    async fn move_lane(&self, lane: &str, to: Option<&str>) -> Result<(), SessionError> {
        let lane = lane.to_owned();
        let to = to.map(str::to_owned);
        self.commit(move |state| {
            state.require_lane(&lane)?;
            state.validate_target(to.as_deref())?;
            Ok((
                SessionMutation::Lane {
                    seq: state.next_sequence(),
                    lane,
                    leaf_id: to,
                },
                (),
            ))
        })
        .await
    }

    async fn append_entry(
        &self,
        entry: ProvisionedEntry,
        lane: &str,
    ) -> Result<Entry, SessionError> {
        let lane = lane.to_owned();
        self.commit(move |state| {
            let parent_id = state.require_lane(&lane)?;
            state.validate_unused_id(entry.id())?;
            let full = entry.into_entry(parent_id, state.next_sequence(), now_ms());
            Ok((
                SessionMutation::Entry {
                    lane: Some(lane),
                    entry: full.clone(),
                },
                full,
            ))
        })
        .await
    }

    async fn append_record(&self, record: NewRecord) -> Result<LaneRecord, SessionError> {
        self.commit(move |state| {
            state.require_lane(record.lane())?;
            state.validate_unused_id(record.id())?;
            if matches!(record, NewRecord::OperationStarted { .. })
                && !state
                    .find_open_operations(record.lane(), Some(1))?
                    .is_empty()
            {
                return Err(SessionError::new(
                    SessionErrorCode::Storage,
                    "Lane already has an open operation",
                ));
            }
            let full = record.into_record(state.next_sequence(), now_ms());
            Ok((
                SessionMutation::Record {
                    record: full.clone(),
                },
                full,
            ))
        })
        .await
    }

    async fn get_entry(&self, id: &str) -> Option<Entry> {
        self.state.lock().await.get_entry(id).cloned()
    }

    async fn find_entries(&self, query: &EntryQuery) -> Result<Vec<Entry>, SessionError> {
        self.state.lock().await.find_entries(query)
    }

    async fn find_entries_on_branch(
        &self,
        query: &BranchEntryQuery,
    ) -> Result<Vec<Entry>, SessionError> {
        self.state.lock().await.find_entries_on_branch(query)
    }

    async fn find_records(&self, query: &RecordQuery) -> Result<Vec<LaneRecord>, SessionError> {
        self.state.lock().await.find_records(query)
    }

    async fn find_open_operations(
        &self,
        lane: &str,
        limit: Option<usize>,
    ) -> Result<Vec<OperationStartedRecord>, SessionError> {
        self.state.lock().await.find_open_operations(lane, limit)
    }

    async fn get_log(&self, options: &LogOptions) -> Result<Vec<LogItem>, SessionError> {
        self.state.lock().await.get_log(options)
    }

    async fn get_name(&self) -> Option<String> {
        self.state.lock().await.get_name().map(str::to_owned)
    }

    async fn set_name(&self, name: Option<&str>) -> Result<(), SessionError> {
        let name = name.map(str::to_owned);
        self.commit(move |state| Ok((SessionMutation::name(state.next_sequence(), name), ())))
            .await
    }

    async fn get_project(&self) -> Option<String> {
        self.state.lock().await.get_project().map(str::to_owned)
    }

    async fn set_project(&self, project: Option<&str>) -> Result<(), SessionError> {
        let project = project.map(str::to_owned);
        self.commit(move |state| Ok((SessionMutation::project(state.next_sequence(), project), ())))
            .await
    }

    async fn get_label(&self, id: &str) -> Option<String> {
        self.state.lock().await.get_label(id).map(str::to_owned)
    }

    async fn set_label(&self, id: &str, label: Option<&str>) -> Result<(), SessionError> {
        let id = id.to_owned();
        let label = label.map(str::to_owned);
        self.commit(move |state| {
            state.validate_target(Some(&id))?;
            Ok((SessionMutation::label(state.next_sequence(), id, label), ()))
        })
        .await
    }

    async fn get_stats(&self) -> SessionStats {
        self.state.lock().await.get_stats()
    }
}
