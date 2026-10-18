//! In-memory session storage and repository.
//!
//! Ported from `@step-harness/agent-core` `src\harness\session\memory.ts`.
//! This is the reference implementation of [`SessionStorage`]; the durable
//! JSONL backend lives in [`crate::session::jsonl`].

use std::collections::HashMap;
use std::sync::Arc;

use step_harness_providers::utils::now_ms;

use crate::session::facade::{Session, SessionStorage};
use crate::session::state::{SessionMutation, SessionState};
use crate::session::types::*;
use crate::session::types::{IdGenerator, UuidV7Generator};

/// In-memory storage backed by a single [`SessionState`].
pub struct InMemorySessionStorage {
    metadata: SessionMetadata,
    state: std::sync::Mutex<SessionState>,
}

impl InMemorySessionStorage {
    pub fn new(metadata: SessionMetadata) -> Self {
        InMemorySessionStorage {
            metadata,
            state: std::sync::Mutex::new(SessionState::default()),
        }
    }

    /// Build a fork of this storage by replaying fork mutations into a new state.
    pub fn fork(
        &self,
        metadata: SessionMetadata,
        options: &ForkOptions,
    ) -> Result<Self, SessionError> {
        let state = self.state.lock().expect("session state poisoned");
        let mutations = state.create_fork_mutations(options)?;
        let forked = InMemorySessionStorage::new(metadata);
        {
            let mut forked_state = forked.state.lock().expect("session state poisoned");
            for mutation in &mutations {
                forked_state.apply_mutation(mutation)?;
            }
        }
        Ok(forked)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, SessionState> {
        self.state.lock().expect("session state poisoned")
    }
}

#[async_trait::async_trait]
impl SessionStorage for InMemorySessionStorage {
    type Metadata = SessionMetadata;

    async fn get_metadata(&self) -> SessionMetadata {
        self.metadata.clone()
    }

    async fn get_lanes(&self) -> Vec<LanePointer> {
        self.lock().get_lanes()
    }

    async fn create_lane(&self, lane: &str, at: Option<&str>) -> Result<(), SessionError> {
        let mut state = self.lock();
        state.validate_new_lane(lane)?;
        state.validate_target(at)?;
        let mutation = SessionMutation::Lane {
            seq: state.next_sequence(),
            lane: lane.to_owned(),
            leaf_id: at.map(str::to_owned),
        };
        state.apply_mutation(&mutation)
    }

    async fn move_lane(&self, lane: &str, to: Option<&str>) -> Result<(), SessionError> {
        let mut state = self.lock();
        state.require_lane(lane)?;
        state.validate_target(to)?;
        let mutation = SessionMutation::Lane {
            seq: state.next_sequence(),
            lane: lane.to_owned(),
            leaf_id: to.map(str::to_owned),
        };
        state.apply_mutation(&mutation)
    }

    async fn append_entry(
        &self,
        entry: ProvisionedEntry,
        lane: &str,
    ) -> Result<Entry, SessionError> {
        let mut state = self.lock();
        let parent_id = state.require_lane(lane)?;
        state.validate_unused_id(entry.id())?;
        let full = entry.into_entry(parent_id, state.next_sequence(), now_ms());
        state.apply_mutation(&SessionMutation::Entry {
            lane: Some(lane.to_owned()),
            entry: full.clone(),
        })?;
        Ok(full)
    }

    async fn append_record(&self, record: NewRecord) -> Result<LaneRecord, SessionError> {
        let mut state = self.lock();
        state.require_lane(record.lane())?;
        state.validate_unused_id(record.id())?;
        if matches!(record, NewRecord::OperationStarted { .. }) {
            let open = state.find_open_operations(record.lane(), Some(1))?;
            if let Some(current) = open.first() {
                return Err(SessionError::new(
                    SessionErrorCode::Storage,
                    format!(
                        "Lane {} already has an open operation {}",
                        record.lane(),
                        current.id
                    ),
                ));
            }
        }
        let full = record.into_record(state.next_sequence(), now_ms());
        state.apply_mutation(&SessionMutation::Record {
            record: full.clone(),
        })?;
        Ok(full)
    }

    async fn get_entry(&self, id: &str) -> Option<Entry> {
        self.lock().get_entry(id).cloned()
    }

    async fn find_entries(&self, query: &EntryQuery) -> Result<Vec<Entry>, SessionError> {
        self.lock().find_entries(query)
    }

    async fn find_entries_on_branch(
        &self,
        query: &BranchEntryQuery,
    ) -> Result<Vec<Entry>, SessionError> {
        self.lock().find_entries_on_branch(query)
    }

    async fn find_records(&self, query: &RecordQuery) -> Result<Vec<LaneRecord>, SessionError> {
        self.lock().find_records(query)
    }

    async fn find_open_operations(
        &self,
        lane: &str,
        limit: Option<usize>,
    ) -> Result<Vec<OperationStartedRecord>, SessionError> {
        self.lock().find_open_operations(lane, limit)
    }

    async fn get_log(&self, options: &LogOptions) -> Result<Vec<LogItem>, SessionError> {
        self.lock().get_log(options)
    }

    async fn get_name(&self) -> Option<String> {
        self.lock().get_name().map(str::to_owned)
    }

    async fn set_name(&self, name: Option<&str>) -> Result<(), SessionError> {
        let mut state = self.lock();
        let mutation = SessionMutation::name(state.next_sequence(), name.map(str::to_owned));
        state.apply_mutation(&mutation)
    }

    async fn get_project(&self) -> Option<String> {
        self.lock().get_project().map(str::to_owned)
    }

    async fn set_project(&self, project: Option<&str>) -> Result<(), SessionError> {
        let mut state = self.lock();
        let mutation = SessionMutation::project(state.next_sequence(), project.map(str::to_owned));
        state.apply_mutation(&mutation)
    }

    async fn get_label(&self, id: &str) -> Option<String> {
        self.lock().get_label(id).map(str::to_owned)
    }

    async fn set_label(&self, id: &str, label: Option<&str>) -> Result<(), SessionError> {
        let mut state = self.lock();
        state.validate_target(Some(id))?;
        let mutation = SessionMutation::label(
            state.next_sequence(),
            id.to_owned(),
            label.map(str::to_owned),
        );
        state.apply_mutation(&mutation)
    }

    async fn get_stats(&self) -> SessionStats {
        self.lock().get_stats()
    }
}

/// In-memory repository over a map of storages.
pub struct InMemorySessionRepo {
    sessions: std::sync::Mutex<HashMap<String, Arc<InMemorySessionStorage>>>,
    id_generator: Arc<dyn IdGenerator>,
}

impl Default for InMemorySessionRepo {
    fn default() -> Self {
        InMemorySessionRepo {
            sessions: std::sync::Mutex::new(HashMap::new()),
            id_generator: Arc::new(UuidV7Generator),
        }
    }
}

impl InMemorySessionRepo {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, Arc<InMemorySessionStorage>>> {
        self.sessions.lock().expect("session repo poisoned")
    }

    fn require_storage(&self, id: &str) -> Result<Arc<InMemorySessionStorage>, SessionError> {
        self.lock().get(id).cloned().ok_or_else(|| {
            SessionError::new(
                SessionErrorCode::NotFound,
                format!("Session not found: {id}"),
            )
        })
    }

    pub async fn create(
        &self,
        options: SessionCreateOptions,
    ) -> Result<Session<SessionMetadata>, SessionError> {
        let id = options.id.unwrap_or_else(|| self.id_generator.next());
        if self.lock().contains_key(&id) {
            return Err(SessionError::new(
                SessionErrorCode::AlreadyExists,
                format!("Session already exists: {id}"),
            ));
        }
        let storage = Arc::new(InMemorySessionStorage::new(SessionMetadata {
            id: id.clone(),
            created_at: now_ms(),
            parent_session_id: options.parent_session_id,
        }));
        self.lock().insert(id, storage.clone());
        Ok(Session::new(storage, Some(self.id_generator.clone())))
    }

    pub async fn open(
        &self,
        metadata: SessionMetadata,
    ) -> Result<Session<SessionMetadata>, SessionError> {
        let storage = self.require_storage(&metadata.id)?;
        Ok(Session::new(storage, Some(self.id_generator.clone())))
    }

    pub async fn list(&self) -> Result<Vec<SessionMetadata>, SessionError> {
        let storages: Vec<Arc<InMemorySessionStorage>> = self.lock().values().cloned().collect();
        let mut metadata = Vec::with_capacity(storages.len());
        for storage in storages {
            metadata.push(storage.get_metadata().await);
        }
        Ok(metadata)
    }

    pub async fn delete(&self, metadata: SessionMetadata) -> Result<(), SessionError> {
        self.lock().remove(&metadata.id);
        Ok(())
    }

    pub async fn fork(
        &self,
        source: SessionMetadata,
        options: ForkOptions,
        id: Option<String>,
    ) -> Result<Session<SessionMetadata>, SessionError> {
        let source_storage = self.require_storage(&source.id)?;
        let id = id.unwrap_or_else(|| self.id_generator.next());
        if self.lock().contains_key(&id) {
            return Err(SessionError::new(
                SessionErrorCode::AlreadyExists,
                format!("Session already exists: {id}"),
            ));
        }
        let storage = Arc::new(source_storage.fork(
            SessionMetadata {
                id,
                created_at: now_ms(),
                parent_session_id: Some(source.id),
            },
            &options,
        )?);
        let forked_id = storage.get_metadata().await.id.clone();
        self.lock().insert(forked_id, storage.clone());
        Ok(Session::new(storage, Some(self.id_generator.clone())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::state::FactKind;
    use crate::types::AgentMessage;

    #[tokio::test]
    async fn storage_enforces_lane_invariants() {
        let storage = InMemorySessionStorage::new(SessionMetadata {
            id: "s".to_owned(),
            created_at: 1,
            parent_session_id: None,
        });
        storage.create_lane("alt", None).await.unwrap();
        assert!(matches!(
            storage.create_lane("alt", None).await.unwrap_err().code,
            SessionErrorCode::AlreadyExists
        ));
        storage.move_lane("alt", None).await.unwrap();
        assert!(matches!(
            storage.move_lane("missing", None).await.unwrap_err().code,
            SessionErrorCode::InvalidLane
        ));
    }

    #[tokio::test]
    async fn one_open_operation_per_lane() {
        let storage = InMemorySessionStorage::new(SessionMetadata {
            id: "s".to_owned(),
            created_at: 1,
            parent_session_id: None,
        });
        let started = NewRecord::OperationStarted {
            id: "run-1".to_owned(),
            lane: "main".to_owned(),
            source_leaf_id: None,
            intent: OperationIntent::Run {
                original_prompt: vec![AgentMessage::user("hi", 1)],
                initial_messages: vec![],
                system_prompt_override: None,
                resume_data: None,
            },
        };
        storage.append_record(started.clone()).await.unwrap();
        // A second start (with a fresh id) is rejected while the first is open.
        let error = storage
            .append_record(NewRecord::OperationStarted {
                id: "run-2".to_owned(),
                lane: "main".to_owned(),
                source_leaf_id: None,
                intent: OperationIntent::Run {
                    original_prompt: vec![AgentMessage::user("hi", 1)],
                    initial_messages: vec![],
                    system_prompt_override: None,
                    resume_data: None,
                },
            })
            .await
            .unwrap_err();
        assert_eq!(error.code, SessionErrorCode::Storage);

        let open = storage.find_open_operations("main", Some(2)).await.unwrap();
        assert_eq!(open.len(), 1);

        storage
            .append_record(NewRecord::OperationFinished {
                id: "fin-1".to_owned(),
                lane: "main".to_owned(),
                run_id: "run-1".to_owned(),
                outcome: OperationOutcome::Completed,
                error: None,
            })
            .await
            .unwrap();
        assert!(
            storage
                .find_open_operations("main", Some(2))
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn repo_forks_copy_branch_state() {
        let repo = InMemorySessionRepo::new();
        let session = repo.create(SessionCreateOptions::default()).await.unwrap();
        session
            .append_message(AgentMessage::user("one", 1))
            .await
            .unwrap();
        let second = session
            .append_message(AgentMessage::user("two", 2))
            .await
            .unwrap();
        session.set_name(Some("named")).await.unwrap();

        let metadata = session.get_metadata().await;
        let forked = repo
            .fork(
                metadata,
                ForkOptions::Branch {
                    entry_id: Some(second),
                    position: Some(crate::session::types::ForkPosition::At),
                },
                None,
            )
            .await
            .unwrap();
        assert_eq!(forked.get_name().await, Some("named".to_owned()));
        assert_eq!(
            forked
                .find_entries(&EntryQuery::default())
                .await
                .unwrap()
                .len(),
            2
        );
        assert_eq!(forked.get_stats().await.message_count, 2);
        // The source keeps its own lane leaf.
        assert!(session.get_leaf_id().await.unwrap().is_some());
    }

    #[tokio::test]
    async fn usage_records_accumulate_stats() {
        let storage = InMemorySessionStorage::new(SessionMetadata {
            id: "s".to_owned(),
            created_at: 1,
            parent_session_id: None,
        });
        storage
            .append_record(NewRecord::Usage {
                id: "u1".to_owned(),
                lane: "main".to_owned(),
                usage: step_harness_providers::types::Usage {
                    input: 10,
                    output: 20,
                    cache_read: 5,
                    cache_write: 0,
                    total_tokens: 35,
                    ..Default::default()
                },
                cause: UsageCause::Tool {
                    run_id: "run-1".to_owned(),
                    entry_id: "e1".to_owned(),
                    tool_call_id: "call-1".to_owned(),
                },
            })
            .await
            .unwrap();
        let stats = storage.get_stats().await;
        assert_eq!(stats.cached_tokens, 5);
        assert_eq!(stats.uncached_tokens, 10);
        assert_eq!(stats.total_tokens, 35);
    }

    #[tokio::test]
    async fn facts_round_trip() {
        let storage = InMemorySessionStorage::new(SessionMetadata {
            id: "s".to_owned(),
            created_at: 1,
            parent_session_id: None,
        });
        let entry = storage
            .append_entry(
                ProvisionedEntry::Message {
                    id: "e1".to_owned(),
                    message: AgentMessage::user("hi", 1),
                    terminate: false,
                },
                "main",
            )
            .await
            .unwrap();
        storage.set_label(entry.id(), Some("label")).await.unwrap();
        assert_eq!(
            storage.get_label(entry.id()).await,
            Some("label".to_owned())
        );
        storage.set_label(entry.id(), None).await.unwrap();
        assert_eq!(storage.get_label(entry.id()).await, None);
        assert!(matches!(
            storage
                .set_label("missing", Some("x"))
                .await
                .unwrap_err()
                .code,
            SessionErrorCode::NotFound
        ));
    }

    #[test]
    fn fork_mutations_skip_records() {
        let state = SessionState::default();
        let mutations = state.create_fork_mutations(&ForkOptions::Tree).unwrap();
        // The main lane pointer only.
        assert_eq!(mutations.len(), 1);
        assert!(matches!(mutations[0], SessionMutation::Lane { .. }));
        let _ = FactKind::Name;
    }
}
