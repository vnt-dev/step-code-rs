//! The session facade and storage contract.
//!
//! Ported from `@step-harness/agent-core` `src\harness\session\session.ts`.
//! [`Session`] owns a [`SessionStorage`] backend and an id generator; lane
//! views are lightweight handles that forward lane-scoped operations.

use std::sync::Arc;

use crate::session::types::*;
use crate::types::AgentMessage;

/// Storage backend contract for durable sessions.
///
/// Implementations must never panic: all failures are [`SessionError`] values.
#[async_trait::async_trait]
pub trait SessionStorage: Send + Sync {
    /// The metadata flavor this storage produces.
    type Metadata: Clone + Send + Sync + 'static;

    async fn get_metadata(&self) -> Self::Metadata;

    // Lanes
    async fn get_lanes(&self) -> Vec<LanePointer>;
    async fn create_lane(&self, lane: &str, at: Option<&str>) -> Result<(), SessionError>;
    async fn move_lane(&self, lane: &str, to: Option<&str>) -> Result<(), SessionError>;

    // Entries and records
    async fn append_entry(
        &self,
        entry: ProvisionedEntry,
        lane: &str,
    ) -> Result<Entry, SessionError>;
    async fn append_record(&self, record: NewRecord) -> Result<LaneRecord, SessionError>;

    // Reads
    async fn get_entry(&self, id: &str) -> Option<Entry>;
    async fn find_entries(&self, query: &EntryQuery) -> Result<Vec<Entry>, SessionError>;
    async fn find_entries_on_branch(
        &self,
        query: &BranchEntryQuery,
    ) -> Result<Vec<Entry>, SessionError>;
    async fn find_records(&self, query: &RecordQuery) -> Result<Vec<LaneRecord>, SessionError>;
    async fn find_open_operations(
        &self,
        lane: &str,
        limit: Option<usize>,
    ) -> Result<Vec<OperationStartedRecord>, SessionError>;
    async fn get_log(&self, options: &LogOptions) -> Result<Vec<LogItem>, SessionError>;

    // Global facts
    async fn get_name(&self) -> Option<String>;
    async fn set_name(&self, name: Option<&str>) -> Result<(), SessionError>;
    async fn get_project(&self) -> Option<String>;
    async fn set_project(&self, project: Option<&str>) -> Result<(), SessionError>;
    async fn get_label(&self, id: &str) -> Option<String>;
    async fn set_label(&self, id: &str, label: Option<&str>) -> Result<(), SessionError>;
    async fn get_stats(&self) -> SessionStats;
}

/// Repository of sessions: create, open, list, delete, fork.
#[async_trait::async_trait]
pub trait SessionRepo: Send + Sync {
    type Metadata: Clone + Send + Sync + 'static;
    type CreateOptions: Clone + Default + Send + Sync + 'static;
    type ListOptions: Default;

    async fn create(
        &self,
        options: Self::CreateOptions,
    ) -> Result<Session<Self::Metadata>, SessionError>;
    async fn open(&self, metadata: Self::Metadata)
    -> Result<Session<Self::Metadata>, SessionError>;
    async fn list(&self, options: Self::ListOptions) -> Result<Vec<Self::Metadata>, SessionError>;
    async fn delete(&self, metadata: Self::Metadata) -> Result<(), SessionError>;
    async fn fork(
        &self,
        source: Self::Metadata,
        options: ForkOptions,
    ) -> Result<Session<Self::Metadata>, SessionError>;
}

/// Shared query/appendix helpers used by [`Session`] and [`LaneView`].
async fn query_entries<S: SessionStorage + ?Sized>(
    storage: &S,
    query: &EntryQuery,
    result_limit: Option<usize>,
) -> Result<Vec<Entry>, SessionError> {
    assert_valid_limit(query.limit)?;
    let effective = match result_limit {
        Some(limit) if Some(limit) != query.limit => EntryQuery {
            limit: Some(limit),
            ..query.clone()
        },
        _ => query.clone(),
    };
    storage.find_entries(&effective).await
}

async fn query_branch_entries<S: SessionStorage + ?Sized>(
    storage: &S,
    default_lane: &str,
    query: &BranchEntryQuery,
    result_limit: Option<usize>,
) -> Result<Vec<Entry>, SessionError> {
    assert_valid_limit(query.limit)?;
    let start = match &query.bounds.start {
        Some(start) => Some(start.clone()),
        None => leaf_id_for_lane(storage, default_lane).await?,
    };
    let Some(start) = start else {
        return Ok(Vec::new());
    };
    let effective = match result_limit {
        Some(limit) if Some(limit) != query.limit => BranchEntryQuery {
            limit: Some(limit),
            ..query.clone()
        },
        _ => query.clone(),
    };
    storage
        .find_entries_on_branch(&BranchEntryQuery {
            bounds: BranchBounds {
                start: Some(start),
                ..effective.bounds
            },
            ..effective
        })
        .await
}

async fn leaf_id_for_lane<S: SessionStorage + ?Sized>(
    storage: &S,
    lane: &str,
) -> Result<Option<String>, SessionError> {
    storage
        .get_lanes()
        .await
        .into_iter()
        .find(|pointer| pointer.lane == lane)
        .map(|pointer| pointer.leaf_id)
        .ok_or_else(|| {
            SessionError::new(
                SessionErrorCode::InvalidLane,
                format!("Lane not found: {lane}"),
            )
        })
}

async fn append_message_to_lane<S: SessionStorage + ?Sized>(
    storage: &S,
    id_generator: &dyn IdGenerator,
    lane: &str,
    message: AgentMessage,
) -> Result<String, SessionError> {
    let entry = storage
        .append_entry(
            ProvisionedEntry::Message {
                id: id_generator.next(),
                message,
                terminate: false,
            },
            lane,
        )
        .await?;
    Ok(entry.id().to_owned())
}

async fn append_custom_entry_to_lane<S: SessionStorage + ?Sized>(
    storage: &S,
    id_generator: &dyn IdGenerator,
    lane: &str,
    custom_type: String,
    data: Option<JsonValue>,
) -> Result<String, SessionError> {
    let entry = storage
        .append_entry(
            ProvisionedEntry::Custom {
                id: id_generator.next(),
                custom_type,
                data,
            },
            lane,
        )
        .await?;
    Ok(entry.id().to_owned())
}

/// A lane-scoped view over a session.
pub struct LaneView<S: SessionStorage + ?Sized> {
    storage: Arc<S>,
    id_generator: Arc<dyn IdGenerator>,
    lane: String,
}

impl<S: SessionStorage + ?Sized> LaneView<S> {
    /// The lane this view is scoped to.
    pub fn lane(&self) -> &str {
        &self.lane
    }

    pub async fn get_leaf_id(&self) -> Result<Option<String>, SessionError> {
        leaf_id_for_lane(self.storage.as_ref(), &self.lane).await
    }

    pub async fn get_entry(&self, id: &str) -> Option<Entry> {
        self.storage.get_entry(id).await
    }

    pub async fn get_stats(&self) -> SessionStats {
        self.storage.get_stats().await
    }

    pub async fn get_name(&self) -> Option<String> {
        self.storage.get_name().await
    }

    pub async fn set_name(&self, name: Option<&str>) -> Result<(), SessionError> {
        self.storage.set_name(name).await
    }

    pub async fn get_project(&self) -> Option<String> {
        self.storage.get_project().await
    }

    pub async fn set_project(&self, project: Option<&str>) -> Result<(), SessionError> {
        self.storage.set_project(project).await
    }

    pub async fn get_label(&self, target_id: &str) -> Option<String> {
        self.storage.get_label(target_id).await
    }

    pub async fn set_label(
        &self,
        target_id: &str,
        label: Option<&str>,
    ) -> Result<(), SessionError> {
        self.storage.set_label(target_id, label).await
    }

    pub async fn find_entries(&self, query: &EntryQuery) -> Result<Vec<Entry>, SessionError> {
        query_entries(self.storage.as_ref(), query, None).await
    }

    pub async fn find_entry(&self, query: &EntryQuery) -> Result<Option<Entry>, SessionError> {
        Ok(query_entries(self.storage.as_ref(), query, Some(1))
            .await?
            .into_iter()
            .next())
    }

    pub async fn find_entries_on_branch(
        &self,
        query: &BranchEntryQuery,
    ) -> Result<Vec<Entry>, SessionError> {
        query_branch_entries(self.storage.as_ref(), &self.lane, query, None).await
    }

    pub async fn find_entry_on_branch(
        &self,
        query: &BranchEntryQuery,
    ) -> Result<Option<Entry>, SessionError> {
        Ok(
            query_branch_entries(self.storage.as_ref(), &self.lane, query, Some(1))
                .await?
                .into_iter()
                .next(),
        )
    }

    pub async fn append_message(&self, message: AgentMessage) -> Result<String, SessionError> {
        append_message_to_lane(
            self.storage.as_ref(),
            self.id_generator.as_ref(),
            &self.lane,
            message,
        )
        .await
    }

    pub async fn append_custom_entry(
        &self,
        custom_type: impl Into<String>,
        data: Option<JsonValue>,
    ) -> Result<String, SessionError> {
        append_custom_entry_to_lane(
            self.storage.as_ref(),
            self.id_generator.as_ref(),
            &self.lane,
            custom_type.into(),
            data,
        )
        .await
    }
}

/// A durable session: the facade over a storage backend.
pub struct Session<M: Clone + Send + Sync + 'static> {
    storage: Arc<dyn SessionStorage<Metadata = M>>,
    id_generator: Arc<dyn IdGenerator>,
}

impl<M: Clone + Send + Sync + 'static> Session<M> {
    pub fn new(
        storage: Arc<dyn SessionStorage<Metadata = M>>,
        id_generator: Option<Arc<dyn IdGenerator>>,
    ) -> Self {
        Session {
            storage,
            id_generator: id_generator.unwrap_or_else(|| Arc::new(UuidV7Generator)),
        }
    }

    pub async fn get_metadata(&self) -> M {
        self.storage.get_metadata().await
    }

    /// A lane-scoped view. The `main` lane returns main-lane operations.
    pub fn view(&self, lane: &str) -> LaneView<dyn SessionStorage<Metadata = M>> {
        LaneView {
            storage: self.storage.clone(),
            id_generator: self.id_generator.clone(),
            lane: lane.to_owned(),
        }
    }

    pub async fn get_leaf_id(&self) -> Result<Option<String>, SessionError> {
        leaf_id_for_lane(self.storage.as_ref(), "main").await
    }

    pub async fn get_entry(&self, id: &str) -> Option<Entry> {
        self.storage.get_entry(id).await
    }

    pub async fn get_stats(&self) -> SessionStats {
        self.storage.get_stats().await
    }

    pub async fn get_name(&self) -> Option<String> {
        self.storage.get_name().await
    }

    pub async fn set_name(&self, name: Option<&str>) -> Result<(), SessionError> {
        self.storage.set_name(name).await
    }

    pub async fn get_project(&self) -> Option<String> {
        self.storage.get_project().await
    }

    pub async fn set_project(&self, project: Option<&str>) -> Result<(), SessionError> {
        self.storage.set_project(project).await
    }

    pub async fn get_label(&self, target_id: &str) -> Option<String> {
        self.storage.get_label(target_id).await
    }

    pub async fn set_label(
        &self,
        target_id: &str,
        label: Option<&str>,
    ) -> Result<(), SessionError> {
        self.storage.set_label(target_id, label).await
    }

    pub async fn find_entries(&self, query: &EntryQuery) -> Result<Vec<Entry>, SessionError> {
        query_entries(self.storage.as_ref(), query, None).await
    }

    pub async fn find_entry(&self, query: &EntryQuery) -> Result<Option<Entry>, SessionError> {
        Ok(query_entries(self.storage.as_ref(), query, Some(1))
            .await?
            .into_iter()
            .next())
    }

    pub async fn find_entries_on_branch(
        &self,
        query: &BranchEntryQuery,
    ) -> Result<Vec<Entry>, SessionError> {
        query_branch_entries(self.storage.as_ref(), "main", query, None).await
    }

    pub async fn find_entry_on_branch(
        &self,
        query: &BranchEntryQuery,
    ) -> Result<Option<Entry>, SessionError> {
        Ok(
            query_branch_entries(self.storage.as_ref(), "main", query, Some(1))
                .await?
                .into_iter()
                .next(),
        )
    }

    pub async fn append_message(&self, message: AgentMessage) -> Result<String, SessionError> {
        append_message_to_lane(
            self.storage.as_ref(),
            self.id_generator.as_ref(),
            "main",
            message,
        )
        .await
    }

    pub async fn append_custom_entry(
        &self,
        custom_type: impl Into<String>,
        data: Option<JsonValue>,
    ) -> Result<String, SessionError> {
        append_custom_entry_to_lane(
            self.storage.as_ref(),
            self.id_generator.as_ref(),
            "main",
            custom_type.into(),
            data,
        )
        .await
    }

    pub async fn get_lanes(&self) -> Vec<LanePointer> {
        self.storage.get_lanes().await
    }

    pub async fn create_lane(&self, lane: &str, at: Option<&str>) -> Result<(), SessionError> {
        self.storage.create_lane(lane, at).await
    }

    pub async fn move_lane(&self, lane: &str, to: Option<&str>) -> Result<(), SessionError> {
        self.storage.move_lane(lane, to).await
    }

    /// Append an entry to a lane, preserving the caller-provided id.
    pub async fn append_entry(
        &self,
        entry: ProvisionedEntry,
        lane: &str,
    ) -> Result<Entry, SessionError> {
        assert_json_serializable(&serde_json::to_value(&entry).unwrap_or(JsonValue::Null))?;
        self.storage.append_entry(entry, lane).await
    }

    /// Append a record; the storage assigns `seq` and `timestamp`.
    pub async fn append_record(&self, record: NewRecord) -> Result<LaneRecord, SessionError> {
        assert_json_serializable(&serde_json::to_value(&record).unwrap_or(JsonValue::Null))?;
        self.storage.append_record(record).await
    }

    pub async fn find_records(&self, query: &RecordQuery) -> Result<Vec<LaneRecord>, SessionError> {
        assert_valid_limit(query.limit)?;
        assert_valid_operation_kind(
            query.record_type.as_deref(),
            query.operation_kind.as_deref(),
        )?;
        self.storage.find_records(query).await
    }

    pub async fn find_open_operations(
        &self,
        lane: &str,
        limit: Option<usize>,
    ) -> Result<Vec<OperationStartedRecord>, SessionError> {
        assert_valid_limit(limit)?;
        self.storage.find_open_operations(lane, limit).await
    }

    pub async fn get_log(&self, options: &LogOptions) -> Result<Vec<LogItem>, SessionError> {
        assert_valid_limit(options.limit)?;
        self.storage.get_log(options).await
    }
}

/// Re-export of the mutation type for storage implementors.
pub use crate::session::state::SessionMutation as Mutation;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::memory::InMemorySessionStorage;

    #[tokio::test]
    async fn appends_and_queries_messages() {
        let storage = Arc::new(InMemorySessionStorage::new(SessionMetadata {
            id: "s1".to_owned(),
            created_at: 1,
            parent_session_id: None,
        }));
        let session = Session::new(storage, None);
        let first = session
            .append_message(AgentMessage::user("one", 1))
            .await
            .unwrap();
        let second = session
            .append_message(AgentMessage::user("two", 2))
            .await
            .unwrap();
        assert_eq!(session.get_leaf_id().await.unwrap(), Some(second.clone()));
        let entries = session.find_entries(&EntryQuery::default()).await.unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].id(), second);
        assert_eq!(entries[1].id(), first);
        assert_eq!(session.get_stats().await.message_count, 2);
    }

    #[tokio::test]
    async fn lane_views_scope_appends() {
        let storage = Arc::new(InMemorySessionStorage::new(SessionMetadata {
            id: "s1".to_owned(),
            created_at: 1,
            parent_session_id: None,
        }));
        let session = Session::new(storage, None);
        session.create_lane("alt", None).await.unwrap();
        let view = session.view("alt");
        let leaf = view
            .append_message(AgentMessage::user("alt", 1))
            .await
            .unwrap();
        assert_eq!(view.get_leaf_id().await.unwrap(), Some(leaf));
        assert_eq!(session.get_leaf_id().await.unwrap(), None);
    }

    #[tokio::test]
    async fn rejects_invalid_queries() {
        let storage = Arc::new(InMemorySessionStorage::new(SessionMetadata {
            id: "s1".to_owned(),
            created_at: 1,
            parent_session_id: None,
        }));
        let session = Session::new(storage, None);
        let error = session
            .find_records(&RecordQuery {
                operation_kind: Some("run".to_owned()),
                ..Default::default()
            })
            .await
            .unwrap_err();
        assert_eq!(error.code, SessionErrorCode::InvalidQuery);
    }
}
