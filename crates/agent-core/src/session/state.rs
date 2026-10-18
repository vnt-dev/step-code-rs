//! The session state machine: an append-only mutation reducer.
//!
//! Ported from `@step-harness/agent-core` `src\harness\session\state.ts`.
//! Every durable write is a [`SessionMutation`] carrying a globally
//! consecutive `seq`; [`SessionState::apply_mutation`] is the only state
//! transition and rejects anything inconsistent.

use serde::{Deserialize, Serialize};
use step_harness_providers::types::Usage;

use crate::session::types::{
    BranchBounds, BranchEntryQuery, Entry, EntryOrder, EntryQuery, ForkOptions, ForkPosition,
    LanePointer, LaneRecord, LogItem, LogOptions, OperationStartedRecord, RecordQuery,
    SessionError, SessionErrorCode, SessionStats, StepAttemptRecord, StepKind,
};

/// A durable session mutation: the unit of persistence.
///
/// The TypeScript source models name and label facts as two variants of the
/// same `kind: "fact"` tag, discriminated by a `fact` field; a serde tag cannot
/// repeat, so both shapes share one variant with optional fields.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum SessionMutation {
    #[serde(rename_all = "camelCase")]
    Entry {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        lane: Option<String>,
        entry: Entry,
    },
    #[serde(rename_all = "camelCase")]
    Record { record: LaneRecord },
    #[serde(rename_all = "camelCase")]
    Lane {
        seq: u64,
        lane: String,
        leaf_id: Option<String>,
    },
    #[serde(rename_all = "camelCase")]
    Fact {
        seq: u64,
        fact: FactKind,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        target_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        project: Option<String>,
    },
}

impl SessionMutation {
    /// A name fact mutation.
    pub fn name(seq: u64, name: Option<String>) -> Self {
        SessionMutation::Fact {
            seq,
            fact: FactKind::Name,
            name,
            target_id: None,
            label: None,
            project: None,
        }
    }

    /// A label fact mutation.
    pub fn label(seq: u64, target_id: String, label: Option<String>) -> Self {
        SessionMutation::Fact {
            seq,
            fact: FactKind::Label,
            name: None,
            target_id: Some(target_id),
            label,
            project: None,
        }
    }

    /// A project fact mutation.
    pub fn project(seq: u64, project: Option<String>) -> Self {
        SessionMutation::Fact {
            seq,
            fact: FactKind::Project,
            name: None,
            target_id: None,
            label: None,
            project,
        }
    }
}

/// Which global fact a mutation sets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactKind {
    Name,
    Label,
    Project,
}

fn invalid_mutation(message: String) -> SessionError {
    SessionError::new(
        SessionErrorCode::InvalidEntry,
        format!("Invalid session mutation: {message}"),
    )
}

/// In-memory session state replayed from mutations.
pub struct SessionState {
    sequence: u64,
    used_ids: std::collections::HashSet<String>,
    entries: Vec<Entry>,
    entries_by_id: std::collections::HashMap<String, Entry>,
    records: Vec<LaneRecord>,
    /// Open operations per lane, in start order (newest last).
    open_operations_by_lane: std::collections::HashMap<String, Vec<OperationStartedRecord>>,
    /// Lane pointers in creation order; `main` always exists.
    lanes: Vec<(String, Option<String>)>,
    log: Vec<LogItem>,
    stats: SessionStats,
    name: Option<String>,
    project: Option<String>,
    labels: std::collections::HashMap<String, String>,
}

impl Default for SessionState {
    fn default() -> Self {
        SessionState {
            sequence: 0,
            used_ids: std::collections::HashSet::new(),
            entries: Vec::new(),
            entries_by_id: std::collections::HashMap::new(),
            records: Vec::new(),
            open_operations_by_lane: std::collections::HashMap::new(),
            lanes: vec![("main".to_owned(), None)],
            log: Vec::new(),
            stats: SessionStats::default(),
            name: None,
            project: None,
            labels: std::collections::HashMap::new(),
        }
    }
}

impl SessionState {
    pub fn next_sequence(&self) -> u64 {
        self.sequence + 1
    }

    pub fn get_lanes(&self) -> Vec<LanePointer> {
        self.lanes
            .iter()
            .map(|(lane, leaf_id)| LanePointer {
                lane: lane.clone(),
                leaf_id: leaf_id.clone(),
            })
            .collect()
    }

    pub fn require_lane(&self, lane: &str) -> Result<Option<String>, SessionError> {
        self.lanes
            .iter()
            .find(|(name, _)| name == lane)
            .map(|(_, leaf_id)| leaf_id.clone())
            .ok_or_else(|| {
                SessionError::new(
                    SessionErrorCode::InvalidLane,
                    format!("Lane not found: {lane}"),
                )
            })
    }

    pub fn validate_new_lane(&self, lane: &str) -> Result<(), SessionError> {
        if self.lanes.iter().any(|(name, _)| name == lane) {
            return Err(SessionError::new(
                SessionErrorCode::AlreadyExists,
                format!("Lane already exists: {lane}"),
            ));
        }
        Ok(())
    }

    pub fn validate_target(&self, target_id: Option<&str>) -> Result<(), SessionError> {
        if let Some(target_id) = target_id
            && !self.entries_by_id.contains_key(target_id)
        {
            return Err(SessionError::new(
                SessionErrorCode::NotFound,
                format!("Entry not found: {target_id}"),
            ));
        }
        Ok(())
    }

    pub fn validate_unused_id(&self, id: &str) -> Result<(), SessionError> {
        if self.used_ids.contains(id) {
            return Err(SessionError::new(
                SessionErrorCode::AlreadyExists,
                format!("Session id already exists: {id}"),
            ));
        }
        Ok(())
    }

    /// Validate a proposed mutation without changing state. Durable storage calls
    /// this under its writer lock before committing any bytes.
    pub fn validate_mutation(&self, mutation: &SessionMutation) -> Result<(), SessionError> {
        let seq = match mutation {
            SessionMutation::Entry { entry, .. } => entry.seq(),
            SessionMutation::Record { record } => record.seq(),
            SessionMutation::Lane { seq, .. } | SessionMutation::Fact { seq, .. } => *seq,
        };
        if self.sequence.checked_add(1) != Some(seq) {
            return Err(invalid_mutation(format!("has non-consecutive seq {seq}")));
        }
        match mutation {
            SessionMutation::Entry { lane, entry } => {
                self.validate_unused_id(entry.id())?;
                if let Some(lane) = lane {
                    let parent = self.require_lane(lane)?;
                    if entry.parent_id() != parent.as_deref() {
                        return Err(invalid_mutation(
                            "does not chain to the lane leaf".to_owned(),
                        ));
                    }
                }
                self.validate_target(entry.parent_id())?;
            }
            SessionMutation::Record { record } => {
                self.require_lane(record.lane())?;
                self.validate_unused_id(record.id())?;
                if let LaneRecord::StepAttempt(attempt) = record
                    && !attempt.is_valid()
                {
                    return Err(invalid_mutation("has an invalid step attempt".to_owned()));
                }
            }
            SessionMutation::Lane { leaf_id, .. } => self.validate_target(leaf_id.as_deref())?,
            SessionMutation::Fact {
                fact: FactKind::Label,
                target_id,
                ..
            } => {
                let target = target_id
                    .as_deref()
                    .ok_or_else(|| invalid_mutation("label fact is missing a target".to_owned()))?;
                self.validate_target(Some(target))?;
            }
            SessionMutation::Fact { .. } => {}
        }
        Ok(())
    }

    /// Apply one mutation. Rejects non-consecutive sequences, duplicate ids,
    /// broken chains, missing lanes/parents, and invalid step attempts.
    pub fn apply_mutation(&mut self, mutation: &SessionMutation) -> Result<(), SessionError> {
        let seq = match mutation {
            SessionMutation::Entry { entry, .. } => entry.seq(),
            SessionMutation::Record { record } => record.seq(),
            SessionMutation::Lane { seq, .. } | SessionMutation::Fact { seq, .. } => *seq,
        };
        if self.sequence.checked_add(1) != Some(seq) {
            return Err(invalid_mutation(format!("has non-consecutive seq {seq}")));
        }

        match mutation {
            SessionMutation::Entry { lane, entry } => {
                if self.used_ids.contains(entry.id()) {
                    return Err(invalid_mutation(format!(
                        "contains duplicate id {}",
                        entry.id()
                    )));
                }
                if let Some(lane) = lane {
                    let Some((_, leaf_id)) = self.lanes.iter().find(|(name, _)| name == lane)
                    else {
                        return Err(invalid_mutation(format!("references missing lane {lane}")));
                    };
                    if entry.parent_id() != leaf_id.as_deref() {
                        return Err(invalid_mutation(
                            "does not chain to the lane leaf".to_owned(),
                        ));
                    }
                }
                if let Some(parent_id) = entry.parent_id()
                    && !self.entries_by_id.contains_key(parent_id)
                {
                    return Err(invalid_mutation(format!(
                        "references missing parent {parent_id}"
                    )));
                }
                self.sequence = seq;
                self.used_ids.insert(entry.id().to_owned());
                self.entries.push(entry.clone());
                self.entries_by_id
                    .insert(entry.id().to_owned(), entry.clone());
                if let Some(lane) = lane
                    && let Some((_, leaf_id)) = self.lanes.iter_mut().find(|(name, _)| name == lane)
                {
                    *leaf_id = Some(entry.id().to_owned());
                }
                self.log.push(LogItem::Entry {
                    seq,
                    entry: entry.clone(),
                });
                if matches!(entry, Entry::Message(_)) {
                    self.stats.message_count += 1;
                }
            }
            SessionMutation::Record { record } => {
                if !self.lanes.iter().any(|(name, _)| name == record.lane()) {
                    return Err(invalid_mutation(format!(
                        "references missing lane {}",
                        record.lane()
                    )));
                }
                if self.used_ids.contains(record.id()) {
                    return Err(invalid_mutation(format!(
                        "contains duplicate id {}",
                        record.id()
                    )));
                }
                if let LaneRecord::StepAttempt(attempt) = record
                    && !attempt.is_valid()
                {
                    return Err(invalid_mutation("has an invalid step attempt".to_owned()));
                }
                self.sequence = seq;
                self.used_ids.insert(record.id().to_owned());
                self.records.push(record.clone());
                match record {
                    LaneRecord::OperationStarted(started) => {
                        self.open_operations_by_lane
                            .entry(started.lane.clone())
                            .or_default()
                            .push(started.clone());
                    }
                    LaneRecord::OperationFinished(finished) => {
                        if let Some(open) = self.open_operations_by_lane.get_mut(&finished.lane) {
                            open.retain(|started| started.id != finished.run_id);
                        }
                    }
                    LaneRecord::Usage(usage) => {
                        accumulate_usage(&mut self.stats, &usage.usage);
                    }
                    _ => {}
                }
                self.log.push(LogItem::Record {
                    seq,
                    record: record.clone(),
                });
            }
            SessionMutation::Lane { lane, leaf_id, .. } => {
                if let Some(leaf_id) = leaf_id
                    && !self.entries_by_id.contains_key(leaf_id)
                {
                    return Err(invalid_mutation(format!(
                        "references missing lane target {leaf_id}"
                    )));
                }
                self.sequence = seq;
                if let Some((_, existing)) = self.lanes.iter_mut().find(|(name, _)| name == lane) {
                    *existing = leaf_id.clone();
                } else {
                    self.lanes.push((lane.clone(), leaf_id.clone()));
                }
                self.log.push(LogItem::Lane {
                    seq,
                    lane: lane.clone(),
                    leaf_id: leaf_id.clone(),
                });
            }
            SessionMutation::Fact {
                fact,
                name,
                target_id,
                label,
                project,
                ..
            } => match fact {
                FactKind::Name => {
                    self.sequence = seq;
                    self.name = name.clone();
                    self.log.push(LogItem::Fact {
                        seq,
                        fact: crate::session::types::FactName::Name,
                        name: name.clone(),
                        project: None,
                    });
                }
                FactKind::Project => {
                    self.sequence = seq;
                    self.project = project.clone();
                    self.log.push(LogItem::Fact {
                        seq,
                        fact: crate::session::types::FactName::Project,
                        name: None,
                        project: project.clone(),
                    });
                }
                FactKind::Label => {
                    let Some(target_id) = target_id else {
                        return Err(invalid_mutation(
                            "label fact is missing a target".to_owned(),
                        ));
                    };
                    if !self.entries_by_id.contains_key(target_id) {
                        return Err(invalid_mutation(format!(
                            "references missing label target {target_id}"
                        )));
                    }
                    self.sequence = seq;
                    match label {
                        Some(label) => {
                            self.labels.insert(target_id.clone(), label.clone());
                        }
                        None => {
                            self.labels.remove(target_id);
                        }
                    }
                    self.log.push(LogItem::Label {
                        seq,
                        target_id: target_id.clone(),
                        label: label.clone(),
                    });
                }
            },
        }
        Ok(())
    }

    pub fn get_entry(&self, id: &str) -> Option<&Entry> {
        self.entries_by_id.get(id)
    }

    pub fn find_entries(&self, query: &EntryQuery) -> Result<Vec<Entry>, SessionError> {
        crate::session::types::assert_valid_limit(query.limit)?;
        let mut results = Vec::new();
        for entry in ordered(&self.entries, query.order) {
            if !matches_entry_query(entry, query) {
                continue;
            }
            results.push(entry.clone());
            if query.limit == Some(results.len()) {
                break;
            }
        }
        Ok(results)
    }

    pub fn find_entries_on_branch(
        &self,
        query: &BranchEntryQuery,
    ) -> Result<Vec<Entry>, SessionError> {
        crate::session::types::assert_valid_limit(query.limit)?;
        let Some(start) = query.bounds.start.as_deref() else {
            return Ok(Vec::new());
        };
        let mut results = Vec::new();
        let entry_query = EntryQuery {
            entry_type: query.entry_type.clone(),
            custom_type: query.custom_type.clone(),
            order: query.order,
            limit: query.limit,
            cursor: query.cursor.clone(),
        };
        if query.order == Some(EntryOrder::OldestFirst) {
            let mut path: Vec<&Entry> = self.walk_to_root(start, Some(&query.bounds))?;
            path.reverse();
            for entry in path {
                let reached_bound = Some(entry.id()) == query.bounds.stop_at_id.as_deref()
                    || Some(entry.entry_type()) == query.bounds.stop_at_type.as_deref();
                if matches_entry_query(entry, &entry_query) {
                    results.push(entry.clone());
                }
                if reached_bound || query.limit == Some(results.len()) {
                    break;
                }
            }
        } else {
            for entry in self.walk_to_root(start, Some(&query.bounds))? {
                if matches_entry_query(entry, &entry_query) {
                    results.push(entry.clone());
                }
                if query.limit == Some(results.len()) {
                    break;
                }
            }
        }
        Ok(results)
    }

    pub fn find_records(&self, query: &RecordQuery) -> Result<Vec<LaneRecord>, SessionError> {
        crate::session::types::assert_valid_limit(query.limit)?;
        let mut results = Vec::new();
        for record in ordered(&self.records, query.order) {
            if !matches_record_query(record, query) {
                continue;
            }
            results.push(record.clone());
            if query.limit == Some(results.len()) {
                break;
            }
        }
        Ok(results)
    }

    /// Unfinished operation starts, newest first.
    pub fn find_open_operations(
        &self,
        lane: &str,
        limit: Option<usize>,
    ) -> Result<Vec<OperationStartedRecord>, SessionError> {
        crate::session::types::assert_valid_limit(limit)?;
        let mut open: Vec<OperationStartedRecord> = self
            .open_operations_by_lane
            .get(lane)
            .cloned()
            .unwrap_or_default();
        open.reverse();
        Ok(match limit {
            Some(limit) => open.into_iter().take(limit).collect(),
            None => open,
        })
    }

    pub fn get_log(&self, options: &LogOptions) -> Result<Vec<LogItem>, SessionError> {
        crate::session::types::assert_valid_limit(options.limit)?;
        let mut results = Vec::new();
        for item in &self.log {
            if options
                .after_seq
                .is_some_and(|after_seq| item.seq() <= after_seq)
            {
                continue;
            }
            results.push(item.clone());
            if options.limit == Some(results.len()) {
                break;
            }
        }
        Ok(results)
    }

    pub fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// The session's project assignment, when one is set.
    pub fn get_project(&self) -> Option<&str> {
        self.project.as_deref()
    }

    pub fn get_label(&self, id: &str) -> Option<&str> {
        self.labels.get(id).map(String::as_str)
    }

    pub fn get_stats(&self) -> SessionStats {
        self.stats
    }

    /// Build the mutations that reconstruct this state under a fork.
    ///
    /// Branch forks copy the path to the target entry plus the name and the
    /// labels on the copied path; tree forks copy everything. Records are never
    /// copied, so token and cost stats restart from zero while `message_count`
    /// is recomputed from the copied entries.
    pub fn create_fork_mutations(
        &self,
        options: &ForkOptions,
    ) -> Result<Vec<SessionMutation>, SessionError> {
        let (copied_entries, fork_lanes) = match options {
            ForkOptions::Tree => (
                self.find_entries(&EntryQuery {
                    order: Some(EntryOrder::OldestFirst),
                    ..Default::default()
                })?,
                self.get_lanes(),
            ),
            ForkOptions::Branch { entry_id, position } => {
                let selected_entry_id = match entry_id {
                    Some(entry_id) => Some(entry_id.clone()),
                    None => self.require_lane("main")?,
                };
                let mut target_id: Option<String> = None;
                if let Some(selected) = &selected_entry_id {
                    let Some(entry) = self.get_entry(selected) else {
                        return Err(SessionError::new(
                            SessionErrorCode::InvalidForkTarget,
                            format!("Fork target is not a message entry: {selected}"),
                        ));
                    };
                    if !matches!(entry, Entry::Message(_)) {
                        return Err(SessionError::new(
                            SessionErrorCode::InvalidForkTarget,
                            format!("Fork target is not a message entry: {selected}"),
                        ));
                    }
                    let position = position.unwrap_or(if entry_id.is_none() {
                        ForkPosition::At
                    } else {
                        ForkPosition::Before
                    });
                    target_id = Some(match position {
                        ForkPosition::At => entry.id().to_owned(),
                        ForkPosition::Before => entry.parent_id().unwrap_or_default().to_owned(),
                    });
                }
                let copied = match &target_id {
                    Some(target_id) => self.find_entries_on_branch(&BranchEntryQuery {
                        bounds: BranchBounds {
                            start: Some(target_id.clone()),
                            ..Default::default()
                        },
                        order: Some(EntryOrder::OldestFirst),
                        ..Default::default()
                    })?,
                    None => Vec::new(),
                };
                (
                    copied,
                    vec![LanePointer {
                        lane: "main".to_owned(),
                        leaf_id: target_id,
                    }],
                )
            }
        };

        let mut mutations = Vec::new();
        let mut sequence = 1u64;
        for entry in &copied_entries {
            let mut forked = entry.clone();
            set_entry_seq(&mut forked, sequence);
            sequence += 1;
            mutations.push(SessionMutation::Entry {
                lane: None,
                entry: forked,
            });
        }
        for pointer in &fork_lanes {
            mutations.push(SessionMutation::Lane {
                seq: sequence,
                lane: pointer.lane.clone(),
                leaf_id: pointer.leaf_id.clone(),
            });
            sequence += 1;
        }
        if self.name.is_some() {
            mutations.push(SessionMutation::name(sequence, self.name.clone()));
            sequence += 1;
        }
        if let Some(project) = &self.project {
            mutations.push(SessionMutation::project(sequence, Some(project.clone())));
            sequence += 1;
        }
        for entry in &copied_entries {
            if let Some(label) = self.labels.get(entry.id()) {
                mutations.push(SessionMutation::label(
                    sequence,
                    entry.id().to_owned(),
                    Some(label.clone()),
                ));
                sequence += 1;
            }
        }
        Ok(mutations)
    }

    /// Walk from `start` toward the root, honoring stop bounds (inclusive).
    fn walk_to_root(
        &self,
        start: &str,
        bounds: Option<&BranchBounds>,
    ) -> Result<Vec<&Entry>, SessionError> {
        let mut path = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let Some(mut current) = self.get_entry(start) else {
            return Err(SessionError::new(
                SessionErrorCode::NotFound,
                format!("Entry not found: {start}"),
            ));
        };
        loop {
            if !visited.insert(current.id()) {
                return Err(SessionError::new(
                    SessionErrorCode::InvalidEntry,
                    format!("Session branch contains a cycle at {}", current.id()),
                ));
            }
            path.push(current);
            let reached_bound = bounds.is_some_and(|bounds| {
                Some(current.id()) == bounds.stop_at_id.as_deref()
                    || Some(current.entry_type()) == bounds.stop_at_type.as_deref()
            });
            let Some(parent_id) = current.parent_id() else {
                break;
            };
            if reached_bound {
                break;
            }
            let Some(parent) = self.get_entry(parent_id) else {
                return Err(SessionError::new(
                    SessionErrorCode::InvalidEntry,
                    format!("Entry not found: {parent_id}"),
                ));
            };
            current = parent;
        }
        Ok(path)
    }
}

fn set_entry_seq(entry: &mut Entry, seq: u64) {
    match entry {
        Entry::Message(entry) => entry.seq = seq,
        Entry::ModelChange(entry) => entry.seq = seq,
        Entry::ThinkingLevelChange(entry) => entry.seq = seq,
        Entry::ActiveToolsChange(entry) => entry.seq = seq,
        Entry::Compaction(entry) => entry.seq = seq,
        Entry::BranchSummary(entry) => entry.seq = seq,
        Entry::Custom(entry) => entry.seq = seq,
    }
}

fn accumulate_usage(stats: &mut SessionStats, usage: &Usage) {
    stats.cached_tokens = stats.cached_tokens.saturating_add(usage.cache_read);
    stats.uncached_tokens = stats
        .uncached_tokens
        .saturating_add(usage.input.saturating_add(usage.cache_write));
    stats.total_tokens = stats.total_tokens.saturating_add(usage.total_tokens);
    stats.cost_total += usage.cost.total;
}

fn ordered<T>(items: &[T], order: Option<EntryOrder>) -> impl Iterator<Item = &T> {
    let newest_first = order != Some(EntryOrder::OldestFirst);
    let count = items.len();
    (0..count)
        .filter_map(move |index| {
            let index = if newest_first {
                count - 1 - index
            } else {
                index
            };
            items.get(index)
        })
        .fuse()
}

fn matches_entry_query(entry: &Entry, query: &EntryQuery) -> bool {
    if query
        .entry_type
        .as_deref()
        .is_some_and(|entry_type| entry.entry_type() != entry_type)
    {
        return false;
    }
    if let Some(custom_type) = &query.custom_type {
        match entry {
            Entry::Custom(custom) if &custom.custom_type == custom_type => {}
            _ => return false,
        }
    }
    if let Some(cursor) = &query.cursor {
        let after = match query.order {
            Some(EntryOrder::OldestFirst) => entry.seq() > cursor.after_seq,
            _ => entry.seq() < cursor.after_seq,
        };
        if !after {
            return false;
        }
    }
    true
}

fn matches_record_query(record: &LaneRecord, query: &RecordQuery) -> bool {
    if query
        .lane
        .as_deref()
        .is_some_and(|lane| record.lane() != lane)
    {
        return false;
    }
    if query
        .record_type
        .as_deref()
        .is_some_and(|record_type| record.record_type() != record_type)
    {
        return false;
    }
    if let Some(run_id) = &query.run_id
        && record.run_id() != Some(run_id.as_str())
    {
        return false;
    }
    if let Some(operation_kind) = &query.operation_kind {
        match record {
            LaneRecord::OperationStarted(started) => {
                let kind = match &started.intent {
                    crate::session::types::OperationIntent::Run { .. } => "run",
                    crate::session::types::OperationIntent::Compaction { .. } => "compaction",
                    crate::session::types::OperationIntent::Navigation { .. } => "navigation",
                };
                if kind != operation_kind {
                    return false;
                }
            }
            _ => return false,
        }
    }
    if query
        .after_seq
        .is_some_and(|after_seq| record.seq() <= after_seq)
    {
        return false;
    }
    true
}

impl LogItem {
    pub fn seq(&self) -> u64 {
        match self {
            LogItem::Entry { seq, .. }
            | LogItem::Record { seq, .. }
            | LogItem::Lane { seq, .. }
            | LogItem::Fact { seq, .. }
            | LogItem::Label { seq, .. } => *seq,
        }
    }
}

/// Ensure a step attempt record satisfies the compaction-reason invariant.
pub fn validate_step_attempt(record: &StepAttemptRecord) -> bool {
    match record.step {
        StepKind::Compaction => record.compaction_reason.is_some(),
        _ => record.compaction_reason.is_none(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::types::{BranchEntryQuery, EntryQuery, ProvisionedEntry};
    use crate::types::AgentMessage;

    fn message_entry(id: &str, seq: u64, parent_id: Option<String>) -> Entry {
        ProvisionedEntry::Message {
            id: id.to_owned(),
            message: AgentMessage::user("hi", 1),
            terminate: false,
        }
        .into_entry(parent_id, seq, 1000 + seq)
    }

    #[test]
    fn applies_entries_in_sequence() {
        let mut state = SessionState::default();
        state
            .apply_mutation(&SessionMutation::Entry {
                lane: Some("main".to_owned()),
                entry: message_entry("e1", 1, None),
            })
            .unwrap();
        state
            .apply_mutation(&SessionMutation::Entry {
                lane: Some("main".to_owned()),
                entry: message_entry("e2", 2, Some("e1".to_owned())),
            })
            .unwrap();
        assert_eq!(state.get_stats().message_count, 2);
        assert_eq!(state.require_lane("main").unwrap(), Some("e2".to_owned()));
    }

    #[test]
    fn rejects_non_consecutive_sequences() {
        let mut state = SessionState::default();
        let error = state
            .apply_mutation(&SessionMutation::Entry {
                lane: Some("main".to_owned()),
                entry: message_entry("e1", 2, None),
            })
            .unwrap_err();
        assert_eq!(error.code, SessionErrorCode::InvalidEntry);
    }

    #[test]
    fn rejects_broken_chains_and_duplicates() {
        let mut state = SessionState::default();
        state
            .apply_mutation(&SessionMutation::Entry {
                lane: Some("main".to_owned()),
                entry: message_entry("e1", 1, None),
            })
            .unwrap();
        // Wrong parent: does not chain to the lane leaf.
        let error = state
            .apply_mutation(&SessionMutation::Entry {
                lane: Some("main".to_owned()),
                entry: message_entry("e2", 2, Some("missing".to_owned())),
            })
            .unwrap_err();
        assert_eq!(error.code, SessionErrorCode::InvalidEntry);
        // Duplicate id.
        let error = state
            .apply_mutation(&SessionMutation::Entry {
                lane: None,
                entry: message_entry("e1", 2, Some("e1".to_owned())),
            })
            .unwrap_err();
        assert_eq!(error.code, SessionErrorCode::InvalidEntry);
    }

    #[test]
    fn walks_branches_to_the_root() {
        let mut state = SessionState::default();
        for (index, id) in ["e1", "e2", "e3"].iter().enumerate() {
            let parent = if index == 0 {
                None
            } else {
                Some(["e1", "e2"][index - 1].to_owned())
            };
            state
                .apply_mutation(&SessionMutation::Entry {
                    lane: Some("main".to_owned()),
                    entry: message_entry(id, index as u64 + 1, parent),
                })
                .unwrap();
        }
        let path = state
            .find_entries_on_branch(&BranchEntryQuery {
                bounds: crate::session::types::BranchBounds {
                    start: Some("e3".to_owned()),
                    ..Default::default()
                },
                ..Default::default()
            })
            .unwrap();
        assert_eq!(
            path.iter().map(Entry::id).collect::<Vec<_>>(),
            vec!["e3", "e2", "e1"]
        );
        let oldest_first = state
            .find_entries_on_branch(&BranchEntryQuery {
                bounds: crate::session::types::BranchBounds {
                    start: Some("e3".to_owned()),
                    stop_at_id: Some("e2".to_owned()),
                    ..Default::default()
                },
                order: Some(EntryOrder::OldestFirst),
                ..Default::default()
            })
            .unwrap();
        // The scan stops after the first bound match, inclusive.
        assert_eq!(
            oldest_first.iter().map(Entry::id).collect::<Vec<_>>(),
            vec!["e2"]
        );
    }

    #[test]
    fn rejects_invalid_queries() {
        let state = SessionState::default();
        let error = state
            .find_entries(&EntryQuery {
                limit: Some(0),
                ..Default::default()
            })
            .unwrap_err();
        assert_eq!(error.code, SessionErrorCode::InvalidQuery);
    }

    #[test]
    fn fork_mutations_renumber_sequences() {
        let mut state = SessionState::default();
        for (index, id) in ["e1", "e2"].iter().enumerate() {
            let parent = if index == 0 {
                None
            } else {
                Some("e1".to_owned())
            };
            state
                .apply_mutation(&SessionMutation::Entry {
                    lane: Some("main".to_owned()),
                    entry: message_entry(id, index as u64 + 1, parent),
                })
                .unwrap();
        }
        state
            .apply_mutation(&SessionMutation::name(3, Some("forked".to_owned())))
            .unwrap();
        let mutations = state
            .create_fork_mutations(&ForkOptions::Branch {
                entry_id: Some("e2".to_owned()),
                position: Some(ForkPosition::At),
            })
            .unwrap();
        // Two entries, one lane pointer, one name fact.
        assert_eq!(mutations.len(), 4);
        let sequences: Vec<u64> = mutations
            .iter()
            .map(|mutation| match mutation {
                SessionMutation::Entry { entry, .. } => entry.seq(),
                SessionMutation::Record { record } => record.seq(),
                SessionMutation::Lane { seq, .. } | SessionMutation::Fact { seq, .. } => *seq,
            })
            .collect();
        assert_eq!(sequences, vec![1, 2, 3, 4]);
    }
}
