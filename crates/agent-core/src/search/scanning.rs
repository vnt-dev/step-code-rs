//! Scanning session search: paged, in-memory scans over session entries.
//!
//! Ported from `@step-harness/agent-core` `src\search\scanning.ts`.
//!
//! The TypeScript source is built from async generators, so hits stream lazily
//! and scanning stops the moment the limit is satisfied. In Rust, [`SessionSearch::search`]
//! collects hits into a `Vec`, but the scan still short-circuits at the limit:
//! [`scan_candidates`] takes a visitor that can stop the walk, so no extra pages
//! are fetched once enough hits have been found.
//!
//! Notable differences from the TypeScript source:
//! - `ScanningReadable` is a trait with three async methods; a blanket-style
//!   adapter [`StorageReadable`] implements it over any [`SessionStorage`].
//! - Async generator sources become boxed futures returning `Vec`s.
//! - Aborting returns a [`SessionError`] instead of throwing an `AbortError`;
//!   see [`abort_error`].
//! - The default text projection serializes the entry with `serde_json`,
//!   matching `JSON.stringify`.

use std::collections::HashSet;
use std::sync::Arc;

use futures::future::BoxFuture;
use serde_json::Value as JsonValue;
use tokio_util::sync::CancellationToken;

use crate::session::facade::{Session, SessionStorage};
use crate::session::types::{
    Entry, EntryCursor, EntryOrder, EntryQuery, SessionError, SessionErrorCode, SessionMetadata,
};

use super::{SessionSearch, SessionSearchHit, SessionSearchOptions};

/// Default page size when neither the query nor the options specify one.
const DEFAULT_PAGE_SIZE: usize = 100;

/// One entry examined by a scanning search.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionSearchCandidate {
    pub entry_id: String,
    pub seq: u64,
    /// Canonical entry type discriminant, e.g. `"message"`.
    pub entry_type: String,
    pub timestamp: u64,
    /// Projected search text for this entry.
    pub text: String,
    /// Extra derived fields; the default projection sets `label` when present.
    pub fields: Option<serde_json::Map<String, JsonValue>>,
}

/// A scanning hit: a [`SessionSearchHit`] with the timestamp and full
/// projected text filled in.
///
/// The TypeScript source declares this as a distinct interface extending
/// `SessionSearchHit`; since the Rust [`SessionSearchHit`] already carries the
/// optional `timestamp` and `snippet` fields, the scanning hit is the same
/// shape and is produced by [`create_default_hit`].
pub type ScanningSessionSearchHit = SessionSearchHit;

/// Read-only view of a session used by the scanning search.
///
/// The TypeScript source derives this from `Pick<SessionStorage, "getMetadata"
/// | "findEntries" | "getLabel">`. It is a trait here so tests can supply an
/// in-memory stand-in without a full storage implementation.
#[async_trait::async_trait]
pub trait ScanningReadable: Send + Sync {
    /// Metadata flavor this readable produces.
    type Metadata: Clone + Send + Sync + 'static;

    async fn get_metadata(&self) -> Self::Metadata;
    async fn find_entries(&self, query: &EntryQuery) -> Result<Vec<Entry>, SessionError>;
    async fn get_label(&self, id: &str) -> Option<String>;
}

/// Projects an entry into searchable text.
pub type ScanningSearchTextProjector<M> =
    Arc<dyn Fn(&M, &Entry, Option<&str>) -> String + Send + Sync>;

/// Decides whether a candidate matches the normalized query text.
pub type ScanningMatchFn<M> = Arc<dyn Fn(&str, &SessionSearchCandidate, &M) -> bool + Send + Sync>;

/// Builds the hit returned for a matching candidate.
pub type ScanningCreateHitFn<M> =
    Arc<dyn Fn(&M, &SessionSearchCandidate) -> SessionSearchHit + Send + Sync>;

/// Computes opaque options handed to a dynamic source, from the query text and
/// the caller's search options.
pub type ScanningSourceOptionsFn =
    Arc<dyn Fn(&str, &SessionSearchOptions) -> Option<JsonValue> + Send + Sync>;

/// A dynamic source of readables, evaluated per search.
pub type ScanningReadableSourceFn<M> = Arc<
    dyn Fn(
            Option<JsonValue>,
        )
            -> BoxFuture<'static, Result<Vec<Arc<dyn ScanningReadable<Metadata = M>>>, SessionError>>
        + Send
        + Sync,
>;

/// Where a scanning search gets its readables.
pub enum ScanningReadableSource<M> {
    /// A fixed list of readables.
    Static(Vec<Arc<dyn ScanningReadable<Metadata = M>>>),
    /// A source function evaluated per search.
    Dynamic(ScanningReadableSourceFn<M>),
}

/// Options for [`scanning_entries`] and [`create_scanning_session_search`].
///
/// `Default` is implemented manually so it does not require `M: Default`.
#[derive(Clone)]
pub struct ScanningReadableOptions<M> {
    /// Text projection; defaults to [`default_project_text`].
    pub project_text: Option<ScanningSearchTextProjector<M>>,
    /// Page size for entry queries; defaults to [`DEFAULT_PAGE_SIZE`].
    pub page_size: Option<usize>,
}

impl<M> Default for ScanningReadableOptions<M> {
    fn default() -> Self {
        ScanningReadableOptions {
            project_text: None,
            page_size: None,
        }
    }
}

/// Options for [`create_scanning_session_search`].
///
/// `Default` is implemented manually so it does not require `M: Default`.
#[derive(Clone)]
pub struct ScanningSessionSearchOptions<M> {
    /// Text projection; defaults to [`default_project_text`].
    pub project_text: Option<ScanningSearchTextProjector<M>>,
    /// Page size for entry queries; defaults to [`DEFAULT_PAGE_SIZE`].
    pub page_size: Option<usize>,
    /// Computes source options from the query; ignored for static sources.
    pub source_options: Option<ScanningSourceOptionsFn>,
    /// Match predicate; defaults to a lowercase substring test.
    pub match_fn: Option<ScanningMatchFn<M>>,
    /// Hit builder; defaults to [`create_default_hit`].
    pub create_hit: Option<ScanningCreateHitFn<M>>,
}

impl<M> Default for ScanningSessionSearchOptions<M> {
    fn default() -> Self {
        ScanningSessionSearchOptions {
            project_text: None,
            page_size: None,
            source_options: None,
            match_fn: None,
            create_hit: None,
        }
    }
}

/// Default text projection: the serialized entry, plus the label when set.
pub fn default_project_text<M>(_metadata: &M, entry: &Entry, label: Option<&str>) -> String {
    let serialized = serde_json::to_string(entry).unwrap_or_default();
    match label {
        Some(label) => format!("{serialized} {label}"),
        None => serialized,
    }
}

/// Default hit builder: the full projected text as the snippet.
pub fn create_default_hit<M>(metadata: &M, candidate: &SessionSearchCandidate) -> SessionSearchHit
where
    M: HasSessionId,
{
    SessionSearchHit {
        session_id: metadata.session_id().to_owned(),
        entry_id: candidate.entry_id.clone(),
        timestamp: Some(candidate.timestamp),
        snippet: Some(candidate.text.clone()),
    }
}

/// Session-id accessor for metadata flavors.
///
/// The TypeScript source reads `metadata.id` on `SessionMetadata`. This trait
/// lets [`create_default_hit`] work with any metadata shape without forcing
/// every implementor to use [`SessionMetadata`] directly.
pub trait HasSessionId {
    fn session_id(&self) -> &str;
}

impl HasSessionId for SessionMetadata {
    fn session_id(&self) -> &str {
        &self.id
    }
}

/// Error returned when a search is aborted through its signal.
///
/// Cancellation is represented by [`SessionErrorCode::Aborted`].
pub fn abort_error() -> SessionError {
    SessionError::new(SessionErrorCode::Aborted, "The operation was aborted")
}

fn is_aborted(signal: Option<&CancellationToken>) -> bool {
    signal.is_some_and(|signal| signal.is_cancelled())
}

/// Effective page size: never below 1, since a limit of 0 is rejected by
/// [`SessionStorage::find_entries`].
fn effective_page_size(limit: Option<usize>, page_size: Option<usize>) -> usize {
    limit.or(page_size).unwrap_or(DEFAULT_PAGE_SIZE).max(1)
}

/// Page through one readable's entries, invoking `on_candidate` for each.
///
/// Returns `false` from `on_candidate` to stop the scan early.
async fn scan_candidates<R, M, F>(
    readable: &R,
    metadata: &M,
    options: &ScanningReadableOptions<M>,
    query: &ScanQuery<'_>,
    mut on_candidate: F,
) -> Result<(), SessionError>
where
    R: ScanningReadable<Metadata = M> + ?Sized,
    F: FnMut(&SessionSearchCandidate) -> bool,
{
    let project_text = options
        .project_text
        .clone()
        .unwrap_or_else(|| Arc::new(default_project_text::<M>));
    let page_size = effective_page_size(None, options.page_size);
    let mut after_seq: u64 = 0;
    let entry_types: Option<HashSet<&str>> = query
        .entry_types
        .map(|types| types.iter().map(String::as_str).collect());

    loop {
        let entries = readable
            .find_entries(&EntryQuery {
                order: Some(EntryOrder::OldestFirst),
                limit: Some(page_size),
                cursor: Some(EntryCursor { after_seq }),
                // A single requested type is pushed down to the storage query.
                entry_type: match query.entry_types {
                    Some(types) if types.len() == 1 => Some(types[0].clone()),
                    _ => None,
                },
                ..Default::default()
            })
            .await?;
        if entries.is_empty() {
            break;
        }
        for entry in &entries {
            if let Some(entry_types) = &entry_types
                && !entry_types.contains(entry.entry_type())
            {
                continue;
            }
            let label = readable.get_label(entry.id()).await;
            let candidate = SessionSearchCandidate {
                entry_id: entry.id().to_owned(),
                seq: entry.seq(),
                entry_type: entry.entry_type().to_owned(),
                timestamp: entry.timestamp(),
                text: project_text(metadata, entry, label.as_deref()),
                fields: label.as_ref().map(|label| {
                    serde_json::Map::from_iter([(
                        "label".to_owned(),
                        JsonValue::String(label.clone()),
                    )])
                }),
            };
            if !on_candidate(&candidate) {
                return Ok(());
            }
        }
        after_seq = entries.last().map(|entry| entry.seq()).unwrap_or(after_seq);
        if entries.len() < page_size {
            break;
        }
    }
    Ok(())
}

/// Entry-type filter shared by [`scan_candidates`] callers.
struct ScanQuery<'a> {
    entry_types: Option<&'a [String]>,
}
/// Enumerate every candidate in a readable, in storage order.
///
/// Unlike [`create_scanning_session_search`], this performs no text matching
/// and no limit handling: it is the raw entry walk with text projection
/// applied.
pub async fn scanning_entries<R, M>(
    readable: &R,
    options: &ScanningReadableOptions<M>,
) -> Result<Vec<SessionSearchCandidate>, SessionError>
where
    R: ScanningReadable<Metadata = M> + ?Sized,
{
    let metadata = readable.get_metadata().await;
    let mut candidates = Vec::new();
    scan_candidates(
        readable,
        &metadata,
        options,
        &ScanQuery { entry_types: None },
        |candidate| {
            candidates.push(candidate.clone());
            true
        },
    )
    .await?;
    Ok(candidates)
}

/// A scanning [`SessionSearch`] over a fixed or dynamic set of readables.
pub struct ScanningSessionSearch<M> {
    source: ScanningReadableSource<M>,
    options: ScanningSessionSearchOptions<M>,
}

impl<M> ScanningSessionSearch<M> {
    /// The readables this search scans, when the source is static.
    pub fn static_source(
        readables: Vec<Arc<dyn ScanningReadable<Metadata = M>>>,
    ) -> ScanningReadableSource<M> {
        ScanningReadableSource::Static(readables)
    }
}

/// Create a scanning session search.
pub fn create_scanning_session_search<M>(
    source: ScanningReadableSource<M>,
    options: ScanningSessionSearchOptions<M>,
) -> ScanningSessionSearch<M>
where
    M: Clone + Send + Sync + 'static,
{
    ScanningSessionSearch { source, options }
}

#[async_trait::async_trait]
impl<M> SessionSearch for ScanningSessionSearch<M>
where
    M: Clone + Send + Sync + HasSessionId + 'static,
{
    async fn search(
        &self,
        text: &str,
        search_options: Option<&SessionSearchOptions>,
    ) -> Result<Vec<SessionSearchHit>, SessionError> {
        let options = search_options.cloned().unwrap_or_default();

        // Query normalization: trim then lowercase.
        let normalized_text = text.trim().to_lowercase();
        if normalized_text.is_empty() {
            return Ok(Vec::new());
        }
        if options.limit == Some(0) {
            return Ok(Vec::new());
        }
        if options.has_empty_entry_types() {
            return Ok(Vec::new());
        }

        let readables: Vec<Arc<dyn ScanningReadable<Metadata = M>>> = match &self.source {
            ScanningReadableSource::Static(readables) => readables.clone(),
            ScanningReadableSource::Dynamic(source) => {
                let source_options = self
                    .options
                    .source_options
                    .as_ref()
                    .and_then(|compute| compute(&normalized_text, &options));
                if is_aborted(options.signal.as_ref()) {
                    return Err(abort_error());
                }
                source(source_options).await?
            }
        };

        let readable_options = ScanningReadableOptions {
            project_text: self.options.project_text.clone(),
            page_size: self.options.page_size,
        };
        let match_fn = self.options.match_fn.clone();
        let create_hit = self.options.create_hit.clone();
        let limit = options.limit;
        let signal = options.signal.clone();

        let mut hits: Vec<SessionSearchHit> = Vec::new();
        let mut seen_session_ids: HashSet<String> = HashSet::new();

        for readable in &readables {
            if is_aborted(signal.as_ref()) {
                return Err(abort_error());
            }
            let metadata = readable.get_metadata().await;
            if !seen_session_ids.insert(metadata.session_id().to_owned()) {
                return Err(SessionError::new(
                    SessionErrorCode::InvalidQuery,
                    format!("Duplicate sessionId: {}", metadata.session_id()),
                ));
            }

            let mut reached_limit = false;
            scan_candidates(
                readable.as_ref(),
                &metadata,
                &readable_options,
                &ScanQuery {
                    entry_types: options.entry_types.as_deref(),
                },
                |candidate| {
                    if is_aborted(signal.as_ref()) {
                        // Surfaced after the scan unwinds; see the check below.
                        return false;
                    }
                    if let Some(entry_types) = options.entry_types.as_deref()
                        && !entry_types.contains(&candidate.entry_type)
                    {
                        return true;
                    }
                    let matches = match &match_fn {
                        Some(match_fn) => match_fn(&normalized_text, candidate, &metadata),
                        None => default_match(&normalized_text, candidate),
                    };
                    if !matches {
                        return true;
                    }
                    let hit = match &create_hit {
                        Some(create_hit) => create_hit(&metadata, candidate),
                        None => create_default_hit(&metadata, candidate),
                    };
                    hits.push(hit);
                    if let Some(limit) = limit
                        && hits.len() >= limit
                    {
                        reached_limit = true;
                        return false;
                    }
                    true
                },
            )
            .await?;

            if is_aborted(signal.as_ref()) {
                return Err(abort_error());
            }
            if reached_limit {
                break;
            }
        }

        Ok(hits)
    }
}

/// Default match: lowercase substring test on the projected text.
fn default_match(query_text: &str, candidate: &SessionSearchCandidate) -> bool {
    candidate.text.to_lowercase().contains(query_text)
}

// ---------------------------------------------------------------------------
// Adapters
// ---------------------------------------------------------------------------

/// Adapter implementing [`ScanningReadable`] over any [`SessionStorage`].
pub struct StorageReadable<S> {
    storage: S,
}

impl<S> StorageReadable<S> {
    pub fn new(storage: S) -> Self {
        StorageReadable { storage }
    }
}

#[async_trait::async_trait]
impl<S> ScanningReadable for StorageReadable<S>
where
    S: SessionStorage + Send + Sync,
{
    type Metadata = S::Metadata;

    async fn get_metadata(&self) -> Self::Metadata {
        self.storage.get_metadata().await
    }

    async fn find_entries(&self, query: &EntryQuery) -> Result<Vec<Entry>, SessionError> {
        self.storage.find_entries(query).await
    }

    async fn get_label(&self, id: &str) -> Option<String> {
        self.storage.get_label(id).await
    }
}

/// [`Session`] already exposes exactly the three read methods, so it implements
/// [`ScanningReadable`] directly.
#[async_trait::async_trait]
impl<M> ScanningReadable for Session<M>
where
    M: Clone + Send + Sync + HasSessionId + 'static,
{
    type Metadata = M;

    async fn get_metadata(&self) -> M {
        self.get_metadata().await
    }

    async fn find_entries(&self, query: &EntryQuery) -> Result<Vec<Entry>, SessionError> {
        self.find_entries(query).await
    }

    async fn get_label(&self, id: &str) -> Option<String> {
        self.get_label(id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::memory::InMemorySessionStorage;
    use crate::session::types::ProvisionedEntry;
    use crate::types::AgentMessage;

    /// Metadata flavor used by the in-memory test stand-ins.
    #[derive(Clone, Debug, PartialEq)]
    struct TestMetadata {
        id: String,
    }

    impl HasSessionId for TestMetadata {
        fn session_id(&self) -> &str {
            &self.id
        }
    }

    /// In-memory readable with a fixed candidate list.
    struct TestReadable {
        metadata: TestMetadata,
        entries: Vec<Entry>,
        labels: std::collections::HashMap<String, String>,
    }

    #[async_trait::async_trait]
    impl ScanningReadable for TestReadable {
        type Metadata = TestMetadata;

        async fn get_metadata(&self) -> TestMetadata {
            self.metadata.clone()
        }

        async fn find_entries(&self, _query: &EntryQuery) -> Result<Vec<Entry>, SessionError> {
            // Honor the cursor and order so paging is exercised.
            let after_seq = _query
                .cursor
                .as_ref()
                .map(|cursor| cursor.after_seq)
                .unwrap_or(0);
            let limit = _query.limit.unwrap_or(usize::MAX);
            let ordered: Vec<Entry> = match _query.order {
                Some(EntryOrder::NewestFirst) => {
                    let mut entries = self.entries.clone();
                    entries.reverse();
                    entries
                }
                _ => self.entries.clone(),
            };
            Ok(ordered
                .into_iter()
                .filter(|entry| entry.seq() > after_seq)
                .take(limit)
                .collect())
        }

        async fn get_label(&self, id: &str) -> Option<String> {
            self.labels.get(id).cloned()
        }
    }

    fn message_entry(id: &str, seq: u64, text: &str) -> Entry {
        Entry::Message(crate::session::types::MessageEntry {
            id: id.to_owned(),
            seq,
            parent_id: None,
            timestamp: 1000 + seq,
            message: AgentMessage::user(text, 1000 + seq),
            terminate: false,
        })
    }

    fn readable(id: &str, entries: Vec<Entry>) -> Arc<TestReadable> {
        Arc::new(TestReadable {
            metadata: TestMetadata { id: id.to_owned() },
            entries,
            labels: std::collections::HashMap::new(),
        })
    }

    fn readables<M, R: ScanningReadable<Metadata = M> + 'static>(
        items: Vec<Arc<R>>,
    ) -> ScanningReadableSource<M> {
        ScanningReadableSource::Static(
            items
                .into_iter()
                .map(|item| item as Arc<dyn ScanningReadable<Metadata = M>>)
                .collect(),
        )
    }

    #[tokio::test]
    async fn scanning_entries_projects_text_and_pages() {
        let entries = vec![
            message_entry("e1", 1, "alpha"),
            message_entry("e2", 2, "beta"),
            message_entry("e3", 3, "gamma"),
        ];
        let source = readable("s1", entries);
        let options = ScanningReadableOptions {
            page_size: Some(2),
            ..Default::default()
        };
        let candidates = scanning_entries(source.as_ref(), &options).await.unwrap();
        assert_eq!(candidates.len(), 3);
        assert_eq!(candidates[0].entry_id, "e1");
        assert_eq!(candidates[2].entry_id, "e3");
        // Default projection includes the serialized entry text.
        assert!(candidates[0].text.contains("alpha"));
        assert_eq!(candidates[0].entry_type, "message");
        assert_eq!(candidates[0].timestamp, 1001);
    }

    #[tokio::test]
    async fn scanning_entries_includes_labels_in_fields_and_text() {
        let mut labels = std::collections::HashMap::new();
        labels.insert("e1".to_owned(), "important".to_owned());
        let source = Arc::new(TestReadable {
            metadata: TestMetadata {
                id: "s1".to_owned(),
            },
            entries: vec![message_entry("e1", 1, "alpha")],
            labels,
        });
        let candidates = scanning_entries(source.as_ref(), &ScanningReadableOptions::default())
            .await
            .unwrap();
        assert_eq!(candidates.len(), 1);
        assert!(candidates[0].text.ends_with(" important"));
        assert_eq!(
            candidates[0].fields.as_ref().and_then(|f| f.get("label")),
            Some(&JsonValue::String("important".to_owned()))
        );
    }

    #[tokio::test]
    async fn search_matches_substrings_case_insensitively() {
        let entries = vec![
            message_entry("e1", 1, "Find the NEEDLE here"),
            message_entry("e2", 2, "nothing to see"),
            message_entry("e3", 3, "a needle in a haystack"),
        ];
        let source = readables(vec![readable("s1", entries)]);
        let search =
            create_scanning_session_search(source, ScanningSessionSearchOptions::default());
        let hits = search.search("needle", None).await.unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].entry_id, "e1");
        assert_eq!(hits[1].entry_id, "e3");
        assert_eq!(hits[0].session_id, "s1");
    }

    #[tokio::test]
    async fn search_normalizes_query_and_short_circuits() {
        let entries = vec![message_entry("e1", 1, "alpha beta")];
        let search = create_scanning_session_search(
            readables(vec![readable("s1", entries)]),
            ScanningSessionSearchOptions::default(),
        );

        // Empty and whitespace-only queries return nothing.
        assert!(search.search("", None).await.unwrap().is_empty());
        assert!(search.search("   ", None).await.unwrap().is_empty());
        // Surrounding whitespace is trimmed before matching.
        assert_eq!(search.search("  alpha  ", None).await.unwrap().len(), 1);
        // A zero limit returns nothing.
        let zero = SessionSearchOptions {
            limit: Some(0),
            ..Default::default()
        };
        assert!(
            search
                .search("alpha", Some(&zero))
                .await
                .unwrap()
                .is_empty()
        );
        // An explicitly empty entry-type list matches nothing.
        let empty_types = SessionSearchOptions {
            entry_types: Some(Vec::new()),
            ..Default::default()
        };
        assert!(
            search
                .search("alpha", Some(&empty_types))
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn search_respects_limit_and_stops_scanning() {
        let entries: Vec<Entry> = (1..=5)
            .map(|index| message_entry(&format!("e{index}"), index, "match me"))
            .collect();
        let source = readable("s1", entries);
        let search = create_scanning_session_search(
            readables(vec![source]),
            ScanningSessionSearchOptions {
                page_size: Some(1),
                ..Default::default()
            },
        );
        let options = SessionSearchOptions {
            limit: Some(2),
            ..Default::default()
        };
        let hits = search.search("match", Some(&options)).await.unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].entry_id, "e1");
        assert_eq!(hits[1].entry_id, "e2");
    }

    #[tokio::test]
    async fn search_filters_by_entry_type() {
        let entries = vec![
            message_entry("e1", 1, "match me"),
            Entry::Custom(crate::session::types::CustomEntry {
                id: "e2".to_owned(),
                seq: 2,
                parent_id: None,
                timestamp: 1002,
                custom_type: "note".to_owned(),
                data: None,
            }),
        ];
        let search = create_scanning_session_search(
            readables(vec![readable("s1", entries)]),
            ScanningSessionSearchOptions::default(),
        );

        // "match me" only appears in the message entry's projected text.
        let message_only = SessionSearchOptions {
            entry_types: Some(vec!["message".to_owned()]),
            ..Default::default()
        };
        let hits = search.search("match", Some(&message_only)).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].entry_id, "e1");

        // The custom entry is filtered out even though the text matches.
        let custom_only = SessionSearchOptions {
            entry_types: Some(vec!["custom".to_owned()]),
            ..Default::default()
        };
        assert!(
            search
                .search("match", Some(&custom_only))
                .await
                .unwrap()
                .is_empty()
        );

        // Without a filter both entries are candidates, but only one matches.
        assert_eq!(search.search("match", None).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn search_honors_abort_signal() {
        let entries = vec![message_entry("e1", 1, "match me")];
        let search = create_scanning_session_search(
            readables(vec![readable("s1", entries)]),
            ScanningSessionSearchOptions::default(),
        );
        let signal = CancellationToken::new();
        signal.cancel();
        let options = SessionSearchOptions {
            signal: Some(signal),
            ..Default::default()
        };
        let error = search.search("match", Some(&options)).await.unwrap_err();
        assert_eq!(error.message, "The operation was aborted");
    }

    #[tokio::test]
    async fn search_rejects_duplicate_session_ids() {
        let entries = vec![message_entry("e1", 1, "match me")];
        let search = create_scanning_session_search(
            readables(vec![
                readable("s1", entries.clone()),
                readable("s1", entries),
            ]),
            ScanningSessionSearchOptions::default(),
        );
        let error = search.search("match", None).await.unwrap_err();
        assert!(error.message.contains("Duplicate sessionId"));
    }

    #[tokio::test]
    async fn search_uses_custom_match_and_hit_builders() {
        let entries = vec![message_entry("e1", 1, "alpha")];
        let source = readables(vec![readable("s1", entries)]);
        let search = create_scanning_session_search(
            source,
            ScanningSessionSearchOptions {
                match_fn: Some(Arc::new(|query, candidate, _metadata| {
                    candidate.entry_id == query
                })),
                create_hit: Some(Arc::new(|_metadata, candidate| SessionSearchHit {
                    session_id: "custom".to_owned(),
                    entry_id: candidate.entry_id.clone(),
                    timestamp: None,
                    snippet: None,
                })),
                ..Default::default()
            },
        );
        let hits = search.search("e1", None).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].session_id, "custom");
        assert!(hits[0].snippet.is_none());
    }

    #[tokio::test]
    async fn session_implements_scanning_readable() {
        let storage = Arc::new(InMemorySessionStorage::new(SessionMetadata {
            id: "s1".to_owned(),
            created_at: 1,
            parent_session_id: None,
        }));
        let session = Session::new(storage, None);
        session
            .append_message(AgentMessage::user("hello world", 1))
            .await
            .unwrap();
        session.set_label("ignored-missing", Some("x")).await.ok();

        let candidates = scanning_entries(&session, &ScanningReadableOptions::default())
            .await
            .unwrap();
        assert_eq!(candidates.len(), 1);
        assert!(candidates[0].text.contains("hello world"));

        let search = create_scanning_session_search(
            ScanningReadableSource::Static(vec![Arc::new(session)]),
            ScanningSessionSearchOptions::default(),
        );
        let hits = search.search("hello", None).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].session_id, "s1");
    }

    #[tokio::test]
    async fn storage_readable_adapts_session_storage() {
        // `StorageReadable` borrows the storage, so the entries are appended
        // before wrapping it.
        let storage = InMemorySessionStorage::new(SessionMetadata {
            id: "s2".to_owned(),
            created_at: 1,
            parent_session_id: None,
        });
        storage
            .append_entry(
                ProvisionedEntry::Message {
                    id: "e1".to_owned(),
                    message: AgentMessage::user("adapt me", 1),
                    terminate: false,
                },
                "main",
            )
            .await
            .unwrap();
        let readable = StorageReadable::new(storage);
        assert_eq!(readable.get_metadata().await.id, "s2");
        let candidates = scanning_entries(&readable, &ScanningReadableOptions::default())
            .await
            .unwrap();
        assert_eq!(candidates.len(), 1);
        assert!(candidates[0].text.contains("adapt me"));
    }

    #[test]
    fn default_match_is_lowercase_substring() {
        let candidate = SessionSearchCandidate {
            entry_id: "e1".to_owned(),
            seq: 1,
            entry_type: "message".to_owned(),
            timestamp: 1,
            text: "Hello WORLD".to_owned(),
            fields: None,
        };
        assert!(default_match("hello", &candidate));
        assert!(default_match("world", &candidate));
        assert!(!default_match("nope", &candidate));
    }

    #[test]
    fn abort_error_names_its_cause() {
        assert_eq!(abort_error().message, "The operation was aborted");
    }
}
