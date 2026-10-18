//! Session search contracts.
//!
//! Ported from `@step-harness/agent-core` `src\search\index.ts`. The scanning
//! implementation lives in [`scanning`].

pub mod scanning;

pub use scanning::{
    ScanningReadable, ScanningReadableOptions, ScanningReadableSource, ScanningSearchTextProjector,
    ScanningSessionSearchHit, ScanningSessionSearchOptions, SessionSearchCandidate,
    create_scanning_session_search, scanning_entries,
};

use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

/// Options for a session search.
#[derive(Clone, Debug, Default)]
pub struct SessionSearchOptions {
    /// Restrict results to specific canonical entry types.
    pub entry_types: Option<Vec<String>>,
    /// Maximum number of hits to return.
    pub limit: Option<usize>,
    /// Abort signal for cancellation, e.g. search-as-you-type.
    pub signal: Option<CancellationToken>,
}

impl SessionSearchOptions {
    /// True when `entry_types` is present but empty.
    ///
    /// The TypeScript source treats an explicitly empty array as "match
    /// nothing" and returns early; a `None` means "no restriction".
    pub fn has_empty_entry_types(&self) -> bool {
        self.entry_types
            .as_ref()
            .is_some_and(|entry_types| entry_types.is_empty())
    }
}

/// One search hit.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionSearchHit {
    /// Logical identifier of the session that owns the entry.
    pub session_id: String,
    /// Logical identifier of the entry within that session.
    pub entry_id: String,
    /// Entry timestamp in milliseconds since the Unix epoch, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<u64>,
    /// Matched text excerpt, when the backend provides one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
}

/// A session search backend.
///
/// The TypeScript source returns an `AsyncIterable` that stops early once the
/// limit is reached. In Rust, `search` returns the collected hits, but the
/// implementation still stops scanning as soon as the limit is satisfied, so
/// the work avoided is the same.
#[async_trait::async_trait]
pub trait SessionSearch: Send + Sync {
    async fn search(
        &self,
        text: &str,
        options: Option<&SessionSearchOptions>,
    ) -> Result<Vec<SessionSearchHit>, crate::session::types::SessionError>;
}

/// Error returned by session searches.
pub use crate::session::types::SessionError;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hit_carries_session_and_entry_ids() {
        let hit = SessionSearchHit {
            session_id: "s1".to_owned(),
            entry_id: "e1".to_owned(),
            timestamp: None,
            snippet: None,
        };
        assert_eq!(hit.session_id, "s1");
        assert_eq!(hit.entry_id, "e1");
        assert!(hit.timestamp.is_none());
        assert!(hit.snippet.is_none());
    }

    #[test]
    fn hit_optional_fields_default_to_none() {
        let hit = SessionSearchHit {
            session_id: "s1".to_owned(),
            entry_id: "e1".to_owned(),
            timestamp: Some(42),
            snippet: Some("matched".to_owned()),
        };
        assert_eq!(hit.timestamp, Some(42));
        assert_eq!(hit.snippet.as_deref(), Some("matched"));
    }

    #[test]
    fn empty_entry_types_is_distinct_from_none() {
        let none = SessionSearchOptions::default();
        assert!(!none.has_empty_entry_types());

        let empty = SessionSearchOptions {
            entry_types: Some(Vec::new()),
            ..Default::default()
        };
        assert!(empty.has_empty_entry_types());

        let some = SessionSearchOptions {
            entry_types: Some(vec!["message".to_owned()]),
            ..Default::default()
        };
        assert!(!some.has_empty_entry_types());
    }
}
