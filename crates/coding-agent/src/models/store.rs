//! File-backed [`ModelsStore`](step_harness_providers::models_store::ModelsStore)
//! caching remote model catalogs in `models-store.json`.
//!
//! The store is a JSON object keyed by provider id; each entry carries the
//! fetched models plus freshness metadata echoed back as `If-None-Match`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Mutex;

use step_harness_providers::models_store::{ModelsStore, ModelsStoreEntry, ModelsStoreError};

/// Error surfaced by the file-backed models store.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct StoreError(pub String);

impl From<std::io::Error> for StoreError {
    fn from(error: std::io::Error) -> Self {
        StoreError(error.to_string())
    }
}

impl From<serde_json::Error> for StoreError {
    fn from(error: serde_json::Error) -> Self {
        StoreError(error.to_string())
    }
}

fn store_error(message: impl Into<String>) -> ModelsStoreError {
    Box::new(StoreError(message.into()))
}
use tokio_util::sync::CancellationToken;

/// Persists provider model catalogs to a single JSON file.
pub struct FileModelsStore {
    path: PathBuf,
    write_lock: Mutex<()>,
}

impl std::fmt::Debug for FileModelsStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileModelsStore")
            .field("path", &self.path)
            .finish()
    }
}

impl FileModelsStore {
    /// Open the store at `path` (created on first write).
    pub fn new(path: impl Into<PathBuf>) -> Self {
        FileModelsStore {
            path: path.into(),
            write_lock: Mutex::new(()),
        }
    }

    fn load(&self) -> BTreeMap<String, ModelsStoreEntry> {
        let Ok(bytes) = std::fs::read(&self.path) else {
            return BTreeMap::new();
        };
        let Ok(text) = String::from_utf8(bytes) else {
            return BTreeMap::new();
        };
        serde_json::from_str(text.trim_start_matches('\u{feff}')).unwrap_or_default()
    }

    fn save(&self, entries: &BTreeMap<String, ModelsStoreEntry>) -> Result<(), ModelsStoreError> {
        let _guard = self.write_lock.lock().expect("models store lock poisoned");
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|error| store_error(format!("cannot create {parent:?}: {error}")))?;
        }
        let text = serde_json::to_string_pretty(entries)
            .map_err(|error| store_error(format!("cannot serialize: {error}")))?;
        let temp = self
            .path
            .with_extension(format!("json.tmp.{}", std::process::id()));
        std::fs::write(&temp, text)
            .map_err(|error| store_error(format!("cannot write {temp:?}: {error}")))?;
        std::fs::rename(&temp, &self.path).map_err(|error| {
            let _ = std::fs::remove_file(&temp);
            store_error(format!("cannot replace {:?}: {error}", self.path))
        })
    }
}

#[async_trait::async_trait]
impl ModelsStore for FileModelsStore {
    async fn read(
        &self,
        provider_id: &str,
        _token: &CancellationToken,
    ) -> Result<Option<ModelsStoreEntry>, ModelsStoreError> {
        Ok(self.load().remove(provider_id))
    }

    async fn write(
        &self,
        provider_id: &str,
        entry: ModelsStoreEntry,
        _token: &CancellationToken,
    ) -> Result<(), ModelsStoreError> {
        let mut entries = self.load();
        entries.insert(provider_id.to_owned(), entry);
        self.save(&entries)
    }

    async fn delete(
        &self,
        provider_id: &str,
        _token: &CancellationToken,
    ) -> Result<(), ModelsStoreError> {
        let mut entries = self.load();
        if entries.remove(provider_id).is_some() {
            self.save(&entries)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn round_trips_entries() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileModelsStore::new(dir.path().join("models-store.json"));
        let token = CancellationToken::new();
        assert!(store.read("step", &token).await.unwrap().is_none());

        let entry = ModelsStoreEntry {
            models: Vec::new(),
            last_modified: Some(42),
            checked_at: Some(7),
            etag: Some("\"abc\"".to_owned()),
        };
        store.write("step", entry, &token).await.unwrap();
        let read = store.read("step", &token).await.unwrap().unwrap();
        assert_eq!(read.last_modified, Some(42));
        assert_eq!(read.etag.as_deref(), Some("\"abc\""));

        store.delete("step", &token).await.unwrap();
        assert!(store.read("step", &token).await.unwrap().is_none());
    }
}
