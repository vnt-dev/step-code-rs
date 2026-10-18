//! Persistent model catalogs keyed by provider id.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::types::Model;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelsStoreEntry {
    pub models: Vec<Model>,
    /// Unix timestamp (ms) from the remote catalog's Last-Modified header.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_modified: Option<u64>,
    /// Unix timestamp (ms) of the last completed remote check.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checked_at: Option<u64>,
    /// Opaque validator from the remote catalog's ETag header, stored verbatim
    /// (quotes included) and echoed back as If-None-Match.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub etag: Option<String>,
}

impl ModelsStoreEntry {
    /// Filter restored entries down to one provider's models.
    pub fn models_for_provider(&self, provider: &str) -> Vec<Model> {
        self.models
            .iter()
            .filter(|model| model.provider.as_str() == provider)
            .cloned()
            .collect()
    }
}

/// Errors surfaced by [`ModelsStore`] implementations; `models(_)` uses serde.
pub type ModelsStoreError = Box<dyn std::error::Error + Send + Sync>;

/// Persistent model catalogs keyed by provider id.
#[async_trait]
pub trait ModelsStore: Send + Sync {
    async fn read(
        &self,
        provider_id: &str,
        token: &CancellationToken,
    ) -> Result<Option<ModelsStoreEntry>, ModelsStoreError>;
    async fn write(
        &self,
        provider_id: &str,
        entry: ModelsStoreEntry,
        token: &CancellationToken,
    ) -> Result<(), ModelsStoreError>;
    async fn delete(
        &self,
        provider_id: &str,
        token: &CancellationToken,
    ) -> Result<(), ModelsStoreError>;
}

/// In-memory [`ModelsStore`].
#[derive(Default)]
pub struct InMemoryModelsStore {
    entries: std::sync::Mutex<std::collections::HashMap<String, ModelsStoreEntry>>,
}

impl InMemoryModelsStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl ModelsStore for InMemoryModelsStore {
    async fn read(
        &self,
        provider_id: &str,
        token: &CancellationToken,
    ) -> Result<Option<ModelsStoreEntry>, ModelsStoreError> {
        if token.is_cancelled() {
            return Err(crate::utils::Aborted.into());
        }
        Ok(self
            .entries
            .lock()
            .map_err(|_| "models store lock poisoned")?
            .get(provider_id)
            .cloned())
    }

    async fn write(
        &self,
        provider_id: &str,
        entry: ModelsStoreEntry,
        token: &CancellationToken,
    ) -> Result<(), ModelsStoreError> {
        if token.is_cancelled() {
            return Err(crate::utils::Aborted.into());
        }
        self.entries
            .lock()
            .map_err(|_| "models store lock poisoned")?
            .insert(provider_id.to_owned(), entry);
        Ok(())
    }

    async fn delete(
        &self,
        provider_id: &str,
        token: &CancellationToken,
    ) -> Result<(), ModelsStoreError> {
        if token.is_cancelled() {
            return Err(crate::utils::Aborted.into());
        }
        self.entries
            .lock()
            .map_err(|_| "models store lock poisoned")?
            .remove(provider_id);
        Ok(())
    }
}

/// Convenience: typed model slice from an entry.
pub fn entry_models(entry: &ModelsStoreEntry) -> &[Model] {
    entry.models.as_slice()
}
