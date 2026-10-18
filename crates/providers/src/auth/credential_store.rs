//! Default in-memory credential store. Apps inject persistent stores.
//! Writes are serialized per provider.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use super::types::{Credential, CredentialInfo, CredentialKind, CredentialStore, ModifierFn};

/// In-memory [`CredentialStore`]: one credential per provider, per-provider
/// write serialization through async locks held across the read-modify-write.
#[derive(Default)]
pub struct InMemoryCredentialStore {
    entries: Mutex<HashMap<String, Option<Credential>>>,
    write_locks: Mutex<HashMap<String, Weak<tokio::sync::Mutex<()>>>>,
}

impl InMemoryCredentialStore {
    pub fn new() -> Self {
        Self::default()
    }

    fn write_lock(&self, provider_id: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self
            .write_locks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        locks.retain(|_, lock| lock.strong_count() > 0);
        if let Some(lock) = locks.get(provider_id).and_then(Weak::upgrade) {
            return lock;
        }
        let lock = Arc::new(tokio::sync::Mutex::new(()));
        locks.insert(provider_id.to_owned(), Arc::downgrade(&lock));
        lock
    }
}

#[async_trait]
impl CredentialStore for InMemoryCredentialStore {
    async fn read(
        &self,
        provider_id: &str,
        token: &CancellationToken,
    ) -> Result<Option<Credential>, Box<dyn std::error::Error + Send + Sync>> {
        if token.is_cancelled() {
            return Err(crate::utils::Aborted.into());
        }
        Ok(self
            .entries
            .lock()
            .map_err(|_| "credential store lock poisoned")?
            .get(provider_id)
            .cloned()
            .flatten())
    }

    async fn list(
        &self,
        token: &CancellationToken,
    ) -> Result<Vec<CredentialInfo>, Box<dyn std::error::Error + Send + Sync>> {
        if token.is_cancelled() {
            return Err(crate::utils::Aborted.into());
        }
        let entries = self
            .entries
            .lock()
            .map_err(|_| "credential store lock poisoned")?;
        Ok(entries
            .iter()
            .filter_map(|(provider_id, credential)| {
                credential.as_ref().map(|credential| CredentialInfo {
                    provider_id: provider_id.clone(),
                    kind: match credential {
                        Credential::ApiKey(_) => CredentialKind::ApiKey,
                        Credential::Oauth(_) => CredentialKind::Oauth,
                    },
                })
            })
            .collect())
    }

    async fn modify(
        &self,
        provider_id: &str,
        modifier: ModifierFn,
        token: &CancellationToken,
    ) -> Result<Option<Credential>, Box<dyn std::error::Error + Send + Sync>> {
        if token.is_cancelled() {
            return Err(crate::utils::Aborted.into());
        }
        // Serialize the whole read-modify-write per provider.
        let write_guard = self.write_lock(provider_id);
        let _guard = tokio::select! {
            biased;
            _ = token.cancelled() => return Err(crate::utils::Aborted.into()),
            guard = write_guard.lock() => guard,
        };

        let current = self
            .entries
            .lock()
            .map_err(|_| "credential store lock poisoned")?
            .get(provider_id)
            .cloned()
            .flatten();

        let next = tokio::select! {
            biased;
            _ = token.cancelled() => return Err(crate::utils::Aborted.into()),
            next = modifier(current.clone()) => next?,
        };
        if token.is_cancelled() {
            return Err(crate::utils::Aborted.into());
        }
        let post_write = next.or(current);
        self.entries
            .lock()
            .map_err(|_| "credential store lock poisoned")?
            .insert(provider_id.to_owned(), post_write.clone());
        Ok(post_write)
    }

    async fn delete(
        &self,
        provider_id: &str,
        token: &CancellationToken,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if token.is_cancelled() {
            return Err(crate::utils::Aborted.into());
        }
        let write_guard = self.write_lock(provider_id);
        let _guard = tokio::select! {
            biased;
            _ = token.cancelled() => return Err(crate::utils::Aborted.into()),
            guard = write_guard.lock() => guard,
        };
        self.entries
            .lock()
            .map_err(|_| "credential store lock poisoned")?
            .remove(provider_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::types::{OAuthCredential, modifier};

    #[tokio::test]
    async fn cancellation_while_waiting_for_a_lock_never_runs_the_modifier() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let store = Arc::new(InMemoryCredentialStore::new());
        let lock = store.write_lock("test");
        let _guard = lock.lock().await;
        let invoked = Arc::new(AtomicUsize::new(0));
        let count = invoked.clone();
        let token = CancellationToken::new();
        let cancel = token.clone();
        let task = tokio::spawn(async move {
            store
                .modify(
                    "test",
                    modifier(move |_| {
                        count.fetch_add(1, Ordering::Relaxed);
                        Box::pin(async { Ok(None) })
                    }),
                    &token,
                )
                .await
        });
        tokio::task::yield_now().await;
        cancel.cancel();
        assert!(
            tokio::time::timeout(std::time::Duration::from_secs(1), task)
                .await
                .unwrap()
                .unwrap()
                .is_err()
        );
        assert_eq!(invoked.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn modify_persists_and_returns_credential() {
        let store = InMemoryCredentialStore::new();
        let token = CancellationToken::new();
        let credential = Credential::Oauth(OAuthCredential {
            refresh: "r".to_owned(),
            access: "a".to_owned(),
            expires: 9999999999999,
            extra: None,
        });
        let stored = credential.clone();
        let outcome = store
            .modify(
                "anthropic",
                modifier(move |_| {
                    let stored = stored.clone();
                    Box::pin(async move { Ok(Some(stored)) })
                }),
                &token,
            )
            .await
            .unwrap();
        assert_eq!(outcome, Some(credential));
        assert!(store.read("anthropic", &token).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn none_modifier_keeps_current_entry() {
        let store = InMemoryCredentialStore::new();
        let token = CancellationToken::new();
        let credential = Credential::ApiKey(Default::default());
        let seeded = credential.clone();
        store
            .modify(
                "openai",
                modifier(move |_| {
                    let seeded = seeded.clone();
                    Box::pin(async move { Ok(Some(seeded)) })
                }),
                &token,
            )
            .await
            .unwrap();
        let outcome = store
            .modify("openai", modifier(|_| Box::pin(async { Ok(None) })), &token)
            .await
            .unwrap();
        assert_eq!(outcome, Some(credential));
    }

    #[tokio::test]
    async fn delete_removes_entry() {
        let store = InMemoryCredentialStore::new();
        let token = CancellationToken::new();
        let credential = Credential::ApiKey(Default::default());
        let seeded = credential.clone();
        store
            .modify(
                "openai",
                modifier(move |_| {
                    let seeded = seeded.clone();
                    Box::pin(async move { Ok(Some(seeded)) })
                }),
                &token,
            )
            .await
            .unwrap();
        store.delete("openai", &token).await.unwrap();
        assert!(store.read("openai", &token).await.unwrap().is_none());
    }
}
