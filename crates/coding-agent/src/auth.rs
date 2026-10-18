//! The file-backed `auth.json` credential store.
//!
//! Ported functionally from `@step-harness/coding-agent`
//! `src/core/auth-storage.ts`. The on-disk format is a JSON object keyed by
//! provider id; each value is a type-tagged [`Credential`]. The file is
//! written atomically (temp file + rename) with 0600 permissions on Unix.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde_json::Value as Json;
use step_harness_providers::auth::{Credential, CredentialInfo, CredentialKind, ModifierFn};
use tokio_util::sync::CancellationToken;

/// File-backed [`CredentialStore`](step_harness_providers::auth::CredentialStore).
pub struct FileCredentialStore {
    path: PathBuf,
    /// In-process serialization; disk writes are atomic so a crash mid-write
    /// can never leave a half-written credential file behind.
    write_lock: Mutex<()>,
}

/// Serialize one credential to its `auth.json` shape.
///
/// The providers' [`Credential`] enum is internally tagged and its payloads
/// are internally tagged too, so a direct `to_value` would emit conflicting
/// `type` fields. The wire shape is written explicitly instead.
fn credential_to_json(credential: &Credential) -> Result<Json, AuthStoreError> {
    let value = match credential {
        Credential::ApiKey(api_key) => {
            let mut map = serde_json::Map::new();
            map.insert("type".to_owned(), Json::String("api_key".to_owned()));
            if let Some(key) = &api_key.key {
                map.insert("key".to_owned(), Json::String(key.clone()));
            }
            if let Some(env) = &api_key.env {
                map.insert("env".to_owned(), serde_json::to_value(env)?);
            }
            Json::Object(map)
        }
        Credential::Oauth(oauth) => {
            let mut map = serde_json::Map::new();
            map.insert("type".to_owned(), Json::String("oauth".to_owned()));
            map.insert("access".to_owned(), Json::String(oauth.access.clone()));
            map.insert("refresh".to_owned(), Json::String(oauth.refresh.clone()));
            map.insert("expires".to_owned(), Json::Number(oauth.expires.into()));
            if let Some(extra) = &oauth.extra {
                for (key, value) in extra {
                    map.insert(key.clone(), value.clone());
                }
            }
            Json::Object(map)
        }
    };
    Ok(value)
}

/// Parse one credential from its `auth.json` shape.
fn credential_from_json(provider_id: &str, raw: &Json) -> Result<Credential, AuthStoreError> {
    let invalid = |error: serde_json::Error| {
        AuthStoreError(format!(
            "invalid auth.json credential for provider \"{provider_id}\": {error}"
        ))
    };
    let kind = raw.get("type").and_then(Json::as_str).ok_or_else(|| {
        AuthStoreError(format!(
            "invalid auth.json credential for provider \"{provider_id}\": missing type"
        ))
    })?;
    match kind {
        "api_key" => Ok(Credential::ApiKey(
            serde_json::from_value(raw.clone()).map_err(invalid)?,
        )),
        "oauth" => Ok(Credential::Oauth(
            serde_json::from_value(raw.clone()).map_err(invalid)?,
        )),
        other => Err(AuthStoreError(format!(
            "invalid auth.json credential for provider \"{provider_id}\": unknown type \"{other}\""
        ))),
    }
}

/// Error type for auth store operations.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct AuthStoreError(pub String);

impl From<std::io::Error> for AuthStoreError {
    fn from(error: std::io::Error) -> Self {
        AuthStoreError(error.to_string())
    }
}

impl From<serde_json::Error> for AuthStoreError {
    fn from(error: serde_json::Error) -> Self {
        AuthStoreError(error.to_string())
    }
}

fn boxed(error: AuthStoreError) -> Box<dyn std::error::Error + Send + Sync> {
    Box::new(error)
}

/// Alias kept for callers that talk about "the auth store".
pub type AuthStore = FileCredentialStore;

impl FileCredentialStore {
    /// Open (or create on first write) the store at `path`.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        FileCredentialStore {
            path: path.into(),
            write_lock: Mutex::new(()),
        }
    }

    /// The backing file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Read and parse the credential map. A missing file is an empty map.
    fn load(&self) -> Result<BTreeMap<String, Credential>, AuthStoreError> {
        let text = match std::fs::read(&self.path) {
            Ok(bytes) => String::from_utf8(bytes)
                .map_err(|error| AuthStoreError(format!("invalid UTF-8: {error}")))?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(BTreeMap::new());
            }
            Err(error) => return Err(AuthStoreError(format!("cannot read auth.json: {error}"))),
        };
        let text = text.trim_start_matches('\u{feff}');
        if text.trim().is_empty() {
            return Ok(BTreeMap::new());
        }
        let value: Json = serde_json::from_str(text)?;
        let map = value
            .as_object()
            .ok_or_else(|| AuthStoreError("auth.json is not a JSON object".to_owned()))?;
        let mut credentials = BTreeMap::new();
        for (provider_id, raw) in map {
            let credential = credential_from_json(provider_id, raw)?;
            credentials.insert(provider_id.clone(), credential);
        }
        Ok(credentials)
    }

    /// Persist the credential map atomically with restrictive permissions.
    fn save(&self, credentials: &BTreeMap<String, Credential>) -> Result<(), AuthStoreError> {
        let _guard = self.write_lock.lock().expect("auth write lock poisoned");
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut document = serde_json::Map::new();
        for (provider_id, credential) in credentials {
            document.insert(provider_id.clone(), credential_to_json(credential)?);
        }
        let text = serde_json::to_string_pretty(&Json::Object(document))?;
        let temp = self
            .path
            .with_extension(format!("json.tmp.{}", std::process::id()));
        std::fs::write(&temp, text)?;
        restrict_permissions(&temp);
        std::fs::rename(&temp, &self.path)?;
        Ok(())
    }
}

/// Best-effort 0600 on the credential file (no-op on Windows).
fn restrict_permissions(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
    #[cfg(not(unix))]
    let _ = path;
}

#[async_trait::async_trait]
impl step_harness_providers::auth::CredentialStore for FileCredentialStore {
    async fn read(
        &self,
        provider_id: &str,
        _token: &CancellationToken,
    ) -> Result<Option<Credential>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(self.load().map_err(boxed)?.remove(provider_id))
    }

    async fn list(
        &self,
        _token: &CancellationToken,
    ) -> Result<Vec<CredentialInfo>, Box<dyn std::error::Error + Send + Sync>> {
        let credentials = self.load().map_err(boxed)?;
        Ok(credentials
            .into_iter()
            .map(|(provider_id, credential)| CredentialInfo {
                provider_id,
                kind: match credential {
                    Credential::ApiKey(_) => CredentialKind::ApiKey,
                    Credential::Oauth(_) => CredentialKind::Oauth,
                },
            })
            .collect())
    }

    async fn modify(
        &self,
        provider_id: &str,
        modifier: ModifierFn,
        _token: &CancellationToken,
    ) -> Result<Option<Credential>, Box<dyn std::error::Error + Send + Sync>> {
        let mut credentials = self.load().map_err(boxed)?;
        let current = credentials.get(provider_id).cloned();
        let outcome = modifier(current).await?;
        match outcome {
            Some(credential) => {
                credentials.insert(provider_id.to_owned(), credential.clone());
                self.save(&credentials).map_err(boxed)?;
                Ok(Some(credential))
            }
            None => Ok(None),
        }
    }

    async fn delete(
        &self,
        provider_id: &str,
        _token: &CancellationToken,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut credentials = self.load().map_err(boxed)?;
        if credentials.remove(provider_id).is_some() {
            self.save(&credentials).map_err(boxed)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use step_harness_providers::auth::{ApiKeyCredential, CredentialStore as _, OAuthCredential};

    fn store_in(name: &str) -> (tempfile::TempDir, FileCredentialStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = FileCredentialStore::new(dir.path().join(format!("{name}.json")));
        (dir, store)
    }

    #[tokio::test]
    async fn stores_and_reads_api_keys() {
        let (_dir, store) = store_in("api-key");
        let token = CancellationToken::new();
        store
            .modify(
                "anthropic",
                Arc::new(|_current| {
                    Box::pin(async {
                        Ok(Some(Credential::ApiKey(ApiKeyCredential {
                            key: Some("sk-test".to_owned()),
                            env: None,
                        })))
                    })
                }),
                &token,
            )
            .await
            .unwrap();
        let credential = store.read("anthropic", &token).await.unwrap().unwrap();
        assert_eq!(
            credential.as_api_key().unwrap().key.as_deref(),
            Some("sk-test")
        );

        let infos = store.list(&token).await.unwrap();
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].provider_id, "anthropic");
        assert_eq!(infos[0].kind, CredentialKind::ApiKey);
    }

    #[tokio::test]
    async fn modify_with_none_leaves_entry_unchanged() {
        let (_dir, store) = store_in("noop");
        let token = CancellationToken::new();
        store
            .modify(
                "step",
                Arc::new(|_| {
                    Box::pin(async {
                        Ok(Some(Credential::Oauth(OAuthCredential {
                            refresh: "step-static-credential".to_owned(),
                            access: "token".to_owned(),
                            expires: u64::MAX,
                            extra: None,
                        })))
                    })
                }),
                &token,
            )
            .await
            .unwrap();
        store
            .modify("step", Arc::new(|_| Box::pin(async { Ok(None) })), &token)
            .await
            .unwrap();
        assert!(store.read("step", &token).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn delete_removes_entry() {
        let (_dir, store) = store_in("delete");
        let token = CancellationToken::new();
        store
            .modify(
                "openai",
                Arc::new(|_| {
                    Box::pin(async {
                        Ok(Some(Credential::ApiKey(ApiKeyCredential {
                            key: Some("k".to_owned()),
                            env: None,
                        })))
                    })
                }),
                &token,
            )
            .await
            .unwrap();
        store.delete("openai", &token).await.unwrap();
        assert!(store.read("openai", &token).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn missing_file_reads_as_empty() {
        let (_dir, store) = store_in("missing");
        let token = CancellationToken::new();
        assert!(store.read("anthropic", &token).await.unwrap().is_none());
        assert!(store.list(&token).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn invalid_entry_reports_the_provider() {
        let (dir, store) = store_in("invalid");
        std::fs::write(store.path(), r#"{"anthropic": {"type": "bogus"}}"#).unwrap();
        let token = CancellationToken::new();
        let error = store.read("anthropic", &token).await.unwrap_err();
        let message = error.to_string();
        assert!(message.contains("anthropic"), "{message}");
        drop(dir);
    }
}
