//! Serialization of file mutations per environment and canonical path.
//!
//! Ported from `@step-harness/agent-core` `src\harness\tools\file-mutation-queue.ts`.
//!
//! The TypeScript source keeps a `WeakMap` of promise chains per environment.
//! Rust environments are shared as `Arc<dyn ExecutionEnv>`, so the registry is
//! keyed by a weak reference to the environment: entries disappear when the
//! environment is dropped, and the per-key async mutex provides the same
//! serialization while holding the lock until the mutation fully settles.

use std::collections::HashMap;
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Weak};

use tokio::sync::Mutex;

use crate::harness::types::{ExecutionEnv, FileError};

/// Registry key identifying one environment instance.
#[derive(Clone)]
struct EnvKey(Weak<dyn ExecutionEnv>);

impl PartialEq for EnvKey {
    fn eq(&self, other: &Self) -> bool {
        Weak::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for EnvKey {}

impl Hash for EnvKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Weak::as_ptr(&self.0).cast::<()>().hash(state);
    }
}

type QueueRegistry = HashMap<(EnvKey, String), Weak<Mutex<()>>>;

fn registry() -> std::sync::MutexGuard<'static, QueueRegistry> {
    static REGISTRY: std::sync::OnceLock<std::sync::Mutex<QueueRegistry>> =
        std::sync::OnceLock::new();
    REGISTRY
        .get_or_init(|| std::sync::Mutex::new(HashMap::new()))
        .lock()
        .expect("mutation queue registry poisoned")
}

/// The queue for one environment and canonical path, pruning dead environments.
fn queue_for(env: &Arc<dyn ExecutionEnv>, key: &str) -> Arc<Mutex<()>> {
    let env_key = EnvKey(Arc::downgrade(env));
    let mut guard = registry();
    guard.retain(|(existing_key, _), queue| {
        existing_key.0.strong_count() > 0 && queue.strong_count() > 0
    });
    let entry = guard.entry((env_key, key.to_owned())).or_default();
    if let Some(queue) = entry.upgrade() {
        return queue;
    }
    let queue = Arc::new(Mutex::new(()));
    *entry = Arc::downgrade(&queue);
    queue
}

/// Canonicalize the nearest existing ancestor so a new file keeps the same
/// queue identity after creation. Fall back only for unsupported backends.
async fn mutation_queue_key(env: &dyn ExecutionEnv, path: &str) -> Result<String, FileError> {
    let absolute_path = env.absolute_path(path, None).await?;
    let mut ancestor = std::path::PathBuf::from(&absolute_path);
    let mut suffix = Vec::new();
    loop {
        match env.canonical_path(&ancestor.to_string_lossy(), None).await {
            Ok(canonical) => {
                suffix.reverse();
                let mut parts = vec![canonical];
                parts.extend(suffix);
                return env.join_path(&parts, None).await;
            }
            Err(error) if error.code == crate::harness::types::FileErrorCode::NotSupported => {
                return Ok(absolute_path);
            }
            Err(error) if error.code == crate::harness::types::FileErrorCode::NotFound => {
                let Some(name) = ancestor
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                else {
                    return Ok(absolute_path);
                };
                suffix.push(name);
                if !ancestor.pop() {
                    return Ok(absolute_path);
                }
            }
            Err(error) => return Err(error),
        }
    }
}

/// Serialize file mutations targeting the same environment and canonical path.
///
/// The lock is held until `future` fully settles — including when it fails — so
/// the next writer observes the previous write's final file state.
pub async fn with_file_mutation_queue<T, E, F, Fut>(
    env: &Arc<dyn ExecutionEnv>,
    path: &str,
    future: F,
) -> Result<T, E>
where
    E: From<FileError>,
    F: FnOnce() -> Fut,
    Fut: Future<Output = Result<T, E>>,
{
    let key = mutation_queue_key(env.as_ref(), path).await?;
    let queue = queue_for(env, &key);
    let _guard = queue.lock().await;
    future().await
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn creating_a_file_does_not_change_its_queue_key() {
        let dir = tempfile::tempdir().unwrap();
        let env = crate::harness::env::NodeExecutionEnv::new(Default::default());
        let path = dir
            .path()
            .join("nested")
            .join("new.txt")
            .display()
            .to_string();
        let before = mutation_queue_key(&env, &path).await.unwrap();
        use crate::harness::types::{FileContent, FileSystem};
        env.write_file(&path, FileContent::Text("created".to_owned()), None)
            .await
            .unwrap();
        assert_eq!(before, mutation_queue_key(&env, &path).await.unwrap());
    }
}
