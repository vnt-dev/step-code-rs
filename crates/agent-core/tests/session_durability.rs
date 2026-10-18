use std::sync::Arc;

use step_harness_agent_core::harness::env::{NodeExecutionEnv, NodeExecutionEnvOptions};
use step_harness_agent_core::harness::types::ExecutionEnv;
use step_harness_agent_core::session::facade::SessionStorage;
use step_harness_agent_core::session::jsonl::{JsonlSessionStorage, types::JsonlV4Header};
use step_harness_agent_core::session::types::{NewRecord, SessionErrorCode, StepKind};

fn environment(dir: &tempfile::TempDir) -> Arc<dyn ExecutionEnv> {
    Arc::new(NodeExecutionEnv::new(NodeExecutionEnvOptions {
        cwd: Some(dir.path().display().to_string()),
        ..Default::default()
    }))
}

fn header() -> JsonlV4Header {
    JsonlV4Header::new("test", 0, "test", None, None)
}

#[tokio::test]
async fn rejected_record_never_changes_the_durable_log() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("session.jsonl").display().to_string();
    let env = environment(&directory);
    let storage = JsonlSessionStorage::create(env.clone(), &path, &header())
        .await
        .unwrap();
    let before = std::fs::read(&path).unwrap();
    let error = storage
        .append_record(NewRecord::StepAttempt {
            id: "invalid".to_owned(),
            lane: "main".to_owned(),
            run_id: "run".to_owned(),
            step: StepKind::Compaction,
            attempt: 1,
            result_entry_id: "summary".to_owned(),
            compaction_reason: None,
        })
        .await
        .unwrap_err();
    assert_eq!(error.code, SessionErrorCode::InvalidEntry);
    assert_eq!(std::fs::read(&path).unwrap(), before);
    storage.create_lane("next", None).await.unwrap();
    drop(storage);
    let loaded = JsonlSessionStorage::load(env, &path).await.unwrap();
    assert_eq!(loaded.get_lanes().await.len(), 2);
}

#[tokio::test]
async fn independent_storage_handles_cannot_write_the_same_session() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("session.jsonl").display().to_string();
    let storage = JsonlSessionStorage::create(environment(&directory), &path, &header())
        .await
        .unwrap();
    assert!(
        JsonlSessionStorage::load(environment(&directory), &path)
            .await
            .is_err()
    );
    assert!(
        JsonlSessionStorage::create(environment(&directory), &path, &header())
            .await
            .is_err()
    );
    drop(storage);
    let loaded = JsonlSessionStorage::load(environment(&directory), &path)
        .await
        .unwrap();
    assert_eq!(loaded.get_lanes().await.len(), 1);
}

#[tokio::test]
async fn torn_tail_repair_preserves_all_acknowledged_mutations() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("session.jsonl").display().to_string();
    let storage = JsonlSessionStorage::create(environment(&directory), &path, &header())
        .await
        .unwrap();
    storage.create_lane("saved", None).await.unwrap();
    drop(storage);
    let valid = std::fs::read(&path).unwrap();
    use std::io::Write;
    std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(b"{\"kind\":")
        .unwrap();
    let loaded = JsonlSessionStorage::load(environment(&directory), &path)
        .await
        .unwrap();
    assert_eq!(loaded.get_lanes().await.len(), 2);
    assert_eq!(std::fs::read(&path).unwrap(), valid);
    loaded.create_lane("after_recovery", None).await.unwrap();
    loaded.drain().await;
    drop(loaded);
    assert_eq!(
        JsonlSessionStorage::load(environment(&directory), &path)
            .await
            .unwrap()
            .get_lanes()
            .await
            .len(),
        3
    );
}

struct ControlledEnv {
    inner: NodeExecutionEnv,
    started: tokio::sync::Notify,
    release: tokio::sync::Notify,
    fail: bool,
}

use step_harness_agent_core::harness::types::*;
#[async_trait::async_trait]
impl FileSystem for ControlledEnv {
    fn cwd(&self) -> &str {
        self.inner.cwd()
    }
    async fn absolute_path(
        &self,
        path: &str,
        signal: Option<&AbortSignal>,
    ) -> Result<String, FileError> {
        self.inner.absolute_path(path, signal).await
    }
    async fn join_path(
        &self,
        parts: &[String],
        signal: Option<&AbortSignal>,
    ) -> Result<String, FileError> {
        self.inner.join_path(parts, signal).await
    }
    async fn read_text_file(
        &self,
        path: &str,
        signal: Option<&AbortSignal>,
    ) -> Result<String, FileError> {
        self.inner.read_text_file(path, signal).await
    }
    async fn read_text_lines(
        &self,
        path: &str,
        options: Option<&ReadTextLinesOptions>,
    ) -> Result<Vec<String>, FileError> {
        self.inner.read_text_lines(path, options).await
    }
    async fn read_binary_file(
        &self,
        path: &str,
        signal: Option<&AbortSignal>,
    ) -> Result<Vec<u8>, FileError> {
        self.inner.read_binary_file(path, signal).await
    }
    async fn write_file(
        &self,
        path: &str,
        content: FileContent,
        signal: Option<&AbortSignal>,
    ) -> Result<(), FileError> {
        self.inner.write_file(path, content, signal).await
    }
    async fn append_file(
        &self,
        path: &str,
        content: FileContent,
        signal: Option<&AbortSignal>,
    ) -> Result<(), FileError> {
        self.inner.append_file(path, content, signal).await
    }
    async fn rename_file(
        &self,
        source: &str,
        destination: &str,
        signal: Option<&AbortSignal>,
    ) -> Result<(), FileError> {
        self.inner.rename_file(source, destination, signal).await
    }
    async fn file_info(
        &self,
        path: &str,
        signal: Option<&AbortSignal>,
    ) -> Result<FileInfo, FileError> {
        self.inner.file_info(path, signal).await
    }
    async fn list_dir(
        &self,
        path: &str,
        signal: Option<&AbortSignal>,
    ) -> Result<Vec<FileInfo>, FileError> {
        self.inner.list_dir(path, signal).await
    }
    async fn canonical_path(
        &self,
        path: &str,
        signal: Option<&AbortSignal>,
    ) -> Result<String, FileError> {
        self.inner.canonical_path(path, signal).await
    }
    async fn exists(&self, path: &str, signal: Option<&AbortSignal>) -> Result<bool, FileError> {
        self.inner.exists(path, signal).await
    }
    async fn create_dir(
        &self,
        path: &str,
        options: Option<&CreateDirOptions>,
    ) -> Result<(), FileError> {
        self.inner.create_dir(path, options).await
    }
    async fn remove(&self, path: &str, options: Option<&RemoveOptions>) -> Result<(), FileError> {
        self.inner.remove(path, options).await
    }
    async fn create_temp_dir(
        &self,
        prefix: Option<&str>,
        signal: Option<&AbortSignal>,
    ) -> Result<String, FileError> {
        self.inner.create_temp_dir(prefix, signal).await
    }
    async fn create_temp_file(
        &self,
        options: Option<&CreateTempFileOptions>,
    ) -> Result<String, FileError> {
        self.inner.create_temp_file(options).await
    }
    async fn lock_file(&self, path: &str) -> Result<Arc<dyn FileLease>, FileError> {
        self.inner.lock_file(path).await
    }
    async fn write_file_atomic(
        &self,
        path: &str,
        content: FileContent,
        overwrite: bool,
    ) -> Result<(), FileError> {
        self.inner.write_file_atomic(path, content, overwrite).await
    }

    async fn append_file_durable(&self, path: &str, content: FileContent) -> Result<(), FileError> {
        self.started.notify_one();
        if self.fail {
            return Err(FileError::new(
                FileErrorCode::Unknown,
                "injected append failure",
                None,
            ));
        }
        self.release.notified().await;
        self.inner.append_file_durable(path, content).await
    }
    async fn cleanup(&self) {
        FileSystem::cleanup(&self.inner).await;
    }
}
#[async_trait::async_trait]
impl Shell for ControlledEnv {
    async fn exec(
        &self,
        command: &str,
        options: Option<&ShellExecOptions>,
    ) -> Result<ShellExecResult, ExecutionError> {
        self.inner.exec(command, options).await
    }
    async fn cleanup(&self) {
        Shell::cleanup(&self.inner).await;
    }
}

#[tokio::test]
async fn cancelled_caller_does_not_interrupt_an_in_flight_commit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.jsonl").display().to_string();
    let env = Arc::new(ControlledEnv {
        inner: NodeExecutionEnv::new(Default::default()),
        started: Default::default(),
        release: Default::default(),
        fail: false,
    });
    let storage = Arc::new(
        JsonlSessionStorage::create(env.clone(), &path, &header())
            .await
            .unwrap(),
    );
    let writing = storage.clone();
    let task = tokio::spawn(async move { writing.create_lane("committed", None).await });
    env.started.notified().await;
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    env.release.notify_one();
    tokio::time::timeout(std::time::Duration::from_secs(1), storage.drain())
        .await
        .unwrap();
    assert_eq!(storage.get_lanes().await.len(), 2);
    drop(storage);
    let loaded = JsonlSessionStorage::load(env, &path).await.unwrap();
    assert_eq!(loaded.get_lanes().await.len(), 2);
}

#[tokio::test]
async fn failed_append_requires_recovery_before_another_write() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.jsonl").display().to_string();
    let env = Arc::new(ControlledEnv {
        inner: NodeExecutionEnv::new(Default::default()),
        started: Default::default(),
        release: Default::default(),
        fail: true,
    });
    let storage = JsonlSessionStorage::create(env.clone(), &path, &header())
        .await
        .unwrap();
    let before = std::fs::read(&path).unwrap();
    assert!(storage.create_lane("failed", None).await.is_err());
    assert!(
        storage
            .create_lane("retry", None)
            .await
            .unwrap_err()
            .message
            .contains("reopened")
    );
    assert_eq!(std::fs::read(&path).unwrap(), before);
    drop(storage);
    assert_eq!(
        JsonlSessionStorage::load(env, &path)
            .await
            .unwrap()
            .get_lanes()
            .await
            .len(),
        1
    );
}

#[tokio::test]
async fn separate_repositories_cannot_create_duplicate_logical_session_ids() {
    use step_harness_agent_core::session::jsonl::JsonlSessionRepo;
    use step_harness_agent_core::session::jsonl::types::{
        JsonlSessionCreateOptions, JsonlSessionListOptions, JsonlSessionRepoOptions,
    };
    let dir = tempfile::tempdir().unwrap();
    let options = JsonlSessionRepoOptions {
        sessions_root: dir.path().join("sessions").display().to_string(),
    };
    let first = JsonlSessionRepo::new(environment(&dir), options.clone());
    let second = JsonlSessionRepo::new(environment(&dir), options);
    let create = JsonlSessionCreateOptions {
        id: Some("same".to_owned()),
        cwd: dir.path().display().to_string(),
        ..Default::default()
    };
    let (a, b) = tokio::join!(first.create(&create), second.create(&create));
    assert_ne!(a.is_ok(), b.is_ok());
    assert_eq!(
        first
            .list(&JsonlSessionListOptions::default())
            .await
            .unwrap()
            .len(),
        1
    );
}
