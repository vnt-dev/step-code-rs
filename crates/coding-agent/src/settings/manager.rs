//! Settings file loading, merge precedence, and persistence.

pub use crate::settings::schema::{
    BranchSummarySettingsFile, CompactionSettingsFile, DEFAULT_COMPACTION_KEEP_RECENT_TOKENS,
    DEFAULT_COMPACTION_RESERVE_TOKENS, DEFAULT_MAX_RETRIES, DEFAULT_RETRY_BASE_DELAY_MS,
    DEFAULT_THINKING_LEVEL, PermissionPresetSetting, PermissionsSettingsFile, QueueModeSetting,
    RetrySettingsFile, Settings, ThinkingBudgetsFile, ThinkingLevelSetting, deep_merge,
};

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::RwLock;

use serde_json::{Map, Value as Json};

use crate::paths::{AppPaths, project_config_dir};
use crate::settings::schema;

/// A settings-loading diagnostic worth surfacing in the UI.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SettingsDiagnostic {
    pub scope: SettingsScope,
    pub path: PathBuf,
    pub message: String,
}

/// Which file a diagnostic came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SettingsScope {
    Global,
    Project,
}

/// Owns the effective settings plus the raw file contents for merge-back.
pub struct SettingsManager {
    paths: AppPaths,
    cwd: PathBuf,
    project_trusted: bool,
    /// Effective settings (defaults ⊕ global ⊕ project ⊕ env overrides).
    effective: RwLock<Settings>,
    /// Raw global file contents (only keys present on disk).
    global_file: RwLock<Json>,
    /// Raw project file contents.
    project_file: RwLock<Json>,
    /// Diagnostics from the last load.
    diagnostics: RwLock<Vec<SettingsDiagnostic>>,
}

fn lock<T>(mutex: &RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    mutex.read().expect("settings mutex poisoned")
}

fn lock_mut<T>(mutex: &RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
    mutex.write().expect("settings mutex poisoned")
}

/// Read a JSON settings file: BOM-stripped, comments-free, object-only.
/// Returns `None` when the file is absent; an `Err` for unreadable/invalid.
fn read_settings_file(path: &std::path::Path) -> Result<Option<Json>, String> {
    let text = match std::fs::read(path) {
        Ok(bytes) => String::from_utf8(bytes).map_err(|error| format!("invalid UTF-8: {error}"))?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("cannot read {path:?}: {error}")),
    };
    let text = text.trim_start_matches('\u{feff}');
    let value: Json =
        serde_json::from_str(text).map_err(|error| format!("invalid JSON in {path:?}: {error}"))?;
    if !value.is_object() {
        return Err(format!("settings file {path:?} is not a JSON object"));
    }
    Ok(Some(value))
}

/// Environment-variable overrides, applied above both files.
fn env_overrides() -> Settings {
    let mut settings = Settings::default();
    if let Some(model) = std::env::var("STEP_MODEL")
        .ok()
        .filter(|value| !value.trim().is_empty())
    {
        // `STEP_MODEL` may be `provider/model` or a bare model id.
        match model.split_once('/') {
            Some((provider, id)) => {
                settings.default_provider = Some(provider.to_owned());
                settings.default_model = Some(id.to_owned());
            }
            None => settings.default_model = Some(model),
        }
    }
    if let Some(provider) = std::env::var("STEP_PROVIDER")
        .ok()
        .filter(|value| !value.trim().is_empty())
    {
        settings.default_provider = Some(provider);
    }
    settings
}

impl SettingsManager {
    /// Load global and (when trusted) project settings for a working directory.
    pub fn create(cwd: impl Into<PathBuf>, paths: AppPaths, project_trusted: bool) -> Self {
        let cwd = cwd.into();
        let mut diagnostics = Vec::new();
        let global = match read_settings_file(&paths.settings_file()) {
            Ok(value) => value.unwrap_or(Json::Object(Map::new())),
            Err(message) => {
                diagnostics.push(SettingsDiagnostic {
                    scope: SettingsScope::Global,
                    path: paths.settings_file(),
                    message,
                });
                Json::Object(Map::new())
            }
        };
        let project_path = project_config_dir(&cwd.to_string_lossy()).join("settings.json");
        let project = if project_trusted {
            match read_settings_file(&project_path) {
                Ok(value) => value.unwrap_or(Json::Object(Map::new())),
                Err(message) => {
                    diagnostics.push(SettingsDiagnostic {
                        scope: SettingsScope::Project,
                        path: project_path,
                        message,
                    });
                    Json::Object(Map::new())
                }
            }
        } else {
            Json::Object(Map::new())
        };

        let mut merged = Json::Object(Map::new());
        deep_merge(&mut merged, global.clone());
        deep_merge(&mut merged, project.clone());
        deep_merge(
            &mut merged,
            serde_json::to_value(env_overrides()).unwrap_or(Json::Object(Map::new())),
        );
        let effective: Settings = serde_json::from_value(merged).unwrap_or_default();

        SettingsManager {
            paths,
            cwd,
            project_trusted,
            effective: RwLock::new(effective),
            global_file: RwLock::new(global),
            project_file: RwLock::new(project),
            diagnostics: RwLock::new(diagnostics),
        }
    }

    /// The effective settings snapshot.
    pub fn get(&self) -> Settings {
        lock(&self.effective).clone()
    }

    /// Diagnostics produced by the last load.
    pub fn diagnostics(&self) -> Vec<SettingsDiagnostic> {
        lock(&self.diagnostics).clone()
    }

    /// Whether project-scoped settings are honored.
    pub fn project_trusted(&self) -> bool {
        self.project_trusted
    }

    /// Apply a patch to the effective settings without persisting.
    pub fn apply_patch(&self, patch: Json) {
        let mut merged = serde_json::to_value(lock(&self.effective).clone()).unwrap_or(Json::Null);
        deep_merge(&mut merged, patch);
        if let Ok(settings) = serde_json::from_value::<Settings>(merged) {
            *lock_mut(&self.effective) = settings;
        }
    }

    /// Persist a patch into the project file when trusted, else the global
    /// file, then recompute the effective settings. Only keys present in the
    /// patch are written; foreign keys in the file are preserved.
    pub fn update(&self, patch: Json) -> Result<(), String> {
        let target = if self.project_trusted {
            project_config_dir(&self.cwd.to_string_lossy()).join("settings.json")
        } else {
            self.paths.settings_file()
        };
        let is_project = self.project_trusted;
        {
            let mut file = if is_project {
                lock_mut(&self.project_file)
            } else {
                lock_mut(&self.global_file)
            };
            if !file.is_object() {
                *file = Json::Object(Map::new());
            }
            deep_merge(&mut file, patch.clone());
        }
        write_settings_file(
            &target,
            if is_project {
                lock(&self.project_file).clone()
            } else {
                lock(&self.global_file).clone()
            },
        )?;

        // Recompute effective settings from both files plus env overrides.
        let mut merged = Json::Object(Map::new());
        deep_merge(&mut merged, lock(&self.global_file).clone());
        deep_merge(&mut merged, lock(&self.project_file).clone());
        deep_merge(
            &mut merged,
            serde_json::to_value(env_overrides()).unwrap_or(Json::Object(Map::new())),
        );
        *lock_mut(&self.effective) = serde_json::from_value(merged).unwrap_or_default();
        Ok(())
    }

    /// Re-read both files from disk and recompute the effective settings.
    pub fn reload(&self) {
        let fresh =
            SettingsManager::create(self.cwd.clone(), self.paths.clone(), self.project_trusted);
        *lock_mut(&self.effective) = fresh.get();
        *lock_mut(&self.global_file) = lock(&fresh.global_file).clone();
        *lock_mut(&self.project_file) = lock(&fresh.project_file).clone();
        *lock_mut(&self.diagnostics) = fresh.diagnostics();
    }

    /// Effective settings as a flat JSON object (for the settings UI).
    pub fn to_json(&self) -> Map<String, Json> {
        match serde_json::to_value(lock(&self.effective).clone()) {
            Ok(Json::Object(map)) => map,
            _ => Map::new(),
        }
    }

    /// Per-model thinking-level lookup (`provider/modelId`).
    pub fn thinking_level_for(
        &self,
        provider: &str,
        model_id: &str,
    ) -> Option<schema::ThinkingLevelSetting> {
        let key = format!("{provider}/{model_id}");
        lock(&self.effective)
            .model_thinking_levels
            .as_ref()
            .and_then(|levels: &BTreeMap<String, _>| levels.get(&key).copied())
    }
}

/// Atomically write a settings file with 2-space indentation.
fn write_settings_file(path: &std::path::Path, contents: Json) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create {parent:?}: {error}"))?;
    }
    let text = serde_json::to_string_pretty(&contents)
        .map_err(|error| format!("cannot serialize settings: {error}"))?;
    let temp = path.with_extension(format!("json.tmp.{}", std::process::id()));
    std::fs::write(&temp, text).map_err(|error| format!("cannot write {temp:?}: {error}"))?;
    std::fs::rename(&temp, path).map_err(|error| {
        let _ = std::fs::remove_file(&temp);
        format!("cannot replace {path:?}: {error}")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn temp_paths(name: &str) -> AppPaths {
        let dir = tempfile::tempdir().unwrap();
        let paths = AppPaths::new(dir.path().join(name));
        paths.ensure_dirs().unwrap();
        // Keep the tempdir alive by leaking it into a static; tests are short.
        std::mem::forget(dir);
        paths
    }

    #[test]
    fn loads_global_and_project_settings_with_project_winning() {
        let paths = temp_paths("global-project");
        std::fs::write(
            paths.settings_file(),
            serde_json::to_string_pretty(&json!({
                "defaultModel": "global-model",
                "defaultProvider": "global-provider",
                "compaction": {"enabled": true}
            }))
            .unwrap(),
        )
        .unwrap();
        let cwd = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(project_config_dir(cwd.path().to_str().unwrap())).unwrap();
        std::fs::write(
            project_config_dir(cwd.path().to_str().unwrap()).join("settings.json"),
            serde_json::to_string_pretty(&json!({"defaultModel": "project-model"})).unwrap(),
        )
        .unwrap();

        let manager = SettingsManager::create(cwd.path(), paths, true);
        let settings = manager.get();
        assert_eq!(settings.default_model.as_deref(), Some("project-model"));
        assert_eq!(
            settings.default_provider.as_deref(),
            Some("global-provider")
        );
        assert_eq!(settings.compaction.unwrap().enabled, Some(true));
        assert!(manager.diagnostics().is_empty());
    }

    #[test]
    fn untrusted_projects_are_ignored() {
        let paths = temp_paths("untrusted");
        let cwd = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(project_config_dir(cwd.path().to_str().unwrap())).unwrap();
        std::fs::write(
            project_config_dir(cwd.path().to_str().unwrap()).join("settings.json"),
            r#"{"defaultModel": "project-model"}"#,
        )
        .unwrap();
        let manager = SettingsManager::create(cwd.path(), paths, false);
        assert_eq!(manager.get().default_model, None);
    }

    #[test]
    fn invalid_json_becomes_a_diagnostic_not_a_panic() {
        let paths = temp_paths("broken");
        std::fs::write(paths.settings_file(), "{not json").unwrap();
        let cwd = tempfile::tempdir().unwrap();
        let manager = SettingsManager::create(cwd.path(), paths, false);
        assert_eq!(manager.diagnostics().len(), 1);
        assert_eq!(manager.get().default_model, None);
    }

    #[test]
    fn project_null_replaces_the_global_value() {
        let paths = temp_paths("null-replace");
        std::fs::write(paths.settings_file(), r#"{"defaultModel": "global-model"}"#).unwrap();
        let cwd = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(project_config_dir(cwd.path().to_str().unwrap())).unwrap();
        std::fs::write(
            project_config_dir(cwd.path().to_str().unwrap()).join("settings.json"),
            r#"{"defaultModel": null}"#,
        )
        .unwrap();
        let manager = SettingsManager::create(cwd.path(), paths, true);
        assert_eq!(manager.get().default_model, None);
    }

    #[test]
    fn update_persists_and_preserves_foreign_keys() {
        let paths = temp_paths("update");
        std::fs::write(
            paths.settings_file(),
            r#"{"defaultModel": "a", "unknownFutureKey": 7}"#,
        )
        .unwrap();
        let cwd = tempfile::tempdir().unwrap();
        let manager = SettingsManager::create(cwd.path(), paths.clone(), false);
        manager.update(json!({"defaultModel": "b"})).unwrap();
        let on_disk: Json =
            serde_json::from_str(&std::fs::read_to_string(paths.settings_file()).unwrap()).unwrap();
        assert_eq!(on_disk["defaultModel"], "b");
        assert_eq!(on_disk["unknownFutureKey"], 7);
        assert_eq!(manager.get().default_model.as_deref(), Some("b"));
    }
}
