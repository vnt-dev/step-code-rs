//! On-disk layout of the Step Code data directory.
//!
//! The desktop app uses a fresh data root (`~/.step-rs/` by default) and does
//! not read legacy layouts. Override the root with `STEP_RS_HOME`.

use std::path::{Path, PathBuf};

/// Environment variable overriding the data root.
pub const HOME_ENV: &str = "STEP_RS_HOME";

/// Directory name of the data root inside the user's home directory.
pub const APP_DIR_NAME: &str = ".step-rs";

/// Directory name of per-project configuration inside a working directory.
pub const PROJECT_CONFIG_DIR_NAME: &str = ".step-rs";

/// Resolved filesystem locations for one data root.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AppPaths {
    root: PathBuf,
}

impl AppPaths {
    /// Build paths for an explicit root.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        AppPaths { root: root.into() }
    }

    /// Discover the data root: `STEP_RS_HOME`, else `~/.step-rs`.
    pub fn discover() -> Self {
        if let Some(root) = std::env::var_os(HOME_ENV).filter(|value| !value.is_empty()) {
            return AppPaths::new(PathBuf::from(root));
        }
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        AppPaths::new(home.join(APP_DIR_NAME))
    }

    /// The data root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Global settings file (`settings.json`).
    pub fn settings_file(&self) -> PathBuf {
        self.root.join("settings.json")
    }

    /// Credential store (`auth.json`, mode 0600).
    pub fn auth_file(&self) -> PathBuf {
        self.root.join("auth.json")
    }

    /// User model configuration (`models.json`).
    pub fn models_file(&self) -> PathBuf {
        self.root.join("models.json")
    }

    /// Cached remote model catalog (`models-store.json`).
    pub fn models_store_file(&self) -> PathBuf {
        self.root.join("models-store.json")
    }

    /// Project trust database (`trust.json`).
    pub fn trust_file(&self) -> PathBuf {
        self.root.join("trust.json")
    }

    /// Root of the session store; sessions live in `<sessions>/<cwd>/<file>.jsonl`.
    pub fn sessions_dir(&self) -> PathBuf {
        self.root.join("sessions")
    }

    /// User-level skill directory.
    pub fn skills_dir(&self) -> PathBuf {
        self.root.join("skills")
    }

    /// User-level prompt template directory.
    pub fn prompts_dir(&self) -> PathBuf {
        self.root.join("prompts")
    }

    /// Log directory for the desktop app's file logger.
    pub fn logs_dir(&self) -> PathBuf {
        self.root.join("logs")
    }

    /// Create the data root and its well-known subdirectories.
    pub fn ensure_dirs(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(self.sessions_dir())?;
        std::fs::create_dir_all(self.skills_dir())?;
        std::fs::create_dir_all(self.prompts_dir())?;
        std::fs::create_dir_all(self.logs_dir())?;
        Ok(())
    }
}

/// The per-project configuration directory for a working directory.
pub fn project_config_dir(cwd: &str) -> PathBuf {
    Path::new(cwd).join(PROJECT_CONFIG_DIR_NAME)
}

/// Encode a working directory into a stable session subdirectory name.
///
/// Non-alphanumeric runs collapse to `-` so the result is a single path
/// segment on every platform; the original cwd is stored in session metadata.
pub fn encode_cwd(cwd: &str) -> String {
    let mut encoded = String::with_capacity(cwd.len());
    let mut pending_dash = false;
    for ch in cwd.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            if pending_dash && !encoded.is_empty() {
                encoded.push('-');
            }
            pending_dash = false;
            encoded.push(ch.to_ascii_lowercase());
        } else {
            pending_dash = true;
        }
    }
    let trimmed = encoded.trim_matches('-').to_owned();
    if trimmed.is_empty() {
        "root".to_owned()
    } else {
        trimmed
    }
}
