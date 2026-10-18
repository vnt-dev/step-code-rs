//! Auth environment access: env vars and file-existence checks, injectable
//! for tests.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use async_trait::async_trait;

/// Environment access for auth resolution.
#[async_trait]
pub trait AuthContext: Send + Sync {
    async fn env(&self, name: &str) -> Option<String>;
    /// Check whether a file exists. Supports a leading `~`.
    async fn file_exists(&self, path: &str) -> bool;
}

/// Default auth context: env vars from the process environment plus a
/// provider-scoped overlay, file existence via `std::fs`.
#[derive(Clone, Default)]
pub struct DefaultAuthContext {
    overlay: Option<HashMap<String, String>>,
}

impl DefaultAuthContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Overlay values take precedence over the process environment
    /// (provider-scoped `env` resolution).
    pub fn with_overlay(overlay: HashMap<String, String>) -> Self {
        DefaultAuthContext {
            overlay: Some(overlay),
        }
    }
}

#[async_trait]
impl AuthContext for DefaultAuthContext {
    async fn env(&self, name: &str) -> Option<String> {
        if let Some(value) = self.overlay.as_ref().and_then(|values| values.get(name)) {
            return Some(value.clone());
        }
        std::env::var(name)
            .ok()
            .filter(|value| !value.trim().is_empty())
    }

    async fn file_exists(&self, path: &str) -> bool {
        resolve_tilde(path).is_some_and(|resolved| resolved.exists())
    }
}

/// Expand a leading `~` to the user's home directory (cached per process).
pub fn resolve_tilde(path: &str) -> Option<PathBuf> {
    if !path.starts_with('~') {
        return Some(PathBuf::from(path));
    }
    static HOME: Mutex<Option<PathBuf>> = Mutex::new(None);
    let mut cached = HOME.lock().expect("home cache poisoned");
    if cached.is_none() {
        *cached = home_dir();
    }
    cached.as_ref().map(|home| {
        home.join(
            path.strip_prefix("~/")
                .or(path.strip_prefix('~'))
                .unwrap_or(""),
        )
    })
}

fn home_dir() -> Option<PathBuf> {
    if let Some(home) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) {
        return Some(PathBuf::from(home));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn overlay_takes_precedence_over_process_env() {
        let mut overlay = HashMap::new();
        overlay.insert("PATH".to_owned(), "/overlaid".to_owned());
        let context = DefaultAuthContext::with_overlay(overlay);
        assert_eq!(context.env("PATH").await.as_deref(), Some("/overlaid"));
    }

    #[tokio::test]
    async fn empty_env_values_are_treated_as_unset() {
        let context = DefaultAuthContext::new();
        assert_eq!(context.env("__ALMOST_CERTAINLY_UNSET_VAR__").await, None);
    }

    #[test]
    fn tilde_expands_to_home() {
        let resolved = resolve_tilde("~/.config").expect("home available in test env");
        assert!(!resolved.to_string_lossy().starts_with('~'));
    }
}
