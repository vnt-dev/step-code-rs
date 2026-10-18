//! Session-scoped resource cleanup registry.

use std::sync::{Arc, Mutex, OnceLock};

type CleanupFn = Arc<dyn Fn(Option<&str>) + Send + Sync>;

static REGISTRY: OnceLock<Mutex<Vec<CleanupFn>>> = OnceLock::new();

fn registry() -> &'static Mutex<Vec<CleanupFn>> {
    REGISTRY.get_or_init(|| Mutex::new(Vec::new()))
}

/// Unregister guard returned by [`register_session_resource_cleanup`]; drops
/// remove the cleanup.
pub struct CleanupRegistration {
    cleanup: CleanupFn,
}

impl Drop for CleanupRegistration {
    fn drop(&mut self) {
        registry()
            .lock()
            .expect("session cleanups poisoned")
            .retain(|registered| !Arc::ptr_eq(registered, &self.cleanup));
    }
}

/// Register a cleanup callback run by [`cleanup_session_resources`]. The
/// returned guard unregisters on drop.
pub fn register_session_resource_cleanup(
    cleanup: impl Fn(Option<&str>) + Send + Sync + 'static,
) -> CleanupRegistration {
    let cleanup: CleanupFn = Arc::new(cleanup);
    registry()
        .lock()
        .expect("session cleanups poisoned")
        .push(cleanup.clone());
    CleanupRegistration { cleanup }
}

/// Run all registered session cleanups. Panicking cleanups are recorded as a
/// failure but do not abort the remaining cleanups.
pub fn cleanup_session_resources(session_id: Option<&str>) -> Result<(), String> {
    let cleanups: Vec<CleanupFn> = registry()
        .lock()
        .expect("session cleanups poisoned")
        .clone();
    let mut first_error = None;
    for cleanup in cleanups {
        let outcome =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cleanup(session_id)));
        if outcome.is_err() && first_error.is_none() {
            first_error = Some("session resource cleanup panicked".to_owned());
        }
    }
    match first_error {
        None => Ok(()),
        Some(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn runs_registered_cleanups_and_guard_unregisters() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_for_cleanup = counter.clone();
        let registration = register_session_resource_cleanup(move |_| {
            counter_for_cleanup.fetch_add(1, Ordering::SeqCst);
        });
        cleanup_session_resources(Some("s1")).unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        drop(registration);
        cleanup_session_resources(Some("s1")).unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }
}
