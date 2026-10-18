//! Default stream function registry.
//!
//! Ported from `@step-harness/agent-core` `src/stream-fn.ts`. The TypeScript
//! module-level singleton is kept so hosts can inject their own model runtime;
//! Rust code may also pass a [`StreamFn`] explicitly to every agent.

use std::sync::{Arc, Mutex};

use crate::types::StreamFn;

static DEFAULT_STREAM_FN: Mutex<Option<Arc<dyn StreamFn>>> = Mutex::new(None);

fn registry() -> std::sync::MutexGuard<'static, Option<Arc<dyn StreamFn>>> {
    DEFAULT_STREAM_FN
        .lock()
        .expect("default stream fn mutex poisoned")
}

/// Install or clear the process-wide default stream function.
pub fn set_default_stream_fn(stream_fn: Option<Arc<dyn StreamFn>>) {
    *registry() = stream_fn;
}

/// The installed default stream function, if any.
pub fn get_default_stream_fn() -> Option<Arc<dyn StreamFn>> {
    registry().clone()
}

/// The installed default stream function.
///
/// # Panics
///
/// Panics when no default stream function is configured. Pass a stream function
/// explicitly or call [`set_default_stream_fn`] first.
pub fn default_stream_fn() -> Arc<dyn StreamFn> {
    get_default_stream_fn().expect(
        "No default stream function configured. Pass streamFn explicitly or call set_default_stream_fn().",
    )
}
