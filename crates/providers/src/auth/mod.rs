//! Provider authentication: credentials, resolution, stores, and login flows.

pub mod context;
pub mod credential_store;
pub mod helpers;
pub mod resolve;
pub mod types;

pub use context::{AuthContext, DefaultAuthContext};
pub use credential_store::InMemoryCredentialStore;
pub use helpers::env_api_key_auth;
pub use resolve::{AuthResolutionOverrides, ModelsError, ModelsErrorCode, resolve_provider_auth};
pub use types::*;
