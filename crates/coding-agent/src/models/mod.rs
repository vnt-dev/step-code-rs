//! Model configuration, catalog, and per-session controller.

pub mod catalog;
pub mod config;
pub mod controller;
pub mod store;

pub use catalog::{
    CatalogProvider, CatalogSnapshot, ModelCatalog, env_api_key_var, provider_ids,
    strip_thinking_suffix,
};
pub use config::{
    DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_TOKENS, ModelEntry, ModelsConfigFile, ProviderEntry,
};
pub use controller::{ModelController, ModelSwitchError};
pub use store::FileModelsStore;
