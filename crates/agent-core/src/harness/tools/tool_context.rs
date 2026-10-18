//! Filesystem and shell context required by the built-in execution tools.

use std::sync::Arc;

use crate::harness::types::ExecutionEnv;

/// Default tool context: just the execution environment.
#[derive(Clone)]
pub struct ExecutionToolContext {
    pub env: Arc<dyn ExecutionEnv>,
}

impl ExecutionToolContext {
    pub fn new(env: Arc<dyn ExecutionEnv>) -> Self {
        ExecutionToolContext { env }
    }
}

/// Anything that can provide an execution environment to the built-in tools.
///
/// Applications can enrich their own context type with extra state and still
/// use the built-in tools by implementing this trait.
pub trait ToolEnvContext {
    fn env(&self) -> &Arc<dyn ExecutionEnv>;
}

impl ToolEnvContext for ExecutionToolContext {
    fn env(&self) -> &Arc<dyn ExecutionEnv> {
        &self.env
    }
}
