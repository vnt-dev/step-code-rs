//! Stateful agent with tool execution, session persistence, context
//! compaction, and event streaming.
//!
//! Rust migration of the TypeScript `@step-harness/agent-core` package, built
//! on [`step_harness_providers`] and [`step_harness_telemetry`].
//!
//! # Architecture
//!
//! - [`types`] — messages, tools, events, loop configuration, and hooks.
//! - [`agent_loop`] — the low-level loop: prompt handling, assistant streaming,
//!   and sequential/parallel tool execution.
//! - [`Agent`] — the stateful runtime: transcript ownership, lifecycle events,
//!   and steering/follow-up queues.
//! - [`harness`] — backend-neutral contracts (filesystem, shell, skills),
//!   built-in tools, session persistence, and context compaction.
//!
//! # Notable differences from the TypeScript source
//!
//! - The `Result` monad is `std::result::Result`; expected failures are values,
//!   not exceptions.
//! - Async callbacks (hooks, tools, listeners) are trait objects.
//! - `AbortSignal` is [`tokio_util::sync::CancellationToken`].
//! - Custom agent messages are enum variants instead of declaration merging.

pub mod agent;
pub mod agent_loop;
pub mod compaction;
pub mod harness;
pub mod proxy;
pub mod search;
pub mod session;
pub mod stream_fn;
pub mod types;

pub use agent::{Agent, AgentError, AgentListener, PromptInput, SubscriptionId};
pub use agent_loop::{
    AgentEventStream, AgentLoopError, DEFAULT_TOOL_CALL_LEAK_RETRIES, agent_loop,
    agent_loop_continue, run_agent_loop, run_agent_loop_continue,
};
pub use harness::messages::convert_to_llm as convert_agent_messages_to_llm;
pub use harness::tools::{
    ExecutionToolContext, ToolEnvContext, create_bash_tool_with_env, create_edit_tool_with_env,
    create_find_tool_with_env, create_grep_tool_with_env, create_ls_tool_with_env,
    create_powershell_tool_with_env, create_read_tool_with_env, create_write_tool_with_env,
};
pub use proxy::{ProxyAssistantMessageEvent, ProxyStreamOptions, stream_proxy};
pub use session::{
    Entry, EntryQuery, ForkOptions, InMemorySessionRepo, InMemorySessionStorage, JsonlSessionRepo,
    JsonlSessionStorage, LaneRecord, ProvisionedEntry, Session, SessionError, SessionMetadata,
    SessionStorage, build_session_context,
};
pub use stream_fn::{default_stream_fn, get_default_stream_fn, set_default_stream_fn};
pub use types::*;
