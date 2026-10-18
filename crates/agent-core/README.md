# step-harness-agent-core

Stateful agent with tool execution, session persistence, context compaction, and event streaming.

Rust migration of the TypeScript `@step-harness/agent-core` package, built on
[`step-harness-providers`](../providers) (models, providers, streaming events) and
[`step-harness-telemetry`](../telemetry) (backend-neutral telemetry). Every module
documents the differences from its TypeScript source; the crate-wide ones are
listed at the bottom of this file.

## Module structure

| Module | What it owns |
| --- | --- |
| [`agent`](src/agent.rs) | The stateful runtime: transcript ownership, lifecycle events, steering and follow-up queues. |
| [`agent_loop`](src/agent_loop.rs) | The low-level loop: prompt handling, assistant streaming, sequential/parallel tool execution, retries. |
| [`harness`](src/harness/mod.rs) | Backend-neutral contracts and built-ins: execution environments, tools, skills, prompt templates, system prompt, session message projection. |
| [`session`](src/session) | Durable sessions: an append-only mutation log with lanes, records, entries, and pluggable storage (in-memory and JSONL). |
| [`compaction`](src/compaction) | Context compaction: threshold estimation, cut-point selection, summary generation, branch summarization, and request-time context projection. |
| [`harness::telemetry`](src/harness/telemetry.rs) | The AI-request and harness telemetry schemas plus the `startAiSpan`/`startHarnessSpan` helpers. |
| [`harness::reducer`](src/harness/reducer.rs) | Pure lane-state recovery: validation and reduction of a lane's durable recovery slice. No IO. |
| [`harness::agent_harness`](src/harness/agent_harness.rs) | The harness contract: lanes, outcomes, rejection unions, and the scaffolded `AgentHarness`. |
| [`search`](src/search/mod.rs) | Session search contracts and the scanning implementation. |
| [`proxy`](src/proxy.rs) | Proxy stream function for apps that route LLM calls through a server. |
| [`stream_fn`](src/stream_fn.rs) | The process-wide default stream function registry. |
| [`types`](src/types.rs) | Messages, tools, events, loop configuration, and hooks. |

## Quick start

### An agent with a stream function and built-in tools

```rust
use std::sync::Arc;

use step_harness_agent_core::harness::env::{NodeExecutionEnv, NodeExecutionEnvOptions};
use step_harness_agent_core::harness::tools::{
    create_bash_tool_with_env, create_read_tool_with_env, create_write_tool_with_env,
};
use step_harness_agent_core::session::Session;
use step_harness_agent_core::session::memory::InMemorySessionStorage;
use step_harness_agent_core::session::types::SessionMetadata;
use step_harness_agent_core::types::{
    AgentOptions, InitialAgentState, NoopStreamFn, QueueMode, ToolExecutionMode,
};
use step_harness_providers::types::Transport;

#[tokio::main]
async fn main() {
    // The execution environment backs every built-in tool.
    let env = Arc::new(NodeExecutionEnv::new(NodeExecutionEnvOptions {
        cwd: None,
        shell_path: None,
        shell_env: None,
    }));

    // Built-in tools are bound to one environment; pass them as the initial state.
    let tools = vec![
        create_bash_tool_with_env(env.clone()),
        create_read_tool_with_env(env.clone(), None),
        create_write_tool_with_env(env.clone()),
    ];

    let agent = step_harness_agent_core::Agent::new(AgentOptions {
        initial_state: Some(InitialAgentState {
            system_prompt: Some("You are a coding agent.".to_owned()),
            tools: Some(tools),
            ..InitialAgentState::default()
        }),
        convert_to_llm: None,
        transform_context: None,
        // Replace with your own `StreamFn` (or `set_default_stream_fn`) to reach a model.
        stream_fn: Arc::new(NoopStreamFn),
        get_api_key: None,
        on_payload: None,
        on_response: None,
        before_tool_call: None,
        after_tool_call: None,
        transform_tool_result: None,
        should_stop_after_turn: None,
        prepare_next_turn: None,
        prepare_next_turn_with_context: None,
        steering_mode: QueueMode::OneAtATime,
        follow_up_mode: QueueMode::OneAtATime,
        session_id: None,
        thinking_budgets: None,
        transport: Transport::Auto,
        max_retry_delay_ms: None,
        tool_execution: ToolExecutionMode::Parallel,
    });

    agent
        .prompt_text("List the files in this directory", None)
        .await
        .unwrap();

    // Durable session: in-memory here, `JsonlSessionStorage` for a JSONL file.
    let session = Session::new(
        Arc::new(InMemorySessionStorage::new(SessionMetadata {
            id: "session-1".to_owned(),
            created_at: 0,
            parent_session_id: None,
        })),
        None,
    );
    let _ = session;
}
```

### The harness contract

`AgentHarness` is the durable-session front end. The execution engine is not part of
this crate yet, so every method that would run work returns
`HarnessRejected::NotImplemented` (or `HarnessRejected::Closed` after `close`).
Configuration management, `close`, and the record-log reducer are real:

```rust
use std::sync::Arc;

use step_harness_agent_core::harness::agent_harness::{AgentHarness, AgentHarnessOptions};
use step_harness_agent_core::session::Session;
use step_harness_agent_core::session::memory::InMemorySessionStorage;
use step_harness_agent_core::session::types::SessionMetadata;
use step_harness_agent_core::types::PromptInput;
use step_harness_providers::{CreateModelsOptions, Models};
use step_harness_telemetry::noop::NoopTelemetryContext;

# async fn example() {
let storage = Arc::new(InMemorySessionStorage::new(SessionMetadata {
    id: "session-1".to_owned(),
    created_at: 0,
    parent_session_id: None,
}));
let options: AgentHarnessOptions<SessionMetadata, NoopTelemetryContext> = AgentHarnessOptions {
    session: Session::new(storage, None),
    models: Models::new(CreateModelsOptions::default()),
    model, // a `step_harness_providers::types::Model`
    thinking_level: None,
    active_tool_names: None,
    tools: Vec::new(),
    tool_context: None,
    system_prompt: None,
    resources: None,
    stream_options: None,
    retry: None,
    compaction: None,
    steering_mode: None,
    follow_up_mode: None,
    tool_execution: None,
    drive: None,
    to_provider_messages: None,
    entry_projectors: Default::default(),
    context: None,
};

let (harness, suspended) = AgentHarness::create(options).await.unwrap();
assert!(suspended.is_empty());

// Lane calls are refused until the execution engine exists; errors are values.
let error = harness
    .prompt(PromptInput::Text("hi".to_owned()), None)
    .await
    .unwrap_err();
assert!(matches!(error, step_harness_agent_core::harness::agent_harness::RunRejected::NotImplemented(_)));

harness.close();
# }
```

### Telemetry schemas

`harness::telemetry` exports the agent-owned span vocabulary as runtime schema data:

```rust
use step_harness_agent_core::harness::telemetry::{
    AGENT_TELEMETRY_SCHEMAS, AI_SPAN_NAMES, HARNESS_SPAN_NAMES,
};

assert_eq!(AI_SPAN_NAMES, ["pi.ai.request"]);
assert_eq!(HARNESS_SPAN_NAMES.len(), 11);
assert_eq!(AGENT_TELEMETRY_SCHEMAS.len(), 2);
```

## Runtime guarantees and compatibility

- Agent invocations atomically reserve the running slot. Dropping a prompt future,
  timing it out, or aborting the task releases runtime state and wakes idle waiters.
  Abort also interrupts listeners and tools that do not cooperate with cancellation.
- Shell processes are owned by process-wrap (Unix process groups / Windows Job
  Objects). Timeout, cancellation, and normal exit share one lifecycle; auxiliary
  readers are aborted when execution is dropped. UTF-8 decoding uses encoding_rs,
  replacing invalid bytes and flushing incomplete sequences at EOF.
- Shell callbacks own output capture: the corresponding stdout/stderr field is
  empty when a callback is supplied. Without callbacks each stream is limited to
  16 MiB and excess output is an explicit error. Built-in bash keeps a bounded
  tail and spools full output; a stalled spool writer reports an error and cancels
  the command rather than building an unbounded in-memory queue.
- Agent event streams use a bounded queue. `result()` discards unread events while
  waiting, so it works without a separate event consumer. Dropping the stream
  cancels its producer. Tool progress snapshots coalesce for slow consumers, and
  parallel tool execution polls at most 16 tools concurrently.
- Skills and prompt templates share serde-saphyr YAML parsing. Multiline scalars,
  escaping and comments are supported; malformed YAML produces diagnostics.
  Fences must occupy complete lines. The previous prompt-template BOM policy
  remains unchanged.
- Local writes publish complete files atomically and preserve existing permissions
  and symlink targets. Atomic replacement changes the file inode; other hard links
  retain the previous inode contents.
- Read offsets/limits are positive integers. Local text reads scan with a fixed
  buffer and retain only a bounded page, including for very long lines. Counting
  total file lines still requires scanning the complete file. Custom backends
  can override `read_file_prefix` / `read_text_page` to avoid the compatibility
  fallback's whole-file read.
- Images use image/png decoding with a 32 MiB source limit, dimensions capped at
  16384, and a 128 MiB decode allocation limit. The built-in processor resizes to
  fit 1600x1600 by default and converts BMP to PNG. `auto_resize_images: false`
  disables resizing; a custom image processor overrides the built-in processor.
  Animated PNG remains unsupported.
- JSONL storage requires exclusive file leases, synced appends, and atomic durable
  publication. NodeExecutionEnv implements these capabilities; custom backends
  must implement `lock_file`, `append_file_durable`, and `write_file_atomic` or
  receive `NotSupported`. A storage handle owns its lease until it is dropped;
  opening another writer or deleting an open session is refused. Sidecar lock
  files are retained because deleting them would break cross-process locking.
- Mutation validation precedes persistence. An in-flight commit finishes even if
  its caller is cancelled; `drain()` waits for it. A failed append disables further
  writes until the storage is dropped and reopened for recovery. Reopen repairs
  a torn final append and validates the complete log. Files are synced before
  acknowledgment; atomic publication also syncs the parent directory on Unix.
  Physical persistence ultimately depends on the filesystem and storage device.
- Proxy transport uses standard SSE parsing, shared HTTP connection pooling and
  immediate external cancellation. Frames are limited to 1 MiB, content blocks
  to 1024, and cumulative message data to 16 MiB. Missing terminal events and
  invalid content ordering are errors. Servers must terminate SSE events with
  blank lines, as required by SSE.
- `SessionErrorCode::Aborted` distinguishes cancelled searches from storage errors.
  File URL conversion uses url, shell lookup uses which, temporary files use
  tempfile, and timestamp formatting uses time.
- Token estimates preserve the ASCII fallback and use a more conservative
  non-ASCII estimate. They are still heuristics, not model tokenizers; callers
  can use `estimate_context_tokens_with` for model-specific estimates and must
  include system prompts, tool schemas and protocol overhead in request budgets.
- `AgentHarness` remains a contract scaffold. `supports_execution()` returns false;
  use `Agent` for execution until the durable harness engine is implemented.

SQLite remains an optional future storage backend for transactional queries or
large session collections; this hardening keeps the established JSONL format.

## Numeric configuration

Token thresholds and budgets are resolved to `u64` before comparison. Projection
trigger thresholds round up; cap and summary budgets round down. Ratios use the
shortest round-trip decimal representation of the configured `f64`, so a ratio
of 0.6 means six tenths without converting token counts to floating point.
Full-width `u64` counts retain their precision.

Projection ratios and summary reserve fractions must be finite and in `(0, 1]`.
`ProjectionOptions::validate()` returns a field-specific `RatioError`. Projection
with invalid ratios returns unchanged messages with `InvalidOptions` as its
skipped reason. `try_pick_summary_max_tokens()` returns a typed validation error;
the original `pick_summary_max_tokens()` API returns zero for invalid fractions.

Public shell timeouts remain fractional seconds for compatibility. Bash and the
local execution environment share validation into `Duration`, preserving
submillisecond timeouts. Positive values that round to zero nanoseconds, non-finite
values, and values above 2147483.647 seconds are rejected before execution.

## Differences from the TypeScript package

Behavioral differences that change observable results:

- **UTF-16 → UTF-8.** Rust strings are UTF-8, so:
  - `shortContentHash` iterates Unicode scalar values while the TypeScript
    original iterates UTF-16 code units. Hashes agree for BMP text and differ for
    astral (non-BMP) characters. Deduplication only compares hashes from the same
    function, so the divergence is self-consistent.
  - Output truncation counts bytes (`str::len`) instead of UTF-16 code units, and
    the surrogate-repair steps in the TypeScript source disappear.
  - Character counts in projection statistics are `usize` scalar counts.
- **`localeCompare` → byte order.** Skill and prompt-template listings sort with
  `name.as_bytes().cmp(...)`; locale-sensitive collation is not reproduced.
- **Usage has no negative adjustment.** Token counts are `u64` and costs are
  `f64`, so the TypeScript `adjustment` usage cause can only record non-negative
  deltas.

API-shape differences:

- **The `Result` monad is `std::result::Result`.** Expected failures are values,
  not exceptions; the TypeScript `matchError` becomes an exhaustive `match`.
- **Async callbacks are trait objects.** Hooks, tools, listeners, projectors, and
  stream functions are `Arc<dyn ... + Send + Sync>` traits implemented with
  `async_trait`, instead of closures or interface objects.
- **`AbortSignal` is `tokio_util::sync::CancellationToken`.**
- **Declaration merging becomes an enum.** The TypeScript extension point for
  custom agent messages is the `AgentMessage` enum; adding a message type means
  adding a variant.
- **Telemetry schemas are runtime data.** The TypeScript schema is a compile-time
  literal that derives span names and exact attribute sets; here the schemas are
  `LazyLock` statics, span names are strings checked in debug builds, and
  `type: "number"` attributes are split by meaning into `unsigned_integer` (HTTP
  status, token counts, chunk counts, delays, attempts, sequences) and `float`
  (cost).
- **Harness rejection unions carry `NotImplemented`.** TypeScript's `unavailable()`
  rejects with `HarnessNotImplemented` even though it is not part of those unions,
  because rejected promises are not type-checked; Rust return values are, so each
  union gains the variant.
- **`HarnessTool` wraps `AgentTool`** instead of intersecting it, and fixes the
  tool details type to `Json` (the TypeScript default is `any`).
- **`AgentHarness` is generic** over the session storage metadata and the
  telemetry context, because `Session` and `TelemetryContext` are.
- **Compaction defaults follow this crate.** `DEFAULT_COMPACTION_SETTINGS` reserves
  24576 tokens rather than the TypeScript harness default of 16384; see
  `harness/compaction/compaction.rs` for why.

## Development

```bash
cargo check -p step-harness-agent-core
cargo test -p step-harness-agent-core
```
