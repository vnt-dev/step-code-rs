# step-harness-providers

Unified LLM API with automatic model discovery and provider configuration —
the Rust migration of the TypeScript `@step-harness/providers` package
(`packages/providers` in the upstream Step-Code repository).

## Layout

| Rust module | TypeScript source |
| --- | --- |
| `src/types.rs` | `src/types.ts` (core types, options, compat, routing) |
| `src/event.rs` | `src/utils/event-stream.ts` + `src/api/lazy.ts` (event stream) |
| `src/error.rs` | `src/auth/resolve.ts` (ModelsError) + `src/utils/error-body.ts` |
| `src/auth/*` | `src/auth/*` (credentials, stores, resolution, login) |
| `src/models.rs` | `src/models.ts` (Provider + Models collection, cost, thinking levels) |
| `src/models_store.rs` | `src/models-store.ts` |
| `src/provider.rs` | `src/provider/*` + `src/dialect/*` + `src/metadata/*` (profiles, declarations, flatten) |
| `src/env_api_keys.rs` | `src/env-api-keys.ts` |
| `src/availability.rs` | `src/availability/probe.ts` |
| `src/session_resources.rs` | `src/session-resources.ts` |
| `src/utils/*` | `src/utils/*` (retry, estimate, overflow, validation, json repair, proxy, ...) |
| `src/api/anthropic_messages.rs` | `src/api/anthropic-messages.ts` |
| `src/api/openai_completions.rs` | `src/api/openai-completions.ts` |
| `src/api/openai_responses.rs` + `openai_responses_shared.rs` | `src/api/openai-responses*.ts` |
| `src/api/{simple_options,transform_messages,constrained_sampling,openai_prompt_cache,copilot_headers,http}.rs` | `src/api/{simple-options,transform-messages,constrained-sampling,openai-prompt-cache,github-copilot-headers}.ts` + HTTP/SSE transport |
| `src/providers/faux.rs` | `src/providers/faux.ts` (scripted test provider) |
| `src/step_provider/*` | `src/step-provider/*` (OAuth device-code login, model discovery) |
| `src/compat.rs` | `src/compat.ts` (legacy global api-registry) |

## Migration decisions

- **Numeric types.** Token counts, context windows, thinking budgets, and
  timeouts are `u64` (the TypeScript source uses untyped `number`). Prices
  and computed costs are `f64` ($/million tokens). Timestamps are `u64`
  milliseconds. HTTP statuses are `u16`. Provider JSON is coerced through
  `types::json_as_token_count` (rejects fractional counts).
- **Mature crates instead of hand-rolled code.** `reqwest` (HTTP + streaming),
  `eventsource-stream` (SSE framing), `serde_json` (JSON, `preserve_order`),
  `jsonschema` (tool-argument validation and coercion, replacing typebox),
  `tokio-util` (`CancellationToken`, replacing `AbortSignal`), `async-trait`,
  `thiserror`, `uuid` (v7), `backon` (backoff), `httpdate` (HTTP dates),
  `jiter` (partial JSON), `moka` (bounded concurrent caches), `rand`, `regex`,
  `url`, `base64`.
- **OpenAI protocol types.** `async-openai` supplies the Responses output-item
  types and function/custom grammar tool definitions for both OpenAI dialects.
  Only `response-types` and `chat-completion-types` are enabled; the SDK does not
  own HTTP requests or retries. The shared transport retains response-header
  observation, injected clients, proxy configuration, cancellation and backpressure.
  Supplier-specific reasoning, usage and event conversion remain in the adapters.
  A compatibility boundary fills omitted empty/metadata fields during Responses
  streaming, ignores unknown output-item types, and rejects malformed supported
  items. Original wire items remain intact for reasoning replay signatures.
  Grammar syntax is serialized as `lark`/`regex`, fixing the previous Responses
  payload that sent internal `openai_lark`/`openai_regex` names.
- **Streaming.** The TypeScript `EventStream` class becomes a tokio
  mpsc + watch pair (`event::AssistantMessageEventStream` /
  `AssistantMessageEventSender`). Adapters spawn a task, push events, and end
  with a terminal `Done`/`Error` event that resolves `result()`.
- **Auth.** `CredentialStore::modify` takes a type-erased
  `Fn(Option<Credential>) -> BoxFuture<ModifyOutcome>` so the double-checked
  OAuth refresh runs under a per-provider async lock (mirroring the TS
  promise-chain serialization).
- **Model catalog.** `models.generated.ts` is empty in this build
  (Step-only build; the catalog is generated from models.dev at build time in
  the TS package) and `builtinProviders()` returns `[]`. The Rust side keeps
  the same empty-catalog machinery; product layers register providers
  explicitly.
- **Partial JSON.** `jiter` parses incomplete argument deltas. Final function
  arguments must be complete JSON objects; malformed input produces an error
  instead of silently repairing or discarding tool arguments.
- **Proxy resolution.** HTTP/HTTPS/ALL proxy variables route by protocol, with
  scoped overrides and reqwest NO_PROXY bypass rules. Invalid configurations
  fail explicitly. Scoped clients share a bounded cache and connection pools.
- **UUIDv7.** `uuid::Uuid::now_v7()` provides timestamp encoding and monotonic
  ordering instead of a custom UUID algorithm.
- **Validation.** Schema validators use a bounded concurrent cache. Invalid
  schemas fail validation rather than allowing arbitrary tool arguments.
- **Cancellation and memory.** Dropping the stream cancels its producer. Caller
  cancellation interrupts requests, retry waits, credential-lock waits, and
  blocked queue sends. The event queue is bounded to 32 entries; schema and
  scoped-client caches are bounded to 256 and 32 entries respectively.
- **Diagnostics.** Credential/request Debug output omits secret values, header
  conversion is shared and marks authentication headers sensitive, and errors
  retain their underlying source where available.

## API adjustments

Options use explicit composition rather than Deref inheritance: request fields
are accessed as `stream_options.base.api_key`, or
`adapter_options.base.base.api_key`. `result().await` returns
`Result<AssistantMessage, EventStreamError>` when a producer exits prematurely;
`Models::complete` still returns an AssistantMessage with an error stop reason.
The redirect-refusing login transport constructor is now fallible.

`StreamOptions::event_snapshot_mode` defaults to `EventSnapshotMode::Full`.
For long outputs, `MetadataOnly` avoids cloning accumulated content for each
partial event. Consumers then accumulate text/thinking/tool-call deltas; end
and terminal events still carry their relevant complete content/final message.

## Not migrated

- `bedrock-provider.js` — a prebuilt AWS Smithy bundle that ships as JS; the
  Step-only build registers no built-in providers.
- `telemetryContext` request plumbing — declared but unused inside the
  TypeScript package, and the workspace telemetry traits are not object-safe.
- WebSocket transports — declared in `StreamOptions.transport` but unused by
  every current adapter (SSE only).
- Browser/bundler concerns (dynamic `import()` lazy loading, `sideEffects`)
  have no Rust equivalent; `api/lazy.ts` collapses into direct module
  functions.

## Dependencies

All workspace registry dependencies were checked against crates.io stable,
non-yanked releases on 2026-10-11. Manifests set the latest stable version as
their minimum; Cargo.lock records the resolved versions. Major-series upgrades
include reqwest 0.13.5, jsonschema 0.58.6, rand 0.10.3 and base64 0.23.1;
async-openai is 0.42.2. Transitive dependencies follow their upstream version
constraints. Reqwest now uses the `rustls` feature and platform certificate
verification. Applications injecting a reqwest Client must use the 0.13 series.

## Tests

`cargo test -p step-harness-providers` covers unit and integration tests
without external network access: SSE/event-stream framing and state machines,
request payload shapes, compat detection and thinking formats, usage/cost
arithmetic, tool-call id normalization, constrained sampling, retry
classification, overflow detection, JSON repair/partial parsing, auth
resolution and OAuth double-checked refresh, model flattening, env-key
discovery, the faux provider's scripted responses (including deferred
fetch/cancel), the Step CLI login state machine, and end-to-end collection
dispatch through `Models`. Conformance fixtures from the TypeScript package
are ported as unit tests with loopback SSE servers.
