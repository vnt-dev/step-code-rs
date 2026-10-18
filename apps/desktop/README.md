# Step Code (Rust + Tauri)

The Rust migration of the `@step-harness/coding-agent` runtime, hosted in a Tauri
desktop app. There is no terminal UI: the embedder (this app) provides the UI and
calls the runtime crates directly.

```
crates/
  telemetry/     typed spans and cancellation-safe telemetry (existing crate)
  providers/     LLM providers, auth, streaming (existing crate)
  agent-core/    Agent, agent loop, tools, sessions, compaction (existing crate)
  coding-agent/  the migrated coding-agent layer: settings, auth, models,
                 permissions, resources, and the AgentSession orchestrator
apps/
  desktop/       the Tauri v2 app (React frontend + Rust bridge)
```

## Running

```bash
# Rust workspace (crates only)
cargo test --workspace
cargo clippy --workspace --all-targets

# Desktop app (its own package; builds into the shared root target dir)
cd apps/desktop
npm install
npm run tauri dev
```

The desktop app is deliberately **not** a workspace member: its build script
emits link-search paths that would leak into other crates' test binaries during
a workspace-wide build. It shares the root `target/` directory through
`apps/desktop/.cargo/config.toml`.
The configured `../../target` path is relative to `apps/desktop`, the parent of
the `.cargo` directory, so artifacts stay inside this repository.

`tauri dev` also watches the workspace's `crates/` directory so runtime and tool
changes rebuild and restart the desktop backend. `cargo check` only checks code;
it does not replace a running desktop executable.

## Data directory

All state lives under `~/.step-rs/` (override with `STEP_RS_HOME`):

```
settings.json       global settings
auth.json           provider credentials (mode 0600)
models.json         provider/model declarations
models-store.json   cached remote model catalogs
trust.json          project trust decisions
sessions/<cwd>/     JSONL session stores
skills/             user-level skills
prompts/            user-level prompt templates
logs/               rolling log files
```

The app logs to `logs/step-code.log` with `tracing`; the filter comes from
`RUST_LOG` and defaults to `info`. The live file rolls over once it reaches
2 MB, keeping the five most recent generations beside it
(`step-code.log.1` … `step-code.log.5`).

Provider HTTP metadata is logged only at `debug`; enable it with
`RUST_LOG=info,http=debug`. It includes the request method, origin and response
status, without HTTP headers, URL paths/parameters or request bodies.

Per-project settings live in `<cwd>/.step-rs/settings.json`.

## Desktop settings

Use the bottom-left **设置** entry. **模型供应商** supports adding, editing and
removing API services, endpoints, dialects and model IDs. Keys entered in the UI
are stored in `auth.json`; leave the key field blank to preserve existing keys.
Editing a model preserves its advanced metadata. The catalog updates immediately
after saving. Stop running tasks before changing providers.

Tool approvals and plan reviews wait indefinitely for your decision, including
while another session is selected. Stopping a task cancels its pending requests.

**Step 账户** opens the system browser for the existing Step OAuth cloud polling
flow. Completion stores the credential and refreshes available models. Login can
be cancelled or reopened. `STEP_API_KEY` remains an independent environment
credential after OAuth logout.

The composer handles IME composition, preserves failed prompts, and supports
steering and queued follow-ups. Sessions support search, in-app rename and delete
confirmation. Reading older messages does not force-scroll to new output.
When an unnamed session receives its first message, the selected model generates
a short name in the background. The saved name updates the sidebar and heading;
manual renames take priority over pending generated names.
Failed generation retries once with a larger output budget; a later message
can retry an unnamed session using its original first message. Explicitly
clearing a name prevents automatic regeneration, including after reopening.

```bash
cd apps/desktop
npm test
npm run build
cargo test --manifest-path src-tauri/Cargo.toml --lib
```

## Web search

`search_web` is included in the default session tools and runs in parallel.
When `settings.json` has a nonempty `defaultTools` allowlist, include
`search_web` there to enable it. The tool takes a `query` string, requests ten
results, and returns Markdown links with snippets capped at 400 Unicode
characters. Its description instructs the model to cite relevant result URLs
in a final `Sources:` section.

Search uses `STEPCODE_SEARCH_API_KEY`, then the stored `step` credential in the
session's `auth.json` (`access` for OAuth or `key` for API keys). `STEP_API_KEY`
is reserved for the model gateway and is never used for search. Embedded hosts
can supply an explicit key through `SearchWebToolOptions`, which takes priority.
The credential and profile are read together at each search call, so existing
sessions follow later logins, account changes and logout. Cloud login persists
the profile selected for that login, including when polling omits it.

The stored login profile selects the mainland/overseas host and the Step Plan
(`/step_plan/v1`) or platform (`/v1`) billing path. Legacy profile `step` maps
to `step_plan`; missing or unknown profiles use the mainland platform endpoint.
`STEPCODE_SEARCH_WEB_MCP_URL` overrides the endpoint; an origin-only override
inherits the login profile's billing path. Full endpoint paths take priority.
Embedded hosts can also override the endpoint and auth path. Standalone tool
construction honors `STEPCODE_AUTH_PATH`, otherwise using `STEP_RS_HOME/auth.json`
or `~/.step-rs/auth.json`.

The remote call uses Streamable HTTP MCP with JSON/SSE responses, paginated tool
discovery and output-schema validation. Calls have a 30-second deadline and
support session cancellation. Failures identify the endpoint and redact the
search credential from diagnostics.

## Browser QA harness

`qa/` renders the real `src/App.tsx` in a plain browser with the Tauri command
and event surface aliased to a stub (`vite.harness.config.ts`), so UI states
that need a live agent — the pre-reply thinking indicator, tool-call approvals —
can be driven and screenshotted without the desktop shell:

```bash
cd apps/desktop
npx vite --config vite.harness.config.ts   # http://localhost:4173/qa/harness.html
```

Drive scenarios from the page console via `window.__qa__`: `agent(event)`
emits a session event, `approval(request)` raises a confirmation, `patchBootstrap`
edits the canned bootstrap, and `calls` records every command the UI invokes.
The production build ignores `qa/`.

## Configuring a provider manually

Add a provider to `~/.step-rs/models.json`:

```json
{
  "providers": {
    "my-provider": {
      "baseUrl": "https://api.example.com/v1",
      "api": "openai-completions",
      "models": [{ "id": "my-model", "contextWindow": 128000, "maxTokens": 8192 }]
    }
  }
}
```

The API key comes from `MY_PROVIDER_API_KEY` in the environment, an inline
`apiKey` field, or `~/.step-rs/auth.json`:

```json
{ "my-provider": { "type": "api_key", "key": "sk-..." } }
```

The built-in `step` provider (StepFun) authenticates through its OAuth device
flow or `STEP_API_KEY` and discovers models from the network.

## Architecture notes

- **Session pipeline** (`crates/coding-agent/src/session/mod.rs`) mirrors the
  TypeScript turn order: validate model and auth → run the agent loop → loop
  over retry, auto-compaction, and queue drains → settle. Persistence flows
  through `message_end` events, so every message is stored exactly once.
- **Approvals** (`permissions.rs`) classify tool calls and shell commands; the
  policy returns a decision and the bridge raises an `approval-request` event
  for the UI to answer. Presets: `ask` (confirm before changes), `bypass`
  (autonomous, confirm risky commands), `autopilot` (full access, no tool
  approvals). Explicit tool denials still apply in every mode. Legacy
  `read-only` settings migrate to `ask`.
- **Events** flow from the session as `SessionEvent` values, forwarded to the
  web UI as Tauri events (`session-event`, `approval-request`).

## Not migrated (yet)

The extension system (jiti-loaded TypeScript extensions), MCP client support,
and the Step product layer (plugins marketplace, feedback, cron, workflows) are
not part of this milestone.
