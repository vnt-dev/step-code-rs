import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import { test } from "node:test";
import ts from "typescript";

const compile = path => ts.transpileModule(readFileSync(new URL(path, import.meta.url), "utf8"), {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 },
}).outputText;
const moduleUrl = source => `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`;
const messagesUrl = moduleUrl(compile("../src/lib/messages.ts"));
const source = compile("../src/lib/context-usage.ts").replace('"./messages"', JSON.stringify(messagesUrl));
const { contextUsage, inputTokens, cacheHitRate } = await import(moduleUrl(source));
const user = timestamp => ({ role: "user", timestamp, content: "task" });
const assistant = (timestamp, usage) => ({ role: "assistant", timestamp, usage, content: [] });
const usage = { input: 100, output: 20, cacheRead: 300, cacheWrite: 100 };

test("input and cache hit rate include all input buckets without output", () => {
  assert.equal(inputTokens(usage), 500);
  assert.equal(cacheHitRate(usage), 60);
  assert.equal(cacheHitRate({ input: 0, output: 20, cacheRead: 0, cacheWrite: 0 }), 0);
});

test("last round sums tool iterations and supplemental prompts in the same execution", () => {
  const transcript = [user(1), assistant(2, usage), user(3), assistant(4, usage)];
  const result = contextUsage(transcript, null, false, [{ firstMessageTimestamp: 1, lastMessageTimestamp: 4, durationMs: 1000 }]);
  assert.equal(inputTokens(result.lastRound), 1000);
  assert.equal(result.lastRound.output, 40);
  assert.equal(cacheHitRate(result.lastRound), 60);
});

test("current context, last round and durable session totals remain independent", () => {
  const stats = { contextTokens: 520, inputTokens: 1000, outputTokens: 200, cacheReadTokens: 3000, cacheWriteTokens: 1000 };
  const result = contextUsage([user(1), assistant(2, usage), user(3), assistant(4, usage)], stats, true,
    [{ firstMessageTimestamp: 1, lastMessageTimestamp: 2, durationMs: 1000 }]);
  assert.equal(result.contextTokens, 520);
  assert.equal(inputTokens(result.lastRound), 500);
  assert.equal(inputTokens(result.total), 5000);
  assert.equal(result.total.output, 200);
  assert.equal(cacheHitRate(result.total), 60);
  assert.deepEqual(contextUsage([], null, false, []).lastRound, { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 });
});

test("new messages never add estimates to the last request's reported context", () => {
  const stats = { contextTokens: 520, inputTokens: 100, outputTokens: 20, cacheReadTokens: 300, cacheWriteTokens: 100 };
  const transcript = [user(1), assistant(2, usage), { role: "toolResult", timestamp: 3, content: [{ type: "text", text: "x".repeat(10000) }] }, user(4)];
  assert.equal(contextUsage(transcript, stats, true, []).contextTokens, 520);
});

test("SQLite usage overrides transcript history without changing current context", () => {
  const empty = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
  const stats = { contextTokens: 520, inputTokens: 999, outputTokens: 999, cacheReadTokens: 0, cacheWriteTokens: 0,
    performance: { lastRoundUsage: empty, sessionUsage: { input: 5, output: 20, cacheRead: 5, cacheWrite: 0 } } };
  const result = contextUsage([user(1), assistant(2, usage)], stats, false, []);
  assert.equal(result.contextTokens, 520);
  assert.deepEqual(result.lastRound, empty);
  assert.equal(inputTokens(result.total), 10);
  assert.equal(result.total.output, 20);
});

test("rounds come from the backend when it has them, otherwise from the transcript", () => {
  const oneRound = [user(1), assistant(2, usage)];
  const twoRounds = [...oneRound, user(3), assistant(4, usage)];
  assert.equal(contextUsage(oneRound, null, false, []).rounds, 1);
  assert.equal(contextUsage(twoRounds, null, false, []).rounds, 2);
  // A compacted transcript keeps the backend count instead of what survived.
  const compacted = { performance: { sessionRounds: 7 } };
  assert.equal(contextUsage(oneRound, compacted, false, []).rounds, 7);
  assert.equal(contextUsage(twoRounds, compacted, false, []).rounds, 7);
  assert.equal(contextUsage([], null, false, []).rounds, 0);
});
