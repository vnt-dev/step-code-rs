import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import { test } from "node:test";
import ts from "typescript";

// The module is pure data mapping, so exercise it directly.
const transpile = relativePath => ts.transpileModule(readFileSync(new URL(`../${relativePath}`, import.meta.url), "utf8"), {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 },
}).outputText;
const moduleUrl = code => `data:text/javascript;base64,${Buffer.from(code).toString("base64")}`;
const messages = moduleUrl(transpile("src/lib/messages.ts"));
const { roundTps, charsPerToken } = await import(moduleUrl(transpile("src/lib/round-tps.ts").replaceAll('from "./messages"', `from "${messages}"`)));

const usage = output => ({ input: 100, output, cacheRead: 0, cacheWrite: 0 });
const stats = performance => ({ inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0, cost: 0, turns: 1, performance });
const reply = (patch = {}) => ({ role: "assistant", content: [], api: "step", provider: "step", model: "step-3", usage: usage(0), stopReason: "pending", timestamp: 1000, ...patch });
const text = length => [{ type: "text", text: "x".repeat(length) }];

test("a finished round reports the exact backend measurement", () => {
  const messages = [reply({ stopReason: "stop", usage: usage(1800) })];
  const performance = { lastRoundTps: 42.6, lastRoundSamples: 2, sessionTps: 38.1, sessionSamples: 3 };
  assert.equal(roundTps(messages, stats(performance), false, 9000), 42.6);
  assert.equal(roundTps(messages, stats(null), false, 9000), null);
  assert.equal(roundTps([], null, false, 9000), null);
});

test("a streaming reply with no finished request reports its own live speed", () => {
  // 400 characters for 200 tokens calibrates the ratio at two characters per token.
  const finished = reply({ content: text(400), usage: usage(200), stopReason: "stop", timestamp: 1 });
  const streaming = reply({ content: text(100), timestamp: 1000 });
  // 50 estimated tokens over two seconds.
  assert.equal(roundTps([finished, streaming], stats(null), true, 3000), 25);
});

test("a running round blends the finished requests with the streaming reply", () => {
  const streaming = reply({ content: text(100), timestamp: 1000 });
  const performance = { lastRoundTps: 45, lastRoundSamples: 1, sessionTps: 45, sessionSamples: 1, lastRoundUsage: usage(1800) };
  // 1800 tokens at 45 tps is 40s of generation, plus 50 estimated tokens over 2s.
  assert.equal(roundTps([streaming], stats(performance), true, 3000), (1800 + 50) / 42000 * 1000);
});

test("finished requests without a measured generation time are left out", () => {
  const streaming = reply({ content: text(100), timestamp: 1000 });
  const performance = { lastRoundTps: null, lastRoundSamples: 0, sessionTps: null, sessionSamples: 0, lastRoundUsage: usage(1800) };
  assert.equal(roundTps([streaming], stats(performance), true, 3000), 25);
});

test("a reply that produced nothing yet shows no number", () => {
  assert.equal(roundTps([reply()], stats(null), true, 5000), null);
  assert.equal(roundTps([], stats(null), true, 5000), null);
});

test("a round waiting on a tool keeps reporting its finished requests", () => {
  const finished = reply({ stopReason: "toolUse", usage: usage(900), timestamp: 1 });
  const performance = { lastRoundTps: 30, lastRoundSamples: 1, sessionTps: 30, sessionSamples: 1, lastRoundUsage: usage(900) };
  assert.equal(roundTps([finished], stats(performance), true, 9000), 30);
});

test("thinking and tool arguments count as produced characters", () => {
  const args = JSON.stringify({ path: "x".repeat(20) }).length;
  const finished = reply({ content: text(400), usage: usage(200), stopReason: "stop", timestamp: 1 });
  const streaming = reply({ content: [
    { type: "thinking", thinking: "x".repeat(50) },
    { type: "text", text: "x".repeat(30) },
    { type: "toolCall", id: "t", name: "read", arguments: { path: "x".repeat(20) } },
  ], timestamp: 1000 });
  assert.equal(roundTps([finished, streaming], stats(null), true, 3000), (50 + 30 + args) / 2 / 2000 * 1000);
});

test("the characters-per-token ratio is measured on finished replies only", () => {
  assert.equal(charsPerToken([]), 2);
  // A reply still streaming has no usage yet and must not calibrate the ratio.
  assert.equal(charsPerToken([reply({ content: text(400) })]), 2);
  assert.equal(charsPerToken([reply({ content: text(400), usage: usage(200), stopReason: "stop", timestamp: 1 })]), 2);
  assert.equal(charsPerToken([reply({ content: text(300), usage: usage(200), stopReason: "stop", timestamp: 1 })]), 1.5);
});

test("a timestamp in the future never produces a speed", () => {
  const streaming = reply({ content: text(100), timestamp: 5000 });
  assert.equal(roundTps([streaming], stats(null), true, 1000), null);
});
