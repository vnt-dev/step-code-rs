import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import { test } from "node:test";
import ts from "typescript";

// Exercise the actual frontend parser using the Rust wire representation.
const source = readFileSync(new URL("../src/lib/messages.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 },
});
const messages = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);

const user = timestamp => ({ role: "user", content: "prompt", timestamp });
const assistant = (timestamp, text) => ({ role: "assistant", content: [{ type: "text", text }], timestamp });

test("completed executions fold intermediate tool turns and preserve only the final answer", () => {
  const final = assistant(4, "result");
  const transcript = [user(1), assistant(2, "checking"), { role: "toolResult", timestamp: 3 }, final];
  const groups = messages.executionGroups(transcript, false, [{ firstMessageTimestamp: 1, lastMessageTimestamp: 4, durationMs: 348000 }]);
  assert.equal(groups.length, 1);
  assert.equal(groups[0].finalAssistant, final);
  assert.equal(groups[0].completed, true);
  assert.equal(groups[0].durationMs, 348000);
});

test("supplemental messages stay expanded during execution and fold into the same execution afterwards", () => {
  const transcript = [user(1), assistant(2, "checking"), user(3), assistant(4, "final")];
  const live = messages.executionGroups(transcript, true);
  assert.equal(live.length, 1);
  assert.equal(live[0].completed, false);
  assert.equal(live[0].finalAssistant, undefined);
  const settled = messages.executionGroups(transcript, false, [{ firstMessageTimestamp: 1, lastMessageTimestamp: 4, durationMs: 4000 }]);
  assert.equal(settled.length, 1);
  assert.equal(settled[0].messages.length, 4);
  assert.equal(settled[0].finalAssistant, transcript[3]);
});

test("previous executions stay folded while the latest execution is running", () => {
  const transcript = [user(1), assistant(2, "previous"), user(3), assistant(4, "working")];
  const groups = messages.executionGroups(transcript, true, [{ firstMessageTimestamp: 1, lastMessageTimestamp: 2, durationMs: 1000 }]);
  assert.equal(groups.length, 2);
  assert.equal(groups[0].completed, true);
  assert.equal(groups[1].completed, false);
});

test("legacy transcripts and failed replies retain their final result without inventing elapsed time", () => {
  const failed = { ...assistant(4, ""), stopReason: "error", errorMessage: "failed" };
  const groups = messages.executionGroups([user(1), assistant(2, "done"), user(3), failed], false);
  assert.equal(groups.length, 2);
  assert.equal(groups[1].finalAssistant, failed);
  assert.equal(groups[0].durationMs, undefined);
});

test("work duration matches the compact Chinese disclosure label", () => {
  assert.equal(messages.formatWorkDuration(348000), "已工作 5 分 48 秒");
  assert.equal(messages.formatWorkDuration(3605000), "已工作 1 小时 0 分 5 秒");
  assert.equal(messages.formatWorkDuration(-1), "已工作 0 秒");
  assert.equal(messages.formatWorkDuration(), "已工作");
});

test("plain user content is a string on the Rust wire", () => {
  const message = { role: "user", content: "Explain this project", timestamp: 1 };
  assert.equal(messages.userText(message), "Explain this project");
  assert.equal(messages.userHasImages(message), false);
});

test("mixed user content retains text and detects attachments", () => {
  const message = { role: "user", timestamp: 2, content: [
    { type: "text", text: "Review this" },
    { type: "image", data: "test", mimeType: "image/png" },
    { type: "text", text: "and suggest a fix" },
  ] };
  assert.equal(messages.userText(message), "Review this\nand suggest a fix");
  assert.equal(messages.userHasImages(message), true);
});

test("assistant answer, reasoning and tool calls remain separate", () => {
  const message = { role: "assistant", timestamp: 3, content: [
    { type: "thinking", thinking: "Inspect the entry point" },
    { type: "text", text: "The app " },
    { type: "toolCall", id: "call-1", name: "read", arguments: { file_path: "src/main.rs" } },
    { type: "text", text: "starts here." },
  ] };
  assert.equal(messages.assistantText(message), "The app starts here.");
  assert.equal(messages.assistantThinking(message), "Inspect the entry point");
  assert.equal(messages.assistantToolCalls(message).length, 1);
});

test("live elapsed counters keep tenths under a minute and switch to m:ss", () => {
  assert.equal(messages.formatElapsed(0), "0.0s");
  assert.equal(messages.formatElapsed(3200), "3.2s");
  assert.equal(messages.formatElapsed(59_900), "59.9s");
  assert.equal(messages.formatElapsed(60_000), "1:00");
  assert.equal(messages.formatElapsed(95_400), "1:35");
  assert.equal(messages.formatElapsed(-50), "0.0s");
});

test("whole-second durations stay compact and never show negative time", () => {
  assert.equal(messages.formatDuration(0), "0s");
  assert.equal(messages.formatDuration(999), "0s");
  assert.equal(messages.formatDuration(45_600), "45s");
  assert.equal(messages.formatDuration(60_000), "1:00");
  assert.equal(messages.formatDuration(3_605_000), "1:00:05");
});

test("thinking durations read as whole seconds, then minutes", () => {
  assert.equal(messages.formatThinkingDuration(0), "1 秒");
  assert.equal(messages.formatThinkingDuration(400), "1 秒");
  assert.equal(messages.formatThinkingDuration(8_400), "8 秒");
  assert.equal(messages.formatThinkingDuration(59_400), "59 秒");
  assert.equal(messages.formatThinkingDuration(65_000), "1 分 5 秒");
  assert.equal(messages.formatThinkingDuration(-20), "1 秒");
});


test("execution duration breakdown is exhaustive and preserves display order", () => {
  const timing = { firstMessageTimestamp: 1, lastMessageTimestamp: 2, durationMs: 130000,
    toolDurationMs: 40000, apiDurationMs: 60000, approvalWaitDurationMs: 20000,
    retryWaitDurationMs: 5000, otherDurationMs: 5000 };
  assert.equal(messages.formatWorkBreakdown(timing), "工具调用 40 秒 · API 调用 1 分 0 秒 · 等待审批 20 秒 · 重试等待 5 秒 · 其他处理 5 秒");
  const groups = messages.executionGroups([user(1), assistant(2, "done")], false, [timing]);
  assert.deepEqual(groups[0].timing, timing);
});

test("duration rounding assigns largest remainders and stable ties", () => {
  const timing = { durationMs: 2999, toolDurationMs: 700, apiDurationMs: 700,
    approvalWaitDurationMs: 700, retryWaitDurationMs: 700, otherDurationMs: 199 };
  assert.deepEqual(messages.workDurationBreakdown(timing), [
    { label: "工具调用", seconds: 1 }, { label: "API 调用", seconds: 1 },
  ]);
  for (let seed = 0; seed < 1000; seed++) {
    const values = [17, 31, 59, 97, 131].map(value => seed * value % 10000);
    const input = { durationMs: values.reduce((sum, value) => sum + value, 0),
      toolDurationMs: values[0], apiDurationMs: values[1], approvalWaitDurationMs: values[2],
      retryWaitDurationMs: values[3], otherDurationMs: values[4] };
    assert.equal(messages.workDurationBreakdown(input).reduce((sum, item) => sum + item.seconds, 0),
      Math.floor(input.durationMs / 1000));
  }
});

test("missing, zero, subsecond and incomplete timings do not invent visible categories", () => {
  assert.equal(messages.formatWorkBreakdown(), "");
  assert.equal(messages.formatWorkBreakdown({ durationMs: 5000 }), "");
  assert.equal(messages.formatWorkBreakdown({ durationMs: 0 }), "");
  assert.equal(messages.formatWorkBreakdown({ durationMs: 999, apiDurationMs: 999 }), "");
  assert.equal(messages.formatWorkBreakdown({ durationMs: 2000, apiDurationMs: 2000 }), "API 调用 2 秒");
  assert.equal(messages.formatWorkBreakdown({ durationMs: 5000, apiDurationMs: 2000 }), "");
});
