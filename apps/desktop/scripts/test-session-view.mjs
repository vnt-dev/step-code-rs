import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import { test } from "node:test";
import ts from "typescript";

const source = readFileSync(new URL("../src/lib/session-view.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 } });
const { emptySessionView, reduceSessionEvent, hydrateSessionView, addApproval, addPlanReview } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);
const approval = (sessionId, requestId) => ({ sessionId, requestId, toolName: "write", argsSummary: sessionId, reason: "confirmation", hazardous: false });
const review = (sessionId, requestId) => ({ sessionId, requestId, planFilePath: `${sessionId}.md`, planContents: `# ${sessionId}` });
const snapshot = (patch = {}) => ({ transcript: [], running: false, approvals: [], planReviews: [], stats: null, ...patch });
const user = (content, timestamp = 1) => ({ role: "user", content, timestamp });
const assistant = (text, timestamp = 2) => ({ role: "assistant", content: [{ type: "text", text }], timestamp });

test("recall rebuilds retained tools and rejects a stale snapshot that contains removed messages", () => {
  const keptCall = { ...assistant('', 2), content: [{ type: 'toolCall', id: 'kept', name: 'read', arguments: { path: 'kept.rs' } }] };
  const keptResult = { role: 'toolResult', toolCallId: 'kept', toolName: 'read', content: [{ type: 'text', text: 'kept result' }], timestamp: 3 };
  const removedCall = { ...assistant('', 6), content: [{ type: 'toolCall', id: 'removed', name: 'read', arguments: {} }] };
  const retained = [user('first'), keptCall, keptResult, assistant('first answer', 4)];
  const original = [...retained, user('recall me', 5), removedCall, assistant('removed answer', 7)];
  const keptTiming = { firstMessageTimestamp: 1, lastMessageTimestamp: 4, durationMs: 3000 };
  const removedTiming = { firstMessageTimestamp: 5, lastMessageTimestamp: 7, durationMs: 2000 };
  const oldSnapshot = snapshot({ transcript: original, executionTimings: [keptTiming, removedTiming], running: true });
  let current = hydrateSessionView(emptySessionView(), oldSnapshot, 0);
  current.thinkingDurations = new Map([[4, 1000], [7, 2000]]);
  current.notices = ['old notice'];
  current.error = 'old error';
  const recalled = reduceSessionEvent(current, { sessionId: 'a', type: 'messagesRecalled', messages: retained, executionTimings: [keptTiming] });
  assert.deepEqual(recalled.messages, retained);
  assert.deepEqual([...recalled.toolCalls.keys()], ['kept']);
  assert.equal(recalled.toolCalls.get('kept').result, 'kept result');
  assert.deepEqual([...recalled.thinkingDurations], [[4, 1000]]);
  assert.equal(recalled.running, false);
  assert.equal(recalled.error, null);
  assert.deepEqual(recalled.notices, []);
  const late = hydrateSessionView(recalled, oldSnapshot, current.revision);
  assert.deepEqual(late.messages, retained);
  assert.deepEqual([...late.toolCalls.keys()], ['kept']);
  assert.deepEqual(late.executionTimings, [keptTiming]);
  assert.equal(late.running, false);
  const newTurn = reduceSessionEvent(recalled, { sessionId: 'a', type: 'agent', event: { type: 'messageEnd', message: user('replacement', 8) } });
  assert.deepEqual(hydrateSessionView(newTurn, oldSnapshot, 0).messages, [...retained, user('replacement', 8)]);
});

test("recalling the first prompt leaves an empty session ready for another prompt", () => {
  const current = hydrateSessionView(emptySessionView(), snapshot({ transcript: [user('first'), assistant('answer')] }), 0);
  const recalled = reduceSessionEvent(current, { sessionId: 'a', type: 'messagesRecalled', messages: [], executionTimings: [] });
  assert.deepEqual(recalled.messages, []);
  assert.equal(recalled.toolCalls.size, 0);
  assert.deepEqual(recalled.executionTimings, []);
  assert.equal(recalled.awaitingResponse, false);
});

test("execution duration survives switching and a snapshot cannot overwrite a newer completion", () => {
  const timing = { firstMessageTimestamp: 1, lastMessageTimestamp: 4, durationMs: 348000 };
  const current = reduceSessionEvent(emptySessionView(), { sessionId: "a", type: "executionCompleted", timing });
  const restored = hydrateSessionView(emptySessionView(), snapshot({ executionTimings: [timing] }), 0);
  assert.deepEqual(restored.executionTimings, [timing]);
  const late = hydrateSessionView(current, snapshot({ executionTimings: [{ ...timing, durationMs: 1000 }] }), 0);
  assert.deepEqual(late.executionTimings, [timing]);
  assert.deepEqual(emptySessionView().executionTimings, []);
});

test("interleaved streaming, tools, errors and completion stay with their originating session", () => {
  const sessions = new Map([['a', emptySessionView()], ['b', emptySessionView()]]);
  const emit = event => sessions.set(event.sessionId, reduceSessionEvent(sessions.get(event.sessionId), event));
  for (const sessionId of ['a', 'b']) emit({ sessionId, type: "agent", event: { type: "agentStart" } });
  emit({ sessionId: 'a', type: "agent", event: { type: "messageEnd", message: user('first user') } });
  emit({ sessionId: 'b', type: "agent", event: { type: "messageEnd", message: user('second user') } });
  emit({ sessionId: 'a', type: "agent", event: { type: "messageUpdate", message: assistant('partial') } });
  emit({ sessionId: 'b', type: "agent", event: { type: "toolExecutionStart", toolCallId: "same", toolName: "read", args: { path: 'b' } } });
  emit({ sessionId: 'a', type: "agent", event: { type: "messageEnd", message: assistant('complete') } });
  emit({ sessionId: 'a', type: "settled" });
  assert.equal(sessions.get('b').running, true);
  assert.deepEqual(sessions.get('a').messages, [user('first user'), assistant('complete')]);
  assert.deepEqual(sessions.get('b').messages, [user('second user')]);
  assert.equal(sessions.get('a').toolCalls.size, 0);
  assert.equal(sessions.get('b').toolCalls.get('same').args.path, 'b');
  emit({ sessionId: 'b', type: "error", message: 'b failed' });
  assert.equal(sessions.get('a').error, null);
  assert.equal(sessions.get('b').error, 'b failed');
});

test("pending approvals and plan reviews remain separate and queued across switches", () => {
  let a = addApproval(addApproval(emptySessionView(), approval('a', 'a:1')), approval('a', 'a:2'));
  const b = addApproval(emptySessionView(), approval('b', 'b:1'));
  a = addPlanReview(a, review('a', 'a:plan'));
  a = reduceSessionEvent(a, { sessionId: 'a', type: 'requestResolved', requestId: 'a:1', kind: 'approval' });
  assert.deepEqual(a.approvals.map(request => request.requestId), ['a:2']);
  assert.deepEqual(a.planReviews.map(request => request.requestId), ['a:plan']);
  assert.deepEqual(b.approvals.map(request => request.requestId), ['b:1']);
  const finished = reduceSessionEvent(a, { sessionId: 'a', type: 'settled' });
  assert.equal(finished.approvals.length, 0);
  assert.equal(finished.planReviews.length, 0);
  assert.equal(b.approvals.length, 1);
});

test("switching back recovers backend approvals without losing newer streaming output", () => {
  const live = reduceSessionEvent(emptySessionView(), { sessionId: 'a', type: 'agent', event: { type: 'messageUpdate', message: assistant('new answer') } });
  const hydrated = hydrateSessionView(live, snapshot({ transcript: [user('a'), assistant('old answer')], approvals: [approval('a', 'a:1')], running: true }), 0);
  assert.deepEqual(hydrated.messages, [user('a'), assistant('new answer')]);
  assert.equal(hydrated.approvals[0].requestId, 'a:1');
});

test("late snapshots cannot resurrect resolved requests or restart settled sessions", () => {
  let live = addApproval(emptySessionView(), approval('a', 'a:1'));
  live = reduceSessionEvent(live, { sessionId: 'a', type: 'requestResolved', requestId: 'a:1', kind: 'approval' });
  const resolved = hydrateSessionView(live, snapshot({ approvals: [approval('a', 'a:1')] }), 1);
  assert.equal(resolved.approvals.length, 0);
  live = reduceSessionEvent(live, { sessionId: 'a', type: 'settled' });
  const settled = hydrateSessionView(live, snapshot({ running: true, planReviews: [review('a', 'a:plan')] }), 1);
  assert.equal(settled.running, false);
  assert.equal(settled.awaitingResponse, false);
  assert.equal(settled.planReviews.length, 0);
});

test("resolving a request during initial load does not hide an ongoing run", () => {
  const live = reduceSessionEvent(emptySessionView(), { sessionId: 'a', type: 'requestResolved', requestId: 'a:1', kind: 'approval' });
  const restored = hydrateSessionView(live, snapshot({ running: true, approvals: [approval('a', 'a:1')] }), 0);
  assert.equal(restored.running, true);
  assert.equal(restored.approvals.length, 0);
});

test("restored transcripts include tool arguments and results", () => {
  const call = { ...assistant(''), content: [{ type: 'toolCall', id: 'read-1', name: 'read', arguments: { path: 'file.rs' } }] };
  const result = { role: 'toolResult', toolCallId: 'read-1', toolName: 'read', content: [{ type: 'text', text: 'source' }], isError: false, timestamp: 3 };
  const restored = hydrateSessionView(emptySessionView(), snapshot({ transcript: [user('read'), call, result] }), 0);
  assert.deepEqual(restored.toolCalls.get('read-1'), { id: 'read-1', name: 'read', args: { path: 'file.rs' }, status: 'done', result: 'source' });
});

test("approval and plan waiting timestamps survive switching away and snapshot recovery", () => {
  let a = addApproval(emptySessionView(), { ...approval('a', 'a:1'), createdAt: 1000 }, 1000);
  a = addPlanReview(a, { ...review('a', 'a:plan'), createdAt: 2000 }, 2000);
  const b = addApproval(emptySessionView(), { ...approval('b', 'b:1'), createdAt: 9000 }, 9000);
  // Switch away for 22 seconds, then restore the same pending requests.
  const restored = hydrateSessionView(a, snapshot({ running: true, approvals: [approval('a', 'a:1')], planReviews: [review('a', 'a:plan')] }), a.revision, 23000);
  assert.equal(restored.approvals[0].createdAt, 1000);
  assert.equal(restored.planReviews[0].createdAt, 2000);
  assert.equal(b.approvals[0].createdAt, 9000);
  assert.equal(23000 - restored.approvals[0].createdAt, 22000);
});

test("first recovery uses backend timestamps and queued requests retain individual start times", () => {
  const recovered = hydrateSessionView(emptySessionView(), snapshot({ approvals: [
    { ...approval('a', 'a:1'), createdAt: 1000 }, { ...approval('a', 'a:2'), createdAt: 5000 }
  ] }), 0, 23000);
  const next = reduceSessionEvent(recovered, { sessionId: 'a', type: 'requestResolved', requestId: 'a:1', kind: 'approval' });
  assert.equal(next.approvals[0].createdAt, 5000);
});

test("legacy requests use first receipt time without resetting on repeated events or hydration", () => {
  const first = addApproval(emptySessionView(), approval('a', 'a:1'), 1000);
  const repeated = addApproval(first, approval('a', 'a:1'), 23000);
  assert.equal(repeated.approvals[0].createdAt, 1000);
  const hydrated = hydrateSessionView(repeated, snapshot({ approvals: [approval('a', 'a:1')] }), repeated.revision, 44000);
  assert.equal(hydrated.approvals[0].createdAt, 1000);
});

test("thinking duration is measured from the first thinking chunk to the first answer chunk", () => {
  const stream = (type, timestamp = 2) => ({ sessionId: 'a', type: 'agent', event: { type: 'messageUpdate', message: assistant('partial', timestamp), assistantMessageEvent: { type } } });
  let view = reduceSessionEvent(emptySessionView(), { sessionId: 'a', type: 'agent', event: { type: 'messageStart', message: assistant('', 2) } }, 1000);
  assert.equal(view.thinkingStart, null);
  view = reduceSessionEvent(view, stream('thinking_start'), 1000);
  assert.deepEqual(view.thinkingStart, { timestamp: 2, startedAt: 1000 });
  view = reduceSessionEvent(view, stream('thinking_delta'), 4000);
  assert.deepEqual(view.thinkingStart, { timestamp: 2, startedAt: 1000 });
  view = reduceSessionEvent(view, stream('text_delta'), 9500);
  assert.equal(view.thinkingStart, null);
  assert.deepEqual(view.thinkingDurations.get(2), 8500);
  // The finished duration stays put and is not rewritten by later chunks.
  view = reduceSessionEvent(view, stream('text_delta'), 12000);
  assert.deepEqual(view.thinkingDurations.get(2), 8500);
});

test("a turn that ends while still thinking closes the clock, and hydration keeps durations", () => {
  let view = reduceSessionEvent(emptySessionView(), { sessionId: 'a', type: 'agent', event: { type: 'messageUpdate', message: assistant('', 7), assistantMessageEvent: { type: 'thinking_start' } } }, 500);
  view = reduceSessionEvent(view, { sessionId: 'a', type: 'settled' }, 4200);
  assert.equal(view.thinkingStart, null);
  assert.deepEqual(view.thinkingDurations.get(7), 3700);
  const restored = hydrateSessionView(view, snapshot(), view.revision);
  assert.deepEqual(restored.thinkingDurations.get(7), 3700);
  const before = view.thinkingDurations;
  assert.equal(restored.thinkingDurations === before, false);
  assert.equal(before.get(7), 3700);
});

test("consecutive turns keep separate thinking clocks", () => {
  const chunk = (type, timestamp) => ({ sessionId: 'a', type: 'agent', event: { type: 'messageUpdate', message: assistant('', timestamp), assistantMessageEvent: { type } } });
  let view = reduceSessionEvent(emptySessionView(), chunk('thinking_start', 2), 0);
  view = reduceSessionEvent(view, chunk('thinking_start', 3), 1000);
  assert.equal(view.thinkingStart.timestamp, 3);
  assert.deepEqual(view.thinkingDurations.get(2), 1000);
  view = reduceSessionEvent(view, chunk('thinking_end', 3), 2500);
  assert.deepEqual(view.thinkingDurations.get(3), 1500);
  assert.equal(view.thinkingStart, null);
});

test("the run start survives switching away, remounting the timer and later events of the same run", () => {
  let a = reduceSessionEvent(emptySessionView(), { sessionId: 'a', type: 'agent', event: { type: 'agentStart' } }, 1000);
  assert.equal(a.runStartedAt, 1000);
  a = reduceSessionEvent(a, { sessionId: 'a', type: 'agent', event: { type: 'messageUpdate', message: assistant('', 2), assistantMessageEvent: { type: 'thinking_delta' } } }, 6000);
  a = reduceSessionEvent(a, { sessionId: 'a', type: 'agent', event: { type: 'toolExecutionEnd', toolCallId: 'same', toolName: 'read', args: {}, result: 'ok' } }, 9000);
  assert.equal(a.runStartedAt, 1000);
  // Switch away for eight seconds and come back: the clock still points at the run.
  const restored = hydrateSessionView(a, snapshot({ running: true }), a.revision, 17000);
  assert.equal(restored.runStartedAt, 1000);
  assert.equal(17000 - restored.runStartedAt, 16000);
  // The other session runs on its own clock.
  const b = reduceSessionEvent(emptySessionView(), { sessionId: 'b', type: 'agent', event: { type: 'agentStart' } }, 3000);
  assert.equal(b.runStartedAt, 3000);
  assert.equal(reduceSessionEvent(b, { sessionId: 'b', type: 'settled' }, 8000).runStartedAt, null);
});

test("a finished run clears the stamp and the next run starts a fresh clock", () => {
  let view = reduceSessionEvent(emptySessionView(), { sessionId: 'a', type: 'agent', event: { type: 'agentStart' } }, 1000);
  view = reduceSessionEvent(view, { sessionId: 'a', type: 'settled' }, 4000);
  assert.equal(view.runStartedAt, null);
  view = reduceSessionEvent(view, { sessionId: 'a', type: 'agent', event: { type: 'agentStart' } }, 9000);
  assert.equal(view.runStartedAt, 9000);
  view = reduceSessionEvent(view, { sessionId: 'a', type: 'error', message: 'boom' }, 9500);
  assert.equal(view.runStartedAt, null);
});

test("a session restored mid-run measures from its own last prompt", () => {
  const transcript = [user('go', 7000), assistant('partial', 8000)];
  const restored = hydrateSessionView(emptySessionView(), snapshot({ running: true, transcript }), 0, 20000);
  assert.equal(restored.runStartedAt, 7000);
  assert.equal(20000 - restored.runStartedAt, 13000);
  assert.equal(hydrateSessionView(emptySessionView(), snapshot({ transcript }), 0, 20000).runStartedAt, null);
});

test("a stale snapshot cannot restart the clock of a session that already settled", () => {
  let live = reduceSessionEvent(emptySessionView(), { sessionId: 'a', type: 'agent', event: { type: 'agentStart' } }, 1000);
  live = reduceSessionEvent(live, { sessionId: 'a', type: 'settled' }, 4000);
  const stale = hydrateSessionView(live, snapshot({ running: true, transcript: [user('go', 1000)] }), 1, 9000);
  assert.equal(stale.running, false);
  assert.equal(stale.runStartedAt, null);
});
