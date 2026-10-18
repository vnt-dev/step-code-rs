//! Browser-only stand-in for the Tauri command/event surface.
//!
//! Served by `vite.harness.config.ts`, which aliases `@tauri-apps/api/core`
//! and `@tauri-apps/api/event` to this module so the real `src/App.tsx` can
//! run in a plain browser. `window.__qa__` drives scripted scenarios
//! (thinking indicator, approval bar) and records every command call.

import type { AgentMessage, ApprovalRequest, Bootstrap, ExecutionTiming, ModelInfo, PlanReviewRequest } from "../src/types";

type Listener = (event: { payload: unknown }) => void;

const listeners = new Map<string, Set<Listener>>();
const calls: string[] = [];

const now = Date.now();
const minute = 60_000;

const model: ModelInfo = {
  provider: "step",
  id: "step-3",
  name: "Step 3",
  reasoning: true,
  supportedThinkingLevels: ["off", "minimal", "low", "medium", "high", "xhigh"],
  contextWindow: 256_000,
  configured: true,
  supportsImages: true,
};
const miniModel: ModelInfo = { ...model, id: "step-3-mini", name: "Step 3 mini", reasoning: false, supportedThinkingLevels: ["off"] };

const bootstrap: Bootstrap = {
  cwd: "fixtures/example-app",
  dataDir: "fixtures/data",
  settings: {},
  models: [model, miniModel],
  currentModel: model,
  thinkingLevel: "medium",
  approvalMode: "ask",
  sessions: [
    { id: "s1", path: "sessions/s1.jsonl", cwd: "fixtures/example-app", createdAt: now - 40 * minute, modifiedAt: now - 2 * minute, parentSessionId: null, name: "调整审批交互", project: null },
    { id: "s2", path: "sessions/s2.jsonl", cwd: "fixtures/example-app", createdAt: now - 3 * 24 * 60 * minute, modifiedAt: now - 90 * minute, parentSessionId: null, name: "修复登录回调", project: null },
  ],
  projects: ["fixtures/example-app"],
  removedProjects: [],
  activeSession: { id: "s1", path: "sessions/s1.jsonl", cwd: "fixtures/example-app", createdAt: now - 40 * minute, modifiedAt: now - 2 * minute, parentSessionId: null, name: "调整审批交互", project: null },
  planActive: false,
  planFilePath: null,
  transcript: [],
  stats: { inputTokens: 18_400, outputTokens: 3_260, cacheReadTokens: 0, cacheWriteTokens: 0, cost: 0.0213, turns: 3,
    performance: { lastRoundTps: 42.6, sessionTps: 38.1, lastRoundSamples: 2, sessionSamples: 3,
      lastRoundUsage: { input: 12_400, output: 1_820, cacheRead: 0, cacheWrite: 0 },
      sessionUsage: { input: 18_400, output: 3_260, cacheRead: 0, cacheWrite: 0 }, sessionRounds: 3 } },
  approvals: [],
  planReviews: [],
  running: false,
  stepAuthenticated: true,
};

const transcript: AgentMessage[] = [
  { role: "user", content: "帮我把命令审批从弹窗改成输入框上方的内嵌卡片，不要影响切换会话。", timestamp: now - 6 * minute },
  { role: "assistant", content: [
    { type: "text", text: "我先读取审批组件，检查它在会话页面中的位置。" },
    { type: "toolCall", id: "history-read", name: "read", arguments: { path: "ApprovalDialog.tsx" } },
  ], api: "step", provider: "step", model: "step-3", usage: { input: 100, output: 50, cacheRead: 0, cacheWrite: 0 }, stopReason: "toolUse", timestamp: now - 5 * minute },
  { role: "toolResult", toolCallId: "history-read", toolName: "read", content: [{ type: "text", text: "审批组件读取完成。" }], isError: false, timestamp: now - 4 * minute },
  { role: "assistant", content: [
    { type: "toolCall", id: "history-edit", name: "edit", arguments: { path: "apps/desktop/src/components/ApprovalBar.tsx", edits: [{ old_string: "modal", new_string: "inline card" }] } },
    { type: "toolCall", id: "history-bash", name: "bash", arguments: { command: "cd apps/desktop && npx tsc --noEmit" } },
  ], api: "step", provider: "step", model: "step-3", usage: { input: 100, output: 50, cacheRead: 0, cacheWrite: 0 }, stopReason: "toolUse", timestamp: now - 3 * minute },
  { role: "toolResult", toolCallId: "history-edit", toolName: "edit", content: [{ type: "text", text: "已把 ApprovalBar 改为内嵌卡片。" }], isError: false, timestamp: now - 2 * minute },
  { role: "toolResult", toolCallId: "history-bash", toolName: "bash", content: [{ type: "text", text: "$ cd apps/desktop && npx tsc --noEmit\ntsc: no errors" }], isError: false, timestamp: now - 90_000 },
  { role: "assistant", content: [
    { type: "toolCall", id: "history-grep", name: "grep", arguments: { pattern: "ApprovalBar", path: "apps/desktop/src" } },
  ], api: "step", provider: "step", model: "step-3", usage: { input: 100, output: 20, cacheRead: 0, cacheWrite: 0 }, stopReason: "toolUse", timestamp: now - 95_000 },
  { role: "toolResult", toolCallId: "history-grep", toolName: "grep", content: [{ type: "text", text: "apps/desktop/src/App.tsx:12:import ApprovalBar" }], isError: false, timestamp: now - 80_000 },
  { role: "assistant", content: [
    { type: "toolCall", id: "history-web", name: "search_web", arguments: { query: "tauri v2 内嵌审批" } },
  ], api: "step", provider: "step", model: "step-3", usage: { input: 100, output: 20, cacheRead: 0, cacheWrite: 0 }, stopReason: "toolUse", timestamp: now - 70_000 },
  { role: "toolResult", toolCallId: "history-web", toolName: "search_web", content: [{ type: "text", text: "搜索服务暂不可用，请稍后重试。" }], isError: true, timestamp: now - 60_000 },
  {
    role: "assistant",
    content: [
      { type: "thinking", thinking: "用户希望审批不再遮挡界面。先看 ApprovalDialog 与 App 的渲染位置，再决定内嵌方案。" },
      { type: "text", text: "好的。我先把现在的实现看一遍：\n\n- 审批目前是居中的模态弹窗（`ApprovalDialog.tsx`），会盖住侧边栏和输入框；\n- 交互目标：改为**输入框上方**的内嵌确认卡片，会话列表保持可操作。\n\n我先确认渲染层级，再动手改。\n\n修改内容（3 个文件，均为小范围补丁）：\n\n| 文件 | 表 | 修改 | 说明 |\n| --- | --- | --- | --- |\n| dal/dataobject/rework/ReworkOrderDetailDO.java | dms_supply_rework_order_detail | 增加 @IgnoreTenantTable | 主单表忽略租户 |\n| dal/dataobject/rework/ReworkOrderRecordDO.java | dms_supply_rework_order_record | 增加 @IgnoreTenantTable | 记录表忽略租户 |\n| dal/dataobject/rework/ReworkOrderSourceSnapshotDO.java | dms_supply_rework_order_source_snapshot | 增加 @IgnoreTenantTable | 来源快照忽略租户 |\n\n修复后（与排产管理 3c97be4e78f 的处理方式一致）：\n\n- 三张表的查询/写入都不再拼接租户条件，历史数据（tenant_id 为旧租户值）可直接查出，新数据不再自动写入 tenant_id；\n- 无需数据迁移 SQL，表结构无变化。" },
    ],
    api: "step",
    provider: "step",
    model: "step-3",
    usage: { input: 12_400, output: 1_820, cacheRead: 0, cacheWrite: 0 },
    stopReason: "stop",
    timestamp: now - 12000,
  },
];


const commandArgs: { command: string; args: Record<string, unknown> }[] = [];
type MockSession = { transcript: AgentMessage[]; running: boolean; approvals: ApprovalRequest[]; planReviews: PlanReviewRequest[]; executionTimings?: ExecutionTiming[] };
const live = new Map<string, MockSession>([
  ["s1", { transcript, running: false, approvals: [], planReviews: [], executionTimings: [{ firstMessageTimestamp: now - 6 * minute, lastMessageTimestamp: now - 12000, durationMs: 348000 }] }],
  ["s2", { transcript: [{ role: "user", content: "会话 B 的消息", timestamp: now - minute }], running: false, approvals: [], planReviews: [] }],
]);
const session = (id: string) => live.get(id)!;

function emit(name: string, payload: unknown) {
  const data = payload as Record<string, any>;
  const id = data?.sessionId ?? bootstrap.activeSession?.id ?? "s1";
  const state = live.get(id);
  if (state && name === "approval-request") state.approvals.push(data as ApprovalRequest);
  if (state && name === "plan-review-request") state.planReviews.push(data as PlanReviewRequest);
  if (state && name === "session-event") {
    if (data.type === "settled" || data.type === "error") { state.running = false; state.approvals = []; state.planReviews = []; }
    if (data.type === "agent") {
      if (data.event.type === "agentStart") state.running = true;
      const message = data.event.message as AgentMessage | undefined;
      if (message) {
        const index = state.transcript.findIndex(previous => previous.role === message.role && previous.timestamp === message.timestamp);
        if (index < 0) state.transcript.push(message); else state.transcript[index] = message;
      }
    }
    if (data.type === "requestResolved") {
      state.approvals = state.approvals.filter(request => request.requestId !== data.requestId);
      state.planReviews = state.planReviews.filter(request => request.requestId !== data.requestId);
    }
  }
  listeners.get(name)?.forEach(handler => handler({ payload }));
}

export async function invoke(command: string, args: Record<string, unknown> = {}): Promise<unknown> {
  calls.push(command);
  commandArgs.push({ command, args });
  const id = String(args.sessionId ?? bootstrap.activeSession?.id ?? "s1");
  switch (command) {
    case "get_bootstrap": {
      const active = bootstrap.sessions.find(item => item.id === id) ?? null;
      const state = active ? session(id) : { transcript: [], running: false, approvals: [], planReviews: [] };
      return structuredClone({ ...bootstrap, activeSession: active, ...state });
    }
    case "get_transcript": return structuredClone(session(id).transcript);
    case "recall_last_user_message": {
      const state = session(id);
      if (state.running) throw new Error("请先停止当前执行，再撤回消息。");
      const index = state.transcript.map(message => message.role).lastIndexOf("user");
      if (index < 0 || state.transcript[index].timestamp !== args.timestamp) throw new Error("会话消息已变化，请刷新后重试。");
      state.transcript = state.transcript.slice(0, index);
      state.executionTimings = (state.executionTimings ?? []).filter(timing => timing.lastMessageTimestamp < Number(args.timestamp));
      emit("session-event", { sessionId: id, type: "messagesRecalled", messages: state.transcript, executionTimings: state.executionTimings });
      return null;
    }
    case "get_stats": return { inputTokens: 18_400, outputTokens: 3_260, cacheReadTokens: 0, cacheWriteTokens: 0, cost: 0.0213, turns: 3,
      performance: { lastRoundTps: 42.6, sessionTps: 38.1, lastRoundSamples: 2, sessionSamples: 3,
        lastRoundUsage: { input: 12_400, output: 1_820, cacheRead: 0, cacheWrite: 0 },
        sessionUsage: { input: 18_400, output: 3_260, cacheRead: 0, cacheWrite: 0 }, sessionRounds: 3 } };
    case "get_providers": return [];
    case "get_usage_statistics": return {
      records: Array.from({ length: 30 }, (_, index) => ({ timestamp: now - index * 24 * 60 * minute, provider: "step", model: "step-3", tokens: 12000 + (index % 7) * 3200 })),
      longestChatMs: 348000,
    };
    case "get_model_health": return [];
    case "list_request_records": return { records: [], nextCursor: null, models: ["step/step-3"], storageError: null };
    case "list_sessions": return structuredClone(bootstrap.sessions);
    case "open_session": {
      bootstrap.activeSession = bootstrap.sessions.find(item => item.id === id)!;
      return structuredClone(bootstrap.activeSession);
    }
    case "new_session": {
      const created = { ...bootstrap.sessions[0], id: `s${bootstrap.sessions.length + 1}`, name: "新会话", cwd: String(args.cwd ?? bootstrap.cwd), createdAt: Date.now(), modifiedAt: Date.now() };
      bootstrap.sessions.unshift(created);
      bootstrap.activeSession = created;
      live.set(created.id, { transcript: [], running: false, approvals: [], planReviews: [] });
      return structuredClone(created);
    }
    case "add_project": {
      const path = "fixtures/another-app";
      bootstrap.projects.push(path);
      return path;
    }
    case "respond_approval":
    case "respond_plan_review": {
      for (const [sessionId, state] of live) {
        if ([...state.approvals, ...state.planReviews].some(request => request.requestId === args.requestId)) {
          emit("session-event", { sessionId, type: "requestResolved", requestId: args.requestId, kind: command === "respond_approval" ? "approval" : "planReview" });
        }
      }
      return null;
    }
    case "prompt":
      emit("session-event", { sessionId: id, type: "agent", event: { type: "agentStart" } });
      emit("session-event", { sessionId: id, type: "agent", event: { type: "messageEnd", message: { role: "user", content: args.text, timestamp: Date.now() } } });
      return null;
    case "abort": emit("session-event", { sessionId: id, type: "settled" }); return null;
    case "steer":
    case "follow_up": return null;
    case "list_skills": return [];
    default: return null;
  }
}

export async function listen(name: string, handler: Listener): Promise<() => void> {
  const handlers = listeners.get(name) ?? new Set<Listener>();
  handlers.add(handler);
  listeners.set(name, handlers);
  return () => { handlers.delete(handler); };
}

export function scenario(action: string) {
  if (action === "approvals") for (const id of ["s1", "s2"]) {
    emit("session-event", { sessionId: id, type: "agent", event: { type: "agentStart" } });
    emit("approval-request", { sessionId: id, requestId: `${id}:call-1`, toolName: "write", argsSummary: `file=${id}.rs`, reason: `${id} 独立审批`, hazardous: false });
  }
  if (action === "plan") emit("plan-review-request", { sessionId: "s1", requestId: "s1:plan", planFilePath: "s1.md", planContents: "# 会话 A 的计划\n\n1. **只处理 A**\n2. 保留 B 的运行状态" });
  if (action === "finish-a") {
    emit("session-event", { sessionId: "s1", type: "agent", event: { type: "messageEnd", message: { ...(transcript[1] as object), content: [{ type: "text", text: "会话 A 已在后台完成" }], timestamp: Date.now() } } });
    emit("session-event", { sessionId: "s1", type: "settled" });
  }
  if (action === "thinking") {
    // A live turn: thinking streams for a few seconds, then a tool call runs.
    const timestamp = Date.now();
    const thinking = "先看审批卡片，再确认它挂在哪个容器上。";
    const message = (content: unknown[], stopReason = "pending") => ({ role: "assistant", content, api: "step", provider: "step", model: "step-3", usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, stopReason, timestamp });
    const send = (delay: number, run: () => void) => setTimeout(run, delay);
    const emitAgent = (event: Record<string, unknown>) => emit("session-event", { sessionId: "s1", type: "agent", event });
    emitAgent({ type: "agentStart" });
    emitAgent({ type: "messageStart", message: message([]) });
    send(400, () => emitAgent({ type: "messageUpdate", message: message([{ type: "thinking", thinking: "先看审批卡片" }]), assistantMessageEvent: { type: "thinking_start", content_index: 0, partial: message([]) } }));
    send(2600, () => emitAgent({ type: "messageUpdate", message: message([{ type: "thinking", thinking }]), assistantMessageEvent: { type: "thinking_delta", content_index: 0, delta: "，再确认它挂在哪个容器上。", partial: message([]) } }));
    send(5200, () => emitAgent({ type: "messageUpdate", message: message([{ type: "thinking", thinking }, { type: "text", text: "我先把审批卡片挂到输入框上方。" }]), assistantMessageEvent: { type: "text_start", content_index: 1, partial: message([]) } }));
    send(5600, () => {
      emitAgent({ type: "messageUpdate", message: message([{ type: "thinking", thinking }, { type: "text", text: "我先把审批卡片挂到输入框上方。" }, { type: "toolCall", id: "live-read", name: "read", arguments: { path: "apps/desktop/src/components/Composer.tsx" } }]), assistantMessageEvent: { type: "tool_call_start", content_index: 2, partial: message([]) } });
      emitAgent({ type: "toolExecutionStart", toolCallId: "live-read", toolName: "read", args: { path: "apps/desktop/src/components/Composer.tsx" } });
    });
    send(8600, () => emitAgent({ type: "toolExecutionEnd", toolCallId: "live-read", toolName: "read", result: { content: [{ type: "text", text: "Composer 读取完成。" }] }, isError: false }));
    send(9000, () => {
      emitAgent({ type: "messageEnd", message: message([{ type: "thinking", thinking }, { type: "text", text: "已改为内嵌卡片。" }, { type: "toolCall", id: "live-read", name: "read", arguments: { path: "apps/desktop/src/components/Composer.tsx" } }], "stop") });
      emit("session-event", { sessionId: "s1", type: "settled" });
    });
  }
}

window.__qa__ = {
  calls, commandArgs, emit,
  agent: (event: Record<string, unknown>) => emit("session-event", { sessionId: bootstrap.activeSession?.id, type: "agent", event }),
  approval: (request: Record<string, unknown>) => emit("approval-request", { sessionId: bootstrap.activeSession?.id, ...request }),
  ready: () => emit("bridge-ready", null),
  patchBootstrap: (patch: Record<string, unknown>) => Object.assign(bootstrap, patch),
};

declare global {
  interface Window {
    __qa__?: {
      calls: string[];
      commandArgs: { command: string; args: Record<string, unknown> }[];
      emit: (name: string, payload: unknown) => void;
      agent: (event: Record<string, unknown>) => void;
      approval: (request: Record<string, unknown>) => void;
      ready: () => void;
      patchBootstrap: (patch: Record<string, unknown>) => void;
    };
  }
}
