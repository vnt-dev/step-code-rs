import type { AgentMessage, ApprovalRequest, AssistantStreamEvent, Bootstrap, ExecutionTiming, PlanReviewRequest, SessionEvent, SessionStats, ToolCallView } from "../types";

export interface SessionView {
  executionTimings: ExecutionTiming[];
  messages: AgentMessage[];
  toolCalls: Map<string, ToolCallView>;
  /** Thinking duration in ms, keyed by the assistant message timestamp. */
  thinkingDurations: Map<number, number>;
  /** Thinking currently streaming: the message timestamp and when it began. */
  thinkingStart: { timestamp: number; startedAt: number } | null;
  /** When the current run began, so live timers survive remounts and switches. */
  runStartedAt: number | null;
  notices: string[];
  approvals: ApprovalRequest[];
  planReviews: PlanReviewRequest[];
  running: boolean;
  awaitingResponse: boolean;
  error: string | null;
  stats: SessionStats | null;
  revision: number;
  /** Last destructive transcript change; older snapshots must not merge removed messages. */
  transcriptRevision: number;
  lifecycleRevision: number;
  approvalsRevision: number;
  planReviewsRevision: number;
  configurationRevision: number;
  configuration: { planActive?: boolean; planFilePath?: string | null; provider?: string; modelId?: string; thinkingLevel?: string };
}

export function emptySessionView(): SessionView {
  return { executionTimings: [], messages: [], toolCalls: new Map(), thinkingDurations: new Map(), thinkingStart: null, runStartedAt: null, notices: [], approvals: [], planReviews: [], running: false, awaitingResponse: false, error: null, stats: null, revision: 0, transcriptRevision: 0, lifecycleRevision: 0, approvalsRevision: 0, planReviewsRevision: 0, configurationRevision: 0, configuration: {} };
}

/** Events that keep the thinking clock of `timestamp` running. */
function keepsThinkingOpen(event: AssistantStreamEvent | undefined): boolean {
  return event?.type === "thinking_start" || event?.type === "thinking_delta";
}

/** Stamp the moment a run begins; later events of the same run keep the first one. */
function beginRun(next: SessionView, current: SessionView, now: number): void {
  if (!current.running) next.runStartedAt = now;
}

/** The prompt that started the run currently in flight, for a session restored mid-run. */
function lastPromptTimestamp(messages: AgentMessage[]): number | null {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role === "user") return message.timestamp;
  }
  return null;
}

/**
 * Advance the thinking clock for one streamed chunk. `thinking` keeps a new
 * turn's clock open, `start` supersedes the previous turn, and anything else
 * closes the clock and records how long the model thought.
 */
function advanceThinkingClock(next: SessionView, kind: "start" | "thinking" | "close", timestamp: number, now: number): void {
  const open = next.thinkingStart;
  const closes = kind === "close" || kind === "start" || (kind === "thinking" && !!open && open.timestamp !== timestamp);
  if (closes && open) {
    next.thinkingDurations = new Map(next.thinkingDurations);
    next.thinkingDurations.set(open.timestamp, Math.max(0, now - open.startedAt));
    next.thinkingStart = null;
  }
  if (kind === "thinking" && (!next.thinkingStart || next.thinkingStart.timestamp !== timestamp)) {
    next.thinkingStart = { timestamp, startedAt: now };
  }
}

function upsertMessage(messages: AgentMessage[], message: AgentMessage): AgentMessage[] {
  const index = messages.findIndex(existing => existing.role === message.role && existing.timestamp === message.timestamp
    && (existing.role !== "toolResult" || message.role !== "toolResult" || existing.toolCallId === message.toolCallId));
  if (index < 0) return [...messages, message];
  const next = [...messages];
  next[index] = message;
  return next;
}

export function renderResult(result: unknown): string {
  if (result == null) return "";
  if (typeof result === "string") return result;
  const record = result as { content?: { type?: string; text?: string }[] };
  if (Array.isArray(record.content)) {
    const text = record.content.filter(block => block.type === "text" && typeof block.text === "string").map(block => block.text).join("\n");
    if (text) return text;
  }
  return JSON.stringify(result, null, 2);
}

function preserveRequestTimes<T extends { requestId: string; createdAt?: number }>(requests: T[], previous: T[], now: number): T[] {
  const times = new Map(previous.map(request => [request.requestId, request.createdAt]));
  return requests.map(request => ({ ...request, createdAt: request.createdAt ?? times.get(request.requestId) ?? now }));
}

type SessionSnapshot = Pick<Bootstrap, "transcript" | "executionTimings" | "running" | "approvals" | "planReviews" | "stats">;

export function hydrateSessionView(current: SessionView, snapshot: SessionSnapshot, revision: number, now = Date.now()): SessionView {
  // Events delivered while the snapshot was in flight are newer than it.
  const changed = current.revision !== revision;
  const replaced = current.transcriptRevision > revision;
  const messages = replaced ? current.messages : changed ? current.messages.reduce(upsertMessage, snapshot.transcript) : snapshot.transcript;
  const toolCalls = new Map<string, ToolCallView>();
  for (const message of messages) {
    if (message.role === "assistant") {
      for (const block of message.content) if (block.type === "toolCall") {
        toolCalls.set(block.id, { id: block.id, name: block.name, args: block.arguments, status: snapshot.running ? "running" : "done" });
      }
    } else if (message.role === "toolResult") {
      const previous = toolCalls.get(message.toolCallId);
      toolCalls.set(message.toolCallId, { id: message.toolCallId, name: message.toolName, args: previous?.args ?? {}, status: message.isError ? "error" : "done", result: renderResult(message) });
    }
  }
  for (const [id, call] of current.toolCalls) if (changed || !toolCalls.has(id)) toolCalls.set(id, call);
  const running = current.lifecycleRevision > revision ? current.running : snapshot.running;
  return { ...current, messages, toolCalls, thinkingDurations: new Map(current.thinkingDurations),
    runStartedAt: current.runStartedAt ?? (running ? lastPromptTimestamp(messages) : null),
    executionTimings: replaced ? current.executionTimings : [...new Map([...(snapshot.executionTimings ?? []), ...current.executionTimings].map(timing => [timing.firstMessageTimestamp, timing])).values()],
    running,
    awaitingResponse: changed ? current.awaitingResponse : snapshot.running && !snapshot.approvals.length && !snapshot.planReviews.length && messages.at(-1)?.role === "user",
    approvals: current.approvalsRevision > revision ? current.approvals : preserveRequestTimes(snapshot.approvals, current.approvals, now),
    planReviews: current.planReviewsRevision > revision ? current.planReviews : preserveRequestTimes(snapshot.planReviews, current.planReviews, now),
    stats: replaced ? current.stats : snapshot.stats,
  };
}

export function reduceSessionEvent(current: SessionView, event: SessionEvent, now = Date.now()): SessionView {
  const next = { ...current, revision: current.revision + 1 };
  const notice = (text: string) => { next.notices = [...current.notices, text]; };
  switch (event.type) {
    case "messagesRecalled": {
      // Rebuild tool state from the retained transcript instead of merging
      // removed tool calls back in from the previous view.
      const retained = hydrateSessionView(emptySessionView(), {
        transcript: event.messages, executionTimings: event.executionTimings,
        running: false, approvals: [], planReviews: [], stats: null,
      }, 0, now);
      const timestamps = new Set(event.messages.filter(message => message.role === "assistant").map(message => message.timestamp));
      return { ...retained, revision: next.revision, transcriptRevision: next.revision,
        lifecycleRevision: next.revision, approvalsRevision: next.revision, planReviewsRevision: next.revision,
        configuration: current.configuration, configurationRevision: current.configurationRevision,
        thinkingDurations: new Map([...current.thinkingDurations].filter(([timestamp]) => timestamps.has(timestamp))),
      };
    }
    case "executionCompleted":
      next.executionTimings = [...current.executionTimings.filter(timing => timing.firstMessageTimestamp !== event.timing.firstMessageTimestamp), event.timing];
      break;
    case "agent": {
      const agent = event.event;
      if (agent.type === "messageStart" || agent.type === "messageUpdate" || agent.type === "messageEnd") {
        next.running = true;
        next.lifecycleRevision = next.revision;
        beginRun(next, current, now);
        next.messages = upsertMessage(current.messages, agent.message);
        if (agent.message.role === "assistant") {
          next.thinkingDurations = new Map(current.thinkingDurations);
          advanceThinkingClock(next,
            agent.type === "messageStart" ? "start"
              : agent.type === "messageEnd" ? "close"
                : keepsThinkingOpen(agent.assistantMessageEvent) ? "thinking" : "close",
            agent.message.timestamp, now);
          next.awaitingResponse = false;
        }
      } else if (agent.type === "agentStart") {
        next.running = true;
        next.lifecycleRevision = next.revision;
        beginRun(next, current, now);
      } else if (agent.type === "toolExecutionStart" || agent.type === "toolExecutionEnd") {
        next.running = true;
        next.lifecycleRevision = next.revision;
        beginRun(next, current, now);
        next.toolCalls = new Map(current.toolCalls);
        const previous = current.toolCalls.get(agent.toolCallId);
        next.toolCalls.set(agent.toolCallId, agent.type === "toolExecutionStart"
          ? { id: agent.toolCallId, name: agent.toolName, args: agent.args, status: "running" }
          : { id: agent.toolCallId, name: agent.toolName, args: previous?.args ?? {}, status: agent.isError ? "error" : "done", result: renderResult(agent.result) });
      }
      break;
    }
    case "settled":
      next.lifecycleRevision = next.revision;
      next.running = false;
      next.awaitingResponse = false;
      next.runStartedAt = null;
      next.approvals = [];
      next.planReviews = [];
      next.approvalsRevision = next.revision;
      next.planReviewsRevision = next.revision;
      if (current.thinkingStart) advanceThinkingClock(next, "close", current.thinkingStart.timestamp, now);
      break;
    case "error":
      next.lifecycleRevision = next.revision;
      next.error = event.message;
      next.running = false;
      next.awaitingResponse = false;
      next.runStartedAt = null;
      next.approvals = [];
      next.planReviews = [];
      next.approvalsRevision = next.revision;
      next.planReviewsRevision = next.revision;
      if (current.thinkingStart) advanceThinkingClock(next, "close", current.thinkingStart.timestamp, now);
      break;
    case "requestResolved":
      if (event.kind === "approval") {
        next.approvals = current.approvals.filter(request => request.requestId !== event.requestId);
        next.approvalsRevision = next.revision;
      } else {
        next.planReviews = current.planReviews.filter(request => request.requestId !== event.requestId);
        next.planReviewsRevision = next.revision;
      }
      break;
    case "compactionStart": notice(`Compacting conversation (${event.reason})…`); break;
    case "compactionEnd": notice(event.errorMessage ? `Compaction failed: ${event.errorMessage}` : "Conversation compacted."); break;
    case "autoRetryStart": notice(`Retrying after provider error (attempt ${event.attempt}/${event.maxAttempts})…`); break;
    case "autoRetryEnd": notice(event.success ? "Retry succeeded." : `Retry gave up: ${event.finalError ?? "unknown error"}`); break;
    case "modelChanged":
      next.configurationRevision = next.revision;
      next.configuration = { ...current.configuration, provider: event.provider, modelId: event.modelId, thinkingLevel: event.thinkingLevel };
      break;
    case "planModeChanged":
      next.configurationRevision = next.revision;
      next.configuration = { ...current.configuration, planActive: event.active, planFilePath: event.planFilePath };
      if (!event.active) notice("计划模式已退出，文件修改已恢复。");
      break;
  }
  return next;
}

export function addApproval(current: SessionView, request: ApprovalRequest, now = Date.now()): SessionView {
  request = preserveRequestTimes([request], current.approvals, now)[0];
  return { ...current, awaitingResponse: false, revision: current.revision + 1, approvalsRevision: current.revision + 1, approvals: [...current.approvals.filter(item => item.requestId !== request.requestId), request] };
}

export function addPlanReview(current: SessionView, request: PlanReviewRequest, now = Date.now()): SessionView {
  request = preserveRequestTimes([request], current.planReviews, now)[0];
  return { ...current, awaitingResponse: false, revision: current.revision + 1, planReviewsRevision: current.revision + 1, planReviews: [...current.planReviews.filter(item => item.requestId !== request.requestId), request] };
}
