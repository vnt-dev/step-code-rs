//! Helpers for reading the wire message shapes.

import type {
  AgentMessage,
  AssistantMessage,
  AssistantToolCall,
  ExecutionTiming,
  TextBlock,
  ToolResultMessage,
  UserContent,
  UserMessage,
} from "../types";

/** Plain text of a user message, if it is a plain-text message. */
export function userText(message: UserMessage): string {
  const content: UserContent = message.content;
  if (typeof content === "string") return content;
  return content.filter((block) => block.type === "text")
    .map((block) => (block as TextBlock).text)
    .join("\n");
}

/** Whether the user message carries images. */
export function userHasImages(message: UserMessage): boolean {
  const content: UserContent = message.content;
  return Array.isArray(content) && content.some((block) => block.type === "image");
}

/** Concatenated assistant text of a message. */
export function assistantText(message: AssistantMessage): string {
  return message.content
    .filter((block): block is TextBlock => block.type === "text")
    .map((block) => block.text)
    .join("");
}

/** Concatenated thinking text of a message. */
export function assistantThinking(message: AssistantMessage): string {
  return message.content
    .filter((block) => block.type === "thinking")
    .map((block) => (block as { thinking: string }).thinking)
    .join("\n");
}

/** Tool calls requested by an assistant message. */
export function assistantToolCalls(message: AssistantMessage): AssistantToolCall[] {
  return message.content.filter((block): block is AssistantToolCall => block.type === "toolCall");
}

/** Text of a tool result message. */
export function toolResultText(message: ToolResultMessage): string {
  return message.content
    .filter((block) => block.type === "text")
    .map((block) => (block as TextBlock).text)
    .join("\n");
}

/** Stable identity for de-duplicating streamed events. */
export function messageKey(message: AgentMessage): string {
  return JSON.stringify(message);
}

/** Whether a message should render in the timeline. */
export function isRenderable(message: AgentMessage): boolean {
  return message.role === "user" || message.role === "assistant";
}

/** Format a token count compactly (1234 -> "1.2k"). */
export function formatTokens(count: number): string {
  if (count < 1000) return String(count);
  if (count < 1_000_000) return `${(count / 1000).toFixed(1)}k`;
  return `${(count / 1_000_000).toFixed(2)}M`;
}

/** Format a cost in USD. */
export function formatCost(cost: number): string {
  if (cost === 0) return "$0.00";
  if (cost < 0.01) return `$${cost.toFixed(4)}`;
  return `$${cost.toFixed(2)}`;
}

/** m:ss below an hour, h:mm:ss above. */
function clockTime(seconds: number): string {
  const total = Math.floor(seconds);
  const minutes = Math.floor(total / 60);
  if (minutes < 60) return `${minutes}:${String(total % 60).padStart(2, "0")}`;
  return `${Math.floor(minutes / 60)}:${String(minutes % 60).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
}

/** Elapsed time with tenths of a second under a minute, m:ss above. */
export function formatElapsed(ms: number): string {
  const seconds = Math.max(0, ms) / 1000;
  return seconds < 60 ? `${seconds.toFixed(1)}s` : clockTime(seconds);
}

/** Elapsed time in whole seconds under a minute, m:ss above. */
export function formatDuration(ms: number): string {
  const seconds = Math.max(0, ms) / 1000;
  return seconds < 60 ? `${Math.floor(seconds)}s` : clockTime(seconds);
}

/** Thinking duration as "8 秒" or "2 分 5 秒", for the collapsed thinking row. */
export function formatThinkingDuration(ms: number): string {
  const seconds = Math.max(1, Math.round(Math.max(0, ms) / 1000));
  if (seconds < 60) return `${seconds} 秒`;
  return `${Math.floor(seconds / 60)} 分 ${seconds % 60} 秒`;
}

export interface ExecutionGroup {
  messages: AgentMessage[];
  completed: boolean;
  durationMs?: number;
  timing?: ExecutionTiming;
  finalAssistant?: AssistantMessage;
}

/** Group a complete execution, including supplemental user messages and tool turns. */
export function executionGroups(messages: AgentMessage[], running: boolean, timings: ExecutionTiming[] = []): ExecutionGroup[] {
  const groups: ExecutionGroup[] = [];
  for (let index = 0; index < messages.length;) {
    const first = messages[index];
    const timing = first.role === "user" ? timings.find(timing => timing.firstMessageTimestamp === first.timestamp) : undefined;
    let end = index + 1;
    if (timing) {
      let finalIndex = -1;
      for (let position = messages.length - 1; position >= index; position--) {
        if (messages[position].timestamp === timing.lastMessageTimestamp) { finalIndex = position; break; }
      }
      if (finalIndex >= index) end = finalIndex + 1;
    }
    if (!timing || end === index + 1) {
      while (end < messages.length && (running || messages[end].role !== "user")) end++;
    }
    const batch = messages.slice(index, end);
    const completed = Boolean(timing) || end < messages.length || !running;
    const finalAssistant = completed ? [...batch].reverse().find((message): message is AssistantMessage => message.role === "assistant") : undefined;
    groups.push({ messages: batch, completed, durationMs: timing?.durationMs, timing, finalAssistant });
    index = end;
  }
  return groups;
}

export function formatWorkDuration(ms?: number): string {
  if (ms === undefined) return "已工作";
  const seconds = Math.floor(Math.max(0, ms) / 1000);
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor(seconds / 60) % 60;
  const parts = [];
  if (hours) parts.push(`${hours} 小时`);
  if (minutes || hours) parts.push(`${minutes} 分`);
  parts.push(`${seconds % 60} 秒`);
  return `已工作 ${parts.join(" ")}`;
}

/** Allocate whole seconds without changing the displayed execution total. */
export function workDurationBreakdown(timing?: ExecutionTiming): { label: string; seconds: number }[] {
  if (!timing) return [];
  const entries = [
    { label: "工具调用", ms: timing.toolDurationMs ?? 0 },
    { label: "API 调用", ms: timing.apiDurationMs ?? 0 },
    { label: "等待审批", ms: timing.approvalWaitDurationMs ?? 0 },
    { label: "重试等待", ms: timing.retryWaitDurationMs ?? 0 },
    { label: "其他处理", ms: timing.otherDurationMs ?? 0 },
  ];
  // Legacy or incomplete records must not invent a partition.
  if (entries.some(entry => !Number.isFinite(entry.ms) || entry.ms < 0)
    || entries.reduce((sum, entry) => sum + entry.ms, 0) !== timing.durationMs) return [];
  const seconds = entries.map(entry => Math.floor(entry.ms / 1000));
  const remaining = Math.floor(timing.durationMs / 1000) - seconds.reduce((sum, value) => sum + value, 0);
  const ranked = entries.map((entry, index) => ({ index, remainder: entry.ms % 1000 }))
    .sort((a, b) => b.remainder - a.remainder || a.index - b.index);
  for (let index = 0; index < remaining; index++) seconds[ranked[index].index]++;
  return entries.map((entry, index) => ({ label: entry.label, seconds: seconds[index] }))
    .filter(entry => entry.seconds > 0);
}

export function formatWorkBreakdown(timing?: ExecutionTiming): string {
  return workDurationBreakdown(timing)
    .map(entry => `${entry.label} ${formatWorkDuration(entry.seconds * 1000).replace(/^已工作 /, "")}`)
    .join(" · ");
}

/** Human label for a stop reason. */
export function stopReasonLabel(reason: string): string {
  switch (reason) {
    case "stop":
      return "done";
    case "length":
      return "length limit";
    case "toolUse":
      return "tool use";
    case "error":
      return "error";
    case "aborted":
      return "aborted";
    default:
      return reason;
  }
}
