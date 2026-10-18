import type { AgentMessage, ExecutionTiming, SessionStats, Usage } from "../types";
import { executionGroups } from "./messages";

export function inputTokens(usage: Usage): number {
  return usage.input + usage.cacheRead + usage.cacheWrite;
}

export function cacheHitRate(usage: Usage): number {
  const input = inputTokens(usage);
  return input > 0 ? usage.cacheRead / input * 100 : 0;
}

export function contextUsage(messages: AgentMessage[], stats: SessionStats | null, running: boolean, timings: ExecutionTiming[]) {
  const groups = executionGroups(messages, running, timings);
  const lastRound: Usage = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
  for (const message of groups.at(-1)?.messages ?? []) {
    if (message.role !== "assistant") continue;
    lastRound.input += message.usage.input;
    lastRound.output += message.usage.output;
    lastRound.cacheRead += message.usage.cacheRead;
    lastRound.cacheWrite += message.usage.cacheWrite;
  }
  const total: Usage = {
    input: stats?.inputTokens ?? 0, output: stats?.outputTokens ?? 0,
    cacheRead: stats?.cacheReadTokens ?? 0, cacheWrite: stats?.cacheWriteTokens ?? 0,
  };
  return { contextTokens: stats?.contextTokens ?? 0, lastRound: stats?.performance?.lastRoundUsage ?? lastRound, total: stats?.performance?.sessionUsage ?? total,
    /** Rounds in this session; one round makes the cumulative row a duplicate. */
    rounds: stats?.performance?.sessionRounds ?? groups.length };
}
