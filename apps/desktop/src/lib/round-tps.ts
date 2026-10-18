//! Average output speed of the last conversation round, in tokens per second.

import type { AgentMessage, AssistantMessage, SessionStats } from "../types";
import { assistantText, assistantThinking, assistantToolCalls } from "./messages";

/** Characters per output token, used until a finished reply calibrates the ratio. */
const CHARS_PER_TOKEN = 2;

/** Characters the model produced, matching what output tokens count. */
function outputChars(message: AssistantMessage): number {
  const calls = assistantToolCalls(message).reduce((sum, call) => sum + JSON.stringify(call.arguments).length, 0);
  return assistantThinking(message).length + assistantText(message).length + calls;
}

/** The reply still streaming, if the session is mid-generation. */
function streamingReply(messages: AgentMessage[]): AssistantMessage | undefined {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role !== "assistant") continue;
    return message.stopReason === "pending" ? message : undefined;
  }
  return undefined;
}

/** Characters per output token, measured on this session's finished replies. */
export function charsPerToken(messages: AgentMessage[]): number {
  let tokens = 0;
  let chars = 0;
  for (const message of messages) {
    if (message.role !== "assistant" || message.usage.output <= 0) continue;
    tokens += message.usage.output;
    chars += outputChars(message);
  }
  return tokens > 0 && chars > 0 ? chars / tokens : CHARS_PER_TOKEN;
}

/**
 * Average output speed of the last round. A finished round reports the exact
 * backend measurement. While one runs, the requests that already finished are
 * blended with the reply currently streaming, whose tokens are estimated from
 * its length against this session's measured characters-per-token ratio, so
 * the number keeps moving before the round settles.
 */
export function roundTps(messages: AgentMessage[], stats: SessionStats | null, running: boolean, now = Date.now()): number | null {
  const performance = stats?.performance;
  const streaming = running ? streamingReply(messages) : undefined;
  if (!streaming) return performance?.lastRoundTps ?? null;
  const doneTps = performance?.lastRoundTps ?? null;
  // Finished requests count only while their generation time is known.
  const doneTokens = doneTps && doneTps > 0 ? performance?.lastRoundUsage?.output ?? 0 : 0;
  const doneMs = doneTokens > 0 && doneTps ? doneTokens / doneTps * 1000 : 0;
  const liveTokens = outputChars(streaming) / charsPerToken(messages);
  const liveMs = Math.max(0, now - streaming.timestamp);
  const tokens = doneTokens + liveTokens;
  const ms = doneMs + liveMs;
  return ms > 0 && tokens > 0 ? tokens / ms * 1000 : null;
}
