//! Wire types shared with the Rust bridge.
//!
//! These mirror the serde shapes emitted by `src-tauri/src/bridge.rs` and the
//! `step-harness-agent-core` message types; the Rust side is the source of
//! truth.

export type ThinkingLevel =
  | "off"
  | "minimal"
  | "low"
  | "medium"
  | "high"
  | "xhigh"
  | "max";

export type ApprovalMode = "ask" | "bypass" | "autopilot";

export interface ModelInfo {
  provider: string;
  id: string;
  name: string;
  reasoning: boolean;
  supportedThinkingLevels: ThinkingLevel[];
  contextWindow: number;
  configured: boolean;
  supportsImages: boolean;
}

export interface SessionSummary {
  id: string;
  path: string;
  cwd: string;
  createdAt: number;
  modifiedAt: number;
  parentSessionId: string | null;
  name: string | null;
  /** Explicit project assignment; null follows the cwd default. */
  project: string | null;
}

export interface SessionStats {
  performance?: { lastRoundTps: number | null; sessionTps: number | null; lastRoundSamples: number; sessionSamples: number; lastRoundUsage?: Usage; sessionUsage?: Usage; sessionRounds?: number };
  performanceError?: string | null;
  /** Total tokens reported by the last request with usage, including output. */
  contextTokens?: number;
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  cacheWriteTokens: number;
  cost: number;
  turns: number;
}

export interface SkillSummary { name: string; description: string; filePath: string; }

export interface Bootstrap {
  executionTimings?: ExecutionTiming[];
  cwd: string;
  dataDir: string;
  settings: Record<string, unknown>;
  models: ModelInfo[];
  currentModel: ModelInfo | null;
  thinkingLevel: string;
  approvalMode: string;
  /** Plan mode is active: only the plan file may be written. */
  planActive: boolean;
  planFilePath: string | null;
  sessions: SessionSummary[];
  projects: string[];
  removedProjects: string[];
  activeSession: SessionSummary | null;
  running: boolean;
  stepAuthenticated: boolean;
  transcript: AgentMessage[];
  stats: SessionStats | null;
  approvals: ApprovalRequest[];
  planReviews: PlanReviewRequest[];
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

export interface TextBlock {
  type: "text";
  text: string;
  textSignature?: string | null;
}
export interface ImageBlock {
  type: "image";
  data: string;
  mimeType: string;
}
export type UserContentBlock = TextBlock | ImageBlock;

/** Rust serializes UserContent without an enum tag. */
export type UserContent = string | UserContentBlock[];

export interface AssistantToolCall {
  type: "toolCall";
  id: string;
  name: string;
  arguments: Record<string, unknown>;
  thoughtSignature?: string | null;
  namespace?: string | null;
}

export type AssistantBlock =
  | TextBlock
  | { type: "thinking"; thinking: string; thinkingSignature?: string | null }
  | AssistantToolCall;

export interface Usage {
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
}

export interface UserMessage {
  role: "user";
  content: UserContent;
  timestamp: number;
}

export interface AssistantMessage {
  role: "assistant";
  content: AssistantBlock[];
  api: string;
  provider: string;
  model: string;
  responseModel?: string | null;
  usage: Usage;
  stopReason: string;
  errorMessage?: string | null;
  timestamp: number;
}

export interface ToolResultMessage {
  role: "toolResult";
  toolCallId: string;
  toolName: string;
  content: UserContentBlock[];
  isError: boolean;
  details?: unknown;
  timestamp: number;
}

export interface BashExecutionMessage {
  role: "bashExecution";
  command: string;
  output: string;
  exitCode: number | null;
  cancelled: boolean;
  timestamp: number;
}

export type AgentMessage =
  | UserMessage
  | AssistantMessage
  | ToolResultMessage
  | BashExecutionMessage
  | { role: "custom" | "branchSummary" | "compactionSummary"; [key: string]: unknown };

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/** A streaming chunk from the provider, forwarded with `messageUpdate`. */
export interface AssistantStreamEvent {
  type:
    | "start"
    | "text_start"
    | "text_delta"
    | "text_end"
    | "thinking_start"
    | "thinking_delta"
    | "thinking_end"
    | "tool_call_start"
    | "tool_call_delta"
    | "tool_call_end"
    | "done"
    | "error";
}

export type AgentEvent =
  | { type: "agentStart" }
  | { type: "agentEnd"; messages: AgentMessage[] }
  | { type: "turnStart" }
  | { type: "turnEnd"; message: AgentMessage; toolResults: ToolResultMessage[] }
  | { type: "messageStart"; message: AgentMessage }
  | { type: "messageUpdate"; message: AgentMessage; assistantMessageEvent: AssistantStreamEvent }
  | { type: "messageEnd"; message: AgentMessage }
  | {
      type: "toolExecutionStart";
      toolCallId: string;
      toolName: string;
      args: Record<string, unknown>;
    }
  | {
      type: "toolExecutionUpdate";
      toolCallId: string;
      toolName: string;
      args: Record<string, unknown>;
      partialResult: unknown;
    }
  | {
      type: "toolExecutionEnd";
      toolCallId: string;
      toolName: string;
      result: unknown;
      isError: boolean;
    };

export interface ExecutionTiming {
  firstMessageTimestamp: number;
  lastMessageTimestamp: number;
  durationMs: number;
  toolDurationMs?: number;
  apiDurationMs?: number;
  approvalWaitDurationMs?: number;
  retryWaitDurationMs?: number;
  otherDurationMs?: number;
}

export type SessionEvent = { sessionId: string } & (
  | { type: "messagesRecalled"; messages: AgentMessage[]; executionTimings: ExecutionTiming[] }
  | { type: "executionCompleted"; timing: ExecutionTiming }
  | { type: "agent"; event: AgentEvent }
  | { type: "settled" }
  | {
      type: "autoRetryStart";
      attempt: number;
      maxAttempts: number;
      delayMs: number;
      errorMessage: string;
    }
  | {
      type: "autoRetryEnd";
      success: boolean;
      attempt: number;
      finalError?: string;
    }
  | { type: "compactionStart"; reason: "manual" | "threshold" | "overflow" }
  | { type: "compactionEnd"; reason: string; errorMessage?: string }
  | { type: "queueUpdate"; steering: string[]; followUp: string[] }
  | { type: "sessionInfoChanged"; name: string | null }
  | { type: "projectChanged"; project: string | null }
  | { type: "entryAppended"; entryId: string }
  | {
      type: "modelChanged";
      provider: string;
      modelId: string;
      thinkingLevel: string;
    }
  | {
      type: "planModeChanged";
      active: boolean;
      planFilePath: string | null;
      planSource: string | null;
    }
  | { type: "requestResolved"; requestId: string; kind: "approval" | "planReview" }
  | { type: "error"; message: string });

export interface ApprovalRequest {
  /** Unix milliseconds when waiting began; optional for older backends. */
  createdAt?: number;
  sessionId: string;
  requestId: string;
  toolName: string;
  argsSummary: string;
  reason: string;
  hazardous: boolean;
}

/** A plan submitted for review by `exit_plan_mode`. */
export interface PlanReviewRequest {
  /** Unix milliseconds when waiting began; optional for older backends. */
  createdAt?: number;
  sessionId: string;
  requestId: string;
  planFilePath: string;
  planContents: string;
}

/** A tool call rendered in the transcript. */
export interface ToolCallView {
  id: string;
  name: string;
  args: Record<string, unknown>;
  status: "running" | "done" | "error";
  result?: string;
}

export interface ModelDraft {
  id: string;
  name?: string;
  contextWindow?: number;
  maxTokens?: number;
  reasoning?: boolean;
  input?: ("text" | "image")[];
  thinkingLevelMap?: Partial<Record<ThinkingLevel, string | null>>;
  // Preserve advanced declarations not edited by this form.
  [key: string]: unknown;
}
export interface ProviderView { id: string; name: string; baseUrl: string; api: string; models: ModelDraft[]; hasApiKey: boolean; configured: boolean; }
export interface ProviderDraft { id: string; name: string; baseUrl: string; api: string; models: ModelDraft[]; apiKey?: string; }
