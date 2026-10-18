//! Typed wrappers around the Tauri command surface.

import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type {
  ProviderView,
  ProviderDraft,
  ApprovalRequest,
  Bootstrap,
  PlanReviewRequest,
  SessionEvent,
  SessionStats,
  SessionSummary,
  AgentMessage,
  ThinkingLevel,
  ImageBlock,
} from "./types";

export const api = {
  openExternalUrl: (url: string) => invoke<void>("open_external_url", { url }),
  skills: (cwd: string) => invoke<import("./types").SkillSummary[]>("list_skills", { cwd }),
  modelHealth: (dayBoundaries: number[], perCall = false) => invoke<import("./lib/request-metrics").HealthPoint[]>("get_model_health", { dayBoundaries, perCall }),
  requestRecords: (filter: import("./lib/request-metrics").RequestFilter) => invoke<import("./lib/request-metrics").RequestPage>("list_request_records", { filter }),
  usageStatistics: () => invoke<import("./lib/usage-statistics").UsageSnapshot>("get_usage_statistics"),
  providers: () => invoke<ProviderView[]>("get_providers"),
  saveProvider: (draft: ProviderDraft) => invoke<void>("save_provider", { ...draft, apiKey: draft.apiKey ?? null }),
  deleteProvider: (id: string) => invoke<void>("delete_provider", { id }),
  loginStep: () => invoke<void>("login_step"),
  cancelStepLogin: () => invoke<void>("cancel_step_login"),
  reopenStepLogin: () => invoke<void>("reopen_step_login"),
  logoutStep: () => invoke<void>("logout_step"),
  bootstrap: (sessionId?: string) => invoke<Bootstrap>("get_bootstrap", { sessionId: sessionId ?? null }),
  transcript: (sessionId?: string) => invoke<AgentMessage[]>("get_transcript", { sessionId: sessionId ?? null }),
  recallLastUserMessage: (sessionId: string, timestamp: number) => invoke<void>("recall_last_user_message", { sessionId, timestamp }),
  stats: (sessionId?: string) => invoke<SessionStats>("get_stats", { sessionId: sessionId ?? null }),

  newSession: (cwd?: string) => invoke<SessionSummary>("new_session", { cwd: cwd ?? null }),
  addProject: () => invoke<string | null>("add_project"),
  removeProject: (path: string) => invoke<void>("remove_project", { path }),
  openSession: (sessionId: string) => invoke<SessionSummary>("open_session", { sessionId }),
  deleteSession: (sessionId: string) => invoke<void>("delete_session", { sessionId }),
  listSessions: () => invoke<SessionSummary[]>("list_sessions"),
  renameSession: (sessionId: string, name: string) => invoke<void>("rename_session", { sessionId, name }),
  setSessionProject: (sessionId: string, project: string) =>
    invoke<void>("set_session_project", { sessionId, project }),

  prompt: (text: string, images: ImageBlock[] = [], sessionId?: string, skills: string[] = []) => invoke<void>("prompt", { text, images, skills, sessionId: sessionId ?? null }),
  abort: (sessionId?: string) => invoke<void>("abort", { sessionId: sessionId ?? null }),
  steer: (text: string, images: ImageBlock[] = [], sessionId?: string, skills: string[] = []) => invoke<void>("steer", { text, images, skills, sessionId: sessionId ?? null }),
  followUp: (text: string, images: ImageBlock[] = [], sessionId?: string, skills: string[] = []) => invoke<void>("follow_up", { text, images, skills, sessionId: sessionId ?? null }),

  setModel: (provider: string, modelId: string, sessionId?: string) =>
    invoke<void>("set_model", { provider, modelId, sessionId: sessionId ?? null }),
  setThinkingLevel: (level: ThinkingLevel, sessionId?: string) => invoke<void>("set_thinking_level", { level, sessionId: sessionId ?? null }),
  setApprovalMode: (mode: string, sessionId?: string) => invoke<void>("set_approval_mode", { mode, sessionId: sessionId ?? null }),
  respondApproval: (requestId: string, approved: boolean) =>
    invoke<void>("respond_approval", { requestId, approved }),
  enterPlanMode: (sessionId?: string) => invoke<string>("enter_plan_mode", { sessionId: sessionId ?? null }),
  exitPlanMode: (sessionId?: string) => invoke<void>("exit_plan_mode", { sessionId: sessionId ?? null }),
  togglePlanMode: (sessionId?: string) => invoke<boolean>("toggle_plan_mode", { sessionId: sessionId ?? null }),
  respondPlanReview: (requestId: string, outcome: "approved" | "feedback" | "dismissed", feedback?: string) =>
    invoke<void>("respond_plan_review", { requestId, outcome, feedback: feedback ?? null }),
  compact: (sessionId?: string) => invoke<void>("compact", { sessionId: sessionId ?? null }),
  updateSettings: (patch: Record<string, unknown>, sessionId?: string) => invoke<void>("update_settings", { patch, sessionId: sessionId ?? null }),
  refreshModels: (allowNetwork: boolean) => invoke<void>("refresh_models", { allowNetwork }),
  reload: () => invoke<void>("reload"),
};

/** Subscribe to forwarded session events. */
export function onSessionEvent(handler: (event: SessionEvent) => void): Promise<() => void> {
  return listen<SessionEvent>("session-event", (event) => handler(event.payload));
}

/** Subscribe to tool-call confirmation requests. */
export function onApprovalRequest(
  handler: (request: ApprovalRequest) => void,
): Promise<() => void> {
  return listen<ApprovalRequest>("approval-request", (event) => handler(event.payload));
}

/** Subscribe to plan review requests. */
export function onPlanReviewRequest(
  handler: (request: PlanReviewRequest) => void,
): Promise<() => void> {
  return listen<PlanReviewRequest>("plan-review-request", (event) => handler(event.payload));
}

/** Subscribe to the bridge-ready signal emitted after the first session opens. */
export function onBridgeReady(handler: () => void): Promise<() => void> {
  return listen("bridge-ready", () => handler());
}

export function onStepLoginEvent(handler: (event: { type: string; message?: string; url?: string }) => void): Promise<() => void> { return listen("step-login-event", (event) => handler(event.payload as { type: string; message?: string; url?: string })); }
export function onRequestRecordUpdated(handler: (event: { sessionId: string }) => void): Promise<() => void> { return listen<{ sessionId: string }>("request-record-updated", event => handler(event.payload)); }
