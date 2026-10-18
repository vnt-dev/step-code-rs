import type { Bootstrap, ModelInfo } from "../types";
import { isImageAttachment, maxImages, type ImageAttachment } from "./images";
import { projectKey } from "./projects";

export interface Draft {
  text: string;
  images?: ImageAttachment[];
  skills?: import("../types").SkillSummary[];
  provider: string;
  modelId: string;
  thinkingLevel: string;
  approvalMode: string;
  /** Draft-only plan-mode flag; the real session turns it on when sent. */
  planActive: boolean;
}
export const draftKey = (path: string) => `project:${projectKey(path)}`;
export const draftStorageKey = "step-code.composer-drafts.v1";

export function supportedDraftThinkingLevel(model: ModelInfo | null | undefined, requested: string): string {
  if (!model?.reasoning) return "off";
  const levels = model.supportedThinkingLevels;
  if (levels.some(level => level === requested)) return requested;
  return levels.includes("medium") ? "medium" : levels[0] ?? "off";
}

export function defaultDraft(bootstrap: Bootstrap | null): Draft {
  const settings = bootstrap?.settings;
  const model = bootstrap?.currentModel ?? bootstrap?.models.find(model => model.configured &&
    model.id === settings?.defaultModel && (!settings.defaultProvider || model.provider === settings.defaultProvider)) ??
    bootstrap?.models.find(model => model.configured);
  const thinkingLevel = bootstrap?.currentModel ? bootstrap.thinkingLevel :
    (typeof settings?.defaultThinkingLevel === "string" ? settings.defaultThinkingLevel : "medium");
  return { text: "", provider: model?.provider ?? "", modelId: model?.id ?? "",
    thinkingLevel: supportedDraftThinkingLevel(model, thinkingLevel), approvalMode: bootstrap?.approvalMode ?? "ask",
    planActive: false };
}

export function parseDrafts(raw: string | null): Record<string, Draft> {
  try {
    const parsed: unknown = JSON.parse(raw ?? "{}");
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return {};
    return Object.fromEntries(Object.entries(parsed).filter(([key, draft]) =>
      key.startsWith("project:") && draft && typeof draft === "object" &&
      ["text", "provider", "modelId", "thinkingLevel", "approvalMode"].every(field => typeof draft[field] === "string")).map(([key, value]) => {
        const draft = value as Draft;
        return [key, { ...draft, ...(draft.skills !== undefined ? { skills: Array.isArray(draft.skills) ? draft.skills.filter(skill => skill && typeof skill.name === "string" && typeof skill.description === "string" && typeof skill.filePath === "string") : [] } : {}), approvalMode: ["ask", "bypass", "autopilot"].includes(draft.approvalMode) ? draft.approvalMode : "ask", planActive: draft.planActive === true,
          ...(draft.images !== undefined ? { images: Array.isArray(draft.images) ? draft.images.filter(isImageAttachment).slice(0, maxImages) : [] } : {}) }];
      }));
  } catch { return {}; }
}
