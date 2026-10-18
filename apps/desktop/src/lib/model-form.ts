import type { ModelDraft, ThinkingLevel } from "../types";

export const thinkingLevels: ThinkingLevel[] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];

/** Decimal SI units: 128k = 128000, 1.5m = 1500000. */
export function parseTokenCount(value: string): number {
  const match = /^(\d+(?:\.\d+)?|\.\d+)\s*([km]?)$/i.exec(value.trim());
  if (!match) throw new Error("请输入正整数或带 k、m 的数值，例如 128k、1.5m。");
  const [whole, fraction = ""] = match[1].split(".");
  const scaled = BigInt((whole || "0") + fraction) * ({ k: 1000n, m: 1000000n }[match[2].toLowerCase()] ?? 1n);
  const divisor = 10n ** BigInt(fraction.length);
  const count = scaled / divisor;
  if (scaled % divisor !== 0n || count <= 0n || count > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error("转换后的 Token 数必须是安全范围内的正整数。");
  return Number(count);
}

export function modelThinkingLevels(model: ModelDraft): ThinkingLevel[] {
  if (!model.reasoning) return [];
  return thinkingLevels.filter(level => model.thinkingLevelMap?.[level] !== null &&
    (!["xhigh", "max"].includes(level) || typeof model.thinkingLevelMap?.[level] === "string"));
}

export function parseThinkingLevels(values: string[]): ThinkingLevel[] {
  const levels = values.map(value => value.trim()).filter(Boolean);
  if (levels.some(level => !thinkingLevels.includes(level as ThinkingLevel))) {
    throw new Error("推理等级支持 off、minimal、low、medium、high、xhigh、max。");
  }
  if (new Set(levels).size !== levels.length) throw new Error("推理等级不能重复。");
  return levels as ThinkingLevel[];
}

export function parseThinkingMap(raw: string, selected: ThinkingLevel[]): ModelDraft["thinkingLevelMap"] {
  const parsed: unknown = raw.trim() ? JSON.parse(raw) : {};
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("推理参数映射必须是 JSON 对象。");
  for (const [key, value] of Object.entries(parsed)) {
    if (!thinkingLevels.includes(key as ThinkingLevel) || (value !== null && (typeof value !== "string" || !value.trim()))) {
      throw new Error("映射键必须是有效推理等级，值必须是非空字符串或 null。");
    }
  }
  const values = parsed as Record<string, string | null>;
  return Object.fromEntries(thinkingLevels.map(level => [level, selected.includes(level)
    ? values[level] ?? (level === "off" ? "none" : level) : null]));
}
