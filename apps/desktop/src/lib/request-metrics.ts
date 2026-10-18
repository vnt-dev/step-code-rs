export interface HealthPoint { day: number; provider: string; model: string; tps: number | null; firstTokenMs: number | null; tpsSamples: number; firstTokenSamples: number; timestamp?: number | null; requestId?: string | null; status?: string | null; }
export type HealthMode = 7 | 30 | "calls";
export interface RequestCursor { startedAt: number; id: string; }
export interface RequestFilter { fromMs?: number; toMs?: number; model?: string; status?: string; cursor?: RequestCursor; }
export interface RequestRecord {
  id: string; logicalId: string; attempt: number;
  context: { sessionId: string; executionId: string | null; purpose: string };
  provider: string; model: string; responseModel: string | null; startedAt: number; endedAt: number | null;
  inputTokens: number | null; cacheReadTokens: number | null; cacheWriteTokens: number | null; outputTokens: number | null;
  firstTokenMs: number | null; generationMs: number | null; tps: number | null; status: string; httpStatus: number | null; errorKind: string | null;
  sessionName: string | null; sessionDeleted: boolean;
}
export interface RequestPage { records: RequestRecord[]; nextCursor: RequestCursor | null; models: string[]; storageError: string | null; }
const palette = ["#65aff5", "#88d8aa", "#ba9aeb", "#e8b971", "#e59bb3", "#72cbd4"];
export function modelColor(id: string): string {
  let hash = 0;
  for (const char of id) hash = (hash * 31 + char.charCodeAt(0)) | 0;
  return palette[(hash >>> 0) % palette.length];
}
export function localDayBoundaries(now = new Date(), days = 7): number[] {
  return Array.from({ length: days + 1 }, (_, i) => { const date = new Date(now); date.setHours(0, 0, 0, 0); date.setDate(date.getDate() - days + 1 + i); return date.getTime(); });
}
export interface ChartPoint { x: number; y: number; }
/** Monotone cubic controls stay inside each interval; null samples break paths. */
export function smoothPaths(points: (ChartPoint | null)[]): string[] {
  const runs: ChartPoint[][] = []; let run: ChartPoint[] = [];
  for (const point of [...points, null]) { if (point) run.push(point); else if (run.length) { runs.push(run); run = []; } }
  return runs.map(items => {
    const slopes = items.slice(1).map((point, i) => (point.y - items[i].y) / (point.x - items[i].x));
    const tangents = items.map((_, i) => {
      if (i === 0) return slopes[0] ?? 0;
      if (i === items.length - 1) return slopes[i - 1];
      const a = slopes[i - 1], b = slopes[i];
      if (a * b <= 0) return 0;
      return Math.sign(a) * Math.min(Math.abs((a + b) / 2), 3 * Math.min(Math.abs(a), Math.abs(b)));
    });
    let path = `M ${items[0].x} ${items[0].y}`;
    for (let i = 1; i < items.length; i++) {
      const a = items[i - 1], b = items[i], dx = (b.x - a.x) / 3;
      path += ` C ${a.x + dx} ${a.y + tangents[i - 1] * dx}, ${b.x - dx} ${b.y - tangents[i] * dx}, ${b.x} ${b.y}`;
    }
    return path;
  });
}
