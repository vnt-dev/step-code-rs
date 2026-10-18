export interface UsageRecord { timestamp: number; provider: string; model: string; tokens: number; }
export interface UsageSnapshot { records: UsageRecord[]; longestChatMs: number; unknownTokenRequests?: number; }
export interface ActivityBucket { date: Date; tokens: number; messages: number; }
export function usageActivity(records: UsageRecord[], now = new Date()) {
  // Columns run Sunday through Saturday, matching the daily activity grid.
  const start = daysBefore(now, 364 + now.getDay());
  const end = daysBefore(now, -1).getTime();
  const daily = new Map<string, { tokens: number; messages: number }>();
  let totalTokens = 0, totalMessages = 0;
  for (const record of records) {
    if (record.timestamp >= end) continue;
    if (record.timestamp < start.getTime()) { totalTokens += record.tokens; totalMessages++; continue; }
    const key = dayKey(new Date(record.timestamp));
    const value = daily.get(key) ?? { tokens: 0, messages: 0 };
    value.tokens += record.tokens; value.messages++; daily.set(key, value);
  }
  const days: ActivityBucket[] = Array.from({ length: 371 }, (_, i) => {
    const date = daysBefore(start, -i);
    return { date, ...(daily.get(dayKey(date)) ?? { tokens: 0, messages: 0 }) };
  });
  const weeks: ActivityBucket[] = [], cumulative: ActivityBucket[] = [];
  for (let column = 0; column < 53; column++) {
    const slice = days.slice(column * 7, column * 7 + 7);
    const tokens = slice.reduce((sum, day) => sum + day.tokens, 0);
    const messages = slice.reduce((sum, day) => sum + day.messages, 0);
    const date = slice[6].date;
    weeks.push({ date, tokens, messages });
    totalTokens += tokens; totalMessages += messages;
    cumulative.push({ date, tokens: totalTokens, messages: totalMessages });
  }
  return { days, weeks, cumulative };
}
export function dayKey(date: Date): string {
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
}
export function daysBefore(date: Date, count: number): Date {
  const result = new Date(date); result.setHours(0, 0, 0, 0); result.setDate(result.getDate() - count); return result;
}
export function summarizeUsage(snapshot: UsageSnapshot, range: number, now = new Date()) {
  const daily = new Map<string, number>();
  const start = daysBefore(now, range - 1).getTime(), end = daysBefore(now, -1).getTime();
  const models = new Map<string, { id: string; label: string; tokens: number; daily: Map<string, number> }>();
  for (const record of snapshot.records) {
    const key = dayKey(new Date(record.timestamp));
    daily.set(key, (daily.get(key) ?? 0) + record.tokens);
    if (record.tokens <= 0 || record.timestamp < start || record.timestamp >= end) continue;
    const id = `${record.provider}/${record.model}`;
    const model = models.get(id) ?? { id, label: record.model, tokens: 0, daily: new Map<string, number>() };
    model.tokens += record.tokens; model.daily.set(key, (model.daily.get(key) ?? 0) + record.tokens); models.set(id, model);
  }
  const active = [...daily.keys()].filter(key => (daily.get(key) ?? 0) > 0).sort();
  let longestStreak = 0, streak = 0, previous = "";
  for (const key of active) {
    const yesterday = dayKey(daysBefore(new Date(`${key}T12:00:00`), 1));
    streak = previous === yesterday ? streak + 1 : 1; longestStreak = Math.max(longestStreak, streak); previous = key;
  }
  let currentStreak = 0;
  const offset = daily.get(dayKey(now)) ? 0 : 1;
  while (daily.get(dayKey(daysBefore(now, currentStreak + offset)))) currentStreak++;
  const dates = Array.from({ length: range }, (_, i) => daysBefore(now, range - 1 - i));
  return { daily, dates, models: [...models.values()].sort((a, b) => b.tokens - a.tokens), total: [...daily.values()].reduce((a, b) => a + b, 0), peak: Math.max(0, ...daily.values()), currentStreak, longestStreak };
}
export function formatTokens(value: number): string {
  if (value >= 1e8) return `${(value / 1e8).toLocaleString("zh-CN", { maximumFractionDigits: 1 })}亿`;
  if (value >= 1e4) return `${(value / 1e4).toLocaleString("zh-CN", { maximumFractionDigits: 1 })}万`;
  return value.toLocaleString("zh-CN");
}
