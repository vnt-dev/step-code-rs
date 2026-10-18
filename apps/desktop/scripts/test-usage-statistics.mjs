import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';
import { test } from 'node:test';
import ts from 'typescript';
const source = ts.transpileModule(readFileSync(new URL('../src/lib/usage-statistics.ts', import.meta.url), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 } }).outputText;
const { summarizeUsage, dayKey, daysBefore, usageActivity, formatTokens } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const now = new Date(2026, 9, 14, 12);
const record = (day, tokens, provider = 'step', model = 'shared') => ({ timestamp: new Date(2026, 9, day, 12).getTime(), tokens, provider, model });
test('all-time totals, daily peak and range model totals use separate scopes', () => {
  const stats = summarizeUsage({ records: [record(1, 100), record(13, 20), record(13, 30, 'other'), record(14, 40)], longestChatMs: 0 }, 7, now);
  assert.equal(stats.total, 190); assert.equal(stats.peak, 100);
  assert.equal(stats.models.length, 2); assert.equal(stats.models.reduce((sum, item) => sum + item.tokens, 0), 90);
  assert.equal(stats.dates.length, 7); assert.equal(dayKey(stats.dates[0]), '2026-10-08');
});
test('streaks span month boundaries and stay current through yesterday', () => {
  const stats = summarizeUsage({ records: [record(11, 1), record(12, 1), record(13, 1)], longestChatMs: 0 }, 30, now);
  assert.equal(stats.currentStreak, 3); assert.equal(stats.longestStreak, 3);
  assert.equal(summarizeUsage({ records: [record(11, 1)], longestChatMs: 0 }, 7, now).currentStreak, 0);
  assert.equal(dayKey(daysBefore(new Date(2026, 10, 1, 12), 1)), '2026-10-31');
});
test('empty data stays zero and future records are excluded from the selected range', () => {
  const empty = summarizeUsage({ records: [], longestChatMs: 0 }, 7, now);
  assert.equal(empty.peak, 0); assert.equal(empty.currentStreak, 0); assert.equal(empty.total, 0);
  assert.equal(summarizeUsage({ records: [record(15, 1)], longestChatMs: 0 }, 7, now).models.length, 0);
});

test('weekly activity groups calendar weeks and cumulative includes older history', () => {
  const older = { timestamp: new Date(2024, 0, 1, 12).getTime(), tokens: 1000, provider: 'step', model: 'test' };
  const data = usageActivity([older, record(3, 10), record(4, 20), record(10, 30), record(11, 40), record(14, 0), record(15, 999)], now);
  const week = key => data.weeks.find(bucket => dayKey(bucket.date) === key);
  const cumulative = key => data.cumulative.find(bucket => dayKey(bucket.date) === key);
  assert.equal(week('2026-10-03').tokens, 10);
  assert.equal(week('2026-10-10').tokens, 50);
  assert.equal(week('2026-10-10').messages, 2);
  assert.equal(week('2026-10-17').tokens, 40);
  assert.equal(week('2026-10-17').messages, 2);
  assert.equal(cumulative('2026-10-10').tokens, 1060);
  assert.equal(cumulative('2026-10-17').tokens, 1100);
  assert.equal(cumulative('2026-10-17').messages, 6);
  assert.equal(data.days.find(bucket => dayKey(bucket.date) === '2026-10-14').messages, 1);
});
test('activity units use compact Chinese token units and zero-token replies have no model share', () => {
  assert.equal(formatTokens(540000000), '5.4亿');
  assert.equal(formatTokens(12000), '1.2万');
  assert.equal(formatTokens(57), '57');
  assert.equal(summarizeUsage({ records: [record(14, 0)], longestChatMs: 0 }, 7, now).models.length, 0);
});
