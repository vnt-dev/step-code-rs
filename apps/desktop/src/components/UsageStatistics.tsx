import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, onRequestRecordUpdated } from "../api";
import { dayKey, formatTokens, summarizeUsage, type UsageSnapshot } from "../lib/usage-statistics";
import Icon from "./Icon";
import UsageActivity from "./UsageActivity";
import RequestAnalytics from "./RequestAnalytics";
import { modelColor, smoothPaths } from "../lib/request-metrics";


export default function UsageStatistics({ active }: { active: boolean }) {
  const [snapshot, setSnapshot] = useState<UsageSnapshot | null>(null);
  const [range, setRange] = useState(7);
  const [mode, setMode] = useState("daily");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null);
  const [hiddenModels, setHiddenModels] = useState<Set<string>>(new Set());
  const [hoveredDay, setHoveredDay] = useState<number | null>(null);
  const request = useRef(0);
  const refresh = useCallback(async () => {
    const id = ++request.current;
    setBusy(true); setError("");
    try {
      const result = await api.usageStatistics();
      if (request.current === id) { setSnapshot(result); setUpdatedAt(new Date()); }
    } catch (e) { if (request.current === id) setError(String(e)); }
    finally { if (request.current === id) setBusy(false); }
  }, []);
  useEffect(() => {
    if (active) void refresh();
    return () => { request.current++; };
  }, [active, refresh]);
  useEffect(() => {
    if (!active) return;
    let disposed = false, unlisten: (() => void) | undefined, timer: ReturnType<typeof setTimeout> | undefined;
    void onRequestRecordUpdated(() => {
      if (timer === undefined) timer = setTimeout(() => { timer = undefined; void refresh(); }, 500);
    }).then(fn => { if (disposed) fn(); else unlisten = fn; }).catch(cause => setError(String(cause)));
    return () => { disposed = true; unlisten?.(); clearTimeout(timer); };
  }, [active, refresh]);
  const now = useMemo(() => updatedAt ?? new Date(), [updatedAt]);
  const stats = useMemo(() => summarizeUsage(snapshot ?? { records: [], longestChatMs: 0 }, range, now), [snapshot, range, now]);
  const color = modelColor;
  const visibleModels = stats.models.filter(model => !hiddenModels.has(model.id));
  const chartMax = Math.max(1, ...visibleModels.flatMap(model => [...model.daily.values()]));
  const rangeTotal = stats.models.reduce((sum, model) => sum + model.tokens, 0);
  let angle = 0;
  const segments = stats.models.map(model => { const start = angle; angle += model.tokens / rangeTotal * 360; return `${color(model.id)} ${start}deg ${angle}deg`; });
  const duration = snapshot?.longestChatMs ?? 0;
  const hours = Math.floor(duration / 3600000), minutes = Math.floor(duration / 60000) % 60;
  const durationLabel = !duration ? "—" : duration < 60000 ? "不足 1 分钟" : hours ? `${hours}小时 ${minutes}分` : `${minutes}分`;
  const selectedDate = stats.dates[hoveredDay ?? range - 1];
  const selectedKey = dayKey(selectedDate);
  const toggleModel = (id: string) => setHiddenModels(previous => {
    const next = new Set(previous);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });
  return <div className="usage-statistics" hidden={!active} aria-busy={busy}>
    <header className="usage-heading"><div><h1>使用统计</h1><p>了解使用习惯，查看各模型的用量分布。</p></div></header>
    <div className="usage-meta"><span><i />所有项目 · 本地请求记录</span><span role="status">{busy ? "正在更新用量…" : updatedAt ? `更新于 ${updatedAt.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })}` : "尚未加载"}</span></div>
    {!!snapshot?.unknownTokenRequests && <p className="usage-caption">{snapshot.unknownTokenRequests.toLocaleString("zh-CN")} 次请求的 Token 尚未完整报告，当前用量仅汇总已报告的部分。</p>}
    {error && <div className="banner banner-error" role="alert"><strong>用量加载失败</strong><p>{snapshot ? "当前显示上次成功加载的数据，重新打开此页面可重试。" : "重新打开此页面可重试。"}</p><details><summary>错误详情</summary>{error}</details></div>}
    {!snapshot ? <div className="usage-loading" role="status"><Icon name="chart" /><strong>{error ? "暂时无法获取使用记录" : "正在整理使用记录"}</strong><p>{error ? "重新打开此页面可重试。" : "正在读取本地请求记录，请稍候。"}</p></div> : <>
      <div className="usage-summary">{[[formatTokens(stats.total), "累计 Token", "全部会话"], [formatTokens(stats.peak), "单日峰值", "历史最高用量"], [durationLabel, "最长会话", "累计执行时长"], [`${stats.currentStreak} 天`, "当前连续使用", `最长连续 ${stats.longestStreak} 天`]].map(([value, label, detail]) => <div key={label}><span>{label}</span><strong title={value}>{value}</strong><small>{detail}</small></div>)}</div>
      <section className="usage-card"><div className="usage-card-heading"><div><h2>使用活跃度</h2><p>最近一年 · {mode === "weekly" ? "每周 Token 用量" : mode === "cumulative" ? "截至每周的累计总用量" : "每日 Token 用量"}</p></div><div className="usage-switch" role="group" aria-label="活动统计方式">{[["daily", "每日"], ["weekly", "每周"], ["cumulative", "累计"]].map(([id, label]) => <button key={id} aria-pressed={mode === id} onClick={() => setMode(id)}>{label}</button>)}</div></div>
        <UsageActivity records={snapshot.records} now={now} mode={mode} active={active} />
      </section>
      <div className="usage-range"><div><h2>用量分析</h2><span>近 {range} 天共使用 <strong>{formatTokens(rangeTotal)}</strong> Token</span></div><div className="usage-switch" role="group" aria-label="用量分析时间范围">{[7, 30].map(days => <button key={days} aria-pressed={range === days} onClick={() => { setRange(days); setHoveredDay(null); }}>近 {days} 天</button>)}</div></div>
      <section className="usage-card"><div className="usage-card-heading"><div><h2>每日用量趋势</h2><p>点击模型图例可显示或隐藏曲线</p></div><span className="usage-unit">Token / 日</span></div>
        <div className="usage-chart-legend">{stats.models.map(model => <button key={model.id} aria-pressed={!hiddenModels.has(model.id)} title={model.id} onClick={() => toggleModel(model.id)}><i style={{ background: color(model.id) }} />{model.label}</button>)}</div>
        {!rangeTotal ? <div className="usage-empty"><Icon name="chart" /><strong>这段时间还没有用量</strong><p>开始对话后，趋势和模型分布会显示在这里。</p></div> : <>
          <svg className="usage-trend" viewBox="0 0 800 250" role="group" aria-label={`近 ${range} 天各模型每日 Token 用量`} onMouseLeave={() => setHoveredDay(null)} onBlur={event => { if (!event.currentTarget.contains(event.relatedTarget)) setHoveredDay(null); }}>
            {[0, 1, 2, 3, 4].map(i => <g key={i}><line className="usage-grid-line" x1="64" x2="780" y1={20 + i * 48} y2={20 + i * 48} /><text x="56" y={24 + i * 48} textAnchor="end">{formatTokens(chartMax * (1 - i / 4))}</text></g>)}
            {hoveredDay !== null && <line className="usage-hover-line" x1={64 + hoveredDay / (range - 1) * 716} x2={64 + hoveredDay / (range - 1) * 716} y1="20" y2="212" />}
            {visibleModels.map(model => <g key={model.id}>{smoothPaths(stats.dates.map((date, x) => ({ x: 64 + x / (range - 1) * 716, y: 212 - (model.daily.get(dayKey(date)) ?? 0) / chartMax * 192 }))).map((path, index) => <path key={index} d={path} fill="none" stroke={color(model.id)} strokeWidth="2.2" />)}{stats.dates.map((date, x) => (range === 7 || hoveredDay === x) && <circle key={dayKey(date)} cx={64 + x / (range - 1) * 716} cy={212 - (model.daily.get(dayKey(date)) ?? 0) / chartMax * 192} r={hoveredDay === x ? 4 : 2.5} fill={color(model.id)} stroke="var(--panel)" strokeWidth="1.5" />)}</g>)}
            {stats.dates.map((date, i) => <g key={i}>{(range === 7 || i % 5 === 0 || i === range - 1) && <text x={64 + i / (range - 1) * 716} y="239" textAnchor="middle">{date.getMonth() + 1}/{date.getDate()}</text>}<rect className="usage-day-target" x={64 + i / (range - 1) * 716 - 716 / (range - 1) / 2} y="12" width={716 / (range - 1)} height="205" fill="transparent" tabIndex={i === (hoveredDay ?? range - 1) ? 0 : -1} role="button" aria-label={`${dayKey(date)}，${(stats.daily.get(dayKey(date)) ?? 0).toLocaleString()} Token，方向键查看相邻日期`} onFocus={() => setHoveredDay(i)} onMouseEnter={() => setHoveredDay(i)} onKeyDown={event => {
              if (event.key === "ArrowLeft" || event.key === "ArrowRight") { event.preventDefault(); const next = Math.max(0, Math.min(range - 1, i + (event.key === "ArrowRight" ? 1 : -1))); const targets = event.currentTarget.ownerSVGElement?.querySelectorAll<SVGRectElement>(".usage-day-target"); targets?.[next]?.focus(); }
            }} /></g>)}
          </svg>
          <div className="usage-chart-detail"><strong>{selectedDate.getMonth() + 1}月{selectedDate.getDate()}日</strong>{visibleModels.length ? visibleModels.map(model => <span key={model.id}><i style={{ background: color(model.id) }} />{model.label}<b>{formatTokens(model.daily.get(selectedKey) ?? 0)}</b></span>) : <span>模型曲线已隐藏，点击图例重新显示</span>}</div>
        </>}
      </section>
      <section className="usage-card"><div className="usage-card-heading"><div><h2>模型用量分布</h2><p>所选时间范围内的 Token 占比</p></div><span className="usage-unit">{stats.models.length} 个模型</span></div><div className="usage-models"><div className="usage-donut" role="img" aria-label={`所选范围总用量 ${formatTokens(rangeTotal)} tokens`} style={{ background: rangeTotal ? `conic-gradient(${segments.join(",")})` : "var(--surface-hover)" }}><div><strong>{formatTokens(rangeTotal)}</strong><span>总 Token</span></div></div><div className="usage-model-list">{stats.models.map(model => <div key={model.id}><i style={{ background: color(model.id) }} /><span><strong title={model.id}>{model.label}</strong><small>{model.id.split("/")[0]} · {formatTokens(model.tokens)} Token</small><span className="usage-model-bar"><i style={{ width: `${model.tokens / rangeTotal * 100}%`, background: color(model.id) }} /></span></span><b>{(model.tokens / rangeTotal * 100).toFixed(1)}%</b></div>)}{!rangeTotal && <p className="usage-caption">暂无模型用量，开始对话后自动记录。</p>}</div></div></section>
      <details className="usage-notes"><summary>统计说明</summary><p>统计来自启用记录后的实际请求，包含对话、压缩、命名及重试；失败或取消请求已报告的 Token 也计入消耗。总 Token 为输入加输出，输入已包含缓存，不重复相加；未报告部分不估算。每周按周日到周六汇总，当前周统计至今；次数按实际请求尝试计数。最长会话为同一会话已结束轮次的执行时长之和，包含工具执行和等待；异常退出而未完成计时的轮次不补算。删除会话后仍保留统计，不导入历史会话。连续使用天数按本地日期和已报告的 Token 计算，今日未使用时延续昨日记录。</p></details>
    </>}
    <RequestAnalytics active={active} />
  </div>;
}
