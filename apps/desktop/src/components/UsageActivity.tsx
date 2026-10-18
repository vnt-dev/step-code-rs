import { useEffect, useId, useMemo, useState, type CSSProperties } from "react";
import { createPortal } from "react-dom";
import { dayKey, formatTokens, usageActivity, type UsageRecord } from "../lib/usage-statistics";

const heatColor = (level: number) => level ? `color-mix(in srgb, var(--accent) ${level}%, var(--surface-hover))` : "var(--surface-hover)";
export default function UsageActivity({ records, now, mode, active }: { records: UsageRecord[]; now: Date; mode: string; active: boolean }) {
  const activity = useMemo(() => usageActivity(records, now), [records, now]);
  const buckets = mode === "daily" ? activity.days : mode === "weekly" ? activity.weeks : activity.cumulative;
  const maximum = Math.max(1, ...buckets.map(bucket => bucket.tokens));
  const months = activity.days.flatMap((day, i) => day.date.getDate() === 1 ? [{ label: `${day.date.getMonth() + 1}月`, column: Math.floor(i / 7) }] : []);
  const [tooltip, setTooltip] = useState<{ index: number; mode: string; style: CSSProperties } | null>(null);
  const [focused, setFocused] = useState<number | null>(null);
  const tooltipId = useId();
  useEffect(() => { if (!active) setTooltip(null); }, [active]);
  useEffect(() => {
    const dismiss = () => setTooltip(null);
    window.addEventListener("scroll", dismiss, true);
    window.addEventListener("resize", dismiss);
    return () => { window.removeEventListener("scroll", dismiss, true); window.removeEventListener("resize", dismiss); };
  }, []);
  const selected = tooltip?.mode === mode ? buckets[tooltip.index] : null;
  const description = (index: number) => {
    const bucket = buckets[index];
    const date = `${bucket.date.getFullYear()}年${bucket.date.getMonth() + 1}月${bucket.date.getDate()}日`;
    return { date: mode === "daily" ? date : mode === "weekly" ? `${date} 当周` : `截至 ${date} 当周累计`, value: `${formatTokens(bucket.tokens)} tokens · ${bucket.messages.toLocaleString("zh-CN")} 次请求` };
  };
  const show = (index: number, element: HTMLElement) => {
    const rect = element.getBoundingClientRect();
    const left = Math.max(8, Math.min(window.innerWidth - 248, rect.left + rect.width / 2 - 120));
    const above = rect.top >= 72;
    setTooltip({ index, mode, style: { left, top: above ? rect.top - 8 : rect.bottom + 8, transform: above ? "translateY(-100%)" : undefined } });
  };
  const lastDay = activity.days.findIndex(day => dayKey(day.date) === dayKey(now));
  const focusIndex = focused !== null && focused < buckets.length ? focused : mode === "daily" ? lastDay : buckets.length - 1;
  return <>
    <div className="usage-heat-scroll" onScroll={() => setTooltip(null)}>
      <div className="usage-months">{months.map((month, i) => <span key={i} style={{ gridColumn: month.column + 1 }}>{month.label}</span>)}</div>
      <div className={mode === "daily" ? "usage-heat-grid" : "usage-week-grid"} onMouseLeave={() => setTooltip(null)}>
        {buckets.map((bucket, index) => {
          const future = mode === "daily" && bucket.date > now;
          const label = description(index);
          const height = bucket.tokens ? Math.max(1, Math.ceil(bucket.tokens / maximum * 7)) : 0;
          return <button key={`${mode}-${dayKey(bucket.date)}`} type="button" className={mode === "daily" ? `usage-heat-cell${future ? " future" : ""}` : "usage-week-column"}
            style={mode === "daily" ? { background: heatColor(bucket.tokens ? 20 + bucket.tokens / maximum * 80 : 0) } : undefined}
            disabled={future} tabIndex={!future && index === focusIndex ? 0 : -1} aria-label={`${label.date}，${label.value}`} aria-describedby={tooltip?.mode === mode && tooltip.index === index ? tooltipId : undefined}
            onMouseEnter={event => show(index, event.currentTarget)} onFocus={event => { setFocused(index); show(index, event.currentTarget); }} onBlur={() => setTooltip(null)} onClick={event => show(index, event.currentTarget)}
            onKeyDown={event => {
              if (event.key === "Escape") { setTooltip(null); return; }
              const delta = event.key === "ArrowLeft" ? mode === "daily" ? -7 : -1 : event.key === "ArrowRight" ? mode === "daily" ? 7 : 1 : event.key === "ArrowUp" ? -1 : event.key === "ArrowDown" ? 1 : 0;
              if (!delta) return;
              event.preventDefault();
              const next = Math.max(0, Math.min(mode === "daily" ? lastDay : buckets.length - 1, index + delta));
              event.currentTarget.parentElement?.querySelectorAll<HTMLButtonElement>("button")[next]?.focus();
            }}>
            {mode !== "daily" && Array.from({ length: 7 }, (_, row) => <i key={row} className={row >= 7 - height ? "filled" : ""} style={{ background: row >= 7 - height ? heatColor(75) : undefined }} />)}
          </button>;
        })}
      </div>
    </div>
    <div className="usage-heat-footer"><span>悬停或用方向键查看{mode === "daily" ? "每日" : "每周"}用量</span><div className="usage-heat-legend">{mode === "daily" ? <><span>少</span>{[0, 25, 50, 75, 100].map(n => <i key={n} style={{ background: heatColor(n) }} />)}<span>多</span></> : <span>柱高表示{mode === "weekly" ? "当周用量" : "截至该周累计用量"}</span>}</div></div>
    {selected && tooltip && createPortal(<div id={tooltipId} role="tooltip" className="usage-activity-tooltip" style={tooltip.style}><strong>{description(tooltip.index).date}</strong><span>{description(tooltip.index).value}</span></div>, document.body)}
  </>;
}
