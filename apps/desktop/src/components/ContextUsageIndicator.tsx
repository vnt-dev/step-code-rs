import { useEffect, useId, useRef, useState, type CSSProperties } from "react";
import { createPortal } from "react-dom";
import type { AgentMessage, ExecutionTiming, SessionStats, Usage } from "../types";
import { cacheHitRate, contextUsage, inputTokens } from "../lib/context-usage";

const tokens = (value: number) => value.toLocaleString("zh-CN");

function UsageDetails({ label, usage, tps, samples, error }: { label: string; usage: Usage; tps?: number | null; samples?: number; error?: string | null }) {
  return <section className="context-usage-section">
    <h4>{label}</h4>
    <dl>
      <div><dt>输入 token</dt><dd>{tokens(inputTokens(usage))}</dd></div>
      <div><dt>输出 token</dt><dd>{tokens(usage.output)}</dd></div>
      <div><dt>缓存命中率</dt><dd>{cacheHitRate(usage).toFixed(2)}%</dd></div>
      <div title={error ? "请求记录不可用" : `按有效输出 Token / 生成时间计算 · ${samples ?? 0} 个有效请求`}><dt>平均输出速度</dt><dd>{tps == null ? "—" : `${tps.toFixed(1)} tps`}</dd></div>
    </dl>
  </section>;
}

export default function ContextUsageIndicator({ messages, stats, running, executionTimings, maxContext }: {
  messages: AgentMessage[]; stats: SessionStats | null; running: boolean;
  executionTimings: ExecutionTiming[]; maxContext: number;
}) {
  const tooltipId = useId();
  const trigger = useRef<HTMLButtonElement>(null);
  const tooltip = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [position, setPosition] = useState<CSSProperties>({});
  const tooltipAnchor = (rect: DOMRect): CSSProperties => ({
    left: Math.max(16, Math.min(rect.left, window.innerWidth - 286)), right: "auto",
    ...(rect.top >= 320 ? { bottom: window.innerHeight - rect.top + 8, top: "auto" } : { top: rect.bottom + 8, bottom: "auto" }),
  });
  useEffect(() => {
    if (!open) return;
    // Streaming shifts the toolbar (tps readout, send/stop swap) and keeps the
    // transcript scrolling, but none of that moves the pointer: closing is tied
    // to real pointer movement only, while scrolls and re-renders re-anchor.
    const outside = (x: number, y: number) => {
      const inside = (rect: DOMRect | undefined) => rect && x >= rect.left - 10 && x <= rect.right + 10 && y >= rect.top - 10 && y <= rect.bottom + 10;
      return !inside(trigger.current?.getBoundingClientRect()) && !inside(tooltip.current?.getBoundingClientRect());
    };
    const onPointerMove = (event: MouseEvent) => { if (outside(event.clientX, event.clientY)) setOpen(false); };
    let frame = 0;
    const reposition = () => {
      frame = 0;
      const rect = trigger.current?.getBoundingClientRect();
      if (!rect || (!rect.top && !rect.left && !rect.width)) { setOpen(false); return; }
      setPosition(tooltipAnchor(rect));
    };
    const schedule = () => { if (!frame) frame = requestAnimationFrame(reposition); };
    window.addEventListener("mousemove", onPointerMove);
    window.addEventListener("scroll", schedule, true);
    window.addEventListener("resize", schedule);
    return () => {
      window.removeEventListener("mousemove", onPointerMove);
      window.removeEventListener("scroll", schedule, true);
      window.removeEventListener("resize", schedule);
      if (frame) cancelAnimationFrame(frame);
    };
  }, [open]);
  // Re-anchor after every render too, so per-turn layout changes stay glued.
  useEffect(() => {
    if (!open) return;
    const rect = trigger.current?.getBoundingClientRect();
    if (!rect || (!rect.top && !rect.left && !rect.width)) { setOpen(false); return; }
    const next = tooltipAnchor(rect);
    setPosition(current => (current.top === next.top && current.bottom === next.bottom && current.left === next.left && current.right === next.right ? current : next));
  });
  const positionTooltip = () => {
    const rect = trigger.current?.getBoundingClientRect();
    if (rect) { setPosition(tooltipAnchor(rect)); setOpen(true); }
  };
  const { contextTokens, lastRound, total, rounds } = contextUsage(messages, stats, running, executionTimings);
  const percent = maxContext > 0 ? contextTokens / maxContext * 100 : 0;
  const available = maxContext > 0;
  const percentage = available ? `${percent.toFixed(2)}%` : "—";
  return <div className="context-usage" onMouseEnter={positionTooltip} onFocus={positionTooltip} onBlur={() => setOpen(false)}>
    <button ref={trigger} type="button" className={`context-usage-trigger${percent >= 90 ? " is-high" : ""}`} aria-label={`上下文占用 ${percentage}`} aria-describedby={open ? tooltipId : undefined} onKeyDown={event => { if (event.key === "Escape") setOpen(false); }}>
      <svg width="18" height="18" viewBox="0 0 20 20" aria-hidden="true">
        <circle className="context-usage-track" cx="10" cy="10" r="7" />
        <circle className="context-usage-fill" cx="10" cy="10" r="7" pathLength="100" strokeDasharray={`${Math.min(100, Math.max(0, percent))} 100`} transform="rotate(-90 10 10)" />
      </svg>
    </button>
    {open && createPortal(<div ref={tooltip} className="context-usage-tooltip is-open" style={position} id={tooltipId} role="tooltip">
      <section className="context-usage-section">
        <h4>上下文详情</h4>
        <dl>
          <div><dt>当前上下文</dt><dd>{tokens(contextTokens)}</dd></div>
          <div><dt>最大上下文</dt><dd>{available ? tokens(maxContext) : "—"}</dd></div>
          <div><dt>占用比例</dt><dd>{percentage}</dd></div>
        </dl>
      </section>
      <UsageDetails label="最后一轮对话" usage={lastRound} tps={stats?.performance?.lastRoundTps} samples={stats?.performance?.lastRoundSamples} error={stats?.performanceError} />
      {/* A single round makes the cumulative row repeat the one above it. */}
      {rounds > 1 && <UsageDetails label="会话累计" usage={total} tps={stats?.performance?.sessionTps} samples={stats?.performance?.sessionSamples} error={stats?.performanceError} />}
    </div>, document.body)}
  </div>;
}
