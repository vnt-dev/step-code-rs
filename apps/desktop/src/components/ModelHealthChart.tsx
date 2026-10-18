import { useEffect, useMemo, useState } from "react";
import { modelColor, smoothPaths, type HealthPoint } from "../lib/request-metrics";

const format = (value: number | null | undefined, digits: number) => value == null ? "—" : value.toLocaleString("zh-CN", { maximumFractionDigits: digits });
const statuses: Record<string, string> = { running: "请求中", success: "成功", failed: "失败", cancelled: "已取消", interrupted: "已中断" };
type Selection = { index: number; left: number; top: number };

export default function ModelHealthChart({ points, dates, models, hidden, metric, perCall }: {
  points: HealthPoint[]; dates: Date[]; models: string[]; hidden: Set<string>;
  metric: "tps" | "firstTokenMs"; perCall: boolean;
}) {
  const [selection, setSelection] = useState<Selection | null>(null);
  const [focusIndex, setFocusIndex] = useState<number | null>(null);
  useEffect(() => { setSelection(null); setFocusIndex(null); }, [perCall, dates.length]);
  const unit = metric === "tps" ? "tokens/s" : "秒";
  const digits = metric === "tps" ? 1 : 3;
  const value = (point: HealthPoint) => metric === "tps" ? point.tps : point.firstTokenMs == null ? null : point.firstTokenMs / 1000;
  const visible = models.filter(id => !hidden.has(id));
  const visiblePoints = useMemo(() => points.filter(point => !hidden.has(`${point.provider}/${point.model}`)), [points, hidden]);
  const series = useMemo(() => {
    const grouped = new Map<string, Map<number, HealthPoint>>();
    for (const point of points) {
      const id = `${point.provider}/${point.model}`;
      if (!grouped.has(id)) grouped.set(id, new Map());
      grouped.get(id)!.set(point.day, point);
    }
    return grouped;
  }, [points]);
  // Call slots preserve every attempt (including identical timestamps), in time order.
  const slots = perCall ? visiblePoints.map(point => point.day) : dates.map((_, index) => index);
  const count = slots.length;
  const maximum = (visiblePoints.reduce((max, point) => Math.max(max, value(point) ?? 0), 0) || 1) * 1.1;
  const x = (index: number) => count <= 1 ? 422 : 64 + index / (count - 1) * 716;
  const positions = new Map(slots.map((slot, index) => [slot, index]));
  const index = Math.min(Math.max(0, selection?.index ?? focusIndex ?? count - 1), count - 1);
  const selected = perCall ? visiblePoints[index] ? [visiblePoints[index]] : [] : visible.map(id => series.get(id)?.get(index)).filter((point): point is HealthPoint => !!point);
  const timeLabel = (point?: HealthPoint, dayIndex = index) => {
    const date = perCall ? point?.timestamp == null ? null : new Date(point.timestamp) : dates[dayIndex];
    return date ? date.toLocaleString("zh-CN", perCall ? { year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit", fractionalSecondDigits: 3, hour12: false } : { year: "numeric", month: "2-digit", day: "2-digit" }) : "—";
  };
  const usable = visiblePoints.some(point => value(point) != null);
  const details = <><strong>{timeLabel(selected[0])}</strong>{perCall && count > 0 && <small>第 {index + 1} / {count} 次调用 · {statuses[selected[0]?.status ?? ""] ?? "—"}</small>}{selected.map(point => <span key={`${point.provider}/${point.model}`}><i style={{ background: modelColor(`${point.provider}/${point.model}`) }} />{point.provider}/{point.model}<b>{format(value(point), digits)} {unit}</b>{!perCall && <small>{metric === "tps" ? point.tpsSamples : point.firstTokenSamples} 个样本</small>}</span>)}{!selected.length && <span>暂无有效样本</span>}</>;
  return <section className="usage-card health-chart">
    <div className="usage-card-heading"><h2>{metric === "tps" ? "Token 输出速度" : "首字 Token 耗时"}</h2><span className="usage-unit">{unit} · {metric === "tps" ? "越高越好" : "越低越好"}</span></div>
    {!count || !visible.length ? <div className="usage-empty"><strong>{models.length ? "所选模型暂无调用记录" : "还没有请求性能记录"}</strong><p>完成新请求后自动显示。</p></div> : <>
      {!usable && <p className="usage-caption">所选记录暂无有效数值，悬停可查看调用信息。</p>}
      <div className="health-plot-scroll"><div className="health-plot">
        <svg className="usage-trend" viewBox="0 0 800 250" role="group" tabIndex={0}
          aria-label={`${perCall ? "最近500次调用" : `近${dates.length}天`}${metric === "tps" ? "输出速度" : "首字耗时"}，方向键查看数据点`}
          onPointerMove={event => {
            const rect = event.currentTarget.getBoundingClientRect();
            const relativeX = (event.clientX - rect.left) / rect.width * 800;
            const next = count <= 1 ? 0 : Math.round(Math.max(0, Math.min(1, (relativeX - 64) / 716)) * (count - 1));
            setSelection({ index: next, left: Math.min(Math.max(8, event.clientX - rect.left + 14), Math.max(8, rect.width - 280)), top: Math.max(8, Math.min(event.clientY - rect.top + 12, rect.height - 135)) });
          }} onPointerLeave={() => setSelection(null)} onBlur={() => { setSelection(null); setFocusIndex(null); }}
          onKeyDown={event => {
            if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
              event.preventDefault();
              const next = event.key === "Home" ? 0 : event.key === "End" ? count - 1 : Math.max(0, Math.min(count - 1, index + (event.key === "ArrowRight" ? 1 : -1)));
              setSelection(null); setFocusIndex(next);
            }
            if (event.key === "Escape") setSelection(null);
          }}>
          {[0, 1, 2, 3, 4].map(i => <g key={i}><line className="usage-grid-line" x1="64" x2="780" y1={20 + i * 48} y2={20 + i * 48} /><text x="56" y={24 + i * 48} textAnchor="end">{format(maximum * (1 - i / 4), digits)}</text></g>)}
          <line className="usage-hover-line" x1={x(index)} x2={x(index)} y1="20" y2="212" />
          {visible.map(id => {
            const data = series.get(id);
            const samples = perCall ? [...(data?.values() ?? [])].map(point => ({ point, position: positions.get(point.day) })) : dates.map((_, i) => ({ point: data?.get(i), position: i }));
            const coordinates = samples.map(({ point, position }) => point && position != null && value(point) != null ? { x: x(position), y: 212 - value(point)! / maximum * 192 } : null);
            const dots = coordinates.filter((point): point is { x: number; y: number } => !!point).map(point => `M ${point.x - 2} ${point.y} a 2 2 0 1 0 4 0 a 2 2 0 1 0 -4 0`).join(" ");
            return <g key={id}>{smoothPaths(coordinates).map((path, i) => <path key={i} d={path} stroke={modelColor(id)} strokeWidth="2" fill="none" />)}<path d={dots} fill={modelColor(id)} /></g>;
          })}
          {selected.map(point => <circle key={`${point.provider}/${point.model}`} cx={x(index)} cy={value(point) == null ? 212 : 212 - value(point)! / maximum * 192} r="4" fill={modelColor(`${point.provider}/${point.model}`)} stroke="var(--panel)" strokeWidth="1.5" />)}
          {slots.map((_, i) => i === 0 || i === count - 1 || (count <= 7 ? true : i % Math.ceil((count - 1) / 6) === 0) ? <text key={i} x={x(i)} y="239" textAnchor="middle">{perCall ? `${i + 1}` : `${dates[i].getMonth() + 1}/${dates[i].getDate()}`}</text> : null)}
        </svg>
        {selection && <div className="health-tooltip" role="tooltip" style={{ left: selection.left, top: selection.top }}>{details}</div>}
      </div></div>
      <div className="usage-chart-detail" aria-live={focusIndex != null ? "polite" : "off"}>{details}</div>
      {perCall && <p className="usage-caption">按调用时间排序，共 {count.toLocaleString("zh-CN")} 次请求尝试；悬停查看精确时间和数值，未报告指标显示“—”。</p>}
    </>}
  </section>;
}
