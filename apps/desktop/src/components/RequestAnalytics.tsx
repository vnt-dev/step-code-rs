import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, onRequestRecordUpdated } from "../api";
import { localDayBoundaries, modelColor, type HealthMode, type HealthPoint, type RequestCursor, type RequestFilter, type RequestPage } from "../lib/request-metrics";
import ModelHealthChart from "./ModelHealthChart";
import { formatTokens } from "../lib/usage-statistics";

const statusLabels: Record<string, string> = { running: "请求中", success: "成功", failed: "失败", cancelled: "已取消", interrupted: "已中断" };
const errorLabels: Record<string,string> = { http:"HTTP 错误",transport:"连接异常",timeout:"请求超时",stream:"流异常",process_exit:"进程退出",producer_dropped:"请求中断" };
const purposes: Record<string, string> = { conversation: "对话", compaction: "上下文压缩", naming: "自动命名" };
const number = (value: number | null | undefined, unit = "", digits = 1) => value == null ? "—" : `${value.toLocaleString("zh-CN", { maximumFractionDigits: digits })}${unit}`;
export default function RequestAnalytics({ active }: { active: boolean }) {
  const [health, setHealth] = useState<HealthPoint[]>([]);
  const [healthMode, setHealthMode] = useState<HealthMode>(7);
  const [loadedMode, setLoadedMode] = useState<HealthMode>(7);
  const [page, setPage] = useState<RequestPage | null>(null);
  const [range, setRange] = useState(7);
  const [model, setModel] = useState("");
  const [status, setStatus] = useState("");
  const [cursors, setCursors] = useState<(RequestCursor | undefined)[]>([undefined]);
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [refresh, setRefresh] = useState(0);
  const [boundaries, setBoundaries] = useState(()=>localDayBoundaries());
  const sequence = useRef(0);
  const healthCache = useRef<Partial<Record<HealthMode, { points: HealthPoint[]; boundaries: number[] }>>>({});
  const tableScroll = useRef<HTMLDivElement>(null);
  useEffect(()=>{tableScroll.current?.scrollTo({top:0});},[range,model,status,cursors]);
  const filter = useMemo<RequestFilter>(()=>({ ...(range ? { fromMs:localDayBoundaries(new Date(),range)[0],toMs:localDayBoundaries().at(-1) } : {}), ...(model?{model}:{}), ...(status?{status}:{}), cursor:cursors.at(-1) }),[range,model,status,cursors,refresh,active]);
  const load = useCallback(async () => {
    const id = ++sequence.current;
    const days = localDayBoundaries(new Date(), healthMode === 30 ? 30 : 7);
    setBusy(true); setError("");
    const results = await Promise.allSettled([api.modelHealth(days, healthMode === "calls"),api.requestRecords(filter)]);
    if(sequence.current!==id) return;
    const failures: string[]=[];
    if(results[0].status==='fulfilled') { healthCache.current[healthMode] = { points: results[0].value, boundaries: days }; setHealth(results[0].value); setBoundaries(days); setLoadedMode(healthMode); } else failures.push(String(results[0].reason));
    if(results[1].status==='fulfilled') { setPage(results[1].value); if(results[1].value.storageError) failures.push(results[1].value.storageError); } else failures.push(String(results[1].reason));
    setError(failures.join('；')); setBusy(false);
  },[filter,healthMode]);
  useEffect(()=>{if(active) void load();return()=>{sequence.current++;};},[active,load,refresh]);
  useEffect(()=>{
    if(!active) return;
    let disposed=false,unlisten:(()=>void)|undefined,timer:ReturnType<typeof setTimeout>|undefined;
    void onRequestRecordUpdated(()=>{if(timer===undefined)timer=setTimeout(()=>{timer=undefined;setRefresh(n=>n+1);},500);}).then(fn=>{if(disposed) fn();else unlisten=fn;}).catch(cause=>setError(String(cause)));
    return()=>{disposed=true;unlisten?.();clearTimeout(timer);};
  },[active]);
  const dates=boundaries.slice(0,-1).map(timestamp=>new Date(timestamp));
  const models=[...new Set(health.map(p=>`${p.provider}/${p.model}`))].sort();
  const resetFilters=(action:()=>void)=>{action();setCursors([undefined]);};
  const switchHealthMode = (mode: HealthMode) => {
    if (mode === healthMode) return;
    const cached = healthCache.current[mode];
    if (cached) { setHealth(cached.points); setBoundaries(cached.boundaries); setLoadedMode(mode); }
    setHealthMode(mode);
  };
  return <div className="request-analytics" aria-busy={busy}>
    <div className="usage-range"><div><h2>模型健康度</h2><span>{healthMode === "calls" ? "最近 500 次调用 · 每次实际请求尝试" : `近 ${healthMode} 天 · 成功请求的每日有效均值`} · 含对话、压缩和命名</span></div><div className="usage-switch" role="group" aria-label="模型健康度显示方式">{([7,30,"calls"] as const).map(mode=><button key={mode} aria-pressed={healthMode===mode} onClick={()=>switchHealthMode(mode)}>{mode==="calls"?"每次调用":`近 ${mode} 天`}</button>)}</div></div>
    {error && <div className="banner banner-error" role="alert"><strong>请求记录暂时不可用</strong><p>已加载的数据会保留，重新打开此页面可重试。</p><details><summary>错误详情</summary>{error}</details></div>}
    {!page && busy && <p className="usage-caption" role="status">正在读取请求性能…</p>}
    <div className="usage-chart-legend health-legend" role="group" aria-label="健康度模型图例">{models.map(id=><button key={id} aria-pressed={!hidden.has(id)} onClick={()=>setHidden(previous=>{const next=new Set(previous);if(next.has(id))next.delete(id);else next.add(id);return next;})}><i style={{background:modelColor(id)}}/>{id}</button>)}</div>
    <ModelHealthChart points={health} dates={dates} models={models} hidden={hidden} metric="tps" perCall={loadedMode==="calls"}/>
    <ModelHealthChart points={health} dates={dates} models={models} hidden={hidden} metric="firstTokenMs" perCall={loadedMode==="calls"}/>
    <section className="usage-card request-records"><div className="usage-card-heading"><div><h2>请求明细</h2><p>每次实际尝试单独记录 · 删除会话后仍保留</p></div><span className="usage-unit">每页 50 条</span></div>
      <div className="request-filters"><label>时间<select aria-label="请求时间范围" value={range} disabled={busy} onChange={event=>resetFilters(()=>setRange(Number(event.target.value)))}><option value={7}>近七天</option><option value={30}>近三十天</option><option value={0}>全部</option></select></label><label>模型<select aria-label="请求模型" value={model} disabled={busy} onChange={event=>resetFilters(()=>setModel(event.target.value))}><option value="">全部模型</option>{page?.models.map(id=><option key={id} value={id}>{id}</option>)}</select></label><label>结果<select aria-label="请求结果" value={status} disabled={busy} onChange={event=>resetFilters(()=>setStatus(event.target.value))}><option value="">全部结果</option>{Object.entries(statusLabels).map(([id,label])=><option key={id} value={id}>{label}</option>)}</select></label></div>
      <div ref={tableScroll} className="request-table-scroll"><table className="request-table"><thead><tr><th>时间 / 用途</th><th>模型 / 会话</th><th>输入 Token</th><th>缓存读 / 写</th><th>输出 Token</th><th>输出速度</th><th>首字耗时</th><th>结果</th></tr></thead><tbody>{page?.records.map(record=><tr key={record.id}><td title={record.id}>{new Date(record.startedAt).toLocaleString('zh-CN',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit'})}<small>{purposes[record.context.purpose]??record.context.purpose} · 尝试 {record.attempt}</small></td><td title={`${record.provider}/${record.model}${record.responseModel?' → '+record.responseModel:''}`}><strong>{record.model}</strong><small>{record.provider} · {record.sessionName||record.context.sessionId.slice(0,8)}{record.sessionDeleted?'（会话已删除）':''}</small></td><td title={number(record.inputTokens)}>{record.inputTokens==null?'—':formatTokens(record.inputTokens)}</td><td title={`${number(record.cacheReadTokens)} / ${number(record.cacheWriteTokens)}`}>{record.cacheReadTokens==null?'—':formatTokens(record.cacheReadTokens)} / {record.cacheWriteTokens==null?'—':formatTokens(record.cacheWriteTokens)}</td><td title={number(record.outputTokens)}>{record.outputTokens==null?'—':formatTokens(record.outputTokens)}</td><td>{number(record.tps,' tps')}</td><td>{number(record.firstTokenMs==null?null:record.firstTokenMs/1000,' 秒',3)}</td><td><span className={`request-result result-${record.status}`}>{statusLabels[record.status]??record.status}</span>{(record.errorKind||record.httpStatus) && <small>{record.httpStatus??''} {record.errorKind ? errorLabels[record.errorKind] ?? record.errorKind : ''}</small>}</td></tr>)}</tbody></table></div>
      {!page?.records.length && <div className="request-empty">{busy?'正在加载请求记录…':error?'请求记录未加载，重新打开此页面可重试。':'当前筛选条件下暂无请求记录'}</div>}
      <div className="request-pagination"><span>第 {cursors.length} 页 · 当前 {page?.records.length??0} 条</span><button disabled={busy||!!error||cursors.length===1} onClick={()=>setCursors(previous=>previous.slice(0,-1))}>上一页</button><button disabled={busy||!!error||!page?.nextCursor} onClick={()=>setCursors(previous=>[...previous,page?.nextCursor??undefined])}>下一页</button></div>
    </section>
    <details className="usage-notes"><summary>性能统计口径</summary><p>输入 Token 包含缓存读写，缓存列单独列示其中的用量。首字耗时从请求开始到首个有效输出，包含思考、正文和工具参数。平均 TPS 按有效输出 Token 总数 / 生成时间总和计算；不包含首字等待、重试退避和工具执行时间。失败、取消、单块输出或缺少 usage 的请求不计入速度均值。空白日期表示没有有效样本，曲线不会跨空白日期连接。历史会话没有准确计时，性能仅从启用记录后的新请求统计。</p></details>
  </div>;
}
