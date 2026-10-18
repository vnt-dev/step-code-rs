import { useEffect, useState } from "react";
import { api, onStepLoginEvent } from "../api";
import type { Bootstrap, ModelDraft, ProviderDraft, ProviderView } from "../types";
import UsageStatistics from "./UsageStatistics";
import Icon from "./Icon";
import ProviderEditor from "./ProviderEditor";
import ModelEditor from "./ModelEditor";

interface Props { bootstrap: Bootstrap; open: boolean; onSave: (patch: Record<string, unknown>) => Promise<void>; onRefresh: () => Promise<void>; onClose: () => void; }
const emptyProvider: ProviderDraft = { id: "", name: "", baseUrl: "", api: "openai-completions", models: [] };
export default function SettingsPanel({ bootstrap, open, onSave, onRefresh, onClose }: Props) {
  const [tab, setTab] = useState("providers");
  const [providers, setProviders] = useState<ProviderView[]>([]);
  const [draft, setDraft] = useState<ProviderDraft | null>(null);
  const [editing, setEditing] = useState(false);
  const [modelEdit, setModelEdit] = useState<{ provider: ProviderView; model: ModelDraft; index: number } | null>(null);
  const [busy, setBusy] = useState(false);
  const [login, setLogin] = useState(false);
  const [loginReady, setLoginReady] = useState(false);
  const [loginMessage, setLoginMessage] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [removing, setRemoving] = useState<string | null>(null);
  const [preferences, setPreferences] = useState<Record<string, unknown>>({});
  useEffect(() => { if (open) void api.providers().then(setProviders).catch(e => setError(String(e))); }, [open]);
  useEffect(() => {
    let disposed = false;
    let unlisten: (() => void) | undefined;
    void onStepLoginEvent(event => {
      if (event.type === "auth_url") { setLoginReady(true); setLoginMessage("请在浏览器中完成授权，登录结果会自动同步。"); }
    }).then(fn => { if (disposed) fn(); else unlisten = fn; }).catch(e => setError(String(e)));
    return () => { disposed = true; unlisten?.(); };
  }, []);
  const perform = async (action: () => Promise<void>, message: string) => {
    setBusy(true); setError(""); setSuccess("");
    try { await action(); await onRefresh(); setProviders(await api.providers()); setSuccess(message); }
    catch (e) { setError(String(e)); }
    finally { setBusy(false); }
  };
  const startLogin = async () => {
    setLogin(true); setLoginReady(false); setError(""); setSuccess(""); setLoginMessage("正在准备安全登录…");
    try { await api.loginStep(); await onRefresh(); setSuccess("Step 登录成功，模型列表已更新。"); }
    catch (e) { setError(String(e)); }
    finally { setLogin(false); setLoginReady(false); }
  };
  const edit = (provider?: ProviderView) => {
    setEditing(Boolean(provider)); setDraft(provider ? { id: provider.id, name: provider.name, baseUrl: provider.baseUrl, api: provider.api, models: provider.models } : { ...emptyProvider });
    setError(""); setSuccess("");
  };
  return <section className="settings-page" hidden={!open}>
    <aside className="settings-navigation">
      <div className="brand settings-brand"><img className="brand-mark" src="/app-icon.png" alt=""/><div>Step Code<small>WORKSPACE SETTINGS</small></div></div>
      <button className="settings-back" onClick={onClose}><Icon name="chevronLeft" />返回工作区</button>
      <nav aria-label="设置分类">
        <p>基础设置</p>
        {([['general', '常规', 'settings'], ['providers', '模型设置', 'model'], ['account', 'Step 账户', 'shield']] as const).map(([id, label, icon]) => <button key={id} className={tab === id ? 'selected' : ''} aria-current={tab === id ? 'page' : undefined} onClick={() => { setTab(id); setError(''); setSuccess(''); }}><Icon name={icon} />{label}</button>)}
        <p>数据与统计</p>
        <button className={tab === 'usage' ? 'selected' : ''} aria-current={tab === 'usage' ? 'page' : undefined} onClick={() => { setTab('usage'); setError(''); setSuccess(''); }}><Icon name="chart" />使用统计</button>
      </nav>
    </aside>
    <div className="settings-content"><div className="settings-content-inner">
    {tab !== 'usage' && <header className="page-heading"><div><h1>{tab === 'general' ? '常规' : tab === 'account' ? 'Step 账户' : '模型设置'}</h1><p>{tab === 'general' ? '调整对话偏好，让工作方式更适合你。' : tab === 'account' ? '管理账户登录与授权。' : '连接模型服务，管理供应商和可用模型。'}</p></div></header>}
    <UsageStatistics active={open && tab === "usage"} />
    {error && <div className="banner banner-error" role="alert">{error}</div>}{success && <div className="banner banner-success" role="status">{success}</div>}
    {tab === "providers" && <div className="settings-body">
      <div className="section-heading"><div><h2>模型供应商</h2><p>使用 Step 账户或连接兼容的 API 服务。</p></div><button className="primary" disabled={busy || bootstrap.running || !!draft} onClick={() => edit()}>＋ 添加供应商</button></div>
      <div className="provider-card"><div className="provider-logo">S</div><div className="provider-info"><strong>Step</strong><p>内置服务 · 浏览器安全授权</p></div><span className={`status-pill ${bootstrap.stepAuthenticated ? "connected" : ""}`}>{bootstrap.stepAuthenticated ? "已登录" : "未登录"}</span><button onClick={() => setTab("account")}>管理账户 →</button></div>
      {providers.map(provider => <div className="provider-group" key={provider.id}>
        <div className="provider-card"><div className="provider-logo">{provider.name.slice(0, 1).toUpperCase()}</div><div className="provider-info"><strong>{provider.name || provider.id}</strong><p title={provider.baseUrl}>{provider.baseUrl}</p><small>{provider.models.length} 个模型 · {provider.api}</small></div><span className={`status-pill ${provider.configured ? "connected" : ""}`}>{provider.configured ? "已配置" : "待配置"}</span><button disabled={busy || bootstrap.running} onClick={() => edit(provider)}>编辑</button><button className="text-danger" disabled={busy || bootstrap.running} onClick={() => setRemoving(provider.id)}>移除</button></div>
        <div className="provider-models"><div className="model-list-heading"><strong>模型列表</strong><button disabled={busy || bootstrap.running} onClick={() => { setError(""); setModelEdit({ provider, model: { id: "" }, index: -1 }); }}>＋ 添加模型</button></div>
          {provider.models.map((model, index) => <div className="provider-model-row" key={model.id}><span><strong>{model.name || model.id}</strong><small>{model.id} · {((model.contextWindow ?? 200000) / 1000).toLocaleString()}k 上下文</small></span><button disabled={busy || bootstrap.running} onClick={() => { setError(""); setModelEdit({ provider, model, index }); }}>编辑模型</button></div>)}
          {!provider.models.length && <p className="muted">还没有模型，添加后即可在工作台使用。</p>}
        </div>
      </div>)}
      {providers.length === 0 && !draft && <div className="settings-empty"><strong>连接你习惯的模型服务</strong><p>添加 API 地址、密钥和模型，即可在工作台切换使用。</p></div>}
      {removing && <div className="settings-card" role="alert"><strong>移除 {removing}？</strong><p>该供应商将从模型选择中移除，会话记录会保留。</p><div className="modal-actions"><button disabled={busy} onClick={() => setRemoving(null)}>取消</button><button className="danger" disabled={busy} onClick={() => void perform(async () => { await api.deleteProvider(removing); setRemoving(null); }, "供应商已移除。")}>确认移除</button></div></div>}
      {bootstrap.running && <p className="muted">任务执行中，请停止任务后修改供应商。</p>}
    </div>}
    {tab === "account" && <div className="settings-body"><div className="settings-card account-card"><div className="account-mark">S</div><h2>连接 Step 账户</h2><p>在默认浏览器中登录并授权，完成后自动返回登录结果。</p><span className={`status-pill ${bootstrap.stepAuthenticated ? "connected" : ""}`}>{bootstrap.stepAuthenticated ? "已登录 Step" : "尚未登录"}</span><div className="account-actions">{login ? <><p role="status">{loginMessage}</p><button disabled={!loginReady} onClick={() => void api.reopenStepLogin().catch(e => setError(String(e)))}>重新打开浏览器</button><button onClick={() => void api.cancelStepLogin().catch(e => setError(String(e)))}>取消登录</button></> : bootstrap.stepAuthenticated ? <button disabled={busy || bootstrap.running} onClick={() => void perform(() => api.logoutStep(), "已退出 Step 账户。")}>退出登录</button> : <button className="primary" disabled={busy || bootstrap.running} onClick={() => void startLogin()}>使用浏览器登录 ↗</button>}</div></div></div>}
    {tab === "general" && <div className="settings-body"><div className="settings-card"><h2>对话偏好</h2>{[["hideThinkingBlock", "隐藏思考过程", "仅显示模型的最终回答。"], ["contextFiles", "加载项目指令", "读取项目中的 AGENTS.md 上下文。"]].map(([key, label, description]) => <label className="preference-row" key={key}><span><strong>{label}</strong><small>{description}</small></span><input type="checkbox" checked={Boolean(preferences[key] ?? bootstrap.settings[key])} onChange={e => setPreferences({ ...preferences, [key]: e.target.checked })}/></label>)}<div className="modal-actions"><button className="primary" disabled={busy || !Object.keys(preferences).length} onClick={() => void perform(async () => { await onSave(preferences); setPreferences({}); }, "设置已保存。")}>{busy ? "保存中…" : "保存设置"}</button></div></div></div>}
    </div></div>
    {draft && <ProviderEditor initial={draft} editing={editing} busy={busy} error={error} onClose={() => setDraft(null)} onSave={async next => {
      if (!editing && providers.some(provider => provider.id === next.id)) { setError("该供应商 ID 已存在。"); return; }
      await perform(async () => { await api.saveProvider(next); setDraft(null); }, "供应商已保存。");
    }} />}
    {modelEdit && <ModelEditor model={modelEdit.model} editing={modelEdit.index >= 0} busy={busy} error={error} onClose={() => setModelEdit(null)} onSave={async model => {
      const { provider, index } = modelEdit;
      if (provider.models.some((item, i) => i !== index && item.id === model.id)) throw new Error("该模型 ID 已存在。");
      const models = index < 0 ? [...provider.models, model] : provider.models.map((item, i) => i === index ? model : item);
      await perform(async () => { await api.saveProvider({ id: provider.id, name: provider.name, baseUrl: provider.baseUrl, api: provider.api, models }); setModelEdit(null); }, "模型已保存。");
    }} />}
  </section>;
}
