import { useEffect, useState } from "react";
import Icon from "./Icon";
import Dropdown from "./Dropdown";
import { groupProjects, projectKey, relativeTime } from "../lib/projects";
import type { SessionSummary } from "../types";
interface Props {
  sessions: SessionSummary[];
  sessionActivity: Record<string, "running" | "approval">;
  projects: string[];
  removedProjects: string[];
  onAddProject: () => Promise<string | null>;
  onRemoveProject: (path: string) => Promise<void>;
  activeId: string | null;
  disabled: boolean;
  settingsOpen: boolean;
  onOpen: (id: string) => Promise<void>;
  onNew: (cwd?: string) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
  onRename: (id: string, name: string) => Promise<void>;
  onSettings: () => void;
}

type SessionAction = { session: SessionSummary; type: "rename" | "delete" };

const label = (session: SessionSummary) => session.name?.trim() || "新会话";

export default function Sidebar(props: Props) {
  const [query, setQuery] = useState("");
  const [action, setAction] = useState<SessionAction | null>(null);
  const [name, setName] = useState("");
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);
  const [projectsOpen, setProjectsOpen] = useState(true);
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 60000);
    return () => window.clearInterval(timer);
  }, []);

  const groups = groupProjects(props.projects, props.sessions, query, props.removedProjects);

  const addProject = async () => {
    setAdding(true);
    try {
      const path = await props.onAddProject();
      if (path) {
        setQuery("");
        setProjectsOpen(true);
        setCollapsed(current => { const next = new Set(current); next.delete(projectKey(path)); return next; });
      }
    } finally { setAdding(false); }
  };

  const toggleGroup = (key: string) =>
    setCollapsed(current => {
      const next = new Set(current);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  const expanded = (key: string) => Boolean(query.trim()) || !collapsed.has(key);

  const openAction = (next: SessionAction) => {
    setActionError(null);
    setAction(next);
    setName(next.session.name ?? "");
  };

  const submit = async () => {
    if (!action) return;
    setBusy(true);
    setActionError(null);
    try {
      if (action.type === "rename") await props.onRename(action.session.id, name);
      else await props.onDelete(action.session.id);
      setAction(null);
    } catch (cause) {
      setActionError(String(cause));
    } finally {
      setBusy(false);
    }
  };

  return <aside className="sidebar">
    <button className="new-session" disabled={props.disabled} onClick={() => void props.onNew()}><Icon name="compose"/><span>新聊天</span></button>
    <label className="session-search"><span>⌕</span><input aria-label="搜索会话" placeholder="搜索会话…" value={query} onChange={e => setQuery(e.target.value)}/></label>
    <div className="projects-heading">
      <button className="projects-toggle" aria-expanded={projectsOpen || Boolean(query.trim())} onClick={() => setProjectsOpen(!projectsOpen)}>项目 <Icon name="chevron" style={{ transform: projectsOpen || query.trim() ? undefined : "rotate(-90deg)" }}/></button>
      <button className={`project-add${adding ? " pending" : ""}`} aria-label="添加项目" title="添加项目 · 选择文件夹" disabled={props.disabled || adding} onClick={() => void addProject()}><Icon name="plus"/></button>
    </div>
    <ul className="session-list">
      {(projectsOpen || query.trim()) && groups.map(({key, path, name: projectName, sessions: items}) => <li className="session-group" key={key}>
        <div className="project-row">
        <button
          type="button"
          className="session-group-header"
          aria-expanded={expanded(key)}
          onClick={() => toggleGroup(key)}
        >
          <span className="project-folder"><Icon name="folder"/><span className="project-caret"><Icon name="chevron" style={{ transform: expanded(key) ? undefined : "rotate(-90deg)" }}/></span></span>
          <span className="session-group-name" title={path}>{projectName}</span>
        </button>
        <button className="project-new" disabled={props.disabled} aria-label={`在 ${projectName} 中新建会话`} title="新建项目会话" onClick={() => { setCollapsed(current => { const next = new Set(current); next.delete(key); return next; }); void props.onNew(path); }}><Icon name="plus"/></button>
        <Dropdown label={`${projectName} 项目操作`} trigger={<span aria-hidden="true">···</span>} className="project-menu" actions placement="auto" disabled={props.disabled} options={[{ value: "remove", label: "移除项目", description: "保留文件、历史会话和草稿，可重新添加" }]} onSelect={() => props.onRemoveProject(path)} />
        </div>
        {expanded(key) && <ul className="session-group-items">
          {items.map(s => <li className={`session-item${s.id === props.activeId && !props.settingsOpen ? " active" : ""}`} key={s.id}>
            <button className="session-open" title={label(s)} aria-current={s.id === props.activeId && !props.settingsOpen ? "page" : undefined} disabled={props.disabled} onClick={() => void props.onOpen(s.id)}>
              <span className="session-name">{label(s)}</span>
              {props.sessionActivity[s.id] ? <span className="session-time" role="status">{props.sessionActivity[s.id] === "approval" ? "等待确认" : "运行中"}</span>
                : <time className="session-time" dateTime={new Date(s.modifiedAt).toISOString()} title={new Date(s.modifiedAt).toLocaleString("zh-CN")}>{relativeTime(s.modifiedAt, now)}</time>}
            </button>
            <div className="session-actions">
              <button aria-label={`重命名 ${label(s)}`} title="重命名" disabled={props.disabled} onClick={() => openAction({ session: s, type: "rename" })}>✎</button>
              <button aria-label={`删除 ${label(s)}`} title={props.sessionActivity[s.id] ? "停止该会话的任务后可删除" : "删除"} disabled={props.disabled || Boolean(props.sessionActivity[s.id])} onClick={() => openAction({ session: s, type: "delete" })}>×</button>
            </div>
          </li>)}
          {!items.length && <li className="project-empty"><span>还没有会话</span><button disabled={props.disabled} onClick={() => void props.onNew(path)}>新建会话</button></li>}
        </ul>}
      </li>)}
      {!groups.length && <li className="session-empty">{query.trim() ? "没有匹配的项目或会话。" : "添加项目文件夹，开始你的工作。"}</li>}
    </ul>
    <div className="sidebar-footer"><button className={`settings-entry${props.settingsOpen ? " selected" : ""}`} onClick={props.onSettings}><span>⚙</span> 设置</button></div>
    {action && <div className="modal-backdrop"><form
      className="modal"
      role="dialog"
      aria-modal="true"
      aria-labelledby="session-action-title"
      onKeyDown={e => { if (e.key === "Escape" && !busy) setAction(null); }}
      onSubmit={e => { e.preventDefault(); void submit(); }}
    >
      <h3 id="session-action-title">
        {action.type === "rename" ? "重命名会话" : "删除会话"}
      </h3>
      {action.type === "rename" && <input autoFocus aria-label="会话名称" value={name} onChange={e => setName(e.target.value)} required/>}
      {action.type === "delete" && <p>确认删除「{label(action.session)}」？此操作无法撤销。</p>}
      {actionError && <p className="error-text" role="alert">{actionError}</p>}
      <div className="modal-actions">
        <button autoFocus={action.type === "delete"} type="button" disabled={busy} onClick={() => setAction(null)}>取消</button>
        <button disabled={busy} className={action.type === "delete" ? "danger" : "primary"}>
          {busy ? "处理中…" : "确认"}
        </button>
      </div>
    </form></div>}
  </aside>;
}
