import type { SessionSummary } from "../types";

export const folderName = (path: string) => path.replace(/[\\/]+$/, "").split(/[\\/]/).pop() || path;
// Windows paths are case insensitive; POSIX paths retain their case.
export const projectKey = (path: string) => {
  const normalized = path.replace(/^\\\\\?\\/, "").replace(/\\/g, "/").replace(/\/+$/, "");
  return /^[a-z]:/i.test(normalized) || normalized.startsWith("//") ? normalized.toLowerCase() : normalized;
};

export function groupProjects(projects: string[], sessions: SessionSummary[], query: string, removedProjects: string[] = []) {
  const removed = new Set(removedProjects.map(projectKey));
  const groups = new Map<string, { key: string; path: string; name: string; sessions: SessionSummary[] }>();
  for (const path of [...projects, ...sessions.map(s => s.cwd)]) {
    const key = projectKey(path);
    if (removed.has(key)) continue;
    if (!groups.has(key)) groups.set(key, { key, path, name: folderName(path), sessions: [] });
  }
  for (const session of sessions) groups.get(projectKey(session.cwd))?.sessions.push(session);
  const search = query.trim().toLowerCase();
  return [...groups.values()].map(group => ({ ...group, sessions: group.sessions
    .filter(s => `${s.name || "新会话"} ${group.name} ${group.path}`.toLowerCase().includes(search))
    .sort((a, b) => b.modifiedAt - a.modifiedAt),
  })).filter(group => !search || group.sessions.length || `${group.name} ${group.path}`.toLowerCase().includes(search));
}

export function relativeTime(timestamp: number, now = Date.now()) {
  const minutes = Math.max(0, Math.floor((now - timestamp) / 60000));
  if (minutes < 1) return "刚刚";
  if (minutes < 60) return `${minutes}分钟`;
  if (minutes < 1440) return `${Math.floor(minutes / 60)}小时`;
  if (minutes < 10080) return `${Math.floor(minutes / 1440)}天`;
  return new Date(timestamp).toLocaleDateString("zh-CN", { month: "numeric", day: "numeric" });
}
