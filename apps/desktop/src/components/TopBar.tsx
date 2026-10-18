import type { Bootstrap } from "../types";
import { formatDuration } from "../lib/messages";
import { useElapsed } from "../lib/useElapsed";
interface Props { bootstrap: Bootstrap | null; running: boolean; approvalPending: boolean; /** When the current run began, so the timer survives session switches. */ runStartedAt: number | null; }
export default function TopBar({ bootstrap, running, approvalPending, runStartedAt }: Props) {
  const elapsedMs = useElapsed(running, 100, runStartedAt ?? undefined);
  const status = approvalPending ? "等待确认" : running ? "正在执行" : "准备就绪";
  // While a confirmation is pending the approval bar owns the waiting clock.
  const showElapsed = running && !approvalPending;
  return <header className="topbar conversation-header">
    <div className="conversation-heading"><strong>{bootstrap?.activeSession?.name || "新会话"}</strong><span title={bootstrap?.cwd}>{bootstrap?.cwd.split(/[\\/]/).pop() || "本地工作空间"}</span></div>
    <div className="conversation-status">
      <span className={`${approvalPending ? "awaiting-state" : running ? "running-state" : "idle-state"}${showElapsed ? " with-elapsed" : ""}`}><i />{status}{showElapsed && <em title="本次任务已运行时间">{formatDuration(elapsedMs)}</em>}</span>
    </div>
  </header>;
}
