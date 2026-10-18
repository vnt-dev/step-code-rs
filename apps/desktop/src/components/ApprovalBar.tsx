//! Inline confirmation for a pending tool call, docked above the composer.
//!
//! Deliberately not a modal: the sidebar, transcript and composer stay
//! interactive while the agent waits, so a pending confirmation never traps
//! the user in a dialog.

import type { ApprovalRequest } from "../types";
import Icon from "./Icon";
import { formatDuration } from "../lib/messages";
import { useElapsed } from "../lib/useElapsed";

interface ApprovalBarProps {
  request: ApprovalRequest;
  onDecide: (requestId: string, approved: boolean) => void;
}

export default function ApprovalBar({ request, onDecide }: ApprovalBarProps) {
  const waitingMs = useElapsed(true, 100, request.createdAt);
  return (
    <section
      className={`approval-bar${request.hazardous ? " approval-danger" : ""}`}
      aria-label="工具执行确认"
      role="status"
    >
      <header className="approval-heading">
        <span className="approval-icon"><Icon name={request.hazardous ? "alert" : "shield"} /></span>
        <div className="approval-title">
          <strong>{request.hazardous ? "确认高风险操作" : "允许执行此操作？"}</strong>
          <span className="approval-reason">{request.reason}</span>
        </div>
        <span className="approval-waiting" title="任务已等待确认的时间">已等待 {formatDuration(waitingMs)}</span>
      </header>
      <div className="approval-body">
        <code className="approval-tool">{request.toolName}</code>
        {request.argsSummary ? <pre className="approval-args">{request.argsSummary}</pre> : null}
      </div>
      <footer className="approval-actions">
        <span className="approval-hint">任务已暂停，等待你的决定</span>
        <button className="approval-deny" onClick={() => onDecide(request.requestId, false)}>拒绝</button>
        <button className="primary approval-allow" onClick={() => onDecide(request.requestId, true)}>允许执行</button>
      </footer>
    </section>
  );
}
