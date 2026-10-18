//! Inline review for a plan submitted by `exit_plan_mode`, docked above the
//! composer.
//!
//! Deliberately not a modal, like the approval bar: the sidebar, transcript
//! and composer stay interactive while the agent waits, so a pending review
//! never traps the user in a dialog.

import { useState } from "react";
import type { PlanReviewRequest } from "../types";
import Icon from "./Icon";
import { Markdown } from "./ToolCard";
import { formatDuration } from "../lib/messages";
import { useElapsed } from "../lib/useElapsed";

interface PlanReviewBarProps {
  request: PlanReviewRequest;
  onDecide: (requestId: string, outcome: "approved" | "feedback" | "dismissed", feedback?: string) => void;
}

export default function PlanReviewBar({ request, onDecide }: PlanReviewBarProps) {
  const waitingMs = useElapsed(true, 100, request.createdAt);
  const [notes, setNotes] = useState("");
  const [editing, setEditing] = useState(false);
  return (
    <section className="plan-review-bar" aria-label="计划审阅" role="status">
      <header className="approval-heading">
        <span className="approval-icon"><Icon name="plan" /></span>
        <div className="approval-title">
          <strong>审阅执行计划</strong>
          <span className="approval-reason">批准后按计划执行；提出修改将留在计划模式继续完善。</span>
        </div>
        <span className="approval-waiting" title="任务已等待审阅的时间">已等待 {formatDuration(waitingMs)}</span>
      </header>
      <div className="approval-body">
        <code className="approval-tool">{request.planFilePath}</code>
        <div className="plan-review-contents"><Markdown text={request.planContents} /></div>
      </div>
      <footer className="approval-actions">
        <span className="approval-hint">任务已暂停，等待你的决定</span>
        {editing ? (
          <div className="plan-review-feedback">
            <textarea
              aria-label="修改意见"
              value={notes}
              onChange={event => setNotes(event.target.value)}
              placeholder="说明需要调整的地方，模型会更新计划后再次提交…"
              rows={3}
            />
            <button className="primary approval-allow" disabled={!notes.trim()} onClick={() => onDecide(request.requestId, "feedback", notes.trim())}>发送修改意见</button>
            <button className="approval-deny" onClick={() => setEditing(false)}>取消</button>
          </div>
        ) : (
          <>
            <button className="approval-deny" onClick={() => onDecide(request.requestId, "dismissed")}>暂不批准</button>
            <button className="approval-deny" onClick={() => setEditing(true)}>提出修改</button>
            <button className="primary approval-allow" onClick={() => onDecide(request.requestId, "approved")}>批准并执行</button>
          </>
        )}
      </footer>
    </section>
  );
}
