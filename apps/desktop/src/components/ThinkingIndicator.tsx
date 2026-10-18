//! Activity indicator shown throughout the agent's response:
//! a pulsing orb, animated dots and a live elapsed timer.

import { formatElapsed } from "../lib/messages";
import { useElapsed } from "../lib/useElapsed";

export default function ThinkingIndicator({ startedAt }: { startedAt?: number }) {
  const elapsed = useElapsed(true, 100, startedAt);
  return (
    <div className="message message-assistant thinking-indicator" role="status" aria-live="polite">
      <span className="thinking-orb" aria-hidden="true" />
      <span className="thinking-label">正在思考</span>
      <span className="thinking-dots" aria-hidden="true"><i /><i /><i /></span>
      <span className="thinking-elapsed" title="已等待时间">{formatElapsed(elapsed)}</span>
    </div>
  );
}
