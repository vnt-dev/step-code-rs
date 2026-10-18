//! Average output speed of the last round, shown beside the context gauge.

import type { AgentMessage, SessionStats } from "../types";
import { roundTps } from "../lib/round-tps";
import { useNow } from "../lib/useElapsed";
import Icon from "./Icon";

export default function RoundSpeed({ messages, stats, running }: {
  messages: AgentMessage[];
  stats: SessionStats | null;
  running: boolean;
}) {
  const now = useNow(running);
  const tps = roundTps(messages, stats, running, now);
  return (
    <span className="round-speed" title={running ? "最后一轮对话的平均输出速度（实时）" : "最后一轮对话的平均输出速度"}>
      <Icon name="bolt" />
      {tps == null ? "—" : `${tps.toFixed(1)} tps`}
    </span>
  );
}
