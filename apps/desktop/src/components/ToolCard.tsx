//! One tool call: a collapsed summary row that expands to arguments and result.

import ReactMarkdown from "react-markdown";
import { useState } from "react";
import type { MouseEvent } from "react";
import { api } from "../api";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import Icon from "./Icon";
import { toolIcon, toolLabel, toolTarget } from "../lib/tool-presentation";
import type { ToolCallView } from "../types";

interface ToolCardProps {
  call: ToolCallView;
}

/** One argument, whitespace-collapsed and capped so long content stays skimmable. */
function argValue(value: unknown): string {
  const text = typeof value === "string" ? value : JSON.stringify(value);
  const oneLine = text.replace(/\s+/g, " ").trim();
  return oneLine.length > 200 ? `${oneLine.slice(0, 197)}…` : oneLine;
}

export default function ToolCard({ call }: ToolCardProps) {
  const label = toolLabel(call.name);
  const target = toolTarget(call.name, call.args);
  const entries = Object.entries(call.args);

  return (
    <details className={`tool-card tool-${call.status}`}>
      <summary className="tool-summary" title={`${label}${target.title ? ` · ${target.title}` : ""}（点击展开）`}>
        <Icon name={toolIcon(call.name)} className="tool-icon" />
        <span className="tool-label">{label}</span>
        <span className="tool-target">
          {target.title && <span className="tool-target-title">{target.title}</span>}
          {target.detail && <span className="tool-target-detail">{target.detail}</span>}
        </span>
        {call.status === "running" && <span className="tool-running-dot" role="status" aria-label="运行中" />}
        {call.status === "error" && <span className="tool-failed">失败</span>}
        <span className="tool-chevron" aria-hidden="true">›</span>
      </summary>
      <div className="tool-detail">
        {entries.length > 0 && (
          <div className="tool-args">
            {entries.map(([key, value]) => (
              <div key={key} className="tool-arg">
                <span className="tool-arg-key">{key}</span>
                <span className="tool-arg-value">{argValue(value)}</span>
              </div>
            ))}
          </div>
        )}
        {call.result ? (
          <pre className="tool-result">{call.result}</pre>
        ) : call.status === "running" ? null : (
          <div className="tool-result tool-result-empty">(no output)</div>
        )}
      </div>
    </details>
  );
}

/** Markdown renderer shared by assistant messages and tool output. */
export function Markdown({ text }: { text: string }) {
  const [linkError, setLinkError] = useState("");
  function openLink(event: MouseEvent<HTMLAnchorElement>, href?: string) {
    event.preventDefault();
    if (event.type === "auxclick" && event.button !== 1) return;
    if (!href) return;
    setLinkError("");
    void api.openExternalUrl(href).catch(error => setLinkError(String(error)));
  }
  return (
    <div className="markdown">
      <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}
        components={{ a: ({ href, children, title }) => (
          <a href={href} title={title} target="_blank" rel="noopener noreferrer"
            onClick={event => openLink(event, href)} onAuxClick={event => openLink(event, href)}>
            {children}
          </a>
        ), table: ({ children }) => (
          // Wide tables scroll here instead of stretching the message list.
          <div className="markdown-table"><table>{children}</table></div>
        ) }}>
        {text}
      </ReactMarkdown>
      {linkError && <p role="alert">{linkError}</p>}
    </div>
  );
}
