//! The message timeline: user prompts, assistant turns with thinking blocks,
//! tool cards, and system notices.

import { useLayoutEffect, useRef, useState } from "react";
import type { AgentMessage, AssistantMessage, ExecutionTiming, ImageBlock, ToolCallView } from "../types";
import {
  assistantText,
  assistantThinking,
  assistantToolCalls,
  userText,
  executionGroups,
  formatWorkDuration,
  formatWorkBreakdown,
  formatThinkingDuration,
} from "../lib/messages";
import { imageSource } from "../lib/images";
import { useElapsed } from "../lib/useElapsed";
import { observeMessageScroll } from "../lib/message-scroll";
import ToolCard, { Markdown } from "./ToolCard";
import Icon from "./Icon";
import ThinkingIndicator from "./ThinkingIndicator";
import ImageLightbox from "./ImageLightbox";

interface MessageListProps {
  messages: AgentMessage[];
  /** The session's transcript is still loading; show a placeholder instead of the welcome screen. */
  loading?: boolean;
  toolCalls: Map<string, ToolCallView>;
  /** Thinking duration in ms per assistant message timestamp. */
  thinkingDurations: Map<number, number>;
  /** Thinking currently streaming, for the live ticking duration. */
  thinkingStart: { timestamp: number; startedAt: number } | null;
  /** When the current run began, so the thinking timer survives session switches. */
  runStartedAt: number | null;
  notices: string[];
  hideThinking: boolean;
  /** A prompt was sent and the model has not replied yet. */
  awaitingResponse: boolean;
  running?: boolean;
  onRecall?: (timestamp: number) => void;
  recallDisabled?: boolean;
  recalling?: boolean;
  executionTimings?: ExecutionTiming[];
  onSuggestion: (text: string) => void;
}

function isUserMessage(message: AgentMessage): message is Extract<AgentMessage, { role: "user" }> {
  return message.role === "user";
}

function isAssistantMessage(
  message: AgentMessage,
): message is Extract<AgentMessage, { role: "assistant" }> {
  return message.role === "assistant";
}

export default function MessageList({
  messages,
  loading = false,
  toolCalls,
  thinkingDurations,
  thinkingStart,
  runStartedAt,
  notices,
  hideThinking,
  running = false,
  onRecall,
  recallDisabled = false,
  recalling = false,
  executionTimings = [],
  onSuggestion,
}: MessageListProps) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<ReturnType<typeof observeMessageScroll> | null>(null);
  const [showJump, setShowJump] = useState(false);
  const [lightbox, setLightbox] = useState<{ images: ImageBlock[]; index: number } | null>(null);

  useLayoutEffect(() => {
    const scroll = observeMessageScroll(viewportRef.current!, contentRef.current!, pinned => setShowJump(!pinned));
    scrollRef.current = scroll;
    return () => {
      scroll.dispose();
      scrollRef.current = null;
    };
  }, []);

  // Pin React updates before paint; the observer also handles later layout
  // changes (images, tool details, composer height and window size).
  useLayoutEffect(() => { scrollRef.current?.follow(); });

  const lastUserMessage = [...messages].reverse().find(isUserMessage);

  const renderMessage = (message: AgentMessage, index: number) => {
    if (isUserMessage(message)) {
      const images = Array.isArray(message.content) ? message.content.filter((block): block is ImageBlock => block.type === "image") : [];
      return <div key={index} className="message message-user">
        {images.length > 0 && <div className="message-images">{images.map((image, imageIndex) =>
          <button key={imageIndex} type="button" className="message-image" aria-label={`放大图片 ${imageIndex + 1}`} onClick={() => setLightbox({ images, index: imageIndex })}>
            <img src={imageSource(image)} alt={`附加图片 ${imageIndex + 1}`} />
          </button>
        )}</div>}
        {userText(message) && <div className="bubble">{userText(message)}</div>}
        {onRecall && message === lastUserMessage && <div className="message-actions">
          <button type="button" className="message-recall" disabled={running || loading || recallDisabled}
            aria-label={recalling ? "撤回中" : "撤回消息"} aria-busy={recalling}
            title={running ? "请先停止当前执行，再撤回消息" : "撤回这条消息，并删除之后的回复和工具记录"}
            onClick={() => onRecall(message.timestamp)}><Icon name="undo" /></button>
        </div>}
      </div>;
    }
    if (isAssistantMessage(message)) return <AssistantTurn key={index} message={message} toolCalls={toolCalls} thinkingDurations={thinkingDurations} thinkingStart={thinkingStart} hideThinking={hideThinking} />;
    return null;
  };

  return (
    <div className="message-list" ref={viewportRef}>
      <div className="message-list-content" ref={contentRef}>
      {!messages.length && loading ? <div className="message-loading" role="status" aria-label="正在加载会话"><span className="thinking-dots" aria-hidden="true"><i /><i /><i /></span></div> : null}
      {!messages.length && !loading && <div className="welcome">
        <div className="welcome-symbol">✳</div><span className="eyebrow">BUILD WITH STEP</span>
        <h1>让想法，成为下一步。</h1><p>从理解项目到交付代码，在这里开始你的工作。</p>
        <div className="welcome-cards">
          <button onClick={() => onSuggestion("请检查当前项目的代码，找出值得改进的问题，并提出具体的修复建议。")}><span>◇</span><strong>改进代码</strong><p>定位问题、优化与验证</p></button>
        </div>
      </div>}
      {notices.map((notice, index) => (
        <div key={`notice-${index}`} className="notice">
          {notice}
        </div>
      ))}

      {executionGroups(messages, running, executionTimings).map((group, index) => (
        <div className="execution-group" key={index}>
          {!group.completed ? group.messages.map(renderMessage) : <>
            {group.messages.filter(isUserMessage).map(renderMessage)}
            <details className="execution-history">
              <summary>
                <span className="execution-duration">{formatWorkDuration(group.durationMs)}
                  {formatWorkBreakdown(group.timing) && <span className="execution-breakdown"> · {formatWorkBreakdown(group.timing)}</span>}
                </span>
                <span className="execution-chevron" aria-hidden="true">›</span>
              </summary>
              <div className="execution-history-content">
                {group.messages.filter(isAssistantMessage).map((message, index) => (
                  <AssistantTurn key={index} message={message} toolCalls={toolCalls} thinkingDurations={thinkingDurations} thinkingStart={thinkingStart} hideThinking={hideThinking} processOnly={message === group.finalAssistant} />
                ))}
              </div>
            </details>
            {group.finalAssistant && <AssistantTurn message={group.finalAssistant} toolCalls={toolCalls} thinkingDurations={thinkingDurations} thinkingStart={thinkingStart} hideThinking={true} resultOnly />}
          </>}
        </div>
      ))}
      {running && <ThinkingIndicator startedAt={runStartedAt ?? undefined} />}
      </div>
      {showJump && messages.length > 0 && <button className="jump-latest" onClick={() => scrollRef.current?.jumpToLatest()}>回到最新消息 ↓</button>}
      {lightbox && <ImageLightbox images={lightbox.images} index={lightbox.index} onIndexChange={index => setLightbox({ ...lightbox, index })} onClose={() => setLightbox(null)} />}
    </div>
  );
}

function AssistantTurn({
  message,
  toolCalls,
  thinkingDurations,
  thinkingStart,
  hideThinking,
  processOnly = false,
  resultOnly = false,
}: {
  message: AssistantMessage;
  toolCalls: Map<string, ToolCallView>;
  thinkingDurations: Map<number, number>;
  thinkingStart: { timestamp: number; startedAt: number } | null;
  hideThinking: boolean;
  processOnly?: boolean;
  resultOnly?: boolean;
}) {
  const text = assistantText(message);
  const thinking = assistantThinking(message);
  const calls = assistantToolCalls(message);
  const failed = message.stopReason === "error";
  const durationMs = thinkingDurations.get(message.timestamp);
  const liveSince = thinkingStart?.timestamp === message.timestamp ? thinkingStart.startedAt : undefined;

  if (processOnly && (!thinking || hideThinking) && calls.length === 0) return null;
  if (resultOnly && !text && !(failed && message.errorMessage)) return null;

  return (
    <div className={`message message-assistant${failed ? " message-failed" : ""}`}>
      {thinking && !hideThinking ? (
        <details className="thinking">
          <summary>
            <Icon name="brain" className="thinking-icon" />
            <span className="thinking-title">思考</span>
            {liveSince !== undefined
              ? <ThinkingDuration since={liveSince} />
              : durationMs !== undefined && <span className="thinking-duration">· 持续了 {formatThinkingDuration(durationMs)}</span>}
            <span className="thinking-chevron" aria-hidden="true">›</span>
          </summary>
          <pre>{thinking}</pre>
        </details>
      ) : null}
      {text && !processOnly ? <Markdown text={text} /> : null}
      {!resultOnly && calls.map((call) => {
        if (call.type !== "toolCall") return null;
        const view = toolCalls.get(call.id);
        if (!view) return null;
        return <ToolCard key={call.id} call={view} />;
      })}
      {failed && message.errorMessage && !processOnly ? (
        <div className="error-text">{message.errorMessage}</div>
      ) : null}
    </div>
  );
}

/** Duration that keeps ticking while the model is still thinking. */
function ThinkingDuration({ since }: { since: number }) {
  const elapsed = useElapsed(true, 1000, since);
  return <span className="thinking-duration">· 持续了 {formatThinkingDuration(elapsed)}</span>;
}
