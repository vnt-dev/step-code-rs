import { useEffect, useRef, type ReactNode } from "react";

export default function SettingsModal({ title, busy, onClose, children }: {
  title: string; busy: boolean; onClose: () => void; children: ReactNode;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const close = useRef(onClose);
  close.current = onClose;
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    (ref.current?.querySelector<HTMLElement>("input:not(:disabled)") ?? ref.current?.querySelector<HTMLElement>("button:not(:disabled)"))?.focus();
    return () => previous?.focus();
  }, []);
  return <div className="modal-backdrop settings-modal-backdrop" onMouseDown={event => {
    if (event.target === event.currentTarget && !busy) close.current();
  }}><div ref={ref} className="modal settings-modal" role="dialog" aria-modal="true" aria-label={title} onKeyDown={event => {
    if (event.key === "Escape" && !busy) { event.stopPropagation(); close.current(); }
    if (event.key === "Tab") {
      const items = Array.from(ref.current?.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled), textarea:not(:disabled), select:not(:disabled), summary, [tabindex="0"]') ?? []);
      const first = items[0], last = items[items.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    }
  }}><header className="settings-modal-heading"><h2>{title}</h2><button type="button" aria-label="关闭弹窗" disabled={busy} onClick={onClose}>×</button></header>{children}</div></div>;
}
