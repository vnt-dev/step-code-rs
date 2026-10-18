import { useEffect, useId, useLayoutEffect, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { createPortal } from "react-dom";
import Icon from "./Icon";
export interface DropdownOption { value: string; label: string; description?: string; group?: string; icon?: ReactNode; disabled?: boolean; }
interface Props {
  label: string; trigger: ReactNode; options: DropdownOption[]; value?: string;
  onSelect: (value: string) => void | Promise<void>;
  onOpen?: () => void;
  restoreFocusOnSelect?: boolean;
  disabled?: boolean; searchable?: boolean; actions?: boolean;
  className?: string; placement?: "top" | "auto"; align?: "start" | "end";
  searchPlaceholder?: string; emptyText?: string;
}
export default function Dropdown({ label, trigger, options, value, onSelect, onOpen, restoreFocusOnSelect = true, disabled, searchable, actions, className = "", placement = "top", align = "start", searchPlaceholder = "搜索模型或供应商…", emptyText = "没有匹配的模型" }: Props) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [position, setPosition] = useState({ left: 0, top: 0, width: 300, maxHeight: 360 });
  const button = useRef<HTMLButtonElement>(null);
  const panel = useRef<HTMLDivElement>(null);
  const id = useId();
  const filtered = options.filter(option => `${option.label} ${option.description ?? ""} ${option.group ?? ""}`.toLowerCase().includes(query.toLowerCase()));
  const close = (restoreFocus = false) => { setOpen(false); setQuery(""); setError(""); if (restoreFocus) requestAnimationFrame(() => button.current?.focus()); };
  useEffect(() => { if (disabled && !pending) setOpen(false); }, [disabled, pending]);
  useLayoutEffect(() => {
    if (!open || !button.current || !panel.current) return;
    const place = () => {
      const rect = button.current!.getBoundingClientRect();
      const above = placement === "top" || window.innerHeight - rect.bottom < Math.min(rect.top, 320);
      const width = Math.min(searchable ? 340 : 300, window.innerWidth - 24);
      const maxHeight = Math.max(100, Math.min(360, (above ? rect.top : window.innerHeight - rect.bottom) - 20));
      const height = Math.min(panel.current!.scrollHeight, maxHeight);
      const left = Math.max(12, Math.min(align === "end" ? rect.right - width : rect.left, window.innerWidth - width - 12));
      setPosition({ left, top: above ? Math.max(12, rect.top - height - 8) : rect.bottom + 8, width, maxHeight });
    };
    place();
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true);
    return () => { window.removeEventListener("resize", place); window.removeEventListener("scroll", place, true); };
  }, [open, query, error, options, placement, align, searchable]);
  useEffect(() => {
    if (!open) return;
    (panel.current?.querySelector("input") ?? panel.current?.querySelector<HTMLButtonElement>("button[aria-selected=true]:not(:disabled)") ?? panel.current?.querySelector<HTMLButtonElement>("button:not(:disabled)"))?.focus();
    const outside = (event: Event) => {
      if (!panel.current?.contains(event.target as Node) && !button.current?.contains(event.target as Node)) close();
    };
    document.addEventListener("pointerdown", outside);
    document.addEventListener("focusin", outside);
    return () => { document.removeEventListener("pointerdown", outside); document.removeEventListener("focusin", outside); };
  }, [open]);
  const keyboard = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Escape") { event.preventDefault(); event.stopPropagation(); close(true); return; }
    if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) return;
    if (event.target instanceof HTMLInputElement && ["Home", "End"].includes(event.key)) return;
    const items = Array.from(panel.current?.querySelectorAll<HTMLButtonElement>(".dropdown-option:not(:disabled)") ?? []);
    if (!items.length) return;
    event.preventDefault();
    const index = items.indexOf(document.activeElement as HTMLButtonElement);
    const next = event.key === "Home" ? 0 : event.key === "End" ? items.length - 1 : event.key === "ArrowDown" ? (index + 1) % items.length : (index - 1 + items.length) % items.length;
    items[next].focus();
  };
  const select = async (next: string) => {
    if (pending) return;
    setPending(true); setError("");
    try { await onSelect(next); close(restoreFocusOnSelect); }
    catch (cause) { setError(String(cause)); }
    finally { setPending(false); }
  };
  return <>
    <button ref={button} type="button" className={`dropdown-trigger ${className}${open ? " is-open" : ""}`} aria-label={label} aria-haspopup={actions ? "menu" : "listbox"} aria-expanded={open} aria-controls={open ? id : undefined} disabled={disabled || pending} onClick={() => { setQuery(""); if (!open) onOpen?.(); setOpen(!open); }} onKeyDown={event => { if (["ArrowDown", "ArrowUp"].includes(event.key)) { event.preventDefault(); if (!open) onOpen?.(); setOpen(true); } }}>
      {trigger}{!actions && <Icon name="chevron" />}
    </button>
    {open && createPortal(<div ref={panel} className="dropdown-panel" style={position} onKeyDown={keyboard}>
      <div className="dropdown-heading">{label}{pending && <span role="status">更新中…</span>}</div>
      {searchable && <input className="dropdown-search" aria-label={`搜索${label}`} placeholder={searchPlaceholder} value={query} onChange={event => setQuery(event.target.value)} />}
      <div id={id} role={actions ? "menu" : "listbox"} aria-label={label} aria-busy={pending} className="dropdown-options">
        {filtered.map((option, index) => <div key={option.value}>
          {option.group && option.group !== filtered[index - 1]?.group && <div className="dropdown-group">{option.group}</div>}
          <button type="button" role={actions ? "menuitem" : "option"} aria-selected={actions ? undefined : value === option.value} disabled={pending || option.disabled} className={`dropdown-option${value === option.value ? " is-selected" : ""}`} onClick={() => void select(option.value)}>
            {option.icon && <span className="dropdown-option-icon">{option.icon}</span>}
            <span className="dropdown-option-copy"><strong>{option.label}</strong>{option.description && <small>{option.description}</small>}</span>
            {!actions && value === option.value && <Icon name="check" />}
          </button>
        </div>)}
        {!filtered.length && <p className="dropdown-empty">{emptyText}</p>}
      </div>
      {error && <p className="dropdown-error" role="alert">{error}</p>}
    </div>, document.body)}
  </>;
}
