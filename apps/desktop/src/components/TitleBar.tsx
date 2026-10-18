import { useEffect, useState } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";

// The QA harness renders the UI in a plain browser without Tauri internals;
// there the bar still draws but the controls stay inert.
const appWindow = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window ? getCurrentWindow() : null;

/** Frameless-window caption bar: drag surface on the left, window controls on the right. */
export default function TitleBar() {
  const [maximized, setMaximized] = useState(false);
  useEffect(() => {
    if (!appWindow) return;
    let disposed = false;
    const sync = () => {
      void appWindow.isMaximized().then(value => { if (!disposed) setMaximized(value); }).catch(() => undefined);
    };
    sync();
    const subscription = appWindow.onResized(sync);
    return () => { disposed = true; void subscription.then(dispose => dispose()).catch(() => undefined); };
  }, []);

  return <header className="titlebar" data-tauri-drag-region>
    <span className="titlebar-title" data-tauri-drag-region>
      <img className="titlebar-mark" src="/app-icon.png" alt=""/>
      Step Code
    </span>
    <div className="titlebar-drag" data-tauri-drag-region/>
    <div className="titlebar-controls">
      <button aria-label="最小化" disabled={!appWindow} onClick={() => void appWindow?.minimize()}>
        <svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="1"><path d="M0 5h10"/></svg>
      </button>
      <button aria-label={maximized ? "向下还原" : "最大化"} disabled={!appWindow} onClick={() => void appWindow?.toggleMaximize()}>
        {maximized
          ? <svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="1"><path d="M2.5.5h7v7"/><rect x=".5" y="2.5" width="7" height="7"/></svg>
          : <svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="1"><rect x=".5" y=".5" width="9" height="9"/></svg>}
      </button>
      <button className="titlebar-close" aria-label="关闭" disabled={!appWindow} onClick={() => void appWindow?.close()}>
        <svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="1"><path d="m0 0 10 10M10 0 0 10"/></svg>
      </button>
    </div>
  </header>;
}
