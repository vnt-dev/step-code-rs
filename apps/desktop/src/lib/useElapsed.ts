import { useEffect, useState } from "react";

/**
 * Milliseconds elapsed since `active` became true, ticking every
 * `intervalMs`. Resets to zero whenever `active` turns false, so a component
 * can mount with the flag already true and still measure from that moment.
 * When `startedAt` is supplied (Unix milliseconds), remounts keep counting
 * from that original timestamp instead of the component mount time.
 */
export function useElapsed(active: boolean, intervalMs = 100, startedAt?: number): number {
  const [elapsed, setElapsed] = useState(() => active && startedAt !== undefined ? Math.max(0, Date.now() - startedAt) : 0);
  useEffect(() => {
    if (!active) {
      setElapsed(0);
      return;
    }
    const mountedAt = performance.now();
    const measure = () => Math.max(0, startedAt === undefined ? performance.now() - mountedAt : Date.now() - startedAt);
    setElapsed(measure());
    const timer = window.setInterval(() => setElapsed(measure()), intervalMs);
    return () => window.clearInterval(timer);
  }, [active, intervalMs, startedAt]);
  return elapsed;
}

/**
 * Current time in Unix milliseconds, refreshed every `intervalMs` while
 * `active`, so a value derived from it keeps ticking without the component
 * owning a timer of its own.
 */
export function useNow(active: boolean, intervalMs = 500): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), intervalMs);
    return () => window.clearInterval(timer);
  }, [active, intervalMs]);
  return now;
}
