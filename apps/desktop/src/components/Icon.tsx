import type { CSSProperties } from "react";
const paths = {
  compose: "M12 4H6a3 3 0 0 0-3 3v11a3 3 0 0 0 3 3h11a3 3 0 0 0 3-3v-6 M16 3a2.1 2.1 0 0 1 3 3l-8 8-4 1 1-4 8-8Z",
  skill: "m4 20 12-12 4 4L8 24 M13 11l4 4 M5 3v4M3 5h4M17 1v4M15 3h4M20 18v4M18 20h4",
  chart: "M4 3v18h17M8 16v-5M13 16V7M18 16V4",
  image: "M3 3h18v18H3Z M3 17l6-6 4 4 3-3 5 5 M15 7h.01",
  folder: "M3 7V5a1 1 0 0 1 1-1h5l2 3h9a1 1 0 0 1 1 1v2M3 7h16a1 1 0 0 1 1 1l-2 11H3a1 1 0 0 1-1-1l1-11Z",
  plus: "M12 5v14M5 12h14", chevron: "m8 10 4 4 4-4", check: "m5 12 4 4L19 6",
  shield: "M12 3 4 6v6c0 5 8 9 8 9s8-4 8-9V6l-8-3Z M9 12l2 2 4-4",
  brain: "M12 5a3 3 0 0 0-6-1 3 3 0 0 0-2 5 4 4 0 0 0 0 7 3 3 0 0 0 5 4l3-1V5Zm0 0a3 3 0 0 1 6-1 3 3 0 0 1 2 5 4 4 0 0 1 0 7 3 3 0 0 1-5 4l-3-1 M7 9l2 2M17 9l-2 2M7 16l2-2M17 16l-2-2",
  model: "m12 3 9 5-9 5-9-5 9-5Zm-9 9 9 5 9-5M3 16l9 5 9-5",
  refresh: "M20 7v5h-5M4 17v-5h5M6 7a7 7 0 0 1 12-1l2 3M4 15l2 3a7 7 0 0 0 12-1",
  undo: "m9 4-5 5 5 5M4 9h10a6 6 0 0 1 0 12h-3",
  compact: "M8 3v5H3M16 3v5h5M8 21v-5H3M16 21v-5h5M3 3l5 5M21 3l-5 5M3 21l5-5M21 21l-5-5",
  settings: "M4 7h16M4 17h16M8 4v6M16 14v6",
  arrow: "M12 19V5m-6 6 6-6 6 6", stop: "M6 6h12v12H6Z", queue: "M4 6h16M4 12h10M4 18h10m3-5 4 4-4 4",
  download: "M12 3v12m-4-4 4 4 4-4M4 21h16",
  chevronLeft: "m14 6-6 6 6 6", chevronRight: "m10 6 6 6-6 6",
  alert: "M12 4 2.5 20h19L12 4Z M12 10v4.5m0 3h.01",
  plan: "M8 4h8a1 1 0 0 1 1 1v1h1a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h1V5a1 1 0 0 1 1-1Z M9 5v2h6V5 M9 12h6M9 16h4",
  bulb: "M9 18h6M10 21h4M12 3a6 6 0 0 1 3.6 10.8c-.6.5-1 1.1-1.2 1.7h-4.8c-.2-.6-.6-1.2-1.2-1.7A6 6 0 0 1 12 3Z",
  closeCircle: "M12 4a8 8 0 1 0 0 16 8 8 0 0 0 0-16Z M9.5 9.5l5 5M14.5 9.5l-5 5",
  pencil: "M4 20h4L20 8l-4-4L4 16v4Z M14.5 5.5l4 4",
  file: "M6 3h8l4 4v14H6V3Z M14 3v4h4",
  filePlus: "M6 3h8l4 4v14H6V3Z M14 3v4h4 M12 12v5M9.5 14.5h5",
  terminal: "M4 5h16v14H4Z M8 10l2.5 2L8 14 M13.5 14.5h3.5",
  search: "M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14Z M16.2 16.2 21 21",
  globe: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18Z M3 12h18 M12 3c2.6 2.6 3.6 5.7 3.6 9s-1 6.4-3.6 9c-2.6-2.6-3.6-5.7-3.6-9S9.4 5.6 12 3Z",
  plugin: "M12 3 4 7v10l8 4 8-4V7l-8-4Z M4 7l8 4 8-4 M12 11v10",
  bolt: "M13 2 4.5 13.5H10L9.5 22 19 9.5h-5.5L13 2Z",
} as const;
export type IconName = keyof typeof paths;
export default function Icon({ name, className, style }: { name: keyof typeof paths; className?: string; style?: CSSProperties }) {
  return <svg aria-hidden="true" className={className} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" style={style}><path d={paths[name]} /></svg>;
}
