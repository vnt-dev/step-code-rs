//! How a tool call summarizes itself on a collapsed transcript row: a Chinese
//! label, an icon, and the single argument worth showing (file, command, query).

import type { IconName } from "../components/Icon";

interface ToolStyle {
  label: string;
  icon: IconName;
}

const TOOL_STYLES: Record<string, ToolStyle> = {
  read: { label: "读取", icon: "file" },
  write: { label: "写入", icon: "filePlus" },
  edit: { label: "编辑", icon: "pencil" },
  bash: { label: "终端", icon: "terminal" },
  powershell: { label: "终端", icon: "terminal" },
  ls: { label: "目录", icon: "folder" },
  find: { label: "查找", icon: "search" },
  grep: { label: "搜索", icon: "search" },
  find_tools: { label: "查找", icon: "search" },
  search_web: { label: "联网", icon: "globe" },
  enter_plan_mode: { label: "计划", icon: "plan" },
  exit_plan_mode: { label: "计划", icon: "plan" },
};

/** Short label naming what the tool does, in the transcript's language. */
export function toolLabel(name: string): string {
  const style = TOOL_STYLES[name];
  if (style) return style.label;
  // Remote MCP tools arrive as `mcp__<server>__<tool>`; the leaf name reads better.
  const leaf = name.split("__").pop();
  return leaf && leaf !== name ? leaf : name;
}

/** Icon matching the tool's category. */
export function toolIcon(name: string): IconName {
  return TOOL_STYLES[name]?.icon ?? "plugin";
}

export interface ToolTarget {
  /** The file name, command or query the call is about. */
  title: string;
  /** Where the title lives: parent directory or search root. */
  detail?: string;
}

/** The one argument that identifies a call, split into title and context. */
export function toolTarget(name: string, args: Record<string, unknown>): ToolTarget {
  const text = (value: unknown): string =>
    typeof value === "string" ? value : value == null ? "" : JSON.stringify(value);
  const path = text(args.path ?? args.file_path);
  const command = text(args.command);
  const pattern = text(args.pattern);
  const query = text(args.query);
  switch (name) {
    case "read":
    case "write":
    case "edit":
      return path ? splitPath(path) : { title: "" };
    case "bash":
    case "powershell":
      return { title: command.replace(/\s+/g, " ").trim() };
    case "ls":
      return path ? splitPath(path) : { title: "." };
    case "find":
    case "grep":
      if (pattern) return path ? { title: pattern, detail: path } : { title: pattern };
      return path ? splitPath(path) : { title: "" };
    case "search_web":
      return { title: query };
    default: {
      // Unknown (MCP) tools: the first string argument is the best guess.
      const first = Object.values(args).find(value => typeof value === "string");
      return { title: typeof first === "string" ? first : "" };
    }
  }
}

function splitPath(path: string): ToolTarget {
  const normalized = path.replace(/\\/g, "/").replace(/\/+$/, "");
  const slash = normalized.lastIndexOf("/");
  if (slash < 0) return { title: normalized };
  return { title: normalized.slice(slash + 1), detail: `${normalized.slice(0, slash + 1)}` };
}
