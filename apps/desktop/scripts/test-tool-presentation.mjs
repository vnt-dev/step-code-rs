import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import { test } from "node:test";
import ts from "typescript";

// The presentation helper is pure data mapping, so exercise it directly.
const source = readFileSync(new URL("../src/lib/tool-presentation.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 },
});
const tools = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);

test("built-in tools get Chinese labels and category icons", () => {
  assert.equal(tools.toolLabel("read"), "读取");
  assert.equal(tools.toolLabel("write"), "写入");
  assert.equal(tools.toolLabel("edit"), "编辑");
  assert.equal(tools.toolLabel("bash"), "终端");
  assert.equal(tools.toolLabel("powershell"), "终端");
  assert.equal(tools.toolLabel("search_web"), "联网");
  assert.equal(tools.toolIcon("edit"), "pencil");
  assert.equal(tools.toolIcon("bash"), "terminal");
  assert.equal(tools.toolIcon("read"), "file");
});

test("unknown and MCP tools fall back to readable names", () => {
  assert.equal(tools.toolLabel("custom_tool"), "custom_tool");
  assert.equal(tools.toolLabel("mcp__lark__docx_create"), "docx_create");
  assert.equal(tools.toolIcon("mcp__lark__docx_create"), "plugin");
});

test("search tools lead with the pattern and keep the root as detail", () => {
  assert.deepEqual(tools.toolTarget("grep", { pattern: "TODO", path: "crates" }), { title: "TODO", detail: "crates" });
  assert.deepEqual(tools.toolTarget("find", { pattern: "*.ts" }), { title: "*.ts" });
  assert.deepEqual(tools.toolTarget("search_web", { query: "tauri v2 sidebar" }), { title: "tauri v2 sidebar" });
});

test("empty arguments stay empty instead of rendering placeholders", () => {
  assert.deepEqual(tools.toolTarget("read", {}), { title: "" });
  assert.deepEqual(tools.toolTarget("ls", {}), { title: "." });
  assert.deepEqual(tools.toolTarget("edit", { path: "" }), { title: "" });
  assert.deepEqual(tools.toolTarget("mcp__server__thing", {}), { title: "" });
});
