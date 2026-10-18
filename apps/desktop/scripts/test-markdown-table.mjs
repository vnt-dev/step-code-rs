import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import assert from "node:assert/strict";
import { test } from "node:test";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ts from "typescript";

const require = createRequire(import.meta.url);
const pkgUrl = name => pathToFileURL(require.resolve(name)).href;
const compilerOptions = { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022, jsx: ts.JsxEmit.ReactJSX };
const transpile = relativePath => ts.transpileModule(readFileSync(new URL(`../${relativePath}`, import.meta.url), "utf8"), { compilerOptions }).outputText;
const moduleUrl = code => `data:text/javascript;base64,${Buffer.from(code).toString("base64")}`;

const react = pkgUrl("react");
const jsxRuntime = pkgUrl("react/jsx-runtime");
/** Load a local module, pointing every import at the URL of the module it resolves to. */
const local = (relativePath, extra = {}) => moduleUrl([...Object.entries({
  '"react/jsx-runtime"': `"${jsxRuntime}"`,
  '"react"': `"${react}"`,
  ...extra,
})].reduce((code, [from, to]) => code.replaceAll(from, to), transpile(relativePath)));

const icon = local("src/components/Icon.tsx");
const presentation = local("src/lib/tool-presentation.ts");
const api = moduleUrl(`export const api = { openExternalUrl: () => Promise.resolve() };`);
const toolCard = local("src/components/ToolCard.tsx", {
  '"react-markdown"': `"${pkgUrl("react-markdown")}"`,
  '"remark-gfm"': `"${pkgUrl("remark-gfm")}"`,
  '"rehype-highlight"': `"${pkgUrl("rehype-highlight")}"`,
  '"../api"': `"${api}"`,
  '"./Icon"': `"${icon}"`,
  '"../lib/tool-presentation"': `"${presentation}"`,
});
const { Markdown } = await import(toolCard);

const render = text => renderToStaticMarkup(React.createElement(Markdown, { text }));
const wideTable = [
  "| 文件 | 表 | 修改 |",
  "| --- | --- | --- |",
  "| dal/dataobject/rework/ReworkOrderDetailDO.java | dms_supply_rework_order_detail | 增加 @IgnoreTenantTable |",
  "| dal/dataobject/rework/ReworkOrderRecordDO.java | dms_supply_rework_order_record | 增加 @IgnoreTenantTable |",
].join("\n");

test("a markdown table renders inside its own scroll container", () => {
  const html = render(wideTable);
  // The wrapper carries the horizontal scroll; the table itself must survive as a real table.
  assert.match(html, /<div class="markdown-table"><table>/);
  assert.match(html, /<thead>/);
  assert.match(html, /<tbody>/);
  assert.match(html, /<th>文件<\/th>/);
  assert.match(html, /<td>dal\/dataobject\/rework\/ReworkOrderDetailDO\.java<\/td>/);
});

test("markdown without a table renders no scroll container", () => {
  const html = render("我先确认渲染层级，**再动手改**。");
  assert.doesNotMatch(html, /markdown-table/);
  assert.match(html, /<strong>再动手改<\/strong>/);
});
