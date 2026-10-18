import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import assert from "node:assert/strict";
import { test } from "node:test";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ts from "typescript";

const require = createRequire(import.meta.url);
const source = readFileSync(new URL("../src/lib/useElapsed.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 } });
const code = outputText.replace('from "react"', `from "${pathToFileURL(require.resolve("react")).href}"`);
const { useElapsed } = await import(`data:text/javascript;base64,${Buffer.from(code).toString("base64")}`);

function mount(now, startedAt, active = true) {
  const originalNow = Date.now;
  Date.now = () => now;
  try {
    function Counter() { return React.createElement("span", null, useElapsed(active, 100, startedAt)); }
    return renderToStaticMarkup(React.createElement(Counter));
  } finally { Date.now = originalNow; }
}

test("remounting the approval counter continues elapsed time instead of resetting it", () => {
  assert.equal(mount(23000, 1000), "<span>22000</span>");
  assert.equal(mount(68000, 1000), "<span>67000</span>");
});

test("a new queued request uses its own starting timestamp", () => {
  assert.equal(mount(68000, 5000), "<span>63000</span>");
});

test("inactive and future timestamps never display negative elapsed time", () => {
  assert.equal(mount(23000, 1000, false), "<span>0</span>");
  assert.equal(mount(23000, 24000), "<span>0</span>");
});
