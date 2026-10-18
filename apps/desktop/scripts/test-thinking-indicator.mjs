import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import assert from "node:assert/strict";
import { test } from "node:test";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ts from "typescript";

const require = createRequire(import.meta.url);
const compilerOptions = { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022, jsx: ts.JsxEmit.ReactJSX };
const react = pathToFileURL(require.resolve("react")).href;
const jsxRuntime = pathToFileURL(require.resolve("react/jsx-runtime")).href;

function transpile(relativePath) {
  const source = readFileSync(new URL(`../${relativePath}`, import.meta.url), "utf8");
  return ts.transpileModule(source, { compilerOptions }).outputText;
}

const moduleUrl = code => `data:text/javascript;base64,${Buffer.from(code).toString("base64")}`;
const messages = moduleUrl(transpile("src/lib/messages.ts"));
const useElapsed = moduleUrl(transpile("src/lib/useElapsed.ts").replaceAll('from "react"', `from "${react}"`));
const indicator = moduleUrl(transpile("src/components/ThinkingIndicator.tsx")
  .replaceAll('from "react/jsx-runtime"', `from "${jsxRuntime}"`)
  .replaceAll('from "../lib/messages"', `from "${messages}"`)
  .replaceAll('from "../lib/useElapsed"', `from "${useElapsed}"`));
const { default: ThinkingIndicator } = await import(indicator);

function render(now, startedAt) {
  const originalNow = Date.now;
  Date.now = () => now;
  try {
    return renderToStaticMarkup(React.createElement(ThinkingIndicator, { startedAt }));
  } finally { Date.now = originalNow; }
}

test("the thinking timer keeps counting from the run start after a remount", () => {
  assert.match(render(45000, 12000), /33\.0s/);
  assert.match(render(45200, 12000), /33\.2s/);
});

test("without a run start the timer falls back to the mount moment", () => {
  assert.match(render(45000, undefined), /0\.0s/);
});
