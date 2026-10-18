import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import { test } from "node:test";
import ts from "typescript";

const source = readFileSync(new URL("../src/lib/model-form.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 } });
const { parseTokenCount, modelThinkingLevels, parseThinkingMap, parseThinkingLevels } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);

test("token input converts decimal k and m exactly, case insensitively", () => {
  for (const [input, expected] of [["128k", 128000], [" 1.5M ", 1500000], ["32 K", 32000], ["8192", 8192], ["1.001k", 1001], [".5m", 500000], ["0.000001m", 1]]) {
    assert.equal(parseTokenCount(input), expected);
  }
});
test("token input rejects malformed, fractional, zero and unsafe counts", () => {
  for (const input of ["", "0", "-1k", "1e6", "1kb", "1.1", "0.0001k", "Infinity", "9007199254740992", "999999999999999999m"]) {
    assert.throws(() => parseTokenCount(input));
  }
});
test("only selected thinking levels survive saving and reopening", () => {
  const selected = ["low", "medium", "high"];
  const map = parseThinkingMap('{"high":"deep"}', selected);
  assert.equal(map.high, "deep");
  assert.equal(map.low, "low");
  assert.equal(map.off, null);
  assert.equal(map.max, null);
  assert.deepEqual(modelThinkingLevels({ reasoning: true, thinkingLevelMap: map }), selected);
  assert.deepEqual(modelThinkingLevels({ reasoning: false }), []);
  assert.deepEqual(modelThinkingLevels({ reasoning: true }), ["off", "minimal", "low", "medium", "high"]);
  for (const raw of ["{", "[]", "null", '{"high":2}', '{"unknown":"low"}', '{"high":""}']) {
    assert.throws(() => parseThinkingMap(raw, selected));
  }
});

test("manually entered reasoning levels trim whitespace and omit empty additions", () => {
  const levels = parseThinkingLevels([" low ", "medium", "high", "", "  "]);
  assert.deepEqual(levels, ["low", "medium", "high"]);
  const map = parseThinkingMap("", levels);
  assert.deepEqual(modelThinkingLevels({ reasoning: true, thinkingLevelMap: map }), levels);
  assert.deepEqual(parseThinkingLevels([""]), []);
  assert.throws(() => parseThinkingLevels(["low", " low "]));
  assert.throws(() => parseThinkingLevels(["unknown"]));
});
