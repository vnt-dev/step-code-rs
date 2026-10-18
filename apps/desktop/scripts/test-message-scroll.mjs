import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import { test } from "node:test";
import ts from "typescript";

const source = readFileSync(new URL("../src/lib/message-scroll.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 } });
const { observeMessageScroll } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);

function mount() {
  let observer;
  globalThis.ResizeObserver = class {
    targets = [];
    disconnected = false;
    constructor(callback) { this.resize = callback; observer = this; }
    observe(target) { this.targets.push(target); }
    disconnect() { this.disconnected = true; }
  };
  const viewport = new EventTarget();
  viewport.scrollHeight = 1200;
  viewport.clientHeight = 400;
  let top = 0;
  Object.defineProperty(viewport, "scrollTop", {
    get: () => top,
    set: value => { top = Math.max(0, Math.min(value, viewport.scrollHeight - viewport.clientHeight)); },
  });
  const content = {};
  const states = [];
  const scroll = observeMessageScroll(viewport, content, pinned => states.push(pinned));
  return { viewport, content, states, scroll, observer, emit: () => viewport.dispatchEvent(new Event("scroll")) };
}

test("opening and layout changes reach the real bottom including padding", () => {
  const { viewport, content, observer } = mount();
  assert.equal(viewport.scrollTop, 800);
  assert.deepEqual(observer.targets, [content, viewport]);
  // A tool result or delayed image changes height without a new message.
  viewport.scrollHeight += 600;
  observer.resize();
  assert.equal(viewport.scrollTop, 1400);
  // A taller composer reduces the transcript viewport.
  viewport.clientHeight = 200;
  observer.resize();
  assert.equal(viewport.scrollTop, 1600);
});

test("a layout-generated scroll event does not disable following after a large chunk", () => {
  const { viewport, states, scroll, emit } = mount();
  viewport.scrollHeight += 800;
  emit();
  scroll.follow();
  assert.equal(viewport.scrollTop, 1600);
  assert.deepEqual(states, []);
});

test("reading older messages suspends following across renders and resizes", () => {
  const { viewport, states, scroll, observer, emit } = mount();
  viewport.scrollTop = 400;
  emit();
  assert.deepEqual(states, [false]);
  viewport.scrollHeight += 600;
  scroll.follow();
  observer.resize();
  assert.equal(viewport.scrollTop, 400);
  // Moving down through history should not resume until close to the bottom.
  viewport.scrollTop = 700;
  emit();
  observer.resize();
  assert.equal(viewport.scrollTop, 700);
  viewport.scrollTop = 1350;
  emit();
  viewport.scrollHeight += 300;
  observer.resize();
  assert.equal(viewport.scrollTop, 1700);
  assert.deepEqual(states, [false, true]);
});

test("jump to latest resumes immediately and stays pinned during streaming", () => {
  const { viewport, states, scroll, observer, emit } = mount();
  viewport.scrollTop = 100;
  emit();
  scroll.jumpToLatest();
  assert.equal(viewport.scrollTop, 800);
  emit();
  viewport.scrollHeight += 700;
  observer.resize();
  assert.equal(viewport.scrollTop, 1500);
  assert.deepEqual(states, [false, true]);
});

test("unmount disconnects observation and scroll handling", () => {
  const { viewport, states, scroll, observer, emit } = mount();
  scroll.dispose();
  assert.equal(observer.disconnected, true);
  viewport.scrollTop = 0;
  emit();
  assert.deepEqual(states, []);
});
