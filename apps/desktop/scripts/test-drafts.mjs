import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import { test } from "node:test";
import ts from "typescript";

const moduleUrl = source => {
  const { outputText } = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 } });
  return `data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`;
};
const projectsUrl = moduleUrl(readFileSync(new URL("../src/lib/projects.ts", import.meta.url), "utf8"));
const imagesUrl = moduleUrl(readFileSync(new URL("../src/lib/images.ts", import.meta.url), "utf8"));
const source = readFileSync(new URL("../src/lib/drafts.ts", import.meta.url), "utf8").replace('"./projects"', JSON.stringify(projectsUrl)).replace('"./images"', JSON.stringify(imagesUrl));
const { draftKey, parseDrafts, defaultDraft, supportedDraftThinkingLevel } = await import(moduleUrl(source));

test("selected skills survive draft restoration with their file identity", () => {
  const skill = { name: "example", description: "Example instructions", filePath: "C:/skills/example/SKILL.md" };
  const draft = { ...defaultDraft(null), skills: [skill] };
  assert.deepEqual(parseDrafts(JSON.stringify({ "project:a": draft }))["project:a"].skills, [skill]);
  const invalid = { ...draft, skills: [null, { name: "missing-path" }, skill] };
  assert.deepEqual(parseDrafts(JSON.stringify({ "project:a": invalid }))["project:a"].skills, [skill]);
});

test("draft storage round-trip isolates projects and preserves all choices", () => {
  const first = { text: "未发送的需求\n下一行", provider: "test", modelId: "one", thinkingLevel: "high", approvalMode: "autopilot", planActive: true };
  const second = { ...first, text: "另一个项目", modelId: "two", planActive: false };
  const saved = parseDrafts(JSON.stringify({ [draftKey("C:/work/app")]: first, [draftKey("D:/work/app")]: second, "session:old": first }));
  assert.deepEqual(saved[draftKey("c:\\work\\app\\")], first);
  assert.deepEqual(saved[draftKey("D:/work/app")], second);
  assert.equal(saved["session:old"], undefined);
  // Drafts stored before the plan switch existed restore with it off.
  const { planActive, ...legacy } = first;
  assert.deepEqual(parseDrafts(JSON.stringify({ [draftKey("C:/work/app")]: legacy }))[draftKey("C:/work/app")], { ...legacy, planActive: false });
});
test("legacy permission modes migrate to confirmation without losing draft text", () => {
  const draft = { ...defaultDraft(null), text: "待发送的需求", approvalMode: "read-only" };
  assert.deepEqual(parseDrafts(JSON.stringify({ "project:a": draft }))["project:a"], { ...draft, approvalMode: "ask" });
});

test("damaged local storage cannot break startup", () => {
  for (const raw of [null, "{", "null", "[]", '{"project:x":null}', '{"project:x":{"text":1}}']) {
    assert.deepEqual(parseDrafts(raw), {});
  }
});
test("first draft uses configured defaults without an active session", () => {
  const models = [{ provider: "test", id: "first", configured: true, reasoning: false, supportedThinkingLevels: ["off"] }, { provider: "test", id: "chosen", configured: true, reasoning: true, supportedThinkingLevels: ["low", "medium", "high"] }];
  const draft = defaultDraft({ models, currentModel: null, thinkingLevel: "off", approvalMode: "ask", settings: { defaultModel: "chosen", defaultProvider: "test", defaultThinkingLevel: "high" } });
  assert.equal(draft.modelId, "chosen");
  assert.equal(draft.thinkingLevel, "high");
  assert.equal(defaultDraft(null).text, "");
});

test("restored drafts and model switches use a supported thinking level", () => {
  const step = { reasoning: true, supportedThinkingLevels: ["low", "medium", "high"] };
  for (const unsupported of ["off", "minimal", "xhigh", "max"]) {
    assert.equal(supportedDraftThinkingLevel(step, unsupported), "medium");
  }
  assert.equal(supportedDraftThinkingLevel(step, "high"), "high");
  assert.equal(supportedDraftThinkingLevel({ reasoning: false, supportedThinkingLevels: ["off"] }, "high"), "off");
  assert.equal(supportedDraftThinkingLevel({ reasoning: true, supportedThinkingLevels: ["high"] }, "medium"), "high");
  const draft = defaultDraft({ models: [{ ...step, configured: true, provider: "step", id: "step" }], settings: { defaultThinkingLevel: "max" } });
  assert.equal(draft.thinkingLevel, "medium");
});


test("image drafts survive reload and invalid attachments are discarded", () => {
  const image = { type: "image", id: "image-1", name: "test.png", mimeType: "image/png", data: "aGVsbG8=" };
  const draft = { ...defaultDraft(null), images: [image] };
  assert.deepEqual(parseDrafts(JSON.stringify({ "project:a": draft }))["project:a"], draft);
  const restored = parseDrafts(JSON.stringify({ "project:a": { ...draft, images: [image, { ...image, mimeType: "text/html" }, { ...image, data: "invalid!" }] } }));
  assert.deepEqual(restored["project:a"].images, [image]);
});


test("image file reading preserves MIME and raw base64 payload", async () => {
  const { readImage, imageSource } = await import(imagesUrl);
  const originalReader = globalThis.FileReader;
  globalThis.FileReader = class {
    readAsDataURL(file) {
      void file.arrayBuffer().then(bytes => {
        this.result = `data:${file.type};base64,${Buffer.from(bytes).toString("base64")}`;
        this.onload();
      });
    }
  };
  try {
    const file = new File([Buffer.from([137, 80, 78, 71])], "shot.png", { type: "image/png" });
    const image = await readImage(file);
    assert.equal(image.name, "shot.png");
    assert.equal(image.mimeType, "image/png");
    assert.equal(image.data, "iVBORw==");
    assert.equal(imageSource(image), "data:image/png;base64,iVBORw==");
    assert.ok(image.id);
    assert.equal((await readImage(file, true)).pasted, true);
    assert.equal(image.pasted, false);
  } finally { globalThis.FileReader = originalReader; }
});

test("downloaded image names count up and use the mime extension", async () => {
  const { imageFileName } = await import(imagesUrl);
  const extensions = { "image/png": "png", "image/jpeg": "jpg", "image/gif": "gif", "image/webp": "webp" };
  assert.equal(imageFileName({ type: "image", data: "aGk=", mimeType: "image/png" }, 0), "图片-1.png");
  for (const [mimeType, extension] of Object.entries(extensions)) {
    assert.equal(imageFileName({ type: "image", data: "aGk=", mimeType }, 2), `图片-3.${extension}`);
  }
  assert.equal(imageFileName({ type: "image", data: "aGk=", mimeType: "image/bmp" }, 0), "图片-1.png");
});

test("unsupported, empty and oversized images are rejected before reading", async () => {
  const { readImage, maxImageBytes } = await import(imagesUrl);
  await assert.rejects(readImage(new File(["svg"], "x.svg", { type: "image/svg+xml" })));
  await assert.rejects(readImage(new File([], "empty.png", { type: "image/png" })));
  await assert.rejects(readImage(new File([new Uint8Array(maxImageBytes + 1)], "large.png", { type: "image/png" })));
});
