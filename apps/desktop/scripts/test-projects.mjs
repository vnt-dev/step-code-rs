import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import { test } from "node:test";
import ts from "typescript";

const source = readFileSync(new URL("../src/lib/projects.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 } });
const { groupProjects, projectKey, relativeTime } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);
const session = (id, cwd, name, modifiedAt = 1) => ({ id, cwd, name, modifiedAt });

test("same folder names in different locations remain separate projects", () => {
  const groups = groupProjects([], [session("a", "C:\\work\\app", "first"), session("b", "D:\\work\\app", "second")], "");
  assert.equal(groups.length, 2);
  assert.deepEqual(groups.map(g => g.sessions.length), [1, 1]);
});
test("registered empty projects survive and Windows path variants deduplicate", () => {
  const groups = groupProjects(["C:\\Work\\app", "D:\\empty"], [session("a", "c:/work/app/", "first")], "");
  assert.equal(groups.length, 2);
  assert.equal(groups[0].sessions.length, 1);
  assert.equal(groups[1].sessions.length, 0);
  assert.equal(projectKey("\\\\?\\C:\\Work\\app"), projectKey("c:/work/app"));
  assert.notEqual(projectKey("/Work/app"), projectKey("/work/app"));
});
test("search finds sessions and project folders, newest sessions come first", () => {
  const sessions = [session("a", "C:\\app", "older", 1), session("b", "C:\\app", "fix sidebar", 2)];
  assert.deepEqual(groupProjects([], sessions, "")[0].sessions.map(s => s.id), ["b", "a"]);
  assert.deepEqual(groupProjects([], sessions, "sidebar")[0].sessions.map(s => s.id), ["b"]);
  assert.equal(groupProjects(["D:\\empty"], sessions, "empty").length, 1);
  assert.equal(groupProjects([], sessions, "missing").length, 0);
});
test("relative timestamps handle fresh and future timestamps", () => {
  const now = 200000000;
  assert.equal(relativeTime(now + 1000, now), "刚刚");
  assert.equal(relativeTime(now - 13 * 3600000, now), "13小时");
});
test("removed projects stay hidden despite stored sessions, search and path variants", () => {
  const sessions = [session("a", "C:\\Work\\app", "old conversation"), session("b", "D:\\other", "kept")];
  for (const query of ["", "app", "old conversation"]) {
    const groups = groupProjects(["C:\\Work\\app", "D:\\other"], sessions, query, ["c:/work/app/"]);
    assert.equal(groups.some(group => group.key === projectKey("C:/work/app")), false);
  }
  assert.equal(sessions.length, 2);
  assert.equal(groupProjects(["C:/Work/app"], sessions, "", []).find(group => group.key === projectKey("C:/work/app")).sessions[0].id, "a");
});
