import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';
import { test } from 'node:test';
import ts from 'typescript';
const source=ts.transpileModule(readFileSync(new URL('../src/lib/request-metrics.ts',import.meta.url),'utf8'),{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ES2022}}).outputText;
const {smoothPaths,localDayBoundaries,modelColor}=await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
test('monotone curves preserve endpoints, separate gaps and bounded controls',()=>{
  const points=[{x:0,y:100},{x:1,y:0},{x:2,y:90},null,{x:4,y:50},{x:5,y:55}];
  const paths=smoothPaths(points); assert.equal(paths.length,2); assert.ok(paths[0].startsWith('M 0 100')); assert.ok(paths[1].startsWith('M 4 50'));
  const commands=paths[0].split(' C ').slice(1);
  commands.forEach((command,i)=>{const numbers=command.match(/-?\d+(?:\.\d+)?/g).map(Number);const low=Math.min(points[i].y,points[i+1].y), high=Math.max(points[i].y,points[i+1].y); assert.ok(numbers[1]>=low&&numbers[1]<=high);assert.ok(numbers[3]>=low&&numbers[3]<=high);});
  assert.deepEqual(smoothPaths([null,null]),[]); assert.deepEqual(smoothPaths([{x:1,y:4}]),['M 1 4']);
});
test('day boundaries use calendar dates and model colors stay independent of sort order',()=>{
  const dates=localDayBoundaries(new Date(2026,9,14,15)); assert.equal(dates.length,8);assert.equal(new Date(dates[0]).getDate(),8);assert.equal(new Date(dates[7]).getDate(),15);assert.ok(dates.every(ms=>new Date(ms).getHours()===0));
  const color=modelColor('provider/model'); ['other/one','other/two'].forEach(modelColor);assert.equal(modelColor('provider/model'),color);
});

test('calendar boundaries preserve 25-hour daylight-saving days',()=>{
  const original=process.env.TZ;
  try {
    process.env.TZ='America/New_York';
    const boundaries=localDayBoundaries(new Date(2026,10,3,12));
    assert.ok(boundaries.slice(1).some((value,i)=>value-boundaries[i]===25*3600000));
    assert.ok(boundaries.every(value=>new Date(value).getHours()===0));
  } finally {if(original===undefined)delete process.env.TZ;else process.env.TZ=original;}
});
