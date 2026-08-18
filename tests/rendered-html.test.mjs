import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const outputRoot = new URL("../out/", import.meta.url);

test("exports the finished homepage", async () => {
  const html = await readFile(new URL("index.html", outputRoot), "utf8");

  assert.match(html, /<html lang="zh-CN"/);
  assert.match(html, /<title>递归之外｜AI 与计算机科学笔记<\/title>/);
  assert.match(html, /在代码与智能之间/);
  assert.match(html, /本期精选/);
  assert.match(html, /文章归档/);
  assert.match(html, /关于本站/);
  assert.match(html, /搜索文章/);
  assert.doesNotMatch(html, /Your site is taking shape|codex-preview/i);
});

test("exports Pages-compatible assets and fallback", async () => {
  const html = await readFile(new URL("index.html", outputRoot), "utf8");

  assert.match(html, /\/_next\/static\/(?:css|chunks)\/[^"']+\.css/);
  assert.match(html, /\/_next\/static\/chunks\/[^"']+\.js/);
  await access(new URL("404.html", outputRoot));
  await access(new URL("icon.svg", outputRoot));
  await access(new URL(".nojekyll", outputRoot));
});
