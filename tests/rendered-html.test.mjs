import assert from "node:assert/strict";
import { access, readFile, readdir } from "node:fs/promises";
import test from "node:test";

const outputRoot = new URL("../out/", import.meta.url);
const articlesRoot = new URL("../content/articles/", import.meta.url);
const articleSlugs = (await readdir(articlesRoot))
  .filter((filename) => filename.endsWith(".md"))
  .map((filename) => filename.slice(0, -3))
  .sort();

test("exports the finished homepage", async () => {
  const html = await readFile(new URL("index.html", outputRoot), "utf8");

  assert.match(html, /<html lang="zh-CN"/);
  assert.match(html, /<title>递归之外｜AI 与计算机科学笔记<\/title>/);
  assert.match(html, /在代码与智能之间/);
  assert.match(html, /本期精选/);
  assert.match(html, /文章归档/);
  assert.match(html, /关于本站/);
  assert.match(html, /搜索文章/);
  for (const slug of articleSlugs) {
    assert.match(html, new RegExp(`href="/articles/${slug}/"`));
  }
  assert.doesNotMatch(html, /Your site is taking shape|codex-preview/i);
});

test("exports a readable page for every article", async () => {
  for (const slug of articleSlugs) {
    const html = await readFile(new URL(`articles/${slug}/index.html`, outputRoot), "utf8");

    assert.match(html, /返回文章归档/);
    assert.match(html, /CONTENTS/);
    assert.match(html, /继续阅读/);
    assert.match(html, /<div class="article-markdown"><p>/);
    assert.match(html, /<h2 id="section-1">/);
    assert.match(html, /END \/ /);
    assert.doesNotMatch(html, /Your site is taking shape|codex-preview/i);
  }
});

test("renders inline and display TeX as KaTeX HTML", async () => {
  const html = await readFile(
    new URL("articles/attention-to-reasoning/index.html", outputRoot),
    "utf8",
  );

  assert.match(html, /class="katex"/);
  assert.match(html, /class="katex-display"/);
  assert.match(html, /annotation encoding="application\/x-tex">\\operatorname{Attention}/);
  assert.doesNotMatch(html, /katex-error/);
});

test("exports Pages-compatible assets and fallback", async () => {
  const html = await readFile(new URL("index.html", outputRoot), "utf8");

  assert.match(html, /\/_next\/static\/(?:css|chunks)\/[^"']+\.css/);
  assert.match(html, /\/_next\/static\/chunks\/[^"']+\.js/);
  await access(new URL("404.html", outputRoot));
  await access(new URL("icon.svg", outputRoot));
  await access(new URL(".nojekyll", outputRoot));
});
