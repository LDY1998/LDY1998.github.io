import "server-only";

import { readdirSync, readFileSync } from "node:fs";
import { basename, extname, join } from "node:path";

import matter from "gray-matter";
import { katex } from "@mdit/plugin-katex";
import MarkdownIt from "markdown-it";

import type { Article, ArticleHeading, ArticleSummary } from "@/lib/article-types";

const articlesDirectory = join(process.cwd(), "content", "articles");
const safeSlugPattern = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

type LoadedArticle = Omit<Article, "index"> & {
  sortDate: string;
};

function requiredString(value: unknown, field: string, filename: string) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new Error(`${filename}: front matter field "${field}" must be a non-empty string.`);
  }

  return value.trim();
}

function optionalString(value: unknown, field: string, filename: string) {
  if (value === undefined) return undefined;
  return requiredString(value, field, filename);
}

function parseDate(value: unknown, filename: string) {
  const raw = value instanceof Date
    ? value.toISOString().slice(0, 10)
    : requiredString(value, "date", filename);
  const normalized = raw.replaceAll(".", "-");

  if (!/^\d{4}-\d{2}-\d{2}$/.test(normalized) || Number.isNaN(Date.parse(`${normalized}T00:00:00Z`))) {
    throw new Error(`${filename}: front matter field "date" must use YYYY-MM-DD or YYYY.MM.DD.`);
  }

  return {
    display: normalized.replaceAll("-", "."),
    sort: normalized,
  };
}

function plainTextExcerpt(markdown: string) {
  return markdown
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/```[\s\S]*?```/g, "")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/!\[[^\]]*]\([^)]*\)/g, "")
    .replace(/\[([^\]]+)]\([^)]*\)/g, "$1")
    .replace(/[>*_~#-]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 120);
}

function estimateReadTime(markdown: string) {
  const characters = markdown.replace(/\s/g, "").length;
  return `${Math.max(1, Math.ceil(characters / 500))} 分钟`;
}

function renderMarkdown(markdown: string) {
  const renderer = new MarkdownIt({
    html: false,
    linkify: true,
    typographer: true,
  }).use(katex);
  const environment = {};
  const tokens = renderer.parse(markdown, environment);
  const headings: ArticleHeading[] = [];

  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index];
    if (token.type !== "heading_open" || token.tag !== "h2") continue;

    const inlineToken = tokens[index + 1];
    const headingText = inlineToken?.children
      ?.map((child) => child.content)
      .join("")
      .trim();
    const heading = {
      id: `section-${headings.length + 1}`,
      text: headingText || inlineToken?.content.trim() || `Section ${headings.length + 1}`,
    };
    token.attrSet("id", heading.id);
    headings.push(heading);
  }

  return {
    headings,
    html: renderer.renderer.render(tokens, renderer.options, environment),
  };
}

function loadArticle(filename: string): LoadedArticle {
  const slug = basename(filename, extname(filename));
  if (!safeSlugPattern.test(slug)) {
    throw new Error(`${filename}: filenames must use lowercase URL-safe slugs such as "my-article.md".`);
  }

  const source = readFileSync(join(articlesDirectory, filename), "utf8");
  const { data, content } = matter(source);
  const markdown = content.trim();

  if (markdown === "") {
    throw new Error(`${filename}: the Markdown body cannot be empty.`);
  }

  const title = requiredString(data.title, "title", filename);
  const category = requiredString(data.category, "category", filename);
  const date = parseDate(data.date, filename);
  const excerpt = optionalString(data.excerpt, "excerpt", filename) ?? plainTextExcerpt(markdown);
  const readTime = optionalString(data.readTime, "readTime", filename) ?? estimateReadTime(markdown);

  if (data.featured !== undefined && typeof data.featured !== "boolean") {
    throw new Error(`${filename}: front matter field "featured" must be true or false.`);
  }

  if (excerpt === "") {
    throw new Error(`${filename}: add an excerpt or at least one text paragraph.`);
  }

  const rendered = renderMarkdown(markdown);

  return {
    slug,
    title,
    excerpt,
    category,
    date: date.display,
    sortDate: date.sort,
    readTime,
    featured: data.featured === true,
    headings: rendered.headings,
    html: rendered.html,
  };
}

export function getAllArticles(): Article[] {
  const filenames = readdirSync(articlesDirectory)
    .filter((filename) => extname(filename).toLowerCase() === ".md");

  if (filenames.length === 0) {
    throw new Error(`No Markdown articles found in ${articlesDirectory}.`);
  }

  const loaded = filenames
    .map(loadArticle)
    .sort((left, right) => right.sortDate.localeCompare(left.sortDate) || left.slug.localeCompare(right.slug));

  const featuredArticles = loaded.filter((article) => article.featured);
  if (featuredArticles.length > 1) {
    throw new Error(`Only one Markdown article may set "featured: true".`);
  }

  return loaded.map((article, index) => ({
    slug: article.slug,
    title: article.title,
    excerpt: article.excerpt,
    category: article.category,
    date: article.date,
    readTime: article.readTime,
    index: String(index + 1).padStart(2, "0"),
    featured: article.featured,
    headings: article.headings,
    html: article.html,
  }));
}

export function getArticleSummaries(): ArticleSummary[] {
  return getAllArticles().map((article) => ({
    slug: article.slug,
    title: article.title,
    excerpt: article.excerpt,
    category: article.category,
    date: article.date,
    readTime: article.readTime,
    index: article.index,
  }));
}

export function getArticle(slug: string) {
  return getAllArticles().find((article) => article.slug === slug);
}

export function getFeaturedArticle() {
  const articles = getAllArticles();
  return articles.find((article) => article.featured) ?? articles[0];
}
