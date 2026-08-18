"use client";

import Link from "next/link";
import { useMemo, useState } from "react";

import type { ArticleSummary } from "@/lib/article-types";

type ArticleArchiveProps = {
  articles: ArticleSummary[];
};

export function ArticleArchive({ articles }: ArticleArchiveProps) {
  const categories = useMemo(
    () => ["全部", ...Array.from(new Set(articles.map((article) => article.category)))],
    [articles],
  );
  const [category, setCategory] = useState("全部");
  const [query, setQuery] = useState("");

  const filteredArticles = useMemo(() => {
    const keyword = query.trim().toLowerCase();
    return articles.filter((article) => {
      const matchesCategory = category === "全部" || article.category === category;
      const matchesQuery =
        !keyword ||
        `${article.title}${article.excerpt}${article.category}`.toLowerCase().includes(keyword);
      return matchesCategory && matchesQuery;
    });
  }, [articles, category, query]);

  return (
    <>
      <div className="archive-tools">
        <div className="filters" role="group" aria-label="按主题筛选">
          {categories.map((item) => (
            <button
              key={item}
              className={category === item ? "active" : ""}
              onClick={() => setCategory(item)}
              aria-pressed={category === item}
            >
              {item}
            </button>
          ))}
        </div>
        <label className="search-box">
          <span aria-hidden="true">⌕</span>
          <span className="sr-only">搜索文章</span>
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="搜索文章"
          />
        </label>
      </div>

      <div className="article-list" aria-live="polite">
        {filteredArticles.length > 0 ? (
          filteredArticles.map((article) => (
            <article className="article-row" key={article.slug}>
              <span className="article-index">{article.index}</span>
              <div className="article-main">
                <div className="article-meta">
                  <span>{article.category}</span>
                  <span>{article.date}</span>
                </div>
                <h3>
                  <Link href={`/articles/${article.slug}/`}>{article.title}</Link>
                </h3>
                <p>{article.excerpt}</p>
              </div>
              <div className="article-side">
                <span>{article.readTime}</span>
                <Link
                  className="arrow"
                  href={`/articles/${article.slug}/`}
                  aria-label={`阅读：${article.title}`}
                >
                  <span aria-hidden="true">↗</span>
                </Link>
              </div>
            </article>
          ))
        ) : (
          <div className="empty-state">
            <span>NULL</span>
            <p>没有找到匹配的文章，换个关键词试试。</p>
          </div>
        )}
      </div>
    </>
  );
}
