"use client";

import { useEffect, useMemo, useState } from "react";

type Article = {
  title: string;
  excerpt: string;
  category: "人工智能" | "计算机科学" | "工程实践";
  date: string;
  readTime: string;
  index: string;
};

const articles: Article[] = [
  {
    title: "从注意力到推理：大语言模型究竟学会了什么？",
    excerpt:
      "越过参数规模与榜单分数，从表征、上下文学习和涌现能力三个角度，重新理解语言模型的能力边界。",
    category: "人工智能",
    date: "2026.08.12",
    readTime: "12 分钟",
    index: "01",
  },
  {
    title: "向量数据库不是魔法：一次语义检索的完整旅程",
    excerpt:
      "Embedding、近似最近邻与重排序如何协作，以及一个看似简单的 RAG 系统最容易在哪些地方失效。",
    category: "工程实践",
    date: "2026.08.04",
    readTime: "9 分钟",
    index: "02",
  },
  {
    title: "为什么递归让人着迷，也让机器头疼",
    excerpt:
      "从调用栈、分治到函数式思维，用几段熟悉的程序，观察问题如何在更小的自身中得到回答。",
    category: "计算机科学",
    date: "2026.07.28",
    readTime: "7 分钟",
    index: "03",
  },
  {
    title: "训练一个模型之前，先训练你的数据直觉",
    excerpt:
      "数据分布、泄漏、偏差与标签质量：决定模型上限的，往往是被我们略过的那些基础问题。",
    category: "人工智能",
    date: "2026.07.17",
    readTime: "10 分钟",
    index: "04",
  },
  {
    title: "把系统变简单：缓存一致性的三个思维模型",
    excerpt:
      "不从协议名词开始，而从读、写和时间出发，建立一套可以迁移到真实工程中的一致性直觉。",
    category: "计算机科学",
    date: "2026.07.06",
    readTime: "11 分钟",
    index: "05",
  },
  {
    title: "AI 应用的评估，不应该止于一个准确率",
    excerpt:
      "建立一套小而有效的评估框架，把离线指标、真实用户体验与失败案例放在同一张地图上。",
    category: "工程实践",
    date: "2026.06.24",
    readTime: "8 分钟",
    index: "06",
  },
];

const categories = ["全部", "人工智能", "计算机科学", "工程实践"] as const;

function SunIcon() {
  return <span aria-hidden="true">☼</span>;
}

function MoonIcon() {
  return <span aria-hidden="true">◐</span>;
}

export default function Home() {
  const [category, setCategory] = useState<(typeof categories)[number]>("全部");
  const [query, setQuery] = useState("");
  const [dark, setDark] = useState(false);

  useEffect(() => {
    const saved = window.localStorage.getItem("theme");
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const shouldUseDark = saved ? saved === "dark" : prefersDark;
    document.documentElement.dataset.theme = shouldUseDark ? "dark" : "light";

    const frame = window.requestAnimationFrame(() => setDark(shouldUseDark));
    return () => window.cancelAnimationFrame(frame);
  }, []);

  const filteredArticles = useMemo(() => {
    const keyword = query.trim().toLowerCase();
    return articles.filter((article) => {
      const matchesCategory = category === "全部" || article.category === category;
      const matchesQuery =
        !keyword ||
        `${article.title}${article.excerpt}${article.category}`.toLowerCase().includes(keyword);
      return matchesCategory && matchesQuery;
    });
  }, [category, query]);

  function toggleTheme() {
    const next = !dark;
    setDark(next);
    document.documentElement.dataset.theme = next ? "dark" : "light";
    window.localStorage.setItem("theme", next ? "dark" : "light");
  }

  return (
    <main>
      <header className="site-header">
        <a className="wordmark" href="#top" aria-label="递归之外首页">
          <span className="mark">[ R ]</span>
          <span>递归之外</span>
        </a>
        <nav className="nav-links" aria-label="主导航">
          <a href="#articles">文章</a>
          <a href="#about">关于</a>
          <button className="theme-toggle" onClick={toggleTheme} aria-label={dark ? "切换到浅色模式" : "切换到深色模式"}>
            {dark ? <SunIcon /> : <MoonIcon />}
          </button>
        </nav>
      </header>

      <section className="hero" id="top">
        <div className="hero-kicker">
          <span>AI × CS</span>
          <span className="rule" />
          <span>ISSUE 08 / 2026</span>
        </div>
        <div className="hero-grid">
          <div>
            <h1>
              在代码与智能之间，
              <br />
              <em>寻找清晰。</em>
            </h1>
          </div>
          <div className="hero-note">
            <p>
              一个关于人工智能、计算机科学与软件工程的个人博客。记录值得被解释的问题，也记录那些还没有答案的问题。
            </p>
            <a className="text-link" href="#articles">
              开始阅读 <span aria-hidden="true">↘</span>
            </a>
          </div>
        </div>
        <div className="signal" aria-hidden="true">
          <span>INPUT</span>
          <div className="signal-line"><i /><i /><i /><i /><i /><i /><i /></div>
          <span>THOUGHT</span>
        </div>
      </section>

      <section className="featured" aria-labelledby="featured-title">
        <div className="section-label">
          <span>01</span>
          <span id="featured-title">本期精选</span>
        </div>
        <article className="featured-card">
          <div className="featured-visual" aria-hidden="true">
            <div className="orbit orbit-one" />
            <div className="orbit orbit-two" />
            <div className="orbit orbit-three" />
            <div className="core">?</div>
            <span className="coordinate c-one">01 / TOKEN</span>
            <span className="coordinate c-two">CONTEXT →</span>
            <span className="coordinate c-three">∑ P(xᵢ|x&lt;i)</span>
          </div>
          <div className="featured-copy">
            <div className="article-meta">
              <span>人工智能</span>
              <span>2026.08.12</span>
              <span>12 分钟阅读</span>
            </div>
            <h2>从注意力到推理：大语言模型究竟学会了什么？</h2>
            <p>
              当我们说模型“理解”了语言时，究竟在说什么？这篇文章越过参数规模和榜单分数，从表征、上下文学习与涌现能力出发，尝试画出更诚实的能力边界。
            </p>
            <a className="read-button" href="#articles">
              阅读文章 <span aria-hidden="true">→</span>
            </a>
          </div>
        </article>
      </section>

      <section className="archive" id="articles" aria-labelledby="archive-title">
        <div className="section-label">
          <span>02</span>
          <span id="archive-title">文章归档</span>
        </div>
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
              <article className="article-row" key={article.index}>
                <span className="article-index">{article.index}</span>
                <div className="article-main">
                  <div className="article-meta">
                    <span>{article.category}</span>
                    <span>{article.date}</span>
                  </div>
                  <h3>{article.title}</h3>
                  <p>{article.excerpt}</p>
                </div>
                <div className="article-side">
                  <span>{article.readTime}</span>
                  <span className="arrow" aria-hidden="true">↗</span>
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
      </section>

      <section className="about" id="about" aria-labelledby="about-title">
        <div className="section-label light-label">
          <span>03</span>
          <span id="about-title">关于本站</span>
        </div>
        <div className="about-grid">
          <h2>
            理解技术，
            <br />
            也理解技术背后的选择。
          </h2>
          <div className="about-copy">
            <p>
              我相信好的技术写作不是把复杂藏起来，而是找到通往复杂的那条清晰路径。这里的文章来自阅读、实验与真实项目，也来自对一个简单问题的反复追问：它为什么这样工作？
            </p>
            <div className="topic-grid">
              <div><span>关注</span><strong>机器学习 / LLM</strong></div>
              <div><span>基础</span><strong>算法 / 系统</strong></div>
              <div><span>实践</span><strong>工程 / 产品</strong></div>
              <div><span>频率</span><strong>每月 2—3 篇</strong></div>
            </div>
          </div>
        </div>
      </section>

      <footer>
        <a className="wordmark footer-mark" href="#top">
          <span className="mark">[ R ]</span>
          <span>递归之外</span>
        </a>
        <p>关于智能与计算的长期笔记。</p>
        <div className="footer-meta">
          <span>© 2026</span>
          <a href="#about">关于</a>
          <a href="#top">返回顶部 ↑</a>
        </div>
      </footer>
    </main>
  );
}
