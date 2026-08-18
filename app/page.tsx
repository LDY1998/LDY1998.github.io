import Link from "next/link";

import { ArticleArchive } from "@/app/components/article-archive";
import { ThemeToggle } from "@/app/components/theme-toggle";
import { getArticleSummaries, getFeaturedArticle } from "@/lib/articles";

export default function Home() {
  const articles = getArticleSummaries();
  const featuredArticle = getFeaturedArticle();

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
          <ThemeToggle />
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
              <span>{featuredArticle.category}</span>
              <span>{featuredArticle.date}</span>
              <span>{featuredArticle.readTime}阅读</span>
            </div>
            <h2>
              <Link href={`/articles/${featuredArticle.slug}/`}>{featuredArticle.title}</Link>
            </h2>
            <p>{featuredArticle.excerpt}</p>
            <Link className="read-button" href={`/articles/${featuredArticle.slug}/`}>
              阅读文章 <span aria-hidden="true">→</span>
            </Link>
          </div>
        </article>
      </section>

      <section className="archive" id="articles" aria-labelledby="archive-title">
        <div className="section-label">
          <span>02</span>
          <span id="archive-title">文章归档</span>
        </div>
        <ArticleArchive articles={articles} />
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
