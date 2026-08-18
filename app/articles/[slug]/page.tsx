import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { ThemeToggle } from "@/app/components/theme-toggle";
import { getAllArticles, getArticle } from "@/lib/articles";

type ArticlePageProps = {
  params: Promise<{ slug: string }>;
};

export const dynamicParams = false;

export function generateStaticParams() {
  return getAllArticles().map((article) => ({ slug: article.slug }));
}

export async function generateMetadata({ params }: ArticlePageProps): Promise<Metadata> {
  const { slug } = await params;
  const article = getArticle(slug);

  if (!article) {
    return { title: "文章未找到｜递归之外" };
  }

  return {
    title: `${article.title}｜递归之外`,
    description: article.excerpt,
  };
}

export default async function ArticlePage({ params }: ArticlePageProps) {
  const { slug } = await params;
  const article = getArticle(slug);

  if (!article) notFound();

  const articles = getAllArticles();
  const articleIndex = articles.findIndex((item) => item.slug === article.slug);
  const nextArticle = articles[(articleIndex + 1) % articles.length];

  return (
    <main className="reading-page">
      <header className="site-header reading-header">
        <Link className="wordmark" href="/" aria-label="返回递归之外首页">
          <span className="mark">[ R ]</span>
          <span>递归之外</span>
        </Link>
        <nav className="nav-links reading-nav" aria-label="文章导航">
          <Link href="/#articles">全部文章</Link>
          <Link href="/#about">关于</Link>
          <ThemeToggle />
        </nav>
      </header>

      <article>
        <header className="article-hero">
          <Link className="article-back" href="/#articles">
            <span aria-hidden="true">←</span> 返回文章归档
          </Link>
          <div className="article-meta article-page-meta">
            <span>{article.index}</span>
            <span>{article.category}</span>
            <span>{article.date}</span>
            <span>{article.readTime}阅读</span>
          </div>
          <h1>{article.title}</h1>
          <p className="article-dek">{article.excerpt}</p>
        </header>

        <div className="article-reading-grid">
          <aside className="article-toc" aria-label="文章目录">
            <span>CONTENTS</span>
            <ol>
              {article.headings.map((heading) => (
                <li key={heading.id}>
                  <a href={`#${heading.id}`}>{heading.text}</a>
                </li>
              ))}
            </ol>
          </aside>

          <div className="article-body">
            <div
              className="article-markdown"
              dangerouslySetInnerHTML={{ __html: article.html }}
            />
            <div className="article-closing article-end">
              <span>END / {article.index}</span>
            </div>
          </div>
        </div>
      </article>

      <section className="next-reading" aria-labelledby="next-reading-title">
        <div>
          <span>继续阅读</span>
          <strong id="next-reading-title">{nextArticle.title}</strong>
        </div>
        <Link href={`/articles/${nextArticle.slug}/`} aria-label={`阅读：${nextArticle.title}`}>
          <span aria-hidden="true">→</span>
        </Link>
      </section>

      <footer className="reading-footer">
        <p>关于智能与计算的长期笔记。</p>
        <Link href="/">返回首页 ↑</Link>
      </footer>
    </main>
  );
}
