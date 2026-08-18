export type ArticleSummary = {
  slug: string;
  title: string;
  excerpt: string;
  category: string;
  date: string;
  readTime: string;
  index: string;
};

export type ArticleHeading = {
  id: string;
  text: string;
};

export type Article = ArticleSummary & {
  featured: boolean;
  headings: ArticleHeading[];
  html: string;
};
