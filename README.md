# 递归之外

“递归之外”是一个使用 Next.js、React、TypeScript 和 Markdown 构建的静态个人博客，内容聚焦人工智能、计算机科学与软件工程。

项目支持响应式布局、文章分类筛选、全文搜索以及可持久化的深浅色主题。生产构建会导出为纯静态文件，并由 GitHub Actions 自动部署到 GitHub Pages。

## Prerequisites

- Node.js `>=22.13.0`
- npm（随 Node.js 一同安装）

## Quick Start

```bash
npm install
npm run dev
```

然后访问 [http://localhost:3000](http://localhost:3000)。

## 新建文章

在 `content/articles/` 中新建一个 Markdown 文件，文件名会直接成为文章地址。例如：

```text
content/articles/my-first-post.md → /articles/my-first-post/
```

文件名请使用小写英文字母、数字和连字符。每篇文章至少需要 `title`、`date` 和 `category` 三项 front matter：

```md
---
title: "我的第一篇文章"
date: "2026-08-17"
category: "工程实践"
excerpt: "可选；省略时会从正文自动提取。"
readTime: "6 分钟"
featured: false
---

这里是文章开头。正文支持普通 Markdown。

## 第一个章节

- 列表
- **粗体**、`行内代码` 与 [链接](https://example.com)

> 引用内容会以重点样式显示。
```

公式支持常见的 TeX 写法：使用 `$E = mc^2$` 插入行内公式，或用独占一段的 `$$...$$` 插入居中公式。公式会在构建时由 KaTeX 渲染，不依赖外部 CDN。

其中 `excerpt`、`readTime` 和 `featured` 都可以省略：阅读时间和摘要会自动计算；如果没有文章设置 `featured: true`，日期最新的文章会成为首页精选。只能有一篇文章设置为精选。

保存后开发服务器会读取该文件；生产构建会自动发现所有 `.md` 文件、生成首页列表，并导出对应的静态文章页面。建议新增文章后运行 `npm run check`。

## 可用命令

- `npm run dev`：启动本地开发服务器
- `npm run build`：生成 GitHub Pages 使用的静态站点到 `out/`
- `npm run start`：在本地预览已经生成的 `out/`
- `npm run preview`：重新构建并预览生产版本
- `npm run lint`：运行 ESLint
- `npm run typecheck`：运行 TypeScript 类型检查
- `npm test`：构建并验证静态输出
- `npm run check`：依次运行全部质量检查

## 项目结构

- `app/`：页面、布局与全局样式
- `content/articles/`：每篇文章一个 Markdown 文件
- `lib/articles.ts`：构建时发现、校验并渲染 Markdown
- `public/`：直接复制到静态站点的公共资源
- `tests/`：对生产静态输出的自动化测试
- `.github/workflows/deploy-pages.yml`：GitHub Pages 构建与部署流程

## GitHub Pages 部署

每次推送到 `master` 后，工作流会：

1. 安装锁定的依赖；
2. 运行代码检查、类型检查、构建和测试；
3. 上传 `out/`；
4. 部署到 GitHub Pages。

首次部署前，请在仓库的 **Settings → Pages → Build and deployment → Source** 中选择 **GitHub Actions**。之后无需提交 `docs/` 或 `out/` 目录。
