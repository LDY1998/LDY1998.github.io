# 递归之外

“递归之外”是一个使用 Next.js、React 和 TypeScript 构建的静态个人博客首页，内容聚焦人工智能、计算机科学与软件工程。

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
