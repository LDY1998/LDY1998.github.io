import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "递归之外｜AI 与计算机科学笔记",
  description: "关于人工智能、计算机科学与软件工程的个人博客。",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}
