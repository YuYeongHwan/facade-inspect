import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Facade Inspect",
  description: "드론 영상 기반 건물 창문 오염도 분석",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="ko" className={`${geistSans.variable} ${geistMono.variable} dark h-full antialiased`}>
      <body className="flex min-h-full flex-col font-sans">
        <header className="sticky top-0 z-40 flex h-14 shrink-0 items-center gap-6 border-b border-neutral-800 bg-neutral-950/90 px-4 backdrop-blur">
          <Link href="/" className="font-semibold tracking-tight">
            Facade Inspect
          </Link>
          <nav className="flex gap-4 text-sm text-neutral-400">
            <Link href="/" className="hover:text-neutral-100">
              검사 결과
            </Link>
            <Link href="/detections" className="hover:text-neutral-100">
              검출 뷰어
            </Link>
          </nav>
        </header>
        <main className="flex flex-1 flex-col">{children}</main>
      </body>
    </html>
  );
}
