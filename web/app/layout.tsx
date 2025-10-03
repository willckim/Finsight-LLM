import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "FinSight — Finance LLM Copilot",
  description: "Finance-focused LLM chat. Clean explanations of ratios, filings, and valuation.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        suppressHydrationWarning
        className={`${geistSans.variable} ${geistMono.variable} min-h-screen bg-app-95 text-app-foreground antialiased`}
      >
        {/* Background glow + soft grid */}
        <div className="fixed inset-0 -z-10">
          <div className="pointer-events-none absolute inset-0 bg-radial-glow" />
          <div className="pointer-events-none absolute inset-0 bg-soft-grid opacity-15" />
        </div>

        <header className="sticky top-0 z-10 border-b border-app-border bg-app-85 backdrop-blur">
          <div className="mx-auto flex max-w-5xl items-center justify-between px-5 py-4">
            <div className="flex items-center gap-3">
              <div className="h-9 w-9 rounded-xl bg-accent-gradient shadow-accent" />
              <div>
                <h1 className="text-2xl font-semibold tracking-tight text-gradient">FinSight</h1>
                <p className="text-[12px] text-app-muted">Finance LLM Copilot · FastAPI · ONNX</p>
              </div>
            </div>
            <span className="rounded-full border border-app-border bg-app-panel px-3 py-1 text-xs text-app-muted">
              Demo
            </span>
          </div>
        </header>

        <main className="mx-auto max-w-5xl px-5 py-8">{children}</main>

        <footer className="border-t border-app-border py-5 text-center text-xs text-app-muted">
          © {new Date().getFullYear()} FinSight · Finance LLM Copilot
        </footer>
      </body>
    </html>
  );
}
