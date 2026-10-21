import type { Metadata } from "next";
import "./globals.css";

const title = "CareerPilot — Multi-Agent Job-Hunt Copilot";
const description =
  "Ask one question; a router sends it to specialist AI agents — résumé/JD fit, live job search, your application tracker, interview prep, and outreach drafting — running in parallel with a live orchestration trace.";
const url = "https://careerpilot-raghu298.netlify.app";

export const metadata: Metadata = {
  metadataBase: new URL(url),
  title,
  description,
  applicationName: "CareerPilot",
  keywords: ["job search", "AI agents", "LangGraph", "MCP", "RAG", "resume", "GenAI"],
  openGraph: {
    title,
    description,
    url,
    siteName: "CareerPilot",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title,
    description,
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
