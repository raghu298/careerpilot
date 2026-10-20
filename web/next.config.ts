import type { NextConfig } from "next";

// Static export: the whole UI is client-rendered and talks to the CareerPilot API
// over fetch, so it deploys to Netlify as static files with no serverless runtime.
const nextConfig: NextConfig = {
  output: "export",
  reactStrictMode: true,
  images: { unoptimized: true },
};

export default nextConfig;
