import type { NextConfig } from "next";

const NODE_ENV=process.env.NODE_ENV??'development'

const nextConfig: NextConfig = {
  output: "export",
  allowedDevOrigins: ["127.0.0.1"],
  basePath:NODE_ENV==='production'?'/linguae':'',
  images: {
    unoptimized: true,
  },
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
