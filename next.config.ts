import type { NextConfig } from "next";

/** Recharts + React 19: align deps and transpile so the client bundle does not load mismatched react-is (webpack "__webpack_modules__ … is not a function"). */
const nextConfig: NextConfig = {
  transpilePackages: ["recharts"],
};

export default nextConfig;
