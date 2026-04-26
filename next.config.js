/** @type {import('next').NextConfig} */
const nextConfig = {
  // Recharts + React 19: ensure Recharts is transpiled in the client bundle.
  // Fixes runtime webpack crashes like: "__webpack_modules__[moduleId] is not a function".
  transpilePackages: ["recharts"],
  webpack: (config) => {
    config.watchOptions = {
      poll: 1000,
      aggregateTimeout: 300,
    };
    return config;
  },
};

module.exports = nextConfig;
