import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      // Point Tailwind's font utilities at the next/font variables. Without this, `font-sans` resolves to the
      // system stack and silently overrides the body font.
      fontFamily: {
        sans: ["var(--font-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
        display: ["var(--font-display)", "Georgia", "serif"],
      },
    },
  },
  plugins: [],
  // Preflight is off on purpose (the inline `S` styles were written against browser defaults); globals.css carries a minimal @layer base reset and the theme variables.
  corePlugins: { preflight: false },
};

export default config;
