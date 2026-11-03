import type { Metadata, Viewport } from "next";
import { Fraunces, Instrument_Sans, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { THEME_KEY } from "@/lib/theme";

// Body: Instrument Sans (clean, slightly condensed). Display: Fraunces, a variable serif whose optical size
// makes headings sharp and characterful. Mono: JetBrains Mono for data and labels.
const fontSans = Instrument_Sans({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
});

const fontDisplay = Fraunces({
  subsets: ["latin"],
  variable: "--font-display",
  axes: ["opsz"],
  style: ["normal", "italic"],
  display: "swap",
});

const fontMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  weight: ["400", "500", "600", "700"],
});

export const metadata: Metadata = {
  title: "RAGBase",
  description: "Document Intelligence & RAG Platform",
  icons: {
    icon: "/favicon.svg",
  },
};

// viewportFit "cover" lets the app draw under the notch; globals.css pads with safe-area insets.
export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: "#09090b",
};

// Dark is the default (CSS :root). Runs before first paint so a saved light choice never flashes dark.
const themeScript = `try{if(localStorage.getItem("${THEME_KEY}")==="light")document.documentElement.dataset.theme="light"}catch(e){}`;

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body
        className={[
          fontSans.variable,
          fontDisplay.variable,
          fontMono.variable,
          "overflow-hidden antialiased",
        ].join(" ")}
      >
        {children}
      </body>
    </html>
  );
}
