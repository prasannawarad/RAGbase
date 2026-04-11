import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAGBase",
  description: "Document Intelligence & RAG Platform",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className="
          h-screen w-screen overflow-hidden
          bg-gradient-to-b from-[#020204] via-[#050510] to-black
          text-white
        "
      >
        {children}
      </body>
    </html>
  );
}
