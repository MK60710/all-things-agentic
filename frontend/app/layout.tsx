import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Atlas · Research Graph",
  description: "Explore papers, evidence, and connections in one living research map.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
