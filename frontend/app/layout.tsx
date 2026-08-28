import type { Metadata } from "next";
import "./globals.css";
import AuthProvider from "./AuthProvider";

export const metadata: Metadata = {
  title: "Atlas · Research Graph",
  description: "Explore papers, evidence, and connections in one living research map.",
};

// Runs before paint (blocking, inline) so the very first frame already has
// the right theme - without this, the page briefly renders the default
// light palette (from :root, unguarded) even for a visitor who already
// chose dark, then snaps to dark once React hydrates and the toggle's own
// effect runs. No dependency on React state, deliberately - it has to run
// before there's a React tree to read state from.
const THEME_INIT_SCRIPT = `(function(){try{var t=localStorage.getItem("atlas-theme");if(t==="dark"||t==="light"){document.documentElement.setAttribute("data-theme",t);}}catch(e){}})();`;

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_INIT_SCRIPT }} />
      </head>
      <body>
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
