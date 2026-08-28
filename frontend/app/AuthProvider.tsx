"use client";

import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { onAuthChange, signInWithGoogle, signOutUser, type User } from "@/lib/firebase";

interface AuthContextValue {
  user: User;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

// Every page under app/ (page.tsx, deep-dive/[paperId]/page.tsx, and any
// future route) renders inside RootLayout, so gating happens once here
// rather than duplicated per-page - a signed-out visitor never reaches
// any route's own code, not just the main page's.
export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be called from inside AuthProvider - every page is wrapped in it via layout.tsx");
  }
  return ctx;
}

export default function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [signingIn, setSigningIn] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => onAuthChange((next) => {
    setUser(next);
    setLoading(false);
  }), []);

  async function handleSignIn() {
    setSigningIn(true);
    setError("");
    try {
      await signInWithGoogle();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Sign-in failed. Try again.");
    } finally {
      setSigningIn(false);
    }
  }

  if (loading) {
    return <div className="auth-loading" role="status" aria-live="polite" />;
  }

  if (!user) {
    return (
      <div className="auth-gate">
        <div className="auth-gate-icon">
          <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
            <path d="M4 19.5V5a2 2 0 0 1 2-2h13v15H6a2 2 0 0 0 0 4h13" />
          </svg>
        </div>
        <h1>Atlas</h1>
        <p>Sign in to build your own research graph - your sessions, papers and graph stay private to your account.</p>
        <button className="auth-gate-signin" onClick={() => void handleSignIn()} disabled={signingIn}>
          <svg width="16" height="16" viewBox="0 0 18 18" aria-hidden="true">
            <path fill="#4285F4" d="M17.64 9.2c0-.64-.06-1.25-.16-1.84H9v3.48h4.84a4.14 4.14 0 0 1-1.8 2.72v2.26h2.92a8.78 8.78 0 0 0 2.68-6.62z" />
            <path fill="#34A853" d="M9 18c2.43 0 4.47-.8 5.96-2.18l-2.92-2.26c-.81.54-1.84.86-3.04.86-2.34 0-4.32-1.58-5.03-3.7H.96v2.33A9 9 0 0 0 9 18z" />
            <path fill="#FBBC05" d="M3.97 10.72A5.4 5.4 0 0 1 3.68 9c0-.6.1-1.18.29-1.72V4.95H.96A9 9 0 0 0 0 9c0 1.45.35 2.83.96 4.05l3.01-2.33z" />
            <path fill="#EA4335" d="M9 3.58c1.32 0 2.5.45 3.44 1.35l2.58-2.58C13.46.89 11.43 0 9 0A9 9 0 0 0 .96 4.95l3.01 2.33C4.68 5.16 6.66 3.58 9 3.58z" />
          </svg>
          {signingIn ? "Signing in…" : "Sign in with Google"}
        </button>
        {error && <p className="auth-gate-error">{error}</p>}
      </div>
    );
  }

  return <AuthContext.Provider value={{ user, signOut: signOutUser }}>{children}</AuthContext.Provider>;
}
