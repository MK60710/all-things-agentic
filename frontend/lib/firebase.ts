import { getApp, getApps, initializeApp, type FirebaseApp } from "firebase/app";
import {
  GoogleAuthProvider,
  getAuth,
  onAuthStateChanged,
  signInWithCustomToken,
  signInWithPopup,
  signOut,
  type Auth,
  type User,
} from "firebase/auth";

// All NEXT_PUBLIC_* - safe to expose client-side, same as
// NEXT_PUBLIC_API_URL already is. These identify the Firebase project,
// they don't authorize anything on their own; real access control lives
// server-side in service/auth.py's ID-token verification.
const firebaseConfig = {
  apiKey: process.env.NEXT_PUBLIC_FIREBASE_API_KEY,
  authDomain: process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN,
  projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID,
  storageBucket: process.env.NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID,
  appId: process.env.NEXT_PUBLIC_FIREBASE_APP_ID,
};

// Lazy on purpose - getAuth() validates the config (throws
// auth/invalid-api-key) the moment it's called, and this module gets
// imported during Next's server-side prerender pass (e.g. building the
// /_not-found page) where no real browser session exists yet and the
// env vars may not even be set (CI, a build without Firebase configured
// yet). Deferring construction until a real caller (always client-side,
// from AuthProvider's useEffect or a real user action) actually needs it
// keeps `next build` itself independent of Firebase being configured.
let cachedAuth: Auth | null = null;

function getFirebaseAuth(): Auth {
  if (cachedAuth) return cachedAuth;
  const app: FirebaseApp = getApps().length ? getApp() : initializeApp(firebaseConfig);
  cachedAuth = getAuth(app);
  return cachedAuth;
}

export type { User };

export function onAuthChange(callback: (user: User | null) => void): () => void {
  return onAuthStateChanged(getFirebaseAuth(), callback);
}

export async function signInWithGoogle(): Promise<void> {
  await signInWithPopup(getFirebaseAuth(), new GoogleAuthProvider());
}

export async function signOutUser(): Promise<void> {
  await signOut(getFirebaseAuth());
}

export async function getIdToken(): Promise<string | null> {
  const user = getFirebaseAuth().currentUser;
  if (!user) return null;
  return user.getIdToken();
}

// Dev/test-only: signs in with a Firebase custom token (minted server-side
// via the Admin SDK, e.g. `firebase_admin.auth.create_custom_token(uid)`)
// instead of a real Google popup - lets an automated test drive the app as
// a real signed-in account without needing anyone's actual Google
// credentials. Exposed on window rather than only exported, so a
// browser-automation tool can call it directly from page-context JS.
// Excluded from production builds - this is exactly as safe as it sounds
// and no safer: anyone with this token can sign in as whatever uid it was
// minted for, so it must never be reachable outside local development.
if (process.env.NODE_ENV !== "production" && typeof window !== "undefined") {
  const testWindow = window as unknown as {
    __authTestSignIn: (token: string) => Promise<void>;
    __authTestGetIdToken: () => Promise<string | null>;
  };
  testWindow.__authTestSignIn = async (token: string) => {
    await signInWithCustomToken(getFirebaseAuth(), token);
  };
  testWindow.__authTestGetIdToken = getIdToken;
}
