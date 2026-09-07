import { create } from "zustand";

import type { User } from "@/types/api";

export type AuthStatus = "checking" | "authenticated" | "unauthenticated";

interface AuthState {
  user: User | null;
  accessToken: string | null;
  status: AuthStatus;
  setSession: (user: User, accessToken: string) => void;
  clearSession: () => void;
}

/**
 * The access token lives here - in memory only, never in localStorage or a
 * readable cookie - per the approved auth architecture. It is lost on page
 * reload by design; `features/auth/useBootstrapSession` re-establishes it on
 * app start by calling /auth/refresh against the httpOnly refresh cookie,
 * which is what makes the session actually persist across reloads.
 */
export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  accessToken: null,
  status: "checking",
  setSession: (user, accessToken) => set({ user, accessToken, status: "authenticated" }),
  clearSession: () => set({ user: null, accessToken: null, status: "unauthenticated" }),
}));
