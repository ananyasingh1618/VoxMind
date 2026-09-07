import { apiFetch } from "@/api/client";
import { readCookie } from "@/lib/cookies";
import type { AccessTokenResponse, User } from "@/types/api";

export function register(email: string, password: string): Promise<User> {
  return apiFetch<User>("/auth/register", { method: "POST", body: { email, password } });
}

export function login(email: string, password: string): Promise<AccessTokenResponse> {
  return apiFetch<AccessTokenResponse>("/auth/login", { method: "POST", body: { email, password } });
}

export function refresh(): Promise<AccessTokenResponse> {
  return apiFetch<AccessTokenResponse>("/auth/refresh", { method: "POST", skipAuthRetry: true });
}

export function logout(): Promise<void> {
  return apiFetch<void>("/auth/logout", { method: "POST" });
}

export function logoutEverywhere(): Promise<void> {
  return apiFetch<void>("/auth/logout-everywhere", { method: "POST" });
}

export function hasRefreshCookie(): boolean {
  // The refresh token itself is httpOnly and unreadable from JS by design;
  // the CSRF cookie is set alongside it in the same response, so its
  // presence is a reasonable client-side hint that a session might exist,
  // worth attempting a silent refresh for on app boot.
  return readCookie("voxmind_csrf") !== null;
}
