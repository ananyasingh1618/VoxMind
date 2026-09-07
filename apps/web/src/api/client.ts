import { readCookie } from "@/lib/cookies";
import { useAuthStore } from "@/stores/authStore";
import type { AccessTokenResponse, ApiErrorBody } from "@/types/api";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/api/v1";
const CSRF_COOKIE_NAME = "voxmind_csrf";

export class ApiError extends Error {
  code: string;
  status: number;
  detail?: unknown;

  constructor(status: number, body: ApiErrorBody) {
    super(body.message);
    this.name = "ApiError";
    this.status = status;
    this.code = body.code;
    this.detail = body.detail;
  }
}

interface RequestOptions {
  method?: string;
  body?: unknown;
  /** Multipart form data (e.g. a file upload) - sent as-is, with no
   * Content-Type header set manually so the browser can add the correct
   * multipart boundary itself. Mutually exclusive with `body`. */
  formData?: FormData;
  skipAuthRetry?: boolean;
}

/**
 * Single-flight guard for /auth/refresh: if several requests hit a 401 at
 * once (e.g. multiple panels fetching on mount), only one refresh call is
 * made and the rest await its result. This is the client-side half of
 * preventing the concurrent-refresh race documented in the backend's
 * RefreshTokenRepository - it stops the race from happening at all in the
 * common case, rather than relying solely on the server absorbing it.
 */
let inFlightRefresh: Promise<AccessTokenResponse | null> | null = null;

async function refreshSession(): Promise<AccessTokenResponse | null> {
  if (!inFlightRefresh) {
    inFlightRefresh = performRefresh().finally(() => {
      inFlightRefresh = null;
    });
  }
  return inFlightRefresh;
}

async function performRefresh(): Promise<AccessTokenResponse | null> {
  const csrfToken = readCookie(CSRF_COOKIE_NAME);
  const response = await fetch(`${API_BASE}/auth/refresh`, {
    method: "POST",
    credentials: "include",
    headers: csrfToken ? { "X-CSRF-Token": csrfToken } : {},
  });
  if (!response.ok) {
    return null;
  }
  return (await response.json()) as AccessTokenResponse;
}

export async function apiFetch<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = "GET", body, formData, skipAuthRetry = false } = options;
  const accessToken = useAuthStore.getState().accessToken;

  const headers: Record<string, string> = {};
  if (!formData) headers["Content-Type"] = "application/json";
  if (accessToken) headers.Authorization = `Bearer ${accessToken}`;
  if (method !== "GET") {
    const csrfToken = readCookie(CSRF_COOKIE_NAME);
    if (csrfToken) headers["X-CSRF-Token"] = csrfToken;
  }

  const response = await fetch(`${API_BASE}${path}`, {
    method,
    credentials: "include",
    headers,
    body: formData ?? (body !== undefined ? JSON.stringify(body) : undefined),
  });

  if (response.status === 204) {
    return undefined as T;
  }

  const payload = await response.json().catch(() => null);

  if (!response.ok) {
    const errorBody: ApiErrorBody = payload?.error ?? {
      code: "unknown_error",
      message: "An unknown error occurred.",
    };

    if (errorBody.code === "session_expired" && !skipAuthRetry) {
      const refreshed = await refreshSession();
      if (refreshed) {
        useAuthStore.getState().setSession(refreshed.user, refreshed.access_token);
        return apiFetch<T>(path, { ...options, skipAuthRetry: true });
      }
      useAuthStore.getState().clearSession();
    }

    throw new ApiError(response.status, errorBody);
  }

  return payload as T;
}
