import { beforeEach, describe, expect, it, vi } from "vitest";

import { apiFetch, ApiError } from "@/api/client";
import { useAuthStore } from "@/stores/authStore";

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("apiFetch", () => {
  beforeEach(() => {
    useAuthStore.setState({ user: null, accessToken: "stale-token", status: "authenticated" });
    vi.restoreAllMocks();
  });

  it("silently refreshes and retries once on a session_expired error, then returns the retried result", async () => {
    const fetchSpy = vi.spyOn(global, "fetch");
    fetchSpy
      .mockResolvedValueOnce(
        jsonResponse(
          { data: null, error: { code: "session_expired", message: "expired" }, request_id: "r1" },
          401,
        ),
      )
      .mockResolvedValueOnce(
        jsonResponse({
          access_token: "fresh-token",
          token_type: "bearer",
          expires_in_seconds: 900,
          user: { id: "1", email: "user@voxmind.dev", created_at: "2026-01-01T00:00:00Z" },
        }),
      )
      .mockResolvedValueOnce(jsonResponse({ ok: true }));

    const result = await apiFetch<{ ok: boolean }>("/conversations");

    expect(result).toEqual({ ok: true });
    expect(fetchSpy).toHaveBeenCalledTimes(3);
    expect(useAuthStore.getState().accessToken).toBe("fresh-token");
  });

  it("clears the session and throws when refresh also fails", async () => {
    vi.spyOn(global, "fetch")
      .mockResolvedValueOnce(
        jsonResponse(
          { data: null, error: { code: "session_expired", message: "expired" }, request_id: "r1" },
          401,
        ),
      )
      .mockResolvedValueOnce(jsonResponse({}, 401));

    await expect(apiFetch("/conversations")).rejects.toBeInstanceOf(ApiError);
    expect(useAuthStore.getState().status).toBe("unauthenticated");
  });

  it("surfaces a typed ApiError with the backend's error code for non-auth failures", async () => {
    vi.spyOn(global, "fetch").mockResolvedValue(
      jsonResponse(
        { data: null, error: { code: "not_found", message: "Conversation not found." }, request_id: "r1" },
        404,
      ),
    );

    await expect(apiFetch("/conversations/does-not-exist")).rejects.toMatchObject({
      code: "not_found",
      status: 404,
    });
  });
});
