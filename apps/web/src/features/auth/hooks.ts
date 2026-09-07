import { useMutation } from "@tanstack/react-query";
import { useEffect, useRef } from "react";

import * as authApi from "@/api/auth";
import { useAuthStore } from "@/stores/authStore";

export function useLogin() {
  const setSession = useAuthStore((s) => s.setSession);
  return useMutation({
    mutationFn: ({ email, password }: { email: string; password: string }) =>
      authApi.login(email, password),
    onSuccess: (data) => setSession(data.user, data.access_token),
  });
}

export function useRegister() {
  return useMutation({
    mutationFn: ({ email, password }: { email: string; password: string }) =>
      authApi.register(email, password),
  });
}

export function useLogout() {
  const clearSession = useAuthStore((s) => s.clearSession);
  return useMutation({
    mutationFn: () => authApi.logout(),
    onSettled: () => clearSession(),
  });
}

export function useLogoutEverywhere() {
  const clearSession = useAuthStore((s) => s.clearSession);
  return useMutation({
    mutationFn: () => authApi.logoutEverywhere(),
    onSettled: () => clearSession(),
  });
}

/**
 * Runs once on app start. The access token lives only in memory, so a page
 * reload always starts with none - this attempts a silent refresh against
 * the httpOnly refresh cookie to re-establish the session, which is what
 * makes "stay logged in across a reload" actually work without storing the
 * access token anywhere persistent.
 */
export function useBootstrapSession() {
  const setSession = useAuthStore((s) => s.setSession);
  const clearSession = useAuthStore((s) => s.clearSession);
  const attempted = useRef(false);

  useEffect(() => {
    if (attempted.current) return;
    attempted.current = true;

    if (!authApi.hasRefreshCookie()) {
      clearSession();
      return;
    }

    authApi
      .refresh()
      .then((data) => setSession(data.user, data.access_token))
      .catch(() => clearSession());
  }, [setSession, clearSession]);
}
