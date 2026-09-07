import { LogOut, Monitor, User as UserIcon } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { useLogout, useLogoutEverywhere } from "@/features/auth/hooks";
import { useAuthStore } from "@/stores/authStore";

export function UserMenu() {
  const [open, setOpen] = useState(false);
  const user = useAuthStore((s) => s.user);
  const navigate = useNavigate();
  const logout = useLogout();
  const logoutEverywhere = useLogoutEverywhere();

  async function handleLogout(everywhere: boolean) {
    setOpen(false);
    if (everywhere) {
      await logoutEverywhere.mutateAsync();
    } else {
      await logout.mutateAsync();
    }
    navigate("/login", { replace: true });
  }

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="menu"
        aria-expanded={open}
        className="flex w-full items-center gap-2 rounded-lg px-2 py-2 text-left text-sm text-[var(--color-text-secondary)] hover:bg-[var(--color-surface-raised)]"
      >
        <span className="flex h-7 w-7 items-center justify-center rounded-full bg-[var(--color-accent-muted)] text-[var(--color-accent-strong)]">
          <UserIcon size={14} />
        </span>
        <span className="truncate">{user?.email}</span>
      </button>

      {open && (
        <div
          role="menu"
          className="absolute bottom-full left-0 mb-2 w-full min-w-[14rem] rounded-lg border border-[var(--color-border-strong)] bg-[var(--color-surface-raised)] p-1 shadow-[var(--shadow-panel)]"
        >
          <button
            role="menuitem"
            type="button"
            onClick={() => handleLogout(false)}
            className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-sm text-[var(--color-text-primary)] hover:bg-[var(--color-surface)]"
          >
            <LogOut size={14} /> Log out
          </button>
          <button
            role="menuitem"
            type="button"
            onClick={() => handleLogout(true)}
            className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-sm text-[var(--color-text-primary)] hover:bg-[var(--color-surface)]"
          >
            <Monitor size={14} /> Log out of all devices
          </button>
        </div>
      )}
    </div>
  );
}
