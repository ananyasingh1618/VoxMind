import { LogOut, Monitor, User as UserIcon } from "lucide-react";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import { useLogout, useLogoutEverywhere } from "@/features/auth/hooks";
import { useAuthStore } from "@/stores/authStore";

export function UserMenu() {
  const [open, setOpen] = useState(false);
  const user = useAuthStore((s) => s.user);
  const navigate = useNavigate();
  const logout = useLogout();
  const logoutEverywhere = useLogoutEverywhere();

  // Manual accessibility review: a keyboard user could already close this
  // by tabbing back to the trigger and re-pressing Enter/Space (native
  // <button> behavior), but Escape-to-close is the conventional,
  // expected pattern for any disclosure widget and was missing entirely.
  useEffect(() => {
    if (!open) return;
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") setOpen(false);
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [open]);

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
      {/* Manual accessibility review (final hardening pass): `role="menu"`/
          `role="menuitem"` previously promised the full ARIA menu keyboard
          pattern (arrow-key navigation between items, Escape to close) per
          the WAI-ARIA Authoring Practices - a role a screen reader
          announces as a real contract with the user. That pattern was
          never actually implemented here (a plain click/Tab-order
          dropdown of two action buttons), which is a real, if subtle,
          accessibility defect on its own - a role promising behavior
          that doesn't exist is worse than no role at all. Fixed by
          dropping the menu roles (this is honestly just a disclosure
          toggle, not an application menu) and keeping only
          `aria-expanded`, which accurately reflects real, existing state
          regardless of role. */}
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center gap-2 rounded-lg px-2 py-2 text-left text-sm text-[var(--color-text-secondary)] hover:bg-[var(--color-surface-raised)]"
      >
        <span className="flex h-7 w-7 items-center justify-center rounded-full bg-[var(--color-accent-muted)] text-[var(--color-accent-strong)]">
          <UserIcon size={14} />
        </span>
        <span className="truncate">{user?.email}</span>
      </button>

      {open && (
        <div className="absolute bottom-full left-0 mb-2 w-full min-w-[14rem] rounded-lg border border-[var(--color-border-strong)] bg-[var(--color-surface-raised)] p-1 shadow-[var(--shadow-panel)]">
          <button
            type="button"
            onClick={() => handleLogout(false)}
            className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-sm text-[var(--color-text-primary)] hover:bg-[var(--color-surface)]"
          >
            <LogOut size={14} /> Log out
          </button>
          <button
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
