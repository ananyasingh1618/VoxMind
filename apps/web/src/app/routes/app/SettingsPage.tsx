import { Card } from "@/components/ui/Card";
import { useAuthStore } from "@/stores/authStore";

export function SettingsPage() {
  const user = useAuthStore((s) => s.user);

  return (
    <div className="mx-auto max-w-xl p-6">
      <h1 className="mb-4 text-lg font-semibold">Settings</h1>
      <Card className="p-5">
        <h2 className="mb-3 text-sm font-medium text-[var(--color-text-secondary)]">Account</h2>
        <dl className="grid grid-cols-[8rem_1fr] gap-y-2 text-sm">
          <dt className="text-[var(--color-text-tertiary)]">Email</dt>
          <dd>{user?.email}</dd>
          <dt className="text-[var(--color-text-tertiary)]">Account ID</dt>
          <dd className="truncate font-mono text-xs">{user?.id}</dd>
          <dt className="text-[var(--color-text-tertiary)]">Member since</dt>
          <dd>{user?.created_at ? new Date(user.created_at).toLocaleDateString() : "—"}</dd>
        </dl>
      </Card>
    </div>
  );
}
