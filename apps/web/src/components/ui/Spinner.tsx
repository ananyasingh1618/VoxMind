import { cn } from "@/lib/cn";

export function Spinner({ className }: { className?: string }) {
  return (
    <span
      role="status"
      aria-label="Loading"
      className={cn(
        "inline-block h-5 w-5 animate-spin rounded-full border-2",
        "border-[var(--color-border-strong)] border-t-[var(--color-accent)]",
        className,
      )}
    />
  );
}
