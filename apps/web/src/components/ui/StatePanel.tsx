import type { ReactNode } from "react";

import { Button } from "@/components/ui/Button";
import { Spinner } from "@/components/ui/Spinner";

/**
 * Shared shape for the loading/empty/error states every data-driven view
 * needs (rule: every screen implements all four states explicitly, not just
 * the happy path).
 */

export function LoadingPanel({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-16 text-[var(--color-text-secondary)]">
      <Spinner />
      <p className="text-sm">{label}</p>
    </div>
  );
}

export function EmptyPanel({
  title,
  description,
  action,
  icon,
  // axe-core `page-has-heading-one`/`heading-order`: this component is
  // reused across many different pages/panels, each with a different
  // surrounding heading context - most already sit under a real page-level
  // <h1>/<h2>, where <h3> (the default, unchanged) is correct. A handful
  // of call sites (e.g. ConversationsListPage's empty state) render this
  // as the *only* content on the page, where it must itself be the page's
  // <h1> - `headingLevel` lets those opt in without moving every other
  // call site off its already-correct default.
  headingLevel = "h3",
}: {
  title: string;
  description?: string;
  action?: ReactNode;
  icon?: ReactNode;
  headingLevel?: "h1" | "h2" | "h3";
}) {
  const Heading = headingLevel;
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-16 text-center">
      {icon}
      <Heading className="text-base font-medium text-[var(--color-text-primary)]">{title}</Heading>
      {description && (
        <p className="max-w-sm text-sm text-[var(--color-text-secondary)]">{description}</p>
      )}
      {action}
    </div>
  );
}

export function ErrorPanel({
  title = "Something went wrong",
  description,
  onRetry,
}: {
  title?: string;
  description?: string;
  onRetry?: () => void;
}) {
  return (
    <div
      role="alert"
      className="flex flex-col items-center justify-center gap-3 py-16 text-center"
    >
      <h3 className="text-base font-medium text-[var(--color-signal-negative)]">{title}</h3>
      {description && (
        <p className="max-w-sm text-sm text-[var(--color-text-secondary)]">{description}</p>
      )}
      {onRetry && (
        <Button variant="secondary" onClick={onRetry}>
          Try again
        </Button>
      )}
    </div>
  );
}
