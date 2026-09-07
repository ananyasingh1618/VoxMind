import { Link } from "react-router-dom";

export function Landing() {
  return (
    <div className="flex min-h-dvh flex-col items-center justify-center gap-8 bg-[var(--color-canvas)] px-6 text-center">
      <div
        aria-hidden="true"
        className="h-24 w-24 rounded-full bg-[radial-gradient(circle_at_30%_30%,var(--color-accent-strong),transparent_70%)] opacity-80 blur-[2px]"
      />
      <div className="max-w-xl space-y-3">
        <h1 className="text-3xl font-semibold tracking-tight text-[var(--color-text-primary)]">
          VoxMind
        </h1>
        <p className="text-[var(--color-text-secondary)]">
          A conversational intelligence platform that listens, understands tone and language, and
          responds with grounded, cited knowledge.
        </p>
      </div>
      <div className="flex gap-3">
        <Link
          to="/register"
          className="inline-flex items-center justify-center rounded-lg bg-[var(--color-accent)] px-4 py-2 text-sm font-medium text-[#0b0b10] transition-colors hover:bg-[var(--color-accent-strong)]"
        >
          Get started
        </Link>
        <Link
          to="/login"
          className="inline-flex items-center justify-center rounded-lg border border-[var(--color-border-strong)] bg-[var(--color-surface-raised)] px-4 py-2 text-sm font-medium text-[var(--color-text-primary)] transition-colors hover:border-[var(--color-accent)]"
        >
          Sign in
        </Link>
      </div>
    </div>
  );
}
