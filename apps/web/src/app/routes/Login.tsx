import { type FormEvent, useState } from "react";
import { Link, type Location, useLocation, useNavigate } from "react-router-dom";

import { ApiError } from "@/api/client";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Input } from "@/components/ui/Input";
import { useLogin } from "@/features/auth/hooks";

export function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [formError, setFormError] = useState<string | null>(null);
  const login = useLogin();
  const navigate = useNavigate();
  const location = useLocation();

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setFormError(null);
    try {
      await login.mutateAsync({ email, password });
      const fromLocation = (location.state as { from?: Location } | null)?.from;
      navigate(fromLocation?.pathname ?? "/app/conversations", { replace: true });
    } catch (error) {
      if (error instanceof ApiError && error.code === "invalid_credentials") {
        setFormError("Incorrect email or password.");
      } else {
        setFormError("Something went wrong. Please try again.");
      }
    }
  }

  return (
    // axe-core `landmark-one-main`/`region`: this page renders outside the
    // authenticated app shell (which already has its own <main>), so it
    // needs its own landmark - was a bare <div> before.
    <main className="flex min-h-dvh items-center justify-center bg-[var(--color-canvas)] px-4">
      <Card className="w-full max-w-sm p-6">
        <h1 className="mb-1 text-lg font-semibold">Sign in</h1>
        <p className="mb-6 text-sm text-[var(--color-text-secondary)]">
          Welcome back to VoxMind.
        </p>
        <form onSubmit={handleSubmit} className="flex flex-col gap-4" noValidate>
          <Input
            label="Email"
            type="email"
            autoComplete="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
          <Input
            label="Password"
            type="password"
            autoComplete="current-password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          {formError && (
            <p role="alert" className="text-sm text-[var(--color-signal-negative)]">
              {formError}
            </p>
          )}
          <Button type="submit" isLoading={login.isPending} className="mt-2">
            Sign in
          </Button>
        </form>
        <p className="mt-5 text-center text-sm text-[var(--color-text-secondary)]">
          Don&apos;t have an account?{" "}
          {/* axe-core `link-in-text-block` (WCAG 1.4.1, Use of Color): this
              link sits inline within body text - color alone (even on
              hover) isn't a sufficient visual distinction, so it's
              underlined by default now, not just on hover. */}
          <Link to="/register" className="text-[var(--color-accent-strong)] underline">
            Create one
          </Link>
        </p>
      </Card>
    </main>
  );
}
