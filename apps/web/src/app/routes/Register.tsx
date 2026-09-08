import { type FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { ApiError } from "@/api/client";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Input } from "@/components/ui/Input";
import { useLogin, useRegister } from "@/features/auth/hooks";

export function Register() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [formError, setFormError] = useState<string | null>(null);
  const register = useRegister();
  const login = useLogin();
  const navigate = useNavigate();

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setFormError(null);
    try {
      await register.mutateAsync({ email, password });
      await login.mutateAsync({ email, password });
      navigate("/app/conversations", { replace: true });
    } catch (error) {
      if (error instanceof ApiError && error.code === "conflict") {
        setFormError("An account with this email already exists.");
      } else if (error instanceof ApiError && error.code === "validation_error") {
        setFormError("Please use a valid email and a password with at least 8 characters.");
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
        <h1 className="mb-1 text-lg font-semibold">Create your account</h1>
        <p className="mb-6 text-sm text-[var(--color-text-secondary)]">
          Start a VoxMind conversation session.
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
            autoComplete="new-password"
            minLength={8}
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          {formError && (
            <p role="alert" className="text-sm text-[var(--color-signal-negative)]">
              {formError}
            </p>
          )}
          <Button type="submit" isLoading={register.isPending || login.isPending} className="mt-2">
            Create account
          </Button>
        </form>
        <p className="mt-5 text-center text-sm text-[var(--color-text-secondary)]">
          Already have an account?{" "}
          {/* axe-core `link-in-text-block` (WCAG 1.4.1, Use of Color): this
              link sits inline within body text - color alone (even on
              hover) isn't a sufficient visual distinction, so it's
              underlined by default now, not just on hover. */}
          <Link to="/login" className="text-[var(--color-accent-strong)] underline">
            Sign in
          </Link>
        </p>
      </Card>
    </main>
  );
}
