import { type InputHTMLAttributes, forwardRef, useId } from "react";

import { cn } from "@/lib/cn";

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label: string;
  error?: string;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ label, error, className, id, ...props }, ref) => {
    const generatedId = useId();
    const inputId = id ?? generatedId;
    const errorId = `${inputId}-error`;

    return (
      <div className="flex flex-col gap-1.5">
        <label htmlFor={inputId} className="text-sm font-medium text-[var(--color-text-secondary)]">
          {label}
        </label>
        <input
          ref={ref}
          id={inputId}
          aria-invalid={Boolean(error)}
          aria-describedby={error ? errorId : undefined}
          className={cn(
            "rounded-lg border border-[var(--color-border-strong)] bg-[var(--color-surface)] px-3 py-2",
            "text-sm text-[var(--color-text-primary)] placeholder:text-[var(--color-text-tertiary)]",
            "outline-none transition-colors focus:border-[var(--color-accent)]",
            error && "border-[var(--color-signal-negative)]",
            className,
          )}
          {...props}
        />
        {error && (
          <p id={errorId} role="alert" className="text-sm text-[var(--color-signal-negative)]">
            {error}
          </p>
        )}
      </div>
    );
  },
);
Input.displayName = "Input";
