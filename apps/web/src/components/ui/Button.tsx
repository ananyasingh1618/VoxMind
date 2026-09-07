import { type ButtonHTMLAttributes, forwardRef } from "react";

import { cn } from "@/lib/cn";

type Variant = "primary" | "secondary" | "ghost" | "danger";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  isLoading?: boolean;
}

const VARIANT_CLASSES: Record<Variant, string> = {
  primary: "bg-[var(--color-accent)] text-[#0b0b10] hover:bg-[var(--color-accent-strong)]",
  secondary:
    "bg-[var(--color-surface-raised)] text-[var(--color-text-primary)] border border-[var(--color-border-strong)] hover:border-[var(--color-accent)]",
  ghost: "bg-transparent text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]",
  danger: "bg-[var(--color-signal-negative)]/90 text-[#1a0a0c] hover:bg-[var(--color-signal-negative)]",
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ variant = "primary", isLoading, className, children, disabled, ...props }, ref) => {
    return (
      <button
        ref={ref}
        disabled={disabled || isLoading}
        className={cn(
          "inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-medium",
          "transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-50",
          VARIANT_CLASSES[variant],
          className,
        )}
        {...props}
      >
        {isLoading && (
          <span
            aria-hidden="true"
            className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-t-transparent"
          />
        )}
        {children}
      </button>
    );
  },
);
Button.displayName = "Button";
