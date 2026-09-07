import { motion, useReducedMotion } from "framer-motion";

import type { VoiceUIState } from "@/types/api";

const STATE_COLORS: Record<VoiceUIState, string> = {
  idle: "var(--color-text-tertiary)",
  listening: "var(--color-accent)",
  processing: "#7dd3fc",
  thinking: "#a78bfa",
  retrieving: "#a78bfa",
  generating: "#f0abfc",
  speaking: "var(--color-signal-positive)",
  interrupted: "#f59e0b",
  error: "var(--color-signal-negative)",
};

const BUSY_STATES: VoiceUIState[] = ["processing", "thinking", "retrieving", "generating"];

/**
 * The voice visualization - its color and motion pattern are driven
 * directly by the real backend pipeline state (`VoiceUIState`, set from
 * genuine SSE stage events - see features/voice/useVoiceTurn.ts), not a
 * decorative loop. `level` (real mic input amplitude, from
 * useVoiceRecorder) only modulates the "listening" pulse; every other state
 * has its own fixed motion so the UI is honest about which real stage the
 * pipeline is in.
 */
export function VoiceOrb({ state, level = 0 }: { state: VoiceUIState; level?: number }) {
  const prefersReducedMotion = useReducedMotion();
  const color = STATE_COLORS[state];
  const isBusy = BUSY_STATES.includes(state);
  const isListening = state === "listening";
  const isSpeaking = state === "speaking";
  const isInterruptedOrError = state === "interrupted" || state === "error";

  return (
    <div className="relative flex h-28 w-28 items-center justify-center">
      {isBusy && !prefersReducedMotion && (
        <motion.span
          aria-hidden="true"
          className="absolute h-full w-full rounded-full"
          style={{
            background: `conic-gradient(from 0deg, transparent, ${color}, transparent 60%)`,
          }}
          animate={{ rotate: 360 }}
          transition={{ repeat: Infinity, duration: 1.6, ease: "linear" }}
        />
      )}

      <motion.span
        aria-hidden="true"
        className="absolute h-[85%] w-[85%] rounded-full"
        style={{
          background: `radial-gradient(circle at 35% 30%, ${color}, ${color}99 55%, transparent 75%)`,
        }}
        animate={
          prefersReducedMotion
            ? { scale: state === "idle" ? 1 : 1.05, opacity: state === "idle" ? 0.5 : 0.9 }
            : isListening
              ? { scale: 1 + level * 0.35, opacity: 0.85 + level * 0.15 }
              : isSpeaking
                ? { scale: [1, 1.08, 1], opacity: [0.85, 1, 0.85] }
                : isInterruptedOrError
                  ? { scale: [1.15, 1], opacity: [1, 0.7] }
                  : isBusy
                    ? { scale: [0.95, 1.02, 0.95], opacity: [0.7, 0.95, 0.7] }
                    : { scale: 1, opacity: 0.5 }
        }
        transition={
          isListening
            ? { type: "spring", stiffness: 220, damping: 20 }
            : isSpeaking || isBusy
              ? { repeat: Infinity, duration: isSpeaking ? 1.1 : 1.4, ease: "easeInOut" }
              : { duration: 0.4 }
        }
      />
      <span className="relative h-3 w-3 rounded-full bg-[var(--color-canvas)]" />
    </div>
  );
}
