import { Mic, Square } from "lucide-react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";

import { VoiceOrb } from "@/components/voice/VoiceOrb";
import { useVoiceTurn } from "@/features/voice/useVoiceTurn";
import type { VoiceUIState } from "@/types/api";

const STATUS_LABEL: Record<VoiceUIState, string> = {
  idle: "Tap the microphone to speak",
  listening: "Listening… tap again to stop",
  processing: "Transcribing your speech…",
  thinking: "Analyzing emotion, sentiment, and memory…",
  retrieving: "Retrieving relevant knowledge…",
  generating: "Generating a response…",
  speaking: "Speaking the response…",
  interrupted: "Interrupted",
  error: "Something went wrong",
};

const LATENCY_LABELS: Record<string, string> = {
  stt_ms: "Speech-to-text",
  analysis_ms: "Emotion/NLP analysis",
  retrieval_ms: "Knowledge retrieval",
  llm_ms: "LLM generation",
  tts_ms: "Speech synthesis",
  total_ms: "Total",
};

export function VoiceSessionControl({ conversationId }: { conversationId: string }) {
  const voice = useVoiceTurn(conversationId);
  const prefersReducedMotion = useReducedMotion();

  const isListening = voice.state === "listening";
  const isBusy = ["processing", "thinking", "retrieving", "generating"].includes(voice.state);
  const isSpeaking = voice.state === "speaking";
  const canInterrupt = isBusy || isSpeaking;

  async function handleMicClick() {
    if (isListening) {
      await voice.stopAndSend();
      return;
    }
    if (voice.state !== "idle" && voice.state !== "interrupted" && voice.state !== "error") return;
    await voice.startListening();
  }

  return (
    <div className="flex flex-col items-center gap-3 border-b border-[var(--color-border)] py-6">
      <VoiceOrb state={voice.state} level={voice.level} />

      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={handleMicClick}
          disabled={
            voice.recorderStatus === "requesting" ||
            voice.recorderStatus === "unsupported" ||
            (isBusy && !isListening)
          }
          aria-pressed={isListening}
          aria-label={isListening ? "Stop and send" : "Start speaking"}
          className="flex h-12 w-12 items-center justify-center rounded-full border border-[var(--color-border-strong)] bg-[var(--color-surface-raised)] text-[var(--color-text-primary)] transition-colors hover:border-[var(--color-accent)] disabled:opacity-50"
        >
          {isListening ? <Square size={16} /> : <Mic size={18} />}
        </button>

        <AnimatePresence>
          {canInterrupt && (
            <motion.button
              type="button"
              initial={prefersReducedMotion ? false : { opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={prefersReducedMotion ? undefined : { opacity: 0, scale: 0.8 }}
              transition={prefersReducedMotion ? { duration: 0 } : undefined}
              onClick={voice.interrupt}
              aria-label="Interrupt"
              className="flex h-9 items-center gap-1.5 rounded-full border border-[var(--color-signal-negative)] px-3 text-xs font-medium text-[var(--color-signal-negative)] transition-colors hover:bg-[var(--color-signal-negative)]/10"
            >
              <Square size={11} fill="currentColor" /> Stop
            </motion.button>
          )}
        </AnimatePresence>
      </div>

      <p role="status" className="max-w-xs text-center text-sm text-[var(--color-text-secondary)]">
        {voice.recorderStatus === "denied"
          ? "Microphone access was denied. Enable it in your browser settings to continue."
          : voice.recorderStatus === "unsupported"
            ? "This browser doesn't support microphone capture."
            : STATUS_LABEL[voice.state]}
      </p>

      {voice.transcript && (
        <p className="max-w-sm text-center text-xs text-[var(--color-text-tertiary)]">
          &ldquo;{voice.transcript}&rdquo;
        </p>
      )}

      {voice.errorMessage && (
        <p role="alert" className="max-w-xs text-center text-xs text-[var(--color-signal-negative)]">
          {voice.errorMessage}
        </p>
      )}

      {voice.stageLatenciesMs && (
        <details className="w-full max-w-xs text-xs text-[var(--color-text-tertiary)]">
          <summary className="cursor-pointer text-center">Timing breakdown</summary>
          <dl className="mt-1 flex flex-col gap-0.5 px-2">
            {Object.entries(voice.stageLatenciesMs).map(([key, ms]) => (
              <div key={key} className="flex justify-between">
                <dt>{LATENCY_LABELS[key] ?? key}</dt>
                <dd>{ms} ms</dd>
              </div>
            ))}
          </dl>
        </details>
      )}
    </div>
  );
}
