import { AlertCircle, FileAudio, Upload } from "lucide-react";
import { useRef, useState } from "react";

import { ApiError } from "@/api/client";
import { TranscriptPanel } from "@/components/conversation/TranscriptPanel";
import { Button } from "@/components/ui/Button";
import { Spinner } from "@/components/ui/Spinner";
import { useAudioAssets, useProcessAudio, useTranscript, useUploadAudio } from "@/features/audio/hooks";
import { cn } from "@/lib/cn";
import type { AudioProcessingJob } from "@/types/api";

function formatBytes(bytes: number | null): string {
  if (bytes === null) return "";
  if (bytes < 1024) return `${bytes} B`;
  return `${(bytes / 1024).toFixed(0)} KB`;
}

function uploadErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.code === "unsupported_audio_format") {
      return "That file isn't a supported audio format (wav, mp3, m4a, ogg, webm).";
    }
    if (error.code === "audio_too_large") {
      return "That file is too large.";
    }
    return error.message;
  }
  return "Upload failed. Please try again.";
}

function processErrorMessage(error: unknown): string {
  return error instanceof ApiError ? error.message : "Processing failed. Please try again.";
}

export function AudioPanel({
  conversationId,
  onSelectMessage,
}: {
  conversationId: string;
  /** Notified whenever the user selects (or a new upload produces) a
   * processed transcript, so a sibling panel (emotion analysis) can follow
   * along without duplicating this component's upload/processing state. */
  onSelectMessage?: (messageId: string | null) => void;
}) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { data: assets, isLoading, isError } = useAudioAssets(conversationId);
  const upload = useUploadAudio(conversationId);
  const process = useProcessAudio(conversationId);
  const [jobsByAsset, setJobsByAsset] = useState<Record<string, AudioProcessingJob>>({});
  const [selectedJob, setSelectedJob] = useState<AudioProcessingJob | null>(null);
  const [processingAssetId, setProcessingAssetId] = useState<string | null>(null);

  const transcriptQuery = useTranscript(conversationId, selectedJob?.message_id ?? null);

  function selectJob(job: AudioProcessingJob) {
    setSelectedJob(job);
    onSelectMessage?.(job.message_id);
  }

  async function handleFileSelected(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    await upload.mutateAsync(file);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  async function handleProcess(audioAssetId: string) {
    setProcessingAssetId(audioAssetId);
    try {
      const job = await process.mutateAsync(audioAssetId);
      setJobsByAsset((prev) => ({ ...prev, [audioAssetId]: job }));
      if (job.status === "completed" && job.message_id) {
        selectJob(job);
      }
    } finally {
      setProcessingAssetId(null);
    }
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-1.5 border-b border-[var(--color-border)] p-2 text-xs font-medium text-[var(--color-text-secondary)]">
        <FileAudio size={13} /> Audio & transcription
      </div>

      <div className="border-b border-[var(--color-border)] p-3">
        <input
          ref={fileInputRef}
          type="file"
          accept=".wav,.mp3,.m4a,.ogg,.webm,audio/*"
          onChange={handleFileSelected}
          className="hidden"
          id="audio-upload-input"
        />
        <Button
          variant="secondary"
          className="w-full justify-center"
          isLoading={upload.isPending}
          onClick={() => fileInputRef.current?.click()}
        >
          <Upload size={14} /> Upload audio
        </Button>
        {upload.isError && (
          <p
            role="alert"
            className="mt-2 flex items-center gap-1 text-xs text-[var(--color-signal-negative)]"
          >
            <AlertCircle size={12} /> {uploadErrorMessage(upload.error)}
          </p>
        )}
      </div>

      <div className="max-h-40 overflow-y-auto border-b border-[var(--color-border)]">
        {isLoading && (
          <div className="flex justify-center py-4">
            <Spinner />
          </div>
        )}
        {isError && (
          <p className="p-3 text-xs text-[var(--color-signal-negative)]">Couldn't load audio files.</p>
        )}
        {!isLoading && !isError && (assets ?? []).length === 0 && (
          <p className="p-3 text-xs text-[var(--color-text-tertiary)]">
            No audio uploaded yet for this conversation.
          </p>
        )}
        {(assets ?? []).map((asset) => {
          const job = jobsByAsset[asset.id];
          const isProcessingThis = processingAssetId === asset.id && process.isPending;
          return (
            <div
              key={asset.id}
              className={cn(
                "flex items-center justify-between gap-2 border-b border-[var(--color-border)] px-3 py-2 text-xs last:border-b-0",
                job && selectedJob?.id === job.id && "bg-[var(--color-surface-raised)]",
              )}
            >
              <button
                type="button"
                onClick={() => job?.message_id && selectJob(job)}
                disabled={!job?.message_id}
                className="flex min-w-0 flex-1 items-center gap-2 text-left disabled:cursor-default"
              >
                <FileAudio size={13} className="shrink-0 text-[var(--color-text-tertiary)]" />
                <span className="truncate">{asset.original_filename ?? "audio"}</span>
                <span className="shrink-0 text-[var(--color-text-tertiary)]">
                  {formatBytes(asset.size_bytes)}
                </span>
              </button>
              {!job && (
                <Button
                  variant="ghost"
                  className="shrink-0 px-2 py-1 text-xs"
                  isLoading={isProcessingThis}
                  onClick={() => handleProcess(asset.id)}
                >
                  Transcribe
                </Button>
              )}
              {job?.status === "completed" && (
                <span className="shrink-0 text-[var(--color-signal-positive)]">Done</span>
              )}
              {job?.status === "failed" && (
                <span
                  className="shrink-0 text-[var(--color-signal-negative)]"
                  title={job.error_message ?? ""}
                >
                  Failed
                </span>
              )}
            </div>
          );
        })}
      </div>

      {process.isError && (
        <p role="alert" className="p-2 text-xs text-[var(--color-signal-negative)]">
          {processErrorMessage(process.error)}
        </p>
      )}

      <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
        {transcriptQuery.isLoading && selectedJob && (
          <div className="flex justify-center py-6">
            <Spinner />
          </div>
        )}
        <div className="min-h-0 flex-1 overflow-y-auto">
          <TranscriptPanel
            segments={transcriptQuery.data?.transcript_segments}
            turns={transcriptQuery.data?.aligned_turns}
          />
        </div>
        {selectedJob?.diarization_status === "unavailable" && (
          <p className="border-t border-[var(--color-border)] p-2 text-xs text-[var(--color-text-tertiary)]">
            Speaker diarization wasn't available for this transcript (not configured on this server).
          </p>
        )}
      </div>
    </div>
  );
}
