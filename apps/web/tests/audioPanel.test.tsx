import { screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import * as audioApi from "@/api/audio";
import { AudioPanel } from "@/components/conversation/AudioPanel";

import { renderWithProviders } from "./testUtils";

vi.mock("@/api/audio");

describe("AudioPanel", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("shows an empty state when no audio has been uploaded", async () => {
    vi.mocked(audioApi.listAudio).mockResolvedValue([]);

    renderWithProviders(<AudioPanel conversationId="conv-1" />);

    expect(await screen.findByText(/no audio uploaded yet/i)).toBeInTheDocument();
  });

  it("lists uploaded audio assets with a transcribe action", async () => {
    vi.mocked(audioApi.listAudio).mockResolvedValue([
      {
        id: "asset-1",
        original_filename: "recording.wav",
        content_type: "audio/wav",
        size_bytes: 2048,
        duration_ms: null,
        sample_rate: null,
        channels: null,
        format: null,
        created_at: "2026-01-01T00:00:00Z",
      },
    ]);

    renderWithProviders(<AudioPanel conversationId="conv-1" />);

    expect(await screen.findByText("recording.wav")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /transcribe/i })).toBeInTheDocument();
  });

  it("shows the not-yet-available transcript placeholder before anything is processed", async () => {
    vi.mocked(audioApi.listAudio).mockResolvedValue([]);

    renderWithProviders(<AudioPanel conversationId="conv-1" />);

    expect(await screen.findByText(/no transcript yet/i)).toBeInTheDocument();
  });
});
