import { screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ConversationWorkspacePage } from "@/app/routes/app/ConversationWorkspacePage";
import * as conversationHooks from "@/features/conversations/hooks";
import type { Conversation } from "@/types/api";

import { renderWithProviders } from "./testUtils";

vi.mock("@/features/conversations/hooks");
vi.mock("@/components/voice/VoiceSessionControl", () => ({
  VoiceSessionControl: () => <div>voice-session-control</div>,
}));
vi.mock("@/components/conversation/MessageTimeline", () => ({
  MessageTimeline: () => <div>message-timeline</div>,
}));
vi.mock("@/components/conversation/MessageComposer", () => ({
  MessageComposer: () => <div>message-composer</div>,
}));
vi.mock("@/components/conversation/AudioPanel", () => ({
  AudioPanel: () => <div>audio-panel-content</div>,
}));
vi.mock("@/components/conversation/IntelligencePanel", () => ({
  IntelligencePanel: () => <div>intelligence-panel-content</div>,
}));
vi.mock("@/components/conversation/KnowledgePanel", () => ({
  KnowledgePanel: () => <div>knowledge-panel-content</div>,
}));

function mockConversation(overrides: Partial<Conversation> = {}) {
  const conversation: Conversation = {
    id: "conv-1",
    title: "Test session",
    status: "active",
    created_at: "2026-01-01T00:00:00Z",
    ended_at: null,
    ...overrides,
  };

  vi.mocked(conversationHooks.useConversation).mockReturnValue({
    data: conversation,
    isLoading: false,
    isError: false,
    refetch: vi.fn(),
  } as unknown as ReturnType<typeof conversationHooks.useConversation>);
  vi.mocked(conversationHooks.useMessages).mockReturnValue({
    data: [],
    isLoading: false,
    isError: false,
    refetch: vi.fn(),
  } as unknown as ReturnType<typeof conversationHooks.useMessages>);
  vi.mocked(conversationHooks.useSendMessage).mockReturnValue({
    mutateAsync: vi.fn(),
    isPending: false,
  } as unknown as ReturnType<typeof conversationHooks.useSendMessage>);
}

function isHidden(node: Element | null): boolean {
  return !!node && node.className.split(/\s+/).includes("hidden");
}

describe("ConversationWorkspacePage responsive tabs", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    mockConversation();
  });

  it("shows chat by default and keeps the other panels reachable but not both visible on mobile", () => {
    renderWithProviders(<ConversationWorkspacePage />);

    expect(screen.getByText("voice-session-control")).toBeInTheDocument();
    const audioContent = screen.getByText("audio-panel-content");
    // The audio panel's mobile wrapper is hidden until its tab is selected -
    // it must never be simultaneously visible with chat on a narrow screen.
    expect(isHidden(audioContent.closest("[class*='rounded-none']"))).toBe(true);
  });

  it("switching to the Audio tab reveals the audio panel wrapper", async () => {
    const { default: userEvent } = await import("@testing-library/user-event");
    renderWithProviders(<ConversationWorkspacePage />);

    await userEvent.click(screen.getByRole("button", { name: /^audio$/i }));

    const audioContent = screen.getByText("audio-panel-content");
    expect(isHidden(audioContent.closest("[class*='rounded-none']"))).toBe(false);
  });

  it("every panel tab is reachable via a labeled, stateful control", () => {
    renderWithProviders(<ConversationWorkspacePage />);

    for (const name of [/^chat$/i, /^audio$/i, /^intelligence$/i, /^knowledge$/i]) {
      const button = screen.getByRole("button", { name });
      expect(button).toHaveAttribute("aria-pressed");
    }
  });
});
