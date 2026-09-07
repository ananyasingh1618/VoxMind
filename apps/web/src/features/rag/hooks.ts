import { useMutation, useQueryClient } from "@tanstack/react-query";

import * as ragApi from "@/api/rag";
import { conversationKeys } from "@/features/conversations/hooks";

export function useAsk(conversationId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (question: string) => ragApi.ask(conversationId, question),
    onSuccess: () => {
      // The real pipeline creates a user Message and (if a provider is
      // configured) an assistant Message - the main timeline should reflect
      // both immediately.
      queryClient.invalidateQueries({ queryKey: conversationKeys.messages(conversationId) });
    },
  });
}
