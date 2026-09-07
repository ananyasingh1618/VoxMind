// Mirrors apps/api/voxmind/schemas/*.py. Kept in sync by hand for now; if
// the backend and frontend types drift, that's a signal worth generating
// these from the OpenAPI schema in a later phase - not a Phase 1 concern.

export interface User {
  id: string;
  email: string;
  created_at: string;
}

export interface AccessTokenResponse {
  access_token: string;
  token_type: string;
  expires_in_seconds: number;
  user: User;
}

export interface Conversation {
  id: string;
  title: string | null;
  status: "active" | "ended";
  created_at: string;
  ended_at: string | null;
}

export type MessageRole = "user" | "assistant";

export interface Message {
  id: string;
  session_id: string;
  role: MessageRole;
  content: string;
  audio_asset_id: string | null;
  created_at: string;
}

export interface ApiErrorBody {
  code: string;
  message: string;
  detail?: unknown;
}

export interface AudioAsset {
  id: string;
  original_filename: string | null;
  content_type: string | null;
  size_bytes: number | null;
  duration_ms: number | null;
  sample_rate: number | null;
  channels: number | null;
  format: string | null;
  created_at: string;
}

export type ProcessingJobStatus = "pending" | "running" | "completed" | "failed";
export type DiarizationStatus = "completed" | "unavailable" | "failed";

export interface AudioProcessingJob {
  id: string;
  audio_asset_id: string;
  message_id: string | null;
  status: ProcessingJobStatus;
  error_code: string | null;
  error_message: string | null;
  diarization_status: DiarizationStatus | null;
  diarization_error: string | null;
  model_versions: Record<string, string | null>;
  stage_durations_ms: Record<string, number>;
  created_at: string;
  completed_at: string | null;
}

export interface TranscriptSegment {
  start_ms: number;
  end_ms: number;
  text: string;
  confidence: number | null;
}

export interface SpeakerSegment {
  speaker_label: string;
  start_ms: number;
  end_ms: number;
  confidence: number | null;
}

export interface AlignedTurn {
  id: string;
  speaker_label: string | null;
  start_ms: number;
  end_ms: number;
  text: string;
}

export interface Transcript {
  message_id: string;
  transcript_segments: TranscriptSegment[];
  speaker_segments: SpeakerSegment[];
  aligned_turns: AlignedTurn[];
}

export type EmotionProcessingJobStatus = "pending" | "running" | "completed" | "failed" | "unavailable";

export interface EmotionProcessingJob {
  id: string;
  message_id: string;
  model_version_id: string | null;
  status: EmotionProcessingJobStatus;
  error_code: string | null;
  error_message: string | null;
  turns_processed: number;
  stage_durations_ms: Record<string, number>;
  created_at: string;
  completed_at: string | null;
}

export interface EmotionPrediction {
  id: string;
  aligned_turn_id: string;
  model_version_id: string;
  predicted_label: string;
  confidence: number;
  probabilities: Record<string, number>;
  created_at: string;
}

export interface ModelVersion {
  id: string;
  component: string;
  version_tag: string;
  task: string | null;
  base_model: string | null;
  label_mapping: Record<string, string> | null;
  trained: boolean;
  trained_at: string | null;
  metrics: Record<string, unknown> | null;
  is_active: boolean;
  created_at: string;
}

// --- Phase 4: NLP ---

export interface Entity {
  text: string;
  label: string;
  start_char: number;
  end_char: number;
}

export interface NlpAnnotation {
  id: string;
  message_id: string;
  sentiment_label: "positive" | "neutral" | "negative";
  sentiment_score: number;
  intent_label: string;
  intent_confidence: number;
  topics: string[];
  entities: Entity[];
  model_versions: Record<string, string>;
  created_at: string;
}

// --- Phase 4: semantic-vocal incongruence ---

export interface IncongruenceSignal {
  id: string;
  aligned_turn_id: string;
  incongruence_score: number;
  semantic_signal: Record<string, unknown>;
  vocal_signal: Record<string, unknown>;
  confidence: number;
  explanation: string;
  signal_category: "analytical_not_diagnostic";
  created_at: string;
}

// --- Phase 4: knowledge / RAG ---

export type KnowledgeDocumentStatus = "pending" | "processing" | "completed" | "failed";

export interface KnowledgeDocument {
  id: string;
  session_id: string;
  title: string;
  original_filename: string;
  content_type: string;
  size_bytes: number;
  status: KnowledgeDocumentStatus;
  error_message: string | null;
  chunk_count: number;
  created_at: string;
  processed_at: string | null;
}

export interface RetrievedChunk {
  chunk_id: string;
  document_id: string;
  text: string;
  citation: string;
  vector_score: number | null;
  lexical_score: number | null;
  rerank_score: number | null;
}

export interface Citation {
  chunk_id: string;
  document_title: string;
  excerpt: string;
}

export type GroundingStatus = "grounded" | "partially_grounded" | "ungrounded" | "unavailable";

export interface Generation {
  id: string;
  provider: string;
  model_name: string | null;
  context_token_estimate: number;
  answer: string;
  raw_model_answer: string;
  citations: Citation[];
  confidence: number;
  evidence_summary: string;
  grounding_status: GroundingStatus;
  grounding_details: Record<string, unknown>;
  error_message: string | null;
  latency_ms: number;
  created_at: string;
}

export interface AskResponse {
  user_message: Message;
  assistant_message: Message | null;
  retrieval_result_id: string;
  retrieved_chunks: RetrievedChunk[];
  generation: Generation;
  guardrail: GuardrailInfo | null;
}

// --- Phase 4: memory ---

export interface ConversationSummary {
  id: string;
  session_id: string;
  summary_text: string;
  covers_message_ids: string[];
  method: "llm_generated" | "extractive_fallback";
  created_at: string;
}

export interface ConversationMemory {
  recent_turns: Message[];
  latest_summary: ConversationSummary | null;
}

// --- Phase 6: guardrails ---

export interface GuardrailInfo {
  decision: "approved" | "modified" | "blocked";
  reasons: string[];
  filtered_chunk_ids: string[];
  safety_flagged: boolean;
  safety_categories: string[];
}

// --- Phase 6: analytics ---

export interface LatencyStats {
  avg_ms: number | null;
  p50_ms: number | null;
  p95_ms: number | null;
  sample_count: number;
}

export interface LatencyOverview {
  stt: LatencyStats;
  analysis: LatencyStats;
  retrieval: LatencyStats;
  llm: LatencyStats;
  tts: LatencyStats;
  end_to_end: LatencyStats;
}

export interface LabelCount {
  label: string;
  count: number;
}

export interface SentimentTrendPoint {
  date: string;
  positive: number;
  neutral: number;
  negative: number;
}

export interface AnalyticsDashboard {
  conversations: { total: number; active: number; ended: number; total_messages: number };
  latency: LatencyOverview;
  emotion: { total_predictions: number; by_label: LabelCount[] };
  intelligence: {
    sentiment_distribution: LabelCount[];
    sentiment_trend: SentimentTrendPoint[];
    intent_distribution: LabelCount[];
    mismatch_event_count: number;
    mismatch_average_score: number | null;
    high_mismatch_event_count: number;
  };
  retrieval: {
    total_queries: number;
    queries_with_results: number;
    average_chunks_retrieved: number | null;
    total_documents: number;
    total_chunks: number;
    rerank_usage_rate: number | null;
  };
  grounding: { by_status: LabelCount[]; total_generations: number };
  model_versions: {
    component: string;
    version_tag: string;
    is_active: boolean;
    trained: boolean;
    metrics: Record<string, unknown> | null;
  }[];
  failures: {
    audio_processing_failed: number;
    emotion_processing_failed: number;
    voice_turns_failed: number;
    voice_turns_interrupted: number;
    llm_unavailable: number;
    guardrail_blocked: number;
    guardrail_modified: number;
  };
  pipeline_health: {
    llm_provider_configured: boolean;
    llm_provider: string;
    tts_provider_configured: boolean;
    tts_provider: string;
    diarization_available: boolean;
    moderation_provider: string;
  };
}

// --- Phase 6: conversation insights / intelligence timeline ---

export interface TimelineEmotionEntry {
  speaker_label: string | null;
  predicted_label: string;
  confidence: number;
}

export interface TimelineIncongruenceEntry {
  incongruence_score: number;
  confidence: number;
  explanation: string;
}

export interface TimelineEntry {
  message_id: string;
  role: string;
  content: string;
  created_at: string;
  sentiment_label: string | null;
  sentiment_score: number | null;
  intent_label: string | null;
  topics: string[];
  emotion: TimelineEmotionEntry[];
  incongruence: TimelineIncongruenceEntry[];
}

export interface Observation {
  text: string;
  basis: string;
}

export interface Interpretation {
  text: string;
  caveat: string;
}

export interface ConversationInsights {
  conversation_id: string;
  message_count: number;
  timeline: TimelineEntry[];
  observations: Observation[];
  interpretations: Interpretation[];
}

// --- Phase 5: real-time voice loop ---

export type VoiceUIState =
  | "idle"
  | "listening"
  | "processing"
  | "thinking"
  | "retrieving"
  | "generating"
  | "speaking"
  | "interrupted"
  | "error";

export interface VoiceStageEvent {
  stage: string;
  status?: string;
  [key: string]: unknown;
}

export interface VoiceTurnDoneEvent {
  stage: "done";
  status: "completed" | "partial" | "failed" | "interrupted";
  voice_turn_id: string;
  user_message_id: string;
  assistant_message_id: string | null;
  answer: string;
  citations: Citation[];
  grounding_status: GroundingStatus;
  audio_url: string | null;
  stage_latencies_ms: Record<string, number>;
}

export type VoiceTurnStatus = "pending" | "completed" | "partial" | "failed" | "interrupted";

export interface VoiceTurn {
  id: string;
  session_id: string;
  user_message_id: string | null;
  assistant_message_id: string | null;
  llm_generation_id: string | null;
  status: VoiceTurnStatus;
  error_code: string | null;
  error_message: string | null;
  audio_content_type: string | null;
  audio_provider: string | null;
  audio_duration_ms: number | null;
  stage_latencies_ms: Record<string, number>;
  created_at: string;
  completed_at: string | null;
}

// --- Phase 7: evaluation ---

export type EvaluationType = "stt" | "emotion" | "retrieval" | "grounding" | "system";
export type EvaluationRunStatus = "completed" | "partial" | "failed" | "unavailable";

export interface EvaluationRunSummary {
  id: string;
  evaluation_type: EvaluationType;
  dataset_version: string | null;
  model_version: string | null;
  configuration: Record<string, unknown>;
  code_version: string | null;
  random_seed: number | null;
  sample_count: number;
  status: EvaluationRunStatus;
  metrics: Record<string, unknown>;
  errors: unknown[];
  notes: string | null;
  started_at: string;
  completed_at: string | null;
  created_at: string;
}

export interface EvaluationTypeSummary {
  evaluation_type: EvaluationType;
  status: "evaluated" | "never_run";
  latest: EvaluationRunSummary | null;
}

export interface EvaluationDashboard {
  types: EvaluationTypeSummary[];
}
