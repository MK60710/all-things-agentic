export interface Citation {
  source_kind: "graph" | "chunk";
  paper_id?: string | null;
  text: string;
  section?: string | null;
  page_start?: number | null;
  page_end?: number | null;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  retrieval_mode: "general" | "graph" | "vector" | "no_results" | "ambiguous";
  confidence?: "confident" | "low";
  candidates?: Array<{ node_id: string; name: string; type: string; description: string; score: number }>;
  clarification_question_id?: string | null;
}
