export interface Citation {
  source_kind: "graph" | "chunk";
  paper_id?: string | null;
  text: string;
  section?: string | null;
  page_start?: number | null;
  page_end?: number | null;
  relation?: string | null;
  node_ids?: string[];
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  retrieval_mode: "general" | "graph" | "vector" | "no_results" | "ambiguous";
  confidence?: "confident" | "low";
  candidates?: Array<{ node_id: string; name: string; type: string; description: string; score: number }>;
  clarification_question_id?: string | null;
}

export interface ClarificationOption {
  id: string;
  label: string;
  description: string;
}

export interface PendingQuestion {
  id: string;
  kind: "entity_merge" | "query_disambiguation";
  question: string;
  options: ClarificationOption[];
  status: "open" | "answered";
  answer_option_id?: string | null;
  provisional_node_id?: string | null;
  candidate_node_id?: string | null;
  score?: number | null;
  query_text?: string | null;
}

export interface GapCitation {
  shared_node_id: string;
  shared_node_name: string;
  connects_to: "node_a" | "node_b";
  relation: string;
  source_paper_id?: string | null;
  source_section?: string | null;
  source_quote: string;
}

export interface GapCandidate {
  node_a_id: string;
  node_b_id: string;
  node_a_name: string;
  node_b_name: string;
  common_neighbor_ids: string[];
  score: number;
  explanation?: string | null;
  citations: GapCitation[];
}

export interface GraphVizNode {
  node_id: string;
  name: string;
  type?: string | null;
  reused_existing_node: boolean;
}

export interface GraphVizEdge {
  edge_id: string;
  source_id: string;
  target_id: string;
  relation: string;
}
