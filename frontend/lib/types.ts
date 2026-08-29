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

export interface NodeCitation {
  paper_id: string;
  section?: string | null;
  source_quote: string;
}

export interface SessionGraphNode {
  node_id: string;
  name: string;
  type: string;
  description: string;
  citations: NodeCitation[];
}

export interface SessionGraphEdge {
  edge_id: string;
  source_id: string;
  target_id: string;
  relation: string;
  source_paper_id?: string | null;
  source_section?: string | null;
  source_quote?: string;
}

export interface FeynmanPrompt {
  node_id: string;
  node_name: string;
  question: string;
}

export interface FeynmanCheckResult {
  node_id: string;
  node_name: string;
  verdict: "strong" | "weak" | "wrong";
  explanation: string;
  citation?: Citation | null;
}

export interface ContradictionCandidate {
  claim_a_id: string;
  claim_b_id: string;
  claim_a_text: string;
  claim_b_text: string;
  explanation: string;
  edge_id: string;
}

export interface GuideDiagramNode {
  id: string;
  label: string;
  detail: string;
}

export interface GuideDiagram {
  title: string;
  nodes: GuideDiagramNode[];
  edges: Array<{ source: string; target: string; label: string }>;
}

export interface GuideSection {
  title: string;
  plain_language: string;
  key_points: string[];
  why_it_matters: string;
  page_start?: number | null;
  page_end?: number | null;
  diagram?: GuideDiagram | null;
}

export interface PaperGuide {
  title: string;
  big_picture: string;
  reading_time_minutes: number;
  sections: GuideSection[];
}

export interface DeepDiveSource {
  text: string;
  section?: string | null;
  page_start?: number | null;
  page_end?: number | null;
}

export interface DeepDiveSection extends GuideSection {
  section_id: string;
  sources: DeepDiveSource[];
}

export interface DeepDiveResponse {
  paper_id: string;
  title: string;
  big_picture: string;
  reading_time_minutes: number;
  sections: DeepDiveSection[];
}

export interface PaperConnectionEvidence {
  topic: string;
  paper_id: string;
  section?: string | null;
  quote: string;
}

export interface PaperConnection {
  paper_a_id: string;
  paper_a_title: string;
  paper_b_id: string;
  paper_b_title: string;
  summary: string;
  shared_topics: string[];
  evidence: PaperConnectionEvidence[];
}
