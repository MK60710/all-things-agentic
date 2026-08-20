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
