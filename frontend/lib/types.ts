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
  retrieval_mode: "graph" | "vector" | "no_results";
}
