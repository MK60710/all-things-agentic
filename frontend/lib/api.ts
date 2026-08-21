import type { GapCandidate, GraphVizEdge, GraphVizNode, PaperGuide, PendingQuestion, QueryResponse } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL;

export interface ChatHistoryItem {
  role: "user" | "assistant";
  text: string;
}

export interface PaperContext {
  id: string;
  title: string;
  authors?: string;
  abstract?: string;
  pdfUrl?: string;
}

export interface PaperSearchResult extends PaperContext {
  published?: string;
}

export interface PaperIngestResult extends PaperContext {
  new_nodes: GraphVizNode[];
  new_edges: GraphVizEdge[];
}

export interface SessionMetadata {
  id: string;
  name: string;
  created_at: string;
}

export async function createSession(name: string): Promise<SessionMetadata> {
  const response = await fetch("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  const data = await response.json() as SessionMetadata & { error?: string };
  if (!response.ok) throw new Error(data.error ?? "Could not create the session");
  return data;
}

export async function listSessions(): Promise<SessionMetadata[]> {
  const response = await fetch("/api/sessions");
  const data = await response.json() as SessionMetadata[] & { error?: string };
  if (!response.ok) throw new Error((data as unknown as { error?: string }).error ?? "Could not load sessions");
  return data;
}

export async function listPapersForSession(sessionId: string): Promise<PaperContext[]> {
  const response = await fetch(`/api/papers?session_id=${encodeURIComponent(sessionId)}`);
  const data = await response.json() as Array<{
    id: string;
    title: string;
    authors?: string;
    abstract?: string;
    pdf_url?: string;
  }> & { error?: string };
  if (!response.ok) throw new Error((data as unknown as { error?: string }).error ?? "Could not load this session's papers");
  return data.map((paper) => ({
    id: paper.id,
    title: paper.title,
    authors: paper.authors,
    abstract: paper.abstract,
    pdfUrl: paper.pdf_url,
  }));
}

export async function detachPaper(paperId: string): Promise<void> {
  const response = await fetch(`/api/papers/${encodeURIComponent(paperId)}/detach`, { method: "POST" });
  if (!response.ok) {
    const data = await response.json().catch(() => ({})) as { error?: string };
    throw new Error(data.error ?? "Could not remove this paper from the session");
  }
}

export async function askAssistant(
  message: string,
  history: ChatHistoryItem[],
  papers?: PaperContext[] | null,
): Promise<QueryResponse> {
  const paperIds = papers?.map((paper) => paper.id) ?? [];
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history, paper_ids: paperIds }),
  });

  const data = await response.json() as Partial<QueryResponse> & { error?: string };
  if (!response.ok) throw new Error(data.error ?? `Assistant request failed (${response.status})`);
  return {
    answer: data.answer ?? "I couldn't produce a response.",
    citations: data.citations ?? [],
    retrieval_mode: data.retrieval_mode ?? (paperIds.length ? "vector" : "general"),
    confidence: data.confidence,
    candidates: data.candidates,
    clarification_question_id: data.clarification_question_id,
  };
}

export async function uploadPaper(
  file: File,
  signal?: AbortSignal,
  sessionId?: string,
): Promise<PaperIngestResult> {
  if (!API_URL) throw new Error("NEXT_PUBLIC_API_URL is not configured");
  const tokenResponse = await fetch("/api/papers/upload-token", { method: "POST", signal });
  const tokenData = await tokenResponse.json() as { token?: string; max_bytes?: number; error?: string };
  if (!tokenResponse.ok || !tokenData.token) {
    throw new Error(tokenData.error ?? "Could not authorize the upload");
  }
  if (tokenData.max_bytes && file.size > tokenData.max_bytes) {
    throw new Error("This PDF is larger than 25 MiB.");
  }
  const body = new FormData();
  body.append("file", file);
  body.append("title", file.name.replace(/\.pdf$/i, ""));
  if (sessionId) body.append("session_id", sessionId);
  const response = await fetch(`${API_URL.replace(/\/$/, "")}/papers`, {
    method: "POST",
    headers: { "X-Upload-Token": tokenData.token },
    body,
    signal,
  });
  const data = await response.json() as PaperIngestResult & { detail?: string };
  if (!response.ok) throw new Error(data.detail ?? `Paper upload failed (${response.status})`);
  return data;
}

export async function ingestArxivPaper(
  paper: PaperSearchResult,
  signal?: AbortSignal,
  sessionId?: string,
): Promise<PaperIngestResult> {
  const response = await fetch("/api/papers/arxiv", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      arxiv_id: paper.id.replace(/^arxiv:/, ""),
      title: paper.title,
      authors: paper.authors,
      abstract: paper.abstract,
      pdfUrl: paper.pdfUrl,
      session_id: sessionId,
    }),
    signal,
  });
  const data = await response.json() as PaperIngestResult & { error?: string };
  if (!response.ok) throw new Error(data.error ?? "Could not ingest the arXiv paper");
  return data;
}

export async function buildPaperGuide(paperId: string): Promise<PaperGuide> {
  const response = await fetch("/api/papers/guide", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ paper_id: paperId }),
  });
  const data = await response.json() as PaperGuide & { error?: string };
  if (!response.ok) throw new Error(data.error ?? "Could not build the paper walkthrough");
  return data;
}

export async function searchPapers(query: string): Promise<PaperSearchResult[]> {
  const response = await fetch(`/api/papers/search?q=${encodeURIComponent(query)}`);
  const data = await response.json() as { papers: PaperSearchResult[]; error?: string };
  if (!response.ok) throw new Error(data.error ?? "Paper search failed");
  return data.papers;
}

export async function listClarifications(sessionId?: string): Promise<PendingQuestion[]> {
  const query = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : "";
  const response = await fetch(`/api/clarifications${query}`);
  if (!response.ok) throw new Error("Could not load clarifications");
  return await response.json() as PendingQuestion[];
}

export async function answerClarification(id: string, optionId: string): Promise<PendingQuestion> {
  const response = await fetch(`/api/clarifications/${encodeURIComponent(id)}/answer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ option_id: optionId }),
  });
  const data = await response.json() as PendingQuestion & { error?: string };
  if (!response.ok) throw new Error((data as { error?: string }).error ?? "Could not answer this question");
  return data;
}

export async function listGaps(limit = 3): Promise<GapCandidate[]> {
  const response = await fetch(`/api/gaps?limit=${limit}`);
  if (!response.ok) throw new Error("Could not load suggestions");
  return await response.json() as GapCandidate[];
}

export async function recordGapFeedback(nodeAId: string, nodeBId: string, interesting: boolean): Promise<void> {
  const response = await fetch("/api/gaps/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ node_a_id: nodeAId, node_b_id: nodeBId, interesting }),
  });
  if (!response.ok) throw new Error("Could not record feedback");
}

export async function recordQueryFeedback(nodeId: string, helpful: boolean): Promise<void> {
  const response = await fetch("/api/query/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ node_id: nodeId, helpful }),
  });
  if (!response.ok) throw new Error("Could not record feedback");
}
