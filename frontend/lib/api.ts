import type { ContradictionCandidate, FeynmanCheckResult, FeynmanPrompt, GapCandidate, GraphVizEdge, GraphVizNode, PaperGuide, PendingQuestion, QueryResponse, SessionGraphEdge, SessionGraphNode } from "./types";

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
  goal?: string | null;
}

export async function createSession(name: string, goal?: string): Promise<SessionMetadata> {
  const response = await fetch("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, goal: goal || undefined }),
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

export async function renameSession(sessionId: string, name: string): Promise<SessionMetadata> {
  const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  const data = await response.json() as SessionMetadata & { error?: string };
  if (!response.ok) throw new Error(data.error ?? "Could not rename the session");
  return data;
}

export async function deleteSession(sessionId: string): Promise<void> {
  const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
  if (!response.ok) {
    const data = await response.json().catch(() => ({})) as { error?: string };
    throw new Error(data.error ?? "Could not delete the session");
  }
}

export interface SessionMessagesResult {
  // Left as opaque records here - lib/api.ts doesn't know page.tsx's own
  // Message shape, and re-declaring/importing it here just to round-trip
  // it back out would be redundant; callers cast to Message[] themselves.
  messages: Record<string, unknown>[];
  compacted: boolean;
}

export async function getSessionMessages(sessionId: string): Promise<SessionMessagesResult> {
  const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/messages`);
  const data = await response.json() as SessionMessagesResult & { error?: string };
  if (!response.ok) throw new Error(data.error ?? "Could not load this session's messages");
  return data;
}

export async function saveSessionMessages(
  sessionId: string,
  messages: Record<string, unknown>[],
): Promise<SessionMessagesResult> {
  const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/messages`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
  });
  const data = await response.json() as SessionMessagesResult & { error?: string };
  if (!response.ok) throw new Error(data.error ?? "Could not save this session's messages");
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

export async function detachPaper(paperId: string, sessionId: string): Promise<void> {
  const response = await fetch(`/api/papers/${encodeURIComponent(paperId)}/detach?session_id=${encodeURIComponent(sessionId)}`, { method: "POST" });
  if (!response.ok) {
    const data = await response.json().catch(() => ({})) as { error?: string };
    throw new Error(data.error ?? "Could not remove this paper from the session");
  }
}

export async function askAssistant(
  message: string,
  history: ChatHistoryItem[],
  papers?: PaperContext[] | null,
  goal?: string | null,
  nodeId?: string,
  sessionId?: string,
): Promise<QueryResponse> {
  const paperIds = papers?.map((paper) => paper.id) ?? [];
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history, paper_ids: paperIds, goal: goal || undefined, node_id: nodeId, session_id: sessionId }),
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

// Mirrors service/routers/papers.py's _sanitize_paper_id(f"arxiv-{arxiv_id}")
// so the frontend can predict an in-flight ingest's paper_id and poll its
// status before the ingest request itself has returned. Arxiv IDs matching
// _ARXIV_ID are almost always already-safe characters (digits/dots, or the
// legacy category/number form with a slash) - this only has to replicate
// the same substitution, not validate the id.
function sanitizeArxivPaperId(arxivId: string): string {
  const cleaned = `arxiv-${arxivId}`.replace(/[^A-Za-z0-9._-]/g, "_");
  return cleaned.replace(/^[._-]+/, "").replace(/[._-]+$/, "") || arxivId;
}

export async function getPaperStatus(arxivId: string): Promise<string> {
  const response = await fetch(`/api/papers/${encodeURIComponent(sanitizeArxivPaperId(arxivId))}/status`);
  if (!response.ok) return "unknown";
  const data = await response.json() as { status?: string };
  return data.status ?? "unknown";
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

export async function listGaps(limit = 3, sessionId?: string): Promise<GapCandidate[]> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (sessionId) params.set("session_id", sessionId);
  const response = await fetch(`/api/gaps?${params.toString()}`);
  if (!response.ok) throw new Error("Could not load suggestions");
  return await response.json() as GapCandidate[];
}

export async function getSessionGraph(
  sessionId: string,
): Promise<{ nodes: SessionGraphNode[]; edges: SessionGraphEdge[] }> {
  const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/graph`);
  const data = await response.json() as { nodes: SessionGraphNode[]; edges: SessionGraphEdge[]; error?: string };
  if (!response.ok) throw new Error(data.error ?? "Could not load the graph");
  return data;
}

export async function getSessionBibliography(sessionId: string): Promise<string> {
  const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/bibliography`);
  if (!response.ok) {
    const data = await response.json().catch(() => ({})) as { error?: string };
    throw new Error(data.error ?? "Could not export citations");
  }
  return response.text();
}

export async function checkForContradictions(sessionId: string): Promise<ContradictionCandidate[]> {
  const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/contradictions/check`, { method: "POST" });
  const data = await response.json() as ContradictionCandidate[] & { error?: string };
  if (!response.ok) throw new Error((data as unknown as { error?: string }).error ?? "Could not check for contradictions");
  return data;
}

export async function getFeynmanPrompts(paperId: string, sessionId: string): Promise<FeynmanPrompt[]> {
  const response = await fetch(`/api/papers/${encodeURIComponent(paperId)}/feynman/prompts?session_id=${encodeURIComponent(sessionId)}`);
  const data = await response.json() as FeynmanPrompt[] & { error?: string };
  if (!response.ok) throw new Error((data as unknown as { error?: string }).error ?? "Could not load a question about this paper");
  return data;
}

export async function checkFeynmanExplanation(paperId: string, nodeId: string, sessionId: string, explanation: string): Promise<FeynmanCheckResult> {
  const response = await fetch(`/api/papers/${encodeURIComponent(paperId)}/feynman/check`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ node_id: nodeId, session_id: sessionId, explanation }),
  });
  const data = await response.json() as FeynmanCheckResult & { error?: string };
  if (!response.ok) throw new Error((data as { error?: string }).error ?? "Could not grade this explanation");
  return data;
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
