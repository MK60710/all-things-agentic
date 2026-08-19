import type { QueryResponse } from "./types";

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

export async function askAssistant(
  message: string,
  history: ChatHistoryItem[],
  paper?: PaperContext | null,
): Promise<QueryResponse> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history: history.slice(-20), paper_id: paper?.id }),
  });

  const data = await response.json() as Partial<QueryResponse> & { error?: string };
  if (!response.ok) throw new Error(data.error ?? `Assistant request failed (${response.status})`);
  return {
    answer: data.answer ?? "I couldn't produce a response.",
    citations: data.citations ?? [],
    retrieval_mode: data.retrieval_mode ?? (paper ? "vector" : "general"),
    confidence: data.confidence,
    candidates: data.candidates,
    clarification_question_id: data.clarification_question_id,
  };
}

export async function uploadPaper(file: File): Promise<PaperContext> {
  if (!API_URL) throw new Error("NEXT_PUBLIC_API_URL is not configured");
  const tokenResponse = await fetch("/api/papers/upload-token", { method: "POST" });
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
  const response = await fetch(`${API_URL.replace(/\/$/, "")}/papers`, {
    method: "POST",
    headers: { "X-Upload-Token": tokenData.token },
    body,
  });
  const data = await response.json() as PaperContext & { detail?: string };
  if (!response.ok) throw new Error(data.detail ?? `Paper upload failed (${response.status})`);
  return data;
}

export async function ingestArxivPaper(paper: PaperSearchResult): Promise<PaperContext> {
  const response = await fetch("/api/papers/arxiv", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      arxiv_id: paper.id.replace(/^arxiv:/, ""),
      title: paper.title,
      authors: paper.authors,
      abstract: paper.abstract,
      pdfUrl: paper.pdfUrl,
    }),
  });
  const data = await response.json() as PaperContext & { error?: string };
  if (!response.ok) throw new Error(data.error ?? "Could not ingest the arXiv paper");
  return data;
}

export async function searchPapers(query: string): Promise<PaperSearchResult[]> {
  const response = await fetch(`/api/papers/search?q=${encodeURIComponent(query)}`);
  if (!response.ok) throw new Error("Paper search failed");
  const data = await response.json() as { papers: PaperSearchResult[] };
  return data.papers;
}
