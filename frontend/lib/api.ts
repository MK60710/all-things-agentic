import type { Citation, QueryResponse } from "./types";

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
  sessionId?: string,
): Promise<QueryResponse | null> {
  if (paper && !API_URL) return null;

  const endpoint = paper ? `${API_URL!.replace(/\/$/, "")}/query` : "/api/chat";
  const body = paper ? { query: message, paper_id: paper.id, paper } : { message, history, sessionId };
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) throw new Error(`Assistant request failed (${response.status})`);
  const data = await response.json() as Partial<QueryResponse> & { text?: string; response?: string };
  return {
    answer: data.answer ?? data.text ?? data.response ?? "I couldn't produce a response.",
    citations: data.citations ?? [],
    retrieval_mode: data.retrieval_mode ?? (paper ? "vector" : "no_results"),
  };
}

export async function uploadPaper(file: File): Promise<PaperContext | null> {
  if (!API_URL) return null;
  const body = new FormData();
  body.append("file", file);
  const response = await fetch(`${API_URL.replace(/\/$/, "")}/papers/upload`, { method: "POST", body });
  if (!response.ok) throw new Error(`Paper upload failed (${response.status})`);
  return response.json() as Promise<PaperContext>;
}

export async function searchPapers(query: string): Promise<PaperSearchResult[]> {
  const response = await fetch(`/api/papers/search?q=${encodeURIComponent(query)}`);
  if (!response.ok) throw new Error("Paper search failed");
  const data = await response.json() as { papers: PaperSearchResult[] };
  return data.papers;
}
