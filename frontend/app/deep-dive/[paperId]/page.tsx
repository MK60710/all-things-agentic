"use client";

import { use, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { askAssistant, getPaperDeepDive, searchPapers, type ChatHistoryItem, type PaperContext, type PaperSearchResult } from "@/lib/api";
import type { DeepDiveResponse, QueryResponse } from "@/lib/types";

type TutorMessage = { role: "user" | "assistant"; text: string; citations?: QueryResponse["citations"] };

export default function DeepDivePage({ params, searchParams }: { params: Promise<{ paperId: string }>; searchParams: Promise<{ session_id?: string }> }) {
  const { paperId } = use(params);
  const { session_id: sessionId = "local" } = use(searchParams);
  const router = useRouter();
  const [deepDive, setDeepDive] = useState<DeepDiveResponse | null>(null);
  const [sectionIndex, setSectionIndex] = useState(0);
  const [messages, setMessages] = useState<TutorMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(true);
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState("");
  const [relatedPapers, setRelatedPapers] = useState<PaperSearchResult[]>([]);

  useEffect(() => {
    getPaperDeepDive(paperId).then(setDeepDive).catch((err) => setError(err instanceof Error ? err.message : "Could not open the deep dive.")).finally(() => setLoading(false));
  }, [paperId]);
  useEffect(() => {
    if (!deepDive) return;
    const canonicalId = deepDive.paper_id.replace(/^arxiv-/, "").toLowerCase();
    searchPapers(deepDive.title).then((papers) => setRelatedPapers(papers.filter((paper) => paper.id.replace(/^arxiv:/, "").toLowerCase() !== canonicalId).slice(0, 4))).catch(() => setRelatedPapers([]));
  }, [deepDive]);

  const section = deepDive?.sections[sectionIndex];
  const storageKey = section ? `atlas-deep-dive-chat-${sessionId}-${paperId}-${section.section_id}` : "";
  useEffect(() => {
    if (!storageKey) return;
    try { setMessages(JSON.parse(window.localStorage.getItem(storageKey) ?? "[]") as TutorMessage[]); } catch { setMessages([]); }
  }, [storageKey]);
  useEffect(() => { if (storageKey) window.localStorage.setItem(storageKey, JSON.stringify(messages)); }, [messages, storageKey]);

  const paper = useMemo<PaperContext | null>(() => deepDive ? { id: deepDive.paper_id, title: deepDive.title } : null, [deepDive]);

  async function askSection() {
    const trimmed = question.trim();
    if (!trimmed || !section || !paper || asking) return;
    const nextMessages = [...messages, { role: "user" as const, text: trimmed }];
    setMessages(nextMessages); setQuestion(""); setAsking(true); setError("");
    try {
      const history: ChatHistoryItem[] = messages.slice(-20).map(({ role, text }) => ({ role, text }));
      const response = await askAssistant(trimmed, history, [paper], null, undefined, sessionId, section.title);
      setMessages([...nextMessages, { role: "assistant", text: response.answer, citations: response.citations }]);
    } catch (err) { setError(err instanceof Error ? err.message : "Could not ask the section tutor."); }
    finally { setAsking(false); }
  }

  if (loading) return <main className="deep-dive-state">Opening your deep dive…</main>;
  if (error && !deepDive) return <main className="deep-dive-state deep-dive-error">{error}</main>;
  if (!deepDive || !section) return <main className="deep-dive-state">This paper has no deep-dive sections yet.</main>;

  return <main className="deep-dive-app">
    <header className="deep-dive-header"><button type="button" onClick={() => router.back()} className="deep-dive-back">← Back to conversation</button><div><small>Deep dive</small><h1>{deepDive.title}</h1></div><span>{sectionIndex + 1} of {deepDive.sections.length}</span></header>
    <div className="deep-dive-layout">
      <article className="deep-dive-reading"><p className="deep-dive-big-picture">{deepDive.big_picture}</p><div className="deep-dive-section-heading"><small>Section {sectionIndex + 1}</small><h2>{section.title}</h2><span>{section.page_start ? `Pages ${section.page_start}${section.page_end && section.page_end !== section.page_start ? `–${section.page_end}` : ""}` : "Source text"}</span></div><p className="deep-dive-explanation">{section.plain_language}</p>{section.key_points.length > 0 && <ul className="deep-dive-points">{section.key_points.map((point) => <li key={point}>{point}</li>)}</ul>}<div className="deep-dive-why"><strong>Why this matters</strong><p>{section.why_it_matters}</p></div><h3 className="deep-dive-source-title">Paper text</h3><div className="deep-dive-source">{section.sources.length === 0 ? <p>No indexed source text is available for this section.</p> : section.sources.map((source, index) => <div key={`${source.page_start}-${index}`}><small>{source.section ?? section.title}{source.page_start ? ` · p. ${source.page_start}${source.page_end && source.page_end !== source.page_start ? `–${source.page_end}` : ""}` : ""}</small><p>{source.text}</p></div>)}</div>{relatedPapers.length > 0 && <section className="deep-dive-related"><small>Optional reading paths</small><h3>Where to go next</h3><p>These arXiv results are relevance-ranked starting points. Use the first group to build context, then the second to go deeper.</p><div className="deep-dive-related-grid">{relatedPapers.map((paper, index) => <a key={paper.id} href={paper.pdfUrl ?? `https://arxiv.org/abs/${paper.id.replace(/^arxiv:/, "")}`} target="_blank" rel="noreferrer"><small>{index < 2 ? "Build context" : "Go deeper"}</small><strong>{paper.title}</strong>{paper.abstract && <span>{paper.abstract.slice(0, 180)}{paper.abstract.length > 180 ? "…" : ""}</span>}</a>)}</div></section>}<nav className="deep-dive-nav"><button type="button" onClick={() => setSectionIndex((index) => index - 1)} disabled={sectionIndex === 0}>← Previous</button><button type="button" onClick={() => setSectionIndex((index) => index + 1)} disabled={sectionIndex + 1 >= deepDive.sections.length}>Next →</button></nav></article>
      <aside className="deep-dive-tutor"><div className="deep-dive-tutor-heading"><small>Section tutor</small><h2>Questions about {section.title}</h2><p>Answers use this section’s source text only.</p></div><div className="deep-dive-tutor-messages">{messages.length === 0 && <p className="deep-dive-tutor-empty">Ask about a method, claim, result, or term in this section.</p>}{messages.map((message, index) => <div key={`${message.role}-${index}`} className={`deep-dive-tutor-message ${message.role}`}><small>{message.role === "user" ? "You" : "Atlas tutor"}</small><p>{message.text}</p>{message.citations?.map((citation, citationIndex) => <span key={`${citation.page_start}-${citationIndex}`} className="deep-dive-citation">{citation.section ?? "Source"}{citation.page_start ? ` · p. ${citation.page_start}` : ""}</span>)}</div>)}</div><form className="deep-dive-tutor-form" onSubmit={(event) => { event.preventDefault(); void askSection(); }}><textarea value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask about this section…" rows={3} disabled={asking}/><button type="submit" disabled={asking || !question.trim()}>{asking ? "Thinking…" : "Ask tutor"}</button></form>{error && <p className="deep-dive-tutor-error">{error}</p>}</aside>
    </div>
  </main>;
}
