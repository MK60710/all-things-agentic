"use client";

import { ChangeEvent, DragEvent, FormEvent, useEffect, useRef, useState } from "react";
import {
  answerClarification,
  askAssistant,
  ingestArxivPaper,
  listClarifications,
  listGaps,
  recordGapFeedback,
  recordQueryFeedback,
  searchPapers,
  uploadPaper,
} from "@/lib/api";
import type { ChatHistoryItem, PaperContext, PaperSearchResult } from "@/lib/api";
import type { Citation, GapCandidate, PendingQuestion, QueryResponse } from "@/lib/types";

type IconName = "atlas" | "plus" | "send" | "spark" | "paper" | "search" | "upload" | "close" | "quote" | "check" | "globe" | "thumbUp" | "thumbDown";
type AddMode = "choose" | "upload" | "search";

interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  citations?: Citation[];
  confidence?: QueryResponse["confidence"];
  candidates?: QueryResponse["candidates"];
  notice?: boolean;
  clarification?: PendingQuestion;
  gaps?: GapCandidate[];
  feedbackGiven?: boolean;
}

function gapKey(candidate: { node_a_id: string; node_b_id: string }) {
  return `${candidate.node_a_id}:${candidate.node_b_id}`;
}

const icons: Record<IconName, React.ReactNode> = {
  atlas: <><circle cx="12" cy="12" r="3"/><path d="M12 3v6M12 15v6M3 12h6M15 12h6M5.6 5.6l4.2 4.2M14.2 14.2l4.2 4.2M18.4 5.6l-4.2 4.2M9.8 14.2l-4.2 4.2"/></>,
  plus: <path d="M12 5v14M5 12h14"/>,
  send: <><path d="M22 2L9.5 14.5M22 2l-8 20-4.5-7.5L2 10z"/></>,
  spark: <><path d="M12 2l1.6 5.2L19 9l-5.4 1.8L12 16l-1.6-5.2L5 9l5.4-1.8z"/><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z"/></>,
  paper: <><path d="M6 3h9l4 4v14H6z"/><path d="M15 3v5h5M9 12h7M9 16h7"/></>,
  search: <><circle cx="10.5" cy="10.5" r="6.5"/><path d="M15.5 15.5L21 21"/></>,
  upload: <><path d="M12 16V3M7 8l5-5 5 5"/><path d="M4 15v5h16v-5"/></>,
  close: <path d="M6 6l12 12M18 6L6 18"/>,
  quote: <><path d="M9 11H4v6h6v-7c0-3-1.5-5-4-6M20 11h-5v6h6v-7c0-3-1.5-5-4-6"/></>,
  check: <path d="M5 12l4 4L19 6"/>,
  globe: <><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c3 3.4 3 14.6 0 18M12 3c-3 3.4-3 14.6 0 18"/></>,
  thumbUp: <path d="M7 10v11H3V10zM7 10l4-7a2 2 0 0 1 2 2v5h5.5a2 2 0 0 1 2 2.4l-1.5 7A2 2 0 0 1 17 21H7"/>,
  thumbDown: <path d="M17 14V3h4v11zM17 14l-4 7a2 2 0 0 1-2-2v-5H5.5a2 2 0 0 1-2-2.4l1.5-7A2 2 0 0 1 7 3h10"/>,
};

function Icon({ name, size = 19 }: { name: IconName; size?: number }) {
  return <svg aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">{icons[name]}</svg>;
}

export default function Home() {
  const fileInput = useRef<HTMLInputElement>(null);
  const composerInput = useRef<HTMLTextAreaElement>(null);
  const chatEnd = useRef<HTMLDivElement>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [paper, setPaper] = useState<PaperContext | null>(null);
  const [addOpen, setAddOpen] = useState(false);
  const [addMode, setAddMode] = useState<AddMode>("choose");
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<PaperSearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [ingestingId, setIngestingId] = useState<string | null>(null);
  const [searchError, setSearchError] = useState("");
  const [expandedCitation, setExpandedCitation] = useState<string | null>(null);
  const [dismissedGapKeys, setDismissedGapKeys] = useState<Set<string>>(new Set());
  const shownClarificationIds = useRef<Set<string>>(new Set());
  const shownGapKeys = useRef<Set<string>>(new Set());

  useEffect(() => { chatEnd.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, loading]);
  // Conversation history is otherwise plain React state - a refresh meant
  // literally starting over, the opposite of "persistent memory... instead
  // of starting over each time".
  useEffect(() => {
    window.localStorage.setItem("atlas-messages", JSON.stringify(messages));
  }, [messages]);
  useEffect(() => {
    const savedPaper = window.localStorage.getItem("atlas-active-paper");
    if (savedPaper) {
      try { setPaper(JSON.parse(savedPaper) as PaperContext); } catch { window.localStorage.removeItem("atlas-active-paper"); }
    }
    const savedMessages = window.localStorage.getItem("atlas-messages");
    if (savedMessages) {
      try {
        const restored = JSON.parse(savedMessages) as Message[];
        // Seed the dedup trackers directly from restored history (not from
        // messages state, which won't have committed yet) so checkGuidance
        // below doesn't re-post a clarification/gap card that's already
        // sitting in the restored conversation.
        restored.forEach((message) => {
          if (message.clarification) shownClarificationIds.current.add(message.clarification.id);
          message.gaps?.forEach((candidate) => shownGapKeys.current.add(gapKey(candidate)));
        });
        setMessages(restored);
      } catch { window.localStorage.removeItem("atlas-messages"); }
    }
    composerInput.current?.focus();
    void checkGuidance();
  }, []);

  // Proactive, unsolicited content: a missing suggestion is fine, an error
  // toast for something the user didn't ask for is not - both halves fail
  // silently and independently.
  async function checkGuidance() {
    try {
      const questions = await listClarifications();
      const fresh = questions.filter((q) => q.status === "open" && !shownClarificationIds.current.has(q.id));
      fresh.forEach((q) => shownClarificationIds.current.add(q.id));
      if (fresh.length) {
        setMessages((current) => [
          ...current,
          ...fresh.map((clarification) => ({ id: crypto.randomUUID(), role: "assistant" as const, text: "", notice: true, clarification })),
        ]);
      }
    } catch {
      // silent
    }
    try {
      const gaps = await listGaps(3);
      const fresh = gaps.filter((g) => !shownGapKeys.current.has(gapKey(g)));
      fresh.forEach((g) => shownGapKeys.current.add(gapKey(g)));
      if (fresh.length) {
        setMessages((current) => [
          ...current,
          { id: crypto.randomUUID(), role: "assistant", text: "A few things worth exploring:", notice: true, gaps: fresh },
        ]);
      }
    } catch {
      // silent
    }
  }

  async function answerClarificationQuestion(messageId: string, question: PendingQuestion, optionId: string) {
    try {
      const answered = await answerClarification(question.id, optionId);
      setMessages((current) => current.map((m) => (m.id === messageId ? { ...m, clarification: answered } : m)));
    } catch (error) {
      setMessages((current) => [...current, {
        id: crypto.randomUUID(),
        role: "assistant",
        text: error instanceof Error ? error.message : "Could not record that answer.",
        notice: true,
      }]);
    }
  }

  function submitQueryFeedback(messageId: string, nodeId: string, helpful: boolean) {
    setMessages((current) => current.map((m) => (m.id === messageId ? { ...m, feedbackGiven: true } : m)));
    void recordQueryFeedback(nodeId, helpful).catch(() => {});
  }

  function askAboutGap(candidate: GapCandidate) {
    const question = `How does ${candidate.node_a_name} relate to ${candidate.node_b_name}?${candidate.explanation ? ` (${candidate.explanation})` : ""}`;
    void recordGapFeedback(candidate.node_a_id, candidate.node_b_id, true).catch(() => {});
    // Gaps come from GapFinder, which reasons over the whole graph, not any
    // one paper - explicitly ignore whatever paper happens to be attached
    // right now, or this silently re-scopes the question to it and can
    // wrongly report "no information" for evidence that's real but lives
    // in a different paper.
    void ask(undefined, question, null);
  }

  function dismissGap(candidate: GapCandidate) {
    setDismissedGapKeys((current) => new Set(current).add(gapKey(candidate)));
    void recordGapFeedback(candidate.node_a_id, candidate.node_b_id, false).catch(() => {});
  }

  async function ask(event?: FormEvent, suggested = query, paperOverride?: PaperContext | null) {
    event?.preventDefault();
    const question = suggested.trim();
    if (!question || loading) return;
    const effectivePaper = paperOverride === undefined ? paper : paperOverride;

    const history: ChatHistoryItem[] = messages
      .filter((message) => !message.notice)
      .slice(-20)
      .map(({ role, text }) => ({ role, text: text.slice(0, 8000) }));
    setMessages((current) => [...current, { id: crypto.randomUUID(), role: "user", text: question }]);
    setQuery("");
    setLoading(true);
    try {
      const response = await askAssistant(question, history, effectivePaper);
      setMessages((current) => [...current, {
        id: crypto.randomUUID(),
        role: "assistant",
        text: response.answer,
        citations: response.citations,
        confidence: response.confidence,
        candidates: response.candidates,
      }]);
    } catch (error) {
      setMessages((current) => [...current, {
        id: crypto.randomUUID(),
        role: "assistant",
        text: error instanceof Error ? `I couldn't reach Gemini: ${error.message}` : "I couldn't reach Gemini.",
        notice: true,
      }]);
    } finally {
      setLoading(false);
      window.setTimeout(() => composerInput.current?.focus(), 50);
    }
  }

  function openAddPaper(mode: AddMode = "choose") {
    setAddMode(mode);
    setUploadError("");
    setSearchError("");
    setAddOpen(true);
  }

  async function handleFile(file?: File) {
    if (!file) return;
    if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
      setUploadError("Please choose a PDF file.");
      return;
    }
    setUploading(true);
    setUploadError("");
    try {
      const uploaded = await uploadPaper(file);
      attachPaper(uploaded);
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : "Upload failed.");
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  function attachPaper(nextPaper: PaperContext) {
    setPaper(nextPaper);
    window.localStorage.setItem("atlas-active-paper", JSON.stringify(nextPaper));
    setMessages((current) => [...current, { id: crypto.randomUUID(), role: "assistant", text: `I've added "${nextPaper.title}" to this conversation. Ask me to summarize it, explain a section, or examine its evidence.`, notice: true }]);
    setAddOpen(false);
    setAddMode("choose");
    window.setTimeout(() => composerInput.current?.focus(), 50);
    void checkGuidance();
  }

  function removePaper() {
    const title = paper?.title;
    setPaper(null);
    window.localStorage.removeItem("atlas-active-paper");
    setMessages((current) => [...current, { id: crypto.randomUUID(), role: "assistant", text: `Paper context removed${title ? `: “${title}”` : ""}. We’re back to general Gemini chat.`, notice: true }]);
  }

  async function runSearch(event: FormEvent) {
    event.preventDefault();
    const term = searchQuery.trim();
    if (term.length < 2 || searching) return;
    setSearching(true);
    setSearchError("");
    try {
      const results = await searchPapers(term);
      setSearchResults(results);
      if (!results.length) setSearchError("No papers found. Try a broader title or topic.");
    } catch (error) {
      setSearchError(error instanceof Error ? error.message : "Online paper search is unavailable right now.");
      setSearchResults([]);
    } finally {
      setSearching(false);
    }
  }

  async function addArxivPaper(result: PaperSearchResult) {
    if (ingestingId) return;
    setIngestingId(result.id);
    setSearchError("");
    try {
      attachPaper(await ingestArxivPaper(result));
    } catch (error) {
      setSearchError(error instanceof Error ? error.message : "Could not read this paper.");
    } finally {
      setIngestingId(null);
    }
  }

  function onDrop(event: DragEvent<HTMLButtonElement>) {
    event.preventDefault();
    void handleFile(event.dataTransfer.files?.[0]);
  }

  return (
    <main className="assistant-app">
      <header className="app-header">
        <div className="brand"><span><Icon name="atlas" size={21}/></span><strong>Atlas</strong></div>
        <div className={`mode-label ${paper ? "paper-mode" : ""}`}>
          {paper ? <><Icon name="paper" size={15}/><span><small>Reading</small><strong>{paper.title}</strong></span><button onClick={removePaper} aria-label="Remove paper"><Icon name="close" size={14}/></button></> : <><span className="online-dot"/><strong>General chat</strong><small>Gemini</small></>}
        </div>
        <button className="add-paper-button" onClick={() => openAddPaper()}><Icon name="plus" size={17}/>{paper ? "Change paper" : "Add paper"}</button>
      </header>

      <section className="conversation-scroll">
        <div className="conversation-content">
          {messages.length === 0 ? (
            <div className="welcome">
              <span className="welcome-icon"><Icon name="spark" size={27}/></span>
              <h1>How can I help?</h1>
              <p>Chat normally with Gemini, or add a research paper whenever you want to go deeper.</p>
              <div className="welcome-actions">
                <button onClick={() => ask(undefined, "Help me understand how AI agents use memory")}>Explain a research topic</button>
                <button onClick={() => ask(undefined, "Help me brainstorm a research question")}>Brainstorm with me</button>
                <button onClick={() => openAddPaper()}><Icon name="paper" size={15}/>Add a paper</button>
              </div>
            </div>
          ) : (
            <div className="message-list">
              {messages.map((message) => {
                const visibleGaps = message.gaps?.filter((g) => !dismissedGapKeys.has(gapKey(g)));
                if (message.gaps && (!visibleGaps || visibleGaps.length === 0)) return null;
                const feedbackNodeId = message.citations?.[0]?.source_kind === "graph" ? message.citations[0].node_ids?.[0] : undefined;

                return <article key={message.id} className={`message ${message.role} ${message.notice ? "notice" : ""}`}>
                  {message.role === "assistant" && <span className="assistant-avatar"><Icon name={message.notice ? "check" : "spark"} size={16}/></span>}
                  <div className="message-body">
                    {message.role === "assistant" && <small>{message.notice ? "Atlas" : "Gemini"}</small>}
                    {message.text && <p>{message.text}</p>}
                    {message.confidence === "low" && <span className="confidence-note">Low-confidence match — check the sources below.</span>}
                    {message.candidates && message.candidates.length > 0 && <div className="candidate-list">{message.candidates.map((candidate) => <div key={candidate.node_id}><strong>{candidate.name}</strong><small>{candidate.type}</small>{candidate.description && <p>{candidate.description}</p>}</div>)}</div>}
                    {message.citations && message.citations.length > 0 && <div className="citations">{message.citations.map((citation, index) => {
                      const key = `${message.id}-${index}`;
                      return <div key={key}><button onClick={() => setExpandedCitation(expandedCitation === key ? null : key)}><Icon name="quote" size={13}/>{citation.section ?? "Source"} · p. {citation.page_start ?? "—"}</button>{expandedCitation === key && <blockquote>“{citation.text}”</blockquote>}</div>;
                    })}</div>}
                    {feedbackNodeId && <div className="feedback-row">
                      {message.feedbackGiven ? <small>Thanks, I&rsquo;ll factor that in.</small> : <>
                        <button onClick={() => submitQueryFeedback(message.id, feedbackNodeId, true)} aria-label="Helpful"><Icon name="thumbUp" size={13}/></button>
                        <button onClick={() => submitQueryFeedback(message.id, feedbackNodeId, false)} aria-label="Not helpful"><Icon name="thumbDown" size={13}/></button>
                      </>}
                    </div>}
                    {message.clarification && <div className="clarification-card">
                      <p>{message.clarification.question}</p>
                      {message.clarification.status === "answered" ? (
                        <small>Answered: {message.clarification.options.find((opt) => opt.id === message.clarification!.answer_option_id)?.label ?? message.clarification.answer_option_id}</small>
                      ) : (
                        <div className="clarification-options">
                          {message.clarification.options.map((option) => <button key={option.id} onClick={() => answerClarificationQuestion(message.id, message.clarification!, option.id)}>{option.label}</button>)}
                        </div>
                      )}
                    </div>}
                    {visibleGaps && visibleGaps.length > 0 && <div className="gap-suggestions">
                      {visibleGaps.map((candidate) => <div key={gapKey(candidate)} className="gap-chip">
                        <button onClick={() => askAboutGap(candidate)}>
                          <strong>{candidate.node_a_name} ↔ {candidate.node_b_name}</strong>
                          {candidate.explanation && <span>{candidate.explanation}</span>}
                        </button>
                        <button className="gap-dismiss" onClick={() => dismissGap(candidate)} aria-label="Not interesting"><Icon name="close" size={12}/></button>
                      </div>)}
                    </div>}
                  </div>
                </article>;
              })}
              {loading && <article className="message assistant"><span className="assistant-avatar"><Icon name="spark" size={16}/></span><div className="message-body"><small>Gemini</small><div className="typing"><i/><i/><i/></div></div></article>}
            </div>
          )}
          <div ref={chatEnd}/>
        </div>
      </section>

      <footer className="composer-area">
        {paper && <div className="paper-context-chip"><Icon name="paper" size={14}/><span>Using <strong>{paper.title}</strong></span><button onClick={removePaper}><Icon name="close" size={13}/></button></div>}
        <form className="composer" onSubmit={(event) => ask(event)}>
          <button type="button" className="composer-add" onClick={() => openAddPaper()} aria-label="Add a paper"><Icon name="plus" size={19}/></button>
          <textarea ref={composerInput} value={query} maxLength={8000} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void ask(); } }} rows={1} placeholder={paper ? "Ask anything about this paper…" : "Message Gemini…"}/>
          <button type="submit" className="send-button" disabled={!query.trim() || loading} aria-label="Send"><Icon name="send" size={18}/></button>
        </form>
        <small className="composer-hint">Gemini can make mistakes. Paper answers include sources when available.</small>
      </footer>

      {addOpen && <div className="add-modal" role="dialog" aria-modal="true" aria-label="Add a research paper">
        <button className="modal-scrim" onClick={() => setAddOpen(false)} aria-label="Close"/>
        <section className="modal-card">
          <header><div><span><Icon name="paper" size={18}/></span><div><strong>Add a research paper</strong><small>Give Gemini a paper to read with you</small></div></div><button onClick={() => setAddOpen(false)} aria-label="Close"><Icon name="close" size={19}/></button></header>

          {addMode === "choose" && <div className="add-choices">
            <button onClick={() => setAddMode("upload")}><span className="choice-icon violet"><Icon name="upload" size={22}/></span><div><strong>Upload a PDF</strong><p>Choose a paper saved on your computer.</p></div></button>
            <button onClick={() => setAddMode("search")}><span className="choice-icon blue"><Icon name="globe" size={22}/></span><div><strong>Search online</strong><p>Find papers on arXiv by title, author, or topic.</p></div></button>
          </div>}

          {addMode === "upload" && <div className="upload-panel">
            <button className="back-button" onClick={() => setAddMode("choose")}>← Back</button>
            <button className="drop-zone" onClick={() => fileInput.current?.click()} onDragOver={(event) => event.preventDefault()} onDrop={onDrop} disabled={uploading}>
              <span><Icon name="upload" size={25}/></span><strong>{uploading ? "Uploading and reading…" : "Choose a PDF or drag it here"}</strong><small>PDF files up to the backend’s configured limit</small>
            </button>
            <input ref={fileInput} className="hidden-input" type="file" accept="application/pdf,.pdf" onChange={(event: ChangeEvent<HTMLInputElement>) => void handleFile(event.target.files?.[0])}/>
            {uploadError && <p className="form-error">{uploadError}</p>}
          </div>}

          {addMode === "search" && <div className="search-panel">
            <button className="back-button" onClick={() => setAddMode("choose")}>← Back</button>
            <form onSubmit={runSearch}><Icon name="search" size={18}/><input autoFocus value={searchQuery} onChange={(event) => setSearchQuery(event.target.value)} placeholder="Search by title, author, or topic…"/><button disabled={searchQuery.trim().length < 2 || searching}>{searching ? "Searching…" : "Search"}</button></form>
            {searchError && <p className="form-error">{searchError}</p>}
            <div className="search-results">{searchResults.map((result) => <article key={result.id}><div><span>arXiv</span><small>{result.published}</small></div><h3>{result.title}</h3><p>{result.authors}</p><button disabled={Boolean(ingestingId)} onClick={() => void addArxivPaper(result)}>{ingestingId === result.id ? "Reading…" : "Add to chat"}</button></article>)}</div>
          </div>}
        </section>
      </div>}
    </main>
  );
}
