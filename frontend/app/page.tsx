"use client";

import { ChangeEvent, DragEvent, FormEvent, useEffect, useRef, useState } from "react";
import { askAssistant, searchPapers, uploadPaper } from "@/lib/api";
import type { ChatHistoryItem, PaperContext, PaperSearchResult } from "@/lib/api";
import type { Citation } from "@/lib/types";

type IconName = "atlas" | "plus" | "send" | "spark" | "paper" | "search" | "upload" | "close" | "quote" | "check" | "globe";
type AddMode = "choose" | "upload" | "search";

interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  citations?: Citation[];
  notice?: boolean;
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
};

function Icon({ name, size = 19 }: { name: IconName; size?: number }) {
  return <svg aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">{icons[name]}</svg>;
}

function localGeneralFallback(): string {
  return "The chat interface is working, but the Gemini chat service is not connected in this environment yet. Set NEXT_PUBLIC_API_URL to the service that exposes /chat; Atlas will then use the regular Gemini agent for general conversation.";
}

function localPaperFallback(paper: PaperContext): string {
  return `“${paper.title}” is attached, but this local frontend does not have a running paper-ingestion service. Connect NEXT_PUBLIC_API_URL to the backend /papers/upload and /query routes to answer from the paper itself.`;
}

export default function Home() {
  const fileInput = useRef<HTMLInputElement>(null);
  const composerInput = useRef<HTMLTextAreaElement>(null);
  const chatEnd = useRef<HTMLDivElement>(null);
  const sessionId = useRef("");
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
  const [searchError, setSearchError] = useState("");
  const [expandedCitation, setExpandedCitation] = useState<string | null>(null);

  useEffect(() => { chatEnd.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, loading]);
  useEffect(() => {
    sessionId.current = crypto.randomUUID();
    composerInput.current?.focus();
  }, []);

  async function ask(event?: FormEvent, suggested = query) {
    event?.preventDefault();
    const question = suggested.trim();
    if (!question || loading) return;

    const history: ChatHistoryItem[] = messages.filter((message) => !message.notice).map(({ role, text }) => ({ role, text }));
    setMessages((current) => [...current, { id: crypto.randomUUID(), role: "user", text: question }]);
    setQuery("");
    setLoading(true);
    try {
      const response = await askAssistant(question, history, paper, sessionId.current || crypto.randomUUID());
      setMessages((current) => [...current, {
        id: crypto.randomUUID(),
        role: "assistant",
        text: response?.answer ?? (paper ? localPaperFallback(paper) : localGeneralFallback()),
        citations: response?.citations,
        notice: !response,
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
      const context = uploaded ?? {
        id: `local:${Date.now()}`,
        title: file.name.replace(/\.pdf$/i, ""),
        authors: "Uploaded PDF",
      };
      attachPaper(context);
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : "Upload failed.");
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  function attachPaper(nextPaper: PaperContext) {
    setPaper(nextPaper);
    setMessages((current) => [...current, { id: crypto.randomUUID(), role: "assistant", text: `I’ve added “${nextPaper.title}” to this conversation. Ask me to summarize it, explain a section, or examine its evidence.`, notice: true }]);
    setAddOpen(false);
    setAddMode("choose");
    window.setTimeout(() => composerInput.current?.focus(), 50);
  }

  function removePaper() {
    const title = paper?.title;
    setPaper(null);
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
    } catch {
      setSearchError("Online paper search is unavailable right now.");
      setSearchResults([]);
    } finally {
      setSearching(false);
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
              {messages.map((message) => <article key={message.id} className={`message ${message.role} ${message.notice ? "notice" : ""}`}>
                {message.role === "assistant" && <span className="assistant-avatar"><Icon name={message.notice ? "check" : "spark"} size={16}/></span>}
                <div className="message-body">
                  {message.role === "assistant" && <small>{message.notice ? "Atlas" : "Gemini"}</small>}
                  <p>{message.text}</p>
                  {message.citations && message.citations.length > 0 && <div className="citations">{message.citations.map((citation, index) => {
                    const key = `${message.id}-${index}`;
                    return <div key={key}><button onClick={() => setExpandedCitation(expandedCitation === key ? null : key)}><Icon name="quote" size={13}/>{citation.section ?? "Source"} · p. {citation.page_start ?? "—"}</button>{expandedCitation === key && <blockquote>“{citation.text}”</blockquote>}</div>;
                  })}</div>}
                </div>
              </article>)}
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
          <textarea ref={composerInput} value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void ask(); } }} rows={1} placeholder={paper ? "Ask anything about this paper…" : "Message Gemini…"}/>
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
            <div className="search-results">{searchResults.map((result) => <article key={result.id}><div><span>arXiv</span><small>{result.published}</small></div><h3>{result.title}</h3><p>{result.authors}</p><button onClick={() => attachPaper(result)}>Add to chat</button></article>)}</div>
          </div>}
        </section>
      </div>}
    </main>
  );
}
