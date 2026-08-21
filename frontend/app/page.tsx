"use client";

import { ChangeEvent, DragEvent, FormEvent, useEffect, useRef, useState } from "react";
import {
  answerClarification,
  askAssistant,
  buildPaperGuide,
  createSession,
  detachPaper,
  ingestArxivPaper,
  listClarifications,
  listGaps,
  listPapersForSession,
  listSessions,
  recordGapFeedback,
  recordQueryFeedback,
  searchPapers,
  uploadPaper,
} from "@/lib/api";
import type { ChatHistoryItem, PaperContext, PaperIngestResult, PaperSearchResult, SessionMetadata } from "@/lib/api";
import type { Citation, GapCandidate, PaperGuide, PendingQuestion, QueryResponse } from "@/lib/types";
import GraphBuildAnimation from "./GraphBuildAnimation";

type IconName = "atlas" | "plus" | "send" | "spark" | "paper" | "search" | "upload" | "close" | "quote" | "check" | "globe" | "thumbUp" | "thumbDown";
type AddMode = "choose" | "upload" | "search";

interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  citations?: Citation[];
  confidence?: QueryResponse["confidence"];
  candidates?: QueryResponse["candidates"];
  guide?: PaperGuide;
  guideLoading?: boolean;
  guideError?: string;
  notice?: boolean;
  clarification?: PendingQuestion;
  // Set instead of `clarification` when more than one question arrives in
  // the same checkGuidance() pass - rendered as one compact batched card
  // instead of N full message bubbles.
  clarifications?: PendingQuestion[];
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

function FlowDiagram({ guide }: { guide: NonNullable<PaperGuide["sections"][number]["diagram"]> }) {
  return <div className="guide-diagram">
    <strong>{guide.title}</strong>
    <div className="flow-track">{guide.nodes.map((node, index) => {
      const next = guide.nodes[index + 1];
      const edge = next ? guide.edges.find((item) => item.source === node.id && item.target === next.id) : undefined;
      return <div className="flow-step" key={node.id}>
        <div className="flow-node"><span>{node.label}</span>{node.detail && <small>{node.detail}</small>}</div>
        {next && <div className="flow-arrow"><i>↓</i>{edge?.label && <small>{edge.label}</small>}</div>}
      </div>;
    })}</div>
  </div>;
}

function GuidedReading({ guide }: { guide: PaperGuide }) {
  return <section className="guided-reading">
    <header><div><span>Guided reading</span><strong>{guide.sections.length} stops · about {guide.reading_time_minutes} min</strong></div><Icon name="spark" size={20}/></header>
    <div className="guide-overview"><small>The big picture</small><p>{guide.big_picture}</p></div>
    <div className="guide-sections">{guide.sections.map((section, index) => <article key={`${section.title}-${index}`}>
      <div className="guide-section-heading"><span>{index + 1}</span><div><small>{section.page_start ? `Pages ${section.page_start}${section.page_end && section.page_end !== section.page_start ? `–${section.page_end}` : ""}` : "Paper section"}</small><h3>{section.title}</h3></div></div>
      <p>{section.plain_language}</p>
      {section.key_points.length > 0 && <ul>{section.key_points.map((point) => <li key={point}>{point}</li>)}</ul>}
      {section.diagram && section.diagram.nodes.length > 0 && <FlowDiagram guide={section.diagram}/>}
      <div className="why-it-matters"><strong>Why this matters</strong><p>{section.why_it_matters}</p></div>
    </article>)}</div>
    <footer>That’s the full walkthrough. Ask a question below and I’ll answer from this paper with page-level sources.</footer>
  </section>;
}

export default function Home() {
  const fileInput = useRef<HTMLInputElement>(null);
  const composerInput = useRef<HTMLTextAreaElement>(null);
  const chatEnd = useRef<HTMLDivElement>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [papers, setPapers] = useState<PaperContext[]>([]);
  const [addOpen, setAddOpen] = useState(false);
  const [addMode, setAddMode] = useState<AddMode>("choose");
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<PaperSearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  // Multiple searches can be in flight at once now - keyed by result.id so
  // each card can show its own "Reading…"/cancel independently instead of
  // one global flag blocking every other result while it's in progress.
  const [ingestingIds, setIngestingIds] = useState<Set<string>>(new Set());
  const ingestControllers = useRef<Map<string, AbortController>>(new Map());
  const uploadController = useRef<AbortController | null>(null);
  const [searchError, setSearchError] = useState("");
  const [expandedCitation, setExpandedCitation] = useState<string | null>(null);
  const [dismissedGapKeys, setDismissedGapKeys] = useState<Set<string>>(new Set());
  // A queue, not a single value - multiple ingests can finish close
  // together now that they're no longer serialized, and each one gets its
  // own full reveal in turn rather than the second silently clobbering the
  // first's still-playing animation.
  const [buildingGraphQueue, setBuildingGraphQueue] = useState<PaperIngestResult[]>([]);
  const shownClarificationIds = useRef<Set<string>>(new Set());
  const shownGapKeys = useRef<Set<string>>(new Set());
  // Tags every ingest write server-side (scripts/clear_session.py can then
  // clear exactly this session's papers/nodes/edges) and scopes chat/
  // clarifications to this session - not displayed itself, currentSession
  // is the display copy. A ref because requests read it synchronously
  // between renders; a plain state field would risk a stale closure.
  const sessionIdRef = useRef<string>("");
  const [currentSession, setCurrentSession] = useState<SessionMetadata | null>(null);
  const [sessions, setSessions] = useState<SessionMetadata[]>([]);
  const [sessionMenuOpen, setSessionMenuOpen] = useState(false);

  useEffect(() => { chatEnd.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, loading]);
  // Conversation history is otherwise plain React state - a refresh meant
  // literally starting over, the opposite of "persistent memory... instead
  // of starting over each time".
  useEffect(() => {
    // Guard against the pre-init render (sessionIdRef not set yet) writing
    // an empty array under a bogus key - initSession/switchToSession below
    // are what actually load a session's messages.
    if (!sessionIdRef.current) return;
    window.localStorage.setItem(`atlas-messages-${sessionIdRef.current}`, JSON.stringify(messages));
  }, [messages]);
  useEffect(() => { void initSession(); }, []);

  async function initSession() {
    let resolved: SessionMetadata | null = null;
    try {
      const list = await listSessions();
      setSessions(list);
      const savedId = window.localStorage.getItem("atlas-session-id");
      resolved = list.find((s) => s.id === savedId) ?? list[0] ?? null;
      if (!resolved) {
        resolved = await createSession("Untitled session");
        setSessions((current) => [resolved!, ...current]);
      }
    } catch {
      // Backend unreachable at startup - fall back to a purely local
      // session id so the app still works; ingest calls just carry it.
      const fallbackId = window.localStorage.getItem("atlas-session-id") ?? crypto.randomUUID();
      resolved = { id: fallbackId, name: "Untitled session", created_at: new Date().toISOString() };
    }
    await switchToSession(resolved);
  }

  async function switchToSession(session: SessionMetadata) {
    sessionIdRef.current = session.id;
    window.localStorage.setItem("atlas-session-id", session.id);
    setCurrentSession(session);
    setSessionMenuOpen(false);
    setBuildingGraphQueue([]);
    setDismissedGapKeys(new Set());

    const savedMessages = window.localStorage.getItem(`atlas-messages-${session.id}`);
    let restoredMessages: Message[] = [];
    const freshClarificationIds = new Set<string>();
    const freshGapKeys = new Set<string>();
    if (savedMessages) {
      try {
        restoredMessages = JSON.parse(savedMessages) as Message[];
        // Seed the dedup trackers directly from restored history (not from
        // messages state, which won't have committed yet) so checkGuidance
        // below doesn't re-post a clarification/gap card that's already
        // sitting in the restored conversation.
        restoredMessages.forEach((message) => {
          if (message.clarification) freshClarificationIds.add(message.clarification.id);
          message.clarifications?.forEach((q) => freshClarificationIds.add(q.id));
          message.gaps?.forEach((candidate) => freshGapKeys.add(gapKey(candidate)));
        });
      } catch { window.localStorage.removeItem(`atlas-messages-${session.id}`); }
    }
    shownClarificationIds.current = freshClarificationIds;
    shownGapKeys.current = freshGapKeys;
    setMessages(restoredMessages);
    setPapers([]);

    try {
      setPapers(await listPapersForSession(session.id));
    } catch (error) {
      setMessages((current) => [...current, {
        id: crypto.randomUUID(),
        role: "assistant",
        text: error instanceof Error ? error.message : "Could not load this session's papers.",
        notice: true,
      }]);
    }

    window.setTimeout(() => composerInput.current?.focus(), 50);
    // A genuinely fresh session - nothing added, nothing asked yet - should
    // start clean. Surfacing corpus-wide gap suggestions before the user
    // has engaged with anything is confusing, not helpful; proactive
    // guidance only makes sense once there's something to be guided about.
    // addPaper() already runs checkGuidance() for the "just added a paper"
    // case; this only covers "returning to a session with history" here.
    if (restoredMessages.length > 0) void checkGuidance();
  }

  async function createAndSwitchToNewSession() {
    const name = window.prompt("Name this session:", "")?.trim();
    if (!name) return;
    setSessionMenuOpen(false);
    try {
      const created = await createSession(name);
      setSessions((current) => [created, ...current]);
      await switchToSession(created);
    } catch (error) {
      setMessages((current) => [...current, {
        id: crypto.randomUUID(),
        role: "assistant",
        text: error instanceof Error ? error.message : "Could not create the session.",
        notice: true,
      }]);
    }
  }

  // Proactive, unsolicited content: a missing suggestion is fine, an error
  // toast for something the user didn't ask for is not - both halves fail
  // silently and independently.
  async function checkGuidance() {
    try {
      const questions = await listClarifications(sessionIdRef.current || undefined);
      const fresh = questions.filter((q) => q.status === "open" && !shownClarificationIds.current.has(q.id));
      fresh.forEach((q) => shownClarificationIds.current.add(q.id));
      if (fresh.length === 1) {
        setMessages((current) => [
          ...current,
          { id: crypto.randomUUID(), role: "assistant" as const, text: "", notice: true, clarification: fresh[0] },
        ]);
      } else if (fresh.length > 1) {
        // More than one at once reads as a wall of doubt-cards, not a
        // partner asking a smart question - one compact batched card
        // instead of N full message bubbles.
        setMessages((current) => [
          ...current,
          { id: crypto.randomUUID(), role: "assistant" as const, text: "", notice: true, clarifications: fresh },
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
      setMessages((current) => current.map((m) => {
        if (m.id !== messageId) return m;
        if (m.clarifications) {
          return { ...m, clarifications: m.clarifications.map((q) => (q.id === answered.id ? answered : q)) };
        }
        return { ...m, clarification: answered };
      }));
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
    // Gaps come from GapFinder, which reasons over the whole graph, not the
    // session's working set - explicitly ignore whatever papers are
    // currently added, or this silently re-scopes the question to them and
    // can wrongly report "no information" for evidence that's real but
    // lives in a paper outside the working set.
    void ask(undefined, question, []);
  }

  function dismissGap(candidate: GapCandidate) {
    setDismissedGapKeys((current) => new Set(current).add(gapKey(candidate)));
    void recordGapFeedback(candidate.node_a_id, candidate.node_b_id, false).catch(() => {});
  }

  async function ask(event?: FormEvent, suggested = query, papersOverride?: PaperContext[] | null) {
    event?.preventDefault();
    const question = suggested.trim();
    if (!question || loading) return;
    const effectivePapers = papersOverride === undefined ? papers : papersOverride;

    const history: ChatHistoryItem[] = messages
      .filter((message) => (
        !message.notice
        && !message.guide
        && !message.guideLoading
        && Boolean(message.text.trim())
      ))
      .slice(-20)
      .map(({ role, text }) => ({ role, text: text.slice(0, 8000) }));
    setMessages((current) => [...current, { id: crypto.randomUUID(), role: "user", text: question }]);
    setQuery("");
    setLoading(true);
    try {
      const response = await askAssistant(question, history, effectivePapers);
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
    const controller = new AbortController();
    uploadController.current = controller;
    setUploading(true);
    setUploadError("");
    try {
      const uploaded = await uploadPaper(file, controller.signal, sessionIdRef.current);
      setBuildingGraphQueue((current) => [...current, uploaded]);
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      setUploadError(error instanceof Error ? error.message : "Upload failed.");
    } finally {
      setUploading(false);
      uploadController.current = null;
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  function cancelUpload() {
    uploadController.current?.abort();
  }

  // announce=false is for the one caller that already sent its own "I've
  // added X" notice right before this (addPaper) - announce=true is for
  // any future caller that hasn't (kept from origin/main's signature since
  // nothing else needs it today, but the option costs nothing to keep).
  async function beginGuidedReading(nextPaper: PaperContext, announce = false) {
    const guideMessageId = crypto.randomUUID();
    setMessages((current) => [
      ...current,
      ...(announce ? [{ id: crypto.randomUUID(), role: "assistant" as const, text: `I've read "${nextPaper.title}". I'm building a guided walkthrough now, starting with the big picture, then moving through the paper section by section.`, notice: true }] : []),
      { id: guideMessageId, role: "assistant", text: "", guideLoading: true },
    ]);
    try {
      const guide = await buildPaperGuide(nextPaper.id);
      setMessages((current) => current.map((message) => message.id === guideMessageId ? { ...message, guideLoading: false, guide } : message));
    } catch (error) {
      const detail = error instanceof Error ? error.message : "Could not build the walkthrough.";
      setMessages((current) => current.map((message) => message.id === guideMessageId ? { ...message, guideLoading: false, guideError: detail, text: "The paper was added and is ready for questions, but I couldn't generate its guided walkthrough." } : message));
    }
  }

  function addPaper(nextPaper: PaperContext) {
    // Papers are sourced from the backend per session (listPapersForSession)
    // now, not localStorage - this is just the optimistic local update for
    // the paper the backend just confirmed ingesting into sessionIdRef.current.
    setPapers((current) => (current.some((existing) => existing.id === nextPaper.id) ? current : [...current, nextPaper]));
    setMessages((current) => [...current, { id: crypto.randomUUID(), role: "assistant", text: `I've added "${nextPaper.title}" to this conversation. Ask me to summarize it, explain a section, or examine its evidence.`, notice: true }]);
    setAddOpen(false);
    setAddMode("choose");
    void beginGuidedReading(nextPaper);
    window.setTimeout(() => composerInput.current?.focus(), 50);
    void checkGuidance();
  }

  function removePaperFromSet(paperId: string) {
    // Optimistic local removal (updates the working set /chat sends as
    // paper_ids immediately), backed by a real server-side detach so the
    // paper doesn't reappear if you switch away and back to this session -
    // listPapersForSession(session_id) is the source of truth on switch.
    setPapers((current) => {
      const removed = current.find((existing) => existing.id === paperId);
      const updated = current.filter((existing) => existing.id !== paperId);
      setMessages((currentMessages) => [...currentMessages, { id: crypto.randomUUID(), role: "assistant", text: `Removed${removed ? `: "${removed.title}"` : ""} from this conversation.${updated.length === 0 ? " We're back to searching everything." : ""}`, notice: true }]);
      return updated;
    });
    void detachPaper(paperId).catch((error) => {
      setMessages((current) => [...current, {
        id: crypto.randomUUID(),
        role: "assistant",
        text: error instanceof Error
          ? `${error.message} - it may reappear if you switch sessions.`
          : "Could not fully remove this paper - it may reappear if you switch sessions.",
        notice: true,
      }]);
    });
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
    if (ingestingIds.has(result.id)) return;
    const controller = new AbortController();
    ingestControllers.current.set(result.id, controller);
    setIngestingIds((current) => new Set(current).add(result.id));
    setSearchError("");
    try {
      const ingested = await ingestArxivPaper(result, controller.signal, sessionIdRef.current);
      setBuildingGraphQueue((current) => [...current, ingested]);
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      setSearchError(error instanceof Error ? error.message : "Could not read this paper.");
    } finally {
      ingestControllers.current.delete(result.id);
      setIngestingIds((current) => {
        const updated = new Set(current);
        updated.delete(result.id);
        return updated;
      });
    }
  }

  function cancelArxivIngest(resultId: string) {
    ingestControllers.current.get(resultId)?.abort();
  }

  function onDrop(event: DragEvent<HTMLButtonElement>) {
    event.preventDefault();
    void handleFile(event.dataTransfer.files?.[0]);
  }

  return (
    <main className="assistant-app">
      <header className="app-header">
        <div className="brand"><span><Icon name="atlas" size={21}/></span><strong>Atlas</strong></div>
        <div className={`mode-label ${papers.length ? "paper-mode" : ""}`}>
          {papers.length > 0 ? (
            <div className="paper-chip-row">
              {papers.map((p) => <span key={p.id} className="paper-chip"><Icon name="paper" size={12}/><strong>{p.title}</strong><button onClick={() => removePaperFromSet(p.id)} aria-label={`Remove ${p.title}`}><Icon name="close" size={11}/></button></span>)}
            </div>
          ) : <><span className="online-dot"/><strong>General chat</strong><small>Gemini</small></>}
        </div>
        <div className="header-actions">
          <div className="session-switcher">
            <button className="session-switcher-toggle" onClick={() => setSessionMenuOpen((open) => !open)}>
              <span>{currentSession?.name ?? "Session"}</span>
            </button>
            {sessionMenuOpen && <div className="session-menu">
              <button className="session-menu-new" onClick={() => void createAndSwitchToNewSession()}><Icon name="plus" size={13}/>New session</button>
              {sessions.map((s) => <button
                key={s.id}
                className={`session-menu-item${s.id === currentSession?.id ? " active" : ""}`}
                onClick={() => void switchToSession(s)}
              >
                <strong>{s.name}</strong>
                <small>{new Date(s.created_at).toLocaleDateString()}</small>
              </button>)}
            </div>}
          </div>
          <button className="add-paper-button" onClick={() => openAddPaper()}><Icon name="plus" size={17}/>{papers.length ? "Add another" : "Add paper"}</button>
        </div>
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
                const isGuideMessage = Boolean(message.guide || message.guideLoading);

                return <article key={message.id} className={`message ${message.role} ${message.notice ? "notice" : ""} ${isGuideMessage ? "guide-message" : ""}`}>
                  {message.role === "assistant" && <span className="assistant-avatar"><Icon name={message.notice ? "check" : "spark"} size={16}/></span>}
                  <div className="message-body">
                    {message.role === "assistant" && <small>{isGuideMessage ? "Atlas guide" : message.notice ? "Atlas" : "Gemini"}</small>}
                    {message.text && <p>{message.text}</p>}
                    {message.guideLoading && <div className="guide-building"><span/><div><strong>Building your guided reading</strong><small>Finding the paper's structure, simplifying each section, and drawing useful visual explanations…</small></div></div>}
                    {message.guide && <GuidedReading guide={message.guide}/>}
                    {message.guideError && <span className="guide-error">{message.guideError}</span>}
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
                    {message.clarifications && message.clarifications.length > 0 && <div className="clarification-batch">
                      <small>{message.clarifications.length} entities might be duplicates</small>
                      {message.clarifications.map((question) => <div key={question.id} className="clarification-row">
                        <p>{question.question}</p>
                        {question.status === "answered" ? (
                          <small>Answered: {question.options.find((opt) => opt.id === question.answer_option_id)?.label ?? question.answer_option_id}</small>
                        ) : (
                          <div className="clarification-options">
                            {question.options.map((option) => <button key={option.id} onClick={() => answerClarificationQuestion(message.id, question, option.id)}>{option.label}</button>)}
                          </div>
                        )}
                      </div>)}
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
        {papers.length > 0 && <div className="paper-context-chip"><Icon name="paper" size={14}/><span>Using <strong>{papers.length === 1 ? papers[0].title : `${papers.length} papers`}</strong></span></div>}
        <form className="composer" onSubmit={(event) => ask(event)}>
          <button type="button" className="composer-add" onClick={() => openAddPaper()} aria-label="Add a paper"><Icon name="plus" size={19}/></button>
          <textarea ref={composerInput} value={query} maxLength={8000} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void ask(); } }} rows={1} placeholder={papers.length ? "Ask anything about these papers…" : "Message Gemini…"}/>
          <button type="submit" className="send-button" disabled={!query.trim() || loading} aria-label="Send"><Icon name="send" size={18}/></button>
        </form>
        <small className="composer-hint">Gemini can make mistakes. Paper answers include sources when available.</small>
      </footer>

      {addOpen && <div className="add-modal" role="dialog" aria-modal="true" aria-label="Add a research paper">
        <button className="modal-scrim" onClick={() => setAddOpen(false)} aria-label="Close"/>
        <section className={`modal-card ${buildingGraphQueue.length > 0 ? "modal-card-fullscreen" : ""}`}>
          <header><div><span><Icon name="paper" size={18}/></span><div><strong>Add a research paper</strong><small>Give Gemini a paper to read with you</small></div></div><button onClick={() => setAddOpen(false)} aria-label="Close"><Icon name="close" size={19}/></button></header>

          {buildingGraphQueue.length > 0 ? (
            <GraphBuildAnimation
              key={buildingGraphQueue[0].id}
              newNodes={buildingGraphQueue[0].new_nodes}
              newEdges={buildingGraphQueue[0].new_edges}
              onComplete={() => {
                addPaper(buildingGraphQueue[0]);
                setBuildingGraphQueue((current) => current.slice(1));
              }}
            />
          ) : <>
          {addMode === "choose" && <div className="add-choices">
            <button onClick={() => setAddMode("upload")}><span className="choice-icon violet"><Icon name="upload" size={22}/></span><div><strong>Upload a PDF</strong><p>Choose a paper saved on your computer.</p></div></button>
            <button onClick={() => setAddMode("search")}><span className="choice-icon blue"><Icon name="globe" size={22}/></span><div><strong>Search online</strong><p>Find papers on arXiv by title, author, or topic.</p></div></button>
          </div>}

          {addMode === "upload" && <div className="upload-panel">
            <button className="back-button" onClick={() => setAddMode("choose")}>← Back</button>
            <button className="drop-zone" onClick={() => fileInput.current?.click()} onDragOver={(event) => event.preventDefault()} onDrop={onDrop} disabled={uploading}>
              <span><Icon name="upload" size={25}/></span><strong>{uploading ? "Uploading and reading…" : "Choose a PDF or drag it here"}</strong><small>PDF files up to the backend’s configured limit</small>
            </button>
            {uploading && <button className="cancel-ingest" onClick={cancelUpload}>Cancel</button>}
            <input ref={fileInput} className="hidden-input" type="file" accept="application/pdf,.pdf" onChange={(event: ChangeEvent<HTMLInputElement>) => void handleFile(event.target.files?.[0])}/>
            {uploadError && <p className="form-error">{uploadError}</p>}
          </div>}

          {addMode === "search" && <div className="search-panel">
            <button className="back-button" onClick={() => setAddMode("choose")}>← Back</button>
            <form onSubmit={runSearch}><Icon name="search" size={18}/><input autoFocus value={searchQuery} onChange={(event) => setSearchQuery(event.target.value)} placeholder="Search by title, author, or topic…"/><button disabled={searchQuery.trim().length < 2 || searching}>{searching ? "Searching…" : "Search"}</button></form>
            {searchError && <p className="form-error">{searchError}</p>}
            <div className="search-results">{searchResults.map((result) => {
              const isIngesting = ingestingIds.has(result.id);
              return <article key={result.id}><div><span>arXiv</span><small>{result.published}</small></div><h3>{result.title}</h3><p>{result.authors}</p>
                {isIngesting ? <div className="ingest-progress"><span>Reading…</span><button className="cancel-ingest" onClick={() => cancelArxivIngest(result.id)}>Cancel</button></div>
                  : <button onClick={() => void addArxivPaper(result)}>Add to chat</button>}
              </article>;
            })}</div>
          </div>}
          </>}
        </section>
      </div>}
    </main>
  );
}
