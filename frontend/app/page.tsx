"use client";

import { ChangeEvent, DragEvent, FormEvent, useEffect, useRef, useState } from "react";
import {
  answerClarification,
  askAssistant,
  buildPaperGuide,
  checkFeynmanExplanation,
  createSession,
  deleteSession,
  detachPaper,
  getFeynmanPrompts,
  getPaperStatus,
  ingestArxivPaper,
  listGaps,
  listPapersForSession,
  listSessions,
  recordGapFeedback,
  recordQueryFeedback,
  renameSession,
  searchPapers,
  uploadPaper,
} from "@/lib/api";
import type { ChatHistoryItem, PaperContext, PaperIngestResult, PaperSearchResult, SessionMetadata } from "@/lib/api";
import type { Citation, FeynmanCheckResult, FeynmanPrompt, GapCandidate, PaperGuide, PendingQuestion, QueryResponse } from "@/lib/types";
import GraphBuildAnimation from "./GraphBuildAnimation";
import ConvergenceRitual from "./ConvergenceRitual";
import GraphExplorer from "./GraphExplorer";
import Tour, { TourStep } from "./Tour";
import PaperMap from "./PaperMap";

type IconName = "atlas" | "plus" | "send" | "paper" | "search" | "upload" | "close" | "quote" | "check" | "globe" | "thumbUp" | "thumbDown" | "rename" | "graph" | "help" | "quiz";
type AddMode = "choose" | "upload" | "search";

interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  citations?: Citation[];
  retrievalMode?: QueryResponse["retrieval_mode"];
  confidence?: QueryResponse["confidence"];
  candidates?: QueryResponse["candidates"];
  // Only set on an "ambiguous" retrieval_mode response - lets
  // selectCandidate mark the ClarificationOrchestrator question answered
  // once the user picks one, instead of leaving it "open" forever. Every
  // never-answered query_disambiguation question is intentionally never
  // session-filtered (see service/routers/clarifications.py's comment -
  // it's meant to be resolved inline, in the same turn) - without marking
  // it answered here, it leaks into every session's proactive
  // clarification poll permanently.
  clarificationQuestionId?: string | null;
  guide?: PaperGuide;
  guideLoading?: boolean;
  guideError?: string;
  // Which paper this guide walkthrough is for - lets the "Test yourself"
  // check at the end scope its questions to this paper's own graph nodes.
  paperId?: string;
  notice?: boolean;
  clarification?: PendingQuestion;
  // Set instead of `clarification` when more than one question arrives in
  // the same checkGuidance() pass - rendered as one compact batched card
  // instead of N full message bubbles.
  clarifications?: PendingQuestion[];
  gaps?: GapCandidate[];
  feedbackGiven?: boolean;
  // Set by the standalone "Test yourself" trigger (openFeynmanCheck) -
  // previously Feynman Check only ever appeared as the forced last stop
  // of Guided Reading, with no other entry point anywhere in the app
  // (confirmed live via a fresh-user audit: a user who skims the guide
  // and jumps straight to chat, a completely normal path, never sees it
  // exists). feynmanPickPaper is set instead when the session has more
  // than one paper and none has been chosen yet for this quiz message.
  feynmanPaperId?: string;
  feynmanPickPaper?: boolean;
}

function gapKey(candidate: { node_a_id: string; node_b_id: string }) {
  return `${candidate.node_a_id}:${candidate.node_b_id}`;
}

// Kept close in length to the original flat "Reading…" label on purpose -
// .ingest-progress is absolutely positioned in a slot sized for that short
// text (see .search-results article's reserved right padding in
// globals.css); a noticeably longer label would overlap the title instead
// of just replacing the badge text.
const INGEST_STAGE_LABELS: Record<string, string> = {
  downloading: "Fetching…",
  // The real, long phase (60-90s+, see service/routers/papers.py's
  // status="extracting" write) previously fell through to the generic
  // "Reading…" fallback below with nothing distinguishing it from the
  // few-second "downloading" phase - confirmed live as a real silent
  // wait with no visible signal that anything was actually happening.
  extracting: "Extracting…",
};

// Two distinct citations can render the same visible label - confirmed
// live: "2. Background · p. 2" and "2 Background · p. 2" back to back,
// same paper/page, section title differing only by a stray period that
// extraction isn't consistent about. Both stay real, separate evidence in
// message.citations (their own quotes aren't lost); this only collapses
// what looks like the same source appearing twice in this one list, same
// content-based approach as GraphExplorer's connections dedup.
function citationDedupeKey(citation: Citation, index: number): string {
  const normalizedSection = (citation.section ?? "").toLowerCase().trim().replace(/^(\d+)\.\s*/, "$1 ");
  if (!citation.paper_id && !normalizedSection && citation.page_start == null) return `unkeyed-${index}`;
  return `${citation.paper_id ?? ""}:${normalizedSection}:${citation.page_start ?? ""}:${citation.page_end ?? ""}`;
}

function dedupeCitations(citations: Citation[]): Citation[] {
  const seen = new Set<string>();
  return citations.filter((citation, index) => {
    const key = citationDedupeKey(citation, index);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function ingestStageLabel(stage: string | undefined): string {
  return (stage && INGEST_STAGE_LABELS[stage]) || "Reading…";
}

const icons: Record<IconName, React.ReactNode> = {
  atlas: <><circle cx="12" cy="12" r="3"/><path d="M12 3v6M12 15v6M3 12h6M15 12h6M5.6 5.6l4.2 4.2M14.2 14.2l4.2 4.2M18.4 5.6l-4.2 4.2M9.8 14.2l-4.2 4.2"/></>,
  plus: <path d="M12 5v14M5 12h14"/>,
  send: <><path d="M22 2L9.5 14.5M22 2l-8 20-4.5-7.5L2 10z"/></>,
  paper: <><path d="M6 3h9l4 4v14H6z"/><path d="M15 3v5h5M9 12h7M9 16h7"/></>,
  search: <><circle cx="10.5" cy="10.5" r="6.5"/><path d="M15.5 15.5L21 21"/></>,
  upload: <><path d="M12 16V3M7 8l5-5 5 5"/><path d="M4 15v5h16v-5"/></>,
  close: <path d="M6 6l12 12M18 6L6 18"/>,
  quote: <><path d="M9 11H4v6h6v-7c0-3-1.5-5-4-6M20 11h-5v6h6v-7c0-3-1.5-5-4-6"/></>,
  check: <path d="M5 12l4 4L19 6"/>,
  globe: <><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c3 3.4 3 14.6 0 18M12 3c-3 3.4-3 14.6 0 18"/></>,
  thumbUp: <path d="M7 10v11H3V10zM7 10l4-7a2 2 0 0 1 2 2v5h5.5a2 2 0 0 1 2 2.4l-1.5 7A2 2 0 0 1 17 21H7"/>,
  thumbDown: <path d="M17 14V3h4v11zM17 14l-4 7a2 2 0 0 1-2-2v-5H5.5a2 2 0 0 1-2-2.4l1.5-7A2 2 0 0 1 7 3h10"/>,
  rename: <path d="M12 20h9M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4Z"/>,
  graph: <><circle cx="6" cy="6" r="2.6"/><circle cx="18" cy="6" r="2.6"/><circle cx="12" cy="18" r="2.6"/><path d="M8.3 6.7L15.7 6.7M7.4 8.2L10.6 15.8M16.6 8.2L13.4 15.8"/></>,
  help: <><circle cx="12" cy="12" r="9"/><path d="M9.1 9a3 3 0 1 1 4.6 2.6c-.9.5-1.7 1.1-1.7 2.4"/><path d="M12 17.5h.01"/></>,
  quiz: <><path d="M9 21h6M10 18h4M8.5 12.5A5 5 0 1 1 15.5 12.5c-.7 1-1.5 1.6-1.5 3H10c0-1.4-.8-2-1.5-3Z"/></>,
};

function Icon({ name, size = 19 }: { name: IconName; size?: number }) {
  return <svg aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">{icons[name]}</svg>;
}

const ATLAS_TOUR_STEPS: TourStep[] = [
  {
    target: ".brand",
    title: "Welcome to Atlas",
    body: "A quick tour of the essentials: five stops, skip anytime.",
    placement: "bottom",
    clickable: false,
  },
  {
    target: ".add-paper-button",
    title: "Add a paper",
    body: "Click here to upload a PDF or search arXiv. Atlas reads it, builds a real knowledge graph from what's inside, and walks you through it section by section automatically.",
    placement: "bottom",
  },
  {
    target: ".composer",
    title: "Ask anything",
    body: "Ask a question here at any time. Once a paper's added, answers cite the exact section and page they came from.",
    placement: "top",
  },
  {
    target: ".graph-explorer-toggle",
    title: "Explore the graph",
    body: "Once you've added a paper, open this to see every idea it extracted as a real, clickable graph, not just a summary.",
    placement: "bottom",
  },
  {
    target: ".session-switcher-toggle",
    title: "Sessions",
    body: "Each session is its own research thread with its own papers and graph. Switch or start a new one here anytime.",
    placement: "bottom",
  },
];

const FLOW_REVEAL_INTERVAL_MS = 200;

function FlowDiagram({ guide }: { guide: NonNullable<PaperGuide["sections"][number]["diagram"]> }) {
  // Builds itself in on a timer (same idea as GraphBuildAnimation's
  // node-by-node reveal) instead of rendering every node at once -
  // remounts (and so replays) each time its parent section becomes the
  // active stop in GuidedReading below, since it's only rendered while
  // that stop is showing.
  const [revealedCount, setRevealedCount] = useState(0);
  // Click a node to focus it: the others dim so the one detail you asked
  // about actually stands out, instead of every node's caption competing
  // for attention at once. Cleared on remount along with the reveal timer
  // so re-entering the section starts from a neutral state, not whatever
  // was focused last time.
  const [focusedId, setFocusedId] = useState<string | null>(null);
  useEffect(() => {
    setRevealedCount(0);
    setFocusedId(null);
    if (guide.nodes.length === 0) return;
    let index = 0;
    const timer = window.setInterval(() => {
      index += 1;
      setRevealedCount(index);
      if (index >= guide.nodes.length) window.clearInterval(timer);
    }, FLOW_REVEAL_INTERVAL_MS);
    return () => window.clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [guide]);

  const revealed = guide.nodes.slice(0, revealedCount);
  return <div className="guide-diagram">
    <strong>{guide.title}</strong>
    <div className="flow-track">{revealed.map((node, index) => {
      const next = revealed[index + 1];
      const edge = next ? guide.edges.find((item) => item.source === node.id && item.target === next.id) : undefined;
      const isFocused = focusedId === node.id;
      const dimmed = focusedId !== null && !isFocused;
      return <div className="flow-step flow-step-in" key={node.id}>
        <button
          type="button"
          className={`flow-node${isFocused ? " focused" : ""}${dimmed ? " dimmed" : ""}`}
          onClick={() => setFocusedId((current) => (current === node.id ? null : node.id))}
        >
          <span>{node.label}</span>{node.detail && <small>{node.detail}</small>}
        </button>
        {next && <div className={`flow-arrow${dimmed ? " dimmed" : ""}`}><i>↓</i>{edge?.label && <small>{edge.label}</small>}</div>}
      </div>;
    })}</div>
  </div>;
}

type GuideStop = { kind: "overview" } | { kind: "section"; section: PaperGuide["sections"][number] };

function GuidedReading({ guide, paperId, sessionId, onViewNodeInGraph }: { guide: PaperGuide; paperId?: string; sessionId?: string; onViewNodeInGraph: (nodeId: string) => void }) {
  const stops: GuideStop[] = [
    { kind: "overview" },
    ...guide.sections.map((section) => ({ kind: "section" as const, section })),
  ];
  const [stopIndex, setStopIndex] = useState(0);
  // Which way we last moved, so the stop can slide in from the direction
  // that matches the nav action (Next/a later dot slides in from the
  // right, Back/an earlier dot from the left) instead of a flat cut.
  const [direction, setDirection] = useState<1 | -1>(1);
  const [quizOpen, setQuizOpen] = useState(false);
  const stop = stops[stopIndex];
  const isLastStop = stopIndex === stops.length - 1;

  function goTo(nextIndex: number) {
    setDirection(nextIndex >= stopIndex ? 1 : -1);
    setStopIndex(Math.max(0, Math.min(stops.length - 1, nextIndex)));
  }

  return <section className="guided-reading">
    <header>
      <div><span>Guided reading</span><strong>Stop {stopIndex + 1} of {stops.length} · about {guide.reading_time_minutes} min</strong></div>
      <Icon name="atlas" size={20}/>
    </header>
    <div className="guide-dots">
      {stops.map((s, index) => (
        <button
          type="button"
          key={s.kind === "overview" ? "overview" : s.section.title}
          className={`guide-dot${index === stopIndex ? " active" : ""}${index < stopIndex ? " visited" : ""}`}
          aria-label={`Jump to stop ${index + 1}`}
          onClick={() => goTo(index)}
        />
      ))}
    </div>
    <div key={stopIndex} className={direction === 1 ? "guide-stop slide-forward" : "guide-stop slide-back"}>
      {stop.kind === "overview" ? (
        <div className="guide-overview"><small>The big picture</small><p>{guide.big_picture}</p></div>
      ) : (
        <div className="guide-sections">
          <article>
            <div className="guide-section-heading">
              <span>{stopIndex}</span>
              <div>
                <small>{stop.section.page_start ? `Pages ${stop.section.page_start}${stop.section.page_end && stop.section.page_end !== stop.section.page_start ? `–${stop.section.page_end}` : ""}` : "Paper section"}</small>
                <h3>{stop.section.title}</h3>
              </div>
            </div>
            <p>{stop.section.plain_language}</p>
            {stop.section.key_points.length > 0 && <ul>{stop.section.key_points.map((point) => <li key={point}>{point}</li>)}</ul>}
            {stop.section.diagram && stop.section.diagram.nodes.length > 0 && <FlowDiagram guide={stop.section.diagram}/>}
            <div className="why-it-matters"><strong>Why this matters</strong><p>{stop.section.why_it_matters}</p></div>
          </article>
        </div>
      )}
    </div>
    <footer className="guide-nav">
      <button type="button" onClick={() => goTo(stopIndex - 1)} disabled={stopIndex === 0}>← Back</button>
      {isLastStop ? (
        paperId && sessionId ? (
          quizOpen ? <FeynmanCheck paperId={paperId} sessionId={sessionId} onViewNodeInGraph={onViewNodeInGraph}/> : (
            <div className="guide-complete">
              <div className="guide-complete-bar"><i/></div>
              <div className="guide-complete-copy"><span className="guide-complete-check"><Icon name="check" size={13}/></span><small>Walkthrough complete. Test your understanding if you’d like.</small></div>
              <button type="button" className="optional-quiz-button" onClick={() => setQuizOpen(true)}>Take the optional quiz</button>
            </div>
          )
        ) : (
          <div className="guide-complete">
            <div className="guide-complete-bar"><i/></div>
            <div className="guide-complete-copy">
              <span className="guide-complete-check"><Icon name="check" size={13}/></span>
              <small>That’s the full walkthrough. Ask a question below and I’ll answer from this paper with page-level sources.</small>
            </div>
          </div>
        )
      ) : (
        <button type="button" onClick={() => goTo(stopIndex + 1)}>Next →</button>
      )}
    </footer>
  </section>;
}

const FEYNMAN_VERDICT_HEADLINE: Record<FeynmanCheckResult["verdict"], string> = {
  strong: "You’ve got it.",
  weak: "Partly there.",
  wrong: "Not quite.",
};

// Matches MAX_EXPLANATION_CHARS in service/routers/feynman.py - this is a
// UX nicety (stops the browser from letting you type past the limit), the
// real enforcement is the backend's Pydantic max_length.
const MAX_EXPLANATION_CHARS = 4000;

type FeynmanPhase = "loading" | "question" | "grading" | "result" | "error" | "done";

// Sits in the exact slot the static "guide-complete" footer used to occupy
// (the last stop of GuidedReading) - this is the Feynman Method's forced-
// retrieval step (write your own explanation before being told the
// answer), graded against this paper's own graph nodes rather than the
// model's general knowledge. See agent/feynman_checker.py.
function FeynmanCheck({ paperId, sessionId, onViewNodeInGraph }: { paperId: string; sessionId: string; onViewNodeInGraph: (nodeId: string) => void }) {
  const [phase, setPhase] = useState<FeynmanPhase>("loading");
  const [prompts, setPrompts] = useState<FeynmanPrompt[]>([]);
  const [promptIndex, setPromptIndex] = useState(0);
  const [explanation, setExplanation] = useState("");
  const [result, setResult] = useState<FeynmanCheckResult | null>(null);
  const [error, setError] = useState("");
  // A React-state guard (checking `phase === "grading"`) is NOT enough here:
  // React 18 batches state updates from synchronous events, so two rapid
  // clicks in the same tick both read the same stale `phase` before either
  // setPhase("grading") call has been applied, and both pass the guard -
  // confirmed live, 3 rapid clicks fired 3 real grading calls. This ref is
  // set synchronously, before any await, so the second click sees it
  // immediately - same fix as the ingestControllers.current guard elsewhere
  // in this file for the identical race class.
  const submittingRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    getFeynmanPrompts(paperId, sessionId)
      .then((loaded) => {
        if (cancelled) return;
        setPrompts(loaded);
        setPhase(loaded.length > 0 ? "question" : "done");
      })
      .catch(() => { if (!cancelled) setPhase("done"); });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [paperId, sessionId]);

  const currentPrompt = prompts[promptIndex];

  async function submit() {
    if (!currentPrompt || !explanation.trim() || submittingRef.current) return;
    submittingRef.current = true;
    setPhase("grading");
    try {
      const graded = await checkFeynmanExplanation(paperId, currentPrompt.node_id, sessionId, explanation.trim());
      setResult(graded);
      setPhase("result");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not grade this explanation.");
      setPhase("error");
    } finally {
      submittingRef.current = false;
    }
  }

  function next() {
    setExplanation("");
    setResult(null);
    setError("");
    if (promptIndex + 1 < prompts.length) {
      setPromptIndex((index) => index + 1);
      setPhase("question");
    } else {
      setPhase("done");
    }
  }

  if (phase === "loading") return <div className="feynman-check"><small>Preparing a quick check on what you just read…</small></div>;

  if (phase === "done") {
    return <div className="guide-complete">
      <div className="guide-complete-bar"><i/></div>
      <div className="guide-complete-copy">
        <span className="guide-complete-check"><Icon name="check" size={13}/></span>
        <small>That’s the full walkthrough. Ask a question below and I’ll answer from this paper with page-level sources.</small>
      </div>
    </div>;
  }

  const graphNodeId = result?.citation?.source_kind === "graph" ? result.citation.node_ids?.[0] : undefined;

  return <div className="feynman-check">
    <div className="feynman-check-header"><span>Test yourself</span><small>{promptIndex + 1} of {prompts.length}</small></div>
    {(phase === "question" || phase === "grading") && currentPrompt && <>
      <p className="feynman-check-question">{currentPrompt.question}</p>
      <textarea
        className="feynman-check-input"
        value={explanation}
        onChange={(event) => setExplanation(event.target.value)}
        placeholder="Explain it in your own words…"
        disabled={phase === "grading"}
        maxLength={MAX_EXPLANATION_CHARS}
        rows={3}
      />
      <div className="feynman-check-actions">
        <button type="button" className="feynman-check-skip" onClick={next}>Skip</button>
        <button type="button" onClick={() => void submit()} disabled={phase === "grading" || !explanation.trim()}>
          {phase === "grading" ? "Grading…" : "Check my understanding"}
        </button>
      </div>
    </>}
    {phase === "error" && <>
      <p className="feynman-check-error">{error}</p>
      <div className="feynman-check-actions">
        <button type="button" className="feynman-check-skip" onClick={next}>Skip</button>
        <button type="button" onClick={() => void submit()}>Try again</button>
      </div>
    </>}
    {phase === "result" && result && <div className={`feynman-verdict feynman-verdict-${result.verdict}`}>
      <strong>{FEYNMAN_VERDICT_HEADLINE[result.verdict]}</strong>
      <p>{result.explanation}</p>
      {graphNodeId && <button className="citation-view-graph" onClick={() => onViewNodeInGraph(graphNodeId)} title="View this node in the graph"><Icon name="graph" size={12}/>View in graph</button>}
      <div className="feynman-check-actions">
        <button type="button" onClick={next}>{promptIndex + 1 < prompts.length ? "Next question" : "Done"}</button>
      </div>
    </div>}
  </div>;
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
  // Replaces two stacked native window.prompt() calls - confirmed live as
  // the one moment in the whole app that looked like a different,
  // unstyled product, and prompt() blocks the entire page/tab while open.
  const [createSessionOpen, setCreateSessionOpen] = useState(false);
  const [newSessionName, setNewSessionName] = useState("");
  const [newSessionGoal, setNewSessionGoal] = useState("");
  const [creatingSession, setCreatingSession] = useState(false);
  const [createSessionError, setCreateSessionError] = useState("");
  const [uploading, setUploading] = useState(false);
  const uploadBatchCancelled = useRef(false);
  const [uploadError, setUploadError] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<PaperSearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  // Multiple searches can be in flight at once now - keyed by result.id so
  // each card can show its own "Reading…"/cancel independently instead of
  // one global flag blocking every other result while it's in progress.
  const [ingestingIds, setIngestingIds] = useState<Set<string>>(new Set());
  const [graphExplorerOpen, setGraphExplorerOpen] = useState(false);
  const [paperMapOpen, setPaperMapOpen] = useState(false);
  const [tourOpen, setTourOpen] = useState(false);
  const [tourSeen, setTourSeen] = useState(false);
  const [graphFocusNodeId, setGraphFocusNodeId] = useState<string | null>(null);
  // Polled from the backend while a card is mid-ingest, purely to turn the
  // flat "Reading…" label into real stage feedback (downloading vs.
  // extracting/writing the guide) - never blocks or gates anything, so a
  // missed poll just leaves that card's label one step stale for a beat.
  const [ingestStage, setIngestStage] = useState<Record<string, string>>({});
  const ingestControllers = useRef<Map<string, AbortController>>(new Map());
  const uploadController = useRef<AbortController | null>(null);
  const [searchError, setSearchError] = useState("");
  const [expandedCitation, setExpandedCitation] = useState<string | null>(null);
  const [dismissedGapKeys, setDismissedGapKeys] = useState<Set<string>>(new Set());
  // Clarifications and gap suggestions used to land as two separate full
  // -content messages right alongside Guided Reading the moment a paper
  // finished ingesting - confirmed as real cognitive overload on a first
  // -time user's very first minute in the app (three simultaneous asks
  // before they'd read a word). Collapsed by default behind one summary
  // card instead - Guided Reading stays the prominent thing on screen,
  // the rest is one click away whenever the user is ready for it.
  const [expandedFindings, setExpandedFindings] = useState<Set<string>>(new Set());
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
  const initRan = useRef(false);
  useEffect(() => {
    // React 18 StrictMode dev-mode double-invokes effects with no cleanup
    // - without this guard, both invocations raced past "no session yet"
    // and each created its own, leaving two "Untitled session" entries
    // from a single page load.
    if (initRan.current) return;
    initRan.current = true;
    void initSession();
    setTourSeen(window.localStorage.getItem("atlas-tour-seen") === "1");
  }, []);
  const sessionSwitcherRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    // graphExplorerOpen was missing here - confirmed live that Escape did
    // nothing while Graph Explorer was open, inconsistent with every other
    // modal in the app (the Add Paper dialog, the session switcher) which
    // this same handler already closes.
    if (!sessionMenuOpen && !addOpen && !graphExplorerOpen && !createSessionOpen) return;
    function onKeyDown(event: KeyboardEvent) {
      if (event.key !== "Escape") return;
      setSessionMenuOpen(false);
      setAddOpen(false);
      setGraphExplorerOpen(false);
      setCreateSessionOpen(false);
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [sessionMenuOpen, addOpen, graphExplorerOpen, createSessionOpen]);
  useEffect(() => {
    // Neither modal had a focus trap - confirmed live: tabbing past
    // Cancel escaped the dialog entirely into background page elements
    // (once landing on a "Remove paper" chip hidden behind the still-open
    // modal), leaving a keyboard-only user acting on content they can't
    // see is focused. Both modals share the same .add-modal/.modal-card
    // markup and are never open at the same time, so one shared trap
    // covers both rather than duplicating this per modal.
    if (!addOpen && !createSessionOpen) return;
    function onKeyDown(event: KeyboardEvent) {
      if (event.key !== "Tab") return;
      const modal = document.querySelector(".add-modal .modal-card");
      if (!modal) return;
      const focusable = modal.querySelectorAll<HTMLElement>(
        'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
      );
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [addOpen, createSessionOpen]);
  useEffect(() => {
    if (!sessionMenuOpen) return;
    function onMouseDown(event: MouseEvent) {
      if (sessionSwitcherRef.current?.contains(event.target as Node)) return;
      setSessionMenuOpen(false);
    }
    document.addEventListener("mousedown", onMouseDown);
    return () => document.removeEventListener("mousedown", onMouseDown);
  }, [sessionMenuOpen]);

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
        restoredMessages = (JSON.parse(savedMessages) as Message[]).map(({ clarification: _clarification, clarifications: _clarifications, ...message }) => message);
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

  function openCreateSession() {
    setSessionMenuOpen(false);
    setNewSessionName("");
    setNewSessionGoal("");
    setCreateSessionError("");
    setCreateSessionOpen(true);
  }

  async function submitCreateSession(event: FormEvent) {
    event.preventDefault();
    const name = newSessionName.trim();
    if (!name || creatingSession) return;
    setCreatingSession(true);
    setCreateSessionError("");
    try {
      const created = await createSession(name, newSessionGoal.trim() || undefined);
      setSessions((current) => [created, ...current]);
      setCreateSessionOpen(false);
      await switchToSession(created);
    } catch (error) {
      setCreateSessionError(error instanceof Error ? error.message : "Could not create the session.");
    } finally {
      setCreatingSession(false);
    }
  }

  async function renameSessionPrompt(session: SessionMetadata) {
    const name = window.prompt("Rename session:", session.name)?.trim();
    if (!name || name === session.name) return;
    try {
      const updated = await renameSession(session.id, name);
      setSessions((current) => current.map((s) => (s.id === updated.id ? updated : s)));
      setCurrentSession((current) => (current?.id === updated.id ? updated : current));
    } catch (error) {
      setMessages((current) => [...current, {
        id: crypto.randomUUID(),
        role: "assistant",
        text: error instanceof Error ? error.message : "Could not rename the session.",
        notice: true,
      }]);
    }
  }

  async function deleteSessionPrompt(session: SessionMetadata) {
    if (!window.confirm(`Delete "${session.name}"? This removes its papers and can't be undone.`)) return;
    try {
      await deleteSession(session.id);
      window.localStorage.removeItem(`atlas-messages-${session.id}`);
      setSessions((current) => current.filter((s) => s.id !== session.id));
      // Reuses initSession's own fallback (pick another session, or
      // create a fresh default one if none remain) instead of
      // duplicating that logic here.
      if (session.id === currentSession?.id) await initSession();
    } catch (error) {
      setMessages((current) => [...current, {
        id: crypto.randomUUID(),
        role: "assistant",
        text: error instanceof Error ? error.message : "Could not delete the session.",
        notice: true,
      }]);
    }
  }

  // Proactive, unsolicited content: a missing suggestion is fine, an error
  // toast for something the user didn't ask for is not - both halves fail
  // silently and independently.
  async function checkGuidance() {
    let gaps: GapCandidate[] | undefined;
    try {
      const fresh = (await listGaps(3, sessionIdRef.current || undefined)).filter(
        (g) => !shownGapKeys.current.has(gapKey(g)),
      );
      fresh.forEach((g) => shownGapKeys.current.add(gapKey(g)));
      if (fresh.length) gaps = fresh;
    } catch {
      // silent
    }
    // One combined, collapsed-by-default card instead of up to two
    // separate full-content messages - see expandedFindings above for why.
    if (gaps) {
      setMessages((current) => [
        ...current,
        { id: crypto.randomUUID(), role: "assistant" as const, text: "", notice: true, gaps },
      ]);
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
    const evidence = candidate.citations
      .slice(0, 4)
      .map((citation) => `${citation.shared_node_name} (${citation.source_section ?? "unknown section"}): “${citation.source_quote}”`)
      .join("\n");
    const question = `Verify this possible graph connection between ${candidate.node_a_name} and ${candidate.node_b_name}. Treat it as a hypothesis, not an established fact. Does the paper text actually support the relationship? Distinguish graph evidence from direct paper evidence.${candidate.explanation ? `\nGraph hypothesis: ${candidate.explanation}` : ""}${evidence ? `\nUnderlying graph evidence:\n${evidence}` : ""}`;
    void recordGapFeedback(candidate.node_a_id, candidate.node_b_id, true).catch(() => {});
    // Gap suggestions are hypotheses from this session's graph. Keep the
    // current papers attached so the tutor can verify the graph claim
    // against their source text instead of answering from graph labels alone.
    void ask(undefined, question, papers);
  }

  function viewNodeInGraph(nodeId: string) {
    setGraphFocusNodeId(nodeId);
    setGraphExplorerOpen(true);
  }

  function askFromGraphExplorer(question: string) {
    // Graph Explorer's nodes come from the whole session's graph
    // (export_session_graph), not necessarily the chat's currently-attached
    // paper set - same unscoped-search reasoning askAboutGap already
    // applies for the identical reason, so a node from a paper the user
    // detached from chat doesn't wrongly come back "no information."
    setGraphExplorerOpen(false);
    void ask(undefined, question, []);
  }

  function selectCandidate(candidate: { node_id: string; name: string }, clarificationQuestionId?: string | null) {
    // Skips search_nodes/ambiguity detection entirely (see
    // QueryAgent.answer's node_id param) - re-asking the same free text
    // would just risk landing on the same ambiguous result again.
    void ask(undefined, `What do you know about "${candidate.name}"?`, undefined, candidate.node_id);
    // Marks the ClarificationOrchestrator question answered - without
    // this it stays "open" forever (query_disambiguation questions are
    // deliberately never session-filtered, see clarifications.py), which
    // means every unresolved ambiguous query leaks into every session's
    // proactive clarification poll permanently. Best-effort: a failure
    // here shouldn't block the real answer above, same as dismissGap's
    // fire-and-forget feedback call.
    if (clarificationQuestionId) {
      void answerClarification(clarificationQuestionId, candidate.node_id).catch(() => {});
    }
  }

  function dismissGap(candidate: GapCandidate) {
    setDismissedGapKeys((current) => new Set(current).add(gapKey(candidate)));
    void recordGapFeedback(candidate.node_a_id, candidate.node_b_id, false).catch(() => {});
  }

  async function ask(event?: FormEvent, suggested = query, papersOverride?: PaperContext[] | null, nodeId?: string) {
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
      const response = await askAssistant(question, history, effectivePapers, currentSession?.goal, nodeId, currentSession?.id);
      setMessages((current) => [...current, {
        id: crypto.randomUUID(),
        role: "assistant",
        text: response.answer,
        citations: response.citations,
        retrievalMode: response.retrieval_mode,
        confidence: response.confidence,
        candidates: response.candidates,
        clarificationQuestionId: response.clarification_question_id,
      }]);
    } catch (error) {
      setMessages((current) => [...current, {
        id: crypto.randomUUID(),
        role: "assistant",
        text: error instanceof Error ? `I couldn't reach Atlas: ${error.message}` : "I couldn't reach Atlas.",
        notice: true,
      }]);
    } finally {
      setLoading(false);
      window.setTimeout(() => composerInput.current?.focus(), 50);
    }
  }

  function openFeynmanCheck() {
    if (papers.length === 0) return;
    const id = crypto.randomUUID();
    if (papers.length === 1) {
      setMessages((current) => [...current, { id, role: "assistant", text: "", feynmanPaperId: papers[0].id }]);
    } else {
      setMessages((current) => [...current, { id, role: "assistant", text: "Which paper do you want to be quizzed on?", feynmanPickPaper: true }]);
    }
  }

  function openAddPaper(mode: AddMode = "choose") {
    setSessionMenuOpen(false);
    setAddMode(mode);
    setUploadError("");
    setSearchError("");
    setAddOpen(true);
  }

  async function handleFiles(files: File[]) {
    if (!files.length) return;
    const validFiles = files.filter((file) => file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf"));
    if (validFiles.length !== files.length) setUploadError("Only PDF files were included; other files were skipped.");
    if (validFiles.length === 0) {
      setUploadError("Please choose PDF files.");
      return;
    }
    // Same synchronous-ref guard as addArxivPaper, and for the same
    // reason: the drop-zone's disabled={uploading} only takes effect
    // after React commits the setUploading(true) below, so a rapid
    // double-drop (or a drop landing while the native file-picker's own
    // change event is still in flight) can start two uploads before that
    // commit lands.
    if (uploadController.current) return;
    uploadBatchCancelled.current = false;
    setUploading(true);
    if (validFiles.length === files.length) setUploadError("");
    const failures: string[] = [];
    try {
      for (const file of validFiles) {
        if (uploadBatchCancelled.current) break;
        const controller = new AbortController();
        uploadController.current = controller;
        try {
          const uploaded = await uploadPaper(file, controller.signal, sessionIdRef.current);
          setBuildingGraphQueue((current) => [...current, uploaded]);
        } catch (error) {
          if (error instanceof DOMException && error.name === "AbortError") break;
          failures.push(`${file.name}: ${error instanceof Error ? error.message : "upload failed"}`);
        } finally {
          uploadController.current = null;
        }
      }
      if (failures.length > 0) {
        setUploadError(`Could not add ${failures.length} file${failures.length === 1 ? "" : "s"}: ${failures.join("; ")}`);
      }
    } finally {
      setUploading(false);
      uploadController.current = null;
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  function cancelUpload() {
    uploadBatchCancelled.current = true;
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
      { id: guideMessageId, role: "assistant", text: "", guideLoading: true, paperId: nextPaper.id },
    ]);
    try {
      const guide = await buildPaperGuide(nextPaper.id);
      setMessages((current) => current.map((message) => message.id === guideMessageId ? { ...message, guideLoading: false, guide } : message));
    } catch (error) {
      const detail = error instanceof Error ? error.message : "Could not build the walkthrough.";
      setMessages((current) => current.map((message) => message.id === guideMessageId ? { ...message, guideLoading: false, guideError: detail, text: "The paper was added and is ready for questions, but I couldn't generate its guided walkthrough." } : message));
    }
  }

  function addPaper(nextPaper: PaperContext, closeModal = true) {
    // Papers are sourced from the backend per session (listPapersForSession)
    // now, not localStorage - this is just the optimistic local update for
    // the paper the backend just confirmed ingesting into sessionIdRef.current.
    setPapers((current) => (current.some((existing) => existing.id === nextPaper.id) ? current : [...current, nextPaper]));
    setMessages((current) => [...current, { id: crypto.randomUUID(), role: "assistant", text: `I've added "${nextPaper.title}" to this conversation. Ask me to summarize it, explain a section, or examine its evidence.`, notice: true }]);
    if (closeModal) setAddOpen(false);
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
    // The notice is posted here, not inside the setPapers updater above -
    // React 18 dev mode double-invokes updater functions to catch impure
    // ones, and a setMessages call nested inside setPapers's updater was
    // firing twice, posting the "Removed: ..." notice twice per click.
    const removed = papers.find((existing) => existing.id === paperId);
    const updated = papers.filter((existing) => existing.id !== paperId);
    setPapers(updated);
    setMessages((current) => [...current, { id: crypto.randomUUID(), role: "assistant", text: `Removed${removed ? `: "${removed.title}"` : ""} from this conversation.${updated.length === 0 ? " We're back to searching everything." : ""}`, notice: true }]);
    void detachPaper(paperId, sessionIdRef.current).catch((error) => {
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
    // ingestControllers.current, not ingestingIds - a ref mutation is
    // visible synchronously to the very next call, while the state-set
    // below only commits on React's next render. A rapid double-click
    // fires both handler invocations before that commit lands, so an
    // ingestingIds-only guard lets both through - confirmed live: it
    // double-ingested the same paper, extracting it twice and duplicating
    // its entity-merge clarifications.
    if (ingestControllers.current.has(result.id)) return;
    const controller = new AbortController();
    ingestControllers.current.set(result.id, controller);
    setIngestingIds((current) => new Set(current).add(result.id));
    setIngestStage((current) => ({ ...current, [result.id]: "downloading" }));
    setSearchError("");
    // Runs alongside the single blocking ingest request below, not instead
    // of it - the backend already writes its status ("downloading", then
    // "extracting") to Firestore as it goes, so a concurrent poll can see
    // those writes mid-request instead of only finding out at the end.
    const pollTimer = window.setInterval(() => {
      void getPaperStatus(result.id.replace(/^arxiv:/, "")).then((status) => {
        setIngestStage((current) => (current[result.id] ? { ...current, [result.id]: status } : current));
      });
    }, 1500);
    try {
      const ingested = await ingestArxivPaper(result, controller.signal, sessionIdRef.current);
      setBuildingGraphQueue((current) => [...current, ingested]);
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      setSearchError(error instanceof Error ? error.message : "Could not read this paper.");
    } finally {
      window.clearInterval(pollTimer);
      ingestControllers.current.delete(result.id);
      setIngestingIds((current) => {
        const updated = new Set(current);
        updated.delete(result.id);
        return updated;
      });
      setIngestStage((current) => {
        const updated = { ...current };
        delete updated[result.id];
        return updated;
      });
    }
  }

  function cancelArxivIngest(resultId: string) {
    ingestControllers.current.get(resultId)?.abort();
  }

  function onDrop(event: DragEvent<HTMLButtonElement>) {
    event.preventDefault();
    void handleFiles(Array.from(event.dataTransfer.files ?? []));
  }

  const convergenceEntries = Array.from(ingestingIds).map((id) => ({
    id,
    title: searchResults.find((result) => result.id === id)?.title ?? "your paper",
    // The real backend stage, resolved to display text here (same helper
    // the search-results ingest badge already uses) - ConvergenceRitual
    // previously received a raw `stage` field and never actually used it,
    // showing only its own cosmetic word rotation with no real signal.
    stageLabel: ingestStageLabel(ingestStage[id]),
  }));

  return (
    <main className="assistant-app">
      <ConvergenceRitual entries={convergenceEntries} active={ingestingIds.size > 0} onSkip={() => {}}/>
      <GraphExplorer sessionId={currentSession?.id ?? null} sessionName={currentSession?.name} active={graphExplorerOpen} onClose={() => setGraphExplorerOpen(false)} onAskInChat={askFromGraphExplorer} focusNodeId={graphFocusNodeId} papers={papers}/>
      <PaperMap sessionId={currentSession?.id ?? null} active={paperMapOpen} onClose={() => setPaperMapOpen(false)} papers={papers}/>
      <Tour steps={ATLAS_TOUR_STEPS} active={tourOpen} onClose={() => { setTourOpen(false); setTourSeen(true); }}/>
      <header className="app-header">
        <div className="brand"><span><Icon name="atlas" size={21}/></span><strong>Atlas</strong></div>
        <div className={`mode-label ${papers.length ? "paper-mode" : ""}`}>
          {papers.length > 0 ? (
            <div className="paper-chip-row">
              {papers.map((p) => <span key={p.id} className="paper-chip"><Icon name="paper" size={12}/><strong>{p.title}</strong><button onClick={() => removePaperFromSet(p.id)} aria-label={`Remove ${p.title}`}><Icon name="close" size={11}/></button></span>)}
            </div>
          ) : <><span className="online-dot"/><strong>General chat</strong><small>Atlas</small></>}
        </div>
        <div className="header-actions">
          <div className="session-switcher" ref={sessionSwitcherRef}>
            <button className="session-switcher-toggle" onClick={() => setSessionMenuOpen((open) => !open)}>
              <span>{currentSession?.name ?? "Session"}</span>
            </button>
            {sessionMenuOpen && <div className="session-menu">
              <button className="session-menu-new" onClick={openCreateSession}><Icon name="plus" size={13}/>New session</button>
              {sessions.map((s) => <div key={s.id} className="session-menu-row">
                <button
                  className={`session-menu-item${s.id === currentSession?.id ? " active" : ""}`}
                  onClick={() => void switchToSession(s)}
                >
                  <strong>{s.name}</strong>
                  <small>{new Date(s.created_at).toLocaleDateString()}</small>
                </button>
                <button className="session-menu-rename" onClick={() => void renameSessionPrompt(s)} aria-label={`Rename ${s.name}`}><Icon name="rename" size={13}/></button>
                <button className="session-menu-delete" onClick={() => void deleteSessionPrompt(s)} aria-label={`Delete ${s.name}`}><Icon name="close" size={13}/></button>
              </div>)}
            </div>}
          </div>
          <button
            className="feynman-check-toggle"
            onClick={openFeynmanCheck}
            disabled={papers.length === 0}
            title={papers.length === 0 ? "Add a paper to test your understanding of it" : "Explain a paper's idea back in your own words, graded against its real evidence"}
            aria-label="Test yourself"
          ><Icon name="quiz" size={17}/></button>
          <button
            className="graph-explorer-toggle"
            onClick={() => setGraphExplorerOpen(true)}
            disabled={papers.length === 0}
            title={papers.length === 0 ? "Add a paper to explore its graph" : "Explore this session's knowledge graph"}
            aria-label="Open graph explorer"
          ><Icon name="graph" size={17}/></button>
          <button className="paper-map-toggle" onClick={() => setPaperMapOpen(true)} disabled={papers.length < 2} title={papers.length < 2 ? "Add at least two papers to map connections" : "Map connections between this session's papers"} aria-label="Map paper connections"><Icon name="graph" size={15}/><span>Map papers <em>BETA</em></span></button>
          <button className="tour-help-button" onClick={() => setTourOpen(true)} title="Replay the walkthrough" aria-label="Replay the walkthrough"><Icon name="help" size={17}/></button>
          <button className="add-paper-button" onClick={() => openAddPaper()}><Icon name="plus" size={17}/>{papers.length ? "Add another" : "Add paper"}</button>
        </div>
      </header>

      <section className="conversation-scroll">
        <div className="conversation-content">
          {messages.length === 0 ? (
            <div className="welcome">
              <span className="welcome-icon"><Icon name="atlas" size={27}/></span>
              <h1>How can I help?</h1>
              <p>Ask Atlas anything, or take the tour to see how paper-grounded research works.</p>
              <div className="welcome-actions">
                <button type="button" className="welcome-tour-button" onClick={() => setTourOpen(true)}><Icon name="help" size={15}/>{tourSeen ? "Retake the tour" : "Take the tour"}</button>
              </div>
            </div>
          ) : (
            <div className="message-list">
              {messages.map((message) => {
                const visibleGaps = message.gaps?.filter((g) => !dismissedGapKeys.has(gapKey(g)));
                const hasClarificationContent = Boolean(message.clarification) || Boolean(message.clarifications?.length);
                const hasVisibleGapContent = Boolean(visibleGaps?.length);
                // A combined findings card can lose all its gaps to
                // dismissal while its clarifications are still open - only
                // hide the whole message once genuinely nothing is left,
                // not just because the gaps half emptied out.
                if (message.gaps && !hasVisibleGapContent && !hasClarificationContent) return null;
                const isFindingsMessage = hasClarificationContent || hasVisibleGapContent;
                const findingsExpanded = expandedFindings.has(message.id);
                const findingsSummaryParts: string[] = [];
                if (message.clarification) findingsSummaryParts.push("1 possible duplicate");
                else if (message.clarifications?.length) findingsSummaryParts.push(`${message.clarifications.length} possible duplicates`);
                if (hasVisibleGapContent) findingsSummaryParts.push(`${visibleGaps!.length} thing${visibleGaps!.length === 1 ? "" : "s"} worth exploring`);
                const dedupedCitations = message.citations ? dedupeCitations(message.citations) : undefined;
                const feedbackNodeId = dedupedCitations?.[0]?.source_kind === "graph" ? dedupedCitations[0].node_ids?.[0] : undefined;
                const isGuideMessage = Boolean(message.guide || message.guideLoading);

                return <article key={message.id} className={`message ${message.role} ${message.notice ? "notice" : ""} ${isGuideMessage ? "guide-message" : ""}`}>
                  {message.role === "assistant" && <span className="assistant-avatar"><Icon name={message.notice ? "check" : "atlas"} size={16}/></span>}
                  <div className="message-body">
                    {message.role === "assistant" && <small>{isGuideMessage ? "Atlas guide" : "Atlas"}</small>}
                    {message.text && <p>{message.text}</p>}
                    {message.guideLoading && <div className="guide-building"><span/><div><strong>Building your guided reading</strong><small>Finding the paper's structure, simplifying each section, and drawing useful visual explanations…</small></div></div>}
                    {message.guide && <GuidedReading guide={message.guide} paperId={message.paperId} sessionId={currentSession?.id} onViewNodeInGraph={viewNodeInGraph}/>}
                    {message.guide && message.paperId && <button type="button" className="deep-dive-launch" onClick={() => window.location.assign(`/deep-dive/${encodeURIComponent(message.paperId!)}?session_id=${encodeURIComponent(currentSession?.id ?? "local")}`)}><strong>Open deeper dive <em>BETA</em></strong><small>Explore the full paper text with a section tutor.</small></button>}
                    {message.guideError && <span className="guide-error">{message.guideError}</span>}
                    {message.feynmanPickPaper && <div className="candidate-list">{papers.map((p) => <button key={p.id} onClick={() => setMessages((current) => current.map((m) => m.id === message.id ? { ...m, feynmanPickPaper: false, feynmanPaperId: p.id } : m))}><strong>{p.title}</strong></button>)}</div>}
                    {message.feynmanPaperId && currentSession?.id && <FeynmanCheck paperId={message.feynmanPaperId} sessionId={currentSession.id} onViewNodeInGraph={viewNodeInGraph}/>}
                    {message.retrievalMode === "vector" && <span className="not-graph-note"><Icon name="globe" size={11}/>From the paper's text directly, not yet verified against the knowledge graph</span>}
                    {message.confidence === "low" && <span className="confidence-note" title="The best matching source for this answer scored below Atlas's confidence threshold, so it's worth double-checking against the sources below.">Low-confidence match: check the sources below.</span>}
                    {message.candidates && message.candidates.length > 0 && <div className="candidate-list">{message.candidates.map((candidate) => <button key={candidate.node_id} onClick={() => selectCandidate(candidate, message.clarificationQuestionId)}><strong>{candidate.name}</strong><small>{candidate.type}</small>{candidate.description && <p>{candidate.description}</p>}</button>)}</div>}
                    {dedupedCitations && dedupedCitations.length > 0 && <div className="citations">{dedupedCitations.map((citation, index) => {
                      const key = `${message.id}-${index}`;
                      const graphNodeId = citation.source_kind === "graph" ? citation.node_ids?.[0] : undefined;
                      return <div key={key}>
                        <button onClick={() => setExpandedCitation(expandedCitation === key ? null : key)}><Icon name="quote" size={13}/>{citation.section ?? "Source"}{citation.page_start != null && ` · p. ${citation.page_start}`}</button>
                        {graphNodeId && <button className="citation-view-graph" onClick={() => viewNodeInGraph(graphNodeId)} title="View this node in the graph"><Icon name="graph" size={12}/>View in graph</button>}
                        {expandedCitation === key && <blockquote>“{citation.text}”</blockquote>}
                      </div>;
                    })}</div>}
                    {feedbackNodeId && <div className="feedback-row">
                      {message.feedbackGiven ? <small>Thanks, I&rsquo;ll factor that in.</small> : <>
                        <button onClick={() => submitQueryFeedback(message.id, feedbackNodeId, true)} aria-label="Helpful"><Icon name="thumbUp" size={13}/></button>
                        <button onClick={() => submitQueryFeedback(message.id, feedbackNodeId, false)} aria-label="Not helpful"><Icon name="thumbDown" size={13}/></button>
                      </>}
                    </div>}
                    {isFindingsMessage && <button
                      type="button"
                      className={`findings-toggle${findingsExpanded ? " expanded" : ""}`}
                      onClick={() => setExpandedFindings((current) => {
                        const next = new Set(current);
                        if (next.has(message.id)) next.delete(message.id); else next.add(message.id);
                        return next;
                      })}
                    >
                      <span>I also found a few things while reading: {findingsSummaryParts.join(", ")}</span>
                      <Icon name="plus" size={13}/>
                    </button>}
                    {findingsExpanded && <>
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
                            <strong>Possible connection to verify: {candidate.node_a_name} ↔ {candidate.node_b_name}</strong>
                            {candidate.explanation && <span>{candidate.explanation}</span>}
                          </button>
                          <button className="gap-dismiss" onClick={() => dismissGap(candidate)} aria-label="Not interesting"><Icon name="close" size={12}/></button>
                        </div>)}
                      </div>}
                    </>}
                  </div>
                </article>;
              })}
              {loading && <article className="message assistant"><span className="assistant-avatar"><Icon name="atlas" size={16}/></span><div className="message-body"><small>Atlas</small><div className="typing"><i/><i/><i/></div></div></article>}
            </div>
          )}
          <div ref={chatEnd}/>
        </div>
      </section>

      <footer className="composer-area">
        {papers.length > 0 && <div className="paper-context-chip"><Icon name="paper" size={14}/><span>Using <strong>{papers.length === 1 ? papers[0].title : `${papers.length} papers`}</strong></span></div>}
        <form className="composer" onSubmit={(event) => ask(event)}>
          <button type="button" className="composer-add" onClick={() => openAddPaper()} aria-label="Add a paper"><Icon name="plus" size={19}/></button>
          <textarea ref={composerInput} value={query} maxLength={8000} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void ask(); } }} rows={1} placeholder={papers.length ? "Ask anything about these papers…" : "Message Atlas…"}/>
          <button type="submit" className="send-button" disabled={!query.trim() || loading} aria-label="Send"><Icon name="send" size={18}/></button>
        </form>
        <small className="composer-hint">Atlas can make mistakes. Paper answers include sources when available.</small>
      </footer>

      {addOpen && <div className="add-modal" role="dialog" aria-modal="true" aria-label="Add a research paper">
        <div className="modal-scrim" onClick={() => setAddOpen(false)} aria-hidden="true"/>
        <section className={`modal-card ${buildingGraphQueue.length > 0 ? "modal-card-fullscreen" : ""}`}>
          <header><div><span><Icon name="paper" size={18}/></span><div><strong>Add a research paper</strong><small>Give Atlas a paper to read with you</small></div></div><button onClick={() => setAddOpen(false)} aria-label="Close"><Icon name="close" size={19}/></button></header>

          {buildingGraphQueue.length > 0 ? (
            <GraphBuildAnimation
              key={buildingGraphQueue[0].id}
              newNodes={buildingGraphQueue[0].new_nodes}
              newEdges={buildingGraphQueue[0].new_edges}
              onComplete={() => {
                const keepModalOpen = buildingGraphQueue.length > 1 || uploading;
                addPaper(buildingGraphQueue[0], !keepModalOpen);
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
              <span><Icon name="upload" size={25}/></span><strong>{uploading ? "Uploading and reading papers one at a time…" : "Choose PDFs or drag them here"}</strong><small>Each PDF follows Atlas’s normal reading and graph-building process.</small>
            </button>
            {uploading && <button className="cancel-ingest" onClick={cancelUpload}>Cancel</button>}
            <input ref={fileInput} className="hidden-input" type="file" accept="application/pdf,.pdf" multiple onChange={(event: ChangeEvent<HTMLInputElement>) => void handleFiles(Array.from(event.target.files ?? []))}/>
            {uploadError && <p className="form-error">{uploadError}</p>}
          </div>}

          {addMode === "search" && <div className="search-panel">
            <button className="back-button" onClick={() => setAddMode("choose")}>← Back</button>
            <form onSubmit={runSearch}><Icon name="search" size={18}/><input autoFocus value={searchQuery} onChange={(event) => setSearchQuery(event.target.value)} placeholder="Search by title, author, or topic…"/><button disabled={searchQuery.trim().length < 2 || searching}>{searching ? "Searching…" : "Search"}</button></form>
            {searchError && <p className="form-error">{searchError}</p>}
            <div className="search-results">{searchResults.map((result) => {
              const isIngesting = ingestingIds.has(result.id);
              return <article key={result.id}><div><span>arXiv</span><small>{result.published}</small></div><h3>{result.title}</h3><p>{result.authors}</p>
                {isIngesting ? <div className="ingest-progress"><span>{ingestStageLabel(ingestStage[result.id])}</span><button className="cancel-ingest" onClick={() => cancelArxivIngest(result.id)}>Cancel</button></div>
                  : <button onClick={() => void addArxivPaper(result)}>Add to chat</button>}
              </article>;
            })}</div>
          </div>}
          </>}
        </section>
      </div>}
      {createSessionOpen && <div className="add-modal" role="dialog" aria-modal="true" aria-label="New session">
        <div className="modal-scrim" onClick={() => setCreateSessionOpen(false)} aria-hidden="true"/>
        <section className="modal-card">
          <header><div><span><Icon name="graph" size={18}/></span><div><strong>New session</strong><small>Each session keeps its own papers and graph</small></div></div><button onClick={() => setCreateSessionOpen(false)} aria-label="Close"><Icon name="close" size={19}/></button></header>
          <form className="create-session-form" onSubmit={submitCreateSession}>
            <label>Name<input autoFocus value={newSessionName} onChange={(event) => setNewSessionName(event.target.value)} placeholder="e.g. Attention mechanisms" maxLength={200}/></label>
            <label>What are you working on? <small>(optional - helps Atlas focus suggestions and answers)</small><textarea value={newSessionGoal} onChange={(event) => setNewSessionGoal(event.target.value)} placeholder="e.g. Comparing efficient-attention methods for my thesis" rows={2} maxLength={300}/></label>
            {createSessionError && <p className="form-error">{createSessionError}</p>}
            <div className="create-session-actions">
              <button type="button" className="back-button" onClick={() => setCreateSessionOpen(false)}>Cancel</button>
              <button type="submit" disabled={!newSessionName.trim() || creatingSession}>{creatingSession ? "Creating…" : "Create session"}</button>
            </div>
          </form>
        </section>
      </div>}
    </main>
  );
}
