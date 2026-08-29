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
  getSessionMessages,
  ingestArxivPaper,
  listGaps,
  listPapersForSession,
  listSessions,
  recordGapFeedback,
  recordQueryFeedback,
  renameSession,
  saveSessionMessages,
  searchPapers,
  uploadPaper,
} from "@/lib/api";
import type { ChatHistoryItem, PaperContext, PaperIngestResult, PaperSearchResult, SessionMetadata } from "@/lib/api";
import type { Citation, FeynmanCheckResult, FeynmanPrompt, GapCandidate, PaperGuide, PendingQuestion, QueryResponse } from "@/lib/types";
import ConvergenceRitual from "./ConvergenceRitual";
import GraphExplorer from "./GraphExplorer";
import Tour, { TourStep } from "./Tour";
import PaperMap from "./PaperMap";
import { useAuth } from "./AuthProvider";

type IconName = "atlas" | "plus" | "send" | "paper" | "search" | "upload" | "close" | "quote" | "check" | "globe" | "thumbUp" | "thumbDown" | "rename" | "graph" | "help" | "quiz" | "book" | "menu" | "papersLink";
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

// Instant-paint cache for the active session, keyed by id - never the
// source of truth (the server-side Firestore load below always still
// runs and overwrites this the moment it resolves), purely so a refresh
// shows the conversation immediately instead of a blank/loading state
// while that real fetch is in flight. Confirmed live as the actual ask:
// "it should instantly load into the session," not just a faster
// backend - the backend fix alone still left a real network round trip
// (even a fast one) between refresh and content appearing.
interface SessionCache {
  session: SessionMetadata;
  messages: Message[];
  papers: PaperContext[];
}

// Every key below is scoped by uid, not just session id - localStorage is
// shared across whoever signs in on this browser, and unlike the
// server-side session load (already safe: get_current_user + owner_uid
// filtering means a foreign session id server-side just silently finds
// nothing), this cache reads real cached content straight from local
// storage with no server round trip at all. Confirmed live as a real gap
// caught before shipping: without the uid scope, User A signing out and
// User B signing in on the same device would have briefly hydrated the
// page with A's actual cached conversation before B's own real fetch
// caught up and corrected it.
function sessionIdStorageKey(uid: string) {
  return `atlas-session-id-${uid}`;
}

function sessionCacheKey(uid: string, sessionId: string) {
  return `atlas-session-cache-${uid}-${sessionId}`;
}

function readSessionCache(uid: string, sessionId: string): SessionCache | null {
  try {
    const raw = window.localStorage.getItem(sessionCacheKey(uid, sessionId));
    return raw ? (JSON.parse(raw) as SessionCache) : null;
  } catch {
    return null;
  }
}

function writeSessionCache(uid: string, cache: SessionCache) {
  try {
    window.localStorage.setItem(sessionCacheKey(uid, cache.session.id), JSON.stringify(cache));
  } catch {
    // Storage full/unavailable/private-browsing - this cache is purely an
    // optimization, so failing silently here just means the next refresh
    // falls back to the normal network-loaded state, not broken behavior.
  }
}

// Kept close in length to the original flat "Reading…" label on purpose -
// this feeds ConvergenceRitual's stageLabel suffix (see convergenceEntries
// in the main component), which sits next to its own cosmetic word
// rotation; a noticeably longer label crowds that line.
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

// A bare heading number ("1", "3.") with no real title reads as a broken
// label, not a source - confirmed live: the extraction prompt never told
// Gemini what source_section should contain, so it sometimes returned
// just the leading digit off a numbered heading instead of the full
// title. Fixed at the prompt itself (agent/gemini_extractor.py) for new
// extractions, but a defensive display-level fallback still matters for
// already-stored edges and for the rare case a model ignores the prompt.
function citationSectionLabel(section: string | null | undefined): string {
  const trimmed = (section ?? "").trim();
  return /^\d+(\.\d+)*\.?$/.test(trimmed) ? "" : trimmed;
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
  // Two documents linked by a connector, distinct from the abstract
  // node-cluster `graph` glyph on purpose - paper-map-toggle sat right
  // next to graph-explorer-toggle using the same icon, confirmed live as
  // a real "what's the difference between these two" moment.
  papersLink: <><rect x="3" y="4" width="8" height="11" rx="1.5"/><rect x="13" y="9" width="8" height="11" rx="1.5"/><path d="M11 9.5L13 14.5"/></>,
  help: <><circle cx="12" cy="12" r="9"/><path d="M9.1 9a3 3 0 1 1 4.6 2.6c-.9.5-1.7 1.1-1.7 2.4"/><path d="M12 17.5h.01"/></>,
  quiz: <><path d="M9 21h6M10 18h4M8.5 12.5A5 5 0 1 1 15.5 12.5c-.7 1-1.5 1.6-1.5 3H10c0-1.4-.8-2-1.5-3Z"/></>,
  book: <><path d="M4 19.5V5a2 2 0 0 1 2-2h13v15H6a2 2 0 0 0 0 4h13"/></>,
  menu: <path d="M4 7h16M4 12h16M4 17h16"/>,
};

function Icon({ name, size = 19 }: { name: IconName; size?: number }) {
  return <svg aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">{icons[name]}</svg>;
}

const ATLAS_TOUR_STEPS: TourStep[] = [
  {
    target: ".welcome-icon",
    title: "Welcome to Atlas",
    body: "A quick nine stop tour. Cancel anytime.",
    placement: "top",
    noHighlight: true,
  },
  {
    target: ".add-paper-button",
    title: "Add a paper",
    body: "Upload a PDF or search arXiv right here. Atlas reads the paper, builds a real knowledge graph from what's inside, and walks you through it section by section.",
    placement: "bottom",
  },
  {
    target: ".composer",
    title: "Ask anything",
    body: "Ask a question here anytime. Once you've added a paper, answers cite the exact section and page they came from.",
    placement: "top",
  },
  {
    target: ".feynman-check-toggle",
    title: "Test yourself",
    body: "Once you've added a paper, explain a concept from it in your own words here. Atlas grades your explanation against the paper's real evidence.",
    placement: "bottom",
  },
  {
    target: ".graph-explorer-toggle",
    title: "Explore the graph",
    body: "Once you've added a paper, open this to see everything it extracted as a real, clickable graph, not just a summary.",
    placement: "bottom",
  },
  {
    target: ".paper-map-toggle",
    title: "Map papers",
    body: "A different view: once you've added at least two papers, this shows how they connect to each other, not everything inside each one.",
    placement: "bottom",
  },
  {
    target: ".app-sidebar-toggle",
    title: "Sessions",
    body: "Each session is its own research thread with its own papers and graph. Switch between sessions or start a new one here.",
    placement: "bottom",
  },
  {
    target: ".user-menu-toggle",
    title: "Your account",
    body: "Switch between light and dark mode or sign out here.",
    placement: "bottom",
  },
  {
    target: ".tour-help-button",
    title: "Need a refresher?",
    body: "Come back here anytime to replay this tour.",
    placement: "bottom",
  },
];

const FLOW_REVEAL_INTERVAL_MS = 200;
const MAX_SESSION_PAPERS = 5;

function FlowDiagram({ guide }: { guide: NonNullable<PaperGuide["sections"][number]["diagram"]> }) {
  // Builds itself in node-by-node on a timer instead of rendering every
  // node at once - remounts (and so replays) each time its parent section becomes the
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
  const { user, signOut } = useAuth();
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  // Reflects whatever's actually showing right now (an explicit choice, or
  // the system default when none was ever made) - not just "did the user
  // pick something." layout.tsx's inline blocking script already applied
  // any stored explicit choice before this component ever mounts (avoids
  // a flash of the wrong theme); this effect only needs to sync React's
  // own state to match what's already on the page, plus resolve the
  // no-stored-choice case against the system preference for the toggle's
  // own icon/label.
  const [theme, setTheme] = useState<"light" | "dark">("light");
  useEffect(() => {
    const stored = window.localStorage.getItem("atlas-theme");
    const resolved =
      stored === "dark" || stored === "light"
        ? stored
        : window.matchMedia("(prefers-color-scheme: dark)").matches
          ? "dark"
          : "light";
    setTheme(resolved);
  }, []);
  function toggleTheme() {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    document.documentElement.setAttribute("data-theme", next);
    window.localStorage.setItem("atlas-theme", next);
  }
  // Desktop: collapses the persistent sidebar to reclaim width, persisted
  // like the theme choice so it stays collapsed across reloads. Mobile:
  // the CSS for a collapsed sidebar only exists inside the >680px media
  // query, so this state has no visible effect there - sessionMenuOpen
  // (below, unrelated pre-existing state) is what the same toggle button
  // drives for the small-screen off-canvas overlay instead. One button,
  // one handler, two different states, because only one of the two is
  // ever visually active for a given viewport.
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  useEffect(() => {
    setSidebarCollapsed(window.localStorage.getItem("atlas-sidebar-collapsed") === "1");
  }, []);
  function toggleSidebar() {
    setSidebarCollapsed((current) => {
      const next = !current;
      window.localStorage.setItem("atlas-sidebar-collapsed", next ? "1" : "0");
      return next;
    });
    setSessionMenuOpen((open) => !open);
  }
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
  const [uploadingFileNames, setUploadingFileNames] = useState<string[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<PaperSearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  // Papers picked but not yet sent for extraction - "Start breaking down"
  // is what actually kicks off ingestion for the whole batch at once, so
  // the full-screen ConvergenceRitual reflects everything the user asked
  // for in one pass instead of starting piecemeal per click.
  const [stagedResults, setStagedResults] = useState<PaperSearchResult[]>([]);
  const [stagedFiles, setStagedFiles] = useState<File[]>([]);
  // Snapshot of staged titles at the moment ingestion actually starts -
  // searchResults/stagedResults are cleared right after so the modal
  // resets cleanly, but ConvergenceRitual still needs a title per
  // in-flight id for as long as that id stays in ingestingIds.
  const [ingestTitles, setIngestTitles] = useState<Record<string, string>>({});
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
  const uploadControllers = useRef<Map<string, AbortController>>(new Map());
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
  // Conversation history is persisted server-side (Firestore) so it
  // survives a different browser/device, not just this one's localStorage
  // - initSession/switchToSession below are what actually load a
  // session's messages.
  const saveMessagesTimer = useRef<number | null>(null);
  useEffect(() => {
    // Guard against the pre-init render (sessionIdRef not set yet) saving
    // an empty array under a bogus id.
    if (!sessionIdRef.current) return;
    const sessionId = sessionIdRef.current;
    if (saveMessagesTimer.current) window.clearTimeout(saveMessagesTimer.current);
    // Debounced, not synchronous like the localStorage write this
    // replaced - this is now a real network round trip, and a fast
    // back-and-forth shouldn't fire one PUT per message.
    saveMessagesTimer.current = window.setTimeout(() => {
      void saveSessionMessages(sessionId, messages as unknown as Record<string, unknown>[])
        .then((result) => {
          // sessionId is captured at the time this save was queued, not
          // re-read from the ref here - guards against a save queued for
          // session A landing after the user has already switched to
          // session B inside this debounce window. Only the compacted
          // case needs to reach back into state; the normal case already
          // matches what was just sent.
          if (result.compacted && sessionIdRef.current === sessionId) {
            setMessages(result.messages as unknown as Message[]);
          }
        })
        .catch(() => {
          // Persistence failure shouldn't be disruptive - the in-memory
          // conversation still works for the rest of this tab session;
          // the next debounced save (or next message) simply retries.
        });
    }, 1000);
    return () => {
      if (saveMessagesTimer.current) window.clearTimeout(saveMessagesTimer.current);
    };
  }, [messages]);
  // Keeps the instant-paint cache current through a live session too, not
  // just at load time - switchToSession already writes it once the
  // server confirms a fresh load, but without this a long chat session's
  // cache would stay frozen at whatever it looked like the moment you
  // switched in, so a refresh after chatting a while would flash that
  // stale snapshot before the real fetch corrected it.
  useEffect(() => {
    if (!currentSession || sessionIdRef.current !== currentSession.id) return;
    writeSessionCache(user.uid, { session: currentSession, messages, papers });
  }, [messages, papers, currentSession]);
  const initRan = useRef(false);
  useEffect(() => {
    // React 18 StrictMode dev-mode double-invokes effects with no cleanup
    // - without this guard, both invocations raced past "no session yet"
    // and each created its own, leaving two "Untitled session" entries
    // from a single page load.
    if (initRan.current) return;
    initRan.current = true;
    // Paint instantly from the last-known snapshot of this exact session,
    // if there is one - initSession() below still runs immediately after
    // and reconciles with the real server state (a stale/deleted session,
    // a message sent from another device, etc always wins), this just
    // means there's something real on screen the moment the page loads
    // instead of a blank state for however long that fetch takes.
    const savedId = window.localStorage.getItem(sessionIdStorageKey(user.uid));
    const cached = savedId ? readSessionCache(user.uid, savedId) : null;
    if (cached) {
      sessionIdRef.current = cached.session.id;
      setCurrentSession(cached.session);
      setMessages(cached.messages);
      setPapers(cached.papers);
    }
    void initSession();
    setTourSeen(window.localStorage.getItem("atlas-tour-seen") === "1");
  }, []);
  const sessionSwitcherRef = useRef<HTMLDivElement | null>(null);
  const userMenuRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    // graphExplorerOpen was missing here - confirmed live that Escape did
    // nothing while Graph Explorer was open, inconsistent with every other
    // modal in the app (the Add Paper dialog, the session switcher) which
    // this same handler already closes.
    if (!sessionMenuOpen && !addOpen && !graphExplorerOpen && !createSessionOpen && !userMenuOpen) return;
    function onKeyDown(event: KeyboardEvent) {
      if (event.key !== "Escape") return;
      setSessionMenuOpen(false);
      setAddOpen(false);
      setGraphExplorerOpen(false);
      setCreateSessionOpen(false);
      setUserMenuOpen(false);
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [sessionMenuOpen, addOpen, graphExplorerOpen, createSessionOpen, userMenuOpen]);
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
  useEffect(() => {
    if (!userMenuOpen) return;
    function onMouseDown(event: MouseEvent) {
      if (userMenuRef.current?.contains(event.target as Node)) return;
      setUserMenuOpen(false);
    }
    document.addEventListener("mousedown", onMouseDown);
    return () => document.removeEventListener("mousedown", onMouseDown);
  }, [userMenuOpen]);

  async function initSession() {
    let resolved: SessionMetadata | null = null;
    try {
      const list = await listSessions();
      setSessions(list);
      const savedId = window.localStorage.getItem(sessionIdStorageKey(user.uid));
      resolved = list.find((s) => s.id === savedId) ?? list[0] ?? null;
      if (!resolved) {
        resolved = await createSession("Untitled session");
        setSessions((current) => [resolved!, ...current]);
      }
    } catch {
      // Backend unreachable at startup - fall back to a purely local
      // session id so the app still works; ingest calls just carry it.
      const fallbackId = window.localStorage.getItem(sessionIdStorageKey(user.uid)) ?? crypto.randomUUID();
      resolved = { id: fallbackId, name: "Untitled session", created_at: new Date().toISOString() };
    }
    await switchToSession(resolved);
  }

  async function switchToSession(session: SessionMetadata) {
    // Captured before sessionIdRef is overwritten below - true when this
    // call is reconciling the session already on screen (the mount-time
    // cache-hydrate case) rather than a genuine switch to a different
    // one. Used below to skip the papers-flash a real switch needs (clear
    // the old session's papers so they're never shown mislabeled as the
    // new session's) but a same-session reconcile does not - there's
    // nothing stale to clear, only real data to quietly confirm or update.
    const isReconcilingSameSession = sessionIdRef.current === session.id;
    // Flush any pending debounced save for the session being left, right
    // now, synchronously with the old session id and its current messages
    // - confirmed live via code review that without this, a message sent
    // right before a fast switch gets silently lost: the save effect's
    // own cleanup (see saveMessagesTimer below) clears the pending timer
    // the moment `messages` next changes (which switching to a new
    // session's restored messages always does), and the timer callback
    // that would have sent it never runs.
    if (saveMessagesTimer.current && sessionIdRef.current && sessionIdRef.current !== session.id) {
      window.clearTimeout(saveMessagesTimer.current);
      saveMessagesTimer.current = null;
      void saveSessionMessages(sessionIdRef.current, messages as unknown as Record<string, unknown>[]).catch(() => {});
    }
    sessionIdRef.current = session.id;
    window.localStorage.setItem(sessionIdStorageKey(user.uid), session.id);
    setCurrentSession(session);
    setSessionMenuOpen(false);
    setBuildingGraphQueue([]);
    setDismissedGapKeys(new Set());

    // Fired together, not one after the other - these two calls don't
    // depend on each other's result, but awaiting messages before even
    // starting the papers fetch made every switch pay for both round
    // trips added together instead of just the slower one. Confirmed
    // live as a real, avoidable chunk of the "switching feels slow"
    // complaint.
    const messagesPromise = getSessionMessages(session.id);
    const papersPromise = listPapersForSession(session.id);

    let restoredMessages: Message[] = [];
    const freshClarificationIds = new Set<string>();
    const freshGapKeys = new Set<string>();
    try {
      const result = await messagesPromise;
      const raw = result.messages as unknown as Message[];
      // Seed the dedup trackers directly from restored history (not from
      // messages state, which won't have committed yet) so checkGuidance
      // below doesn't re-post a clarification/gap card that's already
      // sitting in the restored conversation.
      raw.forEach((message) => {
        if (message.clarification) freshClarificationIds.add(message.clarification.id);
        message.clarifications?.forEach((q) => freshClarificationIds.add(q.id));
        message.gaps?.forEach((candidate) => freshGapKeys.add(gapKey(candidate)));
      });
      // clarification/clarifications are stripped from what actually
      // renders (not just tracked above) so an old, already-resolved
      // question doesn't come back looking like it's still open on
      // reload - gaps are left as-is, only clarification cards had this
      // problem.
      restoredMessages = raw.map(({ clarification: _clarification, clarifications: _clarifications, ...message }) => message);
    } catch {
      // Backend unreachable - degrade to an empty conversation for this
      // switch rather than blocking session switching entirely.
    }
    shownClarificationIds.current = freshClarificationIds;
    shownGapKeys.current = freshGapKeys;
    setMessages(restoredMessages);
    // A real switch clears papers first so the previous session's never
    // sit on screen mislabeled as the new session's while its own fetch
    // is in flight. Reconciling the session already on screen has nothing
    // stale to clear this way - skipping it means the cache-hydrated
    // papers stay visible the whole time instead of flashing empty and
    // back the instant this real fetch confirms them.
    if (!isReconcilingSameSession) setPapers([]);

    let resolvedPapers: PaperContext[] = [];
    try {
      resolvedPapers = await papersPromise;
      setPapers(resolvedPapers);
    } catch (error) {
      setMessages((current) => [...current, {
        id: crypto.randomUUID(),
        role: "assistant",
        text: error instanceof Error ? error.message : "Could not load this session's papers.",
        notice: true,
      }]);
    }
    // Refreshes the instant-paint cache with what the server just
    // confirmed - keeps the next refresh's snapshot current instead of
    // silently drifting further out of date every session this ran for.
    writeSessionCache(user.uid, { session, messages: restoredMessages, papers: resolvedPapers });

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

  function renameSessionPrompt(session: SessionMetadata) {
    const name = window.prompt("Rename session:", session.name)?.trim();
    if (!name || name === session.name) return;
    // Optimistic - another "little thing" per the same reasoning as
    // deleteSessionPrompt: a rename is a single Firestore write, no
    // reason the sidebar should wait on it before showing the new name.
    // Reverted below on failure (unlike removePaperFromSet's silent
    // no-revert, a session sitting under the wrong name is easy to
    // notice and confusing to leave that way).
    setSessions((current) => current.map((s) => (s.id === session.id ? { ...s, name } : s)));
    setCurrentSession((current) => (current?.id === session.id ? { ...current, name } : current));
    renameSession(session.id, name)
      .then((updated) => {
        setSessions((current) => current.map((s) => (s.id === updated.id ? updated : s)));
        setCurrentSession((current) => (current?.id === updated.id ? updated : current));
      })
      .catch((error) => {
        setSessions((current) => current.map((s) => (s.id === session.id ? session : s)));
        setCurrentSession((current) => (current?.id === session.id ? session : current));
        setMessages((current) => [...current, {
          id: crypto.randomUUID(),
          role: "assistant",
          text: error instanceof Error ? error.message : "Could not rename the session.",
          notice: true,
        }]);
      });
  }

  async function deleteSessionPrompt(session: SessionMetadata) {
    if (!window.confirm(`Delete "${session.name}"? This removes its papers and can't be undone.`)) return;
    // Optimistic removal - confirmed live as the actual "takes a long time
    // to disappear from the sidebar" complaint: the row previously stayed
    // visible until delete_session's full backend cascade (paper detach
    // loop, graph pruning, two Firestore deletes) round-tripped, which is
    // real work, not a CSS delay. Restored below if the delete call itself
    // fails.
    const index = sessions.findIndex((s) => s.id === session.id);
    setSessions((current) => current.filter((s) => s.id !== session.id));
    // Switches away from the local, already-updated session list rather
    // than reusing initSession's own listSessions() re-fetch - that fetch
    // would race the in-flight delete below and could still see the
    // about-to-be-deleted session, re-selecting the very session being
    // removed.
    if (session.id === currentSession?.id) {
      const remaining = sessions.filter((s) => s.id !== session.id);
      if (remaining.length > 0) {
        void switchToSession(remaining[0]);
      } else {
        void (async () => {
          const created = await createSession("Untitled session");
          setSessions((current) => [created, ...current]);
          await switchToSession(created);
        })();
      }
    }
    try {
      await deleteSession(session.id);
      // No localStorage cleanup needed - the backend's own delete_session
      // cascade removes this session's stored messages.
    } catch (error) {
      setSessions((current) => {
        if (current.some((s) => s.id === session.id)) return current;
        const restored = [...current];
        restored.splice(Math.min(index, restored.length), 0, session);
        return restored;
      });
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

  function answerClarificationQuestion(messageId: string, question: PendingQuestion, optionId: string) {
    // Optimistic - same "little thing" reasoning as deleteSessionPrompt:
    // the card's own display only ever reads status/answer_option_id (see
    // the "Answered: <label>" line below), both of which are known the
    // instant the user clicks, with no reason to wait on the real
    // entity_merge graph mutation to show the choice was recorded.
    const applyQuestion = (q: PendingQuestion): PendingQuestion => (
      q.id === question.id ? { ...q, status: "answered", answer_option_id: optionId } : q
    );
    setMessages((current) => current.map((m) => {
      if (m.id !== messageId) return m;
      if (m.clarifications) return { ...m, clarifications: m.clarifications.map(applyQuestion) };
      return m.clarification ? { ...m, clarification: applyQuestion(m.clarification) } : m;
    }));
    void answerClarification(question.id, optionId)
      .then((answered) => {
        setMessages((current) => current.map((m) => {
          if (m.id !== messageId) return m;
          if (m.clarifications) {
            return { ...m, clarifications: m.clarifications.map((q) => (q.id === answered.id ? answered : q)) };
          }
          return { ...m, clarification: answered };
        }));
      })
      .catch((error) => {
        setMessages((current) => [...current, {
          id: crypto.randomUUID(),
          role: "assistant",
          text: error instanceof Error ? error.message : "Could not record that answer.",
          notice: true,
        }]);
      });
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

  // Validates and queues files into stagedFiles - does not upload. The
  // actual upload/extract only happens once "Start breaking down" fires
  // handleFiles below with the staged list.
  function stageFiles(files: File[]) {
    if (!files.length) return;
    const committed = papers.length + stagedResults.length + stagedFiles.length;
    const slots = MAX_SESSION_PAPERS - committed;
    if (slots <= 0) {
      setUploadError(`This session is limited to ${MAX_SESSION_PAPERS} papers.`);
      return;
    }
    const pdfFiles = files.filter((file) => file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf"));
    const validFiles = pdfFiles.slice(0, slots);
    const notices: string[] = [];
    if (files.length > slots) notices.push(`You can add up to ${MAX_SESSION_PAPERS} papers per session; only the first ${slots} file${slots === 1 ? "" : "s"} will be staged.`);
    if (validFiles.length !== files.length && files.length <= slots) notices.push("Only PDF files were included; other files were skipped.");
    if (validFiles.length === 0) {
      setUploadError(notices.join(" ") || "Please choose PDF files.");
      return;
    }
    setUploadError(notices.join(" "));
    setStagedFiles((current) => [...current, ...validFiles]);
  }

  async function handleFiles(files: File[]) {
    if (!files.length) return;
    // Same synchronous-ref guard as addArxivPaper, and for the same
    // reason: the drop-zone's disabled={uploading} only takes effect
    // after React commits the setUploading(true) below, so a rapid
    // double-drop (or a drop landing while the native file-picker's own
    // change event is still in flight) can start two uploads before that
    // commit lands.
    if (uploadControllers.current.size > 0) return;
    uploadBatchCancelled.current = false;
    setUploading(true);
    setUploadingFileNames(files.map((file) => file.name));
    const failures: string[] = [];
    try {
      // Upload all selected papers at once. Each request still runs through
      // the exact existing single-paper backend pipeline; concurrency here
      // only prevents the second paper waiting for the first upload to
      // finish. The graph reveals remain queued so walkthroughs appear as
      // separate messages, one below the other, in the conversation.
      await Promise.all(files.map(async (file) => {
        if (uploadBatchCancelled.current) return;
        const controller = new AbortController();
        const controllerKey = `${file.name}:${file.lastModified}:${file.size}`;
        uploadControllers.current.set(controllerKey, controller);
        try {
          const uploaded = await uploadPaper(file, controller.signal, sessionIdRef.current);
          setBuildingGraphQueue((current) => [...current, uploaded]);
        } catch (error) {
          if (!(error instanceof DOMException && error.name === "AbortError")) {
            failures.push(`${file.name}: ${error instanceof Error ? error.message : "upload failed"}`);
          }
        } finally {
          uploadControllers.current.delete(controllerKey);
        }
      }));
      if (failures.length > 0) {
        // uploadError alone isn't enough now that "Start breaking down"
        // closes the modal before this can resolve - uploadError only
        // renders inside the modal's upload-panel, so a failure landing
        // after close would otherwise vanish with nothing telling the
        // user it happened.
        const text = `Could not add ${failures.length} file${failures.length === 1 ? "" : "s"}: ${failures.join("; ")}`;
        setUploadError(text);
        setMessages((current) => [...current, { id: crypto.randomUUID(), role: "assistant", text, notice: true }]);
      }
    } finally {
      setUploading(false);
      setUploadingFileNames([]);
      uploadControllers.current.clear();
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  function cancelUpload() {
    uploadBatchCancelled.current = true;
    uploadControllers.current.forEach((controller) => controller.abort());
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
      setMessages((current) =>
        current.some((message) => message.id === guideMessageId)
          ? current.map((message) => message.id === guideMessageId ? { ...message, guideLoading: false, guide } : message)
          // The placeholder can be gone by the time this resolves - a
          // compaction (see save_session_messages) replaces the whole
          // array with a summary + a short tail while this request was
          // still in flight. Append the finished guide as a fresh
          // message instead of silently losing it to a .map() that finds
          // no match.
          : [...current, { id: guideMessageId, role: "assistant", text: "", guideLoading: false, guide, paperId: nextPaper.id }]
      );
    } catch (error) {
      const detail = error instanceof Error ? error.message : "Could not build the walkthrough.";
      const errorUpdate = { guideLoading: false, guideError: detail, text: "The paper was added and is ready for questions, but I couldn't generate its guided walkthrough." };
      setMessages((current) =>
        current.some((message) => message.id === guideMessageId)
          ? current.map((message) => message.id === guideMessageId ? { ...message, ...errorUpdate } : message)
          : [...current, { id: guideMessageId, role: "assistant" as const, paperId: nextPaper.id, ...errorUpdate }]
      );
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
    if (papers.length >= MAX_SESSION_PAPERS || ingestControllers.current.has(result.id)) return;
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
      void getPaperStatus(result.id.replace(/^arxiv:/, ""), user.uid).then((status) => {
        setIngestStage((current) => (current[result.id] ? { ...current, [result.id]: status } : current));
      });
    }, 1500);
    try {
      const ingested = await ingestArxivPaper(result, controller.signal, sessionIdRef.current);
      setBuildingGraphQueue((current) => [...current, ingested]);
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      // Same reasoning as handleFiles's failure branch - by the time this
      // rejects, "Start breaking down" has already closed the modal that
      // searchError renders inside of, so a chat notice is the only
      // surface guaranteed to still be visible.
      const text = error instanceof Error ? error.message : `Could not read "${result.title}".`;
      setSearchError(text);
      setMessages((current) => [...current, { id: crypto.randomUUID(), role: "assistant", text, notice: true }]);
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

  function onDrop(event: DragEvent<HTMLButtonElement>) {
    event.preventDefault();
    stageFiles(Array.from(event.dataTransfer.files ?? []));
  }

  // Fires everything staged in the modal at once, then closes it - the
  // full-screen ConvergenceRitual is what the user watches from here, not
  // the modal. Titles are snapshotted into ingestTitles first since
  // stagedResults is cleared immediately after (see its own comment).
  function startBreakdown() {
    const toIngest = stagedResults;
    const toUpload = stagedFiles;
    if (toIngest.length === 0 && toUpload.length === 0) return;
    setIngestTitles((current) => {
      const updated = { ...current };
      toIngest.forEach((result) => { updated[result.id] = result.title; });
      return updated;
    });
    setStagedResults([]);
    setStagedFiles([]);
    setAddOpen(false);
    setAddMode("choose");
    toIngest.forEach((result) => void addArxivPaper(result));
    if (toUpload.length) void handleFiles(toUpload);
  }

  // Only reveal finished walkthroughs into the chat once nothing from this
  // batch is still being added - otherwise papers would land in chat
  // staggered mid-animation instead of together once the whole "Start
  // breaking down" batch settles. ingestingIds covers arXiv adds in
  // flight, uploading covers a PDF batch (handleFiles's own Promise.all)
  // in flight - both need to be clear, not just one, since a batch can
  // mix both in the same "Start breaking down" click.
  const readyToShowWalkthroughs = buildingGraphQueue.length > 0 && ingestingIds.size === 0 && !uploading;

  // Drains buildingGraphQueue into real chat messages/graph state once the
  // whole batch is done. The modal is already closed by then (startBreakdown
  // closed it before ingestion even started), so the full-screen
  // ConvergenceRitual renders unobstructed instead of behind modal chrome.
  useEffect(() => {
    if (!readyToShowWalkthroughs) return;
    addPaper(buildingGraphQueue[0], false);
    setBuildingGraphQueue((current) => current.slice(1));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [readyToShowWalkthroughs, buildingGraphQueue]);

  const convergenceEntries = [
    ...Array.from(ingestingIds).map((id) => ({
      id,
      title: ingestTitles[id] ?? "your paper",
      // The real backend stage, resolved to display text here (same helper
      // the search-results ingest badge already uses) - ConvergenceRitual
      // previously received a raw `stage` field and never actually used it,
      // showing only its own cosmetic word rotation with no real signal.
      stageLabel: ingestStageLabel(ingestStage[id]),
    })),
    // Uploads have no per-file backend stage to poll (unlike arXiv ingest),
    // so these only ever show the cosmetic word rotation - still real
    // progress, just without a stageLabel suffix.
    ...uploadingFileNames.map((name, index) => ({ id: `upload-${index}-${name}`, title: name })),
  ];

  return (
    <div className={`app-shell${sidebarCollapsed ? " sidebar-collapsed" : ""}`}>
      <aside className={`app-sidebar${sessionMenuOpen ? " open" : ""}`} ref={sessionSwitcherRef}>
        <div className="app-sidebar-brand"><span><Icon name="atlas" size={19}/></span><strong>Atlas</strong></div>
        <button className="app-sidebar-new" onClick={() => { openCreateSession(); setSessionMenuOpen(false); }}><Icon name="plus" size={14}/>New session</button>
        <div className="app-sidebar-sessions">
          {sessions.map((s) => <div key={s.id} className="app-sidebar-session-row">
            <button
              className={`app-sidebar-session${s.id === currentSession?.id ? " active" : ""}`}
              onClick={() => { void switchToSession(s); setSessionMenuOpen(false); }}
            >
              <strong>{s.name}</strong>
              <small>{new Date(s.created_at).toLocaleDateString()}</small>
            </button>
            <button className="app-sidebar-session-rename" onClick={() => void renameSessionPrompt(s)} aria-label={`Rename ${s.name}`}><Icon name="rename" size={13}/></button>
            <button className="app-sidebar-session-delete" onClick={() => void deleteSessionPrompt(s)} aria-label={`Delete ${s.name}`}><Icon name="close" size={13}/></button>
          </div>)}
        </div>
      </aside>
      {sessionMenuOpen && <div className="app-sidebar-scrim" onClick={() => setSessionMenuOpen(false)} aria-hidden="true"/>}
    <main className="assistant-app">
      <ConvergenceRitual
        entries={convergenceEntries}
        active={ingestingIds.size > 0 || uploading}
        onSkip={() => {
          ingestControllers.current.forEach((controller) => controller.abort());
          cancelUpload();
        }}
      />
      <GraphExplorer sessionId={currentSession?.id ?? null} sessionName={currentSession?.name} active={graphExplorerOpen} onClose={() => setGraphExplorerOpen(false)} onAskInChat={askFromGraphExplorer} focusNodeId={graphFocusNodeId} papers={papers}/>
      <PaperMap sessionId={currentSession?.id ?? null} active={paperMapOpen} onClose={() => setPaperMapOpen(false)} papers={papers}/>
      <Tour steps={ATLAS_TOUR_STEPS} active={tourOpen} onClose={() => { setTourOpen(false); setTourSeen(true); }}/>
      <header className="app-header">
        <button className="app-sidebar-toggle" onClick={toggleSidebar} title={sidebarCollapsed ? "Show sessions" : "Hide sessions"} aria-label="Toggle sessions"><Icon name="menu" size={19}/></button>
        <div className={`mode-label ${papers.length ? "paper-mode" : ""}`}>
          {papers.length > 0 ? (
            <div className="paper-chip-row">
              {papers.map((p) => <span key={p.id} className="paper-chip"><Icon name="paper" size={12}/><strong>{p.title}</strong><button className="paper-chip-deep-dive" onClick={() => window.location.assign(`/deep-dive/${encodeURIComponent(p.id)}?session_id=${encodeURIComponent(currentSession?.id ?? "local")}`)} aria-label={`Open deeper dive for ${p.title}`} title="Open deeper dive"><Icon name="book" size={11}/></button><button onClick={() => removePaperFromSet(p.id)} aria-label={`Remove ${p.title}`}><Icon name="close" size={11}/></button></span>)}
            </div>
          ) : <><span className="online-dot"/><strong>General chat</strong><small>Atlas</small></>}
        </div>
        <div className="header-actions">
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
          ><Icon name="graph" size={17}/><em>ALPHA</em></button>
          <button className="paper-map-toggle" onClick={() => setPaperMapOpen(true)} disabled={papers.length < 2} title={papers.length < 2 ? "Add at least two papers to map connections" : "Map connections between this session's papers"} aria-label="Map paper connections"><Icon name="papersLink" size={15}/><span>Map papers</span><em>ALPHA</em></button>
          <button className="tour-help-button" onClick={() => setTourOpen(true)} title="Replay the walkthrough" aria-label="Replay the walkthrough"><Icon name="help" size={17}/></button>
          <button className="add-paper-button" onClick={() => openAddPaper()}><Icon name="plus" size={17}/>{papers.length ? "Add another" : "Add paper"}</button>
          <div className="user-menu" ref={userMenuRef}>
            <button className="user-menu-toggle" onClick={() => setUserMenuOpen((open) => !open)} aria-label="Account menu">
              {(user.displayName ?? user.email ?? "?").charAt(0).toUpperCase()}
            </button>
            {userMenuOpen && (
              <div className="user-menu-panel">
                <div>
                  <strong>{user.displayName ?? "Signed in"}</strong>
                  <small>{user.email}</small>
                </div>
                <button className="user-menu-item" onClick={() => { toggleTheme(); }}>
                  {theme === "dark" ? (
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><circle cx="12" cy="12" r="4.5"/><path d="M12 2v2.5M12 19.5V22M4.2 4.2l1.8 1.8M18 18l1.8 1.8M2 12h2.5M19.5 12H22M4.2 19.8l1.8-1.8M18 6l1.8-1.8"/></svg>
                  ) : (
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M20.5 14.7A8.5 8.5 0 1 1 9.3 3.5a7 7 0 0 0 11.2 11.2z"/></svg>
                  )}
                  {theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
                </button>
                <button className="user-menu-signout" onClick={() => { setUserMenuOpen(false); void signOut(); }}>Sign out</button>
              </div>
            )}
          </div>
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
                    {message.guide && message.paperId && <button type="button" className="deep-dive-launch" onClick={() => window.location.assign(`/deep-dive/${encodeURIComponent(message.paperId!)}?session_id=${encodeURIComponent(currentSession?.id ?? "local")}`)}><strong>Open deeper dive <em>ALPHA</em></strong><small>Explore the full paper text with a section tutor.</small></button>}
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
                        <button onClick={() => setExpandedCitation(expandedCitation === key ? null : key)}><Icon name="quote" size={13}/>{citationSectionLabel(citation.section) || "Source"}{citation.page_start != null && ` · p. ${citation.page_start}`}</button>
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
        <section className="modal-card">
          <header><div><span><Icon name="paper" size={18}/></span><div><strong>Add a research paper</strong><small>Give Atlas a paper to read with you</small></div></div><button onClick={() => setAddOpen(false)} aria-label="Close"><Icon name="close" size={19}/></button></header>

          {addMode === "choose" && <div className="add-choices">
            <button autoFocus onClick={() => setAddMode("upload")}><span className="choice-icon violet"><Icon name="upload" size={22}/></span><div><strong>Upload a PDF</strong><p>Choose a paper saved on your computer.</p></div></button>
            <button onClick={() => setAddMode("search")}><span className="choice-icon blue"><Icon name="globe" size={22}/></span><div><strong>Search online</strong><p>Find papers on arXiv by title, author, or topic.</p></div></button>
          </div>}

          {addMode === "upload" && <div className="upload-panel">
            <button className="back-button" onClick={() => setAddMode("choose")}>← Back</button>
            <button className="drop-zone" onClick={() => fileInput.current?.click()} onDragOver={(event) => event.preventDefault()} onDrop={onDrop} disabled={papers.length + stagedResults.length + stagedFiles.length >= MAX_SESSION_PAPERS}>
              <span><Icon name="upload" size={25}/></span><strong>{papers.length + stagedResults.length + stagedFiles.length >= MAX_SESSION_PAPERS ? `${MAX_SESSION_PAPERS}-paper limit reached` : `Choose up to ${MAX_SESSION_PAPERS} PDFs or drag them here`}</strong><small>Papers are staged below - nothing is read until you start the breakdown.</small>
            </button>
            <input ref={fileInput} className="hidden-input" type="file" accept="application/pdf,.pdf" multiple disabled={papers.length + stagedResults.length + stagedFiles.length >= MAX_SESSION_PAPERS} onChange={(event: ChangeEvent<HTMLInputElement>) => { stageFiles(Array.from(event.target.files ?? [])); event.target.value = ""; }}/>
            {uploadError && <p className="form-error">{uploadError}</p>}
          </div>}

          {addMode === "search" && <div className="search-panel">
            <button className="back-button" onClick={() => setAddMode("choose")}>← Back</button>
            <form onSubmit={runSearch}><Icon name="search" size={18}/><input autoFocus value={searchQuery} onChange={(event) => setSearchQuery(event.target.value)} placeholder="Search by title, author, or topic…"/><button disabled={searchQuery.trim().length < 2 || searching}>{searching ? "Searching…" : "Search"}</button></form>
            {searchError && <p className="form-error">{searchError}</p>}
            <div className="search-results">{searchResults.map((result) => {
              const isStaged = stagedResults.some((staged) => staged.id === result.id);
              const atLimit = papers.length + stagedResults.length + stagedFiles.length >= MAX_SESSION_PAPERS;
              return <article key={result.id}><div><span>arXiv</span><small>{result.published}</small></div><h3>{result.title}</h3><p>{result.authors}</p>
                {isStaged
                  ? <button className="staged-toggle" onClick={() => setStagedResults((current) => current.filter((staged) => staged.id !== result.id))}><Icon name="check" size={13}/>Added - remove</button>
                  : <button onClick={() => setStagedResults((current) => [...current, result])} disabled={atLimit}>Add to list</button>}
              </article>;
            })}</div>
          </div>}

          {(stagedResults.length > 0 || stagedFiles.length > 0) && <div className="staged-papers">
            <p>{stagedResults.length + stagedFiles.length} paper{stagedResults.length + stagedFiles.length === 1 ? "" : "s"} ready to add</p>
            <ul>
              {stagedResults.map((result) => <li key={result.id}><span>{result.title}</span><button onClick={() => setStagedResults((current) => current.filter((staged) => staged.id !== result.id))} aria-label={`Remove ${result.title}`}><Icon name="close" size={11}/></button></li>)}
              {stagedFiles.map((file, index) => <li key={`${file.name}-${index}`}><span>{file.name}</span><button onClick={() => setStagedFiles((current) => current.filter((_, i) => i !== index))} aria-label={`Remove ${file.name}`}><Icon name="close" size={11}/></button></li>)}
            </ul>
            <button className="start-breakdown-button" onClick={startBreakdown}>Start breaking down {stagedResults.length + stagedFiles.length} paper{stagedResults.length + stagedFiles.length === 1 ? "" : "s"}</button>
          </div>}
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
    </div>
  );
}
