"use client";

import { useEffect, useRef, useState } from "react";

export interface ConvergenceEntry {
  id: string;
  title: string;
  stageLabel?: string;
}

// A rough, honest midpoint of the real pipeline's typical wall-clock cost
// (PDF download + parallel entity extraction + guide generation racing
// each other - see agent/paper_guide.py / gemini_extractor.py). Shown as
// "~Ns left", never as a hard promise - once elapsed passes this, the
// label switches to "Almost there..." instead of counting into negative
// numbers, since this is an estimate, not a real completion signal.
const ESTIMATED_SECONDS = 70;

// A real extraction pass runs 45-90s+ - three words cycling every 1.7s
// repeats itself a dozen times over before it's done, which reads as
// stale rather than alive. A longer, more varied set (same spirit as
// Claude Code's own status-word rotation) keeps the cutscene feeling
// like something is actually still happening, not a stuck loop.
const BUILD_WORDS = [
  "Reading",
  "Parsing",
  "Analyzing",
  "Scanning",
  "Untangling",
  "Cross-referencing",
  "Distilling",
  "Mapping",
  "Connecting",
  "Synthesizing",
  "Tracing citations",
  "Weighing evidence",
  "Building the graph",
];
const WORD_INTERVAL_MS = 1700;

type Phase = "idle" | "active" | "closing" | "leaving";

export default function ConvergenceRitual({
  entries,
  active,
  onSkip,
}: {
  entries: ConvergenceEntry[];
  active: boolean;
  onSkip: () => void;
}) {
  const [phase, setPhase] = useState<Phase>("idle");
  const [flown, setFlown] = useState<Set<string>>(new Set());
  const [wordIndex, setWordIndex] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const startRef = useRef(Date.now());
  const wasActive = useRef(false);

  useEffect(() => {
    if (active && !wasActive.current) {
      startRef.current = Date.now();
      setElapsed(0);
      setFlown(new Set());
      setPhase("active");
    } else if (!active && wasActive.current && phase === "active") {
      setPhase("closing");
      const t1 = window.setTimeout(() => setPhase("leaving"), 2000);
      const t2 = window.setTimeout(() => setPhase("idle"), 3100);
      wasActive.current = active;
      return () => { window.clearTimeout(t1); window.clearTimeout(t2); };
    }
    wasActive.current = active;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active]);

  useEffect(() => {
    if (phase !== "active") return;
    const t = window.setInterval(() => setElapsed(Math.round((Date.now() - startRef.current) / 1000)), 1000);
    return () => window.clearInterval(t);
  }, [phase]);

  useEffect(() => {
    if (phase !== "active") return;
    const t = window.setInterval(() => setWordIndex((i) => (i + 1) % BUILD_WORDS.length), WORD_INTERVAL_MS);
    return () => window.clearInterval(t);
  }, [phase]);

  useEffect(() => {
    const unflown = entries.filter((e) => !flown.has(e.id));
    if (unflown.length === 0) return;
    setFlown((current) => {
      const updated = new Set(current);
      unflown.forEach((e) => updated.add(e.id));
      return updated;
    });
  }, [entries, flown]);

  function skip() {
    setPhase("idle");
    onSkip();
  }

  if (phase === "idle") return null;

  const remaining = ESTIMATED_SECONDS - elapsed;
  const timeLabel = phase === "active" ? (remaining > 0 ? `~${remaining}s left` : "Almost there…") : null;
  const statusWord = BUILD_WORDS[wordIndex];

  return (
    <div className={`convergence-stage${phase === "leaving" ? " leaving" : ""}`}>
      <div className="convergence-corner"><button className="convergence-ghost" onClick={skip}>Skip</button></div>

      <div className="convergence-star-anchor">
        <div className="convergence-halo"/>
        <div className="convergence-star">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <circle cx="12" cy="12" r="3"/>
            <path d="M12 3v6M12 15v6M3 12h6M15 12h6M5.6 5.6l4.2 4.2M14.2 14.2l4.2 4.2M18.4 5.6l-4.2 4.2M9.8 14.2l-4.2 4.2"/>
          </svg>
        </div>
        {/* Nested inside convergence-star-anchor on purpose - convergence-below's
            top:100% is meant to read as "right below the star", the same
            reference box .convergence-halo/.convergence-star already use for
            their own top:50%/left:50% centering. As a sibling of this anchor
            instead (its previous position), top:100% resolved against
            convergence-stage - the full-screen fixed container - pushing the
            status text ~150px below the bottom of the viewport. Confirmed live
            via getBoundingClientRect: the text existed in the DOM with the
            right content the whole time, just never actually on screen. */}
        <div className="convergence-below">
          {phase === "active" && (
            <>
              <p className="convergence-status">
                <span className="convergence-shimmer">{statusWord}…</span>
                {timeLabel && <small>{timeLabel}</small>}
              </p>
            </>
          )}
          {(phase === "closing" || phase === "leaving") && (
            <p className="convergence-closing">This is a guide for reference, built on your knowledge graph.</p>
          )}
        </div>
      </div>

      <div className="convergence-papers">
        {entries.map((entry) => <FlyingPaper key={entry.id} title={entry.title}/>)}
      </div>
    </div>
  );
}

function FlyingPaper({ title }: { title: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [path, setPath] = useState<string | null>(null);

  useEffect(() => {
    const side = Math.floor(Math.random() * 4);
    const pad = -14;
    const vw = window.innerWidth, vh = window.innerHeight;
    const start = side === 0 ? { x: Math.random() * vw, y: pad }
      : side === 1 ? { x: vw - pad, y: Math.random() * vh }
      : side === 2 ? { x: Math.random() * vw, y: vh - pad }
      : { x: pad, y: Math.random() * vh };
    const cx = vw / 2, cy = vh / 2;
    const midX = start.x + (cx - start.x) * 0.5 + (Math.random() - 0.5) * 220;
    const midY = start.y + (cy - start.y) * 0.5 + (Math.random() - 0.5) * 220;
    setPath(`M ${start.x} ${start.y} Q ${midX} ${midY} ${cx} ${cy}`);
  }, []);

  if (!path) return null;
  return (
    <div ref={ref} className="convergence-paper" style={{ offsetPath: `path("${path}")` }}>
      <span>{title}</span>
    </div>
  );
}
