"use client";

import { useCallback, useEffect, useState } from "react";

export interface TourStep {
  target: string;
  title: string;
  body: string;
  placement?: "top" | "bottom" | "left" | "right";
  // Defaults true. The pulsing cursor dot means "click this" - set false
  // for a step whose target is purely a visual anchor (e.g. the "Welcome
  // to Atlas" intro pinned near the logo) rather than something the step
  // is actually telling the user to interact with. Confirmed live: the
  // welcome step's own body never says to click anything, and its
  // target (.brand) isn't even a button - showing the cursor there
  // implied an action that doesn't exist.
  clickable?: boolean;
}

const SPOTLIGHT_PAD = 8;
const CALLOUT_WIDTH = 300;
const CALLOUT_MARGIN = 16;
const ARROW_SIZE = 14;
// Keeps the diamond from visually clipping the callout's own 14px
// border-radius corner - not just an arbitrary margin.
const ARROW_EDGE_MARGIN = 20;

// The callout's own left gets clamped to stay on-screen (see
// calloutPosition below) independently of where the target actually is -
// confirmed live: a target near a viewport edge (e.g. the topbar's
// "+ Add another" button) clamps the callout ~80px away from the
// position that would center it under the target, while the arrow used
// to sit at a hardcoded left:26px regardless, pointing at empty space
// instead of the target. This recomputes the arrow's own left so it
// tracks the target's real horizontal center, clamped to stay within
// the callout's own rounded-corner bounds rather than the callout's
// clamped position.
function arrowOffset(rect: DOMRect, calloutLeft: number) {
  const targetCenterX = rect.left + rect.width / 2;
  const raw = targetCenterX - calloutLeft - ARROW_SIZE / 2;
  return Math.min(Math.max(ARROW_EDGE_MARGIN, raw), CALLOUT_WIDTH - ARROW_EDGE_MARGIN - ARROW_SIZE);
}

function calloutPosition(rect: DOMRect, placement?: TourStep["placement"]) {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  // Rough estimate - real height varies with body text length, but this is
  // only used to decide which side has room, not to size anything.
  const calloutHeight = 160;

  const fitsBelow = rect.bottom + CALLOUT_MARGIN + calloutHeight < vh;
  const fitsAbove = rect.top - CALLOUT_MARGIN - calloutHeight > 0;
  const side: "top" | "bottom" = placement === "top" ? "top" : placement === "bottom" ? "bottom" : fitsBelow ? "bottom" : fitsAbove ? "top" : "bottom";

  // Clamped into the viewport regardless of which branch fired above - a
  // target taller than the viewport (confirmed live: .conversation-scroll)
  // satisfies neither fitsBelow nor fitsAbove, and without this clamp the
  // callout renders past the bottom edge instead of just picking the least
  // -bad spot still fully on-screen.
  const rawTop = side === "bottom" ? rect.bottom + CALLOUT_MARGIN : rect.top - CALLOUT_MARGIN - calloutHeight;
  const top = Math.min(Math.max(CALLOUT_MARGIN, rawTop), vh - calloutHeight - CALLOUT_MARGIN);
  const idealLeft = rect.left + rect.width / 2 - CALLOUT_WIDTH / 2;
  const left = Math.min(Math.max(CALLOUT_MARGIN, idealLeft), vw - CALLOUT_WIDTH - CALLOUT_MARGIN);

  return { top, left, side };
}

export default function Tour({ steps, active, onClose }: { steps: TourStep[]; active: boolean; onClose: () => void }) {
  const [stepIndex, setStepIndex] = useState(0);
  const [rect, setRect] = useState<DOMRect | null>(null);

  const step = steps[stepIndex];

  const finish = useCallback(() => {
    window.localStorage.setItem("atlas-tour-seen", "1");
    onClose();
  }, [onClose]);

  const locate = useCallback(() => {
    if (!step) return;
    const el = document.querySelector(step.target);
    if (!el) {
      // Target isn't in the DOM right now (e.g. mid-remount) - skip this
      // step rather than showing a callout pointing at nothing.
      setStepIndex((i) => {
        if (i + 1 < steps.length) return i + 1;
        finish();
        return i;
      });
      return;
    }
    setRect(el.getBoundingClientRect());
  }, [step, steps.length, finish]);

  useEffect(() => {
    if (active) setStepIndex(0);
  }, [active]);

  useEffect(() => {
    if (!active) return;
    locate();
    window.addEventListener("resize", locate);
    window.addEventListener("scroll", locate, true);
    return () => {
      window.removeEventListener("resize", locate);
      window.removeEventListener("scroll", locate, true);
    };
  }, [active, locate]);

  useEffect(() => {
    if (!active) return;
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") finish();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [active, finish]);

  if (!active || !step || !rect) return null;

  const isLast = stepIndex === steps.length - 1;
  const { top, left, side } = calloutPosition(rect, step.placement);
  const arrowLeft = arrowOffset(rect, left);

  return (
    <div className="tour-overlay" role="dialog" aria-modal="true" aria-label="Atlas walkthrough">
      <div
        className="tour-spotlight"
        style={{
          top: rect.top - SPOTLIGHT_PAD,
          left: rect.left - SPOTLIGHT_PAD,
          width: rect.width + SPOTLIGHT_PAD * 2,
          height: rect.height + SPOTLIGHT_PAD * 2,
        }}
      />
      {step.clickable !== false && (
        <div className="tour-cursor" style={{ top: rect.top + rect.height / 2, left: rect.left + rect.width / 2 }} />
      )}
      <div
        className={`tour-callout tour-callout-${side}`}
        style={{ top, left, width: CALLOUT_WIDTH, "--arrow-left": `${arrowLeft}px` } as React.CSSProperties}
      >
        <div className="tour-callout-progress">
          {steps.map((s, index) => <i key={s.target + index} className={index === stepIndex ? "active" : index < stepIndex ? "done" : ""} />)}
        </div>
        <h4>{step.title}</h4>
        <p>{step.body}</p>
        <div className="tour-callout-actions">
          <button type="button" className="tour-skip" onClick={finish}>Skip</button>
          <div>
            {stepIndex > 0 && <button type="button" onClick={() => setStepIndex((i) => i - 1)}>Back</button>}
            <button type="button" onClick={isLast ? finish : () => setStepIndex((i) => i + 1)}>{isLast ? "Done" : "Next"}</button>
          </div>
        </div>
      </div>
    </div>
  );
}
