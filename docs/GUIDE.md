# How to get the most out of Atlas

Atlas isn't just a chatbot that's read your PDFs - it builds a real
knowledge graph as you add papers, and most of what makes it worth using
only shows up once that graph has more than one paper's evidence in it.
This is the order that actually gets you there.

## 1. Add your first paper and let Guided Reading run

Click **Add a paper** (PDF upload or arXiv id/search). The moment it's
ingested, Atlas automatically starts a **Guided Reading** walkthrough -
the big picture first, then the paper section by section, each with a
small flow diagram of the ideas introduced in that section. You don't
have to trigger this; it just happens.

Read through it rather than skipping straight to chat - it's the
fastest way to know what's actually in the paper before you start
asking it questions, and it's what feeds the next step.

## 2. Take the Feynman Check at the end

When Guided Reading finishes, Atlas asks you to explain one of the
paper's own ideas back in your own words - then grades your explanation
against the paper's real extracted evidence, not its own general
knowledge of the topic. You get a **strong / weak / wrong** verdict plus
a specific, honest reason (never generic praise), with a citation back
to the real passage it graded you against.

This is the single best way to tell if you actually understood the
paper or just skimmed it. If you get "weak" or "wrong," that's real
signal - go back and reread that section rather than treating it as a
formality.

## 3. Add a second, related paper before you judge the tool

This is the part that's easy to miss: contradiction-checking and
gap-finding need more than one paper's evidence in the same session to
say anything interesting. A single-paper session can't disagree with
itself. Add a paper that's actually related to the first one - a
follow-up, a competing method, a paper it cites or that cites it - into
the **same session**, not a new one.

## 4. Ask questions in chat, and read the confidence and citations

Chat answers are grounded in your papers, not generic Gemini knowledge,
and every answer comes with citations back to a real passage. Watch for
two things:
- **Confidence level** - a low-confidence answer means the evidence was
  thin; treat it as a lead to verify, not a settled fact.
- **Clarifying questions** - if your question is ambiguous ("what is
  attention?" when there are three different "attention" entities in
  the graph), Atlas asks which one you meant instead of guessing. Answer
  it - guessing wrong here is what produces confidently-wrong answers.

## 5. Answer clarification questions as they appear

Separately from chat, ingesting a paper sometimes surfaces questions
like *"Extraction added 'GNMT + RL Ensemble' as a new node - is it
actually the same as the existing 'GNMT + RL'?"* This is the graph
asking you to resolve a possible duplicate before it pollutes the graph
with two names for the same real thing. A few seconds spent here keeps
every later feature (contradictions, gaps, citations) working against a
clean graph instead of a fragmented one.

## 6. Open Graph Explorer and check a node's Sources

Click the graph icon to see everything extracted so far as a real node
graph - filterable by type (Paper, Concept, Method, Model, Dataset,
Metric, Claim). Click any node: the panel shows its description, its
connections, and a **Sources** section listing exactly which paper (and
section, when known) it came from. This is the fastest way to verify a
claim rather than take Atlas's word for it - trace it back to the
actual text.

## 7. Run "Check for contradictions" once you have 2+ related papers

In Graph Explorer, click **Check for contradictions**. This compares
CLAIM nodes across your session's papers for genuine disagreement -
not just similar wording, but things that actually conflict. Gemini
only judges pairs that embedding similarity already flagged as plausibly
about the same thing; it doesn't invent candidates. Worth re-running
after adding each new paper to a session, not just once.

## 8. Check the "worth exploring" gap suggestions, and give feedback

Atlas surfaces research gaps on its own - pairs of entities that share
context but have no direct connection in your graph yet, each with a
real explanation of why it might matter. Click one to jump into it in
chat, or dismiss it as **Not interesting**. That feedback isn't just
logged - it actually adjusts what gets ranked higher for you going
forward, so the more you use it in a session, the better the
suggestions get.

## 9. Export your bibliography when you're ready to write

**Export citations** in Graph Explorer's topbar downloads a real BibTeX
file (`.bib`) covering every paper in the current session - ready to
drop into a paper or reference manager. No need to hand-build a
bibliography from the papers you've already read here.

## The pattern underneath all of this

Everything above threads through the same idea: Atlas is most useful
the more real evidence lives in one session's graph. A single paper
gets you Guided Reading and a Feynman Check. A session with several
related papers gets you contradiction detection, gap-finding, and
citation-backed answers that draw on all of them at once. If a feature
feels thin, the fix is usually "add another related paper to this
session," not a different tool.
