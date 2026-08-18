"""Part 5: Clarification Orchestrator.

The shared piece both the extraction path and the query path hand
ambiguous decisions to. It holds pending questions, applies answers back
through the existing correction mechanism (GraphManager.resolve_alias for
entity merges), and exposes a small terminal review loop as the interim
way a person actually sees and answers open questions before Part 8 (the
frontend) exists to do it.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

if TYPE_CHECKING:
    from agent.graph_manager import GraphManager, NodeSearchHit

logger = logging.getLogger(__name__)

QuestionKind = Literal["entity_merge", "query_disambiguation"]

# Sentinel option id for "these are genuinely different things" on an
# entity_merge question - distinct from any real node_id.
DISTINCT_OPTION_ID = "distinct"


@dataclass
class ClarificationOption:
    id: str
    label: str
    description: str = ""


@dataclass
class PendingQuestion:
    id: str
    kind: QuestionKind
    question: str
    options: list[ClarificationOption]
    status: Literal["open", "answered"] = "open"
    answer_option_id: str | None = None
    # entity_merge only: the node extraction already created (so the batch
    # never stalls) and the existing node it might actually be the same as.
    provisional_node_id: str | None = None
    candidate_node_id: str | None = None
    score: float | None = None
    # query_disambiguation only: the original query text, so a caller can
    # re-run the query once an option is chosen.
    query_text: str | None = None


class ClarificationOrchestrator:
    """Holds unresolved ambiguity from both extraction and querying.

    Extraction never blocks on this - apply_extraction_result already
    writes its best guess, and an entity_merge question just tags that
    guess for a later fix via the same resolve_alias mechanism used for
    manual corrections today. Querying returns its question directly in
    the answer instead of waiting on this, since the person asking is
    already there to answer it in the same exchange, but the question is
    still recorded here so a chosen option has somewhere real to apply to.
    """

    def __init__(self, graph_manager: "GraphManager | None" = None):
        self._graph = graph_manager
        self._questions: dict[str, PendingQuestion] = {}

    def register_entity_merge_question(
        self,
        *,
        provisional_node_id: str,
        entity_name: str,
        candidate_node_id: str,
        candidate_name: str,
        candidate_description: str = "",
        score: float | None = None,
    ) -> PendingQuestion:
        question = PendingQuestion(
            id=str(uuid.uuid4()),
            kind="entity_merge",
            question=(
                f'Extraction added "{entity_name}" as a new node. Is it '
                f'actually the same as the existing "{candidate_name}"?'
            ),
            options=[
                ClarificationOption(
                    id=candidate_node_id,
                    label=f"Yes, same as {candidate_name}",
                    description=candidate_description,
                ),
                ClarificationOption(
                    id=DISTINCT_OPTION_ID,
                    label="No, genuinely different",
                ),
            ],
            provisional_node_id=provisional_node_id,
            candidate_node_id=candidate_node_id,
            score=score,
        )
        self._questions[question.id] = question
        return question

    def register_query_disambiguation(
        self, query_text: str, candidates: "list[NodeSearchHit]"
    ) -> PendingQuestion:
        question = PendingQuestion(
            id=str(uuid.uuid4()),
            kind="query_disambiguation",
            question=f'Which of these did you mean by "{query_text}"?',
            options=[
                ClarificationOption(
                    id=hit.node_id, label=hit.name, description=hit.description
                )
                for hit in candidates
            ],
            query_text=query_text,
        )
        self._questions[question.id] = question
        return question

    def pending(self) -> list[PendingQuestion]:
        return [q for q in self._questions.values() if q.status == "open"]

    def get(self, question_id: str) -> PendingQuestion | None:
        return self._questions.get(question_id)

    def answer(self, question_id: str, option_id: str) -> PendingQuestion:
        """Apply a chosen option back to the system.

        entity_merge answers actually mutate the graph (merge or mark
        distinct via GraphManager.resolve_alias). query_disambiguation
        answers just record the choice - re-running the query with the
        chosen node_id is the caller's job, not this method's, since
        nothing here was ever wrong, just ambiguous.
        """
        question = self._questions.get(question_id)
        if question is None:
            raise KeyError(f"no pending question {question_id!r}")
        valid_ids = {opt.id for opt in question.options}
        if option_id not in valid_ids:
            raise ValueError(
                f"{option_id!r} is not a valid option for {question_id!r}"
            )

        if question.kind == "entity_merge":
            if self._graph is None:
                raise RuntimeError(
                    "answering an entity_merge question requires a graph_manager"
                )
            merge = option_id != DISTINCT_OPTION_ID
            self._graph.resolve_alias(
                canonical_id=question.candidate_node_id,
                alias_id=question.provisional_node_id,
                distinct=not merge,
            )

        question.status = "answered"
        question.answer_option_id = option_id
        return question

    def run_terminal_review_loop(
        self,
        input_fn: Callable[[str], str] = input,
        print_fn: Callable[..., None] = print,
    ) -> int:
        """Prints every open question and prompts for an answer right there
        in the terminal - the interim way a person finds out about and
        resolves extraction-side ambiguity until Part 8 (the frontend)
        exists. Returns how many questions got answered."""
        open_questions = self.pending()
        if not open_questions:
            print_fn("No pending clarification questions.")
            return 0

        print_fn(f"{len(open_questions)} question(s) need your input:")
        answered = 0
        for question in open_questions:
            print_fn(f"\n{question.question}")
            for index, option in enumerate(question.options, start=1):
                suffix = f" - {option.description}" if option.description else ""
                print_fn(f"  {index}. {option.label}{suffix}")
            print_fn("  s. skip for now")
            choice = input_fn("> ").strip().lower()
            if choice == "s":
                continue
            try:
                selected = question.options[int(choice) - 1]
            except (ValueError, IndexError):
                print_fn("Not a valid choice, skipping.")
                continue
            self.answer(question.id, selected.id)
            answered += 1
        print_fn(f"\nAnswered {answered} of {len(open_questions)} question(s).")
        return answered
