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
from typing import TYPE_CHECKING, Any, Callable, Literal

if TYPE_CHECKING:
    from agent.graph_manager import GraphManager, NodeSearchHit

logger = logging.getLogger(__name__)

# Sentinel option id for "these are genuinely different things" on an
# entity_merge question - distinct from any real node_id.
DISTINCT_OPTION_ID = "distinct"


@dataclass
class ClarificationOption:
    id: str
    label: str
    description: str = ""


@dataclass(kw_only=True)
class _BaseQuestion:
    id: str
    question: str
    options: list[ClarificationOption]
    status: Literal["open", "answered"] = "open"
    answer_option_id: str | None = None


@dataclass(kw_only=True)
class EntityMergeQuestion(_BaseQuestion):
    """A canonicalization decision that landed in the needs_clarification
    band. provisional_node_id/candidate_node_id are real, required
    node_ids (not str | None like a one-dataclass-for-both-kinds shape
    would leave them) - answer() feeds both straight into
    GraphManager.resolve_alias, which requires actual strings."""

    kind: Literal["entity_merge"] = "entity_merge"
    provisional_node_id: str
    candidate_node_id: str
    score: float | None = None


@dataclass(kw_only=True)
class QueryDisambiguationQuestion(_BaseQuestion):
    """Two or more distinct entities were plausibly what a query meant.
    query_text is required (not the shared-dataclass's str | None) so a
    caller re-running the query with the chosen option always has it."""

    kind: Literal["query_disambiguation"] = "query_disambiguation"
    query_text: str


# A discriminated union rather than one dataclass with every field
# optional - each kind's required fields are actually required at
# construction time, and `if isinstance(question, EntityMergeQuestion)`
# in answer() below is a real type narrowing, not just a convention no
# type system enforces.
PendingQuestion = EntityMergeQuestion | QueryDisambiguationQuestion


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

    def __init__(
        self,
        graph_manager: "GraphManager | None" = None,
        db_client: Any | None = None,
    ):
        self._graph = graph_manager
        self._db = db_client
        self._questions: dict[str, PendingQuestion] = {}
        if self._db is not None:
            self._rehydrate()

    def _rehydrate(self) -> None:
        for snapshot in self._db.collection("clarifications").stream():
            data = snapshot.to_dict()
            try:
                common = dict(
                    id=snapshot.id,
                    question=data["question"],
                    options=[ClarificationOption(**option) for option in data["options"]],
                    status=data.get("status", "open"),
                    answer_option_id=data.get("answer_option_id"),
                )
                if data.get("kind") == "entity_merge":
                    question: PendingQuestion = EntityMergeQuestion(
                        **common,
                        provisional_node_id=data["provisional_node_id"],
                        candidate_node_id=data["candidate_node_id"],
                        score=data.get("score"),
                    )
                elif data.get("kind") == "query_disambiguation":
                    question = QueryDisambiguationQuestion(
                        **common, query_text=data["query_text"]
                    )
                else:
                    continue
            except (KeyError, TypeError, ValueError):
                logger.warning("Ignoring malformed persisted clarification %s", snapshot.id)
                continue
            self._questions[question.id] = question

    def _persist(self, question: PendingQuestion) -> None:
        if self._db is None:
            return
        data: dict[str, Any] = {
            "kind": question.kind,
            "question": question.question,
            "options": [
                {"id": option.id, "label": option.label, "description": option.description}
                for option in question.options
            ],
            "status": question.status,
            "answer_option_id": question.answer_option_id,
        }
        if isinstance(question, EntityMergeQuestion):
            data.update(
                provisional_node_id=question.provisional_node_id,
                candidate_node_id=question.candidate_node_id,
                score=question.score,
            )
        else:
            data["query_text"] = question.query_text
        self._db.collection("clarifications").document(question.id).set(data, merge=True)

    def register_entity_merge_question(
        self,
        *,
        provisional_node_id: str,
        entity_name: str,
        candidate_node_id: str,
        candidate_name: str,
        candidate_description: str = "",
        score: float | None = None,
    ) -> EntityMergeQuestion:
        question = EntityMergeQuestion(
            id=str(uuid.uuid4()),
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
        self._persist(question)
        return question

    def register_query_disambiguation(
        self, query_text: str, candidates: "list[NodeSearchHit]"
    ) -> QueryDisambiguationQuestion:
        question = QueryDisambiguationQuestion(
            id=str(uuid.uuid4()),
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
        self._persist(question)
        return question

    def pending(self) -> list[PendingQuestion]:
        return [q for q in self._questions.values() if q.status == "open"]

    def get(self, question_id: str) -> PendingQuestion | None:
        return self._questions.get(question_id)

    def remove_for_node_ids(self, node_ids: set[str]) -> int:
        """Drop every question referencing any of node_ids as its
        provisional or candidate node, from both memory and Firestore -
        query_disambiguation questions have neither attribute and are
        never touched. Same membership check scripts/clear_session.py
        already uses for the same cleanup, just live rather than
        Firestore-only."""
        stale_ids = [
            question_id
            for question_id, question in self._questions.items()
            if getattr(question, "provisional_node_id", None) in node_ids
            or getattr(question, "candidate_node_id", None) in node_ids
        ]
        for question_id in stale_ids:
            del self._questions[question_id]
            if self._db is not None:
                self._db.collection("clarifications").document(question_id).delete()
        return len(stale_ids)

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

        if isinstance(question, EntityMergeQuestion):
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
        self._persist(question)
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
                choice_number = int(choice)
                if choice_number < 1 or choice_number > len(question.options):
                    # int(choice) - 1 on "0" or a negative number is still
                    # a valid Python list index (wraps to the end), so this
                    # bound check has to happen before indexing - without
                    # it, typing "0" silently selects the last option
                    # instead of being rejected as invalid.
                    raise ValueError(choice)
                selected = question.options[choice_number - 1]
            except ValueError:
                print_fn("Not a valid choice, skipping.")
                continue
            self.answer(question.id, selected.id)
            answered += 1
        print_fn(f"\nAnswered {answered} of {len(open_questions)} question(s).")
        return answered
