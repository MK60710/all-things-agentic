from __future__ import annotations

from scripts.run_extraction_batch import entity_embedding_text


def test_entity_embedding_text_repeats_name_by_weight():
    text = entity_embedding_text("Claude", "A model.", name_weight=4)
    assert text == "Claude Claude Claude Claude: A model."


def test_entity_embedding_text_default_weight_biases_toward_name():
    """The default weighting must make two differently-named entities with
    near-identical, generic descriptions land further apart in hashed-token
    space than the unweighted "name: description" text would - this is the
    concrete fix for the false-positive needs_clarification pairs observed
    live (e.g. "Claude" vs "GPT-4o", both "a large language model used in
    experiments")."""
    from agent.retrieval import LocalHashingEmbedder
    from agent.graph_manager import _cosine_similarity

    embedder = LocalHashingEmbedder()
    description = "A large language model used as a base for experiments."

    weighted_a = embedder(entity_embedding_text("Claude", description))
    weighted_b = embedder(entity_embedding_text("GPT-4o", description))
    weighted_similarity = _cosine_similarity(weighted_a, weighted_b)

    unweighted_a = embedder(f"Claude: {description}")
    unweighted_b = embedder(f"GPT-4o: {description}")
    unweighted_similarity = _cosine_similarity(unweighted_a, unweighted_b)

    assert weighted_similarity < unweighted_similarity
