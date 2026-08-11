from agent.retrieval import ChunkIndex, LocalHashingEmbedder
from agent.schema import ExtractionChunk


def test_chunk_upsert_is_idempotent_and_search_is_ranked():
    index = ChunkIndex(embedding_fn=LocalHashingEmbedder(dimensions=128))
    chunks = [
        "Transformers use self attention for sequence modeling.",
        "Convolutional networks process images with spatial kernels.",
    ]

    first_ids = index.upsert_paper("paper-1", chunks)
    second_ids = index.upsert_paper("paper-1", chunks)

    assert first_ids == second_ids
    assert index.count() == 2
    hits = index.search("transformer self attention", limit=1)
    assert hits[0].chunk_id == first_ids[0]
    assert hits[0].paper_id == "paper-1"


def test_search_can_filter_by_paper():
    index = ChunkIndex()
    index.upsert_paper("paper-1", ["agent planning and tool use"])
    index.upsert_paper("paper-2", ["agent planning with memory"])

    hits = index.search("agent planning", paper_ids={"paper-2"})

    assert [hit.paper_id for hit in hits] == ["paper-2"]


def test_assemble_context_expands_neighbors_and_restores_order():
    index = ChunkIndex()
    chunks = [
        ExtractionChunk(
            text="The experiment setup uses one GPU.",
            ordinal=0,
            page_start=3,
            page_end=3,
            section="Experiments",
        ),
        ExtractionChunk(
            text="Random forest accuracy reached 99.75 percent.",
            ordinal=1,
            page_start=4,
            page_end=4,
            section="Results",
        ),
        ExtractionChunk(
            text="The conclusion discusses future work.",
            ordinal=2,
            page_start=5,
            page_end=5,
            section="Conclusion",
        ),
    ]
    index.upsert_paper("paper-1", chunks)

    context = index.assemble_context(
        "random forest accuracy", limit=1, neighbor_window=1
    )

    assert [hit.ordinal for hit in context.hits] == [0, 1, 2]
    assert context.text.index("experiment setup") < context.text.index("99.75")
    assert "[Paper: paper-1 | Section: Results | Page: 4]" in context.text
