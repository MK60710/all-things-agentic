from agent.research_store import ResearchStore
from agent.retrieval import ChunkIndex
from agent.schema import ExtractionResult


def test_chunk_only_ingestion_does_not_require_graph_or_cloud():
    index = ChunkIndex()
    store = ResearchStore(index)
    extraction = ExtractionResult(
        paper_id="paper-1",
        entities=[],
        relations=[],
        chunks=["A useful research passage."],
    )

    report = store.ingest(extraction)

    assert len(report.chunk_ids) == 1
    assert report.graph is None
    assert index.search("research passage")[0].paper_id == "paper-1"
