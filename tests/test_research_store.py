from agent.clarification_orchestrator import ClarificationOrchestrator
from agent.graph_manager import GraphManager
from agent.research_store import ResearchStore
from agent.retrieval import ChunkIndex
from agent.schema import ExtractedEntity, ExtractionResult, NodeType


def test_chunk_only_ingestion_does_not_require_graph_or_cloud():
    index = ChunkIndex()
    store = ResearchStore(index)
    extraction = ExtractionResult(
        paper_id="paper-1",
        entities=[],
        relations=[],
        chunks=["A useful research passage."],
    )

    report = store.ingest(extraction, owner_uid="owner-1")

    assert len(report.chunk_ids) == 1
    assert report.graph is None
    assert index.search("research passage")[0].paper_id == "paper-1"


def test_ingest_passes_clarification_through_to_apply_extraction_result(fake_db):
    """ingest() is the real join point ExtractionAgent output flows through
    - the Part 5 extraction-side hook only fires end to end if an
    orchestrator handed to ingest() actually reaches
    GraphManager.apply_extraction_result, not just registered directly on
    the GraphManager in isolation like the graph_manager.py unit tests
    already cover."""
    graph = GraphManager(project_id="test", db_client=fake_db)
    index = ChunkIndex()
    store = ResearchStore(index, graph)
    orchestrator = ClarificationOrchestrator(graph_manager=graph)

    existing = ExtractedEntity(
        name="Fine-Tuning", type=NodeType.CONCEPT, description="Existing"
    )
    graph.apply_extraction_result(
        ExtractionResult(
            paper_id="paper-0", entities=[existing], relations=[], chunks=[]
        ),
        "owner-1",
        embedding_fn=lambda entity: [1.0, 0.0],
    )

    extraction = ExtractionResult(
        paper_id="paper-1",
        entities=[
            ExtractedEntity(
                name="Parameter Tuning",
                type=NodeType.CONCEPT,
                description="A tuning approach",
            )
        ],
        relations=[],
        chunks=["Parameter tuning is discussed here."],
    )

    store.ingest(
        extraction,
        entity_embedding_fn=lambda entity: [0.8, 0.6],  # needs_clarification band
        clarification=orchestrator,
     owner_uid="owner-1")

    assert len(orchestrator.pending()) == 1
    assert orchestrator.pending()[0].kind == "entity_merge"
