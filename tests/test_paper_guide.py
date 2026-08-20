from agent.paper_guide import PaperGuideAgent
from agent.retrieval import ChunkIndex
from agent.schema import ExtractionChunk


def test_fallback_guide_preserves_section_order_and_builds_flowchart():
    index = ChunkIndex()
    index.upsert_paper(
        "paper",
        [
            ExtractionChunk(text="Introduction explains the problem and motivation.", ordinal=0, section="Introduction", page_start=1, page_end=1),
            ExtractionChunk(text="Methods explain the model pipeline.", ordinal=1, section="Methods", page_start=2, page_end=2),
            ExtractionChunk(text="Results report improved accuracy.", ordinal=2, section="Results", page_start=3, page_end=3),
        ],
    )

    guide = PaperGuideAgent().generate("Test Paper", index.paper_chunks("paper"))

    assert guide.title == "Test Paper"
    assert [section.title for section in guide.sections] == ["Introduction", "Methods", "Results"]
    assert guide.sections[0].diagram is not None
    assert [node.label for node in guide.sections[0].diagram.nodes] == [
        "Introduction",
        "Methods",
        "Results",
    ]
