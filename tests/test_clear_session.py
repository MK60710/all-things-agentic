from __future__ import annotations

from scripts.clear_session import clear_session


def _seed(
    db,
    *,
    session_id: str,
    paper_id: str,
    node_ids: list[str],
    edge_id: str | None = None,
    chunk_id: str | None = None,
) -> None:
    db.collection("papers").document(paper_id).set({"id": paper_id, "session_id": session_id})
    for node_id in node_ids:
        db.collection("nodes").document(node_id).set({"id": node_id, "session_id": session_id})
    if edge_id:
        db.collection("edges").document(edge_id).set({"id": edge_id, "session_id": session_id})
    if chunk_id:
        db.collection("chunks").document(chunk_id).set({"id": chunk_id, "paper_id": paper_id})


def test_clear_session_deletes_only_the_targeted_sessions_data(fake_db):
    _seed(
        fake_db,
        session_id="session-a",
        paper_id="paper-a",
        node_ids=["node-a1", "node-a2"],
        edge_id="edge-a",
        chunk_id="chunk-a",
    )
    _seed(
        fake_db,
        session_id="session-b",
        paper_id="paper-b",
        node_ids=["node-b1"],
        edge_id="edge-b",
        chunk_id="chunk-b",
    )

    counts = clear_session(fake_db, "session-a")

    assert counts == {"papers": 1, "nodes": 2, "edges": 1, "chunks": 1, "clarifications": 0}
    assert [d.id for d in fake_db.collection("papers").stream()] == ["paper-b"]
    assert {d.id for d in fake_db.collection("nodes").stream()} == {"node-b1"}
    assert [d.id for d in fake_db.collection("edges").stream()] == ["edge-b"]
    assert [d.id for d in fake_db.collection("chunks").stream()] == ["chunk-b"]


def test_clear_session_dry_run_reports_without_deleting(fake_db):
    _seed(fake_db, session_id="session-a", paper_id="paper-a", node_ids=["node-a1"])

    counts = clear_session(fake_db, "session-a", dry_run=True)

    assert counts["nodes"] == 1
    assert [d.id for d in fake_db.collection("nodes").stream()] == ["node-a1"]  # untouched


def test_clear_session_cascades_dangling_clarification_questions(fake_db):
    """A clarification question referencing a node this session created
    must not be left pointing at a node that no longer exists."""
    _seed(fake_db, session_id="session-a", paper_id="paper-a", node_ids=["node-a1"])
    fake_db.collection("clarifications").document("q1").set(
        {"provisional_node_id": "node-a1", "candidate_node_id": "other-node"}
    )
    fake_db.collection("clarifications").document("q2").set(
        {"provisional_node_id": "unrelated", "candidate_node_id": "also-unrelated"}
    )

    counts = clear_session(fake_db, "session-a")

    assert counts["clarifications"] == 1
    assert [d.id for d in fake_db.collection("clarifications").stream()] == ["q2"]
