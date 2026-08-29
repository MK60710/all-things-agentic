"use client";

import { useEffect, useState } from "react";
import { getSessionPaperMap } from "@/lib/api";
import type { PaperContext } from "@/lib/api";
import type { PaperConnection } from "@/lib/types";

export default function PaperMap({ sessionId, active, onClose, papers }: { sessionId: string | null; active: boolean; onClose: () => void; papers: PaperContext[] }) {
  const [connections, setConnections] = useState<PaperConnection[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    if (!active || !sessionId) return;
    setLoading(true); setError("");
    getSessionPaperMap(sessionId).then(setConnections).catch((err) => setError(err instanceof Error ? err.message : "Could not map paper connections.")).finally(() => setLoading(false));
  }, [active, sessionId]);
  if (!active) return null;
  const connectedIds = new Set(connections.flatMap((connection) => [connection.paper_a_id, connection.paper_b_id]));
  const unconnectedPapers = papers.filter((paper) => !connectedIds.has(paper.id));
  return <div className="paper-map-overlay"><button className="modal-scrim" onClick={onClose} aria-label="Close paper map"/><section className="paper-map-card" role="dialog" aria-modal="true" aria-label="Paper connections"><header><div><small>Session research map</small><h2>How your papers connect</h2><p>Only evidence-backed relationships are shown as connections. Papers without one remain available below.</p></div><button onClick={onClose} aria-label="Close">×</button></header>{loading && <p className="paper-map-state">Mapping the papers…</p>}{error && <p className="paper-map-state paper-map-error">{error}</p>}{!loading && !error && connections.length === 0 && <p className="paper-map-state">No evidence-backed connections found yet.</p>}{!loading && !error && connections.length > 0 && <div className="paper-map-list">{connections.map((connection) => <article key={`${connection.paper_a_id}-${connection.paper_b_id}`}><div className="paper-map-papers"><strong>{connection.paper_a_title}</strong><span>↔</span><strong>{connection.paper_b_title}</strong></div><p>{connection.summary}</p>{connection.shared_topics.map((topic) => <span key={topic} className="paper-map-topic">{topic}</span>)}<details><summary>View evidence</summary>{connection.evidence.map((item, index) => <blockquote key={`${item.paper_id}-${index}`}><small>{item.topic} · {item.section ?? "Paper"}</small>{item.quote && <p>“{item.quote}”</p>}</blockquote>)}</details></article>)}</div>}{!loading && !error && unconnectedPapers.length > 0 && <details className="paper-map-unconnected"><summary>Unconnected papers ({unconnectedPapers.length})</summary><p>No evidence-backed relationship was found between these papers and the others in this session yet.</p>{unconnectedPapers.map((paper) => <div key={paper.id}><strong>{paper.title}</strong></div>)}</details>}</section></div>;
}
