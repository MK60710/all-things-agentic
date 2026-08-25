"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import { checkForContradictions, getSessionBibliography, getSessionGraph } from "@/lib/api";
import type { PaperContext } from "@/lib/api";
import { TYPE_COLORS, DEFAULT_NODE_COLOR, typeLabel, relationPhrase } from "@/lib/graphColors";
import type { SessionGraphEdge, SessionGraphNode } from "@/lib/types";

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

const ALL_TYPES = ["PAPER", "CONCEPT", "METHOD", "MODEL", "BENCHMARK_DATASET", "METRIC", "CLAIM"];
const NODE_RADIUS = 5;
const MAX_LABEL_CHARS = 22;

function truncateLabel(name: string): string {
  return name.length > MAX_LABEL_CHARS ? `${name.slice(0, MAX_LABEL_CHARS - 1)}…` : name;
}

// Matches GraphBuildAnimation.tsx's window-based sizing convention - a
// persistent full-screen panel here, so the offsets subtract this
// component's own topbar + filter row instead of that component's modal
// chrome.
const TOPBAR_AND_FILTERS_HEIGHT = 116;

export default function GraphExplorer({
  sessionId,
  active,
  onClose,
  onAskInChat,
  focusNodeId,
  papers,
}: {
  sessionId: string | null;
  active: boolean;
  onClose: () => void;
  onAskInChat: (question: string) => void;
  // Set when opened via a citation's "view in graph" link (see page.tsx's
  // viewNodeInGraph) - once the session's graph loads, the matching node
  // is pre-selected so the panel opens straight to it instead of leaving
  // the user to find one dot among dozens themselves.
  focusNodeId?: string | null;
  // Used only to resolve a node citation's bare paper_id into a real
  // title for display - same paper list page.tsx already holds for the
  // header's paper chips, not a separate fetch.
  papers: PaperContext[];
}) {
  const [nodes, setNodes] = useState<SessionGraphNode[]>([]);
  const [edges, setEdges] = useState<SessionGraphEdge[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [visibleTypes, setVisibleTypes] = useState<Set<string>>(new Set(ALL_TYPES));
  const [selectedNode, setSelectedNode] = useState<SessionGraphNode | null>(null);
  const [size, setSize] = useState({ width: 800, height: 500 });
  // Set when a citation's "view in graph" click points at a node this
  // session's export doesn't have - real case, not just defensive: an
  // unscoped general-chat answer (no papers attached) can cite a graph
  // node from a paper in a different session entirely.
  const [focusNotFound, setFocusNotFound] = useState(false);
  const [checkingContradictions, setCheckingContradictions] = useState(false);
  const [contradictionMessage, setContradictionMessage] = useState<string | null>(null);
  const [exportingCitations, setExportingCitations] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);

  useEffect(() => {
    function measure() {
      setSize({
        width: Math.max(320, window.innerWidth),
        height: Math.max(280, window.innerHeight - TOPBAR_AND_FILTERS_HEIGHT),
      });
    }
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, []);

  useEffect(() => {
    if (!active || !sessionId) return;
    setSelectedNode(null);
    setFocusNotFound(false);
    setLoading(true);
    setError(null);
    getSessionGraph(sessionId)
      .then(({ nodes, edges }) => {
        setNodes(nodes);
        setEdges(edges);
        if (focusNodeId) {
          const match = nodes.find((n) => n.node_id === focusNodeId);
          if (match) setSelectedNode(match); else setFocusNotFound(true);
        }
      })
      .catch((err) => setError(err instanceof Error ? err.message : "Could not load the graph"))
      .finally(() => setLoading(false));
    // focusNodeId deliberately excluded from deps - it's only meant to
    // apply at the moment the explorer opens (active flips true), not
    // trigger a refetch on its own. The effect's closure still sees the
    // latest prop value when it does run, since active and focusNodeId
    // are set together in the same page.tsx update (viewNodeInGraph).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, sessionId]);

  const filteredNodeIds = useMemo(
    () => new Set(nodes.filter((n) => visibleTypes.has(n.type)).map((n) => n.node_id)),
    [nodes, visibleTypes],
  );
  const graphData = useMemo(() => ({
    nodes: nodes
      .filter((n) => filteredNodeIds.has(n.node_id))
      .map((n) => ({ id: n.node_id, name: n.name, type: n.type })),
    links: edges
      .filter((e) => filteredNodeIds.has(e.source_id) && filteredNodeIds.has(e.target_id))
      .map((e) => ({ source: e.source_id, target: e.target_id, relation: e.relation })),
  }), [nodes, edges, filteredNodeIds]);

  const connections = useMemo(() => {
    if (!selectedNode) return [];
    return edges
      .filter((e) => e.source_id === selectedNode.node_id || e.target_id === selectedNode.node_id)
      .map((e) => {
        // Direction matters for the sentence, not just cosmetics: "A
        // outperforms B" and "B outperforms A" are opposite claims - the
        // panel renders subject-verb-object in the edge's real stored
        // order, never assumes the selected node is always the subject.
        const isOutgoing = e.source_id === selectedNode.node_id;
        const otherId = isOutgoing ? e.target_id : e.source_id;
        const other = nodes.find((n) => n.node_id === otherId);
        return {
          edgeId: e.edge_id,
          relation: e.relation,
          name: other?.name ?? otherId,
          direction: isOutgoing ? "outgoing" as const : "incoming" as const,
        };
      });
  }, [selectedNode, edges, nodes]);

  async function handleCheckContradictions() {
    if (!sessionId || checkingContradictions) return;
    setCheckingContradictions(true);
    setContradictionMessage(null);
    try {
      const results = await checkForContradictions(sessionId);
      setContradictionMessage(
        results.length === 0
          ? "No contradictions found."
          : `Found ${results.length} contradiction${results.length === 1 ? "" : "s"}.`,
      );
      if (results.length > 0) {
        // A confirmed contradiction writes a real CONTRADICTS edge - refetch
        // so it shows up on the canvas immediately instead of waiting for
        // the next open.
        const refreshed = await getSessionGraph(sessionId);
        setNodes(refreshed.nodes);
        setEdges(refreshed.edges);
      }
    } catch (err) {
      setContradictionMessage(err instanceof Error ? err.message : "Could not check for contradictions.");
    } finally {
      setCheckingContradictions(false);
    }
  }

  async function handleExportCitations() {
    if (!sessionId || exportingCitations) return;
    setExportingCitations(true);
    setExportError(null);
    try {
      const bibtex = await getSessionBibliography(sessionId);
      const blob = new Blob([bibtex], { type: "application/x-bibtex" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "session.bib";
      link.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setExportError(err instanceof Error ? err.message : "Could not export citations.");
    } finally {
      setExportingCitations(false);
    }
  }

  function toggleType(type: string) {
    setVisibleTypes((current) => {
      const next = new Set(current);
      if (next.has(type)) next.delete(type); else next.add(type);
      return next;
    });
  }

  if (!active) return null;

  return (
    <div className="graph-explorer-stage">
      <div className="graph-explorer-topbar">
        <div className="graph-explorer-title">
          <div className="graph-explorer-title-row">
            <strong>Graph Explorer</strong>
            {!loading && !error && nodes.length > 0 && (
              <small>{nodes.length} things · {edges.length} connections</small>
            )}
          </div>
          <p className="graph-explorer-subtitle">Every dot is something from your papers - a method, a result, an idea. Click one to read about it and see what it connects to.</p>
        </div>
        <button className="graph-explorer-close" onClick={onClose} aria-label="Close graph explorer">×</button>
      </div>
      <div className="graph-explorer-filters">
        {ALL_TYPES.map((type) => (
          <label
            key={type}
            className="graph-explorer-filter"
            style={visibleTypes.has(type) ? { borderColor: TYPE_COLORS[type], background: `${TYPE_COLORS[type]}14` } : undefined}
          >
            <input type="checkbox" checked={visibleTypes.has(type)} onChange={() => toggleType(type)} />
            <span style={{ color: TYPE_COLORS[type] ?? DEFAULT_NODE_COLOR }}>{typeLabel(type)}</span>
          </label>
        ))}
        <button
          className="graph-explorer-contradictions-check"
          onClick={() => void handleCheckContradictions()}
          disabled={checkingContradictions || nodes.length === 0}
        >
          {checkingContradictions ? "Checking…" : "Check for contradictions"}
        </button>
        {contradictionMessage && <small className="graph-explorer-contradictions-result">{contradictionMessage}</small>}
        <button
          className="graph-explorer-export-citations"
          onClick={() => void handleExportCitations()}
          disabled={exportingCitations || nodes.length === 0}
        >
          {exportingCitations ? "Exporting…" : "Export citations"}
        </button>
        {exportError && <small className="graph-explorer-contradictions-result">{exportError}</small>}
      </div>
      <div className="graph-explorer-canvas">
        {focusNotFound && (
          <p className="graph-explorer-focus-miss">That node isn&rsquo;t part of this session&rsquo;s graph. It came from a different session&rsquo;s papers.</p>
        )}
        {loading && <p className="graph-explorer-status">Loading your graph…</p>}
        {error && <p className="graph-explorer-status graph-explorer-error">{error}</p>}
        {!loading && !error && nodes.length === 0 && (
          <p className="graph-explorer-status">Nothing in this session's graph yet.</p>
        )}
        {!loading && !error && nodes.length > 0 && (
          <ForceGraph2D
            graphData={graphData}
            width={size.width}
            height={size.height}
            nodeLabel="name"
            linkColor={() => "#cdc6df"}
            linkWidth={1}
            cooldownTicks={80}
            nodeCanvasObject={(node, ctx, globalScale) => {
              const n = node as SessionGraphNode & { x?: number; y?: number };
              if (n.x == null || n.y == null) return;
              const color = TYPE_COLORS[n.type] ?? DEFAULT_NODE_COLOR;
              const isSelected = selectedNode?.node_id === n.node_id;

              ctx.beginPath();
              ctx.arc(n.x, n.y, isSelected ? NODE_RADIUS + 2 : NODE_RADIUS, 0, 2 * Math.PI);
              ctx.fillStyle = color;
              ctx.fill();
              ctx.lineWidth = (isSelected ? 2.5 : 1.5) / globalScale;
              ctx.strokeStyle = isSelected ? "#2d283c" : "#ffffff";
              ctx.stroke();

              const fontSize = Math.max(10.5 / globalScale, 3.2);
              ctx.font = `600 ${fontSize}px ui-sans-serif, -apple-system, sans-serif`;
              ctx.textAlign = "center";
              ctx.textBaseline = "top";
              const label = truncateLabel(n.name);
              const labelY = n.y + NODE_RADIUS + 3 / globalScale;
              ctx.lineWidth = 3 / globalScale;
              ctx.strokeStyle = "rgba(255,255,255,.88)";
              ctx.strokeText(label, n.x, labelY);
              ctx.fillStyle = "#4a4356";
              ctx.fillText(label, n.x, labelY);
            }}
            nodePointerAreaPaint={(node, paintColor, ctx) => {
              const n = node as SessionGraphNode & { x?: number; y?: number };
              if (n.x == null || n.y == null) return;
              ctx.fillStyle = paintColor;
              ctx.beginPath();
              ctx.arc(n.x, n.y, NODE_RADIUS + 3, 0, 2 * Math.PI);
              ctx.fill();
            }}
            onNodeClick={(node: object) => {
              const match = nodes.find((n) => n.node_id === (node as { id: string }).id);
              if (match) setSelectedNode(match);
            }}
          />
        )}
        {selectedNode && (
          <aside className="graph-explorer-panel">
            <button className="graph-explorer-panel-close" onClick={() => setSelectedNode(null)} aria-label="Close panel">×</button>
            <span className="graph-explorer-panel-type" style={{ color: TYPE_COLORS[selectedNode.type] ?? DEFAULT_NODE_COLOR }}>
              {typeLabel(selectedNode.type)}
            </span>
            <h3>{selectedNode.name}</h3>
            {selectedNode.description && <p>{selectedNode.description}</p>}
            {selectedNode.citations.length > 0 && (
              <div className="graph-explorer-citations">
                <small className="graph-explorer-connections-label">Where this came from</small>
                {selectedNode.citations.map((cit) => {
                  const paper = papers.find((p) => p.id === cit.paper_id);
                  return (
                    <p key={`${cit.paper_id}-${cit.section ?? ""}`} className="graph-explorer-citation">
                      {paper?.title ?? cit.paper_id}
                      {cit.section && <span> · {cit.section}</span>}
                    </p>
                  );
                })}
              </div>
            )}
            <button
              className="graph-explorer-ask"
              onClick={() => onAskInChat(`What do you know about "${selectedNode.name}"?`)}
            >
              Ask about this in chat
            </button>
            <div className="graph-explorer-connections">
              {connections.length > 0 && (
                <small className="graph-explorer-connections-label">How this connects to other things</small>
              )}
              {connections.map((c) => (
                <button
                  key={c.edgeId}
                  className="graph-explorer-connection"
                  onClick={() => onAskInChat(
                    `How does "${selectedNode.name}" relate to "${c.name}"? (${relationPhrase(c.relation)})`,
                  )}
                >
                  {c.direction === "outgoing"
                    ? <><strong>{selectedNode.name}</strong> {relationPhrase(c.relation)} <strong>{c.name}</strong></>
                    : <><strong>{c.name}</strong> {relationPhrase(c.relation)} <strong>{selectedNode.name}</strong></>}
                </button>
              ))}
              {connections.length === 0 && (
                <p className="graph-explorer-no-connections">Nothing else in this session connects to this yet.</p>
              )}
            </div>
          </aside>
        )}
      </div>
    </div>
  );
}
