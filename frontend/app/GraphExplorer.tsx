"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useRef, useState } from "react";
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
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const fgRef = useRef<any>(null);

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

  // The default d3-force charge/link strengths are tuned for small,
  // sparse graphs - confirmed live: with 100+ nodes and this session's
  // real density, the default layout clumps into overlapping label
  // clusters instead of spreading out. This mutates the two forces
  // force-graph already wires up internally (accessed via the ref, not
  // a new d3-force import) rather than sizing them once at prop level,
  // since a fixed constant can't account for how much a given session's
  // graph actually needs to spread - reheating after every graphData
  // change lets it re-settle for the new node/edge count each time.
  //
  // Retries rather than a single fgRef.current check - confirmed live
  // via instrumentation that a plain single check silently no-ops on
  // first load: ForceGraph2D is a next/dynamic({ssr:false}) component,
  // so its ref isn't attached yet the first time graphData populates
  // (fgRef.current was null every time this fired on initial load,
  // confirmed via console logging before this fix). react-force-graph
  // -2d's own type declarations only accept a MutableRefObject for
  // `ref`, not a callback ref, which would otherwise be the cleaner fix
  // for "run exactly when the instance attaches" - polling a few times
  // at a short interval is the workaround available within that
  // constraint, and stops as soon as the ref is present.
  useEffect(() => {
    if (graphData.nodes.length === 0) return;
    let attempts = 0;
    const id = window.setInterval(() => {
      attempts += 1;
      const fg = fgRef.current;
      if (fg) {
        fg.d3Force("charge")?.strength(-340).distanceMax(800);
        fg.d3Force("link")?.distance(90).strength(0.5);
        fg.d3ReheatSimulation();
        window.clearInterval(id);
      } else if (attempts >= 20) {
        window.clearInterval(id);
      }
    }, 100);
    return () => window.clearInterval(id);
  }, [graphData]);

  const connections = useMemo(() => {
    if (!selectedNode) return [];
    const seen = new Set<string>();
    const deduped: { edgeId: string; relation: string; name: string; direction: "outgoing" | "incoming" }[] = [];
    for (const e of edges) {
      if (e.source_id !== selectedNode.node_id && e.target_id !== selectedNode.node_id) continue;
      // Direction matters for the sentence, not just cosmetics: "A
      // outperforms B" and "B outperforms A" are opposite claims - the
      // panel renders subject-verb-object in the edge's real stored
      // order, never assumes the selected node is always the subject.
      const isOutgoing = e.source_id === selectedNode.node_id;
      const otherId = isOutgoing ? e.target_id : e.source_id;
      const other = nodes.find((n) => n.node_id === otherId);
      const name = other?.name ?? otherId;
      const direction = isOutgoing ? "outgoing" as const : "incoming" as const;
      // Two distinct edge_ids can render the exact same sentence -
      // confirmed live (19 such pairs in one real session's graph):
      // extraction can independently emit the same relation twice, each
      // with its own edge_id and possibly its own source quote, so
      // add_edge's per-id idempotency doesn't catch it. Both edges stay
      // real, separate graph data (their citations aren't lost - this
      // only dedupes what renders in this one list), but showing the
      // identical sentence twice here is never useful, so collapse by
      // what the sentence would actually say.
      const key = `${direction}:${relationPhrase(e.relation)}:${name}`;
      if (seen.has(key)) continue;
      seen.add(key);
      deduped.push({ edgeId: e.edge_id, relation: e.relation, name, direction });
    }
    return deduped;
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
            ref={fgRef}
            graphData={graphData}
            width={size.width}
            height={size.height}
            nodeLabel="name"
            linkColor={() => "#cdc6df"}
            linkWidth={1}
            cooldownTicks={400}
            // A small graph (e.g. a single freshly-added paper, ~10 nodes)
            // was confirmed live to render mostly or entirely above the
            // visible viewport with no fixed initial camera position or
            // fit-to-content - looks broken/empty on a first-time user's
            // very first Graph Explorer visit, which is the single most
            // common real case (one paper, not the 100+-node stress-test
            // graph the force tuning above was validated against).
            // onEngineStop fires once the simulation naturally settles -
            // fitting to the real node bounding box there (not a fixed
            // camera guess) works correctly at any graph size, and refires
            // correctly if the user later changes the type filters, which
            // reheats the simulation again.
            onEngineStop={() => fgRef.current?.zoomToFit(400, 60)}
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
