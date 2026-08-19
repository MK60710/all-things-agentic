"use client";

import dynamic from "next/dynamic";
import { useEffect, useRef, useState } from "react";
import type { GraphVizEdge, GraphVizNode } from "@/lib/types";

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

const TYPE_COLORS: Record<string, string> = {
  PAPER: "#2d283c",
  CONCEPT: "#6658d5",
  METHOD: "#347eaa",
  MODEL: "#a34f3e",
  BENCHMARK_DATASET: "#5a9e6f",
  METRIC: "#c98a2e",
  CLAIM: "#8a5ba3",
};
const DEFAULT_NODE_COLOR = "#9b96a4";
const REVEAL_INTERVAL_MS = 200;

interface GraphNode {
  id: string;
  name: string;
  type?: string | null;
  reused: boolean;
}

interface GraphLink {
  source: string;
  target: string;
  relation: string;
}

export default function GraphBuildAnimation({
  newNodes,
  newEdges,
  onComplete,
}: {
  newNodes: GraphVizNode[];
  newEdges: GraphVizEdge[];
  onComplete: () => void;
}) {
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [links, setLinks] = useState<GraphLink[]>([]);
  const [status, setStatus] = useState("Reading the paper...");
  const revealedNodeIds = useRef<Set<string>>(new Set());

  useEffect(() => {
    // Nodes first (so every edge below has both endpoints already present),
    // then edges - both real data from this ingest, just revealed here on a
    // timer for effect rather than as one instant dump.
    const steps: Array<() => void> = [];
    newNodes.forEach((node) => {
      steps.push(() => {
        revealedNodeIds.current.add(node.node_id);
        setNodes((current) => [...current, { id: node.node_id, name: node.name, type: node.type, reused: node.reused_existing_node }]);
        setStatus(node.reused_existing_node ? `Linked "${node.name}" to your graph` : `Found "${node.name}"`);
      });
    });
    newEdges
      .filter((edge) => revealedNodeIds.current.has(edge.source_id) || newNodes.some((n) => n.node_id === edge.source_id))
      .forEach((edge) => {
        steps.push(() => {
          setLinks((current) => [...current, { source: edge.source_id, target: edge.target_id, relation: edge.relation }]);
          setStatus(`Connecting ${edge.relation.toLowerCase().replace(/_/g, " ")}`);
        });
      });

    if (steps.length === 0) {
      onComplete();
      return;
    }

    let index = 0;
    const timer = window.setInterval(() => {
      steps[index]?.();
      index += 1;
      if (index >= steps.length) {
        window.clearInterval(timer);
        setStatus("Added to your knowledge graph");
        window.setTimeout(onComplete, 900);
      }
    }, REVEAL_INTERVAL_MS);
    return () => window.clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="graph-build">
      <ForceGraph2D
        graphData={{ nodes, links }}
        width={520}
        height={280}
        nodeLabel="name"
        nodeColor={(node: object) => {
          const { reused, type } = node as GraphNode;
          return reused ? "#efedfb" : TYPE_COLORS[type ?? ""] ?? DEFAULT_NODE_COLOR;
        }}
        nodeRelSize={5}
        linkColor={() => "#d7d2de"}
        linkDirectionalParticles={1}
        linkDirectionalParticleWidth={2}
        cooldownTicks={80}
        enableZoomInteraction={false}
        enablePanInteraction={false}
      />
      <p className="graph-build-status">{status}</p>
    </div>
  );
}
