// Shared node-color-by-type palette - single source of truth for every
// graph visualization in the app (the ingest build animation and the
// Graph Explorer both key off the same NodeType strings, so a color
// change here doesn't drift between the two).
export const TYPE_COLORS: Record<string, string> = {
  PAPER: "#2d283c",
  CONCEPT: "#6658d5",
  METHOD: "#347eaa",
  MODEL: "#a34f3e",
  BENCHMARK_DATASET: "#5a9e6f",
  METRIC: "#c98a2e",
  CLAIM: "#8a5ba3",
};
export const DEFAULT_NODE_COLOR = "#9b96a4";

// Plain-language labels for the backend's NodeType enum - "BENCHMARK_DATASET"
// means nothing to someone who hasn't read the ontology; "Dataset" does.
// Single source of truth so the Graph Explorer's filter pills and side
// panel never drift out of sync with each other.
export const TYPE_LABELS: Record<string, string> = {
  PAPER: "Paper",
  CONCEPT: "Concept",
  METHOD: "Method",
  MODEL: "Model",
  BENCHMARK_DATASET: "Dataset",
  METRIC: "Metric",
  CLAIM: "Claim",
};

// Plain verb phrases for the backend's EdgeType enum, written so
// "{node A} {phrase} {node B}" reads as an actual sentence instead of a
// label ("value-based prefixes uses TRADE" vs. a bare "USES" tag).
export const RELATION_PHRASES: Record<string, string> = {
  PROPOSES: "introduces",
  USES: "uses",
  EVALUATES_ON: "is evaluated on",
  OUTPERFORMS: "performs better than",
  EXTENDS: "builds on",
  SAME_AS: "is the same as",
  SUPPORTS: "supports",
  CONTRADICTS: "disagrees with",
};

export function relationPhrase(relation: string): string {
  return RELATION_PHRASES[relation] ?? relation.toLowerCase().replace(/_/g, " ");
}

export function typeLabel(type: string): string {
  return TYPE_LABELS[type] ?? type;
}
