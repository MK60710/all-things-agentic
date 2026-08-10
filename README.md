# Interactive Knowledge-Graph Research Partner

All Things Agentic Hackathon (Google/Devpost) — Collaborative Partner track.

An interactive research assistant that reads a batch of academic papers,
builds a typed knowledge graph of their concepts, methods and findings
as it goes, asks clarifying questions when extraction is ambiguous,
answers queries by tracing the graph, and surfaces research gaps from
graph topology.

## Stack
- Google ADK (multi-agent framework)
- Gemini 3.5 via Vertex AI
- networkx (in-memory graph engine) + Firestore (persistence + vector search)
- Cloud Run (deployment)

## Setup
See `pyproject.toml` for dependencies. Requires `gcloud` CLI and a GCP
project with Vertex AI / Cloud Run / Firestore APIs enabled.
