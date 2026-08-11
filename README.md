# Hybrid Research Partner

All Things Agentic Hackathon (Google/Devpost) — Collaborative Partner track.

An interactive research assistant that reads academic papers and uses cheap
chunk retrieval for the common question-answering path. High-confidence,
quoted relationships can additionally be stored in a small typed graph for
cross-paper comparisons and research-gap analysis.

The default ingestion and retrieval path is local: `pdftotext`, deterministic
chunking, and feature-hashed vectors. OCR, Gemini structured extraction,
Firestore persistence, and hosted embeddings are optional upgrades rather
than requirements for every paper or query.

## Stack
- Google ADK (multi-agent framework)
- Optional Gemini structured extraction via Vertex AI
- Local chunk index for retrieval
- networkx graph engine + optional Firestore persistence
- Cloud Run (deployment)

## Setup
Install from `pyproject.toml` or `requirements.txt`. The local path does not
require GCP credentials. Cloud deployment and optional Gemini/Firestore paths
require a configured GCP project.

For fully local OCR on macOS, install Tesseract (Poppler's `pdftoppm` is also
required):

```bash
brew install tesseract poppler
```

For higher-quality local semantic embeddings:

```bash
uv sync --extra local-semantic
```

`FastEmbedEmbedder` downloads its model on first use and then runs through
ONNX Runtime locally. `LocalHashingEmbedder` remains the download-free default.

## Cost controls

- PDF text extraction, chunking, local retrieval, and networkx reads are free.
- OCR runs only when embedded PDF text is unusable.
- Structured model extraction is injected explicitly; the default extractor
  returns chunks and makes no API calls.
- Firestore writes occur only when a Firestore client is supplied.
- The graph is populated only when entities or quoted relations exist.

## Local retrieval

```python
from agent.extraction_agent import ExtractionAgent
from agent.research_store import ResearchStore
from agent.retrieval import ChunkIndex

outcome = ExtractionAgent().extract_one("paper-1", "paper.pdf")
index = ChunkIndex()
ResearchStore(index).ingest(outcome.result)

context = index.assemble_context(
    "What were the main evaluation results?",
    limit=3,
    neighbor_window=1,
)
print(context.text)
```

`assemble_context` retrieves the strongest chunks, includes adjacent context,
deduplicates overlaps, restores paper order, and renders paper/page/section
labels. Its output can be shown directly or passed to an optional answer model.
