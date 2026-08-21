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

## Run the integrated app locally

Copy `.env.example` values into your shell, authenticate Application Default
Credentials, then start the API:

```bash
uv run uvicorn service.app:app --reload --port 8000
```

Copy `frontend/.env.local.example` to `frontend/.env.local`, using the same
`API_SHARED_SECRET`, then start Next.js in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`. General chat, PDF ingestion, arXiv ingestion,
guided visual paper walkthroughs, and paper-grounded chat all use the FastAPI service. PDF uploads are limited
to 25 MiB and use a short-lived browser upload token; the permanent shared
secret stays in the Next.js server process.

## Stack
- Gemini chat and optional structured extraction via Vertex AI
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

## Vertex structured extraction

Vertex AI uses Application Default Credentials rather than an API key:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project "$GOOGLE_CLOUD_PROJECT"
```

```python
from agent.extraction_agent import ExtractionAgent
from agent.gemini_extractor import GeminiStructuredExtractor

extractor = ExtractionAgent(
    structured_extractor=GeminiStructuredExtractor(
        project="all-things-agentic-hack",
        location="global",
    )
)
outcome = extractor.extract_one("paper-1", "paper.pdf")
```

The default model is `gemini-2.5-flash-lite` with thinking disabled and a
2,048-token output cap. Calls are bounded by source-window and per-paper limits.
Only relations with source quotes found in the supplied source window and valid
ontology endpoint signatures are retained.
