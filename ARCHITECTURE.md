# Architecture

Atlas is a FastAPI backend (`service/` + `agent/`) and a Next.js frontend
(`frontend/`), talking to Firestore and Vertex AI (Gemini). The backend
deploys to Cloud Run as a single container (see [`service/DEPLOY.md`](service/DEPLOY.md));
the frontend currently runs as a local Next.js dev server (`npm run dev`) -
no cloud deployment target is configured for it yet.

## Layers

```mermaid
graph LR
    Browser["Browser"]
    Frontend["Frontend<br/>Next.js - UI + a server-side<br/>proxy holding the shared secret<br/>(local dev server, no cloud target yet)"]
    Backend["Backend<br/>FastAPI routers - chat, query, papers,<br/>sessions, gaps, contradictions,<br/>feynman, clarifications, health<br/>(Cloud Run)"]
    Domain["agent/ - domain layer<br/>graph, retrieval, ingestion,<br/>query, judges<br/>(see detail below)"]
    External["Firestore + Vertex AI + arXiv"]

    Browser --> Frontend
    Frontend -- "X-API-Key" --> Backend
    Backend --> Domain
    Domain --> External
```

## Agent layer detail

```mermaid
graph TB
    subgraph Storage["Storage & retrieval"]
        GM["GraphManager<br/>networkx graph, RLock-protected"]
        CI["ChunkIndex<br/>chunk retrieval, RLock-protected"]
        PS["PaperStore"]
    end

    subgraph Ingest["Ingestion"]
        EA["ExtractionAgent<br/>+ GeminiStructuredExtractor"]
    end

    subgraph Reason["Query & reasoning"]
        QA["QueryAgent"]
        CO["ClarificationOrchestrator"]
        CF["ContradictionFinder"]
        GF["GapFinder"]
        FC["FeynmanChecker"]
        GJ["gemini_judge.py<br/>shared lazy-client + structured-call<br/>machinery used by CF / GF / FC"]
    end

    Firestore[("Firestore")]
    Vertex["Vertex AI<br/>Gemini 3.5-flash / 3.5-flash-lite<br/>(global endpoint)"]
    ArXiv["arXiv API"]

    Ingest --> Storage
    Ingest --> Vertex
    Ingest --> ArXiv
    Reason --> Storage
    Reason --> Vertex
    Storage --> Firestore
```

Every request from a Backend router lands in one of these three groups
- ingest a paper, ask a question, or check/quiz against the graph - and
every group ultimately reads or writes through `GraphManager`/`ChunkIndex`/
`PaperStore`, the three classes carrying the concurrency locks described
below.

## Two core request flows

**Ingesting a paper** - `POST /papers` (arXiv id or PDF upload):

```mermaid
sequenceDiagram
    participant U as Browser
    participant API as FastAPI (papers.py)
    participant EA as ExtractionAgent
    participant GX as GeminiStructuredExtractor
    participant GM as GraphManager
    participant CI as ChunkIndex
    participant FS as Firestore

    U->>API: POST /papers {arxiv_id or file}
    API->>EA: extract_one(paper_id, source)
    EA->>GX: structured extraction (entities, quoted relations)
    GX->>GM: entities/edges via apply_extraction_result()
    Note over GM: canonicalize() + add_node()<br/>run under one lock acquisition -<br/>closes a real duplicate-node race<br/>under concurrent ingests
    EA->>CI: upsert_paper() - chunk + embed for retrieval
    GM->>FS: persist nodes/edges
    CI->>FS: persist chunks
    API-->>U: PaperIngestResponse (status, any pending clarification)
```

**Asking a question, with contradiction/gap awareness** - `POST /chat`:

```mermaid
sequenceDiagram
    participant U as Browser
    participant API as FastAPI (chat.py)
    participant QA as QueryAgent
    participant CI as ChunkIndex
    participant GM as GraphManager
    participant V as Vertex AI

    U->>API: POST /chat {message, paper_ids, session_id}
    API->>API: validate paper_ids against real<br/>session membership server-side
    API->>QA: answer(message, paper_ids, history)
    QA->>GM: graph-grounded lookup (entities, edges, citations)
    QA->>CI: assemble_context() - chunk retrieval fallback
    QA->>V: generate_content (grounded answer)
    V-->>QA: answer + citations
    QA-->>U: QueryResult {answer, citations, confidence}
```

## Why the locks

`GraphManager`, `ChunkIndex`, and `PaperStore` are each shared, mutable,
in-process state accessed by FastAPI's concurrently-run sync route
handlers - a chat request reading `ChunkIndex` while a paper ingest
writes to it, or two ingests racing to create the same new graph node.
Each carries its own lock scoped tightly to the state it protects, never
held across slow work (a Gemini call, a Firestore network round trip) -
see the classes themselves for the specific race each lock closes and
the regression test that proves it.

## Deployment

- **Backend**: single container, Cloud Run, `--min-instances=1
  --max-instances=1` (required, not tuned - see `service/DEPLOY.md` for
  why more than one instance would let sessions diverge). Auth is a
  shared-secret `X-API-Key` header, not full user auth - a cost gate on
  an otherwise-public URL.
- **Frontend**: local Next.js dev server only, as of this diagram. Its
  `app/api/*` routes are a real security boundary even locally - they're
  the only place `API_SHARED_SECRET` ever exists outside the backend
  itself, so the browser never sees it.
- **Data**: Firestore (project `all-things-agentic-hack`), same project
  Vertex AI calls resolve against via Application Default Credentials.
