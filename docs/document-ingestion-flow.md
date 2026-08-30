# Document Ingestion Flow

This document describes the complete ingestion path for one PDF, including
local parsing, optional OCR, retrieval indexing, Vertex AI extraction, graph
canonicalization, persistence, retries, and failure isolation.

```mermaid
flowchart TD
    A[PDF submitted with paper_id] --> B{Path allowed and file exists?}
    B -- No --> B1[Return document_ingestion error for this paper]
    B -- Yes --> C[Extract positioned text with PyMuPDF]

    C --> D{Usable embedded text found?}
    D -- Parser failed --> E[Try pdftotext fallback]
    E --> F{Usable text found?}
    D -- Too little text --> G{Local OCR configured?}
    F -- No --> G
    D -- Yes --> H[Split text into pages]
    F -- Yes --> H

    G -- No --> G1[Return document_ingestion error for this paper]
    G -- Yes --> I[Render pages with pdftoppm]
    I --> J[Recognize text with Tesseract]
    J --> H

    H --> K[Reconstruct reading order and detect sections]
    K --> L[Create ordered citation-ready chunks]
    L --> L1[Attach page, section, source, ordinal, and neighbor metadata]

    L1 --> M[Always index chunks for retrieval]
    M --> M1[Create deterministic chunk IDs]
    M1 --> M2[Create local hashing or optional FastEmbed vectors]
    M2 --> M3[Replace stale chunks for the same paper]
    M3 --> M4[Store in memory and optional Firestore]

    L1 --> N{Structured extractor configured?}
    N -- No --> N1[Return chunks with empty entities and relations]
    N -- Yes --> O[Build bounded source windows]
    O --> P{Window already cached?}
    P -- Yes --> Q[Reuse cached structured output]
    P -- No --> R[Send window to Vertex AI Gemini Flash Lite]
    R --> S[Validate JSON against fixed Pydantic ontology]
    S --> Q

    Q --> T[Deduplicate entities and relation candidates]
    T --> U{Each relation has valid endpoint types?}
    U -- No --> U1[Discard invalid relation]
    U -- Yes --> V{Source quote exists in normalized source window?}
    V -- No --> V1[Discard unverified relation]
    V -- Yes --> W[Keep relation as EXTRACTED provenance]

    U1 --> X[Build final ExtractionResult]
    V1 --> X
    W --> X
    N1 --> X

    X --> Y{Graph Manager configured and graph facts exist?}
    Y -- No --> Y1[Finish with retrieval-ready chunks]
    Y -- Yes --> Z[Upsert stable source-paper node]
    Z --> AA[Canonicalize each entity]
    AA --> AB{Normalized string match?}
    AB -- Yes --> AC[Reuse existing typed node]
    AB -- No --> AD[Compare entity embeddings]
    AD --> AE{Similarity band}
    AE -- High --> AC
    AE -- Middle --> AF[Mark needs clarification]
    AE -- Low --> AG[Create new typed node]

    AC --> AH[Resolve typed relation endpoints]
    AF --> AH
    AG --> AH
    AH --> AI[Create stable evidence-backed edges]
    AI --> AJ[Upsert nodes and edges to NetworkX]
    AJ --> AK[Upsert nodes and edges to Firestore]
    AK --> AL[Return ingestion report]

    B1 --> AM[Batch continues with the next paper]
    G1 --> AM
    AL --> AM
    Y1 --> AM
```

### In baby steps

1. The caller supplies one PDF path and a stable `paper_id`.
2. The extractor validates the path when an allowed upload/corpus root is
   configured.
3. PyMuPDF reads positioned text blocks and reconstructs multi-column reading
   order. `pdftotext` is the parser fallback.
4. If the PDF contains too little usable text, the optional local OCR path
   renders its pages and runs Tesseract. OCR-derived pages are tagged `ocr`.
5. The text is separated into pages and sections, then divided into ordered
   chunks. Every chunk retains citation and adjacency metadata.
6. Every chunk enters the retrieval index. Stable IDs make retries safe, and
   reprocessing replaces obsolete chunks instead of leaving stale results.
7. Without a semantic extractor, ingestion stops here with retrieval-ready
   chunks and no graph facts. This path makes no model API call.
8. With `GeminiStructuredExtractor`, bounded source windows are sent to Vertex
   AI. The response must match the fixed entity and relation enums.
9. Relations with impossible endpoint-type combinations are discarded. Every
   remaining relation must contain a quote found in its source window after
   whitespace and PDF-hyphenation normalization.
10. The Graph Manager canonicalizes entities by normalized name first and
    embedding similarity second. High-confidence matches merge, low-confidence
    matches become new nodes, and middle-band matches require clarification.
11. Stable nodes and evidence-backed edges are written to NetworkX and,
    optionally, Firestore. Repeating the same write does not create duplicates.
12. A failure is attached only to the affected paper, so batch processing can
    continue with subsequent PDFs.
