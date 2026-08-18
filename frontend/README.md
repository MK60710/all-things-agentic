# Atlas frontend

Next.js and TypeScript research chat. Atlas starts as a regular Gemini
conversation. A user can optionally upload a PDF or search arXiv and attach a
paper; subsequent answers then use that paper as context and show citations.

## Run locally

```bash
npm install
npm run dev
```

Open `http://localhost:3000`. Without an API URL, the interface uses the
included paper library and sample grounded answers.

## Connect the Python service

Create `.env.local`:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

General chat uses the repository's existing `hello_world` Google ADK agent
with Gemini on Vertex AI. Start the ADK server from the repository root:

```bash
GOOGLE_GENAI_USE_VERTEXAI=TRUE \
GOOGLE_CLOUD_PROJECT=all-things-agentic-hack \
GOOGLE_CLOUD_LOCATION=global \
adk api_server --session_service_uri=memory:// .
```

The Next.js `/api/chat` route handles ADK session creation and `/run` calls.
Point it at the ADK server before starting Next.js:

```bash
ADK_API_URL=http://localhost:8000
ADK_APP_NAME=hello_world
```

The remaining paper features use the research service configured through
`NEXT_PUBLIC_API_URL`:

- `POST /papers/upload` for multipart PDF ingestion
- `POST /query` for paper-grounded questions

The general chat request contains the new message and conversation history.
The paper query contains the question plus the active paper metadata:

```json
{
  "query": "How does graph memory improve retrieval?",
  "paper_id": "paper-id",
  "paper": { "id": "paper-id", "title": "Paper title" }
}
```

The response matches `QueryResult` from `agent/query_agent.py`: `answer`,
`citations`, and `retrieval_mode`.

Online discovery is implemented by the frontend's `/api/papers/search` route,
which searches arXiv. When the Python API is absent, the controls remain fully
interactive and the interface reports that Gemini or ingestion is not
connected instead of returning a fabricated answer.

## Checks

```bash
npm run lint
npm run build
```
