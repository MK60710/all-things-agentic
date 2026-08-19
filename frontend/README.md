# Atlas frontend

Next.js and TypeScript research chat. Atlas starts as a regular Gemini
conversation. A user can optionally upload a PDF or search arXiv and attach a
paper; subsequent answers then use that paper as context and show citations.

## Run locally

Start the FastAPI backend first (see the repository root `README.md`), then:

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

## Connect the Python service

Create `.env.local` (see `.env.local.example`):

```bash
BACKEND_API_URL=http://127.0.0.1:8080
NEXT_PUBLIC_API_URL=http://127.0.0.1:8080
API_SHARED_SECRET=<same value the backend's API_SHARED_SECRET is set to>
```

`BACKEND_API_URL` is what the Next.js server routes call (falls back to
`NEXT_PUBLIC_API_URL` if unset). `NEXT_PUBLIC_API_URL` is also read directly
by the browser for the PDF upload request. `API_SHARED_SECRET` stays
server-side only and is attached as an `X-API-Key` header - the browser
never sees it.

All research features go through Next.js server routes, which proxy to the
FastAPI backend (`service/app.py`) rather than being called from the browser
directly:

- `POST /api/chat` -> backend `POST /chat` - general chat, or paper-grounded
  chat when `paper_id` is set; carries conversation history both ways
- `POST /api/papers/arxiv` -> backend `POST /papers/arxiv` - fetch and
  ingest a paper by arXiv id
- `POST /api/papers/upload-token` -> backend `POST /papers/upload-token` -
  issues a short-lived, single-use upload token
- `GET /api/papers/search` - queries the public arXiv API directly; no
  backend involved

PDF upload is the one path that skips the Next.js proxy: the browser
uploads straight to the backend's `POST /papers`, authenticated with the
token from `/api/papers/upload-token` in an `X-Upload-Token` header, so the
permanent `API_SHARED_SECRET` never reaches client-side JS.

The `/api/chat` response matches `QueryResult` from `agent/query_agent.py`:
`answer`, `citations`, `retrieval_mode`, plus optional
clarification/candidate fields.

When the Python API is absent, the controls remain fully interactive and
the interface reports that the backend is not connected instead of
returning a fabricated answer.

## Checks

```bash
npm run lint
npm run build
```
