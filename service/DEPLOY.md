# Deploying the backend to Cloud Run

The FastAPI service in this directory is built and verified (see the
commit that added it for what was tested). This doc covers the one step
left: the actual deploy.

## No IAM setup needed

Checked live: the project's default compute service account
(`321308278055-compute@developer.gserviceaccount.com`) already has
`roles/editor`, which covers both Firestore and Vertex AI. Nothing to
grant before deploying.

## Deploy command

Pick an `API_SHARED_SECRET` value first (any random string) and share it
with whoever's calling this from the frontend - it's a cost gate on the
open URL, not user auth, but every request needs it in an `X-API-Key`
header once this is set.

```bash
gcloud run deploy all-things-agentic-api \
  --source . \
  --project all-things-agentic-hack \
  --region us-central1 \
  --allow-unauthenticated \
  --min-instances=1 --max-instances=1 --concurrency=4 \
  --timeout=300 --memory=1Gi --cpu=1 \
  --set-env-vars=GOOGLE_CLOUD_PROJECT=all-things-agentic-hack,GOOGLE_CLOUD_LOCATION=us-central1,API_SHARED_SECRET=<pick-a-value>
```

`--min-instances=1 --max-instances=1` is not a tuning choice - it's
required. See `service/state.py`'s module docstring: `ChunkIndex` and
`ClarificationOrchestrator` are in-memory only, no Firestore rehydration,
so more than one running instance would split state across processes
that can't see each other's data.

## Verifying it after deploy

```bash
SERVICE_URL=$(gcloud run services describe all-things-agentic-api \
  --project all-things-agentic-hack --region us-central1 \
  --format='value(status.url)')

curl "$SERVICE_URL/healthz"

curl -X POST "$SERVICE_URL/query" \
  -H "content-type: application/json" \
  -H "X-API-Key: <the-value-you-picked>" \
  -d '{"query":"What is MemoryBank?"}'
```

Don't just check for a 200 - look at the actual JSON body. An empty or
error-shaped 200 isn't a real pass.

## What already got tested (before this deploy step)

- Full local `uvicorn` run against real Firestore + real Vertex AI,
  covering every endpoint: `/healthz`, `/query` (no_results, graph mode,
  and a real ambiguity hit that correctly returned a clarifying
  question), `/query/feedback`, `/clarifications` (list, answer, and a
  confirmed graph mutation from the answer), `/gaps` (real
  `GeminiExplainer` output), `/papers` (a real corpus PDF, real Gemini
  extraction, ~70s round trip, correctly surfaced a partial-extraction
  issue and a pending clarification question through the HTTP response).
- A real `gcloud builds submit` confirming the Dockerfile builds and
  pushes cleanly (image deleted afterward - build-only, this repo's
  session never ran or deployed that image).
- 150 automated tests, including 13 FastAPI `TestClient` tests with no
  live GCP dependency (`tests/test_service.py`).

Not tested: the actual deployed Cloud Run URL, multiple concurrent
users/instances, or anything past a single `/papers` upload per request.
