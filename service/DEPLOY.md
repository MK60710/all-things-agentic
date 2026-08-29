# Production readiness checklist

This is an operator checklist, not a deployment script. Atlas does not deploy
itself, and deployment configuration remains the operator's responsibility.

## Required backend configuration

Set these values on the backend service:

- `ATLAS_ENV=production`
- `GOOGLE_CLOUD_PROJECT`: the project containing Vertex AI and Firestore
- `GOOGLE_CLOUD_LOCATION=global`: required by the configured Gemini models
- `GEMINI_CHAT_MODEL` and `GEMINI_GUIDE_MODEL` when overriding the defaults
- `API_SHARED_SECRET`: a random value of at least 32 characters, supplied
  from Secret Manager rather than committed or exposed to browser JavaScript
- `CORS_ORIGINS`: only the exact HTTPS frontend origin or origins
- `PAPER_STORAGE_BUCKET`: a private Cloud Storage bucket for source PDFs

Production startup fails closed when the shared secret, HTTPS CORS origins, or
paper bucket is missing.

Set these values on the Next.js service:

- `BACKEND_API_URL`: the private server-side backend URL
- `NEXT_PUBLIC_API_URL`: the backend URL used only for token-authorized direct
  PDF uploads
- `API_SHARED_SECRET`: the same server-side secret as the backend
- all `NEXT_PUBLIC_FIREBASE_*` values listed in
  `frontend/.env.local.example`

`API_SHARED_SECRET` must never have a `NEXT_PUBLIC_` prefix.

## Google Cloud resources

Before release:

- Enable Vertex AI, Firestore, Cloud Storage, Firebase Authentication, and
  Secret Manager for the project.
- Create a private PDF bucket with public access prevention and uniform
  bucket-level access enabled. Do not add public object viewers.
- Give the backend service account only the access it needs: Vertex AI User,
  Datastore User, Secret Manager Secret Accessor for the API secret, and
  Storage Object User scoped to the PDF bucket. Do not use the broad Editor
  role for the production runtime.
- Configure a Firestore TTL policy on the `expires_at` field for the
  `rate_limits` collection group so expired fixed-window counters are removed.
- Enable Google as a Firebase sign-in provider and add the final frontend
  hostname to Firebase Authentication's authorized domains.

## Cost controls

Atlas rejects paid operations before Gemini is called when a limit is reached.
The backend, not the browser, owns these limits.

Per-user defaults:

| Operation | Burst limit | Daily limit |
| --- | ---: | ---: |
| Chat/query | 20/minute | 100/day |
| Paper ingestion | 3/hour | 6/day |
| Guide generation | 6/hour | 20/day |
| Contradiction checks | 10/hour | 30/day |
| Feynman checks | 20/hour | 50/day |
| Gap explanations | 10/hour | 30/day |

Global daily ceilings across every account are 500 chat requests, 15 paper
ingestions, 40 guide generations, 60 contradiction runs, 100 Feynman checks,
and 60 gap-explanation runs.

For the initial free release, start with a small monthly Google Cloud budget
(for example, USD 25) and alerts at 50%, 80%, and 100%. A billing budget is an
alert, not a guaranteed hard cap. Connect the 100% notification to a reviewed
operator response or programmatic kill switch before increasing the budget.

## Runtime constraints

Keep the backend at one process and one service instance. `GraphManager`,
`ChunkIndex`, and `ClarificationOrchestrator` rehydrate from Firestore at
startup but do not subscribe to changes made by another running instance.
Scaling beyond one instance can therefore split a live session's in-memory
view. Remove this restriction only after those components use synchronized
storage for every read/write path.

The original PDFs are durable in Cloud Storage. `/tmp` is only scratch space
for validation and parsing. Firestore remains the durable store for paper
metadata, chunks, graph records, sessions, messages, and rate-limit counters.

## Release verification

Before directing users to the app:

1. Run the full backend test suite and the optimized frontend production build.
2. Confirm the health endpoint returns a successful JSON response.
3. Sign in through the final frontend hostname.
4. Upload a PDF and import an arXiv paper, then verify both remain available
   after a backend restart.
5. Verify paper ownership by attempting cross-account session and paper reads.
6. Exercise chat, Deep Dive, graph, paper map, Feynman, contradiction, and gap
   workflows against real Vertex AI.
7. Temporarily lower a test user's limits and confirm the backend returns 429,
   Gemini is not called, and the composer stays disabled after refresh.
8. Confirm the bucket has no public access and Firestore TTL is active.
9. Confirm budget notifications reach every intended recipient.

Do not treat a health response alone as release verification; it intentionally
does not spend money by calling Gemini.
