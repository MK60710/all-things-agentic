# Google Cloud deployment reference

Atlas is deployed. This documents the GCP resources and identities involved,
for anyone picking the project back up later.

## Resources in project `all-things-agentic-hack`

- Billing enabled, Firestore Native mode in `us-central1`.
- Vertex AI, Firestore, Cloud Storage, Firebase/Identity Toolkit, Pub/Sub,
  Billing Budgets, and Secret Manager APIs enabled.
- Runtime identity `hackathon-agent@all-things-agentic-hack.iam.gserviceaccount.com`,
  granted Vertex AI User, Datastore User, and Storage Object User scoped to
  the papers bucket.
- Private bucket `all-things-agentic-hack-atlas-papers` in `us-central1`,
  uniform bucket-level access, public access prevention on.
- Secret Manager secret `atlas-api-shared-secret` (the value is never printed
  or committed - it's referenced by the Cloud Run service, not copied
  anywhere).
- Firestore TTL enabled for collection group `rate_limits`, field
  `expires_at`.
- Pub/Sub topic `atlas-budget-alerts`, wired to a monthly budget on the
  project's billing account with alerts at 50/80/100% spend.
- Google sign-in enabled in Firebase Authentication; authorized domains
  include the production frontend hostname.

These setup actions require project-owner or billing-administrator access -
worth knowing if a non-owner account tries to redo any of them and hits a
permission error.

## Values used at deployment

- `ATLAS_ENV=production`
- `GOOGLE_CLOUD_PROJECT=all-things-agentic-hack`
- `GOOGLE_CLOUD_LOCATION=global`
- `PAPER_STORAGE_BUCKET=all-things-agentic-hack-atlas-papers`
- `CORS_ORIGINS`: the production frontend origin(s)
- `API_SHARED_SECRET`: referenced from Secret Manager secret
  `atlas-api-shared-secret`, never copied into source control
- Backend runtime identity: `hackathon-agent@all-things-agentic-hack.iam.gserviceaccount.com`

The backend must remain at one process and one service instance until the
in-memory graph/index components are synchronized across instances.
