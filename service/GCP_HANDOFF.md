# Google Cloud prerequisite handoff

No application deployment was performed and the Dockerfile was not modified.

## Completed in project `all-things-agentic-hack`

- Confirmed billing is enabled and Firestore Native mode exists in
  `us-central1`.
- Confirmed Vertex AI, Firestore, Cloud Storage, Firebase/Identity Toolkit,
  Pub/Sub, and Billing Budgets APIs are enabled.
- Enabled the Secret Manager API.
- Reused the existing runtime identity
  `hackathon-agent@all-things-agentic-hack.iam.gserviceaccount.com`.
- Confirmed that runtime identity has Vertex AI User and Datastore User.
- Created private bucket `all-things-agentic-hack-atlas-papers` in
  `us-central1` with uniform bucket-level access and public access prevention.
- Granted the runtime identity Storage Object User on that bucket.
- Created Secret Manager secret `atlas-api-shared-secret` with a generated
  64-character-plus value. The value was never printed or committed.
- Enabled Firestore TTL for collection group `rate_limits`, field
  `expires_at`.
- Created Pub/Sub topic `atlas-budget-alerts`.

## Owner or billing-administrator actions still required

The active account `narenram98@gmail.com` lacks the IAM permissions needed for
the following actions:

1. Grant `roles/secretmanager.secretAccessor` on
   `atlas-api-shared-secret` to the `hackathon-agent` runtime identity. The
   secret exists, but the runtime cannot consume it until this is granted.
2. Grant the Cloud Billing budget notification publisher access to
   `atlas-budget-alerts`.
3. Create a project-scoped USD 25 monthly budget on billing account
   `01B2CE-BFFB41-32D8BF`, with current-spend thresholds at 50%, 80%, and
   100%, connected to `atlas-budget-alerts`.
4. Verify Google sign-in is enabled in Firebase Authentication and add the
   final production frontend hostname to authorized domains. The current
   account received HTTP 403 when reading the Identity Toolkit project config.

These are console/IAM tasks. They do not require a code change, Dockerfile
change, or application deployment.

## Values needed when deployment is performed later

- `ATLAS_ENV=production`
- `GOOGLE_CLOUD_PROJECT=all-things-agentic-hack`
- `GOOGLE_CLOUD_LOCATION=global`
- `PAPER_STORAGE_BUCKET=all-things-agentic-hack-atlas-papers`
- `CORS_ORIGINS`: the exact final HTTPS frontend origin
- `API_SHARED_SECRET`: reference Secret Manager secret
  `atlas-api-shared-secret`; do not copy the value into source control
- Backend runtime identity: `hackathon-agent@all-things-agentic-hack.iam.gserviceaccount.com`

The backend must remain at one process and one service instance until the
in-memory graph/index components are synchronized across instances.
