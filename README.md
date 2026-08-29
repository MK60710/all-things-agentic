# Atlas

Atlas is an AI research assistant that helps people read academic papers, ask paper-grounded questions, test their understanding, and see how research ideas connect.

## Features

- Upload PDF papers or add papers from arXiv.
- Read guided, section-by-section explanations.
- Ask questions using retrieved paper context and citations.
- Extract concepts, methods, models, datasets, metrics, and claims.
- Explore relationships in an interactive knowledge graph.
- Check for contradictions and possible research gaps.
- Test understanding with optional knowledge checks.

## Tech stack

- Next.js frontend
- FastAPI backend
- Gemini through the Google GenAI SDK
- Firestore for persistent research data
- Cloud Storage for uploaded papers
- Google Cloud Run for the backend
- Vercel for the frontend

## Reproducible local setup

### Prerequisites

- Python 3.11 or newer
- `uv`
- Node.js and npm
- Poppler, for PDF text extraction
- Docker, if using the container setup below
- Google Cloud CLI, if using Gemini, Firestore, or Cloud Storage locally

On macOS, install Poppler with:

```bash
brew install poppler
```

### Backend

Install the Python dependencies and copy the example environment file:

```bash
uv sync --extra dev
cp .env.example .env
```

For the full application, replace the placeholder project and model values with
your own configuration and authenticate with Google Application Default
Credentials. The application initializes Firestore and Gemini at startup, so
the running backend needs access to that Google Cloud project. Configure the
Google Cloud CLI with your own project ID:

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

Then start the API with:

```bash
uv run uvicorn service.app:app --reload --port 8000
```

The backend health endpoint is available at `http://localhost:8000/health`.

If you only want to run the tests, cloud credentials are not required.

### Frontend

In a second terminal:

```bash
cd frontend
cp .env.local.example .env.local
npm install
npm run dev
```

Open `http://localhost:3000`.

The frontend example file contains placeholders only. Do not commit `.env`,
`.env.local`, API keys, service-account files, or secret values. These values
belong in local environment files or a managed secret store, not in source
control.

### Docker backend

Docker is an alternative to installing the backend dependencies directly. The
Dockerfile packages the FastAPI service and its Poppler PDF dependency. Build
the image from the repository root:

```bash
docker build -t atlas-backend .
```

Run the backend using your local environment file:

```bash
docker run --rm --name atlas-backend \
  --env-file .env \
  -p 8000:8080 \
  atlas-backend
```

If your local configuration uses Google Cloud services, the container also
needs access to your Application Default Credentials. Mount them read-only
when starting the container:

```bash
docker run --rm --name atlas-backend \
  --env-file .env \
  -v ~/.config/gcloud:/root/.config/gcloud:ro \
  -p 8000:8080 \
  atlas-backend
```

Keep the credentials on the host. Do not copy them into the image or commit
them to the repository. The frontend can continue running with `npm run dev`
and will connect to the backend on port 8000.

## Reproducible testing

Run the backend test suite from the repository root:

```bash
uv run pytest
```

Run only the query and retrieval tests when working on paper-grounded answers:

```bash
uv run pytest tests/test_query_agent.py tests/test_retrieval.py
```

Run the frontend type check and production build:

```bash
cd frontend
npm run lint
npm run build
```

The tests use local fakes and fixtures by default, so they do not require cloud credentials or paid model calls. Tests marked `live` require explicitly configured Google Cloud credentials and are skipped unless enabled by the test configuration.

## Optional cloud-backed features

Gemini, Firestore, Firebase Authentication, and Cloud Storage require
credentials and project configuration. Use environment variables locally and
Secret Manager in production. Never put credentials, API keys, service-account
files, private project identifiers, or secret values in source control.

## Project documentation

- [`docs/GUIDE.md`](docs/GUIDE.md): user-facing feature guide
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): system architecture
