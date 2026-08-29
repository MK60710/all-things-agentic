# Atlas backend container. The Next.js frontend is deployed separately on Vercel.
FROM python:3.12-slim-bookworm

# Atlas uses pdftotext for PDF extraction.
RUN apt-get update \
    && apt-get install -y --no-install-recommends poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the backend's production dependencies from the plain requirements
# file so the image remains easy to understand and maintain.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY agent/ ./agent/
COPY service/ ./service/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UPLOAD_ROOT=/tmp/atlas-uploads

EXPOSE 8080

# Cloud Run supplies PORT. One worker is intentional for Atlas's current
# process-local retrieval and clarification state.
CMD ["sh", "-c", "exec uvicorn service.app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
