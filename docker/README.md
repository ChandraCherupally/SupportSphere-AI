# SupportSphere AI — Docker Configuration

This directory contains the primary Docker configuration files for **SupportSphere AI**.

## File Structure

| File | Description |
| :--- | :--- |
| [`Dockerfile`](Dockerfile) | Multi-stage build using `python:3.12-slim` and `uv` package manager for fast installs and security compliance (`appuser`). |
| [`docker-compose.yml`](docker-compose.yml) | Docker Compose specification for building and running the application stack. |
| [`.dockerignore`](.dockerignore) | Specifies pattern rules for excluding non-essential files from the Docker build context. |

---

## Usage Guide

### 1. Environment Setup

Ensure a `.env` file exists at the root of the project with required keys:

```bash
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
GROQ_API_KEY=your_groq_key
```

---

### 2. Run with Docker Compose

To build and launch SupportSphere AI:

```bash
docker compose -f docker/docker-compose.yml up --build -d
```

Access the Streamlit dashboard at **`http://localhost:8501`**.

#### Stream Logs
```bash
docker compose -f docker/docker-compose.yml logs -f supportsphere
```

#### Stop Application
```bash
docker compose -f docker/docker-compose.yml down
```

---

### 3. Run with Docker CLI Direct

#### Build Image
```bash
docker build -t supportsphere-ai:latest -f docker/Dockerfile .
```

#### Run Container
```bash
docker run -d \
  --name supportsphere-ai \
  -p 8501:8501 \
  --env-file .env \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/logs:/app/logs" \
  supportsphere-ai:latest
```
