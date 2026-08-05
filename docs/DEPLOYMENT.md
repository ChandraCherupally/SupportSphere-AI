# 🚀 Deployment & Operations Guide — SupportSphere AI

Welcome to the **SupportSphere AI Deployment & Operations Guide**. This document explains step-by-step how to run SupportSphere AI locally using Python or Docker, and how it is deployed automatically to production on **AWS EC2** using **GitHub Actions CI/CD**.

---

## 📋 Table of Contents
1. [Overview](#1-overview)
2. [Local Setup (Python & uv)](#2-local-setup-python--uv)
3. [Containerization with Docker](#3-containerization-with-docker)
4. [Automated CI/CD Pipeline (GitHub Actions)](#4-automated-cicd-pipeline-github-actions)
5. [AWS EC2 Production Deployment](#5-aws-ec2-production-deployment)
6. [Automated Storage & Memory Maintenance](#6-automated-storage--memory-maintenance)
7. [Troubleshooting & Health Checks](#7-troubleshooting--health-checks)

---

## 1. Overview

SupportSphere AI is built as a **stateless containerized web application**. 

- **Frontend/Dashboard**: Streamlit (`src/app.py`) running on port `8501`.
- **Workflow Engine**: LangGraph 5-node state machine.
- **Vector Database**: Pinecone Cloud Serverless Index.
- **Deployment Host**: AWS EC2 Ubuntu 24.04 LTS instance (`http://52.90.165.65:8501`).
- **Automation**: GitHub Actions CI/CD pipelines trigger on every push to `main`.

---

## 2. Local Setup (Python & uv)

### Prerequisites
- **Python 3.12**
- **`uv`** package manager (recommended for 10x faster installation)
- API Keys for OpenAI, Google Gemini, Groq, or Pinecone

### Quick Steps

```bash
# 1. Clone repository
git clone https://github.com/ChandraCherupally/SupportSphere-AI.git
cd SupportSphere-AI

# 2. Create virtual environment
uv venv

# 3. Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# 4. Install all dependencies
uv sync

# 5. Create .env file with your API keys
cp .env.example .env

# 6. Run local application
uv run streamlit run src/app.py
```

Open **http://localhost:8501** in your browser.

---

## 3. Containerization with Docker

The project includes a production multi-stage [`Dockerfile`](../docker/Dockerfile) and [`docker-compose.yml`](../docker/docker-compose.yml).

### Running with Docker Compose locally

```bash
# Build image and start application container in background
docker compose up -d --build

# View real-time logs
docker compose logs -f

# Stop application
docker compose down
```

---

## 4. Automated CI/CD Pipeline (GitHub Actions)

Deployments are 100% automated using two GitHub Actions workflows in `.github/workflows/`:

```
[ Git Push to main ]
         │
         ▼
 🧪 CI Workflow (.github/workflows/ci.yml)
   ├── Sets up Python 3.12 & uv
   ├── Installs project dependencies
   └── Runs 18 automated unit tests (pytest)
         │
         ▼ (If all tests pass)
 🐳 CD Workflow (.github/workflows/cd.yml)
   ├── Builds production Docker image via Docker Buildx
   ├── Pushes image to Docker Hub (supportsphere-ai:latest)
   ├── SSHs to AWS EC2 instance (52.90.165.65)
   ├── Runs automated disk space & cache pruning
   ├── Generates production .env & docker-compose.yml from GitHub Secrets
   └── Restarts container stack gracefully (docker compose up -d)
```

### GitHub Secrets Setup
Add the following secrets under **Repository Settings -> Secrets and variables -> Actions**:

| Secret Name | Description | Example |
|---|---|---|
| `AWS_HOST` | Public IP address of AWS EC2 instance | `52.90.165.65` |
| `AWS_USERNAME` | SSH login user | `ubuntu` |
| `SSH_PRIVATE_KEY` | Contents of your AWS SSH `.pem` private key | `-----BEGIN RSA PRIVATE KEY-----...` |
| `DOCKERHUB_USERNAME` | Docker Hub username | `your-username` |
| `DOCKERHUB_TOKEN` | Docker Hub Personal Access Token | `dckr_pat_...` |
| `OPENAI_API_KEY` | OpenAI API key | `sk-proj-...` |
| `GOOGLE_API_KEY` | Google Gemini API key | `AIzaSy...` |
| `PINECONE_API_KEY` | Pinecone Vector DB API key | `pcsk_...` |

---

## 5. AWS EC2 Production Deployment

### Live Server URL
Access the live production instance at: **`http://52.90.165.65:8501`**

### Manual Deployment / Server Inspection over SSH

To connect directly to the production server:

```bash
ssh -i /path/to/supportsphere-key.pem ubuntu@52.90.165.65
```

Once connected to EC2, you can manage the application directly:

```bash
cd ~/supportsphere-ai

# Check container status
docker compose ps

# View running container logs
docker compose logs -f --tail=100

# Manually pull latest image and restart
docker compose pull && docker compose up -d
```

---

## 6. Automated Storage & Memory Maintenance

To prevent EC2 servers from running out of disk space during continuous deployments, the CD workflow automatically runs a complete cleanup sequence before every release:

```bash
# Automated CD Cleanup Suite on EC2 Host
docker system prune -af --volumes || true
docker builder prune -af || true
sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/* || true
sudo chmod -R 777 ~/supportsphere-ai/data ~/supportsphere-ai/logs || true
```

What this does:
1. **`docker system prune -af --volumes`**: Deletes all stopped containers, unused networks, dangling images, and anonymous volumes.
2. **`docker builder prune -af`**: Clears Docker's build cache.
3. **`sudo apt-get clean`**: Clears Linux package manager cache.
4. **`sudo chmod -R 777`**: Fixes host file ownership permissions for container log files.

---

## 7. Troubleshooting & Health Checks

### Application Health Endpoint
Streamlit provides an HTTP health check endpoint:
- **Endpoint**: `http://52.90.165.65:8501/_stcore/health`
- **Expected Result**: `200 OK`

### Viewing Log Files
All application logs are saved on the server in `~/supportsphere-ai/logs/`:
- `application.log`: Main application runtime events.
- `pipeline.log`: Node-by-node execution details for support tickets.
- `errors.log`: Exception stack traces and API error logs.
