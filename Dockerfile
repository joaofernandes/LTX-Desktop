FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12, git, ffmpeg
RUN apt-get update && apt-get install -y \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    git curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy backend
COPY backend/ ./backend/

# Install Python dependencies (uses uv with lockfile for reproducible builds)
# Note: gguf is required for GGUF transformer loading
RUN cd backend && uv sync --no-dev && uv pip install gguf

# Copy pre-built frontend (run `pnpm build:web` first)
COPY dist/ ./dist/

EXPOSE 8001

CMD ["uv", "run", "--project", "backend", "python", "ltx2_server.py"]
