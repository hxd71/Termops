# syntax=docker/dockerfile:1

FROM python:3.12-slim

LABEL org.opencontainers.image.title="Termops"
LABEL org.opencontainers.image.description="Terminal-native AI agent for error analysis and remediation"
LABEL org.opencontainers.image.url="https://github.com/hxd71/Termops"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    procps \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir -e ".[dev]" && \
    pip install --no-cache-dir pytest-cov

# Runtime directories
RUN mkdir -p /root/.termops/config /root/.termops/run

# Default config
ENV TERMOPS_WEB_HOST=0.0.0.0
ENV TERMOPS_WEB_PORT=8923
ENV PYTHONUNBUFFERED=1

EXPOSE 8923

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8923/healthz || exit 1

ENTRYPOINT ["termops-agent", "--profile", "demo"]
