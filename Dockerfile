# ═══════════════════════════════════════════════════════════════════════════════
# ALGO TRADING BOT - PRODUCTION DOCKERFILE
# ═══════════════════════════════════════════════════════════════════════════════
#
# This Dockerfile creates a Linux environment for running NautilusTrader,
# which is not available on Windows. The image is optimized for:
#   - Small size (slim base + cache cleanup)
#   - Fast builds (dependency caching via layer ordering)
#   - Development workflow (volume mapping in docker-compose)
#
# Build:  docker build -t trading-bot .
# Run:    docker-compose up
#
# ═══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────
# STAGE 1: Base Image
# ─────────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# Metadata
LABEL maintainer="Quantitative Team <quant@trading.bot>"
LABEL description="Institutional-grade algorithmic trading bot with NautilusTrader"
LABEL version="0.1.0"

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# ─────────────────────────────────────────────────────────────────────────────────
# STAGE 2: System Dependencies
# ─────────────────────────────────────────────────────────────────────────────────
# These packages are required for:
#   - build-essential: Compiling Python C extensions (numpy, pandas, etc.)
#   - libssl-dev: SSL/TLS support for secure connections
#   - pkg-config: Finding library paths during compilation
#   - curl: Health checks and downloading files
#   - git: Installing packages from git repos (if needed)
#   - libffi-dev: Foreign function interface (required by some packages)

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    pkg-config \
    curl \
    git \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ─────────────────────────────────────────────────────────────────────────────────
# STAGE 3: Upgrade pip
# ─────────────────────────────────────────────────────────────────────────────────
RUN pip install --upgrade pip setuptools wheel

# ─────────────────────────────────────────────────────────────────────────────────
# STAGE 4: Install Python Dependencies via pip
# ─────────────────────────────────────────────────────────────────────────────────
# Install all dependencies directly with pip (simpler than Poetry in Docker)
# This ensures all packages are installed to the system Python

# Core trading dependencies
RUN pip install --no-cache-dir \
    nautilus_trader==1.221.0 \
    ccxt==4.5.35 \
    vectorbt==0.26.2 \
    arcticdb==4.5.1

# Utilities and infrastructure
RUN pip install --no-cache-dir \
    loguru==0.7.3 \
    pydantic==2.12.5 \
    pydantic-settings==2.12.0 \
    python-dotenv==1.2.1 \
    pyyaml==6.0.3 \
    click==8.3.1 \
    rich==13.9.4 \
    tqdm==4.67.2

# Data processing (versions compatible with NautilusTrader)
RUN pip install --no-cache-dir \
    polars==0.20.31 \
    protobuf==4.25.8 \
    orjson==3.11.7 \
    redis==5.3.1

# Async and networking
RUN pip install --no-cache-dir \
    httpx==0.27.2 \
    websockets==12.0 \
    uvloop==0.21.0 \
    schedule==1.2.2

# ─────────────────────────────────────────────────────────────────────────────────
# STAGE 5: Application Code
# ─────────────────────────────────────────────────────────────────────────────────
# Copy application code last (changes most frequently)

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY pyproject.toml ./

# ─────────────────────────────────────────────────────────────────────────────────
# STAGE 6: Runtime Configuration
# ─────────────────────────────────────────────────────────────────────────────────

# Create directories for data persistence
RUN mkdir -p /app/data /app/logs /app/reports

# Create non-root user for security (optional but recommended)
RUN useradd --create-home --shell /bin/bash trader \
    && chown -R trader:trader /app

# Switch to non-root user (comment out if volume permissions cause issues)
# USER trader

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import nautilus_trader; print('OK')" || exit 1

# Default command
CMD ["python", "scripts/nautilus_runner.py"]
