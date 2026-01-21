# BooFun Docker Image
# Multi-stage build for optimized production image

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM python:3.11-slim@sha256:89abad2fb0c3633705054018ae09caae4bd0e0febf57fb57a96fd41769c94e12 AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY pyproject.toml setup.cfg ./
COPY src/ ./src/

# Install package with all optional dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir ".[performance,visualization]"

# ============================================================================
# Stage 2: Production Image
# ============================================================================
FROM python:3.11-slim@sha256:89abad2fb0c3633705054018ae09caae4bd0e0febf57fb57a96fd41769c94e12 AS production

LABEL maintainer="Gabriel Taboada <gabtab@berkeley.edu>"
LABEL description="BooFun: Boolean Function Analysis Library"
LABEL version="0.2.0"

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source code
COPY src/ ./src/
COPY notebooks/ ./notebooks/
COPY docs/ ./docs/
COPY pyproject.toml setup.cfg ./

# Create non-root user
RUN useradd --create-home --shell /bin/bash boofun && \
    chown -R boofun:boofun /app

USER boofun

# Expose Jupyter port
EXPOSE 8888

# Default command: start Jupyter
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--notebook-dir=/app/notebooks"]

# ============================================================================
# Stage 3: Development Image
# ============================================================================
FROM python:3.11@sha256:387dc0304019d0c6e347b816377886a15a311c6a42c336ccdf262d3fa2fdbaca AS development

LABEL description="BooFun Development Environment"

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install all dependencies including dev
COPY pyproject.toml setup.cfg ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e ".[all]"

# Copy everything
COPY . .

# Expose ports for Jupyter and docs
EXPOSE 8888 8000

CMD ["/bin/bash"]

# ============================================================================
# Stage 4: Notebook Runner
# ============================================================================
FROM production AS notebook

# Install Jupyter
RUN pip install --no-cache-dir jupyter jupyterlab

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--notebook-dir=/app/notebooks"]
