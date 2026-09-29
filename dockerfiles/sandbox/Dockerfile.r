# Kosmos R Language Execution Sandbox
# Secure, isolated environment for running R code with statistical genetics packages
#
# Build: docker build -t kosmos-sandbox-r:latest -f Dockerfile.r .
# Run:   docker run --rm kosmos-sandbox-r:latest Rscript -e "print('R is working')"

FROM python:3.11-slim

LABEL maintainer="kosmos-ai"
LABEL description="Sandboxed R execution environment for Kosmos AI Scientist"
LABEL version="1.0"

# Set working directory
WORKDIR /workspace

# Set non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for scientific computing + R
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    libxml2-dev \
    libxslt1-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    # R dependencies
    r-base \
    r-base-dev \
    r-recommended \
    # For R packages that need system libs
    libv8-dev \
    libgit2-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libpng-dev \
    libtiff-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements file
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Install rpy2 for Python-R interoperability (optional fallback)
RUN pip install --no-cache-dir rpy2>=3.5.0

# Install R packages for statistical genetics (Mendelian Randomization)
# TwoSampleMR requires devtools and remotes
RUN R -e "install.packages(c('devtools', 'remotes'), repos='https://cloud.r-project.org', quiet=TRUE)" && \
    R -e "install.packages(c('dplyr', 'tidyr', 'ggplot2', 'data.table'), repos='https://cloud.r-project.org', quiet=TRUE)"

# Install TwoSampleMR from GitHub (main MR package)
RUN R -e "remotes::install_github('MRCIEU/TwoSampleMR', quiet=TRUE)" || \
    R -e "install.packages('TwoSampleMR', repos='https://cloud.r-project.org', quiet=TRUE)" || \
    echo "Warning: TwoSampleMR installation failed, continuing anyway"

# Install susieR for fine-mapping
RUN R -e "install.packages('susieR', repos='https://cloud.r-project.org', quiet=TRUE)" || \
    echo "Warning: susieR installation failed, continuing anyway"

# Install additional statistical packages
RUN R -e "install.packages(c('MendelianRandomization', 'MASS', 'survival', 'lme4'), repos='https://cloud.r-project.org', quiet=TRUE)"

# Create non-root user for code execution (security)
RUN useradd -m -u 1000 -s /bin/bash sandbox && \
    chown -R sandbox:sandbox /workspace && \
    mkdir -p /home/sandbox/.local && \
    chown -R sandbox:sandbox /home/sandbox

# Switch to non-root user
USER sandbox

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    HOME=/home/sandbox \
    PATH="/home/sandbox/.local/bin:$PATH" \
    R_LIBS_USER="/home/sandbox/R/library"

# Create R library directory for user
RUN mkdir -p /home/sandbox/R/library

# Health check to verify both Python and R work
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import pandas; import numpy; print('Python OK')" && \
        Rscript -e "library(stats); print('R OK')" || exit 1

# Default command (will be overridden by sandbox)
CMD ["python3", "--version"]
