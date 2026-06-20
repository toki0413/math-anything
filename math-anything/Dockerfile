FROM python:3.12-slim

# Rust toolchain for PyO3
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential pkg-config libssl-dev \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && . $HOME/.cargo/env \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cache layer)
COPY pyproject.toml ./
RUN pip install --no-cache-dir maturin && pip install --no-cache-dir -e .

# Copy source
COPY . .

# Build Rust extension
RUN . $HOME/.cargo/env && maturin develop --release

# Install package
RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["python", "-m", "math_anything.server"]
