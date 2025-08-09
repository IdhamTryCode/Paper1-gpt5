FROM rust:1.75-slim

# Set environment variables
ENV CARGO_NET_GIT_FETCH_WITH_CLI=true
ENV RUST_BACKTRACE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Cargo.toml and Cargo.lock for better caching
COPY Cargo.toml Cargo.lock ./

# Create a dummy main.rs to build dependencies
RUN mkdir -p src && echo "fn main() {}" > src/main.rs
RUN cargo build --release
RUN rm src/main.rs

# Copy application code
COPY src/rust/ ./src/rust/
COPY src/shared/ ./src/shared/
COPY scripts/ ./scripts/
COPY config/ ./config/

# Build all Rust benchmarks
RUN find src/rust -name "Cargo.toml" -execdir cargo build --release \;

# Create necessary directories
RUN mkdir -p /app/data /app/results

# Set permissions
RUN chmod +x scripts/*.sh

# Default command
CMD ["bash", "scripts/setup_environment.sh"] 