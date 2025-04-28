ARG BASE_IMAGE=ubuntu:24.04
ARG TARGET=dev

# Base stage with common setup
FROM ${BASE_IMAGE} AS base

ARG ARCH=aarch64
ENV ARCH=${ARCH}

RUN if [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "amd64" ]; then \
        echo "Unsupported architecture: $ARCH. Must be 'aarch64' or 'amd64'." && exit 1; \
    fi

# Common setup for all images
RUN apt update && apt install -y curl

# Install Rust with specific version
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN RUST_TARGET=$(if [ "$ARCH" = "aarch64" ]; then \
        echo "aarch64-unknown-linux-gnu"; \
    else \
        echo "x86_64-unknown-linux-gnu"; \
    fi) && \
    rustup install nightly-2025-01-16 && \
    rustup default nightly-2025-01-16 && \
    rustup component add rust-src && \
    rustup target add $RUST_TARGET && \
    RUST_LIBDIR=$(rustc --print target-libdir --target=$RUST_TARGET) && \
    echo "export LD_LIBRARY_PATH=\"${RUST_LIBDIR}:\$RUST_LIBDIR\"" >> /etc/bash.bashrc && \
    echo "export RUST_TARGET=\"${RUST_TARGET}\"" >> /etc/bash.bashrc

# Development environment stage
FROM base AS dev

ARG ARCH=aarch64

RUN apt install -y build-essential zstd

RUN RUST_TARGET=$(if [ "$ARCH" = "aarch64" ]; then \
        echo "https://github.com/tig-foundation/llvm/releases/download/aarch64.rc1-19.1.7/llvm.tar.zst"; \
    else \
        echo "https://github.com/tig-foundation/llvm/releases/download/amd64-rc.1%2B19.1.7/llvm.tar.zst"; \
    fi) && \
    curl -L $RUST_TARGET -O && \
    mkdir -p /opt/llvm && \
    tar -xf llvm.tar.zst -C /opt/llvm && \
    mv /opt/llvm/llvm/build/* /opt/llvm/ && \
    rm -rf /opt/llvm/llvm llvm.tar.zst && \
    ln -s /opt/llvm/bin/* /usr/local/bin/

COPY tig-binary/scripts /usr/local/bin/
RUN chmod +x /usr/local/bin/build.sh && \
    chmod +x /usr/local/bin/add_runtime_signature.py

WORKDIR /app

# Runtime environment stage
FROM base AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y git python3 python3-pip
RUN pip3 install blake3 requests randomname --break-system-packages

COPY . /tmp/tig-monorepo
WORKDIR /tmp/tig-monorepo

RUN cargo build -r -p tig-runtime && \
    mv target/release/tig-runtime /usr/local/bin/ && \
    chmod +x /usr/local/bin/tig-runtime && \
    cargo build -r -p tig-worker && \
    mv target/release/tig-worker /usr/local/bin/ && \
    chmod +x /usr/local/bin/tig-worker && \
    mv tig-benchmarker/common /usr/local/lib/ && \
    mv tig-benchmarker/slave.py /usr/local/bin/ && \
    chmod +x /usr/local/bin/slave.py && \
    rm -rf tig-monorepo

WORKDIR /app

# Target selection stage
FROM ${TARGET} AS final