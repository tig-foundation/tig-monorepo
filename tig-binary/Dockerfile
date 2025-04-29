ARG DEVEL_IMAGE=ubuntu:24.04
ARG RUNTIME_IMAGE=ubuntu:24.04
ARG TARGET=devel

# Development environment stage
FROM ${DEVEL_IMAGE} AS devel

RUN ARCH=$(uname -m) && \
    if [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "arm64" ] && [ "$ARCH" != "amd64" ] && [ "$ARCH" != "x86_64" ]; then \
        echo "Unsupported architecture: $ARCH. Must be 'aarch64', 'arm64', 'amd64', or 'x86_64'." && exit 1; \
    fi

# Common setup for all images
RUN apt update && apt install -y curl

# Install Rust with specific version
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN RUST_TARGET=$(if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then \
        echo "aarch64-unknown-linux-gnu"; \
    else \
        echo "x86_64-unknown-linux-gnu"; \
    fi) && \
    rustup install nightly-2025-02-10 && \
    rustup default nightly-2025-02-10 && \
    rustup component add rust-src && \
    rustup target add $RUST_TARGET && \
    RUST_LIBDIR=$(rustc --print target-libdir --target=$RUST_TARGET) && \
    ln -s $RUST_LIBDIR /usr/local/lib/rust && \
    echo "export LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH}:/usr/local/lib/rust\"" >> /etc/bash.bashrc && \
    echo "export RUST_TARGET=\"${RUST_TARGET}\"" >> /etc/bash.bashrc

RUN apt install -y build-essential zstd python3

COPY . /tmp/tig-monorepo
WORKDIR /tmp/tig-monorepo

RUN if command -v nvcc > /dev/null 2>&1; then \
        cargo build -r -p tig-runtime --features cuda && \
        cargo build -r -p tig-worker --features cuda; \
    else \
        cargo build -r -p tig-runtime && \
        cargo build -r -p tig-worker; \
    fi && \
    mv target/release/tig-runtime /usr/local/bin/ && \
    mv target/release/tig-worker /usr/local/bin/ && \
    chmod +x /usr/local/bin/tig-runtime && \
    chmod +x /usr/local/bin/tig-worker && \
    rm -rf tig-monorepo

RUN LLVM_URL=$(if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then \
        echo "https://github.com/tig-foundation/llvm/releases/download/aarch64.rc2%2B19.1.7/llvm.tar.zst"; \
    else \
        echo "https://github.com/tig-foundation/llvm/releases/download/amd64.rc2%2B19.1.7/llvm.tar.zst"; \
    fi) && \
    curl -L $LLVM_URL -O && \
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
FROM ${RUNTIME_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive

COPY --from=devel /usr/local/bin/tig-runtime /usr/local/bin/tig-runtime
COPY --from=devel /usr/local/bin/tig-worker /usr/local/bin/tig-worker
COPY --from=devel /usr/local/lib/rust /usr/local/lib/rust

RUN chmod +x /usr/local/bin/tig-runtime && \
    chmod +x /usr/local/bin/tig-worker && \
    echo "export LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH}:/usr/local/lib/rust\"" >> /etc/bash.bashrc

RUN apt update && apt install -y python3 python3-pip
RUN pip3 install blake3 requests randomname --break-system-packages

COPY tig-benchmarker /app

WORKDIR /app

# Select target image to export
FROM ${TARGET} AS final