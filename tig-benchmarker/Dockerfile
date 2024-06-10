FROM rust:1.77

WORKDIR /app
COPY . .
RUN cargo build -p tig-benchmarker --bins --release --no-default-features --features standalone
RUN cp target/release/tig-benchmarker /usr/local/bin/tig-benchmarker
RUN chmod +x /usr/local/bin/tig-benchmarker

ENTRYPOINT ["tig-benchmarker"]