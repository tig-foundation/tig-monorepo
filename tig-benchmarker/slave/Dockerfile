FROM tig-benchmarker:latest

WORKDIR /app

COPY ./tig-benchmarker/slave .

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app/tig-worker-build

COPY . .
RUN cargo build --release -p tig-worker

RUN mv ./target/release/tig-worker ..

WORKDIR /app
RUN rm -rf ./tig-worker-build

CMD ["python", "slave.py", "master", "./tig-worker"] 