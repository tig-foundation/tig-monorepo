CREATE TABLE IF NOT EXISTS jobs (
    benchmark_id VARCHAR PRIMARY KEY,
    settings JSONB,
    num_nonces INTEGER,
    rand_hash TEXT,
    runtime_config JSONB,
    batch_size INTEGER,
    num_batches INTEGER,
    sampled_nonces JSONB,
    solution_nonces JSONB,
    challenge VARCHAR,
    download_url TEXT,
    last_submit_time BIGINT,
    last_proof_submit_time BIGINT,
    creation_timestamp BIGINT,
    block_started INTEGER,
    merkle_proofs JSONB,
    merkle_root TEXT
);

CREATE TABLE IF NOT EXISTS roots (
    benchmark_id VARCHAR REFERENCES jobs(benchmark_id),
    batch_idx INTEGER,
    root TEXT,
    slave VARCHAR,
    start_epoch BIGINT,
    end_epoch BIGINT,
    solution_nonces JSONB,
    PRIMARY KEY (benchmark_id, batch_idx)
);

CREATE TABLE IF NOT EXISTS proofs (
    benchmark_id VARCHAR REFERENCES jobs(benchmark_id),
    batch_idx INTEGER,
    proofs JSONB,
    slave VARCHAR,
    start_epoch BIGINT,
    end_epoch BIGINT,
    sampled_nonces JSONB,
    PRIMARY KEY (benchmark_id, batch_idx)
);

CREATE TABLE IF NOT EXISTS config (
    config JSONB
);