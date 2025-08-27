CREATE TABLE IF NOT EXISTS config (
    config JSONB
);

CREATE TABLE IF NOT EXISTS job (
    benchmark_id TEXT PRIMARY KEY,
    settings JSONB NOT NULL,
    hyperparameters JSONB,
    num_nonces INTEGER NOT NULL,
    rand_hash TEXT NOT NULL,
    runtime_config JSONB NOT NULL,
    batch_size INTEGER NOT NULL,
    num_batches INTEGER NOT NULL,
    challenge TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    download_url TEXT NOT NULL,
    hash_threshold TEXT NOT NULL,
    average_solution_ratio FLOAT NOT NULL,
    block_started INTEGER NOT NULL,
    sampled_nonces JSONB,
    benchmark_submit_time BIGINT,
    proof_submit_time BIGINT,
    start_time BIGINT,
    end_time BIGINT,
    merkle_root_ready BOOLEAN,
    merkle_proofs_ready BOOLEAN,
    benchmark_submitted BOOLEAN,
    proof_submitted BOOLEAN,
    stopped BOOLEAN
);

CREATE INDEX idx_job_batch_size ON job(batch_size);
CREATE INDEX idx_job_block_started ON job(block_started);
CREATE INDEX idx_job_challenge ON job(challenge);
CREATE INDEX idx_job_benchmark_submit_time ON job(benchmark_submit_time);
CREATE INDEX idx_job_proof_submit_time ON job(proof_submit_time);
CREATE INDEX idx_job_merkle_root_ready ON job(merkle_root_ready);
CREATE INDEX idx_job_merkle_proofs_ready ON job(merkle_proofs_ready);
CREATE INDEX idx_job_benchmark_submitted ON job(benchmark_submitted);
CREATE INDEX idx_job_proof_submitted ON job(proof_submitted);
CREATE INDEX idx_job_stopped ON job(stopped);

CREATE TABLE IF NOT EXISTS job_data (
    benchmark_id TEXT PRIMARY KEY,
    merkle_root TEXT,
    solution_nonces JSONB,
    discarded_solution_nonces JSONB,
    merkle_proofs JSONB,

    FOREIGN KEY (benchmark_id) REFERENCES job(benchmark_id)
);

CREATE TABLE IF NOT EXISTS root_batch (
    benchmark_id TEXT,
    batch_idx INTEGER,
    slave TEXT,
    start_time BIGINT,
    end_time BIGINT,
    ready BOOLEAN,

    PRIMARY KEY (benchmark_id, batch_idx),
    FOREIGN KEY (benchmark_id) REFERENCES job(benchmark_id)
);

CREATE INDEX idx_root_batch_benchmark_id ON root_batch(benchmark_id);
CREATE INDEX idx_root_batch_batch_idx ON root_batch(batch_idx);
CREATE INDEX idx_root_batch_slave ON root_batch(slave);
CREATE INDEX idx_root_batch_start_time ON root_batch(start_time);
CREATE INDEX idx_root_batch_end_time ON root_batch(end_time);
CREATE INDEX idx_root_batch_ready ON root_batch(ready);

CREATE TABLE IF NOT EXISTS proofs_batch (
    benchmark_id TEXT REFERENCES job(benchmark_id),
    batch_idx INTEGER,
    slave TEXT,
    start_time BIGINT,
    end_time BIGINT,
    sampled_nonces JSONB,
    ready BOOLEAN,

    PRIMARY KEY (benchmark_id, batch_idx),
    FOREIGN KEY (benchmark_id, batch_idx) REFERENCES root_batch(benchmark_id, batch_idx)
);

CREATE INDEX idx_proofs_batch_benchmark_id ON proofs_batch(benchmark_id);
CREATE INDEX idx_proofs_batch_batch_idx ON proofs_batch(batch_idx);
CREATE INDEX idx_proofs_batch_slave ON proofs_batch(slave);
CREATE INDEX idx_proofs_batch_start_time ON proofs_batch(start_time);
CREATE INDEX idx_proofs_batch_end_time ON proofs_batch(end_time);
CREATE INDEX idx_proofs_batch_ready ON proofs_batch(ready);

CREATE TABLE IF NOT EXISTS batch_data (
    benchmark_id TEXT,
    batch_idx INTEGER,
    merkle_root TEXT,
    solution_nonces JSONB,
    discarded_solution_nonces JSONB,
    hashes JSONB,
    merkle_proofs JSONB,

    PRIMARY KEY (benchmark_id, batch_idx),
    FOREIGN KEY (benchmark_id, batch_idx) REFERENCES root_batch(benchmark_id, batch_idx)
);

CREATE INDEX idx_proofs_batch_data_benchmark_id ON batch_data(benchmark_id);
CREATE INDEX idx_proofs_batch_data_batch_idx ON batch_data(batch_idx);

INSERT INTO config
SELECT '
{
  "player_id": "0x0000000000000000000000000000000000000000",
  "api_key": "00000000000000000000000000000000",
  "api_url": "https://mainnet-api.tig.foundation",
  "time_between_resubmissions": 60000,
  "max_concurrent_benchmarks": 4,
  "algo_selection": [
    {
      "algorithm_id": "c001_a001",
      "num_nonces": 40,
      "difficulty_range": [0, 0.5],
      "selected_difficulties": [],
      "weight": 1,
      "batch_size": 8
    },
    {
      "algorithm_id": "c002_a001",
      "num_nonces": 40,
      "difficulty_range": [0, 0.5],
      "selected_difficulties": [],
      "weight": 1,
      "batch_size": 8
    },
    {
      "algorithm_id": "c003_a001",
      "num_nonces": 40,
      "difficulty_range": [0, 0.5],
      "selected_difficulties": [],
      "weight": 1,
      "batch_size": 8
    },
    {
      "algorithm_id": "c004_a001",
      "num_nonces": 40,
      "difficulty_range": [0, 0.5],
      "selected_difficulties": [],
      "weight": 1,
      "batch_size": 8
    },
    {
      "algorithm_id": "c005_a001",
      "num_nonces": 40,
      "difficulty_range": [0, 0.5],
      "selected_difficulties": [],
      "weight": 1,
      "batch_size": 8
    }
  ],
  "time_before_batch_retry": 60000,
  "slaves": [
    {
      "name_regex": ".*",
      "algorithm_id_regex": ".*",
      "max_concurrent_batches": 1
    }
  ]
}'
WHERE NOT EXISTS (SELECT 1 FROM config);