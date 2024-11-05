-- Drop tables if they exist (for clean setup)
DROP TABLE IF EXISTS topups;
DROP TABLE IF EXISTS difficulty_data;
DROP TABLE IF EXISTS frauds;
DROP TABLE IF EXISTS proofs;
DROP TABLE IF EXISTS benchmarks;
DROP TABLE IF EXISTS precommits;
DROP TABLE IF EXISTS wasms;
DROP TABLE IF EXISTS algorithms;
DROP TABLE IF EXISTS challenges;
DROP TABLE IF EXISTS players;
DROP TABLE IF EXISTS blocks;

-- Create blocks table
CREATE TABLE blocks (
    id VARCHAR PRIMARY KEY,
    prev_block_id VARCHAR REFERENCES blocks(id),
    height INTEGER NOT NULL,
    round INTEGER NOT NULL,
    eth_block_num VARCHAR,
    fees_paid NUMERIC(38, 18),
    num_confirmed_challenges INTEGER,
    num_confirmed_algorithms INTEGER,
    num_confirmed_benchmarks INTEGER,
    num_confirmed_precommits INTEGER,
    num_confirmed_proofs INTEGER,
    num_confirmed_frauds INTEGER,
    num_confirmed_topups INTEGER,
    num_confirmed_wasms INTEGER,
    num_active_challenges INTEGER,
    num_active_algorithms INTEGER,
    num_active_benchmarks INTEGER,
    num_active_players INTEGER,
    config JSONB NOT NULL,
    data JSONB NOT NULL  
);

-- Create players table
CREATE TABLE players (
    id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    is_multisig BOOLEAN NOT NULL,
    total_fees_paid NUMERIC(38, 18),
    available_fee_balance NUMERIC(38, 18)
);

-- Create challenges table
CREATE TABLE challenges (
    id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    block_confirmed INTEGER NOT NULL,
    round_active INTEGER,
    block_id VARCHAR REFERENCES blocks(id)
);

-- Create algorithms table
CREATE TABLE algorithms (
    id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    player_id VARCHAR NOT NULL REFERENCES players(id),
    challenge_id VARCHAR NOT NULL REFERENCES challenges(id),
    tx_hash VARCHAR,
    block_confirmed INTEGER,
    round_submitted INTEGER,
    round_pushed INTEGER,
    round_merged INTEGER,
    banned BOOLEAN,
    code TEXT,
    block_id VARCHAR REFERENCES blocks(id)
);

-- Create wasms table
CREATE TABLE wasms (
    id SERIAL PRIMARY KEY,
    algorithm_id VARCHAR NOT NULL REFERENCES algorithms(id),
    compile_success BOOLEAN NOT NULL,
    download_url VARCHAR,
    checksum VARCHAR,
    block_confirmed INTEGER NOT NULL
);

-- Create precommits table
CREATE TABLE precommits (
    benchmark_id VARCHAR PRIMARY KEY,
    player_id VARCHAR NOT NULL REFERENCES players(id),
    block_id VARCHAR NOT NULL REFERENCES blocks(id),
    challenge_id VARCHAR NOT NULL REFERENCES challenges(id),
    algorithm_id VARCHAR NOT NULL REFERENCES algorithms(id),
    difficulty JSONB NOT NULL,
    block_started INTEGER NOT NULL,
    num_nonces INTEGER,
    fee_paid NUMERIC(38, 18),
    rand_hash VARCHAR,
    block_confirmed INTEGER
);

-- Create benchmarks table
CREATE TABLE benchmarks (
    id VARCHAR PRIMARY KEY,
    num_solutions INTEGER NOT NULL,
    merkle_root VARCHAR,
    block_confirmed INTEGER NOT NULL,
    sampled_nonces JSONB,
    solution_nonces JSONB,
    player_id VARCHAR NOT NULL REFERENCES players(id)
);

-- Create proofs table
CREATE TABLE proofs (
    benchmark_id VARCHAR PRIMARY KEY REFERENCES benchmarks(id),
    block_confirmed INTEGER,
    submission_delay INTEGER,
    merkle_proofs JSONB,
    player_id VARCHAR REFERENCES players(id)
);

-- Create frauds table
CREATE TABLE frauds (
    benchmark_id VARCHAR PRIMARY KEY REFERENCES benchmarks(id),
    block_confirmed INTEGER NOT NULL,
    allegation TEXT,
    player_id VARCHAR REFERENCES players(id)
);

-- Create difficulty_data table
CREATE TABLE difficulty_data (
    id SERIAL PRIMARY KEY,
    challenge_id VARCHAR NOT NULL REFERENCES challenges(id),
    num_solutions INTEGER NOT NULL,
    num_nonces INTEGER NOT NULL,
    difficulty JSONB NOT NULL
);

-- Create topups table
CREATE TABLE topups (
    id VARCHAR PRIMARY KEY,
    player_id VARCHAR NOT NULL REFERENCES players(id),
    amount NUMERIC(38, 18) NOT NULL,
    block_confirmed INTEGER NOT NULL
);

-- Create jobs table
CREATE TABLE jobs (
    benchmark_id VARCHAR PRIMARY KEY,
    settings JSONB NOT NULL,
    num_nonces INTEGER NOT NULL,
    rand_hash VARCHAR NOT NULL,
    wasm_vm_config JSONB NOT NULL,
    download_url VARCHAR NOT NULL,
    batch_size INTEGER NOT NULL,
    challenge VARCHAR NOT NULL,
    sampled_nonces JSONB,
    merkle_root VARCHAR,
    solution_nonces JSONB,
    merkle_proofs JSONB,
    batch_merkle_proofs JSONB,
    batch_merkle_roots JSONB,
    last_benchmark_submit_time INTEGER NOT NULL,
    last_proof_submit_time INTEGER NOT NULL,
    last_batch_retry_time JSONB NOT NULL
);

-- Create assigned_batches table
CREATE TABLE assigned_batches (
    id SERIAL PRIMARY KEY,
    benchmark_id VARCHAR NOT NULL REFERENCES jobs(benchmark_id),
    batch_idx INTEGER NOT NULL,
    assigned_slave VARCHAR NOT NULL,
    submitted_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    completed_timestamp TIMESTAMP,
    batch_result_id INTEGER REFERENCES batch_results(id),
    UNIQUE (benchmark_id, batch_idx)
);

-- Create batch_results table
CREATE TABLE batch_results (
    id SERIAL PRIMARY KEY,
    benchmark_id VARCHAR NOT NULL REFERENCES jobs(benchmark_id),
    start_nonce INTEGER NOT NULL,
    merkle_root VARCHAR NOT NULL,
    solution_nonces JSONB NOT NULL,
    merkle_proofs JSONB NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    assigned_batch_id INTEGER REFERENCES assigned_batches(id)
);

-- Create precommit_requests table

CREATE TABLE precommit_requests (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR NOT NULL REFERENCES jobs(benchmark_id) ON DELETE CASCADE,
    settings JSONB NOT NULL,
    num_nonces INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create benchmark_requests table

CREATE TABLE benchmark_requests (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR NOT NULL REFERENCES jobs(benchmark_id) ON DELETE CASCADE,
    benchmark_id VARCHAR NOT NULL,
    merkle_root VARCHAR NOT NULL,
    solution_nonces JSONB NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create proof_requests table

CREATE TABLE proof_requests (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR NOT NULL REFERENCES jobs(benchmark_id) ON DELETE CASCADE,
    benchmark_id VARCHAR NOT NULL,
    merkle_proofs JSONB NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create indexes for foreign keys to improve query performance
CREATE INDEX idx_algorithms_player_id ON algorithms(player_id);
CREATE INDEX idx_algorithms_challenge_id ON algorithms(challenge_id);
CREATE INDEX idx_algorithms_block_id ON algorithms(block_id);

CREATE INDEX idx_precommits_player_id ON precommits(player_id);
CREATE INDEX idx_precommits_challenge_id ON precommits(challenge_id);
CREATE INDEX idx_precommits_algorithm_id ON precommits(algorithm_id);

CREATE INDEX idx_difficulty_data_challenge_id ON difficulty_data(challenge_id);

CREATE INDEX idx_proofs_player_id ON proofs(player_id);
CREATE INDEX idx_frauds_player_id ON frauds(player_id);

CREATE INDEX idx_wasms_algorithm_id ON wasms(algorithm_id);

CREATE INDEX idx_jobs_challenge ON jobs(challenge);

CREATE INDEX idx_batch_results_benchmark_id ON batch_results(benchmark_id);
CREATE INDEX idx_assigned_batches_benchmark_id ON assigned_batches(benchmark_id);

-- Create index for faster queries on job_id
CREATE INDEX idx_precommit_requests_job_id ON precommit_requests(job_id);

-- Create index for faster queries on job_id and benchmark_id
CREATE INDEX idx_benchmark_requests_job_id ON benchmark_requests(job_id);
CREATE INDEX idx_benchmark_requests_benchmark_id ON benchmark_requests(benchmark_id);

-- Create index for faster queries on job_id and benchmark_id
CREATE INDEX idx_proof_requests_job_id ON proof_requests(job_id);
CREATE INDEX idx_proof_requests_benchmark_id ON proof_requests(benchmark_id);