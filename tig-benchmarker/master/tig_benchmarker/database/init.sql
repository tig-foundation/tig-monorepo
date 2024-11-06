-- Create blocks table
CREATE TABLE IF NOT EXISTS blocks (
    id VARCHAR PRIMARY KEY,
    prev_block_id VARCHAR,
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
    data JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Create players table
CREATE TABLE IF NOT EXISTS players (
    id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    is_multisig BOOLEAN NOT NULL,
    total_fees_paid NUMERIC(38, 18),
    available_fee_balance NUMERIC(38, 18),
    block_data JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Create challenges table
CREATE TABLE IF NOT EXISTS challenges (
    id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    block_confirmed INTEGER NOT NULL,
    round_active INTEGER,
    block_id VARCHAR REFERENCES blocks(id)
);

-- Create algorithms table
CREATE TABLE IF NOT EXISTS algorithms (
    id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    player_id VARCHAR NOT NULL,
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
CREATE TABLE IF NOT EXISTS wasms (
    id SERIAL PRIMARY KEY,
    algorithm_id VARCHAR NOT NULL REFERENCES algorithms(id),
    compile_success BOOLEAN NOT NULL,
    download_url VARCHAR,
    checksum VARCHAR,
    block_confirmed INTEGER NOT NULL
);

-- Create precommits table
CREATE TABLE IF NOT EXISTS precommits (
    benchmark_id VARCHAR PRIMARY KEY,
    player_id VARCHAR NOT NULL,
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
CREATE TABLE IF NOT EXISTS benchmarks (
    id VARCHAR PRIMARY KEY,
    num_solutions INTEGER NOT NULL,
    merkle_root VARCHAR,
    block_confirmed INTEGER NOT NULL,
    sampled_nonces JSONB,
    solution_nonces JSONB,
    player_id VARCHAR NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create proofs table
CREATE TABLE IF NOT EXISTS proofs (
    benchmark_id VARCHAR PRIMARY KEY REFERENCES benchmarks(id),
    block_confirmed INTEGER,
    submission_delay INTEGER,
    merkle_proofs JSONB,
    player_id VARCHAR
);

-- Create frauds table
CREATE TABLE IF NOT EXISTS frauds (
    benchmark_id VARCHAR PRIMARY KEY REFERENCES benchmarks(id),
    block_confirmed INTEGER NOT NULL,
    allegation TEXT,
    player_id VARCHAR
);

-- Create difficulty_data table
CREATE TABLE IF NOT EXISTS difficulty_data (
    id SERIAL PRIMARY KEY,
    challenge_id VARCHAR NOT NULL REFERENCES challenges(id),
    num_solutions INTEGER NOT NULL,
    num_nonces INTEGER NOT NULL,
    difficulty JSONB NOT NULL
);

-- Create topups table
CREATE TABLE IF NOT EXISTS topups (
    id VARCHAR PRIMARY KEY,
    player_id VARCHAR NOT NULL REFERENCES players(id),
    amount NUMERIC(38, 18) NOT NULL,
    block_confirmed INTEGER NOT NULL
);

-- Create jobs table
CREATE TABLE IF NOT EXISTS jobs (
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
    last_batch_retry_time JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create slave_registry table
CREATE TABLE IF NOT EXISTS slave_registry (
    id SERIAL PRIMARY KEY,
    slave_name VARCHAR(255) UNIQUE NOT NULL,
    num_of_cpus INTEGER NOT NULL,
    num_of_threads INTEGER NOT NULL,
    memory BIGINT NOT NULL,
    registered_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create assigned_batches table
CREATE TABLE IF NOT EXISTS assigned_batches (
    id SERIAL PRIMARY KEY,
    benchmark_id VARCHAR NOT NULL REFERENCES jobs(benchmark_id),
    batch_idx INTEGER NOT NULL,
    assigned_slave INTEGER NOT NULL REFERENCES slave_registry(id),
    submitted_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    completed_timestamp TIMESTAMP,
    -- batch_result_id INTEGER REFERENCES batch_results(id),
    UNIQUE (benchmark_id, batch_idx)
);

-- Create batch_results table
CREATE TABLE IF NOT EXISTS batch_results (
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
CREATE TABLE IF NOT EXISTS precommit_requests (
    id SERIAL PRIMARY KEY,
    challenge_id VARCHAR NOT NULL REFERENCES challenges(id) ON DELETE CASCADE,
    settings JSONB NOT NULL,
    num_nonces INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create benchmark_requests table
CREATE TABLE IF NOT EXISTS benchmark_requests (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR NOT NULL REFERENCES jobs(benchmark_id) ON DELETE CASCADE,
    benchmark_id VARCHAR NOT NULL,
    merkle_root VARCHAR NOT NULL,
    solution_nonces JSONB NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create proof_requests table
CREATE TABLE IF NOT EXISTS proof_requests (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR NOT NULL REFERENCES jobs(benchmark_id) ON DELETE CASCADE,
    benchmark_id VARCHAR NOT NULL,
    merkle_proofs JSONB NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create config table
CREATE TABLE IF NOT EXISTS config (
    id INTEGER PRIMARY KEY DEFAULT 1,
    config_data JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Create indexes for foreign keys to improve query performance
CREATE INDEX IF NOT EXISTS idx_algorithms_player_id ON algorithms(player_id);
CREATE INDEX IF NOT EXISTS idx_algorithms_challenge_id ON algorithms(challenge_id);
CREATE INDEX IF NOT EXISTS idx_algorithms_block_id ON algorithms(block_id);

CREATE INDEX IF NOT EXISTS idx_precommits_player_id ON precommits(player_id);
CREATE INDEX IF NOT EXISTS idx_precommits_challenge_id ON precommits(challenge_id);
CREATE INDEX IF NOT EXISTS idx_precommits_algorithm_id ON precommits(algorithm_id);

CREATE INDEX IF NOT EXISTS idx_difficulty_data_challenge_id ON difficulty_data(challenge_id);

CREATE INDEX IF NOT EXISTS idx_proofs_player_id ON proofs(player_id);
CREATE INDEX IF NOT EXISTS idx_frauds_player_id ON frauds(player_id);

CREATE INDEX IF NOT EXISTS idx_wasms_algorithm_id ON wasms(algorithm_id);

CREATE INDEX IF NOT EXISTS idx_jobs_challenge ON jobs(challenge);

CREATE INDEX IF NOT EXISTS idx_batch_results_benchmark_id ON batch_results(benchmark_id);
CREATE INDEX IF NOT EXISTS idx_assigned_batches_benchmark_id ON assigned_batches(benchmark_id);

-- CREATE INDEX IF NOT EXISTS for faster queries on job_id
-- CREATE INDEX IF NOT EXISTS idx_precommit_requests_job_id ON precommit_requests(job_id);

-- CREATE INDEX IF NOT EXISTS for faster queries on job_id and benchmark_id
CREATE INDEX IF NOT EXISTS idx_benchmark_requests_job_id ON benchmark_requests(job_id);
CREATE INDEX IF NOT EXISTS idx_benchmark_requests_benchmark_id ON benchmark_requests(benchmark_id);

-- CREATE INDEX IF NOT EXISTS for faster queries on job_id and benchmark_id
CREATE INDEX IF NOT EXISTS idx_proof_requests_job_id ON proof_requests(job_id);
CREATE INDEX IF NOT EXISTS idx_proof_requests_benchmark_id ON proof_requests(benchmark_id);

-- CREATE INDEX IF NOT EXISTS on slave_name for faster lookups
CREATE INDEX IF NOT EXISTS idx_slave_registry_slave_name ON slave_registry(slave_name);

-- Create a trigger function to update the updated_at field on row update
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- CREATE OR REPLACE TRIGGERs
CREATE OR REPLACE TRIGGER trigger_update_players_updated_at
BEFORE UPDATE ON players
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

CREATE OR REPLACE TRIGGER trigger_update_blocks_updated_at
BEFORE UPDATE ON blocks
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

CREATE OR REPLACE TRIGGER trigger_update_config_updated_at
BEFORE UPDATE ON config
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Initialize the Config Table only if it's empty
INSERT INTO config (id, config_data)
SELECT 1, $$
{
    "player_id": "0x00469d928a6f35834705972e937070fa154f2f7f",
    "api_key": "256979aea0c51f485e9559a45f29ec74",
    "api_url": "https://testnet-api.tig.foundation",
    "difficulty_sampler_config": {
        "difficulty_ranges": {
            "satisfiability": [
                0.0,
                0.5
            ],
            "vehicle_routing": [
                0.0,
                0.5
            ],
            "knapsack": [
                0.0,
                0.5
            ],
            "vector_search": [
                0.0,
                0.5
            ]
        }
    },
    "job_manager_config": {
        "backup_folder": "jobs",
        "batch_sizes": {
            "satisfiability": 1024,
            "vehicle_routing": 1024,
            "knapsack": 1024,
            "vector_search": 1024
        }
    },
    "submissions_manager_config": {
        "time_between_retries": 60000
    },
    "precommit_manager_config": {
        "max_pending_benchmarks": 4,
        "algo_selection": {
            "satisfiability": {
                "algorithm": "schnoing",
                "num_nonces": 1000,
                "weight": 1.0,
                "base_fee_limit": "10000000000000000"
            },
            "vehicle_routing": {
                "algorithm": "clarke_wright",
                "num_nonces": 1000,
                "weight": 1.0,
                "base_fee_limit": "10000000000000000"
            },
            "knapsack": {
                "algorithm": "dynamic",
                "num_nonces": 1000,
                "weight": 1.0,
                "base_fee_limit": "10000000000000000"
            },
            "vector_search": {
                "algorithm": "optimal_ann",
                "num_nonces": 1000,
                "weight": 1.0,
                "base_fee_limit": "10000000000000000"
            }
        }
    },
    "slave_manager_config": {
        "port": 5115,
        "time_before_batch_retry": 60000,
        "num_nonces_to_sample": 0.5,
        "slaves": [
            {
                "name_regex": ".*",
                "max_concurrent_batches": {
                    "satisfiability": 1,
                    "vehicle_routing": 1,
                    "knapsack": 1,
                    "vector_search": 1
                }
            }
        ]
    }
}
$$
WHERE NOT EXISTS (SELECT 1 FROM config);
