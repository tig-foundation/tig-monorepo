-- Create config table
CREATE TABLE IF NOT EXISTS config (
    id INTEGER PRIMARY KEY DEFAULT 1,
    config_data JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Create slaves table
CREATE TABLE IF NOT EXISTS slaves (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    num_of_cpus INTEGER NOT NULL,
    num_of_threads INTEGER NOT NULL,
    memory BIGINT NOT NULL,
    registered_at TIMESTAMP NOT NULL DEFAULT NOW()
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

-- TODO: Add batch Table
-- -- Create batches table
CREATE TABLE IF NOT EXISTS batches (
    id SERIAL PRIMARY KEY,
    benchmark_id VARCHAR NOT NULL REFERENCES jobs(benchmark_id) ON DELETE CASCADE,
    slave_id INTEGER NOT NULL REFERENCES slaves(id),
    start_nonce INTEGER NOT NULL,
    num_nonces INTEGER NOT NULL,
    settings JSONB NOT NULL,
    wasm_vm_config JSONB NOT NULL,
    download_url VARCHAR NOT NULL,
    rand_hash VARCHAR NOT NULL,
    batch_size INTEGER NOT NULL,
    sampled_nonces JSONB,
    solution_nonces JSONB,
    merkle_root VARCHAR,
    merkle_proofs JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create a trigger function to update the updated_at field on row update
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- CREATE OR REPLACE TRIGGERs
CREATE OR REPLACE TRIGGER trigger_update_batches_updated_at 
BEFORE UPDATE ON batches 
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
        "max_pending_benchmarks": 1,
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
