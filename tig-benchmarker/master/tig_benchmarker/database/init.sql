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