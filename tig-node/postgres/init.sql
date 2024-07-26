CREATE DOMAIN UINT_256 AS NUMERIC
CHECK (VALUE >= 0 AND VALUE < 2^256)
CHECK (SCALE(VALUE) = 0);


CREATE DOMAIN UINT_32 AS NUMERIC
CHECK (VALUE >= 0 AND VALUE < 2^32)
CHECK (SCALE(VALUE) = 0);


CREATE TABLE IF NOT EXISTS sync_state (
  key TEXT NOT NULL,
  value UINT_32 NOT NULL
);


DO $$
BEGIN
INSERT INTO sync_state (key, value) VALUES ('last_processed_height', 241920);
INSERT INTO sync_state (key, value) VALUES ('last_processed_s3_height', 241920);
END $$;


CREATE TABLE IF NOT EXISTS block (
    id TEXT PRIMARY KEY,
    datetime_added TIMESTAMPTZ NOT NULL DEFAULT now(),
    prev_block_id TEXT UNIQUE NOT NULL,
    height UINT_32 UNIQUE NOT NULL,
    round UINT_32 NOT NULL,
    config JSONB NOT NULL,
    eth_block_num UINT_256
);
CREATE INDEX block_prev_block_id_index ON block USING btree (prev_block_id);
CREATE INDEX block_height_index ON block USING btree (height);
CREATE INDEX block_round_index ON block USING btree (round);

CREATE TABLE IF NOT EXISTS challenge (
    id TEXT PRIMARY KEY,
    datetime_added TIMESTAMPTZ NOT NULL DEFAULT now(),
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS algorithm (
    id TEXT PRIMARY KEY,
    datetime_added TIMESTAMPTZ NOT NULL DEFAULT now(),
    name TEXT NOT NULL UNIQUE,
    player_id TEXT NOT NULL,
    challenge_id TEXT NOT NULL,
    tx_hash TEXT NOT NULL UNIQUE,
    FOREIGN KEY (challenge_id) REFERENCES public.challenge(id)
);

CREATE FUNCTION before_block_insert() RETURNS trigger AS $before_block_insert$
    BEGIN
        IF NEW.id <> '00000000000000000000000000000000' THEN
            PERFORM 1 FROM block WHERE id = NEW.prev_block_id;
            IF NOT FOUND THEN
                RAISE EXCEPTION 'Foreign key violation: no block id matches prev_block_id %', NEW.prev_block_id;
            END IF;
        END IF;
        IF NEW.id <> '00000000000000000000000000000000' AND NEW.id = NEW.prev_block_id THEN
            RAISE EXCEPTION 'block id cannot be same as prev_block_id %', NEW.prev_block_id;
        END IF;
        RETURN NEW;
    END;
$before_block_insert$ LANGUAGE plpgsql;


CREATE TRIGGER before_block_insert BEFORE INSERT ON block
    FOR EACH ROW EXECUTE FUNCTION before_block_insert();


CREATE TABLE IF NOT EXISTS block_data (
  block_id TEXT PRIMARY KEY,
  mempool_algorithm_ids JSONB NOT NULL,
  mempool_benchmark_ids JSONB NOT NULL,
  mempool_challenge_ids JSONB NOT NULL,
  mempool_fraud_ids JSONB NOT NULL,
  mempool_proof_ids JSONB NOT NULL,
  mempool_wasm_ids JSONB NOT NULL,
  active_algorithm_ids JSONB NOT NULL,
  active_benchmark_ids JSONB NOT NULL,
  active_challenge_ids JSONB NOT NULL,
  active_player_ids JSONB NOT NULL,
  FOREIGN KEY (block_id) REFERENCES public.block(id)
);


CREATE TABLE IF NOT EXISTS benchmark (
    id TEXT PRIMARY KEY,
    datetime_added TIMESTAMPTZ NOT NULL DEFAULT now(),
    player_id TEXT NOT NULL,
    block_id TEXT NOT NULL,
    challenge_id TEXT NOT NULL,
    algorithm_id TEXT NOT NULL,
    difficulty JSONB NOT NULL,
    block_started UINT_32 NOT NULL,
    num_solutions UINT_32 NOT NULL,
    FOREIGN KEY (block_id) REFERENCES public.block(id),
    FOREIGN KEY (algorithm_id) REFERENCES public.algorithm(id),
    FOREIGN KEY (challenge_id) REFERENCES public.challenge(id),
    CONSTRAINT unique_settings UNIQUE (player_id, block_id, challenge_id, algorithm_id, difficulty)
);
CREATE INDEX benchmark_block_id_index ON public.benchmark USING btree (block_id);
CREATE INDEX benchmark_challenge_id_index ON public.benchmark USING btree (challenge_id);
CREATE INDEX benchmark_algorithm_id_index ON public.benchmark USING btree (algorithm_id);
CREATE INDEX benchmark_player_id_index ON public.benchmark USING btree (player_id);
CREATE INDEX benchmark_block_started_index ON public.benchmark USING btree (block_started);


CREATE TABLE IF NOT EXISTS benchmark_state (
    benchmark_id TEXT PRIMARY KEY,
    block_confirmed UINT_32 NOT NULL,
    sampled_nonces JSONB NOT NULL,
    FOREIGN KEY (benchmark_id) REFERENCES public.benchmark(id)
);


CREATE TABLE IF NOT EXISTS benchmark_data (
    benchmark_id TEXT PRIMARY KEY,
    solutions_meta_data JSONB NOT NULL,
    solution_data JSONB NOT NULL,
    FOREIGN KEY (benchmark_id) REFERENCES public.benchmark(id)
);


CREATE TABLE IF NOT EXISTS proof (
    benchmark_id TEXT PRIMARY KEY,
    datetime_added TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (benchmark_id) REFERENCES public.benchmark(id)
);


CREATE TABLE IF NOT EXISTS proof_state (
    benchmark_id TEXT PRIMARY KEY,
    block_confirmed UINT_32 NOT NULL,
    submission_delay UINT_32 NOT NULL,
    FOREIGN KEY (benchmark_id) REFERENCES public.proof(benchmark_id)
);


CREATE TABLE IF NOT EXISTS proof_data (
    benchmark_id TEXT PRIMARY KEY,
    solutions_data JSONB NOT NULL,
    FOREIGN KEY (benchmark_id) REFERENCES public.proof(benchmark_id)
);


CREATE TABLE IF NOT EXISTS fraud (
    benchmark_id TEXT PRIMARY KEY,
    datetime_added TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (benchmark_id) REFERENCES public.benchmark(id)
);


CREATE TABLE IF NOT EXISTS fraud_state (
    benchmark_id TEXT PRIMARY KEY,
    block_confirmed UINT_32 NOT NULL,
    FOREIGN KEY (benchmark_id) REFERENCES public.fraud(benchmark_id)
);


CREATE TABLE IF NOT EXISTS fraud_data (
    benchmark_id TEXT PRIMARY KEY,
    allegation TEXT NOT NULL,
    FOREIGN KEY (benchmark_id) REFERENCES public.proof(benchmark_id)
);

CREATE INDEX algorithm_name_index ON public.algorithm USING btree (name);
CREATE INDEX algorithm_player_id_index ON public.algorithm USING btree (player_id);
CREATE INDEX algorithm_challenge_id_index ON public.algorithm USING btree (challenge_id);
CREATE INDEX algorithm_tx_hash_index ON public.algorithm USING btree (tx_hash);


CREATE TABLE IF NOT EXISTS algorithm_state (
    algorithm_id TEXT PRIMARY KEY,
    block_confirmed UINT_32 NOT NULL,
    round_submitted UINT_32 NOT NULL,
    round_pushed UINT_32,
    round_merged UINT_32,
    banned TEXT,
    FOREIGN KEY (algorithm_id) REFERENCES public.algorithm(id)
);


CREATE TABLE IF NOT EXISTS algorithm_data (
    algorithm_id TEXT PRIMARY KEY,
    code TEXT NOT NULL,
    FOREIGN KEY (algorithm_id) REFERENCES public.algorithm(id)
);


CREATE TABLE IF NOT EXISTS algorithm_block_data (
    algorithm_id TEXT NOT NULL,
    block_id TEXT NOT NULL,
    num_qualifiers_by_player JSONB,
    adoption UINT_256,
    merge_points UINT_32,
    reward UINT_256,
    round_earnings UINT_256 NOT NULL,
    PRIMARY KEY (algorithm_id, block_id),
    FOREIGN KEY (algorithm_id) REFERENCES public.algorithm(id),
    FOREIGN KEY (block_id) REFERENCES public.block(id)
);
CREATE INDEX algorithm_block_data_algorithm_id_index ON public.algorithm_block_data USING btree (algorithm_id);
CREATE INDEX algorithm_block_data_block_id_index ON public.algorithm_block_data USING btree (block_id);


CREATE TABLE IF NOT EXISTS wasm (
    algorithm_id TEXT PRIMARY KEY,
    datetime_added TIMESTAMPTZ NOT NULL DEFAULT now(),
    download_url TEXT,
    checksum TEXT,
    compile_success BOOLEAN NOT NULL,
    FOREIGN KEY (algorithm_id) REFERENCES public.algorithm(id)
);

CREATE TABLE IF NOT EXISTS wasm_data (
    algorithm_id TEXT PRIMARY KEY,
    wasm_blob BYTEA NOT NULL,
    FOREIGN KEY (algorithm_id) REFERENCES public.wasm(algorithm_id)
);

CREATE TABLE IF NOT EXISTS wasm_state (
    algorithm_id TEXT PRIMARY KEY,
    block_confirmed UINT_32 NOT NULL,
    FOREIGN KEY (algorithm_id) REFERENCES public.wasm(algorithm_id)
);


CREATE TABLE IF NOT EXISTS challenge_state (
    challenge_id TEXT PRIMARY KEY,
    block_confirmed UINT_32 NOT NULL,
    round_active UINT_32,
    FOREIGN KEY (challenge_id) REFERENCES public.challenge(id)
);


CREATE TABLE IF NOT EXISTS challenge_block_data (
    challenge_id TEXT NOT NULL,
    block_id TEXT NOT NULL,
    solution_signature_threshold UINT_32 NOT NULL,
    num_qualifiers UINT_32 NOT NULL,
    qualifier_difficulties JSONB NOT NULL,
    base_frontier JSONB NOT NULL,
    cutoff_frontier JSONB,
    scaled_frontier JSONB NOT NULL,
    scaling_factor DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (challenge_id, block_id),
    FOREIGN KEY (challenge_id) REFERENCES public.challenge(id),
    FOREIGN KEY (block_id) REFERENCES public.block(id)
);
CREATE INDEX challenge_block_data_challenge_id_index ON public.challenge_block_data USING btree (challenge_id);
CREATE INDEX challenge_block_data_block_id_index ON public.challenge_block_data USING btree (block_id);


CREATE TABLE IF NOT EXISTS player_block_data (
    player_id TEXT NOT NULL,
    block_id TEXT NOT NULL,
    num_qualifiers_by_challenge JSONB,
    cutoff UINT_32,
    imbalance UINT_256,
    imbalance_penalty UINT_256,
    influence UINT_256,
    reward UINT_256,
    round_earnings UINT_256 NOT NULL,
    deposit NUMERIC,
    rolling_deposit NUMERIC,
    PRIMARY KEY (player_id, block_id),
    FOREIGN KEY (block_id) REFERENCES public.block(id)
);
CREATE INDEX player_block_data_player_id_index ON public.player_block_data USING btree (player_id);
CREATE INDEX player_block_data_block_id_index ON public.player_block_data USING btree (block_id);
