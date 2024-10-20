# tig-benchmarker

Python scripts that implements a master/slave Benchmarker for TIG. 

# Getting Started

1. Navigate to https://play.tig.foundation/benchmarker
2. Connect your wallet
3. Use the UI to Top-Up your Available Fee Balance by burning some TIG
4. Find your API key: 
    * Run the following command in the console: `JSON.parse(Cookies.get("account"))`
    * `address` is your Mainnet `player_id`
    * `api_key` is your Mainnet API key
5. Clone this repo
    ```
    git clone https://github.com/tig-foundation/tig-monorepo.git
    ```
6. Compile `tig-worker`
    ```
    cd tig-monorepo
    cargo build -p tig-worker --release
    ```
7. Install python requirements
    ```
    # in tig-benchmarker folder
    pip install -r requirements.txt
    ```
8. Edit config.json with your `player_id` and `api_key`
9. Run a master
    ```
    # in tig-benchmarker folder
    python3 master.py <path to config.json>
    ```
10. Connect at least 1 slave to your master
    ```
    python3 slave.py <master_ip> <path to tig-worker>
    ```
    * If running locally, your master_ip should be 0.0.0.0 or localhost
    * tig-worker binary should be located at `tig-monorepo/target/release/tig-worker`

# Benchmarking Process

1. Master submits a precommit (benchmark settings + num_nonces)
2. Once precommit is confirmed, master creates a job and generates "batches"
3. Slaves poll for batches to work on, using `tig-worker` to output the batch merkle_root and nonces for which there are solutions
4. Slaves submit batch results back to master
5. Once all batches are complete, the master calculates the top merkle root, collates solution_nonces, and submits it as a Benchmark
6. When a benchmark is confirmed, master generates priority batches corresponding to the sampled nonces
    * The master stores batch merkle_roots to minimise re-computation
7. Slaves working on a priority batch generate the merkle proof for sampled nonces
8. Slaves submit priority batch results back to master
9. Once all priority batches are complete, the master calculates the merkle proofs and submits it as a proof

# Optimising your Config

1. `difficulty_sampler_config` allows you to set the `difficulty_range` for each challenge. 
    * Every block, each challenge recalculates its `base_frontier` and `scaled_frontier`
    * The difficulties within these 2 frontiers are "sorted" into easiest to hardest (0.0 is easiest, 1.0 is hardest)
    * Benchmarkers can set the `difficulty_range` from which to sample a difficulty. Examples:
        * `[0.0, 1.0]` samples the full range of valid difficulties
        * `[0.0, 0.1]` samples the easiest 10% of valid difficulties
    * Key consideration: easier difficulties may result in more solutions given the same compute, but might not be a qualifier for long if the frontiers get harder

2. `job_manager_config` allows you to set the `batch_size` for each challenge.
    * `batch_size` is the number of nonces that are part of a batch. Must be a power of 2
    * Recommend to pick a `batch_size` for your slave with lowest `num_workers` such that it takes a few seconds to compute (e.g. 5 seconds)
    * `batch_size` shouldn't be too small, or else network latency between master and slave will affect performance
    * To support slaves with different `num_workers`, see `slave_manager_config` below

3. `precommit_manager_config` allows you to control your benchmarks:
    * `max_pending_benchmarks` is the maximum number of pending benchmarks
    * Key consideration: you want batches to always be available for your slaves, but at the same time if you submit benchmarks too slowly, it will have large delays before it will be active
    * `num_nonces` is the number of nonces to compute per benchmark. Recommend to adjust based on the logs which tells you the average number of nonces to find a solution. Example log:
        * `global qualifier difficulty stats for vehicle_routing: (#nonces: 43739782840, #solutions: 22376, avg_nonces_per_solution: 1954763)`
    * `weight` affects how likely the challenge will be picked (weight of 0 will never be picked). Recommend to adjust if the logs warns you to benchmark a specific challenge to increase your cutoff. Example log:
        * `recommend finding more solutions for challenge knapsack to avoid being cut off`

4. `slave_manager_config` allows you to control your slaves:
    * When a slave makes a request, the manager iterates through each slave config one at a time until it finds a regex match. The most specific regexes should be earlier in the list, and the more general regexes should be latter in the list.
    * `max_concurrent_batches` determines how many batches of that challenge a slave can fetch & process concurrently
    * `max_concurrent_batches` also serves as a whitelist of challenges for that slave. If you don't want a slave to benchmark a specific challenge, remove its entry from the list. Example:
        * `{"vector_search": 1}` means the slave will only be given `vector_search` batches

# Master

The master node is responsible for submitting precommits/benchmarks/proofs, generating batches for slaves to work on, and managing the overall benchmarking process.

The current implementation expects only a single instance of `master.py` per `player_id`.

The master does no benchmarking! You need to connect slaves

## Usage

```
usage: master.py [-h] [--verbose] config_path

TIG Benchmarker

positional arguments:
  config_path      Path to the configuration JSON file

options:
  -h, --help       show this help message and exit
  --verbose        Print debug logs
```

# Slave

Slave nodes poll the master for batches to work on and process them using `tig-worker`.

## Usage

```
usage: slave.py [-h] [--download DOWNLOAD] [--workers WORKERS] [--name NAME] [--port PORT] [--verbose] master_ip tig_worker_path

TIG Slave Benchmarker

positional arguments:
  master_ip            IP address of the master
  tig_worker_path      Path to tig-worker executable

options:
  -h, --help           show this help message and exit
  --download DOWNLOAD  Folder to download WASMs to
  --workers WORKERS    Number of workers (default: 8)
  --name NAME          Name for the slave (default: randomly generated)
  --port PORT          Port for master (default: 5115)
  --verbose            Print debug logs
```

# License

[End User License Agreement](../docs/agreements/end_user_license_agreement.pdf)
