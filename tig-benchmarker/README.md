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
8. Run a master
    ```
    # in tig-benchmarker folder
    python3 master.py <player_id> <api_key> <path to config.json>
    ```
9. Connect at least 1 slave to your master
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

1. Your master should generate enough batches for your slaves to be always busy. But at the same time, you should minimise the delay between your precommit and proof submissions. Observe the elapsed ms of your benchmarks and adjust parameters accordingly to target a particular duration per benchmark (e.g. 15s)
    * `max_unresolved_precommits` - how many precommits can be in progress
    * `num_nonces` - number of nonces for a precommit for a particular challenge
    * Increasing/decreasing above will lead to more/less batches in your backlog

2. You want your slaves to be at 100% CPU utilization, but at the same time, batches should be "cheap" to repeat if a slave has issues. Observe the elapsed ms of batches and adjust parameters accordingly to target a particular duration per batch (e.g. 2s)
    * `batch_size` - how many nonces are processed in a batch for a particular challenge. Must be a power of 2

3. Slaves on different compute may be better suited for certain challenges. You can define the challenge selection for a slave based on name patterns (patterns are checked in order):
    * Any slave, any challenge (this should be the last entry)
    ```
    {
        "name_regex": ".*",
        "challenge_selection": null
    }
    ```
    * Example: only vector_search
    ```
    {
        "name_regex": "vector_search-slave-.*",
        "challenge_selection": ["vector_search"]
    }
    ```

4. By default, the benchmarker uses smart challenge selection to maximise your cutoff and to minimise your imbalance. However, if you want to control the chance of a particular challenge being selected, you can set the weight in the algo_selection:
    * Example:
    ```
    "vehicle_routing": {
        "algorithm": "clarke_wright",
        "num_nonces": 1000,
        "base_fee_limit": "10000000000000000",
        "weight": 1.0 <--- weight is relative to other challenges. If other challenge's weights are null/undefined, then this challenge will always be picked
    }
    ```

# Master

The master node is responsible for submitting precommits/benchmarks/proofs, generating batches for slaves to work on, and managing the overall benchmarking process.

The current implementation expects only a single instance of `master.py` per `player_id`.

The master does no benchmarking! You need to connect slaves

## Usage

```
usage: master.py [-h] [--api API] [--backup BACKUP] [--port PORT] [--verbose] player_id api_key config_path

TIG Benchmarker

positional arguments:
  player_id        Player ID
  api_key          API Key
  config_path      Path to the configuration JSON file

options:
  -h, --help       show this help message and exit
  --api API        API URL (default: https://mainnet-api.tig.foundation)
  --backup BACKUP  Folder to save pending submissions and other data
  --port PORT      Port to run the server on (default: 5115)
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
