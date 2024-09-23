# tig-benchmarker

Python scripts that implements a master/slave Benchmarker for TIG. 

# Setting Up

```
# install python requirements
pip install -r requirements.txt
# compile tig-worker
cargo build -p tig-worker --release
```

# Master

The master node is responsible for submitting precommits/benchmarks/proofs, generating batches for slaves to work on, and managing the overall benchmarking process.

The current implementation expects only a single instance of `master.py` per `player_id`.

## Usage

```
usage: python master.py [-h] [--port PORT] [--api API] player_id api_key config_path jobs_folder

TIG Benchmarker

positional arguments:
  player_id    Player ID
  api_key      API Key
  config_path  Path to the configuration file
  jobs_folder  Folder to job jobs until their proofs are submitted

options:
  -h, --help   show this help message and exit
  --port PORT  Port to run the server on (default: 5115)
  --api API    API URL (default: https://mainnet-api.tig.foundation)
```

# Slave

Slave nodes poll the master for batches to work on and process them using `tig-worker`.

## Usage

```
usage: python slave.py [-h] [--workers WORKERS] [--name NAME] [--port PORT] master_ip tig_worker_path tig_algorithms_folder

TIG Slave Benchmarker

positional arguments:
  master_ip             IP address of the master
  tig_worker_path       Path to tig-worker executable
  tig_algorithms_folder
                        Path to tig-algorithms folder. Used to save WASMs

options:
  -h, --help            show this help message and exit
  --workers WORKERS     Number of workers (default: 8)
  --name NAME           Name for the slave (default: randomly generated)
  --port PORT           Port for master (default: 5115)
```

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

# Configuration

The config file (JSON format) should contain the following fields:

```
{
    "max_precommits_per_block": 1,
    "satisfiability": {
        "algorithm": "schnoing",
        "num_nonces": 100000,
        "batch_size": 1024,
        "duration_before_batch_retry": 30000,
        "base_fee_limit": "10000000000000000",
        "weight": 1
    },
    "vehicle_routing": {
        "algorithm": "clarke_wright",
        "num_nonces": 100000,
        "batch_size": 1024,
        "duration_before_batch_retry": 30000,
        "base_fee_limit": "10000000000000000",
        "weight": 1
    },
    "knapsack": {
        "algorithm": "dynamic",
        "num_nonces": 100000,
        "batch_size": 1024,
        "duration_before_batch_retry": 30000,
        "base_fee_limit": "10000000000000000",
        "weight": 1
    },
    "vector_search": {
        "algorithm": "basic",
        "num_nonces": 1000,
        "batch_size": 128,
        "duration_before_batch_retry": 30000,
        "base_fee_limit": "10000000000000000",
        "weight": 0
    }
}
```

* `max_precommits_per_block`: Number of precommits submitted each block
* For each challenge type:
    * `algorithm`: Algorithm to use
    * `num_nonces`: Number of nonces for a precommit
    * `batch_size`: Must be a power of 2
    * `duration_before_batch_retry`: Time in milliseconds before a batch is requeued
    * `base_fee_limit`: Maximum amount of TIG willing to pay for a precommit
    * `weight`: Used in weighted sampling of a challenge for a precommit

**Important:**

Use `tig-worker compute_batch` to determine an appropiate `num_nonces` and `batch_size`.
* It is recommended that each batch takes around 5-10 seconds max. 
* You should target a specific duration for your benchmark
* Example: if you got `2` slaves, and each slave processes a batch of `1000` in `5s`, setting num_nonces to 10,000 should take `10000 / (1000 x 2) x 5 = 25s`

# Finding your API Key

## Mainnet

1. Navigate to https://play.tig.foundation/
2. Connect your wallet
3. Run the following command in the console: `JSON.parse(Cookies.get("account"))`
    * `address` is your Mainnet `player_id`
    * `api_key` is your Mainnet API key

## Testnet

1. Navigate to https://test.tig.foundation/
2. Connect your wallet
3. Run the following command in the console: `JSON.parse(Cookies.get("account"))`
    * `address` is your Testnet `player_id`
    * `api_key` is your Testnet API key

# License

[End User License Agreement](../docs/agreements/end_user_license_agreement.pdf)
