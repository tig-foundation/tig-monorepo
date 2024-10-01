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
    "extensions": [
        <extension_name>,
        ...
        <extension_name>
    ],
    "config": {
        <configs that will be passed to an extension when instantiating it>
    }
}
```

# Extensions

Extensions are event-driven components of the system:
* Events are a tuple of `(<event_name>, **kwargs)`
* Events are collected in a queue and processed every 0.1 seconds
* If an extension has a function `on_<event_name>`, then that function will be called with the `kwargs`
* The master runs in a forever loop, processing events:
    ```python
    last_update = time.time()
    while True:
        now = time.time()
        if now - last_update > 5:
            last_update = now
            await emit('update')
        await process_events(extensions)
        await asyncio.sleep(0.1)
    ```

## data_fetcher.py

Fetches and processes data from an external API.

**Config:**
* `api_url`: TIG API url
* `player_id`: Your player_id

**Event handlers:**
* `on_update`: Fetches the latest block data from the API and processes it

**Events emitted:**
* `proof_confirmed`: When a new proof is confirmed
* `benchmark_confirmed`: When a new benchmark is confirmed
* `precommit_confirmed`: When a new precommit is confirmed
* `new_block`: When a new block is detected

## difficulty_sampler.py

Manages difficulty sampling for different challenges.

**Config:**
* `difficulty_sampler`: Configuration for the difficulty sampler, including:
  * `num_samples`: Number of samples to generate
  * `padding_factor`: Factor for padding the difficulty range
  * `decay`: Decay factor for updating weights
  * `initial_solutions_weight`: Initial weight for solutions
  * `solutions_multiplier`: Multiplier for solution weights

**Event handlers:**
* `on_new_block`: Updates the sampler with new block data
* `on_benchmark_confirmed`: Updates the sampler with confirmed benchmark data

**Events emitted:**
* `difficulty_samples`: Emits new difficulty samples for challenges

## job_manager.py

Manages benchmark jobs, including creation, updating, and pruning.

**Config:**
* `backup_folder`: Folder to store job backups
* `job_manager`: Configuration for each challenge, including:
  * `batch_size`: Size of each batch
  * `ms_delay_between_batch_retries`: Delay between batch retries in milliseconds

**Event handlers:**
* `on_new_block`: Updates job data based on new block information
* `on_precommit_confirmed`: Creates a new job for a confirmed precommit
* `on_benchmark_confirmed`: Updates job data for a confirmed benchmark
* `on_proof_confirmed`: Prunes completed jobs
* `on_update`: Checks job status and emits relevant events
* `on_batch_result`: Processes results from completed batches

**Events emitted:**
* `proof_ready`: When a proof is ready to be submitted
* `benchmark_ready`: When a benchmark is ready to be submitted
* `new_batch`: When a batch needs to be processed/retried

## precommit_manager.py

Manages the creation and submission of precommits.

**Config:**
* `player_id`: Your player_id
* `backup_folder`: Folder to store backups
* `precommit_manager`: Configuration including:
  * `max_precommits_per_block`: Maximum number of precommits to submit per block
  * `algo_selection`: Algorithm selection configuration for each challenge
    * `algorithm` name of algorithm
    * `num_nonces` number of nonces for that precommit
    * `base_fee_limit` if a challenge's base fee exceeds this limit, no precommit will be submitted
    * `weight` relative chance of picking this challenge

**Event handlers:**
* `on_new_block`: Updates challenge and algorithm mappings
* `on_difficulty_samples`: Stores new difficulty samples
* `on_submit_precommit_error`: Handles precommit submission errors
* `on_update`: Selects challenges and creates new precommits

**Events emitted:**
* `precommit_ready`: When a new precommit is ready to be submitted

## slave_manager.py

Manages the distribution of batches to slave nodes and collects results.

**Config:**
* `port`: Port number for the web server

**Event handlers:**
* `on_new_batch`: Adds new batches to the queue
* `on_new_block`: Logs the current number of batches in the queue

**Events emitted:**
* `batch_result`: When a batch result is received from a slave node

## submissions_manager.py

Manages the submission of precommits, benchmarks, and proofs to the API.

**Config:**
* `api_url`: TIG API url
* `api_key`: Your API key
* `backup_folder`: Folder to store submission backups
* `submissions_manager`: Configuration including:
  * `clear_precommits_submission_on_new_block`: Whether to clear pending precommits on new block
  * `max_retries`: Maximum number of retry attempts for failed submissions (null means retry forever)
  * `ms_delay_between_retries`: Delay between retry attempts in milliseconds

**Event handlers:**
* `on_precommit_ready`: Adds a new precommit to the pending submissions
* `on_benchmark_ready`: Adds a new benchmark to the pending submissions
* `on_proof_ready`: Adds a new proof to the pending submissions
* `on_new_block`: Clears expired or confirmed submissions
* `on_update`: Processes pending submissions and initiates the submission process

**Events emitted:**
* `submit_precommit_success`: When a precommit is successfully submitted
* `submit_benchmark_success`: When a benchmark is successfully submitted
* `submit_proof_success`: When a proof is successfully submitted
* `submit_precommit_error`: When a precommit submission fails
* `submit_benchmark_error`: When a benchmark submission fails
* `submit_proof_error`: When a proof submission fails

# License

[End User License Agreement](../docs/agreements/end_user_license_agreement.pdf)
