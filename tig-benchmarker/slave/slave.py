import argparse
import json
import os
import logging
import randomname
import aiohttp
import asyncio
import time
import psutil
import psutil

from tig_benchmarker.merkle_tree import MerkleTree
from tig_benchmarker.structs import MerkleProof, OutputData

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

def now():
    return int(time.time() * 1000)

async def download_wasm(session, download_url, wasm_path):
    if not os.path.exists(wasm_path):
        start = now()
        logger.info(f"downloading WASM from {download_url}")
        async with session.get(download_url) as resp:
            if resp.status != 200:
                raise Exception(f"status {resp.status} when downloading WASM: {await resp.text()}")
            with open(wasm_path, 'wb') as f:
                f.write(await resp.read())
        logger.debug(f"downloading WASM: took {now() - start}ms")
    logger.debug(f"WASM Path: {wasm_path}")

async def run_tig_worker(tig_worker_path, batch, wasm_path, num_workers, output_path):
    start = now()
    cmd = [
        tig_worker_path, "compute_batch",
        json.dumps(batch["settings"]), 
        batch["rand_hash"], 
        str(batch["start_nonce"]), 
        str(batch["num_nonces"]),
        str(batch["batch_size"]), 
        wasm_path,
        "--mem", str(batch["wasm_vm_config"]["max_memory"]),
        "--fuel", str(batch["wasm_vm_config"]["max_fuel"]),
        "--workers", str(num_workers),
        "--output", f"{output_path}/{batch['benchmark_id']}_{batch['start_nonce']}_{batch['batch_size']}",
    ]
    # if batch["sampled_nonces"]:
    #     cmd += ["--sampled", *map(str, batch["sampled_nonces"])]
    logger.info(f"computing batch: {' '.join(cmd)}")
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise Exception(f"tig-worker failed: {stderr.decode()}")
    result = json.loads(stdout.decode())
    logger.info(f"computing batch took {now() - start}ms")
    logger.debug(f"batch result: {result}")
    return result

async def register_slave(session, master_ip, master_port, slave_name):
    # Get Slave ID from disk
    slave_id = None

    # Check if slave_id.txt exists
    if os.path.exists("slave_id.txt"):
        with open("slave_id.txt", "r") as f:
            slave_id = f.read()

    if slave_id is None:
        # Register Slave with Master
        start = now()
        logger.info(f"Registering slave '{slave_name}' with master at {master_ip}:{master_port}")

        slave_data = {
            "name": slave_name,
            "num_of_cpus": psutil.cpu_count(logical=False),
            "num_of_threads": psutil.cpu_count(logical=True),
            "memory": psutil.virtual_memory().total,
        }

        async with session.post(f"http://{master_ip}:{master_port}/register-slave", json=slave_data) as resp:
            if resp.status != 200:
                raise Exception(f"status {resp.status} when registering slave: {await resp.text()}")
            data = await resp.json()
            logger.debug(f"Registering slave took {now() - start}ms")
            logger.info(f"Slave '{slave_name}' registered with ID {slave_id}")
            with open("slave_id.txt", "w") as f:
                f.write(str(data.get("id")))
    else:
        logger.info(f"Slave '{slave_name}' already registered with ID {slave_id}")

    return slave_id


async def process_batch(session, master_ip, master_port, tig_worker_path, download_wasms_folder, num_workers, batch, headers, output_path):
    try:
        batch_id = f"{batch['benchmark_id']}_{batch['start_nonce']}"
        logger.info(f"Processing batch {batch_id}: {batch}")

        # Step 1: Check if batch is already processed and call murkle proofs
        if (len(batch["sampled_nonces"]) > 0 and os.path.isdir(f"{output_path}/{batch['benchmark_id']}_{batch['start_nonce']}_{batch['batch_size']}")):
            sample_nonces = batch["sampled_nonces"]
            start_nonce = int(batch["start_nonce"])
            batch_size = int(batch["batch_size"])

            leafs = {}
            for nonce in sample_nonces:
                file_path = f"{output_path}/{batch['benchmark_id']}_{batch['start_nonce']}_{batch['batch_size']}/{nonce}.json"
                try:
                    with open(file_path) as f:
                        leafs[nonce] = OutputData.from_dict(json.load(f))
                except FileNotFoundError:
                    print(f"File not found: {file_path}")
                except json.JSONDecodeError:
                    print(f"Invalid JSON in file: {file_path}")

            merkle_tree = MerkleTree(
                [x.to_merkle_hash() for x in leafs.values()],
                batch_size
            )

            merkle_proofs = [
                MerkleProof(
                    leaf=leafs[n],
                    branch=merkle_tree.calc_merkle_branch(branch_idx=n - start_nonce)
                ).to_dict()
                for n in sample_nonces
            ]


            # Submit proofs to the server
            start = now()
            submit_url = f"http://{master_ip}:{master_port}/submit-merkle-proofs/{batch_id}"
            logger.info(f"posting merkle proofs to {submit_url}")

            async with session.post(f"{submit_url}", json={"merkle_proofs":merkle_proofs}, headers=headers) as resp:
                response_text = await resp.text() 

                logger.info(f"response text: {resp.status}")

                if resp.status != 200:
                    raise Exception(f"status {resp.status} when posting merkel proofs to master: {response_text}")
                logger.debug(f"posting merkel proofs took {now() - start} ms")

        else:
            # Step 2: Download WASM
            wasm_path = os.path.join(download_wasms_folder, f"{batch['settings']['algorithm_id']}.wasm")
            await download_wasm(session, batch['download_url'], wasm_path)

            # Step 3: Run tig-worker
            result = await run_tig_worker(tig_worker_path, batch, wasm_path, num_workers, output_path)

            # Step 4: Submit results
            start = now()
            submit_url = f"http://{master_ip}:{master_port}/submit-batch-result/{batch_id}"
            logger.info(f"posting results to {submit_url}")
            async with session.post(submit_url, json=result, headers=headers) as resp:
                response_text = await resp.text() 

                logger.info(f"response text: {resp.status}")

                if resp.status != 200:
                    raise Exception(f"status {resp.status} when posting results to master: {response_text}")
                logger.debug(f"posting results took {now() - start} ms")
               

    except Exception as e:
        logger.error(f"Error processing batch {batch_id}: {e}")


async def main(
    master_ip: str,
    tig_worker_path: str,
    download_wasms_folder: str,
    num_workers: int,
    slave_name: str,
    master_port: int,
    output_path: str
):
    if not os.path.exists(tig_worker_path):
        raise FileNotFoundError(f"tig-worker not found at path: {tig_worker_path}")
    os.makedirs(download_wasms_folder, exist_ok=True)
    
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                # Step 0: Register Slave
                slave_id = await register_slave(session, master_ip, master_port, slave_name)

                headers = {
                    "User-Agent": slave_id
                }

                # Step 1: Query for job
                start = now()
                get_batch_url = f"http://{master_ip}:{master_port}/get-batches"
                logger.info(f"Fetching job from {get_batch_url}")
                async with session.get(get_batch_url, headers=headers) as resp:
                    if resp.status != 200:
                        raise Exception(f"status {resp.status} when fetching job: {await resp.text()}")
                    logger.info(f"response text: {await resp.text()}")
                    batches = await resp.json(content_type=None)
                logger.debug(f"fetching job: took {now() - start}ms")

                # Process batches concurrently
                tasks = [
                    process_batch(session, master_ip, master_port, tig_worker_path, download_wasms_folder, num_workers, batch, headers, output_path)
                    for batch in batches
                ]
                await asyncio.gather(*tasks)

            except Exception as e:
                logger.error(e)
                await asyncio.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TIG Slave Benchmarker")
    parser.add_argument("master_ip", help="IP address of the master")
    parser.add_argument("tig_worker_path", help="Path to tig-worker executable")
    parser.add_argument("--download", type=str, default="wasms", help="Folder to download WASMs to (default: wasms)")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers (default: 8)")
    parser.add_argument("--name", type=str, default=randomname.get_name(), help="Name for the slave (default: randomly generated)")
    parser.add_argument("--port", type=int, default=5115, help="Port for master (default: 5115)")
    parser.add_argument("--verbose", action='store_true', help="Print debug logs")
    parser.add_argument('--output', type=str, default="output", help="Folder to output results to (default: output)")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        format='%(levelname)s - [%(name)s] - %(message)s',
        level=logging.DEBUG if args.verbose else logging.INFO
    )

    asyncio.run(main(args.master_ip, args.tig_worker_path, args.download, args.workers, args.name, args.port, args.output))