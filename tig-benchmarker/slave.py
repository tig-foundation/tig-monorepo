import argparse
import json
import os
import logging
import randomname
import aiohttp
import asyncio
import subprocess
import time

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

async def run_tig_worker(tig_worker_path, batch, wasm_path, num_workers):
    start = now()
    cmd = [
        tig_worker_path, "compute_batch",
        json.dumps(batch["settings"]), 
        batch["rand_hash"], 
        str(batch["start_nonce"]), 
        str(batch["num_nonces"]),
        str(batch["batch_size"]), 
        wasm_path,
        "--mem", str(batch["runtime_config"]["max_memory"]),
        "--fuel", str(batch["runtime_config"]["max_fuel"]),
        "--workers", str(num_workers),
    ]
    if batch["sampled_nonces"]:
        cmd += ["--sampled", *map(str, batch["sampled_nonces"])]
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

async def process_batch(session, master_ip, master_port, tig_worker_path, download_wasms_folder, num_workers, batch, headers):
    try:
        batch_id = f"{batch['benchmark_id']}_{batch['start_nonce']}"
        logger.info(f"Processing batch {batch_id}: {batch}")

        # Step 2: Download WASM
        wasm_path = os.path.join(download_wasms_folder, f"{batch['settings']['algorithm_id']}.wasm")
        await download_wasm(session, batch['download_url'], wasm_path)

        # Step 3: Run tig-worker
        result = await run_tig_worker(tig_worker_path, batch, wasm_path, num_workers)

        # Step 4: Submit results
        start = now()
        submit_url = f"http://{master_ip}:{master_port}/submit-batch-result/{batch_id}"
        logger.info(f"posting results to {submit_url}")
        async with session.post(submit_url, json=result, headers=headers) as resp:
            if resp.status != 200:
                raise Exception(f"status {resp.status} when posting results to master: {await resp.text()}")
        logger.debug(f"posting results took {now() - start} ms")

    except Exception as e:
        logger.error(f"Error processing batch {batch_id}: {e}")

async def main(
    master_ip: str,
    tig_worker_path: str,
    download_wasms_folder: str,
    num_workers: int,
    slave_name: str,
    master_port: int
):
    if not os.path.exists(tig_worker_path):
        raise FileNotFoundError(f"tig-worker not found at path: {tig_worker_path}")
    os.makedirs(download_wasms_folder, exist_ok=True)

    headers = {
        "User-Agent": slave_name
    }
    
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                # Step 1: Query for job
                start = now()
                get_batch_url = f"http://{master_ip}:{master_port}/get-batches"
                logger.info(f"fetching job from {get_batch_url}")
                async with session.get(get_batch_url, headers=headers) as resp:
                    if resp.status != 200:
                        raise Exception(f"status {resp.status} when fetching job: {await resp.text()}")
                    batches = await resp.json(content_type=None)
                logger.debug(f"fetching job: took {now() - start}ms")

                # Process batches concurrently
                tasks = [
                    process_batch(session, master_ip, master_port, tig_worker_path, download_wasms_folder, num_workers, batch, headers)
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
    
    args = parser.parse_args()
    
    logging.basicConfig(
        format='%(levelname)s - [%(name)s] - %(message)s',
        level=logging.DEBUG if args.verbose else logging.INFO
    )

    asyncio.run(main(args.master_ip, args.tig_worker_path, args.download, args.workers, args.name, args.port))