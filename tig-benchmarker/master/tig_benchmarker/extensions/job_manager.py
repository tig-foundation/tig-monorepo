import os
import logging
from dataclasses import dataclass
from tig_benchmarker.merkle_tree import MerkleBranch, MerkleTree
from tig_benchmarker.structs import *
from tig_benchmarker.utils import *

from tig_benchmarker.database.init import SessionLocal
from sqlalchemy.exc import SQLAlchemyError

from tig_benchmarker.database.models.index import JobModel
from typing import Dict, List

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

@dataclass
class JobManagerConfig(FromDict):
    batch_sizes: Dict[str, int]

class JobManager:
    def __init__(self, config: JobManagerConfig):
        self.config = config
        # Initialize database session
        self.db_session = SessionLocal()
        logger.info("JobManager initialized and connected to the database.")

    def on_new_block(
        self,
        block: Block,
        precommits: Dict[str, Precommit],
        benchmarks: Dict[str, Benchmark],
        proofs: Dict[str, Proof],
        challenges: Dict[str, Challenge],
        wasms: Dict[str, Wasm],
        **kwargs
    ):
        try:
            # Create a mapping of existing jobs' benchmark_ids
            existing_jobs = self.db_session.query(JobModel).all()
            job_map = {job.benchmark_id: job for job in existing_jobs}

            # Mapping challenge IDs to their names
            challenge_id_2_name = {
                c.id: c.details.name
                for c in challenges.values()
            }

            # Create jobs from confirmed precommits
            for benchmark_id, precommit in precommits.items():
                if benchmark_id in job_map or benchmark_id in proofs:
                    continue  # Skip if job already exists or has a proof

                logger.info(f"Creating job from confirmed precommit {benchmark_id}")
                challenge_name = challenge_id_2_name.get(precommit.settings.challenge_id)

                # Retrieve the download URL from wasms
                download_url = next(
                    (w.details.download_url for w in wasms.values() if w.algorithm_id == precommit.settings.algorithm_id), 
                    None
                )

                # Instantiate a new Job dataclass
                job = Job(
                    benchmark_id=benchmark_id,
                    settings=precommit.settings,
                    num_nonces=precommit.details.num_nonces,
                    rand_hash=precommit.state.rand_hash,
                    wasm_vm_config=block.config.get("wasm_vm"),
                    batch_size=self.config.batch_sizes.get(challenge_name),
                    challenge=challenge_name,
                    download_url=download_url
                )

                # Convert to JobModel and add to the database
                job_model = JobModel.from_dataclass(job)
                self.db_session.add(job_model)
                job_map[benchmark_id] = job_model  # Update the mapping

            # Update jobs from confirmed benchmarks
            for benchmark_id, benchmark in benchmarks.items():
                if benchmark_id in proofs:
                    continue  # Skip if proof already exists

                job_model = job_map.get(benchmark_id)
                if not job_model:
                    logger.warning(f"No existing job found for benchmark_id {benchmark_id} during update.")
                    continue

                if job_model.sampled_nonces:
                    continue  # Skip if already updated

                logger.info(f"Updating job from confirmed benchmark {benchmark_id}")
                job_model.sampled_nonces = benchmark.state.sampled_nonces
                # Reset last_batch_retry_time for all batches
                job_model.last_batch_retry_time = [0] * job_model.num_batches

            # Prune jobs based on proofs and precommits
            # Identify benchmark_ids to prune
            benchmarks_with_proofs = set(proofs.keys())
            benchmarks_not_in_precommits = set(job_map.keys()) - set(precommits.keys())

            benchmarks_to_prune = benchmarks_with_proofs.union(benchmarks_not_in_precommits)

            for benchmark_id in benchmarks_to_prune:
                job_model = job_map.get(benchmark_id)
                if job_model:
                    logger.info(f"Pruning job {benchmark_id}")
                    self.db_session.delete(job_model)

            # Commit all changes
            self.db_session.commit()
            logger.info("All job creations, updates, and prunings have been committed to the database.")
        except SQLAlchemyError as e:
            self.db_session.rollback()
            logger.error(f"Database error during on_new_block: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during on_new_block: {e}")
            raise
                
    def run(self):
        try:
            # Fetch all jobs from the database
            jobs = self.db_session.query(JobModel).all()
            now = int(time.time() * 1000)

            for job_model in jobs:
                if job_model.merkle_root is not None:
                    continue  # Skip if Merkle root is already calculated

                num_batches_ready = sum(
                    root is not None for root in job_model.batch_merkle_roots
                )
                logger.info(f"Benchmark {job_model.benchmark_id}: (batches: {num_batches_ready} of {job_model.num_batches} ready, #solutions: {len(job_model.solution_nonces)})")

                if num_batches_ready != job_model.num_batches:
                    continue  # Wait until all batches are ready

                # Calculate the minimum retry time
                if job_model.last_batch_retry_time:
                    start_time = min(job_model.last_batch_retry_time)
                else:
                    start_time = now

                logger.info(f"Benchmark {job_model.benchmark_id}: ready, took {(now - start_time) / 1000} seconds")

                # Calculate Merkle root
                merkle_tree = MerkleTree(
                    job_model.batch_merkle_roots,
                    1 << (job_model.num_batches - 1).bit_length()
                )
                job_model.merkle_root = merkle_tree.calc_merkle_root()

            # Process proofs
            for job_model in jobs:
                if (
                    not job_model.sampled_nonces or
                    len(job_model.merkle_proofs) == len(job_model.sampled_nonces)
                ):
                    continue  # Skip if benchmark not confirmed or already processed

                logger.info(f"Proof {job_model.benchmark_id}: (merkle_proof: {len(job_model.batch_merkle_proofs)} of {len(job_model.sampled_nonces)} ready)")

                if (
                    len(job_model.batch_merkle_proofs) != len(job_model.sampled_nonces) or
                    any(root is None for root in job_model.batch_merkle_roots)
                ):
                    continue  # Not all proofs are ready

                logger.info(f"Proof {job_model.benchmark_id}: ready")
                depth_offset = (job_model.batch_size - 1).bit_length()
                merkle_tree = MerkleTree(
                    job_model.batch_merkle_roots, 
                    1 << (job_model.num_batches - 1).bit_length()
                )

                # Process each batch's Merkle proof
                for batch_idx, nonces in job_model.sampled_nonces_by_batch_idx.items():
                    merkle_branch = merkle_tree.calc_merkle_branch(batch_idx)
                    upper_stems = [
                        (d + depth_offset, h) 
                        for d, h in merkle_branch.stems
                    ]

                    for nonce in set(nonces):
                        proof_data = job_model.batch_merkle_proofs.get(str(nonce))
                        if proof_data:
                            merkle_proof = MerkleProof(
                                leaf=proof_data['leaf'],
                                branch=MerkleBranch(int(proof_data['branch']['stems']) + upper_stems)
                            )
                            job_model.merkle_proofs[str(nonce)] = merkle_proof.to_dict()

            # Commit all changes
            self.db_session.commit()
            logger.info("All job runs have been processed and committed to the database.")
        except SQLAlchemyError as e:
            self.db_session.rollback()
            logger.error(f"Database error during run: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during run: {e}")
            raise