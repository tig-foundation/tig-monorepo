import json
import logging
import os
import requests
from sqlalchemy import false, null
from tig_benchmarker.structs import *
from tig_benchmarker.utils import *
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

from tig_benchmarker.database.init import SessionLocal
from tig_benchmarker.database.models.index import (
    BlockModel, AlgorithmModel, WasmModel, PlayerModel, PrecommitModel,
    BenchmarkModel, ProofModel, FraudModel, ChallengeModel,
    DifficultyDataModel, precise_to_float
)

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

def _get(url: str) -> Dict[str, Any]:
    logger.debug(f"Fetching from {url}")
    resp = requests.get(url, timeout=10)  # Added timeout for robustness
    if resp.status_code == 200:
        return json.loads(resp.text)
    else:
        if resp.headers.get("Content-Type") == "text/plain":
            err_msg = f"status code {resp.status_code} from {url}: {resp.text}"
        else:
            err_msg = f"status code {resp.status_code} from {url}"
        logger.error(err_msg)
        raise Exception(err_msg)

class DataFetcher:
    def __init__(self, api_url: str, player_id: str):
        self.api_url = api_url
        self.player_id = player_id
        self.db_session = SessionLocal()

    def run(self) -> dict:
        logger.debug("fetching latest block")
        block_data = _get(f"{self.api_url}/get-block?include_data")
        block = Block.from_dict({k: v for k, v in block_data["block"].items() if k != 'data'})
        block.data = block_data["block"]["data"]
        
        # Check if block exists in DB
        existing_block = self.db_session.query(BlockModel).filter_by(id=block.id).first()
        if existing_block:
            logger.debug("No new block data found in DB")
            # Fetch data from the database
            return self._fetch_from_db(block.id)

        logger.info(f"New block detected @ height {block.details.height}, fetching data")
        urls = [
            f"{self.api_url}/get-algorithms?block_id={block.id}",
            f"{self.api_url}/get-players?player_type=benchmarker&block_id={block.id}",
            f"{self.api_url}/get-benchmarks?player_id={self.player_id}&block_id={block.id}",
            f"{self.api_url}/get-challenges?block_id={block.id}",
            f"{self.api_url}/get-fee-balance?player_id={self.player_id}&block_id={block.id}"
        ]
        
        with ThreadPoolExecutor(max_workers=4) as executor: # Defined max workers as there are 4 process to be executed in parallel.
            algorithms_data, players_data, benchmarks_data, challenges_data, fee_data = list(executor.map(_get, urls))

        # Parse algorithms and wasms
        algorithms = {a["id"]: Algorithm.from_dict(a) for a in algorithms_data.get("algorithms", [])}
        wasms = {w["algorithm_id"]: Wasm.from_dict(w) for w in algorithms_data.get("wasms", [])}


        
        # Parse player
        dummy_player = {
            "id": self.player_id,
            "details": {
                "name": self.player_id,
                "is_multisig": False
            },
            "state": {
                "total_fees_paid":  fee_data['state']['total_fees_paid'],
                "available_fee_balance": fee_data['state']['available_fee_balance']
            }
        }

        logger.info(f"Player: {dummy_player}")

        player = next((Player.from_dict(p) for p in players_data.get("players", []) if p["id"] == self.player_id), Player.from_dict(dummy_player))
        
        # Parse precommits, benchmarks, proofs, frauds
        precommits = {b["benchmark_id"]: Precommit.from_dict(b) for b in benchmarks_data.get("precommits", [])}
        benchmarks = {b["id"]: Benchmark.from_dict(b) for b in benchmarks_data.get("benchmarks", [])}
        proofs = {p["benchmark_id"]: Proof.from_dict(p) for p in benchmarks_data.get("proofs", [])}
        frauds = {f["benchmark_id"]: Fraud.from_dict(f) for f in benchmarks_data.get("frauds", [])}

        # Parse challenges
        challenges = {c["id"]: Challenge.from_dict(c) for c in challenges_data.get("challenges", [])}
        
        # Fetch difficulty data for each challenge
        difficulty_urls = [
            f"{self.api_url}/get-difficulty-data?block_id={block.id}&challenge_id={c_id}"
            for c_id in challenges
        ]
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            difficulty_responses = list(executor.map(_get, difficulty_urls))
        
        difficulty_data = {
            c_id: [DifficultyData.from_dict(d) for d in resp.get("data", [])]
            for c_id, resp in zip(challenges, difficulty_responses)
        }

        # Store all fetched data into the database.
        self._store_in_db(
            block=block,
            algorithms=algorithms,
            wasms=wasms,
            player=player,
            precommits=precommits,
            benchmarks=benchmarks,
            proofs=proofs,
            frauds=frauds,
            challenges=challenges,
            difficulty_data=difficulty_data
        )

        # Return data
        return {
            "block": block,
            "algorithms": algorithms,
            "wasms": wasms,
            "player": player,
            "precommits": precommits,
            "benchmarks": benchmarks,
            "proofs": proofs,
            "frauds": frauds,
            "challenges": challenges,
            "difficulty_data": difficulty_data
        }
    
    def _store_in_db(
        self,
        block: Block,
        algorithms: Dict[str, Algorithm],
        wasms: Dict[str, Wasm],
        player: Player,
        precommits: Dict[str, Precommit],
        benchmarks: Dict[str, Benchmark],
        proofs: Dict[str, Proof],
        frauds: Dict[str, Fraud],
        challenges: Dict[str, Challenge],
        difficulty_data: Dict[str, list]
    ):
        session = self.db_session
        try:
            # Store block
            block_model = BlockModel.from_dataclass(block)
            session.merge(block_model)
            
            # Store challenges
            for challenge in challenges.values():
                challenge_model = ChallengeModel.from_dataclass(challenge)
                session.merge(challenge_model)
                
            # Store player
            if player:
                logger.info(f"player: {PlayerModel.from_dataclass(player)}")

                existing_player = session.query(PlayerModel).filter_by(id=player.id).first()
                if existing_player:
                    existing_player.block_data = player.block_data
                    existing_player.is_multisig = player.details.is_multisig
                    existing_player.total_fees_paid = precise_to_float(player.state.total_fees_paid)
                    existing_player.available_fee_balance = precise_to_float(player.state.available_fee_balance)
                    session.merge(existing_player)
                else:
                    player_model = PlayerModel.from_dataclass(player)
                    session.merge(player_model)

            # Store algorithms
            for alg in algorithms.values():
                alg.block_data.block_id = block.id
                alg_model = AlgorithmModel.from_dataclass(alg)
                session.merge(alg_model)

            # Store wasms
            for wasm in wasms.values():
                wasm_model = WasmModel.from_dataclass(wasm)
                session.merge(wasm_model)


            # Store precommits
            for precommit in precommits.values():
                logger.info(f"Storing precommits for block {precommit}")
                precommit_model = PrecommitModel.from_dataclass(precommit)
                session.merge(precommit_model)

            # Store benchmarks
            for benchmark in benchmarks.values():
                benchmark_model = BenchmarkModel.from_dataclass(benchmark)
                session.merge(benchmark_model)

            # Store proofs
            for proof in proofs.values():
                proof_model = ProofModel.from_dataclass(proof)
                session.merge(proof_model)

            # Store frauds
            for fraud in frauds.values():
                fraud_model = FraudModel.from_dataclass(fraud)
                session.merge(fraud_model)

            # Store difficulty data
            for c_id, difficulties in difficulty_data.items():
                for difficulty in difficulties:
                    difficulty_model = DifficultyDataModel.from_dataclass(c_id, difficulty)
                    session.add(difficulty_model)

            # Commit all changes
            session.commit()
            logger.info("Data successfully stored in the database")
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing data in DB: {e}")
            raise
        finally:
            session.close()
            
    def _fetch_from_db(self, block_id: str) -> dict:
        session = self.db_session
        try:
            # Fetch block
            block_model = session.query(BlockModel).filter_by(id=block_id).first()
            if not block_model:
                logger.error(f"Block with id {block_id} not found in DB")
                raise Exception(f"Block with id {block_id} not found in DB")
            block = block_model.to_dataclass()

            # Fetch algorithms
            algorithm_models = session.query(AlgorithmModel).filter_by(block_id=block_id).all()
            algorithms = {alg.id: alg.to_dataclass() for alg in algorithm_models}

            # Fetch wasms
            wasm_models = session.query(WasmModel).filter(WasmModel.algorithm_id.in_(algorithms.keys())).all()
            wasms = {wasm.algorithm_id: wasm.to_dataclass() for wasm in wasm_models}

            # Fetch player
            player_model = session.query(PlayerModel).filter_by(id=self.player_id).first()
            player = player_model.to_dataclass() if player_model else None

            # Fetch precommits
            precommit_models = session.query(PrecommitModel).filter_by(player_id=self.player_id).all()
            precommits = {pre.benchmark_id: pre.to_dataclass() for pre in precommit_models}

            # Fetch benchmarks
            benchmark_models = session.query(BenchmarkModel).filter_by(player_id=self.player_id).all()
            benchmarks = {bench.id: bench.to_dataclass() for bench in benchmark_models}

            # Fetch proofs
            proof_models = session.query(ProofModel).filter_by(player_id=self.player_id).all()
            proofs = {proof.benchmark_id: proof.to_dataclass() for proof in proof_models}

            # Fetch frauds
            fraud_models = session.query(FraudModel).filter_by(player_id=self.player_id).all()
            frauds = {fraud.benchmark_id: fraud.to_dataclass() for fraud in fraud_models}

            # Fetch challenges
            challenge_models = session.query(ChallengeModel).filter_by(block_id=block_id).all()
            challenges = {challenge.id: challenge.to_dataclass() for challenge in challenge_models}

            # Fetch difficulty data
            difficulty_data = {}
            for challenge_id in challenges.keys():
                difficulty_models = session.query(DifficultyDataModel).filter_by(challenge_id=challenge_id).all()
                difficulty_data[challenge_id] = [dm.to_dataclass() for dm in difficulty_models]

            return {
                "block": block,
                "algorithms": algorithms,
                "wasms": wasms,
                "player": player,
                "precommits": precommits,
                "benchmarks": benchmarks,
                "proofs": proofs,
                "frauds": frauds,
                "challenges": challenges,
                "difficulty_data": difficulty_data
            }
        except Exception as e:
            logger.error(f"Error fetching data from DB: {e}")
            raise
        finally:
            session.close()
