const { Pool } = require('pg');
const axios = require('axios');
const AdmZip = require('adm-zip');
const queries = require('./queries');

const pool = new Pool({
  user: process.env.POSTGRES_USER,
  host: 'postgres',
  database: process.env.POSTGRES_DB,
  password: process.env.POSTGRES_PASSWORD,
  port: 5432,
});

const environment = process.env.NODE_ENV;
const S3_PUBLIC_URL = `https://files.tig.foundation/node_historic_data/${environment}`;
const API_ENDPOINT = `https://${environment}-api.tig.foundation/get-block`;
const INITIAL_HEIGHT = 241920;

let latestNodeHeight = INITIAL_HEIGHT;
let firstMessageReceived = false;
let syncInProgress = false;

const getLastProcessedHeight = async () => {
  try {
    const res = await pool.query("SELECT value FROM sync_state WHERE key = 'last_processed_height'");
    return parseInt(res.rows[0].value, 10);
  } catch (error) {
    console.error('error getting last processed height:', error);
    return INITIAL_HEIGHT;
  }
};

const setLastProcessedHeight = async (height) => {
  console.log('setting last processed height:', height);
  try {
    await pool.query("UPDATE sync_state SET value = $1 WHERE key = 'last_processed_height'", [height]);
  } catch (error) {
    console.error('error setting last processed height:', error);
  }
};

const setS3SyncHeight = async (height) => {
  try {
    await pool.query("UPDATE sync_state SET value = $1 WHERE key = 'last_processed_s3_height'", [height]);
  } catch (error) {
    console.error('error setting s3 sync height:', error);
  }
};

const getS3SyncHeight = async () => {
  try {
    const res = await pool.query("SELECT value FROM sync_state WHERE key = 'last_processed_s3_height'");
    return parseInt(res.rows[0].value, 10);
  } catch (error) {
    console.error('error getting s3 sync height:', error);
    return INITIAL_HEIGHT;
  }
};

const consumeMessage = async (messageContent) => {
  const data = JSON.parse(messageContent);

  try {
    const client = await pool.connect();
    try {
      await client.query('BEGIN');

      const blockQuery = queries.block;
      for (const row of data.block) {
        await client.query(blockQuery, [
          row.id,
          row.datetime_added,
          row.prev_block_id,
          row.height,
          row.round,
          JSON.stringify(row.config),
          row.eth_block_num,
        ]);
      }

      const blockDataQuery = queries.block_data;
      for (const row of data.block_data) {
        await client.query(blockDataQuery, [
          row.block_id,
          JSON.stringify(row.mempool_algorithm_ids),
          JSON.stringify(row.mempool_benchmark_ids),
          JSON.stringify(row.mempool_fraud_ids),
          JSON.stringify(row.mempool_proof_ids),
          JSON.stringify(row.mempool_wasm_ids),
          JSON.stringify(row.active_algorithm_ids),
          JSON.stringify(row.active_benchmark_ids),
          JSON.stringify(row.active_challenge_ids),
          JSON.stringify(row.active_player_ids),
          JSON.stringify(row.mempool_challenge_ids)
        ]);
      }

      const benchmarkQuery = queries.benchmark;
      for (const row of data.benchmark) {
        await client.query(benchmarkQuery, [
          row.id,
          row.datetime_added,
          row.player_id,
          row.block_id,
          row.challenge_id,
          row.algorithm_id,
          JSON.stringify(row.difficulty),
          row.block_started,
          row.num_solutions,
        ]);
      }

      const benchmarkStateQuery = queries.benchmark_state;
      for (const row of data.benchmark_state) {
        await client.query(benchmarkStateQuery, [
          row.benchmark_id,
          row.block_confirmed,
          JSON.stringify(row.sampled_nonces),
        ]);
      }

      const benchmarkDataQuery = queries.benchmark_data;
      for (const row of data.benchmark_data) {
        await client.query(benchmarkDataQuery, [
          row.benchmark_id,
          JSON.stringify(row.solutions_meta_data),
          JSON.stringify(row.solution_data)
        ]);
      }

      const proofQuery = queries.proof;
      for (const row of data.proof) {
        await client.query(proofQuery, [
          row.benchmark_id,
          row.datetime_added
        ]);
      }

      const proofStateQuery = queries.proof_state;
      for (const row of data.proof_state) {
        await client.query(proofStateQuery, [
          row.benchmark_id,
          row.block_confirmed,
          row.submission_delay
        ]);
      }

      const proofDataQuery = queries.proof_data;
      for (const row of data.proof_data) {
        await client.query(proofDataQuery, [
          row.benchmark_id,
          JSON.stringify(row.solutions_data)
        ]);
      }

      const fraudQuery = queries.fraud;
      for (const row of data.fraud) {
        await client.query(fraudQuery, [
          row.benchmark_id,
          row.datetime_added,
        ]);
      }

      const fraudStateQuery = queries.fraud_state;
      for (const row of data.fraud_state) {
        await client.query(fraudStateQuery, [
          row.benchmark_id,
          row.block_confirmed
        ]);
      }

      const fraudDataQuery = queries.fraud_data;
      for (const row of data.fraud_data) {
        await client.query(fraudDataQuery, [
          row.benchmark_id,
          JSON.stringify(row.allegation)
        ]);
      }

      const challengeQuery = queries.challenge;
      for (const row of data.challenge) {
        await client.query(challengeQuery, [
          row.id,
          row.datetime_added,
          row.name
        ]);
      }

      const algorithmQuery = queries.algorithm;
      for (const row of data.algorithm) {
        await client.query(algorithmQuery, [
          row.id,
          row.datetime_added,
          row.name,
          row.player_id,
          row.challenge_id,
          row.tx_hash
        ]);
      }

      const algorithmStateQuery = queries.algorithm_state;
      for (const row of data.algorithm_state) {
        await client.query(algorithmStateQuery, [
          row.algorithm_id,
          row.block_confirmed,
          row.round_submitted,
          row.round_pushed,
          row.round_merged,
          row.banned
        ]);
      }

      const algorithmDataQuery = queries.algorithm_data;
      for (const row of data.algorithm_data) {
        await client.query(algorithmDataQuery, [
          row.algorithm_id,
          JSON.stringify(row.code)
        ]);
      }

      const wasmQuery = queries.wasm;
      for (const row of data.wasm) {
        await client.query(wasmQuery, [
          row.algorithm_id,
          row.datetime_added,
          row.download_url,
          row.checksum,
          row.compile_success
        ]);
      }

      const wasmStateQuery = queries.wasm_state;
      for (const row of data.wasm_state) {
        await client.query(wasmStateQuery, [
          row.algorithm_id,
          row.block_confirmed
        ]);
      }

      const wasmDataQuery = queries.wasm_data;
      for (const row of data.wasm_data) {
        await client.query(wasmDataQuery, [
          row.algorithm_id,
          Buffer.from(row.wasm_blob, 'utf8')
        ]);
      }

      const challengeStateQuery = queries.challenge_state;
      for (const row of data.challenge_state) {
        await client.query(challengeStateQuery, [
          row.challenge_id,
          row.block_confirmed,
          row.round_submitted
        ]);
      }

      const challengeBlockDataQuery = queries.challenge_block_data;
      for (const row of data.challenge_block_data) {
        await client.query(challengeBlockDataQuery, [
          row.challenge_id,
          row.block_id,
          row.solution_signature_threshold,
          row.num_qualifiers,
          JSON.stringify(row.qualifier_difficulties),
          JSON.stringify(row.base_frontier),
          JSON.stringify(row.scaled_frontier),
          row.scaling_factor,
          JSON.stringify(row.cutoff_frontier)
        ]);
      }

      const playerBlockDataQuery = queries.player_block_data;
      for (const row of data.player_block_data) {
        await client.query(playerBlockDataQuery, [
          row.block_id,
          row.player_id,
          JSON.stringify(row.num_qualifiers_by_challenge),
          row.cutoff,
          row.imbalance,
          row.imbalance_penalty,
          row.influence,
          row.reward,
          row.round_earnings,
          row.deposit,
          row.rolling_deposit
        ]);
      }

      const algorithmBlockDataQuery = queries.algorithm_block_data;
      for (const row of data.algorithm_block_data) {
        await client.query(algorithmBlockDataQuery, [
          row.block_id,
          row.algorithm_id,
          JSON.stringify(row.num_qualifiers_by_player),
          row.adoption,
          row.merge_points,
          row.reward,
          row.round_earnings
        ]);
      }

      await client.query('COMMIT');
      // console.log('data saved to db for block', data.block_height);
    } catch (error) {
      await client.query('ROLLBACK');
      console.error(`error saving data to db for height ${data.block_height}:`, error);
    } finally {
      client.release();
    }
  } catch (error) {
    console.error('error connecting to db:', error);
  }
};

const downloadS3Data = async (startHeight, endHeight) => {
  syncInProgress = true;
  for (let height = startHeight; height <= endHeight; height++) {
    const fileName = `data_${height}.zip`;
    const url = `${S3_PUBLIC_URL}/${fileName}`;
    let retries = 0;
    const maxRetries = 5;

    while (retries < maxRetries) {
      try {
        const response = await axios.get(url, { responseType: 'arraybuffer' });
        const zip = new AdmZip(response.data);
        const zipEntries = zip.getEntries();

        const jsonEntry = zipEntries.find(entry => entry.entryName === `data_${height}.json`);
        if (jsonEntry) {
          const jsonData = JSON.parse(jsonEntry.getData().toString('utf8'));
          await consumeMessage(JSON.stringify(jsonData));
          console.log(`Downloaded and processed data for block ${height}`);
          await setLastProcessedHeight(height);
          await setS3SyncHeight(height);
          break;
        } else {
          throw new Error(`data_${height}.json not found in the ZIP file`);
        }
      } catch (error) {
        retries++;
        console.error(`error downloading and processing data for block ${height} from S3:`, error);
        if (retries < maxRetries) {
          await new Promise(resolve => setTimeout(resolve, 10000));
        }
      }
  }
  if (retries === maxRetries) {
    console.error(`Failed to download data for block ${height} after ${maxRetries} attempts`);
  }
}
  syncInProgress = false;
};

const checkAndDownloadData = async () => {
  if (syncInProgress) return; // if syncInProgress, exit
  try {
    const response = await axios.get(API_ENDPOINT);
    const latestHeight = response.data.block.details.height;

    let lastProcessedHeight = await getLastProcessedHeight();
    console.log(`Last processed height: ${lastProcessedHeight}`);
    console.log(`Latest block height returned by API: ${latestHeight}`);
    if (latestHeight > lastProcessedHeight) {
      console.log(`------------------------------------------------------------`)
      console.log(`NEW BLOCK HEIGHT DETECTED: ${latestHeight}`);
      console.log(`Downloading and processing data from block ${lastProcessedHeight + 1} to ${latestHeight}...`);
      console.log(`------------------------------------------------------------`)

      await downloadS3Data(lastProcessedHeight + 1, latestHeight);

      firstMessageReceived = true;
      syncInProgress = false;
    } else {
      console.log('No new blocks detected, skipping...');
    }
  } catch (error) {
    console.error('Error checking latest block height:', error);
  }
};

const startCheckingLatestHeight = () => {
  setInterval(async () => {
    await checkAndDownloadData();

    // after done with initial sync, keep checking every 5 secs
    if (firstMessageReceived && !syncInProgress) {
      setInterval(checkAndDownloadData, 5000);
    }
  }, 10000);
};

// Initial sync
const initialSync = async () => {
  let pgReady = false;
  while (!pgReady) {
    try {
      await pool.query('SELECT 1');
      pgReady = true;
    } catch (error) {
      console.log('Waiting for PostgreSQL to be ready...');
      await new Promise(resolve => setTimeout(resolve, 5000));
    }
  }

  const lastProcessedHeight = await getLastProcessedHeight();

  // first block processing done individually
  if (lastProcessedHeight <= INITIAL_HEIGHT) {
    await downloadS3Data(INITIAL_HEIGHT, INITIAL_HEIGHT);
    await setLastProcessedHeight(INITIAL_HEIGHT);
  }

  startCheckingLatestHeight();
};

initialSync();
