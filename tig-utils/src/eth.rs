use crate::number::PreciseNumber;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Transfer {
    pub erc20: String,
    pub sender: String,
    pub receiver: String,
    pub amount: PreciseNumber,
    pub log_idx: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LinearLock {
    pub locker: String,
    pub erc20: String,
    pub owner: String,
    pub can_cancel: bool,
    pub can_transfer: bool,
    pub start_timestamp: u64,
    pub cliff_timestamp: u64,
    pub end_timestamp: u64,
    pub amount: PreciseNumber,
    pub log_idx: usize,
}

#[cfg(feature = "web3")]
mod web3_feature {
    use std::str::FromStr;

    use crate::PreciseNumber;
    use anyhow::{anyhow, Result};
    use hex::{self, ToHex};
    use web3::{contract::*, signing::*, types::*};

    pub fn recover_address_from_msg_and_sig(msg: &str, sig: &str) -> Result<String> {
        let hash_msg = hash_message(msg.as_bytes());
        let recovery = Recovery::from_raw_signature(
            hash_msg,
            hex::decode(sig.trim_start_matches("0x"))
                .map_err(|e| web3::Error::InvalidResponse(e.to_string()))?,
        )?;
        let (signature, recovery_id) = recovery.as_signature().unwrap();
        let address = recover(hash_msg.as_bytes(), &signature, recovery_id)?;
        Ok(format!("0x{}", address.encode_hex::<String>()))
    }

    pub const ERC20_TRANSFER_TOPIC: &str =
        "ddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef";

    pub async fn get_transfer(
        rpc_url: &str,
        tx_hash: &str,
        log_idx: Option<usize>,
    ) -> Result<super::Transfer> {
        let transport = web3::transports::Http::new(rpc_url)?;
        let eth = web3::Web3::new(transport).eth();

        let tx_hash = H256::from_slice(hex::decode(tx_hash.trim_start_matches("0x"))?.as_slice());
        let receipt = eth
            .transaction_receipt(tx_hash)
            .await?
            .ok_or_else(|| anyhow!("Receipt for transaction {} not found", tx_hash))?;

        if !receipt.status.is_some_and(|x| x.as_u64() == 1) {
            return Err(anyhow!("Transaction not confirmed"));
        }

        // Find the Transfer event log
        let transfer_topic = H256::from_slice(&hex::decode(ERC20_TRANSFER_TOPIC)?);
        let (log_idx, transfer_log) = receipt
            .logs
            .iter()
            .enumerate()
            .find(|(idx, log)| {
                (log_idx.is_none() || log_idx.is_some_and(|i| i == *idx))
                    && !log.topics.is_empty()
                    && log.topics[0] == transfer_topic
            })
            .ok_or_else(|| anyhow!("No ERC20 transfer event found"))?;

        if transfer_log.topics.len() != 3 {
            return Err(anyhow!("Invalid Transfer event format"));
        }

        // Extract transfer details from the event
        let erc20 = format!("0x{}", hex::encode(&transfer_log.address.as_bytes()));
        let sender = format!(
            "0x{}",
            hex::encode(&transfer_log.topics[1].as_bytes()[12..])
        );
        let receiver = format!(
            "0x{}",
            hex::encode(&transfer_log.topics[2].as_bytes()[12..])
        );
        let amount = PreciseNumber::from_hex_str(&hex::encode(&transfer_log.data.0))?;

        Ok(super::Transfer {
            erc20,
            sender,
            receiver,
            amount,
            log_idx,
        })
    }

    pub const SABLIERV2_CREATELOCKUPLINEARSTREAM_TOPIC: &str =
        "44cb432df42caa86b7ec73644ab8aec922bc44c71c98fc330addc75b88adbc7c";

    pub async fn get_linear_lock(
        rpc_url: &str,
        tx_hash: &str,
        log_idx: Option<usize>,
    ) -> Result<super::LinearLock> {
        let transport = web3::transports::Http::new(rpc_url)?;
        let eth = web3::Web3::new(transport).eth();

        let tx_hash = H256::from_slice(hex::decode(tx_hash.trim_start_matches("0x"))?.as_slice());
        let receipt = eth
            .transaction_receipt(tx_hash)
            .await?
            .ok_or_else(|| anyhow!("Receipt for transaction {} not found", tx_hash))?;

        if !receipt.status.is_some_and(|x| x.as_u64() == 1) {
            return Err(anyhow!("Transaction not confirmed"));
        }

        // Find the Transfer event log
        let linear_lock_topic =
            H256::from_slice(&hex::decode(SABLIERV2_CREATELOCKUPLINEARSTREAM_TOPIC)?);
        let (log_idx, transfer_log) = receipt
            .logs
            .iter()
            .enumerate()
            .find(|(idx, log)| {
                (log_idx.is_none() || log_idx.is_some_and(|i| i == *idx))
                    && !log.topics.is_empty()
                    && log.topics[0] == linear_lock_topic
            })
            .ok_or_else(|| anyhow!("No ERC20 transfer event found"))?;

        if transfer_log.topics.len() != 4 {
            return Err(anyhow!("Invalid CreateLinearLockStream event format"));
        }

        // Extract transfer details from the event
        let locker = format!("0x{}", hex::encode(&transfer_log.address.as_bytes()));
        let owner = format!(
            "0x{}",
            hex::encode(&transfer_log.topics[2].as_bytes()[12..])
        );
        let erc20 = format!(
            "0x{}",
            hex::encode(&transfer_log.topics[3].as_bytes()[12..])
        );
        let amount = PreciseNumber::from_hex_str(&hex::encode(&transfer_log.data.0[64..96]))?;
        let can_cancel = transfer_log.data.0[159] == 1;
        let can_transfer = transfer_log.data.0[191] == 1;
        let start_timestamp = u64::from_be_bytes(transfer_log.data.0[216..224].try_into().unwrap());
        let cliff_timestamp = u64::from_be_bytes(transfer_log.data.0[248..256].try_into().unwrap());
        let end_timestamp = u64::from_be_bytes(transfer_log.data.0[280..288].try_into().unwrap());

        Ok(super::LinearLock {
            locker,
            erc20,
            owner,
            amount,
            can_cancel,
            can_transfer,
            start_timestamp,
            cliff_timestamp,
            end_timestamp,
            log_idx,
        })
    }

    pub const GNOSIS_SAFE_ABI: &str = r#"[
        {
            "inputs": [
                {
                    "name": "_dataHash",
                    "type": "bytes32"
                },
                {
                    "name": "_signature",
                    "type": "bytes"
                }
            ],
            "name": "isValidSignature",
            "outputs": [
                {
                    "name": "",
                    "type": "bytes4"
                }
            ],
            "payable": false,
            "stateMutability": "view",
            "type": "function"
        }
    ]"#;

    pub async fn is_valid_gnosis_safe_sig(
        rpc_url: &str,
        address: &str,
        msg: &str,
        sig: &str,
    ) -> Result<()> {
        let transport = web3::transports::Http::new(rpc_url)?;
        let eth = web3::Web3::new(transport).eth();

        let gnosis_safe = Contract::from_json(
            eth.clone(),
            H160::from_str(&address)?,
            GNOSIS_SAFE_ABI.as_bytes(),
        )
        .unwrap();
        let result: Result<Vec<u8>, _> = gnosis_safe
            .query(
                "isValidSignature",
                (
                    hash_message(msg.as_bytes()),
                    hex::decode(sig.trim_start_matches("0x"))?,
                ),
                None,
                Options::default(),
                None,
            )
            .await;
        match result {
            Ok(_) => Ok(()),
            Err(e) => Err(anyhow!("Failed query isValidSignature: {:?}", e)),
        }
    }

    pub const ENS_REVERSE_RECORDS_ADDRESS: &str = "0x3671aE578E63FdF66ad4F3E12CC0c0d71Ac7510C";
    pub const ENS_REVERSE_RECORDS_ABI: &str = r#"[
        {
            "inputs": [
                {
                    "internalType": "address[]",
                    "name": "addresses",
                    "type": "address[]"
                }
            ],
            "name": "getNames",
            "outputs": [
                {
                    "internalType": "string[]",
                    "name": "r",
                    "type": "string[]"
                }
            ],
            "stateMutability": "view",
            "type": "function"
        }
    ]"#;

    pub async fn lookup_ens_name(rpc_url: &str, address: &str) -> Result<Option<String>> {
        let transport = web3::transports::Http::new(rpc_url)?;
        let eth = web3::Web3::new(transport).eth();

        let reverse_records = Contract::from_json(
            eth.clone(),
            H160::from_str(ENS_REVERSE_RECORDS_ADDRESS)?,
            ENS_REVERSE_RECORDS_ABI.as_bytes(),
        )?;

        let addresses = vec![H160::from_str(address.trim_start_matches("0x"))?];

        let names: Vec<String> = reverse_records
            .query("getNames", (addresses,), None, Options::default(), None)
            .await?;

        // Return first name or none if empty
        Ok(if !names[0].is_empty() {
            Some(names[0].clone())
        } else {
            None
        })
    }
}

#[cfg(feature = "web3")]
pub use web3_feature::*;
