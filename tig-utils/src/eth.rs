use crate::number::PreciseNumber;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Transaction {
    pub sender: String,
    pub receiver: String,
    pub amount: PreciseNumber,
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

    pub async fn get_transaction(
        rpc_url: &str,
        erc20_address: &str,
        tx_hash: &str,
    ) -> Result<super::Transaction> {
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
        let erc20_address = H160::from_str(erc20_address.trim_start_matches("0x")).unwrap();
        let transfer_topic = H256::from_slice(&hex::decode(ERC20_TRANSFER_TOPIC)?);
        let transfer_log = receipt
            .logs
            .iter()
            .find(|log| {
                log.address == erc20_address
                    && !log.topics.is_empty()
                    && log.topics[0] == transfer_topic
            })
            .ok_or_else(|| anyhow!("No ERC20 transfer event found"))?;

        if transfer_log.topics.len() != 3 {
            return Err(anyhow!("Invalid Transfer event format"));
        }

        // Extract transfer details from the event
        let sender = format!(
            "0x{}",
            hex::encode(&transfer_log.topics[1].as_bytes()[12..])
        );
        let receiver = format!(
            "0x{}",
            hex::encode(&transfer_log.topics[2].as_bytes()[12..])
        );
        let amount = PreciseNumber::from_hex_str(&hex::encode(&transfer_log.data.0))?;

        Ok(super::Transaction {
            sender,
            receiver,
            amount,
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
}

#[cfg(feature = "web3")]
pub use web3_feature::*;
