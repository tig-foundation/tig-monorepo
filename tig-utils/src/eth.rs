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

    pub async fn get_transaction(rpc_url: &str, tx_hash: &str) -> Result<super::Transaction> {
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
        let tx = eth
            .transaction(TransactionId::Hash(tx_hash))
            .await?
            .ok_or_else(|| anyhow!("Transaction {} not found", tx_hash))?;
        Ok(super::Transaction {
            sender: format!("{:?}", tx.from.ok_or_else(|| anyhow!("Sender not found"))?),
            receiver: format!(
                "{:?}",
                receipt.to.ok_or_else(|| anyhow!("Receiver not found"))?
            ),
            amount: PreciseNumber::from_dec_str(&tx.value.to_string())?,
        })
    }

    pub const PROXY_CREATION_TOPIC_ID: &str =
        "0x4f51faf6c4561ff95f067657e43439f0f856d97c04d9ec9070a6199ad418e235";
    pub const GNOSIS_SAFE_PROXY_CONTRACT_ADDRESS: &str =
        "0xc22834581ebc8527d974f8a1c97e1bea4ef910bc";
    pub const GNOSIS_SAFE_ABI: &str = r#"[
        {
            "inputs":[],
            "name":"getOwners",
            "outputs":[
                {
                    "internalType":"address[]",
                    "name":"",
                    "type":"address[]"
                }
            ],
            "stateMutability":"view",
            "type":"function"
        }
    ]"#;

    pub async fn get_gnosis_safe_owners(rpc_url: &str, address: &str) -> Result<Vec<String>> {
        let transport = web3::transports::Http::new(rpc_url)?;
        let eth = web3::Web3::new(transport).eth();

        let gnosis_safe = Contract::from_json(
            eth.clone(),
            H160::from_str(&address)?,
            GNOSIS_SAFE_ABI.as_bytes(),
        )
        .unwrap();
        let owners: Vec<Address> = gnosis_safe
            .query("getOwners", (), None, Options::default(), None)
            .await
            .map_err(|e| anyhow!("Failed query getOwners: {:?}", e))?;
        Ok(owners.iter().map(|x| format!("{:?}", x)).collect())
    }

    pub async fn get_gnosis_safe_address(rpc_url: &str, tx_hash: &str) -> Result<String> {
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
        if !receipt
            .to
            .is_some_and(|x| format!("{:?}", x) == GNOSIS_SAFE_PROXY_CONTRACT_ADDRESS)
        {
            return Err(anyhow!("Not a Create Gnosis Safe transaction"));
        }
        match receipt
            .logs
            .iter()
            .find(|log| {
                log.topics
                    .iter()
                    .any(|topic| format!("{:?}", topic) == PROXY_CREATION_TOPIC_ID)
            })
            .map(|log| format!("0x{}", hex::encode(&log.data.0.as_slice()[12..32])))
        {
            None => Err(anyhow!("No ProxyCreation event found")),
            Some(gnosis_safe_address) => Ok(gnosis_safe_address),
        }
    }
}

#[cfg(feature = "web3")]
pub use web3_feature::*;
