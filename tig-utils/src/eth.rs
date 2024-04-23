use crate::number::PreciseNumber;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Transaction {
    pub sender: String,
    pub receiver: String,
    pub amount: PreciseNumber,
}

#[cfg(feature = "web3")]
mod web3_feature {
    use crate::json::dejsonify;
    use anyhow::{anyhow, Result};
    use hex::{self, ToHex};
    use web3::{signing::*, types::*};

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
        let tx = eth
            .transaction(TransactionId::Hash(tx_hash))
            .await?
            .ok_or_else(|| anyhow!("Transaction {} not found", tx_hash))?;

        Ok(super::Transaction {
            sender: tx.from.unwrap().to_string(),
            receiver: receipt.to.unwrap().to_string(),
            amount: dejsonify(&tx.value.to_string())?,
        })
    }
}

#[cfg(feature = "web3")]
pub use web3_feature::*;
