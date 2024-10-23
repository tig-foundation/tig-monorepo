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
        let to = format!(
            "{:?}",
            receipt.to.ok_or_else(|| anyhow!("Receiver not found"))?
        );
        if to != erc20_address {
            return Err(anyhow!(
                "Transaction not interacting with erc20 contract '{}'",
                erc20_address
            ));
        }
        let tx = eth
            .transaction(TransactionId::Hash(tx_hash))
            .await?
            .ok_or_else(|| anyhow!("Transaction {} not found", tx_hash))?;
        if hex::encode(&tx.input.0[0..4]) != "a9059cbb" {
            return Err(anyhow!("Not a ERC20 transfer transaction"));
        };
        Ok(super::Transaction {
            sender: format!("{:?}", tx.from.ok_or_else(|| anyhow!("Sender not found"))?),
            receiver: format!("0x{}", hex::encode(&tx.input.0[16..36])),
            amount: PreciseNumber::from_hex_str(&hex::encode(&tx.input.0[36..68]))?,
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
