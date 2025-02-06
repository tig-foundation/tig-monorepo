use std::str::FromStr;

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
