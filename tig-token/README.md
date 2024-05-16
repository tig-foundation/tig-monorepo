# tig-token

Solidity code for TIG token

## Deployed ERC20

* Base [0x0C03Ce270B4826Ec62e7DD007f0B716068639F7B](https://basescan.org/token/0x0C03Ce270B4826Ec62e7DD007f0B716068639F7B)
* Sepolia Base [0x3366feee9bbe5b830df9e1fa743828732b13959a](https://sepolia.basescan.org/token/0x3366feee9bbe5b830df9e1fa743828732b13959a)

## Deployment Process

1. Copy TIGToken.sol to [Remix](https://remix.ethereum.org/)

2. Right click file and flatten 

3. Right click file and compile (note solidity compiler version)

4. Deploy, copying in ABI and Bytecode
    ```python3
    from web3 import Web3
    from web3.gas_strategies.rpc import rpc_gas_price_strategy
    import json

    RPC_URL = "FIXME"
    BYTECODE = "FIXME"
    ABI = json.loads("""FIXME""")

    w3 = Web3(Web3.HTTPProvider(RPC_URL))

    TIGToken = w3.eth.contract(abi=ABI, bytecode=BYTECODE)
    account = w3.eth.default_account

    gas_price = rpc_gas_price_strategy(w3)
    if (temp := input(f"Rpc gas price {gas_price}. Press enter to continue or provide an amount: ")) != "":
        gas_price = int(temp)
    tx = TIGToken.constructor().build_transaction({
        'from': account.address,
        'nonce': w3.eth.get_transaction_count(account.address),
        'gasPrice': gas_price,
    })
    tx.update({'gas': w3.eth.estimate_gas(tx)})
    input(f"Will cost {tx['gas']} ETH (ctrl+c to cancel)")
    signed_tx = account.sign_transaction(tx)
    print(f"Sending transaction...")
    tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    print(f"Transaction sent: {tx_hash.hex()}")
    print(f"Waiting for transaction to be confirmed...")
    w3.eth.wait_for_transaction_receipt(tx_hash)
    print(f"Transaction confirmed")
    ```

5. Goto relevant etherscan site and verify contract using flattened code

# License

Copyright (c) 2024 TIG UG, Kurfurstendamm 15, 10719 Berlin, Deutschland

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.