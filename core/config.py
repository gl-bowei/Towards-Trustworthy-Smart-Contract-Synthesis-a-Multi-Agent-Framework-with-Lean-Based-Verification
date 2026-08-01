import os

# PUBLIC LOCAL-TEST KEYS ONLY. Never fund or use these accounts on a live network.
# Anvil default mnemonic: "test test test test test test test test test test test junk"
# Derived using standard path "m/44'/60'/0'/0/x".

RPC_URL = os.getenv("FOUNDRY_ETH_RPC_URL", "http://127.0.0.1:8545")

ANVIL_CONFIG = {
    "rpc_url": RPC_URL,
    "accounts": {
        "Deployer": {
            "role": "SYSTEM_DEPLOYER",
            "private_key": "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
            "address": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
        },
        # --- Honest Users (2 accounts) ---
        "Alice": {
            "role": "HONEST_USER",
            "private_key": "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d",
            "address": "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"
        },
        "Bob": {
            "role": "HONEST_USER",
            "private_key": "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a",
            "address": "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC"
        },
        # --- Malicious Attackers (2 accounts) ---
        "Mallory": {
            "role": "MALICIOUS_ATTACKER",
            "private_key": "0x2a871d0798f97d79848a013d4936a73bf4cc922c825d33c1cf7073dff6d409c6",
            "address": "0xa0Ee7A142d267C1f36714E4a8F75612F20a79720"
        },
        "Sybil": {
            "role": "MALICIOUS_ATTACKER",
            "private_key": "0xdbda1821b80551c9d6596375f539d69396146a10a701c14533085f38a9562436",
            "address": "0xC1922C99D6158fEc868aD1289aec0be2F129C35C"
        }
    }
}

def get_simulation_agents():
    """Returns a dict of all agents excluding the Deployer."""
    return {k: v for k, v in ANVIL_CONFIG["accounts"].items() if k != "Deployer"}
