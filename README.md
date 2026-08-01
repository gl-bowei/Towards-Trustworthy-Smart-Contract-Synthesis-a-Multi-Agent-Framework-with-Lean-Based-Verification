# LeVer: Trustworthy Smart Contract Synthesis with Lean-Based Verification

<p align="center">
  <b>Official repository for</b><br>
  <i>Towards Trustworthy Smart Contract Synthesis: A Multi-Agent Framework with Lean-Based Verification</i><br>
  <b>ACL 2026 Main Conference</b>
</p>

<p align="center">
  Bowei Zhang, Hanbing Liu, Qixin Tian, Siyu Chen, Ziyuan Wang, Qi Qi
</p>

<p align="center">
  <img alt="status" src="https://img.shields.io/badge/status-core%20code%20available-brightgreen">
  <img alt="venue" src="https://img.shields.io/badge/ACL-2026-blue">
  <img alt="type" src="https://img.shields.io/badge/research-prototype-purple">
</p>

> LeVer is a multi-agent framework for generating **trustworthy smart contracts** from natural-language requirements.  
> It integrates **LLM-based Solidity synthesis**, **Lean-based auto-formalization**, **formal proof search**, and **adversarial sandbox testing** into a closed-loop refinement pipeline.

> [!IMPORTANT]
> This repository currently provides the minimum public implementation of the
> main requirement-to-Solidity pipeline. The paper-scale experiment and full
> reproducibility package will be released separately.

---

## Overview

Large language models make smart contract development much more accessible, but generated contracts remain difficult to trust. Since smart contracts are immutable after deployment and often directly manage financial assets, even subtle logic errors can lead to irreversible losses.

LeVer addresses this trust gap through two complementary assurance mechanisms:

- **Formal verification with Lean 4**: selected safety properties are translated into Lean theorems and checked in a theorem-proving environment.
- **Adversarial dynamic testing**: an attacker agent searches for concrete exploit traces in a sandbox environment, especially for behaviors that are difficult to verify statically.

By combining these two mechanisms, LeVer supports an iterative workflow in which contracts are **generated, verified, attacked, repaired, and re-verified**.

---

## Framework

<p align="center">
  <img src="Framework_new_01.png" alt="LeVer Framework" width="95%">
</p>

At a high level, LeVer is built around four collaborative roles:

| Component | Description |
|---|---|
| **Coder Agent** | Generates Solidity contracts from user requirements and repairs them based on verifier and attacker feedback. |
| **Verifier Agent** | Extracts safety properties, auto-formalizes Solidity logic into Lean, and attempts to prove the corresponding theorems. |
| **Attacker Agent** | Simulates adversarial transaction sequences to discover concrete exploits and counterexamples. |
| **Analyst** | Converts failed proofs and attack traces into structured repair feedback and new formal properties. |

---

## Key Features

- **Closed-loop synthesis and repair**  
  Generated contracts are iteratively refined using both proof failures and discovered attack traces.

- **Lean-based auto-formalization**  
  Solidity storage, execution context, inputs, and transition logic are translated into functional Lean definitions.

- **Property-driven assurance**  
  Security is framed through formal safety invariants rather than relying only on vulnerability pattern matching.

- **Attack-to-property refinement**  
  Concrete exploits discovered during dynamic testing are distilled into new formal properties for subsequent verification.

- **Dual assurance mechanism**  
  Formal proofs provide mathematical guarantees for selected properties, while adversarial testing improves empirical robustness against complex attacks.

---

## Core Code Release

The public entry point exposes the requirement-to-Solidity workflow in three
modes:

| Mode | Lean | Foundry, semantic probe, agent simulation | Slither and LLM audit |
|---|---|---|---|
| `lean_only` | Yes; blocking | No | No |
| `no_lean` | No | Yes; used by the iteration policy | Yes; reported only |
| `full` | Yes; non-blocking | Yes; used by the iteration policy | Yes; reported only |

All LLM requests use the exact model name `gpt-5`. The CLI rejects any other
model name and the code implements no model fallback.

### Prerequisites

The release has been smoke-tested with:

- Python 3.11 and `openai` 2.8.1;
- Foundry (`forge` and `anvil`) 1.5.0;
- Solidity compiler 0.8.33;
- Lean 4.12.0, selected by the generated `lean-toolchain` file;
- Slither 0.10.0 for the two non-Lean modes.

Equivalent compatible versions may work, but have not been validated by this
release. Ensure `git`, `forge`, `anvil`, and `lean` are on `PATH`; `full` and
`no_lean` additionally require `slither`.

### Installation

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

mkdir -p libs_cache
git clone --depth 1 https://github.com/foundry-rs/forge-std.git \
  libs_cache/forge-std
git clone --depth 1 https://github.com/OpenZeppelin/openzeppelin-contracts.git \
  libs_cache/openzeppelin-contracts
```

Slither is an external evaluation dependency and is intentionally separate from
the main Python requirements:

```bash
python -m pip install "slither-analyzer==0.10.0"
export SLITHER_COMMAND="$(pwd)/.venv/bin/slither"
```

If the Solidity libraries live elsewhere, set `SOLIDITY_LIB_PATH` to the
directory containing `forge-std/` and `openzeppelin-contracts/`.

### API Configuration

Set the API key in the shell. `LLM_BASE_URL` is optional and should only point
to an OpenAI-compatible endpoint that provides the exact `gpt-5` model.

```bash
export LLM_API_KEY="your-api-key"
# Optional: export LLM_BASE_URL="https://api.openai.com/v1"
```

The code also accepts `OPENAI_API_KEY` when `LLM_API_KEY` is unset. Never commit
API keys, `.env` files, or generated experiment artifacts.

The account keys in `core/config.py` are the public, deterministic Anvil test
accounts. They are only for the local sandbox: never fund them or use them on a
live network.

### Running the Three Modes

Run commands from the repository root:

```bash
# Lean only
python -m core.main_pipeline \
  --mode lean_only \
  --req-file examples/simple_counter_requirement.txt \
  --exp-name quickstart \
  --run-id lean

# All non-Lean method stages
python -m core.main_pipeline \
  --mode no_lean \
  --req-file examples/simple_counter_requirement.txt \
  --exp-name quickstart \
  --run-id no_lean

# Lean and all non-Lean method stages
python -m core.main_pipeline \
  --mode full \
  --req-file examples/simple_counter_requirement.txt \
  --exp-name quickstart \
  --run-id full
```

Use `python -m core.main_pipeline --help` for iteration, attack-round, model,
and Slither-command options. Each run creates an isolated workspace at
`experiments/<exp-name>/<run-id>/`.

### Outputs and Decision Policy

The main output is `result.json`, which records the selected mode, requested
model, decision policy, and separate Lean, non-Lean, and external-evaluation
outcomes. Each workspace also contains per-iteration metrics, `Target.sol`, a
checkpoint, execution logs, and the generated Foundry and Lean workspaces when
enabled.

The public pipeline preserves the original lenient policy, recorded as
`legacy_core_lenient`:

- Foundry and the default semantic-probe policy can trigger another iteration.
- Lean must pass in `lean_only`; incomplete Lean proofs are non-blocking in
  `full` and are recorded separately.
- Slither and the LLM audit are external experiment measurements. They do not
  trigger repair or determine the iteration result.
- Agent infrastructure failure may be accepted when no concrete breach was
  observed, matching the original core implementation.

A `passed: true` result means only that the configured criteria passed under
this policy. It is not a proof that a generated contract is secure, and no
generated contract should be deployed without independent review and testing.
The exact smoke configuration, results, and known limitations are documented in
[`SMOKE_RESULTS.md`](SMOKE_RESULTS.md).

---

## Main Results

LeVer consistently improves both **trustworthiness assurance** and **standard correctness** across frontier foundation models.

<p align="center">
  <img src="main_results.png" alt="Main results on the Smart Contract Generation Benchmark" width="95%">
</p>

**Highlights:**
- LeVer substantially reduces the **Attack Success Rate (AR)** compared with both direct generation and FSM-based baselines.
- It consistently improves **Verification Rate (VR)** and the **average number of verified properties**, showing stronger formal assurance.
- These gains also translate into better practical correctness, with improved **Slither** and **Foundry** pass rates.
- Ablation results further show that the **Verifier** and **Attacker** play complementary roles: the verifier strengthens formal guarantees, while the attacker exposes hard-to-prove dynamic vulnerabilities and feeds them back into refinement.
