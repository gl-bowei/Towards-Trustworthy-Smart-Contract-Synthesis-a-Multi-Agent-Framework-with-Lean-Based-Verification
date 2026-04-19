# LeVer: Trustworthy Smart Contract Synthesis with Lean-Based Verification

<p align="center">
  <b>Official repository for</b><br>
  <i>Towards Trustworthy Smart Contract Synthesis: A Multi-Agent Framework with Lean-Based Verification</i><br>
  <b>ACL 2026 Main Conference</b>
</p>

<p align="center">
  <b>Bowei Zhang</b>, Hanbing Liu, Qixin Tian, Siyu Chen, Ziyuan Wang, Qi Qi
</p>

<p align="center">
  <img alt="status" src="https://img.shields.io/badge/status-coming%20soon-orange">
  <img alt="venue" src="https://img.shields.io/badge/ACL-2026-blue">
  <img alt="type" src="https://img.shields.io/badge/research-prototype-purple">
</p>

> LeVer is a multi-agent framework for generating **trustworthy smart contracts** from natural-language requirements.  
> It integrates **LLM-based Solidity synthesis**, **Lean-based auto-formalization**, **formal proof search**, and **adversarial sandbox testing** into a closed-loop refinement pipeline.

> [!IMPORTANT]
> The codebase is currently being cleaned and organized for public release.

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

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{zhang2026lever,
  title     = {Towards Trustworthy Smart Contract Synthesis: A Multi-Agent Framework with Lean-Based Verification},
  author    = {Bowei Zhang and Hanbing Liu and Qixin Tian and Siyu Chen and Ziyuan Wang and Qi Qi},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)},
  year      = {2026}
}
