# LeVer: Trustworthy Smart Contract Synthesis with Lean-Based Verification

> Official repository for **"Towards Trustworthy Smart Contract Synthesis: A Multi-Agent Framework with Lean-Based Verification."**

LeVer is a multi-agent framework for generating trustworthy smart contracts from natural-language requirements. It combines LLM-based Solidity synthesis, Lean-based auto-formalization, formal proof search, and adversarial sandbox testing in a closed-loop refinement pipeline.

> **Repository status:** the codebase is currently being cleaned and organized for public release. 
---

## Why LeVer?

Large language models make smart contract development easier, but generated contracts are difficult to trust. Since smart contracts are immutable after deployment and often directly manage financial assets, even subtle logic errors can cause irreversible losses.

LeVer addresses this trust gap by combining two complementary assurance mechanisms:

1. **Formal verification with Lean 4** - selected safety properties are translated into Lean theorems and checked by the Lean compiler.
2. **Adversarial dynamic testing** - an attacker agent searches for concrete exploit traces in a sandbox environment, especially for properties that are hard to prove statically.

Together, these components allow contracts to be generated, verified, attacked, repaired, and re-verified in an iterative loop.

---

## Framework Overview

![]()

At a high level, LeVer consists of four roles:

| Component | Role |
|---|---|
| **Coder Agent** | Generates Solidity contracts from requirements and repairs them using verifier/attacker feedback. |
| **Verifier Agent** | Extracts safety properties, auto-formalizes Solidity logic into Lean, and attempts to prove theorems. |
| **Attacker Agent** | Simulates adversarial transaction sequences to discover concrete exploits and counterexamples. |
| **Analyst** | Converts failed proofs and attack traces into structured repair feedback and new formal properties. |

---

## Key Features

- **Closed-loop synthesis and repair**: generated contracts are repeatedly refined using proof failures and attack traces.
- **Lean-based auto-formalization**: Solidity storage, execution context, inputs, and transition logic are modeled as functional Lean definitions.
- **Property-driven security assurance**: vulnerabilities are addressed through safety invariants rather than only pattern matching.
- **Attack-to-property refinement**: concrete exploits found by the attacker are distilled into new formal properties for future verification.
- **Dual assurance**: formal proofs provide mathematical guarantees for selected properties, while dynamic attacks improve empirical robustness.

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{lever2026,
  title        = {Towards Trustworthy Smart Contract Synthesis: A Multi-Agent Framework with Lean-Based Verification},
  author       = {Anonymous},
  year         = {2026},
  note         = {Under review}
}
```

The citation will be updated after publication.

---

## Security Notice

LeVer is a research prototype. Generated or verified smart contracts should **not** be deployed with real funds without independent review, testing, and professional security auditing.

---

## License

The open-source license will be added before the official code release. Suggested options include:

- MIT or Apache-2.0 for framework code;
- CC BY 4.0 or a dataset-specific license for documentation and released benchmark artifacts.

Please update this section according to the final release policy.

---

## Contact

For questions, please open a GitHub issue after the repository is public.

