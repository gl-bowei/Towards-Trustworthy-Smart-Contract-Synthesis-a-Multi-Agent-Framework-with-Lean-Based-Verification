# Three-mode smoke results

Latest policy-validation run executed on 2026-07-14 from the same natural-language requirement in
`examples/simple_counter_requirement.txt`.

| Mode | Requested model | Result | Executed method stages |
| --- | --- | --- | --- |
| `lean_only` | `gpt-5` | PASS | Solidity compile; Lean definitions compile; 1/1 theorem proved |
| `no_lean` | `gpt-5` | PASS | Foundry; semantic probe; agent simulation |
| `full` | `gpt-5` | PASS | Lean plus all non-Lean method stages |

Slither and LLM audit were also executed for `no_lean` and `full` as external
experiment measurements. They are not method-stage pass/fail gates.

The CLI rejects any model name other than the exact request name `gpt-5`; no
fallback model is configured. The direct API availability check returned model
identifier `gpt-5-2025-08-07` for the `gpt-5` request.

Smoke settings were intentionally small: one pipeline iteration, one attack
round, 32 configured fuzz runs, and a one-theorem Lean budget. These results
show that all three main workflows execute end to end; they are not paper-scale
coverage or reproducibility results.

## Issues found and fixed during the smoke

- Slither was initially unavailable because of a polluted system Python. It now
  supports an explicit `SLITHER_COMMAND` and was tested from an isolated virtual
  environment. Its result is an external experiment metric, not an iteration
  gate.
- Generated semantic probes could combine `vm.expectRevert` with a low-level
  `.call` result assertion and produce a false failure. The generator prompt and
  deterministic sanitizer now reject that combination.
- `result.json` now records the exact requested model and separate Lean/non-Lean
  stage results, including the `full` mode.

## Decision policy

The active decision policy matches the original core implementation. Foundry
and the default semantic-probe policy can trigger iteration. Lean is
non-blocking in `full` and required in `lean_only`. Slither and LLM audit are
reported after execution but do not affect iteration. Agent infrastructure
failure may be accepted when no concrete breach was observed.

## Observations that remain outside this smoke's success claim

- The semantic-obligation extractor incorrectly inferred a per-user-cap
  obligation for `SimpleCounter`; downstream stages marked it not applicable.
- The one-theorem budget does not prove every selected property, and the Lean
  counter model uses unbounded `Nat` rather than an exact EVM `uint256` model.
- Slither reported zero High findings and one Medium finding in both non-Lean
  runs. Passing here means the configured High-severity gate passed; the Medium
  finding still requires manual triage before making a security claim.

Local run artifacts live under `experiments/` and the isolated Slither runtime
under `.venv-slither/`. Both are intentionally excluded by `.gitignore` and
must not be included in the public source release.
