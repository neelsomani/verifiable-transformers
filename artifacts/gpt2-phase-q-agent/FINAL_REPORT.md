# Phase Q autonomous Codex terminal report

Date: 2026-07-25
Branch: `codex/phase-q-agent`

## Outcome

The desired GPT-2 flagship endpoint was not achieved. The adaptive exploratory
continuation recovered exact bounded behavior and the perplexity gate, but the
full unsampled migration/lesion gate failed. Consequently re-extraction, FP32
encoder sanity, and the four bounded SMT properties were not run.

This does not alter the preregistered result: Phase Q had already reached its
kill criterion after the two preserved 10,000-step failures. Everything added
here is explicitly adaptive bounded-domain engineering, not preregistration
and not held-out generalization.

## Achieved gates

- Frozen domain D: 1,280 unique quote prompts, unchanged prompt-set SHA-256
  `3d590ce66edc1e83e054735523674c1d7c77af4297fd230b2a17d3209917bd48`.
- Symbolic coverage: retained heads `7.11` and `9.0` are frozen restricted-DSL
  programs; the selected circuit has zero active neural-attention bilinear
  terms.
- Best exploratory checkpoint:
  `artifacts/gpt2-program-healed-bounded-quote-exploratory-probe-v2/`.
  Full agreement = 1.0 (1,280/1,280), circuit agreement = 1.0
  (1,280/1,280), OWT perplexity = 25.25059924330555, below
  28.617593822841776.

## Failed gate

At the best exact checkpoint, the unsampled lesion sweep reported:

- joint full agreement after removing both programs: 1.0;
- joint core agreement: 0.5;
- `7.11` necessary in core but not full;
- `9.0` necessary in neither full nor core;
- migration pass: false.

The longer exploratory continuation preserved four complete gates at steps
100, 200, 300, and 400. Full/circuit accuracy stayed 1.0 and perplexity
improved from 25.10097 to 24.96165, but the lesion matrix did not change.
That run was stopped at step 421 because KL-to-uniform can shrink a signed
margin toward zero without requiring its argmax to flip. Its checkpoint-250
and gate history remain under
`artifacts/gpt2-program-healed-bounded-quote-exploratory-full-v1/`.

A final 50-step counterfactual probe used opposite-P targets only on ablated
forwards. It retained exact full/circuit behavior and perplexity 25.29012, but
the lesion matrix was again unchanged. Evidence is under
`artifacts/gpt2-program-healed-bounded-quote-exploratory-probe-v3/`.

## Root-cause findings

1. The original core-aware objective never supervised the actual full forward
   used by the gate. Its sampled-forward loss could be nearly zero while the
   full model collapsed.
2. The program-context wrapper accepts `**kwargs`; Transformers 4.49 therefore
   inferred support for accumulation-aware loss kwargs and skipped the normal
   division by `gradient_accumulation_steps=4`. The old ~13.3 logged loss
   versus ~3.2 OWT loss exposed this fourfold scaling error.
3. Programs were frozen, programmed heads had no Q/K parameters, candidate
   targets were valid binary indices, and ordinary eight-rank DDP averaging
   was otherwise correct.
4. Uniform bypass suppression optimizes margin magnitude, not the categorical
   lesion event. Direct counterfactual supervision supplies the right gradient
   but did not overcome the simultaneous intact/full/core constraints within
   the bounded probe.

The deterministic gradient evidence is
`artifacts/gpt2-phase-q-agent/healing_objective_diagnostic.json`.

## Code and tests

Changes:

- correct custom mean-loss accumulation scaling;
- add optional direct full-forward behavior loss;
- add early wrong-direction termination;
- make failed-result persistence rank-zero safe;
- add optional counterfactual bypass targets;
- add a deterministic objective/gradient diagnostic;
- add focused regression tests.

Focused tests: `26 passed`.

The full-suite result and durable commit/push identifiers are recorded in the
final journal entries and the handoff message.

## Artifact index

- Machine journal: `artifacts/gpt2-phase-q-agent/agent_journal.jsonl`
- Evidence manifest: `artifacts/gpt2-phase-q-agent/evidence_manifest.json`
- Protocols: `protocol_exploratory_probe_v1.json`,
  `protocol_exploratory_probe_v2.json`, `protocol_exploratory_full_v1.json`,
  `protocol_exploratory_probe_v3.json`
- Preregistered failures:
  `artifacts/gpt2-program-healed-bounded-quote-core-aware/` and
  `artifacts/gpt2-program-healed-bounded-quote-core-aware-final/`
- Exploratory exact checkpoint:
  `artifacts/gpt2-program-healed-bounded-quote-exploratory-probe-v2/`
- Falsified long continuation:
  `artifacts/gpt2-program-healed-bounded-quote-exploratory-full-v1/`
- Terminal counterfactual probe:
  `artifacts/gpt2-program-healed-bounded-quote-exploratory-probe-v3/`

## Remaining limitation

The evidence does not show that migration is impossible in principle. It shows
that the preregistered objective, the corrected full-supervision objective,
its measured longer continuation, and a short direct counterfactual lesion
objective all failed the same locked migration matrix. A future experiment
would need a newly registered mechanism-localization strategy and budget; it
must not be described as continuation of the preregistered Phase Q result.
