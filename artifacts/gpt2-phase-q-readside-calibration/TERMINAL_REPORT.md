# Terminal report

The constrained causal-integration follow-up stopped at the mandatory
deterministic FP32 preflight. No search, optimization, calibration step, OWT
evaluation, migration sweep, extraction, or SMT verification was run.

## Causal evidence

The implemented rungs 1–2 read-side path decomposes a programmed head's
bias-free `W_O` contribution at a specific retained reader. A scalar or
768-channel diagonal transform is applied there, without changing the head
residual write seen by other readers. Focused tests prove:

- scalar calibration changes only the selected reader contribution;
- diagonal calibration is channelwise after `W_O`;
- selected-edge zero makes the calibration delta identically zero; and
- an unselected reader is bitwise unchanged.

Together with the existing per-head and pre-healing audit tests, 11 focused
tests passed.

No trained causal claim is made because the preflight stop preceded every
ladder rung.

## Task and identity result

The frozen state-B decisions and mismatch-ID sets reproduced exactly for all
four preflight paths:

| Path | Correct | Decisions/IDs | Max stored-margin difference | `1e-5` |
|---|---:|---|---:|---|
| native full | 1257/1280 | exact | 1.7642974853515625e-5 | fail |
| joint program-head lesion | 1256/1280 | exact | 1.621246337890625e-5 | fail |
| whole selected-circuit node-zero | 710/1280 | exact | 7.867813110351562e-6 | pass |
| selected-edge zero | 820/1280 | exact | 6.556510925292969e-6 | pass |

The locked policy requires every baseline path to be within `1e-5`. Native
full and joint-head lesion exceed it, so the aggregate preflight fails.
Epsilon was not changed.

The pre-healing audit does not store the two raw candidate logits; it stores
their decision and candidate margin. The preflight therefore compared every
available stored per-input quantity and recorded fresh candidate-logit hashes
for any future, separately authorized same-path investigation.

OWT was not evaluated because the stop rule fired first.

## Limitation and disposition

This terminal result is a reproducibility-boundary result, not evidence that
read-side calibration succeeds or fails. The discrepancy is small enough to
leave every projected decision and mismatch set unchanged, but it is larger
than the preregistered numerical identity tolerance on two paths. The protocol
explicitly forbids relaxing epsilon or continuing after that observation.

All prior artifacts remain unchanged. The registration is commit `1046efc`.
The follow-up made zero optimizer steps and did not run any ladder rung.
