# Amended continuation terminal report

This report closes the user-authorized numerical replay continuation registered
in `PROTOCOL_AMENDMENT_2026-07-27.md`. It is separate from, and does not modify,
the preserved earlier `TERMINAL_REPORT.md` or `preflight.json`.

## Corrected deterministic reference

State B was reconstructed in the pinned FP32 configuration and the complete
seven-path baseline battery was executed twice back-to-back. Candidate logits
and candidate margins were bitwise identical on every path (maximum absolute
difference `0.0`, below the unchanged `1e-5` epsilon). Decisions and
mismatch-ID sets were exact between the pair and against the older audit. The
older audit was used only as the provenance/semantic reference; its unpinned
logits were not compared numerically.

## Registered ladder

All three rungs reached the early task and OWT gates:

| Rung | First exact gate | Native full | Circuit only | OWT perplexity | Budget |
|---|---:|---:|---:|---:|---:|
| scalar read-side | step 100 | 1280/1280 | 1280/1280 | 25.6691 | 28.6176 |
| diagonal read-side | step 200 | 1280/1280 | 1280/1280 | 25.7157 | 28.6176 |
| program-local `W_V/W_O` | step 200 | 1280/1280 | 1280/1280 | 25.5707 | 28.6176 |

The scalar constants were `1.923632025718689` and `1.8428443670272827`
(`16136595/8388608` and `15458899/8388608` exactly as FP32 rationals). The
diagonal checkpoint hash is
`82d5404126625e4df4303b3443b7a726e9844196f31e00ce9d1ec72f5bac42b7`.
Programs remained frozen throughout. Original GPT-2 parameters remained
frozen in rungs 1–2. In rung 3, only the registered head-local `W_V`, `b_V`,
and bias-free `W_O` slices changed; the maximum change outside those permitted
slices was exactly `0.0`.

## Terminal causal result

Every rung failed the full unsampled migration/lesion gate. For the final
program-local fallback:

| Path | Correct on `D` |
|---|---:|
| native full | 1280/1280 |
| controlled full | 1280/1280 |
| circuit only | 1280/1280 |
| full minus `attn_7_h_11` | 1280/1280 |
| core minus `attn_7_h_11` | 640/1280 |
| full minus `attn_9_h_0` | 1280/1280 |
| core minus `attn_9_h_0` | 1280/1280 |
| full minus both programs | 1256/1280 |
| core minus both programs | 640/1280 |

Thus both program heads are individually bypassed in the full graph, and
`attn_9_h_0` is also bypassed in the core. Joint necessity is present, but the
registered criterion requires individual necessity and non-bypass.

The mandatory rung-3 joint-program lesion and whole-selected-node-zero
identities both match the paired deterministic baseline with exact decisions,
exact mismatch-ID sets, and `0.0` maximum candidate-logit and margin
difference. Selected-edge zero is recorded only as observational at the rung-3
causal boundary, as registered.

## Disposition

The read-side follow-up terminates scientifically after the third registered
rung. No new rung, objective, threshold, or epsilon was introduced.
Re-extraction, FP32 encoder sanity, and the four SMT properties were not run
because staged evaluation gates them behind migration, which failed. No
verification claim is made for this continuation.

Compact evidence is in `paired_baseline.json`, `rung1_scalar.json`,
`rung1_migration_sweep.json`, `rung2_diagonal.json`,
`rung2_migration_sweep.json`, `rung3_program_local.json`, and
`rung3_causal_migration.json`. Model weights and optimizer/checkpoint state
remain outside Git.

