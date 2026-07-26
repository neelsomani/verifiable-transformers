# Phase Q lesion-conditioned causal chase report

Date: 2026-07-26

## Outcome

The subsequently authorized causal chase stopped at the registered
diffuse/unprogrammable-backup condition. It did not establish the programmed
quote circuit as the healed model's causal mechanism.

This result is distinct from the earlier penalty-only terminal result. The
earlier probes and `artifacts/gpt2-phase-q-agent/FINAL_REPORT.md` remain
unchanged. Those probes showed that undirected penalties did not cross the
lesion boundary. This chase instead held both programmed heads unavailable
throughout extraction and localized the neural route that preserved exact
quotation-closing behavior.

## Reproduction

The saved `checkpoint-100` and its exported model have identical
`model.safetensors` SHA-256
`4a70e2f9045a3859dd9f5e41c528595f40d48b45646f64bbe23d9a1a20687c70`.
The export was used because Trainer checkpoints omit the `programs.json`
sidecar.

- Full agreement with P on D: 1.0 (1,280/1,280).
- Circuit-only agreement with P on D: 1.0 (1,280/1,280).
- Joint program lesion, full-model agreement: 1.0.
- Joint program lesion, circuit-only agreement: 0.5.
- OpenWebText validation perplexity: 25.2516468 on 11,059 examples, below the
  unchanged 28.6175938 budget. The saved training evaluation was 25.2505992.

## Lesion-conditioned extraction

The extraction reference permanently excluded outputs from `attn_7_h_11` and
`attn_9_h_0`. Every candidate was therefore evaluated with both known program
heads unavailable. The domain was the unchanged 1,280-row D, zero ablation was
used, and `min_agreement` remained 1.0.

The completed threshold-0.005 fixed point is exact but diffuse:

- 81 retained edges;
- 11 neural attention heads;
- 11 MLP nodes, spanning MLP 0 through MLP 10;
- candidate accuracy and projected agreement both 1.0;
- five edge overlaps with the original 11-edge circuit, all non-program
  scaffold edges;
- zero overlap with the original programmed attention heads.

The retained neural heads are `3.8`, `4.3`, `5.5`, `5.9`, `6.2`, `6.9`,
`7.6`, `7.9`, `8.8`, `9.10`, and `10.3`. The full circuit is preserved at
`artifacts/gpt2-circuits-bounded-quote/lesion-chase-round1-sweep/quote_close_t0.005/`.

All six registered thresholds were launched. The adjacent runs also exposed
exact-guarded early-head and MLP routes during traversal. Once the lowest
threshold's formal fixed point itself contained 81 edges and 22 neural
computational nodes, the authorized diffuse/unprogrammable stop condition was
met and the unfinished larger-threshold jobs were terminated.

## Decision

No new programs were synthesized, no structural suppression or healing run was
started, and no chase round was consumed. The localized backup is not one
fittable attention head or one specific gateable path: it is a multi-layer
neural route containing eleven MLPs and eleven attention heads. Expanding the
mandatory symbolic core to cover only its attention heads would leave the
identified MLP backup computation intact; structurally gating the whole route
would be broad model surgery rather than the authorized directed suppression
of a specific path.

Because the causal migration prerequisite failed, FP32 SMT sanity and the four
verification properties were not run. The result remains a bounded-domain
causal-localization finding and is not a held-out generalization claim.
