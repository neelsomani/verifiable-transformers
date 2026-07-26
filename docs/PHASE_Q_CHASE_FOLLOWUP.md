# Phase Q causal-circuit chase follow-up

This direction supersedes only the exploratory continuation's conclusion that
the failed penalty-only migration probes were terminal. Preserve those probes
and their report unchanged as evidence. The user has clarified that the
programmed quote circuit must be the causal mechanism for the healed model's
quotation-closing behavior; a sufficient but fully bypassable surrogate is not
an acceptable endpoint.

## Current evidence

The exploratory 100-step checkpoint at
`artifacts/gpt2-program-healed-bounded-quote-exploratory-probe-v2/checkpoint-100`
has:

- full agreement on `D`: 1.0;
- circuit-only agreement on `D`: 1.0;
- OpenWebText perplexity: 25.25059924330555, within the unchanged 28.617593822841776
  budget;
- both retained attention heads replaced by frozen DSL programs; and
- failed migration: jointly lesioning the programs leaves full-model agreement
  at 1.0.

Longer KL-to-uniform and direct opposite-target penalty probes did not cross the
categorical lesion boundary. Do not launch another loss-weight or duration
variant before localizing the actual backup route.

## Authorized next experiment: lesion-conditioned circuit chasing

1. Load the saved exact programmed checkpoint and reproduce all current metrics.
2. Lesion the complete intended symbolic circuit, at minimum both programmed
   heads `7.11` and `9.0`, while preserving the full model's surviving
   quotation-closing behavior.
3. Run per-head extraction on that lesioned model over the unchanged 1,280-row
   bounded domain `D`. The extraction must be conditioned on the original
   programmed heads remaining unavailable, so it exposes the surviving backup
   route instead of rediscovering the known circuit.
4. Record the backup circuit, its exact agreement, retained heads/edges, and its
   overlap with the original circuit.
5. Chase the concrete backup:
   - Prefer synthesizing programs for fittable newly retained attention heads
     and expanding the mandatory symbolic core.
   - If a specific non-circuit path cannot be programmed, suppress or
     structurally gate that identified path during healing; do not apply an
     undirected whole-model penalty and call it localization.
6. Heal again with frozen programs, exact full and circuit-only supervision,
   the unchanged perplexity budget, and explicit early gates.
7. Repeat lesion-conditioned extraction only if a new exact backup remains.
   Retain the existing two-round chase guard from `AGENTS.md`.
8. Success requires:
   - exact full and circuit-only agreement on every row of `D`;
   - perplexity at most 28.617593822841776;
   - joint symbolic-circuit ablation breaks the full model's projected
     quotation-closing behavior;
   - no exact neural backup circuit remains under the registered extraction
     procedure;
   - re-extraction of the unlesioned healed model retains the programmed
     mechanism; and
   - FP32 SMT sanity and all four verification properties pass.

Do not weaken, filter, or alter `D`, and do not turn the result into a held-out
generalization claim. If both chase rounds expose fresh exact backups or the
backup circuit is effectively diffuse/unprogrammable, stop with that concrete
causal-localization finding.

Update the Phase Q journal and final report to distinguish the earlier
penalty-only terminal result from this subsequently authorized chase. Commit
coherent code and small evidence to `codex/phase-q-agent`. A missing GitHub
credential must not stop scientific work; leave local commits for transport by
the supervising Codex instance.
