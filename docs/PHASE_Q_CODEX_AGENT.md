# Phase Q autonomous Codex brief

You are operating directly on the rented eight-H100 host in
`/workspace/verifiable-transformers`. Work autonomously and persist while safe,
scientifically valid work remains. Read `AGENTS.md` in full before acting, then
inspect the repository, the complete Phase Q implementation, and every existing
Phase Q result artifact.

## Objective

Finish the bounded-domain `quote_close` symbolic-attention experiment honestly.
The desired technical endpoint is:

1. every retained attention head is replaced by a frozen restricted-DSL
   program, leaving zero active neural-attention bilinear terms in the selected
   circuit;
2. the healed full model and circuit-only forward both agree with `P(x)` on all
   1,280 frozen rows of declared domain `D`;
3. OpenWebText perplexity is at most `28.617593822841776`;
4. the full unsampled lesion/migration sweep passes; and
5. FP32 encoder-vs-PyTorch sanity and all four bounded properties verify.

If that endpoint is technically unattainable, finish the experiment by
producing a complete, evidence-backed terminal diagnosis and documentation.
Do not manufacture success by weakening a gate.

## Scientific status and authorization

The preregistered Phase Q track has already reached its kill criterion. Preserve
that fact. Do not edit old evidence or retroactively call later attempts
preregistered. The user has now authorized an explicitly labeled
**exploratory bounded-domain engineering continuation** so Codex can diagnose
and iterate on implementation and optimization. Any later success is an
adaptively engineered bounded-domain result, not held-out generalization and
not the preregistered Phase Q result.

The two failed healing attempts are evidence:

- `artifacts/gpt2-program-healed-bounded-quote-core-aware/`
- `artifacts/gpt2-program-healed-bounded-quote-core-aware-final/`

The second attempt started with circuit-only accuracy `1.0` and full accuracy
`0.9820312261581421`, but ended with full accuracy
`0.18515625596046448`; perplexity passed at `24.428697815344098`. The selected
programs and exact initial circuit must not be discarded. The distributed
exit-code traceback is a consequence of the gate process returning nonzero,
not the root cause.

## Operating rules

- Diagnose before launching another full 10,000-step job. Inspect loss values,
  gradients, DDP scaling, auxiliary cadence, sampled-forward semantics, program
  parameter freezing, candidate-target construction, and the gate history.
- Add cheap, deterministic regression tests and short diagnostic probes first.
  A new training attempt must have early gate checks and must be terminated
  promptly if task agreement moves materially in the wrong direction.
- Preserve every `healing_results.json`, migration report, gate history, config,
  and diagnostic result, including failures. Use a new output directory for
  every materially different attempt. Never overwrite old evidence.
- Do not delete or move existing artifacts. Do not commit model weights,
  checkpoints, optimizer state, caches, datasets, generated archives, or
  `dump_code.py`.
- Never weaken exact agreement, the perplexity budget, symbolic coverage, or
  lesion/verification gates. Never filter, duplicate, or adaptively remove
  rows from `D`.
- Do not reopen a held-out-generalization claim. Keep all claims explicitly
  bounded to the frozen 1,280-row `D`.
- Do not modify `README.md` until the experiment has a terminal result.
- Avoid unrelated refactors. Run the focused tests after every code change and
  the full suite before committing.
- Use the existing processed dataset at
  `/dev/shm/openwebtext-gpt2-block1024`; do not recompute OpenWebText.
- `NCCL_NVLS_ENABLE=0` is required for multi-GPU work on this host.
- Monitor disk space. The large Hugging Face cache is intentional and must not
  be removed without explicit user approval.
- Do not push or rewrite `main`. Create or reuse branch
  `codex/phase-q-agent`, commit coherent code/evidence summaries there, and
  push that branch for durability. Small JSON/text evidence belongs in Git;
  weights and checkpoint state do not.

## Execution discipline

Maintain a machine-readable journal under
`artifacts/gpt2-phase-q-agent/agent_journal.jsonl` containing timestamps,
hypotheses, commands, results, and decisions. Lead with the cheapest
discriminating experiment. Before each expensive run, write a JSON protocol
record stating the hypothesis, exact configuration, output path, unchanged
gates, and kill condition.

Continue through diagnosis, implementation, tests, short probes, and at most
the justified full run needed to decide the current hypothesis. If blocked by
authentication, unavailable external authority, irrecoverable infrastructure,
or a genuinely required user decision, stop with the exact blocker in
`artifacts/gpt2-phase-q-agent/BLOCKED.md`. Otherwise, do not stop merely because
the work is slow or difficult.

At the end, write
`artifacts/gpt2-phase-q-agent/FINAL_REPORT.md` with the achieved endpoint or
terminal finding, exact artifact paths, commands, gate values, commits, and
remaining limitations.
