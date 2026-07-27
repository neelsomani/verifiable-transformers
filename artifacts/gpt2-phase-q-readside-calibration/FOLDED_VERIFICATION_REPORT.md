# Phase Q folded verification report

The redirected exact folded battery passed on all 1,280 ordered prompts of the
frozen bounded domain `D`. No model parameter, program, selected edge, OWT
budget, exactness gate, or causal gate was changed.

## Exactness gate

`folded_verification/anchor_equality.json` records exact-rational equality
between the folded evaluator and an independent uncontracted monolithic
evaluator on all eight preserved synthetic sequences. Equality is literal
rational equality, with no tolerance.

The old `max_length=3` log remains unchanged at SHA-256
`ff54f7aa046961eb75c37d9a5d26fe8af2f0d3c5aea89bad05a3be0510c99fd7`.
It is only a smoke-domain construction artifact. Seq1 is `[10, 1, 10]`; that
exact token sequence is outside `D`. Its recorded monolithic violation result
is SAT and the folded evaluator reproduces the same counterexample. The reason
is semantic, not numerical: token ID 1 is an output quote token but is not one
of the four manifest opener-context IDs recognized by the restricted scan
program, so the program takes its exact self-fallback. The other smoke-domain
outcomes are also preserved in the anchor file. None is represented as a
result on `D`.

## Full-domain result

The manifest SHA-256 is
`4b017206e3182ddf52356808ad073c2b1ffbaf225f577c802b93141c739a91ff`;
the ordered-example-ID hash is
`6c769c18764b847e0d9bb927df2cc0fab68148ef3996cb02b231fae76f079ecc`.
All four registered properties passed:

- functional equivalence: 1,280/1,280 exact decisions agree with `P(x)`;
- content invariance: every row agrees with the exact projected-decision anchor
  for its registered single/double quote stratum;
- circuit-internal edge necessity: all three fixed edges have exact witnesses;
- continuous robustness: 1,280/1,280 pass at epsilon `1/100`.

The smallest certified robustness radius is the exact rational
`106302753145838114582811419361320437872636088662797442126380209384817807/7016749180801418050551083983878063208043793439326543902527951887925248000`
(approximately `0.015149857919489826`), on
`v4:quote_close:single:24620fc35863514c`.

The proof records attribute 5,911,040 assertions. One-time exact constant
contraction took 16.036 seconds; per-input encode/fold work totaled 50.578
seconds. Solve time is recorded as zero because every post-fold query is a
ground exact comparison, and the registered final-residual robustness LP is
solved in closed form as exact margin divided by the L1 norm of the candidate
readout difference. This is the exact optimum, not a numerical solver result.

Every used LeakyReLU branch has an exact sign certificate. The smallest
absolute preactivation over the per-input records is
`102075552812051/4611686018427387904`, strictly positive. The registered
robustness perturbation is applied at the final-residual interface, downstream
of `mlp_0`; consequently that perturbation cannot cross an MLP branch and each
certificate covers the entire claimed final-residual region. No split or
`UNKNOWN` case occurred.

The resumable per-input proof objects are in
`folded_verification/per_input.jsonl`; their SHA-256 and all pinned model,
program, calibrated-constant, circuit, and topology hashes are recorded in
`folded_verification/summary.json`.
