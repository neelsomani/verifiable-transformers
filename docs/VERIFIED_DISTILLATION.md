# Program-Mediated Attention and Normalization Removal: Verified Zero-Bilinear Circuits at Two Scales

This section reports an end-to-end extension of the direct-verification pipeline in which the two components previously identified as the principal encoding obstacles — normalization and QK attention — are removed from trained models' task circuits by post-training surgery rather than by from-scratch architectural replacement. At small scale, we (i) remove LayerNorm entirely via attenuation fine-tuning, (ii) replace the attention heads of each extracted task circuit with frozen symbolic programs that compute attention weights from token positions alone, and (iii) fine-tune the surrounding network under objectives that force the task mechanism to run through the installed programs. At GPT-2 scale, the successful route instead freezes and hash-verifies every non-program parameter, calibrates only program-local value/output parameters, re-extracts the causal core, and verifies a constant-folded exact encoding. The resulting task circuits contain no bilinear attention terms and no normalization branches. All four circuit-level properties — projected functional equivalence, content invariance, edge necessity, and continuous final-residual robustness — are verified for both small-model tasks on their exhaustive 128-input domains and for GPT-2 quote closing on its declared, hash-pinned 1,280-prompt domain.

Two negative results obtained along the way are reported with equal weight, because each motivates a method introduced to overcome it. First, branch-certified robustness verification of a retrained BandNorm model returns *unknown* at every perturbation radius tested, exposing a seed-dependent incompleteness of certificate-based proving that normalization removal eliminates by construction. Second, naive fine-tuning after program installation silently reroutes the task mechanism around the installed programs — even in an 8K-parameter model — and a single round of mechanism-pinning proved insufficient once a circuit involved multiple program heads. The healing objectives introduced below are the direct responses.

All claims in this section are bounded, projected, and circuit-level in the sense of Section 2, and attach to the specific checkpoints named in the artifact index.

## 1. Setup and per-head extraction

Experiments use the small-model configuration of Section 4 (32-token vocabulary, sequence length 6, d_model 16, 2 layers, 2 heads per layer, sparsemax attention, LeakyReLU MLPs) and the two exhaustive syntax tasks, quote closing and bracket type, each with a 128-sequence domain and a two-token candidate projection.

Circuit extraction is refined from block granularity to head granularity: the coarse graph's `attn_i` nodes are split into per-head nodes `attn_i_h_j`, with per-head masking applied to the head outputs before the output projection. As a regression check, we re-ran the block-level extractions and confirmed that the union of retained per-head edges reproduces each block-level circuit, with projected agreement 1.000 between the head-union circuit and the block circuit on both tasks, and maximum absolute logit differences below 2 × 10⁻⁵ between the controlled and native forward passes.

Three model generations appear below. Their shared architecture differs only in normalization and in which attention heads are programmatic:

| Model | Norm | Parameters | Physical QK parameters | Program heads |
|---|---|---:|---:|---|
| BandNorm (matched retrain) | Signed L1 BandNorm | 7,712 | 1,088 | none |
| Norm-free | none | 7,584 | 1,088 | none |
| Program-healed (final) | none | 6,768 | 272 | attn_0_h_0, attn_0_h_1, attn_1_h_1 |

The BandNorm model is a fresh retrain under conditions matched to the other runs; it is not the checkpoint underlying Table 2, a distinction that matters for the robustness finding in Section 3.

## 2. LayerNorm removal survives sparsemax attention

Prior work shows that LayerNorm can be removed from pretrained GPT-2-family models after fine-tuning with small loss degradation [Baroni et al., 2025]. Whether removal is compatible with sparsemax attention was an open question: sparsemax support sets are sensitive to score scale in a way softmax is not, and normalization is one of the mechanisms controlling that scale.

At small scale the answer is affirmative. We train a sparsemax + LeakyReLU model with standard LayerNorm to 100% candidate accuracy on both tasks, then remove normalization by a sequential schedule that attenuates each LayerNorm's data-dependent standard deviation toward a fixed constant, followed by exact folding of the resulting affine maps into adjacent weights (maximum folding error 3.1 × 10⁻⁵). Candidate accuracy remains 1.000 on both tasks throughout the schedule and after folding. Mean candidate margins grow from approximately 1.0 to 109.3 (quote) and 91.6 (bracket) after removal — the expected consequence of removing scale control, and a phenomenon to monitor at larger scale, since unbounded residual growth is precisely the failure mode of the weaker normalization replacements in Appendix D.

The verification consequence is structural. Without a final normalization layer, the decision tail G_T reduces to an affine map followed by an argmax comparison, so the continuous robustness property requires no branch certificates of any kind: the *unknown* outcome is eliminated from the method's range, not merely avoided in practice.

## 3. Branch-certified robustness is incomplete at branch boundaries

The matched BandNorm retrain verifies projected functional equivalence, content invariance, and edge necessity for both tasks, but continuous robustness returns UNKNOWN_BRANCH_UNSTABLE for both. Because the published Table 2 model verified all four properties, we diagnosed the discrepancy rather than narrate it away.

We first confirmed that the perturbation radius matches the original runs (ε₀ = 0.01 for both tasks), then swept ε ∈ {ε₀/10, ε₀/4, ε₀/2, ε₀} on the matched BandNorm model and on the norm-free model:

| Model | Task | ε₀/10 | ε₀/4 | ε₀/2 | ε₀ | Classification |
|---|---|---|---|---|---|---|
| BandNorm (matched) | quote_close | unknown (64/128 certified) | unknown | unknown | unknown | branch-adjacent |
| BandNorm (matched) | bracket_type | unknown (64/128) | unknown (64/128) | unknown (64/128) | unknown (64/128) | branch-adjacent |
| Norm-free | quote_close | verified | verified | verified | verified | — |
| Norm-free | bracket_type | verified | verified | verified | verified | — |

Two features of this table fix its interpretation. First, the classification is branch-adjacent rather than margin-thin: instability persists at the smallest radius tested, so the final BandNorm operating point lies essentially on a branch boundary for a subset of inputs, independent of ε. Second, the number of decision violations is zero at every radius — the solver never found a perturbation that flips the projected decision. The correct statement is therefore not that the matched model is non-robust, but that branch-certified proving is *incomplete*: BandNorm is continuous and piecewise-affine, so crossing a branch boundary does not imply a decision change, yet the traced-branch method cannot reason across the boundary and must return unknown. Whether a given seed's operating point sits inside branch interiors (the Table 2 checkpoint) or on a boundary (the matched retrain) is not controlled by the training objective, so certifiability under this method is seed-dependent. The exactly-half pattern (64 of 128 inputs unstable) suggests the boundary condition is input-class-dependent, affecting one of the two structural classes in each domain.

This finding strengthens the case for removal over replacement on the provability axis: normalization removal does not make the robustness proofs easier — it makes this entire failure mode inexpressible.

## 4. Attention programs: DSL, synthesis, and acceptance

We define a restricted rule-list DSL for attention programs. A program consists of a default weight and an ordered list of rules; each rule adds a constant weight to a (query, key) pair when its conditions hold, and conditions test only token positions, relative distances, and token identities. Per-row scores are normalized to a distribution. Programs are therefore functions of tokens and positions alone: they read no hidden state, and for any concrete input the resulting attention matrix is a matrix of rational constants.

Programs are synthesized against a head's attention maps over the task domain by enumerating a candidate space (120 candidates per head here; no LM proposer was required at this scale). Candidates are ranked by attention-map similarity but *accepted* by a behavioral criterion: exact projected agreement — installing the candidate as the head's attention, with all else fixed, must leave the model's projected decision unchanged on every input in the domain. This deliberately admits programs coarser than the attention they replace; the healing stage absorbs the difference, and the formal claims ultimately attach to the healed model, not to attention-map fidelity.

The accepted programs are compact and interpretable:

| Head | Task | Program | Support IoU | Projected agreement |
|---|---|---|---:|---:|
| attn_0_h_0 | quote_close | weight 1 if relative_distance = 1 (previous-token head) | 0.708 | 1.000 |
| attn_0_h_1 | bracket_type | weight 1 if relative_distance ≤ 3, +1 if key_position = 3 (local window, boosted at the opener slot) | 0.690 | 1.000 |
| attn_1_h_1 | quote_close (chase round) | weight 1 if key_position ∈ {2, 3} (content and opener slots) | 0.875 | 1.000 |

Sparsemax materially eases synthesis: the teacher heads' mean support size is 1.33 positions, versus 3.5 under a softmax counterfactual computed from the same scores, so the targets are nearly one-hot and well matched to rule-list programs with exact zeros.

Program installation deletes the head's query and key projections outright (physical QK parameters drop from 1,088 to 272 in the final model, the remainder belonging to the one head that stays neural) and retains the value and output projections as ordinary trainable parameters. The installed program is frozen thereafter. In the SMT encoding, a program head contributes only linear constraints: its attention weights are rational constants per input, so both the score bilinearity and the attention-weighted value bilinearity vanish for that head.

## 5. Healing and mechanism migration

Fine-tuning after installation ("healing") is where the central difficulty of this approach lives, and it required three successive objectives.

**Plain healing fails silently.** Fine-tuning the non-frozen parameters under the ordinary task loss restores 100% candidate accuracy within 10 steps. But a post-hoc migration check — re-extracting circuits and ablating the installed heads — shows the recovered behavior does not run through the programs: ablating the quote program head changes the model's candidate distribution by a KL of 7.4 × 10⁻⁶ and leaves projected agreement at 1.000. The program is decorative; gradient descent rebuilt the routing elsewhere. That this bypass arises in an 8K-parameter model with two heads per layer indicates it is the default behavior of healing, not a large-model phenomenon.

**Ablation-aware healing pins single heads.** We augment the loss with three terms: a circuit-only loss (forward pass with all non-circuit edges ablated), a sampled-ablation loss (non-circuit edges kept with probability 0.25 per step, so no backup path is reliably available during training), and an explicit bypass penalty that punishes correct projected behavior in a forward pass with the intended program head ablated. Under this objective the bypass is trained away — by step 20, ablating the intended head collapses agreement to 0.5 — and the migration check passes: each intended head is individually necessary and no neural path outside the program heads restores the behavior.

**Mechanism drift and the chase round.** The ablation-aware healed model satisfies the migration criterion, yet exact re-extraction reveals that the quote circuit changed shape: where the pre-healing circuit was emb → attn_0_h_0 → mlp_0 → mlp_1 → logits, the healed circuit routes emb → attn_0_h_0 → mlp_0 → attn_1_h_1 → logits, acquiring a new downstream dependency on a neural attention head that the original mechanism did not use. The migration criterion is satisfiable while the mechanism drifts within the permitted graph. We recorded this as a standing artifact, then attempted one chase round: synthesize a program for attn_1_h_1 from the healed model's attention maps (accepted at projected agreement 1.000), install it frozen, and re-heal. Plain and single-head-penalty re-healing both failed the migration check with two intended quote heads; passing required a third objective, *core-aware healing*, which applies the bypass penalty to the combined intended-head set and to each intended head individually, evaluated on both the full graph and the pre-registered circuits. One chase round sufficed; the iteration guard (two rounds maximum before reporting drift as a negative result) was not triggered.

The three-objective progression is itself a result: mechanism location is not preserved by behavioral fine-tuning, is enforceable for single heads by ablation-aware training, and requires jointly-and-individually targeted suppression once a circuit's intended mechanism spans multiple heads. This instantiates, in a minimal setting, the training-time redundancy control discussed as an open problem in Section 7.

## 6. Verification results

The final model carries three frozen program heads (attn_0_h_0, attn_0_h_1, attn_1_h_1) and one neural head (attn_1_h_0), which lies outside both task circuits. Re-extraction on the final checkpoint, the exhaustive PyTorch sanity check, the SMT-encoder-versus-PyTorch sanity check, the migration check, and the four property verifications all pass:

| Task | Circuit edges | Program heads in circuit | Neural heads in circuit | QK bilinear terms | Equivalence | Invariance | Edge necessity | Robustness |
|---|---:|---|---|---:|---|---|---|---|
| quote_close | 5 | attn_0_h_0, attn_1_h_1 | none | 0 | verified | verified | verified | verified |
| bracket_type | 4 | attn_0_h_1 | none | 0 | verified | verified | verified | verified |

Ablating any single program head in either circuit collapses candidate accuracy to 0.5 (chance), and ablating all program heads leaves no neural path that restores the behavior. Both circuit encodings contain no bilinear terms and no normalization branches: given the per-input trace, every constraint is affine, and the robustness property quantifies a continuous ℓ∞ ball through a purely affine decision tail. To our knowledge these are the first formally verified transformer task circuits whose attention is entirely symbolic at inference time.

## 7. Encoding-cost accounting

Raw solver totals are not comparable across circuits of different sizes, so the cost table enforces a claim policy: qualitative properties (certificate requirements, reachable statuses) are reported per run; quantitative comparisons across topologies use only normalized, instrumented columns (assertions per edge, solve seconds per edge, assertions attributable to normalization per norm instance); and raw totals are quoted only within explicitly matched-topology groups.

Assertion attribution instruments each encoder with a source category (norm, attention, MLP, embedding/residual, decision). The BandNorm encoding attributes approximately 50 assertions to each normalization instance. For quote_close, the threshold sweeps of the matched BandNorm and norm-free models happen to contain circuits of identical edge count (5 edges, both at threshold 0.001, both at projected agreement 1.000), giving one controlled head-to-head: the BandNorm circuit encodes at ~113,000 assertions per edge versus ~66,000 for the norm-free circuit (≈1.7×), and the BandNorm run's robustness is unprovable by the branch-certified method at this seed while the norm-free run verifies all four properties. No equal-edge pair exists for bracket_type, and no forced comparison is reported. The final program-healed quote circuit is the cheapest entry in the table at ~31,000 assertions per edge — roughly half the norm-free neural circuit — quantifying the encoding value of eliminating the bilinear terms in addition to the norm.

## 8. Claim discipline and limitations

At small scale, the verified object is the final healed hybrid checkpoint, not the originally trained model; this pipeline is verified distillation or editing-for-verifiability, not post-hoc interpretation of the pre-surgery network. The task circuits are zero-bilinear, although a neural head remains outside both circuits. The small-scale guarantees are bounded to the 128-input domains, the two-token projections, and zero-ablation semantics, per Section 1.4. Their position-conditioned attention programs exploit fixed templates, and their behavioral acceptance does not imply attention-map identity.

At GPT-2 scale, the verified object is the rung-3 program-local W_V/W_O calibrated checkpoint named in the evidence manifest. Every other parameter is frozen and hash-verified, so the causal no-leakage claim is deductive rather than inferred from individual-head non-redundancy. The guarantee is exact only over the frozen 1,280-prompt quote domain D and the two-token quote projection; it is not an unrestricted-language or held-out-generalization theorem. The preserved synthetic counterexample outside D demonstrates that boundary. Bracket type is not verified at this scale: its only exactly-generalizing selected circuit retains all 144 attention heads, which is reported as a localization limit rather than hidden as unfinished surgery.

## 9. Artifact index

Norm removal: `artifacts/small_layer_norm/` (source), `artifacts/small_norm_free/` (model + `removal_metrics.json`). Per-head regression: `artifacts/per-head-block-regression.json`. Robustness diagnosis: `artifacts/robustness_eps_sweep.json`; matched BandNorm model and circuits under `artifacts/small_band_norm_matched*/`. Programs: `artifacts/small_programs/` (layer-0 synthesis), `artifacts/small_program_chase_round1/` (layer-1 synthesis). Healing generations: `artifacts/small_program_healed/` (plain; failing migration report preserved at `small_program_healed_circuits/migration_report.json`), `artifacts/small_program_healed_ablation_aware/`, `artifacts/small_program_healed_chase_round1_core_aware/` (final model of record). Drift record: `artifacts/mechanism_drift.json` (includes a SHA-256 of the plain-heal failure report). Chase summary: `artifacts/program_chase_report.json`. Final small-scale circuits and verification outputs: `artifacts/small_program_healed_chase_round1_core_aware_circuits/`. Cost accounting: `artifacts/small-unified-cost-table.{json,csv}`, matched-topology selection under `artifacts/small_matched_topology/`.

GPT-2 Phase Q: `artifacts/gpt2-phase-q-agent/evidence_manifest.json` is the evidence root and pins the checkpoint, programs, selected circuit, domain, causal gates, verification records, and their hashes. The calibrated FP32 checkpoint is kept outside Git at `artifacts/gpt2-phase-q-readside-calibration/fp32_export/model.safetensors`; its SHA-256 is recorded in the manifest. The exact verification summary, resumable per-input records, anchor regression, and narrative report are under `artifacts/gpt2-phase-q-readside-calibration/folded_verification/` and `artifacts/gpt2-phase-q-readside-calibration/FOLDED_VERIFICATION_REPORT.md`.

## 10. GPT-2-scale bounded verification

### 10.1 Generalization boundary and declared domain

The held-out-generalization track ended at protocol v4. On a fresh 512-prompt
gate, the full norm-free model was exact on both tasks, the selected bracket
circuit was exact, and the selected quote circuit scored 511/512. The
exactly-generalizing bracket circuit retained 340 edges and all 144 attention
heads. No v5 was run. These results establish a sparsity-versus-exactness
frontier, not a failure of bounded verification.

The flagship continuation therefore declares D as the ordered union of the
frozen v4 development and gate manifests: 1,280 unique, hash-pinned quote
prompts, balanced 640/640 by quote class. There is no held-out split in this
claim. Every verification statement below is scoped to D.

### 10.2 Symbolic replacement and causal pinning

The first bounded extraction retained heads `7.11` and `9.0`, and both were
replaced by frozen restricted-DSL programs. Ordinary whole-model healing could
recover exact behavior and stay within the perplexity budget, but its lesion
result could not establish that the programs carried the mechanism. A
pre-healing causal audit then supplied the missing baseline: in the untouched
model, jointly ablating those two heads already left 1,256/1,280 prompts
correct, while ablating the selected circuit reduced accuracy to 710/1,280.
The redundant route therefore predated healing.

The successful follow-up did not fine-tune the whole model. It froze and
hash-verified every non-program parameter and trained only program-local
parameters on a registered ladder. The model of record is rung 3, which
calibrates program-local W_V/W_O parameters. It reaches 1,280/1,280 agreement
in both full and circuit-only forwards, with OpenWebText perplexity 25.5707
against the locked 28.6176 budget. The joint program-set lesion remains
1,256/1,280 and the registered whole-circuit lesion remains 710/1,280.
Because every trainable contribution disappears under the corresponding
lesion and all other parameters are unchanged, no bypass can be introduced by
calibration.

The original individual-head-necessity gate was amended under documented
registration. It had imported a migration diagnostic into a constrained design
where leakage identity already proves the relevant no-bypass fact, and it
instead rejected native redundancy: `7.11` is load-bearing, while `9.0` is
auxiliary even circuit-internally. The amended causal gate requires leakage
identity, joint necessity of the program set, whole-circuit necessity, and
circuit-internal edge necessity in the formal property.

### 10.3 Re-extracted causal core

Re-extraction on all of D selected exactly three edges:

```text
emb → mlp_0 → attn_7_h_11 → logits
```

Head `9.0` remains a frozen program in the full checkpoint but is absent from
the selected circuit. The verified circuit therefore contains one symbolic
program head, no neural attention head, no QK computation, and no normalization.
Its mechanism shape qualitatively converges with the string-closing circuit
reported by Gao et al. — an early MLP constructs quote features and a later
copy head carries the quote type — here recovered independently from a densely
trained model. This comparison concerns mechanism shape, not circuit stability.

### 10.4 Exact folded verification

The FP32 encoder agrees with PyTorch to a maximum candidate-logit error of
1.11 × 10⁻⁸. The independent monolithic exact encoder builds in approximately
494 seconds and validates 9,216 LeakyReLU trace constraints as SAT. The
registered verification battery uses a semantics-preserving constant fold:
LeakyReLU signs are certified externally in exact rational arithmetic,
per-input forward computations contract to exact rational candidate logits,
and the four properties reduce to ground comparisons in linear real
arithmetic. Robustness is an exact linear margin computation at the
final-residual interface. Folded and monolithic evaluators agree by literal
rational equality on all eight preserved anchor sequences.

All four properties pass over every row of D:

| Property | Result |
|---|---|
| Projected functional equivalence | 1,280/1,280 |
| Content invariance | 1,280/1,280 |
| Circuit-internal edge necessity | all 3 edges; 640 decision-changing witnesses per edge |
| Continuous robustness | 1,280/1,280 at ε = 0.01 |

The minimum exact per-input robustness radius is approximately 0.01514986,
more than 1.5 times the locked ε. The proof records contain 5,911,040
categorized assertions and 62 unique exact MLP sign signatures. There are no
normalization branches or bilinear terms, so *unknown* is structurally
unreachable.

One legacy max-length-3 sequence, `[10, 1, 10]`, is a genuine counterexample
outside D: token 1 is an output quote token but not one of the four registered
opener-context IDs, so the restricted scan program takes its exact
self-fallback. The folded and monolithic evaluators reproduce the same result.
This constructive boundary exhibit is preserved; it does not weaken any claim
over D.

### 10.5 Scale boundary

Quote closing is the verified GPT-2 flagship. Bracket type remains a measured
localization boundary: its only exactly-generalizing circuit retained all 144
attention heads, so no claim of a small verified bracket circuit is made.
Together, the two tasks show why the final claim is bounded and task-specific:
program replacement plus causally constrained calibration can produce a
formally verified GPT-2-derived circuit, but sparse exact localization is not
guaranteed for every behavior.
