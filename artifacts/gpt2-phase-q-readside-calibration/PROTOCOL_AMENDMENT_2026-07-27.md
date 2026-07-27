# Protocol amendment: paired deterministic numerical reference

Authorized 2026-07-27 from `f40412b082879d35ed12e6ecc74e1bec72101e88`.
This amendment preserves `TERMINAL_REPORT.md` and `preflight.json` unchanged as
the terminal record of the earlier preregistered execution. It authorizes a
new continuation and corrects only the numerical reference pairing. It does
not relax the registered `1e-5` epsilon, change a rung, or add an objective.

Before any calibration search or optimization:

1. Reconstruct state B in the pinned FP32 deterministic configuration and run
   the complete baseline battery twice back-to-back with identical domain
   order, batch size, dtype, kernels, deterministic flags, and intervention
   semantics.
2. Require a maximum absolute difference of at most `1e-5` for both candidate
   logits and candidate margins between the two newly paired runs. Decisions
   and mismatch-ID sets must be exactly equal.
3. Use that paired deterministic baseline as the numerical identity reference
   for every post-calibration lesion gate.
4. Retain the older audit as the provenance and semantic reference. Aggregate
   decisions and mismatch-ID sets must match it exactly, but logits and margins
   produced by that earlier unpinned execution are not numerical comparators.
5. If the paired current runs fail `1e-5`, diagnose and fix deterministic
   execution without relaxing epsilon. No calibration search or optimization
   may begin until the paired baseline passes.

After passage, execute the already registered ladder and locked gates in order:
scalar read-side, diagonal read-side if needed, then the registered
program-local `W_V/W_O` fallback if needed. Evaluation is staged: task and
identity first; OWT only for a task/identity-passing candidate; migration,
re-extraction, FP32 encoder sanity, and four SMT properties only after all
earlier gates pass. Frozen programs, original GPT-2 parameter hashes, the OWT
threshold, all causal criteria, and the terminal rule remain unchanged.

