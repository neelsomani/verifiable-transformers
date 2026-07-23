#!/usr/bin/env python3
"""Run the quote-close Phase-C bounded-domain symbolic-head continuation.

This runner does not reopen protocol v4's stopped held-out-generalization
track. It treats the frozen union of all 1,280 v4 prompts per task as a declared
finite specification domain D, runs only quote_close, and makes no claim about
prompts outside D.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import traceback
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.gpt2.behavior_domains import load_domain_manifest
from scripts.gpt2.run_phase_c import (
    PhaseCRunner,
    PipelineError,
    atomic_json,
    behavior_scan_gate,
    checkpoint_ready,
    load_json,
    migration_state,
    resolved,
    scaling_complete,
    smt_sanity_state,
    utc_now,
)


TASK = "quote_close"
TASKS = (TASK,)


class BoundedQuoteRunner(PhaseCRunner):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.run_dir = self.artifacts / "gpt2-phase-c-bounded-quote-run"
        self.log_dir = self.run_dir / "logs"
        self.status_path = self.run_dir / "run_status.json"
        self.heal_config = self.run_dir / "gpt2_program_healing_h100.json"
        self.domain_config = (
            self.root / "configs/gpt2_behavior_domain_bounded_v1.json"
        )
        self.domain_dir = (
            self.artifacts / "gpt2-behavior-domain-bounded-v1"
        )
        self.synthesis_manifest = self.domain_dir / "domain.json"
        self.gate_manifest = None
        self.status = self._load_status()

    @staticmethod
    def file_sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def bounded_manifest_complete(self) -> bool:
        try:
            config = load_json(self.domain_config)
            manifest = load_domain_manifest(self.synthesis_manifest)
            index = load_json(self.domain_dir / "manifest_index.json")
            return (
                manifest["protocol_id"]
                == "gpt2_behavior_domain_bounded_v1"
                and manifest["split"] == "bounded"
                and manifest.get("held_out_gate") is False
                and int(manifest["summary"][TASK]["rows"]) == 1280
                and manifest["summary"][TASK]["prompt_set_sha256"]
                == config["locked_prompt_set_sha256"][TASK]
                and index["manifest_sha256"]
                == self.file_sha256(self.synthesis_manifest)
                and index.get("held_out_gate") is False
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False

    def ensure_bounded_domain(self) -> None:
        self.run_logged(
            self.python_command(
                "scripts/gpt2/build_bounded_behavior_domain.py",
                "--config",
                str(self.domain_config),
                "--output",
                str(self.synthesis_manifest),
            ),
            log_name="00-build-bounded-domain.log",
        )
        if not self.bounded_manifest_complete():
            raise PipelineError("The locked 1,280-prompt bounded domain is invalid")

    def ensure_bounded_behavior_scan(self) -> None:
        expected_rows = int(
            load_json(self.synthesis_manifest)["summary"][TASK]["rows"]
        )
        output_root = self.artifacts / "gpt2-circuits-bounded-quote/base-scan"
        report_path = output_root / "behavior_scan/behavior_scan.json"
        if report_path.is_file():
            passed, _ = behavior_scan_gate(
                report_path,
                self.base_model,
                expected_rows=expected_rows,
                expected_manifest=self.synthesis_manifest,
            )
            if passed:
                self.log("REUSE: base model is exact on bounded D")
                return
        self.run_logged(
            self.python_command(
                "scripts/gpt2/extract.py",
                "--model_path",
                str(self.base_model),
                "--scan_behaviors",
                "--domain_manifest",
                str(self.synthesis_manifest),
                "--batch_size",
                "16",
                "--output_dir",
                str(output_root),
            ),
            log_name="01-behavior-scan-bounded.log",
            extra_environment={
                "CUDA_VISIBLE_DEVICES": str(self.gpus[0]),
                "NVIDIA_TF32_OVERRIDE": "0",
            },
        )
        passed, failures = behavior_scan_gate(
            report_path,
            self.base_model,
            expected_rows=expected_rows,
            expected_manifest=self.synthesis_manifest,
        )
        if not passed:
            raise PipelineError(
                "Base-model bounded-domain check failed: " + "; ".join(failures)
            )

    def ensure_quote_circuit(self, model: Path, label: str, root_name: str) -> Path:
        sweep_root = self.artifacts / root_name / label
        selected_root = self.artifacts / root_name / f"{label}-selected"
        self.run_sweeps(
            model, sweep_root, f"bounded-{label}", tasks=TASKS
        )
        self.ensure_selected_circuits(
            model,
            sweep_root,
            selected_root,
            selection_manifest=self.synthesis_manifest,
            candidate_manifests=(self.synthesis_manifest,),
            selection_strategy="minimum_edges",
            tasks=TASKS,
        )
        return selected_root

    def synthesis_complete(self, output_dir: Path, circuit_root: Path) -> bool:
        try:
            result = load_json(output_dir / "synthesis_results.json")
            programs = load_json(output_dir / "programs.json")
            expected_sha = self.file_sha256(self.synthesis_manifest)
            return (
                resolved(result["model_path"]) == self.base_model
                and resolved(result["circuit_root"]) == circuit_root
                and result.get("success") is True
                and set(result.get("tasks", {})) == {TASK}
                and result["domain"][TASK]["manifest_sha256"] == expected_sha
                and result["base_accuracy_against_reference"][TASK] == 1.0
                and bool(programs)
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False

    def ensure_synthesis(self, circuit_root: Path) -> Path:
        output_dir = self.artifacts / "gpt2-programs-bounded-quote"
        if self.synthesis_complete(output_dir, circuit_root):
            self.log("REUSE: bounded quote program synthesis")
            return output_dir
        rows = int(load_json(self.synthesis_manifest)["summary"][TASK]["rows"])
        self.run_logged(
            self.python_command(
                "scripts/gpt2/synthesize_programs.py",
                "--model_path",
                str(self.base_model),
                "--circuit_root",
                str(circuit_root),
                "--output_dir",
                str(output_dir),
                "--num_examples",
                str(rows),
                "--domain_manifest",
                str(self.synthesis_manifest),
                "--healable_agreement",
                "1.0",
                "--tasks",
                TASK,
            ),
            log_name="03-synthesize-quote-programs.log",
            extra_environment={
                "CUDA_VISIBLE_DEVICES": str(self.gpus[0]),
                "NVIDIA_TF32_OVERRIDE": "0",
            },
        )
        if not self.synthesis_complete(output_dir, circuit_root):
            raise PipelineError("Bounded quote synthesis produced no usable programs")
        return output_dir

    def selected_programs_complete(self, output_dir: Path) -> bool:
        try:
            report = load_json(output_dir / "joint_program_report.json")
            programs = load_json(output_dir / "programs_selected.json")
            expected_sha = self.file_sha256(self.synthesis_manifest)
            return (
                report.get("success") is True
                and report.get("mode") == "bounded_domain"
                and report.get("tasks") == [TASK]
                and report.get("require_all_circuit_heads") is True
                and report.get("all_circuit_heads_replaced") is True
                and report["bounded_manifest"][TASK]["manifest_sha256"]
                == expected_sha
                and report["final_bounded"]["exact_full_and_circuit"] is True
                and bool(programs)
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False

    def ensure_selected_programs(
        self, circuit_root: Path, synthesis_root: Path
    ) -> Path:
        output_dir = synthesis_root / "bounded-selection"
        if self.selected_programs_complete(output_dir):
            self.log("REUSE: fully replaced exact quote program set on D")
            selected = output_dir / "programs_selected.json"
            self.ensure_zero_bilinear_coverage(circuit_root, selected)
            return selected
        return_code = self.run_logged(
            self.python_command(
                "scripts/gpt2/select_joint_program_subset.py",
                "--model_path",
                str(self.base_model),
                "--synthesis_results",
                str(synthesis_root / "synthesis_results.json"),
                "--circuit_root",
                str(circuit_root),
                "--programs",
                str(synthesis_root / "programs.json"),
                "--bounded_manifest",
                str(self.synthesis_manifest),
                "--output_dir",
                str(output_dir),
                "--batch_size",
                "32",
                "--minimum_program_heads",
                "1",
                "--minimum_program_heads_per_task",
                "1",
                "--require_all_circuit_heads",
                "--tasks",
                TASK,
            ),
            log_name="04-select-bounded-quote-programs.log",
            extra_environment={
                "CUDA_VISIBLE_DEVICES": str(self.gpus[0]),
                "NVIDIA_TF32_OVERRIDE": "0",
            },
            accepted=(0, 2),
        )
        if return_code != 0 or not self.selected_programs_complete(output_dir):
            raise PipelineError(
                "Quote programs did not replace every retained attention head "
                "while preserving exact full and circuit-only behavior on D"
            )
        selected = output_dir / "programs_selected.json"
        self.ensure_zero_bilinear_coverage(circuit_root, selected)
        return selected

    def ensure_zero_bilinear_coverage(
        self, circuit_root: Path, programs_path: Path
    ) -> dict:
        circuit = load_json(circuit_root / TASK / "circuit.json")
        programs = load_json(programs_path)
        retained = sorted(
            {
                node
                for edge in circuit["edges"]
                for node in (
                    (edge["source"], edge["target"])
                    if isinstance(edge, dict)
                    else edge
                )
                if node.startswith("attn_")
            }
        )
        installed = {
            f"attn_{key.replace('.', '_h_')}" for key in programs
        }
        neural = sorted(set(retained) - installed)
        report = {
            "task": TASK,
            "circuit": str(circuit_root / TASK / "circuit.json"),
            "programs": str(programs_path),
            "retained_attention_heads": retained,
            "installed_program_heads": sorted(installed),
            "retained_neural_attention_heads": neural,
            "zero_active_neural_attention_bilinear_terms": not neural,
            "pass": bool(retained) and not neural,
        }
        atomic_json(circuit_root / TASK / "symbolic_coverage.json", report)
        if not report["pass"]:
            raise PipelineError(
                "Selected quote circuit retains neural attention heads: "
                + ", ".join(neural or ["no retained program head"])
            )
        return report

    def healing_complete(self, output_dir: Path) -> bool:
        try:
            result = load_json(output_dir / "healing_results.json")
            return (
                checkpoint_ready(output_dir)
                and result.get("success") is True
                and result.get("behavior_domain_mode") == "bounded_domain"
                and result.get("tasks") == [TASK]
                and result.get("final_projected_agreement", {}).get(TASK) == 1.0
                and float(result["final_eval_perplexity"])
                <= 28.617593822841776
                and result.get("migration_pass") is True
                and result.get("suppression_coverage_pass") is True
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False

    def ensure_healing(self, circuit_root: Path, programs: Path) -> Path:
        output_dir = self.artifacts / "gpt2-program-healed-bounded-quote-core-aware"
        if self.healing_complete(output_dir):
            self.log("REUSE: passing bounded quote healing result")
            return output_dir
        command = self.torchrun_command(
            "scripts/gpt2/heal_programs.py",
            "--model_path",
            str(self.base_model),
            "--programs",
            str(programs),
            "--processed_dataset_dir",
            str(self.dataset),
            "--output_dir",
            str(output_dir),
            "--reference_eval_perplexity",
            "24.67033950244981",
            "--config",
            str(self.heal_config),
            "--bounded_behavior_manifest",
            str(self.synthesis_manifest),
            "--ablation_aware",
            "--circuit_root",
            str(circuit_root),
            "--tasks",
            TASK,
        )
        try:
            self.run_logged(
                command,
                log_name="05-heal-bounded-quote-core-aware.log",
                extra_environment={
                    "CUDA_VISIBLE_DEVICES": ",".join(map(str, self.gpus))
                },
            )
        except PipelineError:
            if not (output_dir / "healing_results.json").is_file():
                raise
        if not self.healing_complete(output_dir):
            raise PipelineError(
                "Bounded quote healing completed without passing every locked gate"
            )
        return output_dir

    def ensure_healed_quote_circuit(
        self, model: Path, synthesis_root: Path, programs: Path
    ) -> Path:
        sweep_root = self.artifacts / "gpt2-circuits-bounded-quote/healed"
        selected_root = (
            self.artifacts / "gpt2-circuits-bounded-quote/healed-selected"
        )
        self.run_sweeps(model, sweep_root, "bounded-healed", tasks=TASKS)
        self.ensure_selected_circuits(
            model,
            sweep_root,
            selected_root,
            synthesis_results=synthesis_root / "synthesis_results.json",
            installed_programs_path=programs,
            selection_manifest=self.synthesis_manifest,
            candidate_manifests=(self.synthesis_manifest,),
            selection_strategy="minimum_edges",
            tasks=TASKS,
        )
        self.ensure_zero_bilinear_coverage(selected_root, programs)
        return selected_root

    def ensure_bounded_migration(self, model: Path, circuits: Path) -> None:
        report_path = model / "migration_report.json"
        state = migration_state(report_path)
        if state != "passed":
            rows = int(
                load_json(self.synthesis_manifest)["summary"][TASK]["rows"]
            )
            self.run_logged(
                self.python_command(
                    "scripts/gpt2/check_program_migration.py",
                    "--model_path",
                    str(model),
                    "--circuit_root",
                    str(circuits),
                    "--output",
                    str(report_path),
                    "--num_examples",
                    str(rows),
                    "--domain_manifest",
                    str(self.synthesis_manifest),
                    "--tasks",
                    TASK,
                ),
                log_name="06-migration-bounded-quote.log",
                extra_environment={
                    "CUDA_VISIBLE_DEVICES": str(self.gpus[0]),
                    "NVIDIA_TF32_OVERRIDE": "0",
                },
                accepted=(0, 2),
            )
        if migration_state(report_path) != "passed":
            raise PipelineError("Bounded quote migration/lesion gate failed")

    def ensure_quote_verification(self, model: Path, circuits: Path) -> None:
        verification_root = model / "verification" / TASK
        sanity_path = verification_root / "smt_sanity.json"
        if smt_sanity_state(sanity_path) != "passed":
            self.run_logged(
                self.python_command(
                    "scripts/gpt2/test_smt_encoder.py",
                    "--model_path",
                    str(model),
                    "--circuit_path",
                    str(circuits / TASK / "circuit.json"),
                    "--task",
                    TASK,
                    "--output_json",
                    str(sanity_path),
                ),
                log_name="07-smt-sanity-quote.log",
            )
        if smt_sanity_state(sanity_path) != "passed":
            raise PipelineError("Quote SMT encoder sanity check failed")
        scaling_path = verification_root / "scaling_summary.json"
        if not scaling_complete(scaling_path, TASK):
            self.run_logged(
                self.python_command(
                    "scripts/gpt2/scale_verification.py",
                    "--model_path",
                    str(model),
                    "--circuit_path",
                    str(circuits / TASK / "circuit.json"),
                    "--task",
                    TASK,
                    "--start_length",
                    "3",
                    "--max_length",
                    "8",
                    "--output_root",
                    str(verification_root),
                ),
                log_name="07-scale-verification-quote.log",
            )
        if not scaling_complete(scaling_path, TASK):
            raise PipelineError("Missing quote verification scaling summary")

    def build_bounded_cost_table(self, healed_model: Path, synthesis_root: Path) -> None:
        self.run_logged(
            self.python_command(
                "scripts/gpt2/build_cost_table.py",
                "--removal_metrics",
                "artifacts/gpt2-norm-free/removal_metrics.json",
                "--removal_wikitext_metrics",
                "artifacts/gpt2-norm-free/wikitext_eval_final.json",
                "--program_metrics",
                str(healed_model / "healing_results.json"),
                "--synthesis_metrics",
                str(synthesis_root / "synthesis_results.json"),
                "--output_json",
                "artifacts/gpt2-unified-cost-table.json",
                "--output_csv",
                "artifacts/gpt2-unified-cost-table.csv",
            ),
            log_name="08-build-cost-table.log",
        )

    def package_bounded(
        self, healed_model: Path, healed_circuits: Path
    ) -> dict:
        evidence_archive = self.resolve_from_root(self.args.evidence_archive)
        model_archive = self.resolve_from_root(self.args.model_archive)
        evidence_archive.parent.mkdir(parents=True, exist_ok=True)
        model_archive.parent.mkdir(parents=True, exist_ok=True)
        candidates = [
            self.domain_config,
            self.domain_dir,
            self.artifacts / "gpt2-circuits-bounded-quote",
            self.artifacts / "gpt2-programs-bounded-quote",
            healed_model,
            self.run_dir,
            self.artifacts / "gpt2-unified-cost-table.json",
            self.artifacts / "gpt2-unified-cost-table.csv",
        ]
        relative = [
            str(path.relative_to(self.root)) for path in candidates if path.exists()
        ]
        self.run_logged(
            [
                "tar",
                "--exclude=*/checkpoint-*",
                "--exclude=*/model.safetensors",
                "--exclude=*/optimizer.pt",
                "--exclude=*/scheduler.pt",
                "--exclude=*/rng_state*.pth",
                "--exclude=*/training_args.bin",
                "-czf",
                str(evidence_archive),
                *relative,
            ],
            log_name="09-package-evidence.log",
        )
        self.run_logged(
            [
                "tar",
                "--exclude=./checkpoint-*",
                "--exclude=./optimizer.pt",
                "--exclude=./scheduler.pt",
                "--exclude=./rng_state*.pth",
                "-cf",
                str(model_archive),
                "-C",
                str(healed_model),
                ".",
            ],
            log_name="09-package-model.log",
        )
        evidence_sha = self.file_sha256(evidence_archive)
        model_sha = self.file_sha256(model_archive)
        evidence_checksum = self.write_checksum(evidence_archive, evidence_sha)
        model_checksum = self.write_checksum(model_archive, model_sha)
        result = {
            "evidence_archive": str(evidence_archive),
            "evidence_sha256": evidence_sha,
            "evidence_checksum_file": str(evidence_checksum),
            "evidence_size_bytes": evidence_archive.stat().st_size,
            "model_archive": str(model_archive),
            "model_sha256": model_sha,
            "model_checksum_file": str(model_checksum),
            "model_size_bytes": model_archive.stat().st_size,
            "healed_model": str(healed_model),
            "healed_circuit_root": str(healed_circuits),
        }
        atomic_json(self.run_dir / "package_manifest.json", result)
        return result

    def execute(self) -> None:
        self.acquire_lock()
        os.chdir(self.root)
        self.stage("preflight", self.run_preflight)
        self.stage("bounded_domain", self.ensure_bounded_domain)
        self.stage("bounded_behavior_scan", self.ensure_bounded_behavior_scan)
        self.stage("reference_eval", self.ensure_reference)
        base_circuits = self.stage(
            "quote_circuit_extraction",
            lambda: self.ensure_quote_circuit(
                self.base_model,
                "base",
                "gpt2-circuits-bounded-quote",
            ),
        )
        synthesis_root = self.stage(
            "quote_program_synthesis",
            lambda: self.ensure_synthesis(base_circuits),
        )
        selected_programs = self.stage(
            "bounded_program_composition",
            lambda: self.ensure_selected_programs(base_circuits, synthesis_root),
        )
        healed_model = self.stage(
            "bounded_core_aware_healing",
            lambda: self.ensure_healing(base_circuits, selected_programs),
        )
        healed_circuits = self.stage(
            "healed_quote_circuit_extraction",
            lambda: self.ensure_healed_quote_circuit(
                healed_model, synthesis_root, selected_programs
            ),
        )
        self.stage(
            "bounded_migration",
            lambda: self.ensure_bounded_migration(healed_model, healed_circuits),
        )
        self.stage(
            "quote_verification",
            lambda: self.ensure_quote_verification(healed_model, healed_circuits),
        )
        self.stage(
            "cost_table",
            lambda: self.build_bounded_cost_table(healed_model, synthesis_root),
        )
        package = self.stage(
            "packaging",
            lambda: self.package_bounded(healed_model, healed_circuits),
        )
        self.update_status(
            status="completed",
            stage="done",
            completed_at_utc=utc_now(),
            claim_scope=(
                "quote_close exact only on the declared 1,280-prompt bounded D; "
                "no held-out-generalization claim"
            ),
            healed_model=str(healed_model),
            healed_circuit_root=str(healed_circuits),
            packages=package,
        )
        self.log("BOUNDED QUOTE CONTINUATION COMPLETE")
        self.log(f"Evidence archive: {package['evidence_archive']}")
        self.log(f"Model archive:    {package['model_archive']}")


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", default=str(repository))
    parser.add_argument("--base_model", default="artifacts/gpt2-norm-free")
    parser.add_argument(
        "--processed_dataset_dir",
        default=os.environ.get(
            "PROCESSED_DATASET_DIR", "/dev/shm/openwebtext-gpt2-block1024"
        ),
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--minimum_free_gb", type=float, default=200.0)
    parser.add_argument(
        "--evidence_archive",
        default="/workspace/phase-c-bounded-quote-evidence.tar.gz",
    )
    parser.add_argument(
        "--model_archive",
        default="/workspace/phase-c-bounded-quote-model.tar",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.gpus.strip():
        print("ERROR: --gpus must contain at least one GPU index", file=sys.stderr)
        return 2
    runner = BoundedQuoteRunner(args)

    def handle_signal(signum, _frame):
        runner.log(f"Received signal {signum}; stopping owned child processes")
        runner.terminate_children()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    try:
        runner.execute()
        return 0
    except KeyboardInterrupt:
        runner.update_status(status="interrupted", stage=runner.status["stage"])
        runner.log("Interrupted safely; rerun the same command to resume.")
        return 130
    except Exception as error:
        runner.terminate_children()
        runner.update_status(
            status="failed",
            stage=runner.status["stage"],
            error=str(error),
            traceback=traceback.format_exc(),
        )
        runner.log(f"BOUNDED QUOTE CONTINUATION STOPPED: {error}")
        runner.log(f"Status artifact: {runner.status_path}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
