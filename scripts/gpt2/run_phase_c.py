#!/usr/bin/env python3
"""Run the complete GPT-2 Phase C pipeline with resumable artifact gates.

The runner is intentionally conservative:

* completed, structurally valid artifacts are reused;
* healing automatically resumes its newest Trainer checkpoint;
* all threshold sweeps use one process per GPU with dynamic scheduling;
* protocol-v1 evidence remains untouched in its original artifact paths;
* protocol v2 uses disjoint unique synthesis/gate manifests and checks programs
  jointly before any healing starts;
* healing uses the toy-proven core-aware objective and a complete unsampled
  lesion gate; a failed run stops the pipeline;
* interruption terminates only children owned by this runner, never the shell.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Iterable, Sequence


TASKS = ("quote_close", "bracket_type")
THRESHOLDS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
EXPECTED_VARIANTS = {
    "norm_variant": "none",
    "attn_variant": "sparsemax",
    "activation_variant": "leaky_relu",
}


class PipelineError(RuntimeError):
    """A fail-closed pipeline error that leaves completed artifacts intact."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def checkpoint_ready(path: Path) -> bool:
    return (
        (path / "config.json").is_file()
        and (path / "model_info.json").is_file()
        and (
            (path / "model.safetensors").is_file()
            or (path / "pytorch_model.bin").is_file()
        )
    )


def behavior_scan_gate(
    path: Path,
    expected_model: Path | None = None,
    *,
    expected_rows: int = 128,
    expected_manifest: Path | None = None,
) -> tuple[bool, list[str]]:
    try:
        report = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return False, [f"unreadable behavior scan: {error}"]
    failures = []
    if expected_model is not None:
        try:
            recorded_model = resolved(report["model_path"])
        except (KeyError, OSError, RuntimeError):
            recorded_model = None
        if recorded_model != expected_model:
            failures.append(
                f"model path is {recorded_model}, expected {expected_model}"
            )
    if expected_manifest is not None:
        recorded = report.get("domain_manifest")
        try:
            recorded_path = resolved(recorded) if recorded else None
        except (OSError, RuntimeError):
            recorded_path = None
        if recorded_path != expected_manifest:
            failures.append(
                f"domain manifest is {recorded_path}, expected {expected_manifest}"
            )
        expected_sha = hashlib.sha256(expected_manifest.read_bytes()).hexdigest()
        if report.get("domain_manifest_sha256") != expected_sha:
            failures.append("domain manifest digest is stale or missing")
    results = report.get("results", {})
    for task in TASKS:
        task_result = results.get(task, {})
        if task_result.get("n_examples_used") != expected_rows:
            failures.append(
                f"{task} used {task_result.get('n_examples_used')} of "
                f"{expected_rows} examples"
            )
        if float(task_result.get("binary_accuracy", math.nan)) != 1.0:
            failures.append(
                f"{task} accuracy against P(x)="
                f"{task_result.get('binary_accuracy')!r}; required 1.0"
            )
    return not failures, failures


def circuit_artifact_complete(
    path: Path,
    *,
    task: str,
    threshold: float,
    model_path: Path,
    domain_manifest: Path | None = None,
) -> bool:
    try:
        circuit = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    recorded_model = circuit.get("model_path")
    try:
        model_matches = recorded_model is not None and resolved(recorded_model) == model_path
    except (OSError, RuntimeError):
        model_matches = False
    domain_matches = True
    expected_rows = 128
    if domain_manifest is not None:
        try:
            manifest = load_json(domain_manifest)
            expected_rows = int(manifest["summary"][task]["rows"])
            expected_sha = hashlib.sha256(domain_manifest.read_bytes()).hexdigest()
            domain_matches = (
                circuit.get("domain", {}).get("manifest_sha256") == expected_sha
                and circuit.get("domain", {}).get("protocol_id")
                == "gpt2_behavior_domain_v2"
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            domain_matches = False
    return (
        model_matches
        and circuit.get("task") == task
        and circuit.get("metric") == "candidate_kl"
        and math.isclose(float(circuit.get("threshold", math.nan)), threshold)
        and float(circuit.get("min_agreement", math.nan)) == 1.0
        and int(circuit.get("n_examples", -1)) == expected_rows
        and domain_matches
        and circuit.get("granularity") == "head"
        and int(circuit.get("n_heads", -1)) > 1
        and isinstance(circuit.get("edges"), list)
        and isinstance(circuit.get("scores"), dict)
    )


def selected_circuit_complete(
    path: Path,
    selection_path: Path,
    model_path: Path,
    required_heads: set[str] | None = None,
) -> bool:
    try:
        circuit = load_json(path)
        selection = load_json(selection_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    try:
        recorded_model = resolved(circuit["model_path"])
    except (KeyError, OSError, RuntimeError):
        recorded_model = None
    nodes = set()
    for edge in circuit.get("edges", []):
        if isinstance(edge, dict):
            nodes.update((edge.get("source"), edge.get("target")))
        elif isinstance(edge, list) and len(edge) == 2:
            nodes.update(edge)
    return (
        recorded_model == model_path
        and circuit.get("granularity") == "head"
        and isinstance(circuit.get("edges"), list)
        and float(selection.get("projected_agreement", math.nan)) == 1.0
        and selection.get("task") == circuit.get("task")
        and (required_heads is None or required_heads <= nodes)
    )


def synthesis_state(
    output_dir: Path, base_model: Path, domain_manifest: Path | None = None
) -> str:
    result_path = output_dir / "synthesis_results.json"
    programs_path = output_dir / "programs.json"
    if not result_path.is_file() or not programs_path.is_file():
        return "missing"
    try:
        result = load_json(result_path)
        programs = load_json(programs_path)
        model_matches = resolved(result["model_path"]) == base_model
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return "malformed"
    domain_matches = True
    if domain_manifest is not None:
        expected_sha = hashlib.sha256(domain_manifest.read_bytes()).hexdigest()
        domain_matches = (
            result.get("domain_manifest") == str(domain_manifest)
            and all(
                result.get("domain", {})
                .get(task, {})
                .get("manifest_sha256")
                == expected_sha
                for task in TASKS
            )
        )
    if not model_matches or not domain_matches:
        return "malformed"
    if result.get("success") and programs:
        return "passed"
    return "failed"


def behavior_domains_complete(directory: Path, config_path: Path | None = None) -> bool:
    try:
        synthesis = load_json(directory / "synthesis.json")
        gate = load_json(directory / "gate.json")
        legacy = load_json(directory / "legacy_regression.json")
        index = load_json(directory / "manifest_index.json")
        if index.get("protocol_id") != "gpt2_behavior_domain_v2":
            return False
        if config_path is not None:
            config = load_json(config_path)
            config_digest = hashlib.sha256(
                json.dumps(
                    config, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            if index.get("config_sha256") != config_digest:
                return False
            for split in ("synthesis", "gate", "legacy_regression"):
                path = directory / f"{split}.json"
                if (
                    index.get("manifests", {}).get(split, {}).get("sha256")
                    != hashlib.sha256(path.read_bytes()).hexdigest()
                ):
                    return False
        synthesis_prompts = set()
        gate_prompts = set()
        for task in TASKS:
            synthesis_rows = synthesis["examples"][task]
            gate_rows = gate["examples"][task]
            legacy_rows = legacy["examples"][task]
            if len(synthesis_rows) != 256 or len(gate_rows) != 256:
                return False
            if len({row["prompt"] for row in synthesis_rows}) != 256:
                return False
            if len({row["prompt"] for row in gate_rows}) != 256:
                return False
            if len(legacy_rows) != 16 or len(
                {row["prompt"] for row in legacy_rows}
            ) != 16:
                return False
            synthesis_prompts.update(row["prompt"] for row in synthesis_rows)
            gate_prompts.update(row["prompt"] for row in gate_rows)
        return synthesis_prompts.isdisjoint(gate_prompts)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def joint_program_state(
    output_dir: Path,
    synthesis_manifest: Path | None = None,
    gate_manifest: Path | None = None,
) -> str:
    report_path = output_dir / "joint_program_report.json"
    programs_path = output_dir / "programs_selected.json"
    if not report_path.is_file() or not programs_path.is_file():
        return "missing"
    try:
        report = load_json(report_path)
        programs = load_json(programs_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return "malformed"
    if not programs or not isinstance(report.get("final_gate"), dict):
        return "failed"
    if synthesis_manifest is not None and gate_manifest is not None:
        synthesis_sha = hashlib.sha256(synthesis_manifest.read_bytes()).hexdigest()
        gate_sha = hashlib.sha256(gate_manifest.read_bytes()).hexdigest()
        if not all(
            report.get("synthesis_manifest", {})
            .get(task, {})
            .get("manifest_sha256")
            == synthesis_sha
            and report.get("gate_manifest", {})
            .get(task, {})
            .get("manifest_sha256")
            == gate_sha
            for task in TASKS
        ):
            return "malformed"
    return "passed" if report.get("success") is True else "failed"


def healing_state(
    output_dir: Path,
    reference_perplexity: float,
    train_manifest: Path | None = None,
    gate_manifest: Path | None = None,
) -> str:
    result_path = output_dir / "healing_results.json"
    if not result_path.is_file():
        return "missing"
    try:
        result = load_json(result_path)
        recorded_reference = float(result["reference_eval_perplexity"])
        final_perplexity = float(result["final_eval_perplexity"])
        budget = float(result["perplexity_budget"])
        agreements = result["final_projected_agreement"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return "malformed"
    structurally_valid = (
        math.isclose(recorded_reference, reference_perplexity, rel_tol=0.0, abs_tol=1e-9)
        and all(task in agreements for task in TASKS)
        and all(math.isfinite(float(agreements[task])) for task in TASKS)
        and math.isfinite(final_perplexity)
        and math.isfinite(budget)
        and result.get("reference_target")
        == "explicit_reference_program_P(x)"
        and isinstance(result.get("behavior_train_manifest"), str)
        and isinstance(result.get("behavior_gate_manifest"), str)
        and result.get("migration_pass") in {True, False}
        and result.get("suppression_coverage_pass") in {True, False}
        and checkpoint_ready(output_dir)
    )
    if train_manifest is not None and gate_manifest is not None:
        train_sha = hashlib.sha256(train_manifest.read_bytes()).hexdigest()
        gate_sha = hashlib.sha256(gate_manifest.read_bytes()).hexdigest()
        structurally_valid = structurally_valid and all(
            result.get("behavior_domain", {})
            .get("train", {})
            .get(task, {})
            .get("manifest_sha256")
            == train_sha
            and result.get("behavior_domain", {})
            .get("gate", {})
            .get(task, {})
            .get("manifest_sha256")
            == gate_sha
            for task in TASKS
        )
    if not structurally_valid:
        return "malformed"
    gates_pass = (
        all(float(agreements[task]) == 1.0 for task in TASKS)
        and final_perplexity <= budget
        and result.get("migration_pass") is True
        and result.get("suppression_coverage_pass") is True
    )
    if result.get("success") is not gates_pass:
        return "malformed"
    return "passed" if gates_pass else "failed"


def migration_state(path: Path) -> str:
    if not path.is_file():
        return "missing"
    try:
        report = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return "malformed"
    if not isinstance(report.get("tasks"), dict):
        return "malformed"
    return "passed" if report.get("migration_pass") is True else "failed"


def smt_sanity_state(path: Path) -> str:
    if not path.is_file():
        return "missing"
    try:
        report = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return "malformed"
    return "passed" if report.get("status") == "PASSED" else "failed"


def scaling_complete(path: Path, task: str) -> bool:
    try:
        summary = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return summary.get("task") == task and bool(summary.get("attempts"))


def tail(path: Path, lines: int = 60) -> str:
    try:
        content = path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])


class PhaseCRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = resolved(args.repo_root)
        self.artifacts = self.root / "artifacts"
        self.run_dir = self.artifacts / "gpt2-phase-c-v2-run"
        self.log_dir = self.run_dir / "logs"
        self.status_path = self.run_dir / "run_status.json"
        self.base_model = self.resolve_from_root(args.base_model)
        self.dataset = self.resolve_from_root(args.processed_dataset_dir)
        self.gpus = tuple(int(value) for value in args.gpus.split(",") if value.strip())
        self.heal_config = self.run_dir / "gpt2_program_healing_h100.json"
        self.domain_dir = self.artifacts / "gpt2-behavior-domains-v2"
        self.synthesis_manifest = self.domain_dir / "synthesis.json"
        self.gate_manifest = self.domain_dir / "gate.json"
        self.legacy_manifest = self.domain_dir / "legacy_regression.json"
        self.children: set[subprocess.Popen] = set()
        self.children_lock = threading.Lock()
        self.print_lock = threading.Lock()
        self.cancelled = threading.Event()
        self.lock_handle = None
        self.status = self._load_status()

    def resolve_from_root(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.root / path
        return path.resolve()

    def _load_status(self) -> dict:
        try:
            status = load_json(self.status_path)
        except (OSError, ValueError, json.JSONDecodeError):
            status = {}
        return {
            "schema_version": 1,
            "status": status.get("status", "pending"),
            "stage": status.get("stage", "not_started"),
            "started_at_utc": status.get("started_at_utc", utc_now()),
            "updated_at_utc": utc_now(),
            "completed_stages": list(status.get("completed_stages", [])),
            "base_model": str(self.base_model),
            "processed_dataset_dir": str(self.dataset),
            "gpus": list(self.gpus),
        }

    def log(self, message: str) -> None:
        with self.print_lock:
            timestamp = dt.datetime.now().astimezone().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}", flush=True)

    def update_status(self, *, status: str, stage: str, **extra) -> None:
        self.status.update(
            {
                "status": status,
                "stage": stage,
                "updated_at_utc": utc_now(),
                **extra,
            }
        )
        atomic_json(self.status_path, self.status)

    def complete_stage(self, stage: str) -> None:
        if stage not in self.status["completed_stages"]:
            self.status["completed_stages"].append(stage)
        self.update_status(status="running", stage=stage)

    def acquire_lock(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.run_dir / "runner.lock"
        self.lock_handle = lock_path.open("w", encoding="utf-8")
        try:
            fcntl.flock(
                self.lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError as error:
            raise PipelineError(
                f"Another Phase C runner owns {lock_path}; do not start a duplicate"
            ) from error
        self.lock_handle.write(f"pid={os.getpid()} started={utc_now()}\n")
        self.lock_handle.flush()

    def base_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            {
                "NCCL_NVLS_ENABLE": "0",
                "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": "1",
                "PYTHONUNBUFFERED": "1",
            }
        )
        return environment

    def _register(self, process: subprocess.Popen) -> None:
        with self.children_lock:
            self.children.add(process)

    def _unregister(self, process: subprocess.Popen) -> None:
        with self.children_lock:
            self.children.discard(process)

    def terminate_children(self) -> None:
        self.cancelled.set()
        with self.children_lock:
            children = list(self.children)
        for process in children:
            if process.poll() is not None:
                continue
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if all(process.poll() is not None for process in children):
                return
            time.sleep(0.25)
        for process in children:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass

    def run_logged(
        self,
        command: Sequence[str],
        *,
        log_name: str,
        extra_environment: dict[str, str] | None = None,
        accepted: Iterable[int] = (0,),
        echo: bool = True,
    ) -> int:
        if self.cancelled.is_set():
            raise PipelineError("Pipeline cancellation requested")
        log_path = self.log_dir / log_name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        environment = self.base_environment()
        if extra_environment:
            environment.update(extra_environment)
        printable = " ".join(command)
        self.log(f"RUN: {printable}")
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"\n[{utc_now()}] RUN {printable}\n")
            log_handle.flush()
            process = subprocess.Popen(
                list(command),
                cwd=self.root,
                env=environment,
                stdout=subprocess.PIPE if echo else log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            self._register(process)
            try:
                if echo:
                    assert process.stdout is not None
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log_handle.write(line)
                return_code = process.wait()
            finally:
                self._unregister(process)
        if return_code not in set(accepted):
            raise PipelineError(
                f"Command failed with status {return_code}: {printable}\n"
                f"Log: {log_path}\n{tail(log_path)}"
            )
        return return_code

    def _external_pipeline_processes(self) -> list[tuple[int, str]]:
        matches = []
        proc_root = Path("/proc")
        if not proc_root.is_dir():
            return matches
        owned = {process.pid for process in self.children}
        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid == os.getpid() or pid in owned:
                continue
            try:
                command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                    errors="replace"
                )
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            relevant = (
                "scripts/gpt2/extract.py" in command
                and "--extract_circuit" in command
            ) or "scripts/gpt2/select_sweep_circuit.py" in command
            if relevant:
                matches.append((pid, command.strip()))
        return matches

    def wait_for_external_extraction(self) -> None:
        quiet_since = None
        last_report = 0.0
        while True:
            matches = self._external_pipeline_processes()
            now = time.monotonic()
            if matches:
                quiet_since = None
                if now - last_report >= 60 or last_report == 0:
                    self.log(
                        "Waiting for the manually launched C2 extraction to finish; "
                        f"{len(matches)} extraction/selection processes remain"
                    )
                    last_report = now
                time.sleep(10)
                continue
            if quiet_since is None:
                quiet_since = now
            if now - quiet_since >= 20:
                return
            time.sleep(2)

    def python_command(self, script: str, *arguments: str) -> list[str]:
        return [sys.executable, script, *map(str, arguments)]

    def torchrun_command(self, script: str, *arguments: str) -> list[str]:
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={len(self.gpus)}",
            script,
            *map(str, arguments),
        ]

    def validate_base_model(self) -> None:
        if not checkpoint_ready(self.base_model):
            raise PipelineError(f"Incomplete Phase C base model: {self.base_model}")
        info = load_json(self.base_model / "model_info.json")
        mismatches = {
            key: (value, info.get(key))
            for key, value in EXPECTED_VARIANTS.items()
            if info.get(key) != value
        }
        if mismatches:
            raise PipelineError(f"Phase C base has wrong variants: {mismatches}")
        weights = self.base_model / "model.safetensors"
        if weights.is_file() and weights.stat().st_size < 100_000_000:
            raise PipelineError(f"Folded model weights are unexpectedly small: {weights}")

    def write_heal_config(self) -> None:
        source = load_json(self.root / "configs/gpt2_program_healing.json")
        source.update(
            {
                "train_batch_size_per_device": 8,
                "eval_batch_size_per_device": 8,
                "gradient_accumulation_steps": 4,
            }
        )
        atomic_json(self.heal_config, source)

    def run_preflight(self) -> None:
        self.validate_base_model()
        self.write_heal_config()
        self.run_logged(
            self.python_command(
                "scripts/gpt2/cluster_preflight.py",
                "--processed_dataset_dir",
                str(self.dataset),
                "--base_model",
                str(self.base_model),
                "--output",
                "artifacts/gpt2-cluster-preflight.json",
                "--expected_gpus",
                str(len(self.gpus)),
                "--minimum_free_gb",
                str(self.args.minimum_free_gb),
            ),
            log_name="00-preflight.log",
        )

    def ensure_behavior_domains(self) -> None:
        config_path = self.root / "configs/gpt2_behavior_domain_v2.json"
        if behavior_domains_complete(self.domain_dir, config_path):
            self.log("REUSE: locked unique behavior-domain v2 manifests")
            return
        self.run_logged(
            self.python_command(
                "scripts/gpt2/build_behavior_domains.py",
                "--tokenizer_path",
                str(self.base_model),
                "--output_dir",
                str(self.domain_dir),
                "--config",
                "configs/gpt2_behavior_domain_v2.json",
            ),
            log_name="00-build-behavior-domains.log",
        )
        if not behavior_domains_complete(self.domain_dir, config_path):
            raise PipelineError("Behavior-domain v2 manifests failed validation")

    def ensure_behavior_scan(self) -> None:
        for split, manifest in (
            ("synthesis", self.synthesis_manifest),
            ("gate", self.gate_manifest),
        ):
            output_root = self.artifacts / f"gpt2-circuits-v2/base-scan-{split}"
            path = output_root / "behavior_scan/behavior_scan.json"
            if path.is_file():
                passed, failures = behavior_scan_gate(
                    path,
                    self.base_model,
                    expected_rows=256,
                    expected_manifest=manifest,
                )
                if passed:
                    self.log(f"REUSE: base model is exact on v2 {split} split")
                    continue
                report = load_json(path)
                expected_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
                if report.get("domain_manifest_sha256") == expected_sha:
                    raise PipelineError(
                        f"Behavior v2 {split} gate failed: "
                        + "; ".join(failures)
                    )
                self.log(
                    f"REBUILD: stale {split} behavior scan uses an older manifest"
                )
            self.run_logged(
                self.python_command(
                    "scripts/gpt2/extract.py",
                    "--model_path",
                    str(self.base_model),
                    "--scan_behaviors",
                    "--domain_manifest",
                    str(manifest),
                    "--batch_size",
                    "16",
                    "--output_dir",
                    str(output_root),
                ),
                log_name=f"01-behavior-scan-{split}.log",
                extra_environment={
                    "CUDA_VISIBLE_DEVICES": str(self.gpus[0]),
                    "NVIDIA_TF32_OVERRIDE": "0",
                },
            )
            passed, failures = behavior_scan_gate(
                path,
                self.base_model,
                expected_rows=256,
                expected_manifest=manifest,
            )
            if not passed:
                raise PipelineError(
                    f"Behavior v2 {split} gate failed: " + "; ".join(failures)
                )

    def reference_metrics(self) -> dict:
        return load_json(self.artifacts / "gpt2-phase-c-reference-eval.json")

    def reference_complete(self) -> bool:
        path = self.artifacts / "gpt2-phase-c-reference-eval.json"
        try:
            result = load_json(path)
            removal = load_json(self.base_model / "removal_metrics.json")
            return (
                resolved(result["model_path"]) == self.base_model
                and resolved(result["processed_dataset_dir"]) == self.dataset
                and int(result["eval_examples"]) > 0
                and math.isclose(
                    float(result["eval_loss"]),
                    float(removal["post_fold_eval_loss"]),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False

    def ensure_reference(self) -> None:
        if self.reference_complete():
            result = self.reference_metrics()
            self.log(
                "REUSE: reference eval "
                f"loss={result['eval_loss']:.7f}, ppl={result['eval_perplexity']:.4f}"
            )
            return
        self.run_logged(
            self.torchrun_command(
                "scripts/gpt2/evaluate_checkpoint.py",
                "--model_path",
                str(self.base_model),
                "--processed_dataset_dir",
                str(self.dataset),
                "--output",
                "artifacts/gpt2-phase-c-reference-eval.json",
                "--batch_size_per_device",
                "8",
            ),
            log_name="02-reference-eval.log",
            extra_environment={"CUDA_VISIBLE_DEVICES": ",".join(map(str, self.gpus))},
        )
        if not self.reference_complete():
            raise PipelineError("Reference evaluation did not reproduce the folded model")

    def _sweep_jobs(
        self, model: Path, output_root: Path, *, announce_reuse: bool = True
    ) -> list[tuple[str, float]]:
        jobs = []
        for threshold in THRESHOLDS:
            for task in TASKS:
                circuit_path = output_root / f"{task}_t{threshold}/circuit.json"
                if circuit_artifact_complete(
                    circuit_path,
                    task=task,
                    threshold=threshold,
                    model_path=model,
                    domain_manifest=self.synthesis_manifest,
                ):
                    if announce_reuse:
                        self.log(
                            f"REUSE: {output_root.name} {task} threshold={threshold}"
                        )
                else:
                    jobs.append((task, threshold))
        return jobs

    def _sweep_worker(
        self,
        gpu: int,
        work: queue.Queue,
        model: Path,
        output_root: Path,
        label: str,
        failure: threading.Event,
    ) -> None:
        while not failure.is_set() and not self.cancelled.is_set():
            try:
                task, threshold = work.get_nowait()
            except queue.Empty:
                return
            output_dir = output_root / f"{task}_t{threshold}"
            log_name = f"sweeps/{label}-{task}-{threshold}.log"
            try:
                self.log(f"SWEEP START gpu={gpu} task={task} threshold={threshold}")
                self.run_logged(
                    self.python_command(
                        "scripts/gpt2/extract.py",
                        "--model_path",
                        str(model),
                        "--extract_circuit",
                        task,
                        "--n_examples",
                        "256",
                        "--domain_manifest",
                        str(self.synthesis_manifest),
                        "--threshold",
                        str(threshold),
                        "--metric",
                        "candidate_kl",
                        "--min_agreement",
                        "1.0",
                        "--trim_rounds",
                        "0",
                        "--output_dir",
                        str(output_dir),
                    ),
                    log_name=log_name,
                    extra_environment={
                        "CUDA_VISIBLE_DEVICES": str(gpu),
                        "NVIDIA_TF32_OVERRIDE": "0",
                    },
                    echo=False,
                )
                circuit_path = output_dir / "circuit.json"
                if not circuit_artifact_complete(
                    circuit_path,
                    task=task,
                    threshold=threshold,
                    model_path=model,
                    domain_manifest=self.synthesis_manifest,
                ):
                    raise PipelineError(
                        f"Sweep produced an invalid circuit artifact: {circuit_path}"
                    )
                self.log(f"SWEEP DONE  gpu={gpu} task={task} threshold={threshold}")
            except Exception:
                failure.set()
                raise
            finally:
                work.task_done()

    def run_sweeps(self, model: Path, output_root: Path, label: str) -> None:
        output_root.mkdir(parents=True, exist_ok=True)
        jobs = self._sweep_jobs(model, output_root)
        if not jobs:
            self.log(f"REUSE: all twelve {label} threshold circuits")
            return
        work: queue.Queue = queue.Queue()
        for job in jobs:
            work.put(job)
        failure = threading.Event()
        self.log(
            f"Launching {len(jobs)} missing {label} sweep jobs across "
            f"{min(len(jobs), len(self.gpus))} GPUs"
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(jobs), len(self.gpus))
        ) as executor:
            futures = [
                executor.submit(
                    self._sweep_worker,
                    gpu,
                    work,
                    model,
                    output_root,
                    label,
                    failure,
                )
                for gpu in self.gpus[: min(len(jobs), len(self.gpus))]
            ]
            pending = set(futures)
            while pending:
                done, pending = concurrent.futures.wait(
                    pending,
                    timeout=60,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    try:
                        future.result()
                    except Exception:
                        failure.set()
                        self.terminate_children()
                        for other in pending:
                            other.cancel()
                        raise
                if pending and not done:
                    complete = 12 - len(
                        self._sweep_jobs(
                            model, output_root, announce_reuse=False
                        )
                    )
                    self.log(f"{label} sweep heartbeat: {complete}/12 thresholds complete")
        remaining = self._sweep_jobs(
            model, output_root, announce_reuse=False
        )
        if remaining:
            raise PipelineError(f"Incomplete {label} sweep jobs: {remaining}")

    def ensure_selected_circuits(
        self,
        model: Path,
        sweep_root: Path,
        selected_root: Path,
        *,
        synthesis_results: Path | None = None,
        installed_programs_path: Path | None = None,
    ) -> None:
        selected_root.mkdir(parents=True, exist_ok=True)
        synthesis = None
        installed_programs = set()
        if synthesis_results is not None:
            synthesis = load_json(synthesis_results)
            if installed_programs_path is not None:
                installed_programs = set(load_json(installed_programs_path))
            else:
                installed_programs = set(synthesis.get("programs", {}))
        for task in TASKS:
            circuit_path = selected_root / task / "circuit.json"
            selection_path = selected_root / task / "selection.json"
            required_heads = None
            if synthesis is not None:
                required_heads = {
                    f"attn_{key.replace('.', '_h_')}"
                    for key, report in synthesis.get("tasks", {}).get(task, {}).items()
                    if report.get("accepted") and key in installed_programs
                }
            if selected_circuit_complete(
                circuit_path,
                selection_path,
                model_path=model,
                required_heads=required_heads,
            ):
                self.log(f"REUSE: selected {selected_root.name}/{task} circuit")
                continue
            command = self.python_command(
                "scripts/gpt2/select_sweep_circuit.py",
                "--sweep_dir",
                str(sweep_root),
                "--task",
                task,
                "--output_root",
                str(selected_root),
            )
            if synthesis_results is not None:
                command.extend(["--synthesis_results", str(synthesis_results)])
            if installed_programs_path is not None:
                command.extend(
                    ["--installed_programs", str(installed_programs_path)]
                )
            self.run_logged(
                command,
                log_name=f"select-{selected_root.name}-{task}.log",
            )
            if not selected_circuit_complete(
                circuit_path,
                selection_path,
                model,
                required_heads,
            ):
                raise PipelineError(f"Invalid selected circuit: {circuit_path}")

    def ensure_synthesis(self, selected_root: Path) -> None:
        output_dir = self.artifacts / "gpt2-programs-v2"
        state = synthesis_state(
            output_dir, self.base_model, self.synthesis_manifest
        )
        if state == "passed":
            self.log("REUSE: accepted restricted-DSL synthesis")
            return
        if state == "failed":
            raise PipelineError(
                "C3 produced no usable programs; preserve its counterexample report"
            )
        self.run_logged(
            self.python_command(
                "scripts/gpt2/synthesize_programs.py",
                "--model_path",
                str(self.base_model),
                "--circuit_root",
                str(selected_root),
                "--output_dir",
                str(output_dir),
                "--num_examples",
                "256",
                "--domain_manifest",
                str(self.synthesis_manifest),
                "--healable_agreement",
                "1.0",
            ),
            log_name="04-synthesize-programs.log",
            extra_environment={
                "CUDA_VISIBLE_DEVICES": str(self.gpus[0]),
                "NVIDIA_TF32_OVERRIDE": "0",
            },
        )
        state = synthesis_state(
            output_dir, self.base_model, self.synthesis_manifest
        )
        if state != "passed":
            raise PipelineError(f"C3 synthesis ended in state {state}")

    def ensure_joint_programs(self) -> Path:
        programs_root = self.artifacts / "gpt2-programs-v2"
        output_dir = programs_root / "joint"
        state = joint_program_state(
            output_dir, self.synthesis_manifest, self.gate_manifest
        )
        if state == "passed":
            self.log("REUSE: globally exact joint program subset on v2 gate")
            return output_dir / "programs_selected.json"
        if state == "failed":
            raise PipelineError(
                "C3 programs are not jointly exact on the untouched v2 gate; "
                "preserve the report and repair synthesis before healing"
            )
        return_code = self.run_logged(
            self.python_command(
                "scripts/gpt2/select_joint_program_subset.py",
                "--model_path",
                str(self.base_model),
                "--synthesis_results",
                str(programs_root / "synthesis_results.json"),
                "--circuit_root",
                str(self.artifacts / "gpt2-circuits-v2/base-selected"),
                "--programs",
                str(programs_root / "programs.json"),
                "--synthesis_manifest",
                str(self.synthesis_manifest),
                "--gate_manifest",
                str(self.gate_manifest),
                "--output_dir",
                str(output_dir),
                "--batch_size",
                "32",
            ),
            log_name="04b-select-joint-program-subset.log",
            extra_environment={
                "CUDA_VISIBLE_DEVICES": str(self.gpus[0]),
                "NVIDIA_TF32_OVERRIDE": "0",
            },
            accepted=(0, 2),
        )
        state = joint_program_state(
            output_dir, self.synthesis_manifest, self.gate_manifest
        )
        expected = "passed" if return_code == 0 else "failed"
        if state != expected or state != "passed":
            raise PipelineError(
                "Joint program composition did not pass both locked v2 tasks; "
                f"state={state}"
            )
        return output_dir / "programs_selected.json"

    def ensure_diagnostic(
        self,
        *,
        base_model: Path,
        programs: Path,
        circuits: Path,
        manifest: Path,
        output: Path,
        healed_model: Path | None = None,
    ) -> None:
        if output.is_file():
            try:
                report = load_json(output)
                expected_manifest_sha = hashlib.sha256(
                    manifest.read_bytes()
                ).hexdigest()
                domain_matches = all(
                    report["domain"][task]["manifest_sha256"]
                    == expected_manifest_sha
                    for task in TASKS
                )
                matches = (
                    resolved(report["base_model"]) == base_model
                    and resolved(report["programs"]) == programs
                    and resolved(report["circuit_root"]) == circuits
                    and domain_matches
                    and (
                        healed_model is None
                        or resolved(report["healed_model"]) == healed_model
                    )
                )
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                matches = False
            if matches:
                self.log(f"REUSE: program-composition diagnostic {output}")
                return
        command = self.python_command(
            "scripts/gpt2/diagnose_program_composition.py",
            "--base_model",
            str(base_model),
            "--programs",
            str(programs),
            "--circuit_root",
            str(circuits),
            "--domain_manifest",
            str(manifest),
            "--output",
            str(output),
            "--batch_size",
            "32",
        )
        if healed_model is not None:
            command.extend(["--healed_model", str(healed_model)])
        self.run_logged(
            command,
            log_name=f"diagnostic-{output.parent.name}-{output.stem}.log",
            extra_environment={
                "CUDA_VISIBLE_DEVICES": str(self.gpus[0]),
                "NVIDIA_TF32_OVERRIDE": "0",
            },
        )

    def ensure_preheal_diagnostics(
        self, base_selected: Path, selected_programs: Path
    ) -> None:
        # Preserve and diagnose v1 if its remote artifacts are present. Missing
        # v1 files do not block the v2 pipeline.
        old_programs = self.artifacts / "gpt2-programs/programs.json"
        old_circuits = self.artifacts / "gpt2-circuits/base-selected"
        old_healed = self.artifacts / "gpt2-program-healed"
        if old_programs.is_file() and old_circuits.is_dir():
            self.ensure_diagnostic(
                base_model=self.base_model,
                programs=old_programs,
                circuits=old_circuits,
                manifest=self.legacy_manifest,
                output=self.artifacts
                / "gpt2-program-diagnostics-v1/composition.json",
                healed_model=old_healed if checkpoint_ready(old_healed) else None,
            )
        self.ensure_diagnostic(
            base_model=self.base_model,
            programs=selected_programs,
            circuits=base_selected,
            manifest=self.synthesis_manifest,
            output=self.artifacts
            / "gpt2-programs-v2/preheal_composition_diagnostic.json",
        )

    def ensure_healing(
        self,
        output_dir: Path,
        reference_perplexity: float,
        *,
        ablation_aware: bool,
        programs: Path,
        circuit_root: Path | None = None,
        allow_gate_failure: bool = False,
    ) -> str:
        train_manifest = (
            self.synthesis_manifest if self.synthesis_manifest.is_file() else None
        )
        gate_manifest = self.gate_manifest if self.gate_manifest.is_file() else None
        state = healing_state(
            output_dir,
            reference_perplexity,
            train_manifest,
            gate_manifest,
        )
        label = "ablation-aware" if ablation_aware else "ordinary"
        if state == "passed":
            self.log(f"REUSE: passing {label} healing result")
            return state
        if state == "failed":
            if allow_gate_failure:
                self.log(
                    f"REUSE: completed {label} healing result that missed its "
                    "C4 gates; preserving it and selecting the fallback"
                )
                return state
            raise PipelineError(f"The completed {label} healing run failed its C4 gates")
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
            str(reference_perplexity),
            "--config",
            str(self.heal_config),
            "--behavior_train_manifest",
            str(self.synthesis_manifest),
            "--behavior_gate_manifest",
            str(self.gate_manifest),
        )
        if ablation_aware:
            if circuit_root is None:
                raise PipelineError("Ablation-aware healing requires base circuits")
            command.extend(
                ["--ablation_aware", "--circuit_root", str(circuit_root)]
            )
        try:
            self.run_logged(
                command,
                log_name=f"05-heal-{label}.log",
                extra_environment={
                    "CUDA_VISIBLE_DEVICES": ",".join(map(str, self.gpus))
                },
            )
        except PipelineError:
            # heal_programs writes the complete model-of-record and gate report,
            # then deliberately exits nonzero when either locked gate fails.
            # torchrun surfaces that child exit as status 1, so the artifact is
            # the only reliable distinction from an infrastructure crash.
            state = healing_state(
                output_dir,
                reference_perplexity,
                train_manifest,
                gate_manifest,
            )
            if state != "failed":
                raise
        state = healing_state(
            output_dir,
            reference_perplexity,
            train_manifest,
            gate_manifest,
        )
        if state == "passed":
            return state
        if state == "failed" and allow_gate_failure:
            self.log(
                f"Completed {label} healing missed its C4 gates; preserving "
                "the result and selecting the fallback"
            )
            return state
        raise PipelineError(f"{label.capitalize()} healing ended in state {state}")

    def ensure_migration(
        self,
        model: Path,
        circuit_root: Path,
        report_path: Path,
        *,
        require_pass: bool,
    ) -> str:
        state = migration_state(report_path)
        if state in {"passed", "failed"}:
            self.log(f"REUSE: migration report {report_path} ({state})")
        else:
            return_code = self.run_logged(
                self.python_command(
                    "scripts/gpt2/check_program_migration.py",
                    "--model_path",
                    str(model),
                    "--circuit_root",
                    str(circuit_root),
                    "--output",
                    str(report_path),
                    "--num_examples",
                    "256",
                    "--domain_manifest",
                    str(self.gate_manifest),
                ),
                log_name=f"migration-{model.name}.log",
                extra_environment={
                    "CUDA_VISIBLE_DEVICES": str(self.gpus[0]),
                    "NVIDIA_TF32_OVERRIDE": "0",
                },
                accepted=(0, 2),
            )
            state = migration_state(report_path)
            expected_state = "passed" if return_code == 0 else "failed"
            if state != expected_state:
                raise PipelineError(
                    f"Migration exit status and artifact disagree: {return_code}, {state}"
                )
        if require_pass and state != "passed":
            raise PipelineError(f"Required migration check failed: {report_path}")
        return state

    def choose_healed_model(
        self,
        reference_perplexity: float,
        base_selected: Path,
        selected_programs: Path,
    ) -> tuple[Path, Path]:
        synthesis_results = (
            self.artifacts / "gpt2-programs-v2/synthesis_results.json"
        )
        fallback_model = self.artifacts / "gpt2-program-healed-v2-core-aware"
        fallback_sweeps = self.artifacts / "gpt2-circuits-v2/healed-core-aware"
        fallback_selected = self.artifacts / (
            "gpt2-circuits-v2/healed-core-aware-selected"
        )
        self.ensure_healing(
            fallback_model,
            reference_perplexity,
            ablation_aware=True,
            programs=selected_programs,
            circuit_root=base_selected,
        )
        self.ensure_diagnostic(
            base_model=self.base_model,
            programs=selected_programs,
            circuits=base_selected,
            manifest=self.gate_manifest,
            output=fallback_model / "composition_diagnostic_gate.json",
            healed_model=fallback_model,
        )
        self.run_sweeps(fallback_model, fallback_sweeps, "healed-v2-core-aware")
        self.ensure_selected_circuits(
            fallback_model,
            fallback_sweeps,
            fallback_selected,
            synthesis_results=synthesis_results,
            installed_programs_path=selected_programs,
        )
        self.ensure_migration(
            fallback_model,
            fallback_selected,
            fallback_model / "migration_report.json",
            require_pass=True,
        )
        return fallback_model, fallback_selected

    def ensure_verification(self, model: Path, circuits: Path) -> None:
        for task in TASKS:
            verification_root = model / "verification" / task
            sanity_path = verification_root / "smt_sanity.json"
            sanity = smt_sanity_state(sanity_path)
            if sanity == "failed":
                raise PipelineError(f"Existing SMT sanity check failed: {sanity_path}")
            if sanity != "passed":
                self.run_logged(
                    self.python_command(
                        "scripts/gpt2/test_smt_encoder.py",
                        "--model_path",
                        str(model),
                        "--circuit_path",
                        str(circuits / task / "circuit.json"),
                        "--task",
                        task,
                        "--output_json",
                        str(sanity_path),
                    ),
                    log_name=f"07-smt-sanity-{task}.log",
                )
                if smt_sanity_state(sanity_path) != "passed":
                    raise PipelineError(f"SMT sanity check failed: {sanity_path}")
            else:
                self.log(f"REUSE: passing {task} SMT sanity check")

            scaling_path = verification_root / "scaling_summary.json"
            if scaling_complete(scaling_path, task):
                self.log(f"REUSE: {task} verification scaling summary")
                continue
            self.run_logged(
                self.python_command(
                    "scripts/gpt2/scale_verification.py",
                    "--model_path",
                    str(model),
                    "--circuit_path",
                    str(circuits / task / "circuit.json"),
                    "--task",
                    task,
                    "--start_length",
                    "3",
                    "--max_length",
                    "8",
                    "--output_root",
                    str(verification_root),
                ),
                log_name=f"07-scale-verification-{task}.log",
            )
            if not scaling_complete(scaling_path, task):
                raise PipelineError(f"Missing verification scaling summary: {scaling_path}")

    def build_cost_table(self, healed_model: Path) -> None:
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
                "artifacts/gpt2-programs-v2/synthesis_results.json",
                "--output_json",
                "artifacts/gpt2-unified-cost-table.json",
                "--output_csv",
                "artifacts/gpt2-unified-cost-table.csv",
            ),
            log_name="08-build-cost-table.log",
        )

    def sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def write_checksum(self, path: Path, checksum: str) -> Path:
        checksum_path = Path(str(path) + ".sha256")
        checksum_path.write_text(f"{checksum}  {path.name}\n", encoding="utf-8")
        return checksum_path

    def package(self, healed_model: Path, healed_circuits: Path) -> dict:
        evidence_archive = self.resolve_from_root(self.args.evidence_archive)
        model_archive = self.resolve_from_root(self.args.model_archive)
        evidence_archive.parent.mkdir(parents=True, exist_ok=True)
        model_archive.parent.mkdir(parents=True, exist_ok=True)

        candidates = [
            self.artifacts / "gpt2-cluster-preflight.json",
            self.artifacts / "gpt2-phase-c-base.json",
            self.artifacts / "gpt2-phase-c-reference-eval.json",
            self.artifacts / "gpt2-unified-cost-table.json",
            self.artifacts / "gpt2-unified-cost-table.csv",
            self.artifacts / "gpt2-norm-free/wikitext_eval_final.json",
            self.artifacts / "gpt2-behavior-domains-v2",
            self.artifacts / "gpt2-circuits-v2",
            self.artifacts / "gpt2-programs-v2",
            self.artifacts / "gpt2-program-diagnostics-v1",
            self.artifacts / "gpt2-program-healed-v2-core-aware",
            self.run_dir,
        ]
        relative_paths = [
            str(path.relative_to(self.root)) for path in candidates if path.exists()
        ]
        evidence_command = [
            "tar",
            "--exclude=*/checkpoint-*",
            "--exclude=*/model.safetensors",
            "--exclude=*/pytorch_model.bin",
            "--exclude=*/optimizer.pt",
            "--exclude=*/scheduler.pt",
            "--exclude=*/scaler.pt",
            "--exclude=*/rng_state*.pth",
            "--exclude=*/training_args.bin",
            "--exclude=*.pt",
            "-czf",
            str(evidence_archive),
            *relative_paths,
        ]
        self.run_logged(evidence_command, log_name="09-package-evidence.log")

        model_command = [
            "tar",
            "--exclude=./checkpoint-*",
            "--exclude=./optimizer.pt",
            "--exclude=./scheduler.pt",
            "--exclude=./scaler.pt",
            "--exclude=./rng_state*.pth",
            "-cf",
            str(model_archive),
            "-C",
            str(healed_model),
            ".",
        ]
        self.run_logged(model_command, log_name="09-package-model.log")

        evidence_hash = self.sha256(evidence_archive)
        model_hash = self.sha256(model_archive)
        evidence_checksum = self.write_checksum(evidence_archive, evidence_hash)
        model_checksum = self.write_checksum(model_archive, model_hash)
        result = {
            "evidence_archive": str(evidence_archive),
            "evidence_sha256": evidence_hash,
            "evidence_checksum_file": str(evidence_checksum),
            "evidence_size_bytes": evidence_archive.stat().st_size,
            "model_archive": str(model_archive),
            "model_sha256": model_hash,
            "model_checksum_file": str(model_checksum),
            "model_size_bytes": model_archive.stat().st_size,
            "healed_model": str(healed_model),
            "healed_circuit_root": str(healed_circuits),
        }
        atomic_json(self.run_dir / "package_manifest.json", result)
        return result

    def stage(self, name: str, function):
        self.log(f"STAGE START: {name}")
        self.update_status(status="running", stage=name)
        result = function()
        self.complete_stage(name)
        self.log(f"STAGE DONE:  {name}")
        return result

    def execute(self) -> None:
        self.acquire_lock()
        os.chdir(self.root)
        self.update_status(status="running", stage="waiting_for_existing_extraction")
        self.wait_for_external_extraction()
        self.stage("preflight", self.run_preflight)
        self.stage("behavior_domain_v2", self.ensure_behavior_domains)
        self.stage("behavior_scan", self.ensure_behavior_scan)
        self.stage("reference_eval", self.ensure_reference)

        base_sweeps = self.artifacts / "gpt2-circuits-v2/base"
        base_selected = self.artifacts / "gpt2-circuits-v2/base-selected"
        self.stage(
            "base_circuit_sweeps",
            lambda: self.run_sweeps(self.base_model, base_sweeps, "base"),
        )
        self.stage(
            "base_circuit_selection",
            lambda: self.ensure_selected_circuits(
                self.base_model, base_sweeps, base_selected
            ),
        )
        self.stage("program_synthesis", lambda: self.ensure_synthesis(base_selected))
        selected_programs = self.stage(
            "joint_program_gate", self.ensure_joint_programs
        )
        self.stage(
            "preheal_diagnostics",
            lambda: self.ensure_preheal_diagnostics(
                base_selected, selected_programs
            ),
        )

        reference_perplexity = float(self.reference_metrics()["eval_perplexity"])
        healed_model, healed_circuits = self.stage(
            "healing_and_migration",
            lambda: self.choose_healed_model(
                reference_perplexity, base_selected, selected_programs
            ),
        )
        self.update_status(
            status="running",
            stage="healing_and_migration",
            healed_model=str(healed_model),
            healed_circuit_root=str(healed_circuits),
        )
        self.stage(
            "verification",
            lambda: self.ensure_verification(healed_model, healed_circuits),
        )
        self.stage("cost_table", lambda: self.build_cost_table(healed_model))
        package_result = self.stage(
            "packaging", lambda: self.package(healed_model, healed_circuits)
        )
        self.update_status(
            status="completed",
            stage="done",
            completed_at_utc=utc_now(),
            healed_model=str(healed_model),
            healed_circuit_root=str(healed_circuits),
            packages=package_result,
        )
        self.log("PHASE C COMPLETE")
        self.log(f"Evidence archive: {package_result['evidence_archive']}")
        self.log(f"Model archive:    {package_result['model_archive']}")


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", default=str(repository))
    parser.add_argument(
        "--base_model", default="artifacts/gpt2-norm-free"
    )
    parser.add_argument(
        "--processed_dataset_dir",
        default=os.environ.get(
            "PROCESSED_DATASET_DIR", "/dev/shm/openwebtext-gpt2-block1024"
        ),
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--minimum_free_gb", type=float, default=200.0)
    parser.add_argument(
        "--evidence_archive", default="/workspace/phase-c-evidence.tar.gz"
    )
    parser.add_argument(
        "--model_archive", default="/workspace/phase-c-healed-model.tar"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.gpus.strip():
        print("ERROR: --gpus must contain at least one GPU index", file=sys.stderr)
        return 2
    runner = PhaseCRunner(args)

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
        runner.log(
            "Phase C interrupted safely. Re-run the same command to reuse outputs "
            "and resume healing checkpoints."
        )
        return 130
    except Exception as error:
        runner.terminate_children()
        runner.update_status(
            status="failed",
            stage=runner.status["stage"],
            error=str(error),
            traceback=traceback.format_exc(),
        )
        runner.log(f"PHASE C STOPPED: {error}")
        runner.log(f"Status artifact: {runner.status_path}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
