#!/usr/bin/env python3
"""Fail-fast validation for the 8-GPU A4/C execution environment."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import torch


REQUIRED_FILES = (
    "configs/sparsemax_leaky_layernorm.json",
    "configs/gpt2_layernorm_removal.json",
    "configs/gpt2_program_healing.json",
    "scripts/gpt2/train.py",
    "scripts/gpt2/remove_layernorm.py",
    "scripts/gpt2/evaluate_checkpoint.py",
    "scripts/gpt2/extract.py",
    "scripts/gpt2/synthesize_programs.py",
    "scripts/gpt2/heal_programs.py",
    "scripts/gpt2/check_program_migration.py",
    "scripts/gpt2/test_smt_encoder.py",
    "scripts/gpt2/scale_verification.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed_dataset_dir", required=True)
    parser.add_argument("--bandnorm_model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected_gpus", type=int, default=8)
    parser.add_argument("--minimum_free_gb", type=float, default=200.0)
    parser.add_argument(
        "--allow_missing_bandnorm",
        action="store_true",
        help="Development-only; the real run needs the fallback checkpoint.",
    )
    return parser.parse_args()


def checkpoint_ready(path: str) -> bool:
    return (
        os.path.isfile(os.path.join(path, "config.json"))
        and os.path.isfile(os.path.join(path, "model_info.json"))
        and (
            os.path.isfile(os.path.join(path, "model.safetensors"))
            or os.path.isfile(os.path.join(path, "pytorch_model.bin"))
        )
    )


def processed_dataset_ready(path: str) -> bool:
    return os.path.isdir(path) and (
        os.path.isfile(os.path.join(path, "_READY"))
        or (
            os.path.isfile(os.path.join(path, "dataset_dict.json"))
            and os.path.isdir(os.path.join(path, "train"))
            and os.path.isdir(os.path.join(path, "validation"))
        )
    )


def main() -> None:
    args = parse_args()
    checks = {}
    checks["python"] = {
        "value": sys.version,
        "passed": sys.version_info >= (3, 10),
    }
    gpu_count = torch.cuda.device_count()
    checks["gpu_count"] = {
        "value": gpu_count,
        "expected": args.expected_gpus,
        "passed": gpu_count == args.expected_gpus,
        "devices": [torch.cuda.get_device_name(index) for index in range(gpu_count)],
    }
    checks["distributed"] = {
        "available": torch.distributed.is_available(),
        "passed": torch.distributed.is_available(),
    }
    missing_files = [path for path in REQUIRED_FILES if not os.path.isfile(path)]
    checks["pipeline_files"] = {
        "missing": missing_files,
        "passed": not missing_files,
    }
    invalid_json = []
    for path in (
        "configs/sparsemax_leaky_layernorm.json",
        "configs/gpt2_layernorm_removal.json",
        "configs/gpt2_program_healing.json",
        "configs/gpu_run_manifest.json",
    ):
        try:
            with open(path) as handle:
                json.load(handle)
        except Exception as error:
            invalid_json.append({"path": path, "error": str(error)})
    checks["config_json"] = {"invalid": invalid_json, "passed": not invalid_json}
    dataset_ready = processed_dataset_ready(args.processed_dataset_dir)
    checks["processed_dataset"] = {
        "path": os.path.abspath(args.processed_dataset_dir),
        "passed": dataset_ready,
    }

    storage_probe = os.path.abspath(args.processed_dataset_dir)
    while not os.path.exists(storage_probe):
        parent = os.path.dirname(storage_probe)
        if parent == storage_probe:
            break
        storage_probe = parent
    free_gb = shutil.disk_usage(storage_probe).free / (1024**3)
    checks["storage"] = {
        "probe": storage_probe,
        "free_gb": free_gb,
        "minimum_free_gb": args.minimum_free_gb,
        "passed": free_gb >= args.minimum_free_gb,
    }
    bandnorm_ready = checkpoint_ready(args.bandnorm_model)
    checks["bandnorm_fallback"] = {
        "path": os.path.abspath(args.bandnorm_model),
        "ready": bandnorm_ready,
        "required": not args.allow_missing_bandnorm,
        "passed": bandnorm_ready or args.allow_missing_bandnorm,
    }
    checks["bf16"] = {
        "supported_all_devices": bool(gpu_count) and all(
            torch.cuda.get_device_capability(index)[0] >= 8
            for index in range(gpu_count)
        ),
    }
    checks["bf16"]["passed"] = checks["bf16"]["supported_all_devices"]
    output = {
        "status": "PASSED" if all(value["passed"] for value in checks.values()) else "FAILED",
        "processed_dataset_dir": os.path.abspath(args.processed_dataset_dir),
        "checks": checks,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))
    if output["status"] != "PASSED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
