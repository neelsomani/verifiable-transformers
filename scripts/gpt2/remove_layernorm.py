#!/usr/bin/env python3
"""Distributed GPT-2 LayerNorm attenuation, fine-tuning, and exact folding."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.gpt2.extract import load_model_with_variants
from scripts.gpt2.train import (
    create_optimizer_with_weight_decay_exclusions,
    validate_checkpoint_compatibility,
)
from scripts.norm_removal import (
    fold_attenuated_layernorms,
    install_attenuated_layernorms,
    update_attenuation_schedule,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processed_dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--config", default="configs/gpt2_layernorm_removal.json"
    )
    parser.add_argument(
        "--baseline_eval_loss",
        type=float,
        default=None,
        help="Source-model eval loss used only to report removal degradation.",
    )
    parser.add_argument(
        "--bandnorm_eval_loss",
        type=float,
        default=None,
        help=(
            "Optional invocation-time copy of the preregistered absolute BandNorm "
            "gate. If supplied, it must match bandnorm_eval_loss_gate in the config."
        ),
    )
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument(
        "--postprocess_checkpoint",
        default=None,
        help=(
            "Recover a completed attenuation run from checkpoint-N and perform "
            "only the fold checks, final evaluation, and artifact save."
        ),
    )
    return parser.parse_args()


def load_json(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def validate_bandnorm_eval_loss_gate(cfg: dict, supplied: float | None) -> float:
    gate = float(cfg["bandnorm_eval_loss_gate"])
    if supplied is not None and not math.isclose(
        float(supplied), gate, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            f"--bandnorm_eval_loss={supplied} contradicts the preregistered "
            f"bandnorm_eval_loss_gate={gate}"
        )
    return gate


def load_trained_removal_checkpoint(
    model, checkpoint: str, *, required_step: int
) -> int:
    state_path = os.path.join(checkpoint, "trainer_state.json")
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"Missing trainer state: {state_path}")
    state = load_json(state_path)
    global_step = int(state["global_step"])
    if global_step < required_step:
        raise ValueError(
            f"Postprocess checkpoint stopped at step {global_step}; "
            f"required at least {required_step}"
        )

    weights_path = os.path.join(checkpoint, "model.safetensors")
    if os.path.isfile(weights_path):
        from safetensors.torch import load_file

        state_dict = load_file(weights_path)
    else:
        weights_path = os.path.join(checkpoint, "pytorch_model.bin")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Missing model weights in {checkpoint}")
        state_dict = torch.load(weights_path, map_location="cpu")

    incompatible = model.load_state_dict(state_dict, strict=False)
    ignored_missing = validate_checkpoint_compatibility(model, incompatible)
    if ignored_missing:
        model.tie_weights()
    return global_step


def validate_attenuation_endpoint(entries) -> dict[str, dict[str, float | bool]]:
    endpoint = {}
    errors = []
    for entry in entries:
        attenuation = float(entry.module.attenuation.item())
        fixed_std = float(entry.module.fixed_std.item())
        calibrating = bool(entry.module.calibrating.item())
        endpoint[entry.name] = {
            "attenuation": attenuation,
            "fixed_std": fixed_std,
            "calibrating": calibrating,
        }
        if attenuation != 1.0:
            errors.append(f"{entry.name}: attenuation={attenuation}")
        if calibrating:
            errors.append(f"{entry.name}: calibration still enabled")
        if not math.isfinite(fixed_std) or fixed_std <= 0.0:
            errors.append(f"{entry.name}: invalid fixed_std={fixed_std}")
    if errors:
        raise RuntimeError(
            "LayerNorm attenuation checkpoint is not at the fixed-std endpoint: "
            + "; ".join(errors)
        )
    return endpoint


@torch.no_grad()
def fold_and_measure_in_fp32(model, sample: torch.Tensor) -> dict[str, float]:
    """Check the algebraic fold outside Accelerate's lossy BF16 wrapper."""
    parameter_dtypes = {parameter.dtype for parameter in model.parameters()}
    if parameter_dtypes != {torch.float32}:
        raise RuntimeError(
            "FP32 fold validation requires every model parameter to be float32; "
            f"found {sorted(map(str, parameter_dtypes))}"
        )
    prepared_forward = model.forward
    original_forward = getattr(model, "_original_forward", None)
    if original_forward is not None:
        model.forward = original_forward
    try:
        before_fold = model(input_ids=sample).logits.float()
        fold_attenuated_layernorms(model, compute_dtype=torch.float64)
        after_fold = model(input_ids=sample).logits.float()
        difference = after_fold - before_fold
        relative_l2_error = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(
            before_fold
        ).clamp_min(torch.finfo(torch.float32).tiny)
        top1_agreement = (
            after_fold.argmax(dim=-1) == before_fold.argmax(dim=-1)
        ).float().mean()
        return {
            "max_abs_diff": float(difference.abs().max().item()),
            "relative_l2_error": float(relative_l2_error.item()),
            "top1_agreement": float(top1_agreement.item()),
        }
    finally:
        if original_forward is not None:
            model.forward = prepared_forward


class NormRemovalScheduleCallback(TrainerCallback):
    def __init__(self, entries, cfg: dict, output_dir: str):
        self.entries = entries
        self.cfg = cfg
        self.output_dir = output_dir
        self.history = []

    def _update(self, step: int) -> dict[str, float]:
        return update_attenuation_schedule(
            self.entries,
            step,
            calibration_steps=self.cfg["calibration_steps"],
            transition_steps=self.cfg["transition_steps"],
            gap_steps=self.cfg["gap_steps"],
        )

    def on_train_begin(self, args, state, control, **kwargs):
        self._update(state.global_step)
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        self._update(state.global_step)
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        schedule = self._update(state.global_step)
        record = {
            "step": state.global_step,
            "schedule": schedule,
            "fixed_std": {
                entry.name: float(entry.module.fixed_std.item())
                for entry in self.entries
            },
            "logs": dict(logs or {}),
        }
        self.history.append(record)
        if state.is_world_process_zero:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(
                os.path.join(self.output_dir, "removal_schedule.json"), "w"
            ) as handle:
                json.dump(self.history, handle, indent=2)
        return control


def schedule_end(entries, cfg: dict) -> int:
    return (
        cfg["calibration_steps"]
        + (len(entries) - 1) * cfg["gap_steps"]
        + cfg["transition_steps"]
    )


def write_status(output_dir: str, stage: str, **extra) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "stage": stage,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    with open(os.path.join(output_dir, "run_status.json"), "w") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)
    bandnorm_eval_loss_gate = validate_bandnorm_eval_loss_gate(
        cfg, args.bandnorm_eval_loss
    )
    set_seed(cfg["seed"])
    os.makedirs(args.output_dir, exist_ok=True)
    write_status(args.output_dir, "initializing", source_checkpoint=args.checkpoint)

    model = load_model_with_variants(args.checkpoint, "cpu").float()
    source_info_path = os.path.join(args.checkpoint, "model_info.json")
    if not os.path.exists(source_info_path):
        source_info_path = os.path.join(os.path.dirname(args.checkpoint), "model_info.json")
    source_info = load_json(source_info_path)
    expected = {
        "norm_variant": "layernorm",
        "attn_variant": "sparsemax",
        "activation_variant": "leaky_relu",
    }
    for key, value in expected.items():
        if source_info.get(key) != value:
            raise ValueError(
                f"A4 removal requires {key}={value!r}, got {source_info.get(key)!r}"
            )

    entries = install_attenuated_layernorms(
        model, momentum=cfg["ema_momentum"]
    )
    end_step = schedule_end(entries, cfg)
    max_steps = args.max_steps or cfg["max_steps"]
    if max_steps < end_step:
        raise ValueError(
            f"max_steps={max_steps} ends before full attenuation at step {end_step}"
        )
    recovered_step = None
    if args.postprocess_checkpoint is not None:
        recovered_step = load_trained_removal_checkpoint(
            model, args.postprocess_checkpoint, required_step=max_steps
        )
        validate_attenuation_endpoint(entries)
    if cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    datasets = load_from_disk(args.processed_dataset_dir)
    train_dataset = datasets["train"]
    eval_dataset = datasets["validation"]
    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(
            range(min(args.max_train_samples, len(train_dataset)))
        )
    if args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(
            range(min(args.max_eval_samples, len(eval_dataset)))
        )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=cfg["train_batch_size_per_device"],
        per_device_eval_batch_size=cfg["eval_batch_size_per_device"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        max_grad_norm=cfg["max_grad_norm"],
        adam_beta1=cfg["adam_beta1"],
        adam_beta2=cfg["adam_beta2"],
        adam_epsilon=cfg["adam_epsilon"],
        warmup_steps=cfg["warmup_steps"],
        max_steps=max_steps,
        lr_scheduler_type=cfg["lr_scheduler_type"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_steps=cfg["save_steps"],
        logging_steps=cfg["logging_steps"],
        save_total_limit=cfg["save_total_limit"],
        dataloader_num_workers=cfg["dataloader_num_workers"],
        bf16=cfg["bf16"],
        fp16=cfg["fp16"],
        report_to="none",
    )
    callback = NormRemovalScheduleCallback(entries, cfg, args.output_dir)
    if args.postprocess_checkpoint is not None:
        history_path = os.path.join(args.output_dir, "removal_schedule.json")
        if os.path.isfile(history_path):
            callback.history = load_json(history_path)
    optimizer = create_optimizer_with_weight_decay_exclusions(
        model,
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        betas=(cfg["adam_beta1"], cfg["adam_beta2"]),
        eps=cfg["adam_epsilon"],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=default_data_collator,
        callbacks=[callback],
        optimizers=(optimizer, None),
    )

    if args.postprocess_checkpoint is None:
        write_status(args.output_dir, "training", schedule_end=end_step, max_steps=max_steps)
        trainer.train()
    else:
        write_status(
            args.output_dir,
            "postprocessing_recovered_checkpoint",
            schedule_end=end_step,
            max_steps=max_steps,
            recovered_checkpoint=args.postprocess_checkpoint,
            recovered_step=recovered_step,
        )
    final_schedule = callback._update(max_steps)
    if not all(value == 1.0 for value in final_schedule.values()):
        raise RuntimeError("Training ended before every LayerNorm reached attenuation=1")
    attenuation_endpoint = validate_attenuation_endpoint(entries)

    pre_fold_metrics = trainer.evaluate()
    sample = next(iter(trainer.get_eval_dataloader()))["input_ids"][
        : cfg["fold_validation_batch_size"]
    ]
    sample = sample.to(trainer.model.device)
    trainer.model.eval()
    local_fold_metrics = fold_and_measure_in_fp32(trainer.model, sample)
    local_fold_tensor = torch.tensor(
        [
            local_fold_metrics["max_abs_diff"],
            local_fold_metrics["relative_l2_error"],
            local_fold_metrics["top1_agreement"],
        ],
        device=trainer.model.device,
        dtype=torch.float64,
    )
    gathered_fold_metrics = trainer.accelerator.gather(local_fold_tensor)
    gathered_fold_metrics = gathered_fold_metrics.reshape(-1, 3)
    fold_max_abs_diff = float(gathered_fold_metrics[:, 0].max().item())
    fold_relative_l2_error = float(gathered_fold_metrics[:, 1].max().item())
    fold_top1_agreement = float(gathered_fold_metrics[:, 2].min().item())
    post_fold_metrics = trainer.evaluate()
    fold_eval_loss_delta = float(
        post_fold_metrics["eval_loss"] - pre_fold_metrics["eval_loss"]
    )
    fold_validation = {
        "arithmetic_dtype": "float64",
        "forward_dtype": "float32",
        "validation_batch_size_per_rank": cfg["fold_validation_batch_size"],
        "max_abs_diff": fold_max_abs_diff,
        "max_abs_tolerance": cfg["fold_max_abs_tolerance_fp32"],
        "relative_l2_error": fold_relative_l2_error,
        "relative_l2_tolerance": cfg["fold_relative_l2_tolerance_fp32"],
        "top1_agreement": fold_top1_agreement,
        "top1_agreement_min": cfg["fold_top1_agreement_min"],
        "pre_fold_eval_loss": float(pre_fold_metrics["eval_loss"]),
        "post_fold_eval_loss": float(post_fold_metrics["eval_loss"]),
        "eval_loss_delta": fold_eval_loss_delta,
        "eval_loss_delta_tolerance": cfg["fold_eval_loss_delta_tolerance"],
    }
    fold_violations = []
    if fold_max_abs_diff > cfg["fold_max_abs_tolerance_fp32"]:
        fold_violations.append("fp32 max absolute logit difference exceeded tolerance")
    if fold_relative_l2_error > cfg["fold_relative_l2_tolerance_fp32"]:
        fold_violations.append("fp32 relative L2 logit error exceeded tolerance")
    if fold_top1_agreement < cfg["fold_top1_agreement_min"]:
        fold_violations.append("fp32 top-1 agreement fell below threshold")
    if abs(fold_eval_loss_delta) > cfg["fold_eval_loss_delta_tolerance"]:
        fold_violations.append("BF16 eval-loss delta exceeded tolerance")
    fold_validation["status"] = "passed" if not fold_violations else "failed"
    fold_validation["violations"] = fold_violations
    if trainer.is_world_process_zero():
        with open(os.path.join(args.output_dir, "fold_validation.json"), "w") as handle:
            json.dump(fold_validation, handle, indent=2)
    if fold_violations:
        write_status(
            args.output_dir,
            "fold_validation_failed",
            fold_validation=fold_validation,
        )
        raise RuntimeError("Affine fold validation failed: " + "; ".join(fold_violations))
    trainer.save_model(args.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)

    eval_loss = float(post_fold_metrics["eval_loss"])
    baseline_eval_loss = args.baseline_eval_loss
    if baseline_eval_loss is None:
        source_eval_path = os.path.join(args.checkpoint, "eval_results.json")
        if os.path.exists(source_eval_path):
            baseline_eval_loss = float(load_json(source_eval_path)["eval_loss"])
    result = {
        "status": "passed",
        "method": "sequential_fixed_std_attenuation_and_affine_folding",
        "source_checkpoint": args.checkpoint,
        "schedule_end": end_step,
        "max_steps": max_steps,
        "recovered_checkpoint": args.postprocess_checkpoint,
        "recovered_step": recovered_step,
        "attenuation_endpoint": attenuation_endpoint,
        "fold_validation": fold_validation,
        "fold_max_abs_diff": fold_max_abs_diff,
        "fold_relative_l2_error": fold_relative_l2_error,
        "fold_top1_agreement": fold_top1_agreement,
        "fold_eval_loss_delta": fold_eval_loss_delta,
        "pre_fold_eval_loss": float(pre_fold_metrics["eval_loss"]),
        "post_fold_eval_loss": eval_loss,
        "post_fold_perplexity": math.exp(eval_loss),
        "baseline_eval_loss": baseline_eval_loss,
        "bandnorm_eval_loss": args.bandnorm_eval_loss,
        "bandnorm_eval_loss_gate": bandnorm_eval_loss_gate,
        "decision_rule": cfg["decision_rule"],
    }
    if baseline_eval_loss is not None:
        result["removal_loss_delta"] = eval_loss - baseline_eval_loss
    result["decision"] = (
        "norm_free" if eval_loss < bandnorm_eval_loss_gate else "bandnorm"
    )
    if args.bandnorm_eval_loss is not None:
        result["beats_measured_bandnorm"] = eval_loss < args.bandnorm_eval_loss

    if trainer.is_world_process_zero():
        with open(os.path.join(args.output_dir, "removal_metrics.json"), "w") as handle:
            json.dump(result, handle, indent=2)
        with open(os.path.join(args.output_dir, "model_info.json"), "w") as handle:
            json.dump(
                {
                    "model_name": source_info["model_name"],
                    "num_parameters": sum(p.numel() for p in trainer.model.parameters()),
                    "norm_variant": "none",
                    "attn_variant": "sparsemax",
                    "activation_variant": "leaky_relu",
                    "source_checkpoint": args.checkpoint,
                },
                handle,
                indent=2,
            )
        write_status(args.output_dir, "completed", **result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
