#!/usr/bin/env python3
"""Distributed OWT healing for synthesized GPT-2 program heads."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys

import torch
from datasets import load_from_disk
from transformers import (
    GPT2Tokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.gpt2.extract import (
    BEHAVIOR_GENERATORS,
    build_circuit_graph,
    controlled_forward,
    get_candidate_token_ids,
    load_model_with_variants,
    select_last_real_logits,
)
from scripts.gpt2.synthesize_programs import projected_decisions
from scripts.gpt2.train import (
    create_optimizer_with_weight_decay_exclusions,
    find_latest_checkpoint,
)
from scripts.programs import install_program_heads, load_programs, save_programs


TASKS = ("quote_close", "bracket_type")
HEAD_PATTERN = re.compile(r"^attn_(\d+)_h_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--programs", required=True)
    parser.add_argument("--processed_dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--reference_eval_perplexity", type=float, required=True)
    parser.add_argument("--config", default="configs/gpt2_program_healing.json")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Explicit Trainer checkpoint; otherwise the latest output checkpoint is used.",
    )
    parser.add_argument(
        "--disable_auto_resume",
        action="store_true",
        help="Start from the base model even if output_dir contains checkpoints.",
    )
    parser.add_argument(
        "--ablation_aware",
        action="store_true",
        help=(
            "Add short-domain circuit, stochastic non-circuit ablation, and "
            "neural-bypass suppression losses for the C5 migration fallback."
        ),
    )
    parser.add_argument(
        "--circuit_root",
        default=None,
        help="Required with --ablation_aware; selected pre-healing task circuits.",
    )
    return parser.parse_args()


def load_json(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def behavior_domain(tokenizer, cfg, device):
    result = {}
    for task in TASKS:
        examples = BEHAVIOR_GENERATORS[task](cfg["behavior_examples"])
        encoded = tokenizer(
            [example.prompt for example in examples],
            return_tensors="pt",
            padding=True,
        )
        result[task] = {
            "input_ids": encoded["input_ids"].to(device),
            "attention_mask": encoded["attention_mask"].to(device),
            "candidates": get_candidate_token_ids(task, tokenizer),
        }
    return result


def decisions_for_model(model, domains, batch_size):
    device = next(model.parameters()).device
    return {
        task: projected_decisions(
            model,
            values["input_ids"].to(device),
            values["attention_mask"].to(device),
            values["candidates"],
            batch_size,
        )
        for task, values in domains.items()
    }


class HealingGateCallback(TrainerCallback):
    """Stop only when both preregistered C4 acceptance gates are satisfied."""

    def __init__(
        self,
        domains,
        reference,
        batch_size,
        required_agreement,
        perplexity_budget,
        output_dir,
    ):
        self.domains = domains
        self.reference = reference
        self.batch_size = batch_size
        self.required_agreement = required_agreement
        self.perplexity_budget = perplexity_budget
        self.output_dir = output_dir
        self.history = []

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        decisions = decisions_for_model(model, self.domains, self.batch_size)
        agreements = {
            task: float(
                (decisions[task] == self.reference[task]).float().mean().item()
            )
            for task in TASKS
        }
        eval_loss = float((metrics or {})["eval_loss"])
        perplexity = math.exp(eval_loss)
        agreement_pass = all(
            value >= self.required_agreement for value in agreements.values()
        )
        perplexity_pass = perplexity <= self.perplexity_budget
        record = {
            "step": int(state.global_step),
            "projected_agreement": agreements,
            "required_projected_agreement": self.required_agreement,
            "eval_loss": eval_loss,
            "eval_perplexity": perplexity,
            "perplexity_budget": self.perplexity_budget,
            "agreement_pass": agreement_pass,
            "perplexity_pass": perplexity_pass,
            "acceptance_pass": agreement_pass and perplexity_pass,
        }
        self.history.append(record)
        if state.is_world_process_zero:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(
                os.path.join(self.output_dir, "healing_gate_history.json"), "w"
            ) as handle:
                json.dump(self.history, handle, indent=2)
        if record["acceptance_pass"]:
            control.should_training_stop = True
        return control


def load_circuit_edges(circuit_root: str) -> dict[str, set[tuple[str, str]]]:
    result = {}
    for task in TASKS:
        with open(os.path.join(circuit_root, task, "circuit.json")) as handle:
            circuit = json.load(handle)
        result[task] = {
            (edge["source"], edge["target"])
            if isinstance(edge, dict)
            else tuple(edge)
            for edge in circuit["edges"]
        }
    return result


class AblationAwareProgramTrainer(Trainer):
    """OWT Trainer with exact residual-edge auxiliary losses on short domains."""

    def __init__(
        self,
        *args,
        behavior_domains,
        reference_decisions,
        circuit_edges,
        program_nodes,
        healing_config,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.behavior_domains = behavior_domains
        self.reference_decisions = reference_decisions
        self.circuit_edges = circuit_edges
        self.program_nodes = set(program_nodes)
        self.healing_config = healing_config
        base = self.accelerator.unwrap_model(self.model)
        self.circuit_graph = build_circuit_graph(
            base.config.n_layer, base.config.n_head
        )
        self.full_circuit_edges = self.circuit_graph.get_edges()
        self.intended_program_nodes = {}
        for task, edges in circuit_edges.items():
            intended = sorted(
                node
                for node in self.program_nodes
                if any(node in edge for edge in edges)
            )
            if not intended:
                raise RuntimeError(
                    f"Ablation-aware healing needs a program head in {task}'s circuit"
                )
            self.intended_program_nodes[task] = intended
        self.last_ablation_aware_metrics = None

    @staticmethod
    def _candidate_loss(logits, attention_mask, candidates, targets):
        rows = select_last_real_logits(logits, attention_mask)[:, candidates]
        return torch.nn.functional.cross_entropy(rows, targets.to(rows.device))

    @staticmethod
    def _uniform_penalty(logits, attention_mask, candidates):
        rows = select_last_real_logits(logits, attention_mask)[:, candidates]
        uniform = torch.full_like(rows, 1.0 / len(candidates))
        return torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(rows, dim=-1),
            uniform,
            reduction="batchmean",
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss
        # C4's perplexity gate must remain the ordinary held-out language-model
        # loss.  Auxiliary circuit losses are training-only and must not leak
        # into Trainer.evaluate()'s reported eval_loss.
        if not model.training:
            return (loss, outputs) if return_outputs else loss
        every = int(self.healing_config["ablation_aware_aux_every_steps"])
        step = int(self.state.global_step)
        if every < 1:
            raise ValueError("ablation_aware_aux_every_steps must be positive")
        if step % every != 0:
            return (loss, outputs) if return_outputs else loss

        base_model = self.accelerator.unwrap_model(model)
        circuit_loss = loss.new_zeros(())
        sampled_loss = loss.new_zeros(())
        bypass_penalty = loss.new_zeros(())
        keep_probability = float(
            self.healing_config[
                "ablation_aware_non_circuit_keep_probability"
            ]
        )
        if not 0.0 <= keep_probability <= 1.0:
            raise ValueError("non-circuit keep probability must be in [0, 1]")
        rng = random.Random(int(self.healing_config["seed"]) + step)
        aux_batch_size = int(
            self.healing_config.get(
                "ablation_aware_behavior_batch_size",
                self.healing_config["behavior_batch_size"],
            )
        )
        if aux_batch_size < 1:
            raise ValueError("ablation_aware_behavior_batch_size must be positive")
        aux_examples = {}

        for task in TASKS:
            domain = self.behavior_domains[task]
            runtime_device = next(base_model.parameters()).device
            domain_size = domain["input_ids"].size(0)
            sample_size = min(aux_batch_size, domain_size)
            aux_examples[task] = sample_size
            round_index = step // every
            start = (round_index * sample_size) % domain_size
            indices = [
                (start + offset) % domain_size for offset in range(sample_size)
            ]
            input_ids = domain["input_ids"][indices].to(runtime_device)
            attention_mask = domain["attention_mask"][indices].to(runtime_device)
            candidates = domain["candidates"]
            targets = self.reference_decisions[task][indices]
            core = self.circuit_edges[task]
            core_logits = controlled_forward(
                base_model,
                input_ids,
                attention_mask,
                core,
                self.circuit_graph,
            )
            circuit_loss = circuit_loss + self._candidate_loss(
                core_logits, attention_mask, candidates, targets
            )

            sampled = set(core)
            sampled.update(
                edge
                for edge in self.full_circuit_edges - core
                if rng.random() < keep_probability
            )
            sampled_logits = controlled_forward(
                base_model,
                input_ids,
                attention_mask,
                sampled,
                self.circuit_graph,
            )
            sampled_loss = sampled_loss + self._candidate_loss(
                sampled_logits, attention_mask, candidates, targets
            )

            bypass = {
                edge
                for edge in self.full_circuit_edges
                if edge[0] not in self.intended_program_nodes[task]
            }
            bypass_logits = controlled_forward(
                base_model,
                input_ids,
                attention_mask,
                bypass,
                self.circuit_graph,
            )
            bypass_penalty = bypass_penalty + self._uniform_penalty(
                bypass_logits, attention_mask, candidates
            )

        total = (
            loss
            + float(self.healing_config["ablation_aware_circuit_loss_weight"])
            * circuit_loss
            + float(self.healing_config["ablation_aware_sampled_loss_weight"])
            * sampled_loss
            + float(self.healing_config["ablation_aware_bypass_penalty_weight"])
            * bypass_penalty
        )
        self.last_ablation_aware_metrics = {
            "step": step,
            "behavior_examples_per_task": aux_examples,
            "owt_loss": float(loss.detach().item()),
            "circuit_loss": float(circuit_loss.detach().item()),
            "sampled_loss": float(sampled_loss.detach().item()),
            "bypass_penalty": float(bypass_penalty.detach().item()),
            "total_loss": float(total.detach().item()),
        }
        return (total, outputs) if return_outputs else total


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)
    if args.ablation_aware and not args.circuit_root:
        raise ValueError("--circuit_root is required with --ablation_aware")
    set_seed(cfg["seed"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    model = load_model_with_variants(args.model_path, str(device))
    model.config.use_cache = False
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    domains = behavior_domain(tokenizer, cfg, device)
    reference = decisions_for_model(
        model, domains, cfg["behavior_batch_size"]
    )

    programs = load_programs(args.programs)
    if not programs:
        raise RuntimeError(
            "No fittable program heads were synthesized; preserve the synthesis "
            "counterexample report and stop before C4 healing."
        )
    variant = load_json(
        os.path.join(args.model_path, "model_info.json")
    ).get("attn_variant", "sparsemax")
    install_program_heads(model, programs, attention_variant=variant)
    initial = decisions_for_model(model, domains, cfg["behavior_batch_size"])
    initial_agreement = {
        task: float((initial[task] == reference[task]).float().mean().item())
        for task in TASKS
    }

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
    if cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

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
        max_steps=args.max_steps or cfg["max_steps"],
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
    optimizer = create_optimizer_with_weight_decay_exclusions(
        model,
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        betas=(cfg["adam_beta1"], cfg["adam_beta2"]),
        eps=cfg["adam_epsilon"],
    )
    perplexity_budget = args.reference_eval_perplexity * (
        1 + cfg["relative_perplexity_increase_budget"]
    )
    gate_callback = HealingGateCallback(
        domains,
        reference,
        cfg["behavior_batch_size"],
        cfg["projected_agreement_required"],
        perplexity_budget,
        args.output_dir,
    )
    trainer_class = AblationAwareProgramTrainer if args.ablation_aware else Trainer
    trainer_kwargs = {}
    if args.ablation_aware:
        trainer_kwargs.update(
            {
                "behavior_domains": domains,
                "reference_decisions": reference,
                "circuit_edges": load_circuit_edges(args.circuit_root),
                "program_nodes": {
                    f"attn_{layer}_h_{head}" for layer, head in programs
                },
                "healing_config": cfg,
            }
        )
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=default_data_collator,
        callbacks=[gate_callback],
        optimizers=(optimizer, None),
        **trainer_kwargs,
    )
    resume_checkpoint = args.resume_from_checkpoint
    if resume_checkpoint is None and not args.disable_auto_resume:
        resume_checkpoint = find_latest_checkpoint(args.output_dir)
        if resume_checkpoint is not None and local_rank == 0:
            print(f"Auto-resuming healing from checkpoint: {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    eval_metrics = trainer.evaluate()
    final = decisions_for_model(
        trainer.model, domains, cfg["behavior_batch_size"]
    )
    final_agreement = {
        task: float((final[task] == reference[task]).float().mean().item())
        for task in TASKS
    }
    eval_perplexity = math.exp(float(eval_metrics["eval_loss"]))
    budget = perplexity_budget
    agreement_pass = all(
        value >= cfg["projected_agreement_required"]
        for value in final_agreement.values()
    )
    perplexity_pass = eval_perplexity <= budget
    result = {
        "source_model": args.model_path,
        "resume_from_checkpoint": resume_checkpoint,
        "programs": args.programs,
        "program_heads": [f"{layer}.{head}" for layer, head in sorted(programs)],
        "programs_frozen": True,
        "method": (
            "owt_plus_exact_short_domain_stochastic_edge_ablation_and_bypass_suppression"
            if args.ablation_aware
            else "owt_healing"
        ),
        "ablation_aware": args.ablation_aware,
        "circuit_root": args.circuit_root,
        "ablation_aware_last_metrics": getattr(
            trainer, "last_ablation_aware_metrics", None
        ),
        "initial_projected_agreement": initial_agreement,
        "final_projected_agreement": final_agreement,
        "projected_agreement_required": cfg["projected_agreement_required"],
        "reference_eval_perplexity": args.reference_eval_perplexity,
        "final_eval_loss": float(eval_metrics["eval_loss"]),
        "final_eval_perplexity": eval_perplexity,
        "perplexity_budget": budget,
        "relative_perplexity_increase_budget": cfg[
            "relative_perplexity_increase_budget"
        ],
        "perplexity_budget_provenance": cfg["perplexity_budget_provenance"],
        "agreement_pass": agreement_pass,
        "perplexity_pass": perplexity_pass,
        "stopping_criteria": (
            "projected agreement on both registered behavior domains must equal "
            "the configured threshold and eval perplexity must stay within the "
            "pre-registered budget"
        ),
        "gate_history": gate_callback.history,
        "success": agreement_pass and perplexity_pass,
        "global_step": int(trainer.state.global_step),
        "epoch": None if trainer.state.epoch is None else float(trainer.state.epoch),
        "effective_global_batch_size": (
            int(trainer.args.world_size)
            * int(cfg["train_batch_size_per_device"])
            * int(cfg["gradient_accumulation_steps"])
        ),
    }
    trainer.save_model(args.output_dir)
    trainer.save_state()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)
        save_programs(programs, os.path.join(args.output_dir, "programs.json"))
        with open(os.path.join(args.output_dir, "run_config.json"), "w") as handle:
            json.dump(cfg, handle, indent=2)
        source_info = load_json(os.path.join(args.model_path, "model_info.json"))
        source_info.update(
            {
                "program_heads": result["program_heads"],
                "source_model": args.model_path,
            }
        )
        with open(os.path.join(args.output_dir, "model_info.json"), "w") as handle:
            json.dump(source_info, handle, indent=2)
        with open(os.path.join(args.output_dir, "healing_results.json"), "w") as handle:
            json.dump(result, handle, indent=2)
        print(json.dumps(result, indent=2))
    if not result["success"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
