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
    build_circuit_graph,
    controlled_forward,
    get_candidate_token_ids,
    load_behavior_examples,
    load_model_with_variants,
    select_last_real_logits,
)
from scripts.gpt2.behavior_domains import reference_program_targets
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
            "rotating joint/individual full-and-core bypass suppression."
        ),
    )
    parser.add_argument(
        "--circuit_root",
        default=None,
        help="Required with --ablation_aware; selected pre-healing task circuits.",
    )
    parser.add_argument(
        "--behavior_train_manifest",
        default=None,
        help="Synthesis/train behavior manifest; legacy repeated rows if omitted.",
    )
    parser.add_argument(
        "--behavior_gate_manifest",
        default=None,
        help="Untouched behavior gate manifest; defaults to the train manifest.",
    )
    parser.add_argument(
        "--bounded_behavior_manifest",
        default=None,
        help=(
            "Declared finite behavior domain used for both auxiliary training "
            "and exhaustive acceptance; makes no held-out claim."
        ),
    )
    parser.add_argument(
        "--tasks", nargs="+", choices=TASKS, default=list(TASKS)
    )
    parser.add_argument(
        "--minimum_initial_agreement",
        type=float,
        default=1.0,
        help=(
            "Registered floor for the programs-installed pre-heal full and "
            "circuit-only forwards. Final acceptance remains exact."
        ),
    )
    return parser.parse_args()


def load_json(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def behavior_domain(tokenizer, cfg, device, manifest_path=None, tasks=TASKS):
    result = {}
    for task in tasks:
        examples, provenance = load_behavior_examples(
            task, cfg["behavior_examples"], manifest_path
        )
        encoded = tokenizer(
            [example.prompt for example in examples],
            return_tensors="pt",
            padding=True,
        )
        candidates = get_candidate_token_ids(task, tokenizer)
        result[task] = {
            "examples": examples,
            "input_ids": encoded["input_ids"].to(device),
            "attention_mask": encoded["attention_mask"].to(device),
            "candidates": candidates,
            "targets": reference_program_targets(
                examples, tokenizer, candidates
            ),
            "provenance": provenance,
        }
    return result


def reference_program_decisions(domains):
    return {task: values["targets"] for task, values in domains.items()}


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


def controlled_decisions_for_edges(
    model, domains, task, edges, graph, batch_size
):
    values = domains[task]
    device = next(model.parameters()).device
    decisions = []
    with torch.no_grad():
        for start in range(0, values["input_ids"].size(0), batch_size):
            input_ids = values["input_ids"][start : start + batch_size].to(device)
            attention_mask = values["attention_mask"][
                start : start + batch_size
            ].to(device)
            logits = controlled_forward(
                model,
                input_ids,
                attention_mask,
                edges,
                graph,
            )
            rows = select_last_real_logits(logits, attention_mask)
            decisions.append(
                rows[:, values["candidates"]].argmax(dim=-1).cpu()
            )
    return torch.cat(decisions)


def full_lesion_sweep(
    model,
    domains,
    reference,
    circuit_edges,
    program_nodes,
    batch_size,
):
    """Run the unsampled full/core C5 lesion matrix on the acceptance domain."""

    base = model
    graph = build_circuit_graph(base.config.n_layer, base.config.n_head)
    full_edges = graph.get_edges()
    full_decisions = decisions_for_model(base, domains, batch_size)
    reports = {}
    for task in domains:
        core = circuit_edges[task]
        intended = sorted(
            node for node in program_nodes if any(node in edge for edge in core)
        )
        if not intended:
            raise RuntimeError(f"{task} has no installed program in its circuit")
        circuit_decisions = controlled_decisions_for_edges(
            base, domains, task, core, graph, batch_size
        )
        individual = {}
        for node in intended:
            full_without = {edge for edge in full_edges if edge[0] != node}
            core_without = {edge for edge in core if edge[0] != node}
            full_lesion = controlled_decisions_for_edges(
                base, domains, task, full_without, graph, batch_size
            )
            core_lesion = controlled_decisions_for_edges(
                base, domains, task, core_without, graph, batch_size
            )
            full_agreement = float(
                (full_lesion == full_decisions[task]).float().mean().item()
            )
            core_agreement = float(
                (core_lesion == circuit_decisions).float().mean().item()
            )
            individual[node] = {
                "full_agreement_after_lesion": full_agreement,
                "core_agreement_after_lesion": core_agreement,
                "necessary_in_full": full_agreement < 1.0,
                "necessary_in_core": core_agreement < 1.0,
            }
        intended_set = set(intended)
        joint_full = {
            edge for edge in full_edges if edge[0] not in intended_set
        }
        joint_core = {edge for edge in core if edge[0] not in intended_set}
        joint_full_decisions = controlled_decisions_for_edges(
            base, domains, task, joint_full, graph, batch_size
        )
        joint_core_decisions = controlled_decisions_for_edges(
            base, domains, task, joint_core, graph, batch_size
        )
        joint_full_agreement = float(
            (joint_full_decisions == full_decisions[task]).float().mean().item()
        )
        joint_core_agreement = float(
            (joint_core_decisions == circuit_decisions).float().mean().item()
        )
        circuit_accuracy = float(
            (circuit_decisions == reference[task]).float().mean().item()
        )
        task_pass = (
            circuit_accuracy == 1.0
            and joint_full_agreement < 1.0
            and joint_core_agreement < 1.0
            and all(
                value["necessary_in_full"] and value["necessary_in_core"]
                for value in individual.values()
            )
        )
        reports[task] = {
            "intended_program_heads": intended,
            "circuit_accuracy_against_P": circuit_accuracy,
            "joint_full_agreement_after_lesion": joint_full_agreement,
            "joint_core_agreement_after_lesion": joint_core_agreement,
            "individual_ablations": individual,
            "pass": task_pass,
        }
    return {
        "tasks": reports,
        "pass": all(report["pass"] for report in reports.values()),
    }


class HealingGateCallback(TrainerCallback):
    """Stop only when every configured acceptance condition is satisfied."""

    def __init__(
        self,
        domains,
        reference,
        batch_size,
        required_agreement,
        perplexity_budget,
        output_dir,
        lesion_context=None,
    ):
        self.domains = domains
        self.reference = reference
        self.batch_size = batch_size
        self.required_agreement = required_agreement
        self.perplexity_budget = perplexity_budget
        self.output_dir = output_dir
        self.lesion_context = lesion_context
        self.tasks = tuple(domains)
        self.ablation_trainer = None
        self.history = []

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        decisions = decisions_for_model(model, self.domains, self.batch_size)
        agreements = {
            task: float(
                (decisions[task] == self.reference[task]).float().mean().item()
            )
            for task in self.tasks
        }
        eval_loss = float((metrics or {})["eval_loss"])
        perplexity = math.exp(eval_loss)
        agreement_pass = all(
            value >= self.required_agreement for value in agreements.values()
        )
        perplexity_pass = perplexity <= self.perplexity_budget
        basic_pass = agreement_pass and perplexity_pass
        migration_sweep = None
        coverage_pass = True
        coverage = None
        if basic_pass and self.lesion_context is not None:
            if self.ablation_trainer is None:
                raise RuntimeError("Core-aware gate is missing its trainer")
            coverage = {
                task: dict(counts)
                for task, counts in self.ablation_trainer.suppression_visit_counts.items()
            }
            minimum_visits = int(
                self.lesion_context.get("minimum_suppression_visits", 1)
            )
            coverage_pass = all(
                count >= minimum_visits
                for counts in coverage.values()
                for count in counts.values()
            )
            if coverage_pass:
                migration_sweep = full_lesion_sweep(
                    model,
                    self.domains,
                    self.reference,
                    self.lesion_context["circuit_edges"],
                    self.lesion_context["program_nodes"],
                    self.batch_size,
                )
        migration_pass = (
            True if migration_sweep is None and self.lesion_context is None
            else bool(migration_sweep and migration_sweep["pass"])
        )
        record = {
            "step": int(state.global_step),
            "projected_agreement": agreements,
            "required_projected_agreement": self.required_agreement,
            "eval_loss": eval_loss,
            "eval_perplexity": perplexity,
            "perplexity_budget": self.perplexity_budget,
            "agreement_pass": agreement_pass,
            "perplexity_pass": perplexity_pass,
            "suppression_visit_counts": coverage,
            "suppression_coverage_pass": coverage_pass,
            "full_unsampled_lesion_sweep": migration_sweep,
            "migration_pass": migration_pass,
            "acceptance_pass": (
                basic_pass and coverage_pass and migration_pass
            ),
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


def load_circuit_edges(
    circuit_root: str, tasks=TASKS
) -> dict[str, set[tuple[str, str]]]:
    result = {}
    for task in tasks:
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
        self.tasks = tuple(behavior_domains)
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
        self.suppression_visit_counts = {
            task: {node: 0 for node in nodes}
            for task, nodes in self.intended_program_nodes.items()
        }
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
        # Trainer calls compute_loss once per gradient-accumulation microbatch.
        # Run the expensive auxiliary matrix only on the synchronized boundary
        # so visit counts mean optimizer steps rather than microbatches.
        if step % every != 0 or not self.accelerator.sync_gradients:
            return (loss, outputs) if return_outputs else loss

        base_model = self.accelerator.unwrap_model(model)
        circuit_loss = loss.new_zeros(())
        sampled_loss = loss.new_zeros(())
        joint_bypass_penalty = loss.new_zeros(())
        individual_bypass_penalty = loss.new_zeros(())
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

        individual_heads_per_aux = int(
            self.healing_config.get("core_aware_individual_heads_per_aux", 1)
        )
        if individual_heads_per_aux < 1:
            raise ValueError("core_aware_individual_heads_per_aux must be positive")

        for task_index, task in enumerate(self.tasks):
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

            intended_nodes = set(self.intended_program_nodes[task])
            bypass_full = {
                edge
                for edge in self.full_circuit_edges
                if edge[0] not in intended_nodes
            }
            bypass_core = {
                edge for edge in core if edge[0] not in intended_nodes
            }
            bypass_full_logits = controlled_forward(
                base_model,
                input_ids,
                attention_mask,
                bypass_full,
                self.circuit_graph,
            )
            bypass_core_logits = controlled_forward(
                base_model,
                input_ids,
                attention_mask,
                bypass_core,
                self.circuit_graph,
            )
            joint_bypass_penalty = (
                joint_bypass_penalty
                + self._uniform_penalty(
                    bypass_full_logits, attention_mask, candidates
                )
                + self._uniform_penalty(
                    bypass_core_logits, attention_mask, candidates
                )
            )

            nodes = self.intended_program_nodes[task]
            rotation_start = (round_index * individual_heads_per_aux + task_index) % len(nodes)
            selected_nodes = [
                nodes[(rotation_start + offset) % len(nodes)]
                for offset in range(min(individual_heads_per_aux, len(nodes)))
            ]
            for node in selected_nodes:
                self.suppression_visit_counts[task][node] += 1
                individual_full = {
                    edge for edge in self.full_circuit_edges if edge[0] != node
                }
                individual_core = {edge for edge in core if edge[0] != node}
                individual_full_logits = controlled_forward(
                    base_model,
                    input_ids,
                    attention_mask,
                    individual_full,
                    self.circuit_graph,
                )
                individual_core_logits = controlled_forward(
                    base_model,
                    input_ids,
                    attention_mask,
                    individual_core,
                    self.circuit_graph,
                )
                individual_bypass_penalty = (
                    individual_bypass_penalty
                    + self._uniform_penalty(
                        individual_full_logits, attention_mask, candidates
                    )
                    + self._uniform_penalty(
                        individual_core_logits, attention_mask, candidates
                    )
                )

        joint_weight = float(
            self.healing_config.get(
                "core_aware_joint_bypass_penalty_weight",
                self.healing_config["ablation_aware_bypass_penalty_weight"],
            )
        )
        individual_weight = float(
            self.healing_config.get(
                "core_aware_individual_bypass_penalty_weight",
                self.healing_config["ablation_aware_bypass_penalty_weight"],
            )
        )
        total = (
            loss
            + float(self.healing_config["ablation_aware_circuit_loss_weight"])
            * circuit_loss
            + float(self.healing_config["ablation_aware_sampled_loss_weight"])
            * sampled_loss
            + joint_weight * joint_bypass_penalty
            + individual_weight * individual_bypass_penalty
        )
        self.last_ablation_aware_metrics = {
            "step": step,
            "behavior_examples_per_task": aux_examples,
            "owt_loss": float(loss.detach().item()),
            "circuit_loss": float(circuit_loss.detach().item()),
            "sampled_loss": float(sampled_loss.detach().item()),
            "joint_bypass_penalty": float(
                joint_bypass_penalty.detach().item()
            ),
            "individual_bypass_penalty": float(
                individual_bypass_penalty.detach().item()
            ),
            "suppression_visit_counts": {
                task: dict(counts)
                for task, counts in self.suppression_visit_counts.items()
            },
            "total_loss": float(total.detach().item()),
        }
        return (total, outputs) if return_outputs else total


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.minimum_initial_agreement <= 1.0:
        raise ValueError("--minimum_initial_agreement must be between 0 and 1")
    tasks = tuple(dict.fromkeys(args.tasks))
    bounded_mode = args.bounded_behavior_manifest is not None
    if bounded_mode and (
        args.behavior_train_manifest is not None
        or args.behavior_gate_manifest is not None
    ):
        raise ValueError(
            "--bounded_behavior_manifest cannot be combined with train/gate manifests"
        )
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
    train_manifest = (
        args.bounded_behavior_manifest
        if bounded_mode
        else args.behavior_train_manifest
    )
    gate_manifest = (
        args.bounded_behavior_manifest
        if bounded_mode
        else args.behavior_gate_manifest or args.behavior_train_manifest
    )
    train_domains = behavior_domain(
        tokenizer, cfg, device, train_manifest, tasks
    )
    gate_domains = behavior_domain(
        tokenizer, cfg, device, gate_manifest, tasks
    )
    train_reference = reference_program_decisions(train_domains)
    gate_reference = reference_program_decisions(gate_domains)
    base_train = decisions_for_model(
        model, train_domains, cfg["behavior_batch_size"]
    )
    base_gate = decisions_for_model(
        model, gate_domains, cfg["behavior_batch_size"]
    )
    train_accuracy = {
        task: float(
            (base_train[task] == train_reference[task]).float().mean().item()
        )
        for task in tasks
    }
    gate_accuracy = {
        task: float(
            (base_gate[task] == gate_reference[task]).float().mean().item()
        )
        for task in tasks
    }
    base_accuracy = (
        {"bounded": train_accuracy}
        if bounded_mode
        else {
            "train": train_accuracy,
            "gate": gate_accuracy,
        }
    )
    if train_manifest and any(
        value != 1.0
        for split in base_accuracy.values()
        for value in split.values()
    ):
        raise RuntimeError(
            "The base model is not exact against P(x) on the locked behavior domain. "
            "Do not filter failing examples or start healing."
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
    initial = decisions_for_model(model, gate_domains, cfg["behavior_batch_size"])
    initial_agreement = {
        task: float(
            (initial[task] == gate_reference[task]).float().mean().item()
        )
        for task in tasks
    }
    if gate_manifest and any(
        value < args.minimum_initial_agreement
        for value in initial_agreement.values()
    ):
        raise RuntimeError(
            "The jointly selected programs are below the registered pre-heal "
            "agreement floor on the locked acceptance domain."
        )
    loaded_circuit_edges = (
        load_circuit_edges(args.circuit_root, tasks)
        if args.ablation_aware
        else None
    )
    initial_circuit_accuracy = None
    if args.ablation_aware:
        graph = build_circuit_graph(model.config.n_layer, model.config.n_head)
        acceptance_splits = (
            (("bounded", train_domains, train_reference),)
            if bounded_mode
            else (
                ("train", train_domains, train_reference),
                ("gate", gate_domains, gate_reference),
            )
        )
        initial_circuit_accuracy = {
            split: {} for split, _, _ in acceptance_splits
        }
        for split, split_domains, split_reference in acceptance_splits:
            for task in tasks:
                decisions = controlled_decisions_for_edges(
                    model,
                    split_domains,
                    task,
                    loaded_circuit_edges[task],
                    graph,
                    cfg["behavior_batch_size"],
                )
                accuracy = float(
                    (decisions == split_reference[task]).float().mean().item()
                )
                initial_circuit_accuracy[split][task] = accuracy
        if any(
            value < args.minimum_initial_agreement
            for split in initial_circuit_accuracy.values()
            for value in split.values()
        ):
            raise RuntimeError(
                "The programmed circuit-only forward is below the registered "
                "pre-heal agreement floor."
            )

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
    installed_program_nodes = {
        f"attn_{layer}_h_{head}" for layer, head in programs
    }
    lesion_context = None
    if args.ablation_aware:
        lesion_context = {
            "circuit_edges": loaded_circuit_edges,
            "program_nodes": installed_program_nodes,
            "minimum_suppression_visits": int(
                cfg.get("core_aware_minimum_suppression_visits", 1)
            ),
        }
    gate_callback = HealingGateCallback(
        gate_domains,
        gate_reference,
        cfg["behavior_batch_size"],
        cfg["projected_agreement_required"],
        perplexity_budget,
        args.output_dir,
        lesion_context=lesion_context,
    )
    trainer_class = AblationAwareProgramTrainer if args.ablation_aware else Trainer
    trainer_kwargs = {}
    if args.ablation_aware:
        trainer_kwargs.update(
            {
                "behavior_domains": train_domains,
                "reference_decisions": train_reference,
                "circuit_edges": loaded_circuit_edges,
                "program_nodes": installed_program_nodes,
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
    if args.ablation_aware:
        gate_callback.ablation_trainer = trainer
    resume_checkpoint = args.resume_from_checkpoint
    if resume_checkpoint is None and not args.disable_auto_resume:
        resume_checkpoint = find_latest_checkpoint(args.output_dir)
        if resume_checkpoint is not None and local_rank == 0:
            print(f"Auto-resuming healing from checkpoint: {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    eval_metrics = trainer.evaluate()
    final = decisions_for_model(
        trainer.model, gate_domains, cfg["behavior_batch_size"]
    )
    final_agreement = {
        task: float(
            (final[task] == gate_reference[task]).float().mean().item()
        )
        for task in tasks
    }
    eval_perplexity = math.exp(float(eval_metrics["eval_loss"]))
    budget = perplexity_budget
    agreement_pass = all(
        value >= cfg["projected_agreement_required"]
        for value in final_agreement.values()
    )
    perplexity_pass = eval_perplexity <= budget
    final_gate_record = gate_callback.history[-1] if gate_callback.history else None
    migration_pass = (
        bool(final_gate_record and final_gate_record.get("migration_pass"))
        if args.ablation_aware
        else True
    )
    suppression_coverage_pass = (
        bool(
            final_gate_record
            and final_gate_record.get("suppression_coverage_pass")
        )
        if args.ablation_aware
        else True
    )
    result = {
        "source_model": args.model_path,
        "resume_from_checkpoint": resume_checkpoint,
        "programs": args.programs,
        "program_heads": [f"{layer}.{head}" for layer, head in sorted(programs)],
        "replacement_fraction": (
            len(programs) / model.config.n_layer / model.config.n_head
        ),
        "programs_frozen": True,
        "method": (
            "owt_plus_exact_P_targets_stochastic_edge_ablation_and_"
            "rotating_joint_and_individual_full_and_core_bypass_suppression"
            if args.ablation_aware
            else "owt_healing"
        ),
        "ablation_aware": args.ablation_aware,
        "circuit_root": args.circuit_root,
        "behavior_domain_mode": (
            "bounded_domain" if bounded_mode else "train_and_gate"
        ),
        "tasks": list(tasks),
        "claim_scope": (
            "exact only on the declared finite bounded domain; no held-out claim"
            if bounded_mode
            else "locked train and gate domains"
        ),
        "bounded_behavior_manifest": (
            args.bounded_behavior_manifest if bounded_mode else None
        ),
        "behavior_train_manifest": None if bounded_mode else train_manifest,
        "behavior_gate_manifest": None if bounded_mode else gate_manifest,
        "behavior_domain": (
            {
                "bounded": {
                    task: train_domains[task]["provenance"] for task in tasks
                }
            }
            if bounded_mode
            else {
                "train": {
                    task: train_domains[task]["provenance"] for task in tasks
                },
                "gate": {
                    task: gate_domains[task]["provenance"] for task in tasks
                },
            }
        ),
        "reference_target": "explicit_reference_program_P(x)",
        "base_projected_accuracy": base_accuracy,
        "ablation_aware_last_metrics": getattr(
            trainer, "last_ablation_aware_metrics", None
        ),
        "initial_projected_agreement": initial_agreement,
        "minimum_initial_agreement": args.minimum_initial_agreement,
        "initial_circuit_accuracy_against_P": initial_circuit_accuracy,
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
        "migration_pass": migration_pass,
        "suppression_coverage_pass": suppression_coverage_pass,
        "full_unsampled_lesion_sweep": (
            None
            if final_gate_record is None
            else final_gate_record.get("full_unsampled_lesion_sweep")
        ),
        "stopping_criteria": (
            "projected accuracy against P(x) on every task in the declared "
            "bounded domain must equal the configured threshold; eval perplexity "
            "must stay within the pre-registered budget; core-aware runs also "
            "require suppression coverage and the complete unsampled full/core "
            "lesion sweep"
            if bounded_mode
            else "projected accuracy against P(x) on both locked gate domains "
            "must equal the configured threshold; eval perplexity must stay "
            "within the pre-registered budget; core-aware runs also require "
            "suppression coverage and the complete unsampled full/core lesion sweep"
        ),
        "gate_history": gate_callback.history,
        "success": (
            agreement_pass
            and perplexity_pass
            and migration_pass
            and suppression_coverage_pass
        ),
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
