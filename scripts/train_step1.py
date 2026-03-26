import argparse
import json
import math
import os
import re
import shutil
import time
from datetime import datetime, timezone
from itertools import chain
import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
import transformers


class EvalLossThresholdStopCallback(TrainerCallback):
    def __init__(self, target_eval_loss: float):
        self.target_eval_loss = target_eval_loss

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return control
        if eval_loss <= self.target_eval_loss:
            print(
                f"Early stopping triggered: eval_loss={eval_loss:.4f} <= target={self.target_eval_loss:.4f}"
            )
            control.should_training_stop = True
        return control


def evaluate_causal_lm_perplexity(model, input_ids: torch.Tensor, block_size: int, stride: int):
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    nlls = []
    seq_len = input_ids.size(1)
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + block_size, seq_len)
        target_len = end - prev_end
        ids = input_ids[:, begin:end]
        labels = ids.clone()
        labels[:, :-target_len] = -100

        with torch.no_grad():
            outputs = model(ids, labels=labels)
            neg_log_likelihood = outputs.loss * target_len
        nlls.append(neg_log_likelihood)
        prev_end = end

        if end == seq_len:
            break

    total_nll = torch.stack(nlls).sum()
    avg_nll = (total_nll / seq_len).item()
    ppl = math.exp(avg_nll)
    return avg_nll, ppl, int(seq_len)


def load_wikitext_input_ids(tokenizer, split: str, max_samples: int | None):
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    joined_text = "\n\n".join(dataset["text"])
    return tokenizer(joined_text, return_tensors="pt").input_ids


class WikiTextEvalCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        output_dir: str,
        split: str,
        block_size: int,
        stride: int,
        max_samples: int | None,
        eval_every_n_evals: int,
        target_ppl: float | None,
    ):
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.split = split
        self.block_size = block_size
        self.stride = stride
        self.max_samples = max_samples
        self.eval_every_n_evals = eval_every_n_evals
        self.target_ppl = target_ppl
        self.eval_counter = 0
        self.wikitext_input_ids = None
        self.target_reached_marker = os.path.join(output_dir, "wikitext_target_reached.json")

    def _evaluate_and_save(self, model, step: int):
        if self.wikitext_input_ids is None:
            self.wikitext_input_ids = load_wikitext_input_ids(
                self.tokenizer,
                split=self.split,
                max_samples=self.max_samples,
            )

        loss, ppl, seq_len = evaluate_causal_lm_perplexity(
            model,
            input_ids=self.wikitext_input_ids,
            block_size=self.block_size,
            stride=self.stride,
        )
        metrics = {
            "dataset": "wikitext-103-raw-v1",
            "split": self.split,
            "step": int(step),
            "loss": loss,
            "perplexity": ppl,
            "seq_len": seq_len,
            "max_samples": self.max_samples,
        }
        step_path = os.path.join(self.output_dir, f"wikitext_eval_step_{int(step)}.json")
        latest_path = os.path.join(self.output_dir, "wikitext_eval_latest.json")
        with open(step_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        with open(latest_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        print(
            f"WikiText eval at step {int(step)}: loss={loss:.4f}, perplexity={ppl:.4f}"
        )
        return metrics

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, model=None, metrics=None, **kwargs):
        self.eval_counter += 1

        if os.path.isfile(self.target_reached_marker):
            control.should_training_stop = True
            return control

        if self.eval_every_n_evals <= 0 or self.eval_counter % self.eval_every_n_evals != 0:
            return control

        if state.is_world_process_zero:
            wikitext_metrics = self._evaluate_and_save(model=model, step=state.global_step)
            if metrics is not None:
                metrics["eval_wikitext_loss"] = wikitext_metrics["loss"]
                metrics["eval_wikitext_perplexity"] = wikitext_metrics["perplexity"]
            if self.target_ppl is not None and wikitext_metrics["perplexity"] <= self.target_ppl:
                with open(self.target_reached_marker, "w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "target_ppl": self.target_ppl,
                            "achieved_ppl": wikitext_metrics["perplexity"],
                            "step": int(state.global_step),
                        },
                        handle,
                        indent=2,
                    )
                print(
                    f"Early stopping triggered: WikiText perplexity={wikitext_metrics['perplexity']:.4f} <= "
                    f"target={self.target_ppl:.4f}"
                )

        if os.path.isfile(self.target_reached_marker):
            control.should_training_stop = True
        return control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 1 baseline: GPT-2 small on OpenWebText")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/step1_gpt2_small_openwebtext.json",
        help="Path to JSON config file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/step1-gpt2-small-openwebtext",
        help="Training output directory.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming dataset mode for OWT.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Optional cap for local smoke tests.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=10000,
        help="Optional eval cap.",
    )
    parser.add_argument(
        "--early_stop_eval_loss",
        type=float,
        default=None,
        help="Stop training once eval_loss is <= this threshold.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Explicit checkpoint path to resume from.",
    )
    parser.add_argument(
        "--disable_auto_resume",
        action="store_true",
        help="Disable automatic resume from latest checkpoint in output_dir.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override max training steps from config.",
    )
    parser.add_argument(
        "--processed_dataset_dir",
        type=str,
        default=None,
        help="Path to save/load preprocessed tokenized dataset.",
    )
    parser.add_argument(
        "--preprocessing_num_proc",
        type=int,
        default=None,
        help="Number of CPU processes for dataset.map preprocessing.",
    )
    parser.add_argument(
        "--evaluate_wikitext_at_end",
        action="store_true",
        help="Run WikiText-103 evaluation after training completes.",
    )
    parser.add_argument(
        "--use_wikitext_as_dev",
        action="store_true",
        help="Use periodic WikiText eval as an opt-in dev criterion mode.",
    )
    parser.add_argument(
        "--target_wikitext_ppl",
        type=float,
        default=None,
        help="Optional target perplexity for early stopping based on WikiText-103.",
    )
    parser.add_argument(
        "--wikitext_eval_every_n_evals",
        type=int,
        default=0,
        help="Run WikiText eval every N Trainer eval events (0 disables periodic WikiText eval).",
    )
    parser.add_argument(
        "--wikitext_split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
    )
    parser.add_argument(
        "--wikitext_max_samples",
        type=int,
        default=None,
        help="Optional sample cap for WikiText evaluation.",
    )
    parser.add_argument(
        "--wikitext_block_size",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--wikitext_stride",
        type=int,
        default=1024,
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def find_latest_checkpoint(output_dir: str):
    if not os.path.isdir(output_dir):
        return None
    pattern = re.compile(r"^checkpoint-(\d+)$")
    candidates = []
    for name in os.listdir(output_dir):
        match = pattern.match(name)
        if match:
            candidates.append((int(match.group(1)), os.path.join(output_dir, name)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def count_params(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def write_run_status(output_dir: str, status: str, stage: str, extra: dict = None) -> None:
    payload = {
        "status": status,
        "stage": stage,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "run_status.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def get_distributed_context():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world_size


def processed_dataset_ready_marker_path(processed_dataset_dir: str) -> str:
    return os.path.join(processed_dataset_dir, "_READY")


def is_processed_dataset_ready(processed_dataset_dir: str) -> bool:
    if not os.path.isdir(processed_dataset_dir):
        return False
    if os.path.isfile(processed_dataset_ready_marker_path(processed_dataset_dir)):
        return True
    dataset_dict_file = os.path.join(processed_dataset_dir, "dataset_dict.json")
    train_dir = os.path.join(processed_dataset_dir, "train")
    validation_dir = os.path.join(processed_dataset_dir, "validation")
    return os.path.isfile(dataset_dict_file) and os.path.isdir(train_dir) and os.path.isdir(validation_dir)


def mark_processed_dataset_ready(processed_dataset_dir: str) -> None:
    with open(processed_dataset_ready_marker_path(processed_dataset_dir), "w", encoding="utf-8") as handle:
        json.dump({"ready": True, "updated_at_utc": datetime.now(timezone.utc).isoformat()}, handle)


def default_processed_dataset_dir(args, cfg) -> str:
    train_tag = "all" if args.max_train_samples is None else str(args.max_train_samples)
    eval_tag = "all" if args.max_eval_samples is None else str(args.max_eval_samples)
    dataset_tag = cfg["dataset_name"].replace("/", "-")
    return os.path.join(
        "artifacts",
        "processed",
        f"{dataset_tag}_block{cfg['block_size']}_train{train_tag}_eval{eval_tag}",
    )


def tokenize_and_group(
    raw_datasets: DatasetDict,
    tokenizer,
    block_size: int,
    preprocessing_num_proc: int,
) -> DatasetDict:
    def tokenize_function(batch):
        return tokenizer(batch["text"])

    tokenized = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=preprocessing_num_proc,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    def group_texts(batch):
        concatenated = {
            key: list(chain.from_iterable(batch[key]))
            for key in batch.keys()
        }
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        result = {
            key: [tokens[i : i + block_size] for i in range(0, total_length, block_size)]
            for key, tokens in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    grouped = tokenized.map(
        group_texts,
        batched=True,
        num_proc=preprocessing_num_proc,
        desc=f"Grouping into blocks of {block_size}",
    )
    return grouped


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if transformers.__version__ != "5.3.0":
        raise RuntimeError(
            f"Expected transformers==5.3.0, found {transformers.__version__}. "
            "Install with: pip install transformers==5.3.0"
        )
    set_seed(cfg["seed"])
    os.makedirs(args.output_dir, exist_ok=True)
    write_run_status(args.output_dir, status="running", stage="initializing")

    try:
        preprocessing_num_proc = args.preprocessing_num_proc
        if preprocessing_num_proc is None:
            preprocessing_num_proc = cfg.get("preprocessing_num_proc")
        if preprocessing_num_proc is None:
            cpu_count = os.cpu_count() or 1
            preprocessing_num_proc = max(1, cpu_count // 2)

        processed_dataset_dir = args.processed_dataset_dir
        if processed_dataset_dir is None:
            processed_dataset_dir = cfg.get("processed_dataset_dir")
        if processed_dataset_dir is None:
            processed_dataset_dir = default_processed_dataset_dir(args, cfg)

        write_run_status(
            args.output_dir,
            status="running",
            stage="building_model",
            extra={
                "processed_dataset_dir": processed_dataset_dir,
                "preprocessing_num_proc": preprocessing_num_proc,
            },
        )

        model_name = cfg["model_name"]
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_config = GPT2Config.from_pretrained(model_name)
        if hasattr(model_config, "n_positions"):
            model_config.n_positions = cfg["block_size"]
        if hasattr(model_config, "n_ctx"):
            model_config.n_ctx = cfg["block_size"]
        model = GPT2LMHeadModel(model_config)
        model_num_params = count_params(model)
        print(f"Model params: {model_num_params:,}")
        with open(os.path.join(args.output_dir, "model_info.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "model_name": model_name,
                    "num_parameters": model_num_params,
                },
                handle,
                indent=2,
            )
        if cfg.get("gradient_checkpointing", False):
            model.gradient_checkpointing_enable()

        dataset_name = cfg["dataset_name"]
        rank, world_size = get_distributed_context()
        is_main_process = rank == 0
        if args.streaming:
            raise ValueError("Streaming mode is not supported with Trainer in this script.")
        else:
            if is_processed_dataset_ready(processed_dataset_dir):
                write_run_status(
                    args.output_dir,
                    status="running",
                    stage="loading_processed_dataset",
                    extra={"processed_dataset_dir": processed_dataset_dir},
                )
                print(f"Loading preprocessed dataset from: {processed_dataset_dir}")
                lm_datasets = load_from_disk(processed_dataset_dir)
            else:
                if world_size > 1 and not is_main_process:
                    write_run_status(
                        args.output_dir,
                        status="running",
                        stage="waiting_for_processed_dataset",
                        extra={"processed_dataset_dir": processed_dataset_dir, "rank": rank},
                    )
                    print(
                        f"Rank {rank} waiting for rank 0 to preprocess dataset at: {processed_dataset_dir}"
                    )
                    while not is_processed_dataset_ready(processed_dataset_dir):
                        time.sleep(10)
                    lm_datasets = load_from_disk(processed_dataset_dir)
                else:
                    write_run_status(
                        args.output_dir,
                        status="running",
                        stage="preprocessing_dataset",
                        extra={"processed_dataset_dir": processed_dataset_dir, "rank": rank},
                    )
                    if os.path.isdir(processed_dataset_dir):
                        print(f"Removing incomplete processed dataset dir: {processed_dataset_dir}")
                        shutil.rmtree(processed_dataset_dir)
                    raw_train = load_dataset(dataset_name, split="train[:-1%]")
                    raw_eval = load_dataset(dataset_name, split="train[-1%:]")
                    if args.max_train_samples is not None:
                        raw_train = raw_train.select(range(min(args.max_train_samples, len(raw_train))))
                    if args.max_eval_samples is not None:
                        raw_eval = raw_eval.select(range(min(args.max_eval_samples, len(raw_eval))))

                    raw_datasets = DatasetDict({"train": raw_train, "validation": raw_eval})
                    lm_datasets = tokenize_and_group(
                        raw_datasets,
                        tokenizer,
                        cfg["block_size"],
                        preprocessing_num_proc,
                    )
                    os.makedirs(os.path.dirname(processed_dataset_dir), exist_ok=True)
                    lm_datasets.save_to_disk(processed_dataset_dir)
                    mark_processed_dataset_ready(processed_dataset_dir)
                    print(f"Saved preprocessed dataset to: {processed_dataset_dir}")

        data_collator = default_data_collator

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            do_train=True,
            do_eval=True,
            per_device_train_batch_size=cfg["train_batch_size_per_device"],
            per_device_eval_batch_size=cfg["eval_batch_size_per_device"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            learning_rate=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
            max_grad_norm=cfg.get("max_grad_norm", 1.0),
            adam_beta1=cfg["adam_beta1"],
            adam_beta2=cfg["adam_beta2"],
            adam_epsilon=cfg["adam_epsilon"],
            max_steps=args.max_steps if args.max_steps is not None else cfg["max_steps"],
            warmup_steps=cfg["warmup_steps"],
            lr_scheduler_type=cfg["lr_scheduler_type"],
            eval_strategy="steps",
            eval_steps=cfg["eval_steps"],
            save_steps=cfg["save_steps"],
            logging_steps=cfg["logging_steps"],
            save_total_limit=cfg["save_total_limit"],
            dataloader_num_workers=cfg["dataloader_num_workers"],
            bf16=cfg["bf16"],
            fp16=cfg["fp16"],
            torch_compile=cfg.get("torch_compile", False),
            report_to=cfg["report_to"],
        )

        early_stop_eval_loss = args.early_stop_eval_loss
        if early_stop_eval_loss is None:
            early_stop_eval_loss = cfg.get("early_stop_eval_loss")

        wikitext_eval_every_n_evals = args.wikitext_eval_every_n_evals
        if args.use_wikitext_as_dev and wikitext_eval_every_n_evals <= 0:
            wikitext_eval_every_n_evals = 1
        if args.target_wikitext_ppl is not None and wikitext_eval_every_n_evals <= 0:
            wikitext_eval_every_n_evals = 1

        if (args.use_wikitext_as_dev or args.target_wikitext_ppl is not None) and args.early_stop_eval_loss is None:
            early_stop_eval_loss = None

        callbacks = []
        if early_stop_eval_loss is not None:
            callbacks.append(EvalLossThresholdStopCallback(target_eval_loss=float(early_stop_eval_loss)))
        if wikitext_eval_every_n_evals > 0:
            callbacks.append(
                WikiTextEvalCallback(
                    tokenizer=tokenizer,
                    output_dir=args.output_dir,
                    split=args.wikitext_split,
                    block_size=args.wikitext_block_size,
                    stride=args.wikitext_stride,
                    max_samples=args.wikitext_max_samples,
                    eval_every_n_evals=wikitext_eval_every_n_evals,
                    target_ppl=args.target_wikitext_ppl,
                )
            )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=lm_datasets["train"],
            eval_dataset=lm_datasets["validation"],
            processing_class=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
        )

        resume_checkpoint = args.resume_from_checkpoint
        if resume_checkpoint is None and not args.disable_auto_resume:
            resume_checkpoint = find_latest_checkpoint(args.output_dir)
            if resume_checkpoint is not None:
                print(f"Auto-resuming from checkpoint: {resume_checkpoint}")

        write_run_status(
            args.output_dir,
            status="running",
            stage="training",
            extra={
                "resume_from_checkpoint": resume_checkpoint,
                "max_steps": training_args.max_steps,
            },
        )

        train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        write_run_status(args.output_dir, status="running", stage="final_eval")
        eval_metrics = trainer.evaluate()
        eval_metrics["perplexity"] = math.exp(eval_metrics["eval_loss"])
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

        final_wikitext_metrics = None
        if args.evaluate_wikitext_at_end or args.target_wikitext_ppl is not None:
            if rank == 0:
                wikitext_input_ids = load_wikitext_input_ids(
                    tokenizer,
                    split=args.wikitext_split,
                    max_samples=args.wikitext_max_samples,
                )
                wt_loss, wt_ppl, wt_seq_len = evaluate_causal_lm_perplexity(
                    trainer.model,
                    input_ids=wikitext_input_ids,
                    block_size=args.wikitext_block_size,
                    stride=args.wikitext_stride,
                )
                final_wikitext_metrics = {
                    "dataset": "wikitext-103-raw-v1",
                    "split": args.wikitext_split,
                    "loss": wt_loss,
                    "perplexity": wt_ppl,
                    "seq_len": wt_seq_len,
                    "max_samples": args.wikitext_max_samples,
                }
                with open(os.path.join(args.output_dir, "wikitext_eval_final.json"), "w", encoding="utf-8") as handle:
                    json.dump(final_wikitext_metrics, handle, indent=2)
                print(
                    f"Final WikiText eval: loss={wt_loss:.4f}, perplexity={wt_ppl:.4f}"
                )

        write_run_status(
            args.output_dir,
            status="completed",
            stage="done",
            extra={
                "final_train_loss": metrics.get("train_loss"),
                "final_eval_loss": eval_metrics.get("eval_loss"),
                "final_eval_perplexity": eval_metrics.get("perplexity"),
                "final_wikitext_perplexity": None if final_wikitext_metrics is None else final_wikitext_metrics["perplexity"],
            },
        )
    except KeyboardInterrupt:
        write_run_status(args.output_dir, status="interrupted", stage="interrupted")
        raise
    except Exception as error:
        write_run_status(
            args.output_dir,
            status="failed",
            stage="failed",
            extra={"error": str(error)},
        )
        raise


if __name__ == "__main__":
    main()
