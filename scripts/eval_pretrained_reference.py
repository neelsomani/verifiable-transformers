import argparse
import json
import math
import os

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained GPT-2 on OWT validation and WikiText-103"
    )
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--owt_eval_percent", type=float, default=1.0)
    parser.add_argument("--owt_max_samples", type=int, default=10000)
    parser.add_argument("--wikitext_split", type=str, default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--wikitext_max_samples", type=int, default=None)
    parser.add_argument("--output_json", type=str, default="artifacts/pretrained-gpt2-reference-metrics.json")
    return parser.parse_args()


def compute_ppl_and_loss(model, tokenizer, text: str, block_size: int, stride: int):
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)

    nlls = []
    seq_len = input_ids.size(1)
    prev_end = 0

    for begin in tqdm(range(0, seq_len, stride), desc="Evaluating", leave=False):
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


def load_owt_validation_text(percent: float, max_samples: int):
    train_end = max(0.0, 100.0 - percent)
    split = f"train[{train_end:.2f}%:]"
    dataset = load_dataset("openwebtext", split=split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return "\n\n".join(dataset["text"])


def load_wikitext_text(split: str, max_samples: int):
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return "\n\n".join(dataset["text"])


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    model.eval()

    print("Running OpenWebText validation evaluation...")
    owt_text = load_owt_validation_text(args.owt_eval_percent, args.owt_max_samples)
    owt_loss, owt_ppl, owt_seq_len = compute_ppl_and_loss(
        model,
        tokenizer,
        owt_text,
        args.block_size,
        args.stride,
    )

    print("Running WikiText-103 evaluation...")
    wt_text = load_wikitext_text(args.wikitext_split, args.wikitext_max_samples)
    wt_loss, wt_ppl, wt_seq_len = compute_ppl_and_loss(
        model,
        tokenizer,
        wt_text,
        args.block_size,
        args.stride,
    )

    result = {
        "model_name": args.model_name,
        "evaluation": {
            "openwebtext_validation": {
                "eval_percent": args.owt_eval_percent,
                "max_samples": args.owt_max_samples,
                "loss": owt_loss,
                "perplexity": owt_ppl,
                "seq_len": owt_seq_len,
            },
            "wikitext103": {
                "split": args.wikitext_split,
                "max_samples": args.wikitext_max_samples,
                "loss": wt_loss,
                "perplexity": wt_ppl,
                "seq_len": wt_seq_len,
            },
        },
    }

    print(json.dumps(result, indent=2))

    if args.output_json:
        output_dir = "/".join(args.output_json.split("/")[:-1])
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
