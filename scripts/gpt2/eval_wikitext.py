import argparse
import json
import math
import os
import sys

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.gpt2.extract import load_model_with_variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate causal LM perplexity on WikiText-103")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model or HF model id.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
    )
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if os.path.isdir(args.model_path):
        # Local experiment checkpoints require model_info.json to reconstruct
        # custom norm, attention, activation, and program-head variants before
        # their weights are loaded. AutoModelForCausalLM would silently build a
        # vanilla GPT-2 graph and produce meaningless perplexities.
        model = load_model_with_variants(args.model_path, str(device))
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)
    model.eval()

    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=args.split)
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    joined_text = "\n\n".join(dataset["text"])
    encodings = tokenizer(joined_text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    nlls = []
    seq_len = input_ids.size(1)
    prev_end = 0

    for begin in tqdm(range(0, seq_len, args.stride), desc="Evaluating"):
        end = min(begin + args.block_size, seq_len)
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
    loss = (total_nll / seq_len).item()
    ppl = math.exp(loss)

    result = {
        "dataset": "wikitext-103-raw-v1",
        "split": args.split,
        "loss": loss,
        "perplexity": ppl,
        "seq_len": int(seq_len),
        "max_samples": args.max_samples,
    }
    print(json.dumps(result, indent=2))

    if args.output_json is not None:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
