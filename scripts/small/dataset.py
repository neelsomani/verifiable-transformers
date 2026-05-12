"""
Dataset generation for the small verifiable Transformer.

Generates exhaustive training data for three symbolic tasks:
1. quote_close: Match opening quote (' or ")
2. bracket_type: Match opening bracket ([ or {)
3. add_mod_5: Addition modulo 5
"""

import itertools
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset

from . import vocab


class SmallVerifiableDataset(Dataset):
    """
    Exhaustive dataset for small verifiable Transformer tasks.

    Generates all possible examples for each task:
    - quote_close: 2 * 4^3 = 128 examples
    - bracket_type: 2 * 4^3 = 128 examples
    - add_mod_5: 5 * 5 = 25 examples

    Total: 281 examples
    """

    def __init__(self, task_sampling: str = "balanced"):
        """
        Args:
            task_sampling: How to sample tasks during training
                - "balanced": Equal probability per task (1/3 each)
                - "proportional": Sample proportional to dataset size
                - "all": Include all examples once per epoch
        """
        self.task_sampling = task_sampling

        # Generate exhaustive examples for each task
        self.quote_examples = self._generate_quote_examples()
        self.bracket_examples = self._generate_bracket_examples()
        self.add_examples = self._generate_add_examples()

        # Combine all examples
        self.all_examples = (
            self.quote_examples +
            self.bracket_examples +
            self.add_examples
        )

        # Store per-task examples for evaluation
        self.task_examples = {
            "quote_close": self.quote_examples,
            "bracket_type": self.bracket_examples,
            "add_mod_5": self.add_examples,
        }

        print(f"Dataset created:")
        print(f"  quote_close: {len(self.quote_examples)} examples")
        print(f"  bracket_type: {len(self.bracket_examples)} examples")
        print(f"  add_mod_5: {len(self.add_examples)} examples")
        print(f"  total: {len(self.all_examples)} examples")
        print(f"  task_sampling: {task_sampling}")

    def _generate_quote_examples(self) -> List[Dict]:
        """
        Generate all quote_close examples.

        Pattern: BOS TASK_QUOTE x1 quote x2 x3 -> quote
        where x1, x2, x3 are content tokens and quote is ' or "
        """
        examples = []

        for x1, x2, x3 in itertools.product(vocab.CONTENT_TOKENS, repeat=3):
            for quote in [vocab.SINGLE_QUOTE, vocab.DOUBLE_QUOTE]:
                input_ids = [
                    vocab.BOS,
                    vocab.TASK_QUOTE,
                    x1,
                    quote,
                    x2,
                    x3
                ]
                target = quote

                examples.append({
                    "input_ids": input_ids,
                    "target": target,
                    "task": "quote_close",
                    "task_token": vocab.TASK_QUOTE,
                })

        return examples

    def _generate_bracket_examples(self) -> List[Dict]:
        """
        Generate all bracket_type examples.

        Pattern: BOS TASK_BRACKET x1 open x2 x3 -> close
        where x1, x2, x3 are content tokens
        and (open, close) is ([ , ]) or ({ , })
        """
        examples = []

        bracket_pairs = [
            (vocab.LEFT_BRACKET, vocab.RIGHT_BRACKET),
            (vocab.LEFT_BRACE, vocab.RIGHT_BRACE),
        ]

        for x1, x2, x3 in itertools.product(vocab.CONTENT_TOKENS, repeat=3):
            for open_bracket, close_bracket in bracket_pairs:
                input_ids = [
                    vocab.BOS,
                    vocab.TASK_BRACKET,
                    x1,
                    open_bracket,
                    x2,
                    x3
                ]
                target = close_bracket

                examples.append({
                    "input_ids": input_ids,
                    "target": target,
                    "task": "bracket_type",
                    "task_token": vocab.TASK_BRACKET,
                })

        return examples

    def _generate_add_examples(self) -> List[Dict]:
        """
        Generate all add_mod_5 examples.

        Pattern: BOS TASK_ADD digit_a PLUS digit_b EQ -> digit_((a+b) mod 5)
        where digit_a, digit_b are in {0, 1, 2, 3, 4}
        """
        examples = []

        for a in range(5):
            for b in range(5):
                a_token = vocab.value_to_digit(a)
                b_token = vocab.value_to_digit(b)
                result = (a + b) % 5
                result_token = vocab.value_to_digit(result)

                input_ids = [
                    vocab.BOS,
                    vocab.TASK_ADD,
                    a_token,
                    vocab.PLUS,
                    b_token,
                    vocab.EQ
                ]
                target = result_token

                examples.append({
                    "input_ids": input_ids,
                    "target": target,
                    "task": "add_mod_5",
                    "task_token": vocab.TASK_ADD,
                    "operand_a": a,
                    "operand_b": b,
                    "result": result,
                })

        return examples

    def __len__(self) -> int:
        if self.task_sampling == "all":
            return len(self.all_examples)
        else:
            # For balanced/proportional sampling, we define epoch length
            # as 3x the largest task (to ensure good coverage)
            return 3 * max(
                len(self.quote_examples),
                len(self.bracket_examples),
                len(self.add_examples)
            )

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a training example.

        For balanced sampling: randomly select a task, then an example from that task
        For proportional sampling: sample from all examples
        For all sampling: iterate through all examples
        """
        if self.task_sampling == "all":
            example = self.all_examples[idx % len(self.all_examples)]
        elif self.task_sampling == "balanced":
            # Randomly select a task with equal probability
            import random
            task_name = random.choice(["quote_close", "bracket_type", "add_mod_5"])
            task_examples = self.task_examples[task_name]
            example = random.choice(task_examples)
        else:  # proportional
            import random
            example = random.choice(self.all_examples)

        # Convert to tensors
        input_ids = torch.tensor(example["input_ids"], dtype=torch.long)
        target = torch.tensor(example["target"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "target": target,
            "task": example["task"],
            "task_token": example["task_token"],
        }


def get_eval_dataset(task_name: str) -> List[Dict]:
    """
    Get exhaustive evaluation dataset for a specific task.

    Args:
        task_name: One of "quote_close", "bracket_type", "add_mod_5"

    Returns:
        List of examples with input_ids and target
    """
    dataset = SmallVerifiableDataset(task_sampling="all")
    return dataset.task_examples[task_name]


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for DataLoader.

    All sequences have the same length (6 tokens), so no padding needed.
    Only returns model-consumed tensors (input_ids, targets).
    Task metadata is stripped before feeding into the model.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    targets = torch.stack([item["target"] for item in batch])

    return {
        "input_ids": input_ids,
        "targets": targets,
    }


if __name__ == "__main__":
    # Test dataset generation
    print("Testing dataset generation...")
    print("=" * 50)

    dataset = SmallVerifiableDataset(task_sampling="balanced")

    # Print a few examples from each task
    for task_name in ["quote_close", "bracket_type", "add_mod_5"]:
        examples = dataset.task_examples[task_name]
        print(f"\n{task_name} examples (first 3):")
        for i, ex in enumerate(examples[:3]):
            input_str = vocab.tokens_to_str(ex["input_ids"])
            target_str = vocab.token_to_str(ex["target"])
            print(f"  {i+1}. {input_str} -> {target_str}")

    # Test __getitem__
    print("\n" + "=" * 50)
    print("Testing __getitem__ (5 random samples):")
    for i in range(5):
        item = dataset[i]
        input_str = vocab.tokens_to_str(item["input_ids"].tolist())
        target = item["target"].item()
        target_str = vocab.token_to_str(target)
        print(f"  {item['task']}: {input_str} -> {target_str}")
