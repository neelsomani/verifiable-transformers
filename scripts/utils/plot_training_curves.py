import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training/eval loss curves from Trainer state")
    parser.add_argument(
        "--run_dir",
        type=str,
        default="artifacts/gpt2-baseline",
        help="Run output directory containing trainer_state.json",
    )
    parser.add_argument(
        "--trainer_state_path",
        type=str,
        default=None,
        help="Optional explicit path to trainer_state.json",
    )
    parser.add_argument(
        "--output_png",
        type=str,
        default=None,
        help="Output PNG path. Defaults to <run_dir>/training_curves.png",
    )
    return parser.parse_args()


def load_log_history(trainer_state_path: str):
    with open(trainer_state_path, "r", encoding="utf-8") as handle:
        state = json.load(handle)
    return state.get("log_history", [])


def extract_points(log_history):
    train_steps = []
    train_losses = []
    eval_steps = []
    eval_losses = []

    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue
        if "loss" in entry:
            train_steps.append(step)
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(step)
            eval_losses.append(entry["eval_loss"])

    return train_steps, train_losses, eval_steps, eval_losses


def main() -> None:
    args = parse_args()

    trainer_state_path = args.trainer_state_path
    if trainer_state_path is None:
        trainer_state_path = os.path.join(args.run_dir, "trainer_state.json")

    if not os.path.isfile(trainer_state_path):
        raise FileNotFoundError(f"trainer_state.json not found at: {trainer_state_path}")

    output_png = args.output_png
    if output_png is None:
        output_png = os.path.join(args.run_dir, "training_curves.png")

    log_history = load_log_history(trainer_state_path)
    train_steps, train_losses, eval_steps, eval_losses = extract_points(log_history)

    if not train_steps and not eval_steps:
        raise ValueError("No train/eval loss points found in trainer_state.json log_history")

    plt.figure(figsize=(10, 6))
    if train_steps:
        plt.plot(train_steps, train_losses, label="train_loss", linewidth=1.5)
    if eval_steps:
        plt.plot(eval_steps, eval_losses, label="eval_loss", linewidth=2.0)

    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
    plt.savefig(output_png, dpi=150)
    print(f"Saved plot to: {output_png}")


if __name__ == "__main__":
    main()
