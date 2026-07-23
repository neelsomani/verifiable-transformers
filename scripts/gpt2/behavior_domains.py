"""Versioned behavior domains for GPT-2 circuit experiments.

Protocol v1 is retained byte-for-byte for regression measurements. Protocols
v2 through v4 are deterministic, unique-prompt domains with disjoint
development and gate material. The bounded-v1 continuation is the frozen union
of v4's two splits and makes no held-out-generalization claim. The generator
never consults model predictions when constructing or selecting examples.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


TASKS = ("quote_close", "bracket_type")
PROTOCOL_ID = "gpt2_behavior_domain_v2"
SUPPORTED_PROTOCOL_IDS = {
    "gpt2_behavior_domain_v2",
    "gpt2_behavior_domain_v3",
    "gpt2_behavior_domain_v4",
    "gpt2_behavior_domain_bounded_v1",
}
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class BehaviorExample:
    """One projected two-candidate next-token example."""

    prompt: str
    correct_token: str
    incorrect_token: str
    example_id: str | None = None
    task: str | None = None
    split: str | None = None
    stratum: str | None = None
    template_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BehaviorExample":
        return cls(
            prompt=value["prompt"],
            correct_token=value["correct_token"],
            incorrect_token=value["incorrect_token"],
            example_id=value.get("example_id"),
            task=value.get("task"),
            split=value.get("split"),
            stratum=value.get("stratum"),
            template_id=value.get("template_id"),
            metadata=value.get("metadata", {}),
        )


LEGACY_TEMPLATES = {
    "quote_close": {
        "single": [
            "x = 'hello world",
            "print('hello world",
            "message = 'foo bar",
            "return 'some text",
            "data.append('value",
            "name = 'alice",
            "s = 'test string",
            "key = 'item",
        ],
        "double": [
            'x = "hello world',
            'print("hello world',
            'message = "foo bar',
            'return "some text',
            'data.append("value',
            'name = "alice',
            's = "test string',
            'key = "item',
        ],
    },
    "bracket_type": {
        "bracket": [
            "x = [a, b, c",
            "items = [foo, bar",
            "return [x, y, z",
            "data = [one, two",
            "arr = [p, q, r",
            "vals = [red, blue",
            "tmp = [left, right",
            "out = [first, second",
        ],
        "brace": [
            "x = {a, b, c",
            "items = {foo, bar",
            "return {x, y, z",
            "data = {one, two",
            "arr = {p, q, r",
            "vals = {red, blue",
            "tmp = {left, right",
            "out = {first, second",
        ],
    },
}


V2_FRAMES = (
    ("assign", "{name} = {open}{body}"),
    ("return", "return {open}{body}"),
    ("call", "emit({name}, {open}{body}"),
    ("branch", "if ready: {name} = {open}{body}"),
    ("append", "values.append({open}{body}"),
    ("attribute", "record.{key} = {open}{body}"),
    ("log", "logger.info({open}{body}"),
    ("two_statement", "result = normalize({name}); payload = {open}{body}"),
)

V2_NAMES = (
    "alpha",
    "buffer",
    "current_value",
    "delta7",
    "entry",
    "final_result",
    "green_item",
    "header_value",
    "index2",
    "job_name",
    "known_tokens",
    "local_cache",
    "message_body",
    "next_output",
    "pending_items",
    "query_result",
)

V2_KEYS = (
    "active",
    "backup",
    "content",
    "default",
    "event",
    "field",
    "group",
    "header",
)

V2_QUOTE_BODIES = (
    "amber river",
    "blue meadow",
    "calm horizon",
    "distant signal",
    "evening report",
    "fresh result",
    "gentle current",
    "hidden pathway",
    "internal value",
    "jagged outline",
    "known answer",
    "local record",
    "moving target",
    "new message",
    "open channel",
    "pending update",
)

V2_BRACKET_BODIES = (
    "amber, river, stone",
    "blue, meadow",
    "calm, horizon, cloud",
    "delta, signal",
    "evening, report, note",
    "fresh, result",
    "green, current, path",
    "hidden, value",
    "inner, item, token",
    "jagged, outline",
    "known, answer, key",
    "local, record",
    "moving, target, point",
    "new, message",
    "open, channel, port",
    "pending, update",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def prompt_set_sha256(examples: Iterable[BehaviorExample]) -> str:
    rows = [
        {
            "example_id": example.example_id,
            "prompt": example.prompt,
            "correct_token": example.correct_token,
            "incorrect_token": example.incorrect_token,
        }
        for example in examples
    ]
    return canonical_json_sha256(rows)


def reference_program_targets(
    examples: Iterable[BehaviorExample], tokenizer, candidate_token_ids: list[int]
):
    """Return candidate indices for the explicit reference program P(x)."""

    import torch

    lookup = {token_id: index for index, token_id in enumerate(candidate_token_ids)}
    targets = []
    for example in examples:
        encoded = tokenizer.encode(example.correct_token, add_special_tokens=False)
        if len(encoded) != 1 or encoded[0] not in lookup:
            raise ValueError(
                f"Reference token {example.correct_token!r} for {example.example_id} "
                "is not one of the projected candidates"
            )
        targets.append(lookup[encoded[0]])
    return torch.tensor(targets, dtype=torch.long)


def legacy_examples(task: str, n: int = 128) -> list[BehaviorExample]:
    """Reproduce the original repeated-template domain exactly."""

    if task not in TASKS:
        raise ValueError(f"Unknown behavior task: {task}")
    if n % 2:
        raise ValueError("The legacy domain requires an even example count")
    if task == "quote_close":
        strata = (("single", "'", '"'), ("double", '"', "'"))
    else:
        strata = (("bracket", "]", "}"), ("brace", "}", "]"))
    result = []
    for stratum, correct, incorrect in strata:
        templates = LEGACY_TEMPLATES[task][stratum]
        for index in range(n // 2):
            template_index = index % len(templates)
            opener = correct if task == "quote_close" else "[" if correct == "]" else "{"
            prompt = templates[template_index]
            result.append(
                BehaviorExample(
                    prompt=prompt,
                    correct_token=correct,
                    incorrect_token=incorrect,
                    example_id=f"legacy_v1:{task}:{stratum}:{index:04d}",
                    task=task,
                    split="legacy_regression",
                    stratum=stratum,
                    template_id=f"legacy_{stratum}_{template_index}",
                    metadata={
                        "protocol": "legacy_v1",
                        "repeat_index": index // len(templates),
                        "opener": opener,
                        "opener_char_index": prompt.rfind(opener),
                    },
                )
            )
    return result


def _task_strata(task: str) -> tuple[tuple[str, str, str], ...]:
    if task == "quote_close":
        return (("single", "'", "'"), ("double", '"', '"'))
    if task == "bracket_type":
        return (("bracket", "[", "]"), ("brace", "{", "}"))
    raise ValueError(f"Unknown behavior task: {task}")


def _candidate_pool(
    task: str, seed: int, protocol_id: str = PROTOCOL_ID
) -> dict[str, list[BehaviorExample]]:
    bodies = V2_QUOTE_BODIES if task == "quote_close" else V2_BRACKET_BODIES
    pools: dict[str, list[BehaviorExample]] = {}
    for stratum, opener, closer in _task_strata(task):
        incorrect = '"' if closer == "'" else "'" if closer == '"' else "}" if closer == "]" else "]"
        rows = []
        for frame_id, frame in V2_FRAMES:
            for name in V2_NAMES:
                for body in bodies:
                    key = V2_KEYS[
                        int(_sha256_bytes(f"{seed}:{name}:{body}".encode())[:8], 16)
                        % len(V2_KEYS)
                    ]
                    prompt = frame.format(
                        name=name, key=key, open=opener, body=body
                    )
                    opener_char_index = prompt.rfind(opener)
                    row_hash = _sha256_bytes(
                        f"{seed}:{task}:{stratum}:{frame_id}:{prompt}".encode()
                    )
                    rows.append(
                        BehaviorExample(
                            prompt=prompt,
                            correct_token=closer,
                            incorrect_token=incorrect,
                            example_id=(
                                f"{protocol_id.rsplit('_', 1)[-1]}:{task}:"
                                f"{stratum}:{row_hash[:16]}"
                            ),
                            task=task,
                            stratum=stratum,
                            template_id=frame_id,
                            metadata={
                                "protocol": protocol_id,
                                "name": name,
                                "key": key,
                                "body": body,
                                "opener": opener,
                                "opener_char_index": opener_char_index,
                                "selection_hash": row_hash,
                            },
                        )
                    )
        # Some frames intentionally omit ``name`` or ``key``.  Enumerating the
        # Cartesian product can therefore produce the same prompt more than
        # once; collapse it before deterministic split selection.
        rows_by_prompt = {row.prompt: row for row in rows}
        pools[stratum] = sorted(
            rows_by_prompt.values(),
            key=lambda row: row.metadata["selection_hash"],
        )
    return pools


def candidate_pool_capacity(
    *, seed: int, protocol_id: str = PROTOCOL_ID
) -> dict[str, dict[str, int]]:
    """Return deterministic unique-prompt capacity for every task stratum."""

    return {
        task: {
            stratum: len(rows)
            for stratum, rows in _candidate_pool(task, seed, protocol_id).items()
        }
        for task in TASKS
    }


def generate_v2_splits(
    *,
    seed: int,
    split_sizes: dict[str, int],
    protocol_id: str = PROTOCOL_ID,
) -> dict[str, dict[str, list[BehaviorExample]]]:
    """Generate balanced, disjoint protocol splits without model filtering."""

    output = {split: {} for split in split_sizes}
    for task in TASKS:
        pools = _candidate_pool(task, seed, protocol_id)
        offsets = {stratum: 0 for stratum in pools}
        for split, total_size in split_sizes.items():
            if total_size % len(pools):
                raise ValueError(
                    f"{split} size {total_size} is not balanced across task strata"
                )
            per_stratum = total_size // len(pools)
            selected = []
            for stratum, rows in pools.items():
                start = offsets[stratum]
                stop = start + per_stratum
                if stop > len(rows):
                    raise ValueError(
                        f"Requested v2 splits exceed the {task}/{stratum} pool"
                    )
                selected.extend(rows[start:stop])
                offsets[stratum] = stop
            selected = sorted(selected, key=lambda row: row.metadata["selection_hash"])
            output[split][task] = [
                BehaviorExample(**{**asdict(row), "split": split})
                for row in selected
            ]
    all_prompts = [
        row.prompt
        for tasks in output.values()
        for rows in tasks.values()
        for row in rows
    ]
    if len(all_prompts) != len(set(all_prompts)):
        raise AssertionError("v2 synthesis/gate prompts must be globally unique")
    return output


def load_domain_manifest(path: str | Path) -> dict[str, Any]:
    with open(path) as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported behavior manifest schema: {path}")
    examples = manifest.get("examples")
    if not isinstance(examples, dict) or set(examples) != set(TASKS):
        raise ValueError(f"Manifest must contain exactly {TASKS}: {path}")
    manifest["loaded_examples"] = {
        task: [BehaviorExample.from_dict(row) for row in rows]
        for task, rows in examples.items()
    }
    for task, rows in manifest["loaded_examples"].items():
        expected = manifest["summary"][task]["prompt_set_sha256"]
        if prompt_set_sha256(rows) != expected:
            raise ValueError(f"Prompt digest mismatch for {task}: {path}")
    return manifest


def write_domain_manifest(
    path: str | Path,
    *,
    split: str,
    examples: dict[str, list[BehaviorExample]],
    config: dict[str, Any],
    tokenizer_metadata: dict[str, Any],
    validation: dict[str, Any],
    protocol_id: str = PROTOCOL_ID,
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": protocol_id,
        "split": split,
        "generator_config_sha256": canonical_json_sha256(config),
        "tokenizer": tokenizer_metadata,
        "selection_policy": (
            "balanced deterministic hash order; no model outputs consulted"
        ),
        "summary": {
            task: {
                "rows": len(rows),
                "unique_prompts": len({row.prompt for row in rows}),
                "prompt_set_sha256": prompt_set_sha256(rows),
                "strata": {
                    stratum: sum(row.stratum == stratum for row in rows)
                    for stratum in sorted({row.stratum for row in rows})
                },
                "opener_token_positions": validation[task][
                    "opener_token_positions"
                ],
            }
            for task, rows in examples.items()
        },
        "validation": validation,
        "examples": {
            task: [asdict(row) for row in rows] for task, rows in examples.items()
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return payload
