"""Adapters for the iterative, Hayes-style LM program-proposal loop."""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from typing import Iterable, Sequence


def _json_payloads(text: str) -> list[dict]:
    """Extract program dictionaries from a command's JSON/JSONL response."""
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1]).strip()

    values = []
    try:
        values.append(json.loads(stripped))
    except json.JSONDecodeError:
        for line in stripped.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    programs: list[dict] = []
    for value in values:
        if isinstance(value, list):
            programs.extend(item for item in value if isinstance(item, dict))
        elif isinstance(value, dict) and isinstance(value.get("programs"), list):
            programs.extend(
                item for item in value["programs"] if isinstance(item, dict)
            )
        elif isinstance(value, dict):
            programs.append(value)
    return programs


@dataclass(frozen=True)
class CommandProgramProposer:
    """Call an arbitrary LM CLI with the restricted prompt on standard input.

    The command must print either one program JSON object, a JSON list of
    programs, ``{"programs": [...]}``, or JSONL. The synthesis harness still
    validates every proposal with ``AttentionProgram.from_dict``; this adapter
    never executes model-generated code.
    """

    command: tuple[str, ...]
    timeout_seconds: float = 120.0

    def __init__(
        self,
        command: Sequence[str] | str,
        timeout_seconds: float = 120.0,
    ) -> None:
        words = shlex.split(command) if isinstance(command, str) else list(command)
        if not words:
            raise ValueError("LM proposer command must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("LM proposer timeout must be positive")
        object.__setattr__(self, "command", tuple(words))
        object.__setattr__(self, "timeout_seconds", float(timeout_seconds))

    def __call__(self, prompt: str) -> Iterable[dict]:
        completed = subprocess.run(
            self.command,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise RuntimeError(
                f"LM proposer exited with {completed.returncode}: {stderr}"
            )
        return _json_payloads(completed.stdout)

    def provenance(self) -> dict:
        return {
            "adapter": "command_stdin_json_stdout",
            "command": list(self.command),
            "timeout_seconds": self.timeout_seconds,
        }
