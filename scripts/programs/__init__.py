"""Restricted attention programs, synthesis, and neural integration."""

from .dsl import AttentionProgram, Condition, Rule
from .module import (
    ProgramAttentionHead,
    ProgrammedAttention,
    install_program_heads,
    load_programs,
    save_programs,
)
from .synthesis import SynthesisHarness, SynthesisResult
from .proposer import CommandProgramProposer

__all__ = [
    "AttentionProgram",
    "Condition",
    "Rule",
    "ProgramAttentionHead",
    "ProgrammedAttention",
    "install_program_heads",
    "load_programs",
    "save_programs",
    "SynthesisHarness",
    "SynthesisResult",
    "CommandProgramProposer",
]
