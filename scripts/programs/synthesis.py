"""IoU-ranked synthesis with exact-map or projected-behavior acceptance."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
from typing import Callable, Iterable, Optional, Sequence

import torch

from .dsl import AttentionProgram, Condition, Rule, program_from_conditions


ProjectedEvaluator = Callable[[AttentionProgram], float]
ProgramProposer = Callable[[str], Iterable[dict]]


@dataclass
class CandidateScore:
    program: AttentionProgram
    support_iou: float
    mean_absolute_error: float
    exact_attention_agreement: bool
    projected_agreement: Optional[float]


@dataclass
class SynthesisResult:
    program: AttentionProgram
    accepted: bool
    acceptance_reason: str
    score: CandidateScore
    candidates_evaluated: int
    projected_candidates_evaluated: int = 0

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "acceptance_reason": self.acceptance_reason,
            "candidates_evaluated": self.candidates_evaluated,
            "projected_candidates_evaluated": self.projected_candidates_evaluated,
            "support_iou": self.score.support_iou,
            "mean_absolute_error": self.score.mean_absolute_error,
            "exact_attention_agreement": self.score.exact_attention_agreement,
            "projected_agreement": self.score.projected_agreement,
            "program": self.program.to_dict(),
        }


def support_iou(predicted: torch.Tensor, target: torch.Tensor, epsilon: float) -> float:
    predicted_support = predicted > epsilon
    target_support = target > epsilon
    union = predicted_support | target_support
    intersection = predicted_support & target_support
    per_row = torch.where(
        union.sum(dim=-1) == 0,
        torch.ones_like(union.sum(dim=-1), dtype=torch.float32),
        intersection.sum(dim=-1).float() / union.sum(dim=-1).float(),
    )
    return per_row.mean().item()


def restricted_synthesis_prompt(feedback: str = "") -> str:
    """Prompt for an optional Hayes-style proposer, restricted to this DSL."""
    return (
        "Synthesize an attention program as JSON using only DSL version 1. "
        "Allowed features: query_token, key_token, query_position, key_position, "
        "relative_distance, absolute_distance, token_match. Allowed comparisons: "
        "==, !=, <, <=, >, >=, in. Rules add nonnegative rational weights; rows "
        "are normalized by their branch-constant sum and are causal. Do not use "
        "embeddings, NLP libraries, parsers, learned features, or free-form code. "
        f"Return a JSON object accepted by AttentionProgram.from_dict. {feedback}"
    )


class SynthesisHarness:
    """Enumerate restricted programs, optionally augmented by an LM proposer."""

    def __init__(
        self,
        *,
        support_epsilon: float = 1e-7,
        exact_tolerance: float = 1e-6,
        healable_projected_agreement: float = 1.0,
        proposer: Optional[ProgramProposer] = None,
        proposer_rounds: int = 2,
        projected_candidates: int = 64,
        max_token_values: int = 32,
        max_conjunction_values: int = 12,
    ):
        self.support_epsilon = support_epsilon
        self.exact_tolerance = exact_tolerance
        self.healable_projected_agreement = healable_projected_agreement
        self.proposer = proposer
        self.proposer_rounds = proposer_rounds
        self.projected_candidates = projected_candidates
        self.max_token_values = max_token_values
        self.max_conjunction_values = max_conjunction_values

    def _base_candidates(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> list[AttentionProgram]:
        _, length = input_ids.shape
        if attention_mask is None:
            token_values = input_ids.detach().cpu().flatten().tolist()
        else:
            token_values = input_ids[attention_mask.bool()].detach().cpu().tolist()
        counts = {}
        for token in token_values:
            counts[int(token)] = counts.get(int(token), 0) + 1
        tokens = [
            token
            for token, _ in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )[: self.max_token_values]
        ]
        candidates = [AttentionProgram((), default_weight=1, name="uniform_causal")]

        single_conditions: list[tuple[str, tuple[Condition, ...]]] = [
            ("token_match", (Condition("token_match", "==", True),)),
        ]
        for position in range(length):
            single_conditions.append(
                (f"key_position_{position}", (Condition("key_position", "==", position),))
            )
        for distance in range(length):
            single_conditions.extend(
                [
                    (f"relative_{distance}", (Condition("relative_distance", "==", distance),)),
                    (f"distance_le_{distance}", (Condition("relative_distance", "<=", distance),)),
                ]
            )
        for token in tokens:
            single_conditions.append(
                (f"key_token_{token}", (Condition("key_token", "==", int(token)),))
            )

        for name, conditions in single_conditions:
            candidates.append(program_from_conditions(conditions, name=name))

        # Conjunctions cover task-conditioned routing (for example: a query
        # task token selecting a particular key token) without adding a richer
        # language primitive.
        conjunction_tokens = tokens[: self.max_conjunction_values]
        for query_token in conjunction_tokens:
            for key_token in conjunction_tokens:
                conditions = (
                    Condition("query_token", "==", int(query_token)),
                    Condition("key_token", "==", int(key_token)),
                )
                candidates.append(
                    program_from_conditions(
                        conditions,
                        name=f"query_{query_token}_key_{key_token}",
                    )
                )
        return candidates

    def _score(
        self,
        program: AttentionProgram,
        input_ids: torch.Tensor,
        target: torch.Tensor,
        projected_evaluator: Optional[ProjectedEvaluator],
        attention_mask: Optional[torch.Tensor],
    ) -> CandidateScore:
        predicted = program.weights(input_ids, dtype=target.dtype)
        if attention_mask is None:
            scored_prediction = predicted
            scored_target = target
            mae = (predicted - target).abs().mean().item()
            exact = torch.allclose(
                predicted,
                target,
                atol=self.exact_tolerance,
                rtol=0.0,
            )
        else:
            valid = attention_mask.bool()
            valid_entries = valid.unsqueeze(-1) & valid.unsqueeze(1)
            scored_prediction = torch.where(
                valid_entries, predicted, torch.zeros_like(predicted)
            )
            scored_target = torch.where(
                valid_entries, target, torch.zeros_like(target)
            )
            differences = (predicted - target).abs()[valid_entries]
            mae = differences.mean().item() if differences.numel() else 0.0
            exact = bool(
                (differences <= self.exact_tolerance).all().item()
            )
        iou = support_iou(
            scored_prediction, scored_target, self.support_epsilon
        )
        projected = projected_evaluator(program) if projected_evaluator else None
        return CandidateScore(program, iou, mae, exact, projected)

    @staticmethod
    def _rank_key(score: CandidateScore) -> tuple[float, float, float]:
        projected = -1.0 if score.projected_agreement is None else score.projected_agreement
        return score.support_iou, projected, -score.mean_absolute_error

    def synthesize(
        self,
        input_ids: torch.Tensor,
        target_weights: torch.Tensor,
        *,
        projected_evaluator: Optional[ProjectedEvaluator] = None,
        attention_mask: Optional[torch.Tensor] = None,
        extra_candidates: Iterable[AttentionProgram] = (),
    ) -> SynthesisResult:
        if target_weights.shape != (*input_ids.shape, input_ids.shape[1]):
            raise ValueError(
                "target_weights must have shape [batch, query_position, key_position]"
            )

        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")
        candidates = self._base_candidates(input_ids, attention_mask)
        candidates.extend(extra_candidates)
        scored = [
            self._score(
                candidate,
                input_ids,
                target_weights,
                None,
                attention_mask,
            )
            for candidate in candidates
        ]

        # Additive pairs are generated from the strongest simple supports. This
        # is still a finite restricted search; IoU is only a ranking metric.
        strongest = sorted(scored, key=self._rank_key, reverse=True)[:8]
        for left, right in combinations(strongest, 2):
            combined = AttentionProgram(
                rules=left.program.rules + right.program.rules,
                default_weight=left.program.default_weight + right.program.default_weight,
                name=f"{left.program.name}+{right.program.name}",
            )
            scored.append(
                self._score(
                    combined,
                    input_ids,
                    target_weights,
                    None,
                    attention_mask,
                )
            )

        if self.proposer is not None:
            feedback = ""
            for _ in range(self.proposer_rounds):
                prompt = restricted_synthesis_prompt(feedback)
                for raw_program in self.proposer(prompt):
                    try:
                        program = AttentionProgram.from_dict(raw_program)
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                        continue
                    scored.append(
                        self._score(
                            program,
                            input_ids,
                            target_weights,
                            None,
                            attention_mask,
                        )
                    )
                best = max(scored, key=self._rank_key)
                feedback = (
                    f"Best support IoU={best.support_iou:.6f}, "
                    f"MAE={best.mean_absolute_error:.6g}. Improve it."
                )

        projected_count = 0
        if projected_evaluator is not None:
            # IoU ranks the finite search. Only the strongest candidates pay
            # for a model forward on the projected behavior domain.
            ranked_indices = sorted(
                range(len(scored)),
                key=lambda index: self._rank_key(scored[index]),
                reverse=True,
            )[: self.projected_candidates]
            for index in ranked_indices:
                scored[index] = self._score(
                    scored[index].program,
                    input_ids,
                    target_weights,
                    projected_evaluator,
                    attention_mask,
                )
                projected_count += 1
                if (
                    scored[index].projected_agreement is not None
                    and scored[index].projected_agreement
                    >= self.healable_projected_agreement
                ):
                    # Candidates are in descending IoU order, so the first
                    # accepted one is already the best-ranked valid program.
                    break
            accepted_scores = [
                score
                for score in scored
                if score.projected_agreement is not None
                and score.projected_agreement >= self.healable_projected_agreement
            ]
            accepted = bool(accepted_scores)
            # IoU ranks the search, but cannot veto a program that satisfies the
            # preregistered behavioral acceptance metric.
            best = max(accepted_scores or scored, key=self._rank_key)
            reason = (
                "exact projected agreement"
                if best.projected_agreement == 1.0
                else "within predeclared healing threshold"
                if accepted
                else "projected agreement below threshold"
            )
        else:
            best = max(scored, key=self._rank_key)
            accepted = best.exact_attention_agreement
            reason = "exact attention-map agreement" if accepted else "best IoU only"

        return SynthesisResult(
            program=best.program,
            accepted=accepted,
            acceptance_reason=reason,
            score=best,
            candidates_evaluated=len(scored),
            projected_candidates_evaluated=projected_count,
        )
