#!/usr/bin/env python3
"""Cheap single-GPU diagnostics for the exploratory Phase Q healing objective."""

import argparse
import json
import math
from pathlib import Path

import torch

from scripts.gpt2.heal_programs import (
    behavior_domain,
    controlled_forward,
    load_circuit_edges,
    reference_program_decisions,
)
from scripts.gpt2.extract import (
    build_circuit_graph,
    load_model_with_variants,
    select_last_real_logits,
)
from scripts.programs import ProgrammedAttention, install_program_heads, load_programs


def candidate_loss(logits, attention_mask, candidates, targets):
    rows = select_last_real_logits(logits, attention_mask)[:, candidates]
    return torch.nn.functional.cross_entropy(rows, targets.to(rows.device))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--programs", required=True)
    parser.add_argument("--circuit_root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model_with_variants(args.model_path, str(device))
    model.config.use_cache = False
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    domains = behavior_domain(
        tokenizer, cfg, device, args.manifest, tasks=("quote_close",)
    )
    references = reference_program_decisions(domains)
    programs = load_programs(args.programs)
    variant = json.loads(
        Path(args.model_path, "model_info.json").read_text()
    ).get("attn_variant", "sparsemax")
    install_program_heads(model, programs, attention_variant=variant)
    model.train()

    domain = domains["quote_close"]
    n = min(args.batch_size, domain["input_ids"].size(0))
    input_ids = domain["input_ids"][:n]
    attention_mask = domain["attention_mask"][:n]
    targets = references["quote_close"][:n]
    candidates = domain["candidates"]
    graph = build_circuit_graph(model.config.n_layer, model.config.n_head)
    core = load_circuit_edges(args.circuit_root, ("quote_close",))["quote_close"]

    probe_parameter = model.transformer.h[7].attn.c_proj.weight
    losses = {}
    gradients = {}

    full_logits = model(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=False
    ).logits
    losses["full"] = candidate_loss(
        full_logits, attention_mask, candidates, targets
    )
    core_logits = controlled_forward(
        model, input_ids, attention_mask, core, graph
    )
    losses["circuit"] = candidate_loss(
        core_logits, attention_mask, candidates, targets
    )
    intended = {f"attn_{layer}_h_{head}" for layer, head in programs}
    bypass_edges = {
        edge for edge in graph.get_edges() if edge[0] not in intended
    }
    bypass_logits = controlled_forward(
        model, input_ids, attention_mask, bypass_edges, graph
    )
    losses["bypass_uniform"] = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(
            select_last_real_logits(bypass_logits, attention_mask)[:, candidates],
            dim=-1,
        ),
        torch.full(
            (n, len(candidates)),
            1.0 / len(candidates),
            device=device,
        ),
        reduction="batchmean",
    )
    for name, loss in losses.items():
        grad = torch.autograd.grad(
            loss, probe_parameter, retain_graph=True, allow_unused=True
        )[0]
        gradients[name] = (
            torch.zeros_like(probe_parameter) if grad is None else grad.detach()
        )

    def cosine(left, right):
        l = gradients[left].float().flatten()
        r = gradients[right].float().flatten()
        denom = l.norm() * r.norm()
        return None if denom == 0 else float((l @ r / denom).item())

    programmed_layers = {
        str(index): {
            "program_heads": sorted(attn.programs),
            "neural_heads": list(attn.neural_heads),
            "query_key_parameters_for_program_heads": 0,
        }
        for index, block in enumerate(model.transformer.h)
        if isinstance((attn := block.attn), ProgrammedAttention)
    }
    payload = {
        "scope": "exploratory bounded-domain diagnostic",
        "rows": n,
        "losses": {name: float(value.detach().item()) for name, value in losses.items()},
        "probe_parameter": "transformer.h.7.attn.c_proj.weight",
        "gradient_norms": {
            name: float(value.float().norm().item())
            for name, value in gradients.items()
        },
        "gradient_cosines": {
            "full_vs_circuit": cosine("full", "circuit"),
            "full_vs_bypass_uniform": cosine("full", "bypass_uniform"),
            "circuit_vs_bypass_uniform": cosine(
                "circuit", "bypass_uniform"
            ),
        },
        "programmed_layers": programmed_layers,
        "program_objects_have_parameters": any(
            isinstance(value, torch.nn.Parameter)
            for program in programs.values()
            for value in vars(program).values()
        ),
        "targets": {
            "values": sorted(set(int(value) for value in targets.cpu().tolist())),
            "candidate_count": len(candidates),
            "valid_candidate_indices": all(
                0 <= int(value) < len(candidates) for value in targets
            ),
        },
        "finite": all(
            math.isfinite(float(value.detach().item())) for value in losses.values()
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
