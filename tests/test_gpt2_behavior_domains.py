import json
from pathlib import Path

from scripts.gpt2.behavior_domains import (
    candidate_pool_capacity,
    generate_v2_splits,
    legacy_examples,
    load_domain_manifest,
    prompt_set_sha256,
    write_domain_manifest,
)
from scripts.gpt2.build_bounded_behavior_domain import build


TASKS = ("quote_close", "bracket_type")


def test_v2_splits_are_unique_balanced_disjoint_and_stable():
    splits = generate_v2_splits(
        seed=1337, split_sizes={"synthesis": 256, "gate": 256}
    )
    expected_hashes = {
        ("synthesis", "quote_close"): (
            "6e361c9d49518ded905fa94f27661342b03b7532791f165b79f5f2754401e4b8"
        ),
        ("synthesis", "bracket_type"): (
            "d75263467fc9fcd4951763af37a89faab2130df290c35290c9c79c7feadff3e0"
        ),
        ("gate", "quote_close"): (
            "2672b5af812f70b101dcdf3be1c31a7e8c0b436f57a3b7d49a18d6cc4471c1ff"
        ),
        ("gate", "bracket_type"): (
            "85a26d808c3da14e5704e63fe1f5232ee30e6e69bf2d84eda63b21c75d062240"
        ),
    }
    all_synthesis = set()
    all_gate = set()
    for split, destination in (
        ("synthesis", all_synthesis),
        ("gate", all_gate),
    ):
        for task in TASKS:
            rows = splits[split][task]
            prompts = {row.prompt for row in rows}
            assert len(rows) == len(prompts) == 256
            assert {row.split for row in rows} == {split}
            if task == "quote_close":
                assert all(
                    row.correct_token == row.metadata["opener"]
                    for row in rows
                )
                assert all(
                    row.incorrect_token != row.metadata["opener"]
                    for row in rows
                )
            assert sorted(
                sum(row.stratum == stratum for row in rows)
                for stratum in {row.stratum for row in rows}
            ) == [128, 128]
            assert prompt_set_sha256(rows) == expected_hashes[(split, task)]
            destination.update(prompts)
    assert all_synthesis.isdisjoint(all_gate)


def test_v3_promotes_all_v2_rows_and_locks_the_next_fresh_gate():
    v2 = generate_v2_splits(
        seed=1337, split_sizes={"synthesis": 256, "gate": 256}
    )
    v3 = generate_v2_splits(
        seed=1337,
        split_sizes={"development": 512, "gate": 256},
        protocol_id="gpt2_behavior_domain_v3",
    )
    expected_hashes = {
        ("development", "quote_close"): (
            "3cf737914da1c4cf2e09b866a0ac1f03b15f626c48a9c4cb2407a660c967da2e"
        ),
        ("development", "bracket_type"): (
            "24666321904fca7ade6ffc9fc080f4b5405d9237eac2c00ecf829dd81c8c33c4"
        ),
        ("gate", "quote_close"): (
            "b66912f5382381406a60a0873074c55f0f49f3a5f5a652f10757d324aff6eba3"
        ),
        ("gate", "bracket_type"): (
            "68f4384a90e7c0e406729eeb02f10970d8e6e950b148ffdbc6ba29e594e0fd17"
        ),
    }
    for task in TASKS:
        burned = {
            row.prompt
            for split in ("synthesis", "gate")
            for row in v2[split][task]
        }
        development = {row.prompt for row in v3["development"][task]}
        fresh_gate = {row.prompt for row in v3["gate"][task]}
        assert len(development) == 512
        assert development == burned
        assert len(fresh_gate) == 256
        assert fresh_gate.isdisjoint(burned)
        for split in ("development", "gate"):
            assert prompt_set_sha256(v3[split][task]) == expected_hashes[(split, task)]


def test_v4_promotes_all_v3_rows_and_has_balanced_final_gate_capacity():
    v3 = generate_v2_splits(
        seed=1337,
        split_sizes={"development": 512, "gate": 256},
        protocol_id="gpt2_behavior_domain_v3",
    )
    v4 = generate_v2_splits(
        seed=1337,
        split_sizes={"development": 768, "gate": 512},
        protocol_id="gpt2_behavior_domain_v4",
    )
    capacity = candidate_pool_capacity(
        seed=1337, protocol_id="gpt2_behavior_domain_v4"
    )
    assert capacity == {
        "quote_close": {"single": 1184, "double": 1184},
        "bracket_type": {"bracket": 1187, "brace": 1187},
    }
    expected_hashes = {
        ("development", "quote_close"): (
            "67e3d82e4c4bfaa15075e634c30432049e115b6af84cce3f19fe51d70c2f3f76"
        ),
        ("development", "bracket_type"): (
            "22b090e4bdb2094fd739fd25111bed6129ff6310961468915cc9bba0b07a023d"
        ),
        ("gate", "quote_close"): (
            "de3f65cd2496a2735d12e735a5b9fd26d2afc163048aeb4eb9ae6d4811b930ec"
        ),
        ("gate", "bracket_type"): (
            "f6b710cd3f468781884b7e78d58f2cfd618923cf8f44952df5db0feb27749e74"
        ),
    }
    for task in TASKS:
        burned = {
            row.prompt
            for split in ("development", "gate")
            for row in v3[split][task]
        }
        development = {row.prompt for row in v4["development"][task]}
        fresh_gate = {row.prompt for row in v4["gate"][task]}
        assert development == burned
        assert fresh_gate.isdisjoint(burned)
        assert len(development) == 768
        assert len(fresh_gate) == 512
        assert sorted(
            sum(row.stratum == stratum for row in v4["gate"][task])
            for stratum in {row.stratum for row in v4["gate"][task]}
        ) == [256, 256]
        for split in ("development", "gate"):
            assert prompt_set_sha256(v4[split][task]) == expected_hashes[(split, task)]


def test_bounded_v1_is_the_exact_frozen_union_without_a_gate(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "bounded" / "domain.json"
    index = build(
        root / "configs/gpt2_behavior_domain_bounded_v1.json",
        output,
        root,
    )
    bounded = load_domain_manifest(output)
    development = load_domain_manifest(
        root / "artifacts/gpt2-behavior-domains-v4/development.json"
    )
    gate = load_domain_manifest(
        root / "artifacts/gpt2-behavior-domains-v4/gate.json"
    )
    assert bounded["protocol_id"] == "gpt2_behavior_domain_bounded_v1"
    assert bounded["split"] == "bounded"
    assert bounded["held_out_gate"] is False
    assert index["held_out_gate"] is False
    expected_hashes = {
        "quote_close": (
            "3d590ce66edc1e83e054735523674c1d7c77af4297fd230b2a17d3209917bd48"
        ),
        "bracket_type": (
            "0ae81b74475f8b988325299387da78732421901f15f7b9845031fc805dea9afe"
        ),
    }
    for task in TASKS:
        bounded_rows = bounded["loaded_examples"][task]
        source_rows = (
            development["loaded_examples"][task]
            + gate["loaded_examples"][task]
        )
        assert [row.example_id for row in bounded_rows] == [
            row.example_id for row in source_rows
        ]
        assert len({row.prompt for row in bounded_rows}) == 1280
        assert prompt_set_sha256(bounded_rows) == expected_hashes[task]


def test_legacy_v1_128_rows_are_sixteen_prompts_repeated_eight_times():
    for task in TASKS:
        rows = legacy_examples(task, 128)
        counts = {}
        for row in rows:
            counts[row.prompt] = counts.get(row.prompt, 0) + 1
        assert len(rows) == 128
        assert len(counts) == 16
        assert set(counts.values()) == {8}


def test_manifest_round_trip_checks_prompt_digest(tmp_path):
    tasks = {
        task: legacy_examples(task, 16)
        for task in TASKS
    }
    validation = {
        task: {"opener_token_positions": [1, 2]}
        for task in TASKS
    }
    path = tmp_path / "domain.json"
    write_domain_manifest(
        path,
        split="legacy_regression",
        examples=tasks,
        config={"seed": 1337},
        tokenizer_metadata={"vocab_sha256": "test"},
        validation=validation,
    )
    loaded = load_domain_manifest(path)
    assert len(loaded["loaded_examples"]["quote_close"]) == 16

    payload = json.loads(path.read_text())
    payload["examples"]["quote_close"][0]["prompt"] += " changed"
    path.write_text(json.dumps(payload))
    try:
        load_domain_manifest(path)
    except ValueError as error:
        assert "Prompt digest mismatch" in str(error)
    else:
        raise AssertionError("tampered domain manifest was accepted")
