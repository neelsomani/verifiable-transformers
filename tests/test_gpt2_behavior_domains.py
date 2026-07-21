import json

from scripts.gpt2.behavior_domains import (
    generate_v2_splits,
    legacy_examples,
    load_domain_manifest,
    prompt_set_sha256,
    write_domain_manifest,
)


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
