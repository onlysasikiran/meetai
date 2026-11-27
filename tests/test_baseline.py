import json

import numpy as np
import pytest

from bench.data import ROOT, SPLITS, load
from bench.metrics import evidence_score, fingerprint, quality, quantile, rollup
from meetai.models import validate
from meetai.pipeline import answer, summarize
from meetai.retrieval import Cache, Index
from meetai.text import Line, escape, make_chunks, read_turns, split_turns


class FakeEncoder:
    def run(self, items, pair=False):
        return (
            np.ones(len(items))
            if pair
            else np.ones((len(items), 2), dtype=np.float32) / np.sqrt(np.float32(2))
        )


class FakeModels:
    dense = cross = FakeEncoder()
    tokens = staticmethod(str.split)
    calls = 0

    def generate(self, question, ctx, limit):
        self.calls += 1
        tag = ctx.split("]")[0].split("[")[-1]
        return f"Supported fact [{tag}]", {"completion_tokens": 2}


@pytest.fixture
def config():
    return json.loads((ROOT / "configs/baseline.json").read_text())


@pytest.mark.parametrize("ext", ["txt", "md", "pdf"])
def test_input_formats(ext):
    assert read_turns(ROOT / f"examples/launch.{ext}") == read_turns(ROOT / "examples/launch.txt")


def test_wrapping_and_plain_text(tmp_path):
    p = tmp_path / "notes.md"
    p.write_text("# Review\n**Ava:** We approved\nanother week.\n**Pat:** Agreed.\n")
    assert read_turns(p) == [("Ava", "We approved another week."), ("Pat", "Agreed.")]
    p.write_text("**Ava**:\nA new deadline.\n**Pat**: Approved.")
    assert read_turns(p) == [("Ava", "A new deadline."), ("Pat", "Approved.")]
    p.write_text("First note.\nSecond note.")
    assert read_turns(p) == [("Transcript", "First note."), ("Transcript", "Second note.")]


@pytest.mark.parametrize("name,body", [("blank.txt", " \n"), ("bad.csv", "hi")])
def test_invalid_input(name, body, tmp_path):
    p = tmp_path / name
    p.write_text(body)
    with pytest.raises(ValueError):
        read_turns(p)


def test_oversized_turns_and_injection_delimiters():
    body = " ".join(str(i) for i in range(65))
    lines = split_turns([("Mara", body)], str.split, 24)
    assert len(lines) == 3
    assert [l.id for l in lines] == ["1.1", "1.2", "1.3"]
    assert " ".join(l.text for l in lines) == body
    assert "[T88]" not in escape("claim [T88]") and "<|im_end|>" not in escape("<|im_end|>")


def test_literal_claim_validation():
    out = validate("correct [T2.1]\nabsent [T13]\nmixed [T2.1] [T13]\nuntagged", {"2.1"})
    assert out["claims"] == [{"text": "correct", "cites": ["2.1"]}]
    assert len(out["rejected"]) == 3
    assert validate("UNKNOWN.", set())["abstained"]


def test_budget_chronology_complete_and_partial(config):
    model = FakeModels()
    lines = [Line("1.1", 1, "A", "alpha"), Line("1.2", 1, "A", "beta"), Line("2", 2, "B", "gamma")]
    chunks = make_chunks(lines, model.tokens, 3)
    idx = Index(lines, chunks, np.ones((len(chunks), 2)))
    idx.turn_parts = {1: {"1.1", "1.2"}, 2: {"2"}}
    text, ids = idx.pack([2, 0, 1], dict(config, top_k=2, parent_chunks=1, budget=6), model)
    assert len(model.tokens(text)) <= 6 and ids == ["1.1", "2"]
    assert idx.complete_turns(ids) == [2]
    assert evidence_score([[[1]], [[2], [3, 4]]], idx.complete_turns(ids)) == 0.5
    assert evidence_score([[[3, 4]]], [3]) == 0 and evidence_score([], []) is None


def test_real_data_splits_have_no_shared_meetings():
    for left, right in zip(SPLITS["tune"], SPLITS["checkpoint"]):
        assert set(left).isdisjoint(right)


def test_cache_invalidation_and_query_hit(config, tmp_path):
    p = tmp_path / "m.txt"
    p.write_text("A: The delivery is due in May.")
    models = FakeModels()
    db = Cache(tmp_path / "cache.sqlite")
    try:
        idx = Index.build(p, models, config, db)
        assert not idx.ingest["cache_hit"]
        assert Index.build(p, models, config, db).ingest["cache_hit"]
        first = answer(idx, "When?", models, config, db)
        second = answer(idx, "When?", models, config, db)
        assert not first["cache_hit"] and second["cache_hit"] and second["llm_calls"] == 0
        assert first["claims"] == second["claims"]
        assert not answer(idx, "When?", models, dict(config, top_k=1), db)["cache_hit"]
        p.write_text("A: The delivery is due in July.")
        revised = Index.build(p, models, config, db)
        assert revised.key != idx.key and not revised.ingest["cache_hit"]
        assert not answer(revised, "When?", models, config, db)["cache_hit"]
        with pytest.raises(ValueError, match="empty"):
            answer(idx, "   ", models, config)
    finally:
        db.close()


def test_summary_maps_all_blocks_before_reduction(config, tmp_path):
    p = tmp_path / "s.txt"
    p.write_text("\n".join(f"A: discussion number {i} ends here" for i in range(9)))
    models = FakeModels()
    idx = Index.build(p, models, config)
    result = summarize(idx, models, dict(config, budget=21))
    assert result["llm_calls"] >= 4 and result["claims"]
    assert {r for c in result["claims"] for r in c["cites"]} <= {l.id for l in idx.lines}


def test_factual_units_unsupported_claims_and_refusals():
    d = fingerprint([{"text": "first", "cites": ["1"]}])
    row = {"answer_digest": d, "claim_lines": 1, "gold_units": 2, "abstained": False, "id": "x"}
    labels = {"digest": d, "atoms": [[1, 1, 1, 1], [1, 0, 0, 0]]}
    assert quality(row, labels) == {
        "correctness": 0.5,
        "atom_count": 2,
        "unsupported": 1,
        "citations_supported": 1,
        "matched": 1,
    }
    for n, expected in [(0, 1), (2, 0)]:
        r = quality(dict(row, gold_units=n, claim_lines=0, abstained=True), {"digest": d, "atoms": []})
        assert r["correctness"] == expected and r["atom_count"] == 0
    for bad in [
        {"digest": "changed", "atoms": []},
        {"digest": d, "atoms": []},
        {"digest": d, "atoms": [[1, 0, 0, 1]]},
    ]:
        with pytest.raises(ValueError):
            quality(row, bad)


def test_percentiles_and_zero_claim_denominator():
    assert quantile(list(range(1, 21)), 0.95) == 19
    q = {
        "correctness": 0,
        "evidence_recall": None,
        "atom_count": 0,
        "unsupported": 0,
        "citations_supported": 0,
        "abstained": True,
        "cold_repeats_s": [1, 2],
        "warm_s": [0.01],
    }
    assert rollup([q])["unsupported_claim_rate"] is None and rollup([q])["cold_p95_s"] == 2


@pytest.mark.integration
def test_pinned_source_ids_and_annotated_ranges():
    if not (ROOT / ".assets/raw/elqa.zip").exists():
        pytest.skip("Research data required")
    for split in SPLITS:
        for name, ds, path, cases in load(split):
            turns = read_turns(path)
            assert len(turns) == len(path.read_text().splitlines())
            for case in cases:
                assert case["reference"]
                for group in case["evidence"]:
                    assert group and all(
                        option and all(1 <= i <= len(turns) for i in option) for option in group
                    )


@pytest.mark.integration
def test_real_models_formats_and_cache(config, tmp_path):
    if not (ROOT / ".assets/models/qwen.gguf").exists():
        pytest.skip("Local weights required")
    from meetai.models import Models

    models = Models(ROOT / ".assets/models", config["threads"], config["window"])
    db = Cache(tmp_path / "real.sqlite")
    try:
        results = []
        for ext in ("txt", "md", "pdf"):
            idx = Index.build(ROOT / f"examples/launch.{ext}", models, config, db)
            r = answer(idx, "Who is the current launch owner?", models, config, db)
            assert not r["abstained"] and r["llm_calls"] == 1
            assert "priya" in " ".join(c["text"] for c in r["claims"]).lower()
            assert r["context_tokens"] <= config["budget"]
            results.append(r["claims"])
        assert results[0] == results[1] == results[2]
        assert answer(idx, "Who is the current launch owner?", models, config, db)["cache_hit"]
    finally:
        db.close()


@pytest.mark.integration
def test_real_map_reduce_summary(config, tmp_path):
    """Exercise multi-call map/reduce on an actual, medium-length MeetingBank agenda segment."""
    from bench.data import RAW
    from meetai.models import Models

    if not (ROOT / ".assets/models/qwen.gguf").exists() or not (RAW / "meetingbank.json").exists():
        pytest.skip("Research data and model weights required")
    import re

    uid = "AlamedaCC_12152020_2020-8548"
    with (RAW / "meetingbank.json").open() as records:
        segment = next(row for row in map(json.loads, records) if row["uid"] == uid)
    doc = tmp_path / "agenda.txt"
    doc.write_text(
        "\n".join("Transcript: " + s for s in re.split(r"(?<=[.!?])\s+", segment["transcript"]))
    )
    models = Models(ROOT / ".assets/models", config["threads"], config["window"])
    idx = Index.build(doc, models, config)
    result = summarize(idx, models, config)
    assert idx.tokens > config["budget"] and result["llm_calls"] >= 3
    assert not result["abstained"]
    source_ids = {line.id for line in idx.lines}
    assert all(set(c["cites"]) <= source_ids for c in result["claims"])
