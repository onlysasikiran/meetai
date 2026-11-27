import hashlib
import json

import numpy as np
import pytest

from bench.data import ROOT
from meetai.grounding import validated_answer, verify_claims
from meetai.linked_index import LinkedIndex, names
from meetai.linked_retrieval import pack_rows, retrieve
from meetai.minutes import extract_records
from meetai.pipeline import answer, summarize
from meetai.retrieval import Cache, Index
from meetai.text import CITE


class Encoder:
    def run(self, items, pair=False):
        if pair:
            return np.array(
                [len(set(q.lower().split()) & set(p.lower().split())) for q, p in items], dtype=float
            )
        vectors = np.zeros((len(items), 12), dtype=np.float32)
        for i, text in enumerate(items):
            for word in text.lower().split():
                vectors[
                    i,
                    int.from_bytes(hashlib.blake2s(word.encode(), digest_size=2).digest(), "big") % 12,
                ] += 1
        return vectors / np.maximum(1, np.linalg.norm(vectors, axis=1, keepdims=True))


class ModelsDouble:
    dense = cross = Encoder()
    tokens = staticmethod(str.split)

    def support_scores(self, pairs):
        return [(0.01, 0.98) if "invented" in h.lower() else (0.99, 0.01) for _, h in pairs]

    def generate(self, q, text, limit):
        first = text.splitlines()[0]
        ref = CITE.match(first)[1]
        return f"{first.split(': ', 1)[1]} [T{ref}]", {}


@pytest.fixture
def cfg():
    return json.loads((ROOT / "configs/meetai.json").read_text())


def build(path, body, cfg, models=None, cache=None):
    path.write_text(body)
    return Index.build(path, models or ModelsDouble(), cfg, cache)


def test_graph_cache_identity_layout_and_degree(tmp_path, cfg):
    db = Cache(tmp_path / "graphs.sqlite")
    try:
        body = "\n".join(f"Lia: Batch {i} has a distinct ledger entry." for i in range(22))
        p = tmp_path / "notes.txt"
        config = dict(cfg, chunk_tokens=8, block_min=2, block_max=3)
        idx = build(p, body, config, cache=db)
        assert isinstance(idx, LinkedIndex)
        assert [c for lo, hi in idx.blocks for c in range(lo, hi)] == list(range(len(idx.chunks)))
        assert all(1 <= hi - lo <= 3 for lo, hi in idx.blocks)
        assert all(
            len(links) <= config["link_degree"] and all(abs(i - j) > 1 for j, _ in links)
            for i, links in enumerate(idx.links)
        )
        assert any(idx.links)
        again = Index.build(p, ModelsDouble(), config, db)
        assert (
            again.ingest["graph_cache_hit"] and idx.links == again.links and idx.blocks == again.blocks
        )
        changed = Index.build(p, ModelsDouble(), dict(config, link_degree=1), db)
        assert (
            changed.ingest["cache_hit"]
            and not changed.ingest["graph_cache_hit"]
            and idx.key != changed.key
        )
        p.write_text(body + "\nLia: The last estimate was rejected.")
        fresh = Index.build(p, ModelsDouble(), config, db)
        assert fresh.key != idx.key and not fresh.ingest["graph_cache_hit"]
    finally:
        db.close()


def test_embedding_link_without_shared_words():
    from meetai.text import Line

    lines = [
        Line(str(i + 1), i + 1, "S", t) for i, t in enumerate(["cats", "wheels", "soil", "felines"])
    ]
    idx = LinkedIndex(
        lines, [[0], [1], [2], [3]], np.array([[1, 0], [0, 1], [-1, 0], [1, 0]], dtype=np.float32)
    )
    result = idx._organize(
        dict(block_min=1, block_max=2, topic_boundary=0.7, link_degree=2, semantic_floor=0.8)
    )
    assert [3, "semantic"] in result["links"][0]


def test_simple_vs_expanded_and_source_order(tmp_path, cfg):
    m = ModelsDouble()
    idx = build(
        tmp_path / "notes.txt",
        "Ria: The lamp is blue.\nPat: The original launch owner was Dev.\nPat: Priya is now the launch owner.",
        cfg,
        m,
    )
    _, ids, _, trace = retrieve(idx, "What is blue?", m, cfg)
    assert not trace["expanded"] and trace["budget"] == cfg["simple_budget"]
    context, ids, _, trace = retrieve(idx, "Who is the current launch owner?", m, cfg)
    assert (
        trace["expanded"]
        and trace["route"] == "current"
        and ids == sorted(ids, key=idx.by_id.__getitem__)
    )
    assert len(m.tokens(context)) <= cfg["budget"]
    assert CITE.findall(context) == ids


def test_alias_links_and_similar_project(tmp_path, cfg):
    idx = build(
        tmp_path / "names.txt",
        "Al: Orion is also known as Linen.\nBo: Linen must clear the Iris gate.\nCy: Orion West must clear the Granite gate.",
        cfg,
    )
    assert names("Orion West is separate.") == {"orion west"}
    assert idx.focus("What does orion west require?") == {"orion west"}
    assert idx.focus("Compare Orion and Orion West.") == {"orion", "orion west"}
    assert idx.bridges({"orion"}) == [("linen", 0)]
    trace = {"focus": ["orion"], "route": "fact"}

    def check(ref, ids):
        return verify_claims(
            [{"text": "Iris", "cites": [ref]}],
            ids,
            idx,
            ModelsDouble(),
            cfg,
            "Which gate must Orion clear?",
            trace,
        )

    assert check("2", ["1", "2", "3"])[0][0]["cites"] == ["1", "2"]
    assert check("2", ["2", "3"])[1][0]["reason"] == "missing_subject_evidence"
    assert check("3", ["1", "2", "3"])[1][0]["reason"] == "different_named_subject"


def test_citation_rejection_and_stale_date(tmp_path, cfg):
    idx = build(
        tmp_path / "notes.txt",
        "Ria: We moved the Luna deadline from May 8 to May 24.\nBo: The room is ready.",
        cfg,
    )
    trace = {"focus": ["luna"], "route": "current", "facets": ["deadline"]}
    for text, reason in (
        ("The current deadline is May 8", "superseded_date"),
        ("June 30", "number_missing_from_citations"),
        ("Invented detail", "insufficient_citation_support"),
    ):
        ok, errors = verify_claims(
            [{"text": text, "cites": ["1"]}],
            ["1"],
            idx,
            ModelsDouble(),
            cfg,
            "What is the current Luna deadline?",
            trace,
        )
        assert not ok and errors[0]["reason"] == reason
    for raw in ("Unknown", "The answer [T202]", "The answer [T1] [T202]", "No reference"):
        assert validated_answer(raw, ["1"], idx, ModelsDouble(), cfg)["abstained"]
    parts = validated_answer(
        "We moved the deadline [T1], and the room is ready [T2].", ["1", "2"], idx, ModelsDouble(), cfg
    )
    assert [c["cites"] for c in parts["claims"]] == [["1"], ["2"]]


def test_packing_keeps_fragment_identity_and_deduplicates(tmp_path, cfg):
    body = " ".join(f"word{i}" for i in range(69))
    idx = build(
        tmp_path / "parts.txt",
        "Al: " + body + "\nJo: Identical remark.\nJo: Identical remark.",
        dict(cfg, chunk_tokens=28),
    )
    rows = pack_rows(idx, list(reversed(range(len(idx.lines)))), ModelsDouble(), 34)
    assert (
        rows == sorted(set(rows))
        and len([i for i in rows if idx.lines[i].text == "Identical remark."]) == 1
    )
    assert 1 not in idx.complete_turns([idx.lines[i].id for i in rows])
    assert any("." in idx.lines[i].id for i in rows)


def test_structured_fields_fail_closed(tmp_path, cfg):
    idx = build(tmp_path / "record.txt", "Eve: Ana will write the checklist.", cfg)
    item = {
        "kind": "commitment",
        "text": "Ana will write the checklist.",
        "cites": ["T1"],
        "owner": "Ana",
        "deadline": "June 9",
    }
    run = lambda raw: extract_records(raw, ["1"], idx, ModelsDouble(), cfg)
    records, errors = run(json.dumps({"items": [item]}))
    assert records[0]["owner"] == {"value": "Ana", "cites": ["1"]} and records[0]["deadline"] is None
    assert errors[0]["reason"] == "field_absent_from_citations"
    assert not run(json.dumps({"items": [dict(item, cites=["6"])]}))[0]
    for raw in ('{"items":[', json.dumps({"items": [dict(item, kind="guess")]})):
        assert run(raw) == ([], [{"reason": "invalid_or_incomplete_json"}])


def test_summary_reduces_multiple_levels_using_original_lines(tmp_path, cfg):
    m = ModelsDouble()
    config = dict(cfg, budget=22, chunk_tokens=10, block_min=1, block_max=2, summary_lines_per_block=4)
    idx = build(
        tmp_path / "large.txt",
        "\n".join(f"Al: We approved the delivery of shipment number {i}." for i in range(18)),
        config,
        m,
    )
    summary = summarize(idx, m, config)
    assert summary["trace"]["reduction_levels"] >= 2 and summary["llm_calls"] > 3
    assert summary["claims"] and set(summary["claims"][0]["cites"]) <= set(summary["ids"])
    assert all(
        idx.lines[idx.by_id[r]].text.rstrip(".") in c["text"]
        for c in summary["claims"]
        for r in c["cites"]
    )


def test_answers_do_not_cross_systems(tmp_path, cfg):
    p = tmp_path / "answer.txt"
    db = Cache(tmp_path / "answer.sqlite")
    try:
        m = ModelsDouble()
        baseline = json.loads((ROOT / "configs/baseline.json").read_text())
        flat = build(p, "Al: The lamp is blue.", baseline, m, db)
        linked = Index.build(p, m, cfg, db)
        assert type(flat) is Index and type(linked) is LinkedIndex
        assert not answer(flat, "What is blue?", m, baseline, db)["cache_hit"]
        assert not answer(linked, "What is blue?", m, cfg, db)["cache_hit"]
        warm = answer(linked, "What is blue?", m, cfg, db)
        assert warm["cache_hit"] and warm["llm_calls"] == 0
    finally:
        db.close()


@pytest.mark.integration
def test_real_nli_rejects_mentioned_person_as_owner(cfg, tmp_path):
    from meetai.models import Models

    if not (ROOT / ".assets/models/nli/model.onnx").exists():
        pytest.skip("Local NLI weights required")
    # Only the ONNX verifier is used in this focused test.
    model = object.__new__(Models)
    model.root, model.threads = ROOT / ".assets/models", cfg["threads"]
    idx = build(
        tmp_path / "roles.txt", "Pat: Mira received the invitation.\nPat: The new owner is Leon.", cfg
    )
    good, bad = verify_claims(
        [
            {"text": "Mira is the new owner", "cites": ["1"]},
            {"text": "Leon is the new owner", "cites": ["2"]},
        ],
        ["1", "2"],
        idx,
        model,
        cfg,
    )
    assert [c["cites"] for c in good] == [["2"]] and len(bad) == 1


def test_alias_ambiguity_and_negation_do_not_create_bridge(tmp_path, cfg):
    idx = build(
        tmp_path / "aliases.txt",
        "A: Orion is also known as Linen.\nB: Lyra is also known as Linen.\nC: Vega is not also known as Jasper.",
        cfg,
    )
    assert not idx.bridges({"orion"}) and not idx.bridges({"linen"}) and not idx.bridges({"vega"})


def test_timestamped_source_keeps_speaker_and_fragment_time(tmp_path, cfg):
    idx = build(
        tmp_path / "agenda.md",
        "# Agenda\n[00:03:12] **Ana:** We approved the quote.\nwith one condition.\n00:18 Bo: "
        + " ".join(["detail"] * 45),
        dict(cfg, chunk_tokens=28),
    )
    assert idx.lines[0].speaker == "Ana" and idx.lines[0].text.endswith("with one condition.")
    assert idx.lines[0].render().startswith("[T1] (00:03:12) Ana:")
    assert [(x.id, x.timestamp) for x in idx.lines[1:]] == [("2.1", "00:18"), ("2.2", "00:18")]


def test_prefix_citations_and_repeated_headings(tmp_path, cfg):
    idx = build(tmp_path / "prefix.txt", "A: We approved the launch.\nB: I will send it.", cfg)
    raw = "Approved decisions: [T1] A: We approved the launch.\nCurrent actions: [T2] B: I will send it.\nCommitments: [T2] B: I will send it."
    claims = validated_answer(raw, ["1", "2"], idx, ModelsDouble(), cfg)["claims"]
    assert [c["cites"] for c in claims] == [["1"], ["2"]]
    assert "approved the launch" in claims[0]["text"] and "send it" in claims[1]["text"]
