import json

import pytest

from bench.metrics import evidence_score
from bench.phase3_data import PROTOCOL, scenario
from bench.phase3_scale import pdf_transcript
from bench.phase3_score import packets
from bench.run import write
from meetai.text import read_turns


def test_new_rosters_have_distinct_turns_and_two_hop_evidence():
    for seed in PROTOCOL["heldout"]["MeetAI-Stress"]["seeds"]:
        turns, cases = scenario(seed, 20000)
        assert len(turns) == len(set(turns)) == 20000
        assert len(cases[2]["facts"]) == 2
        group = cases[3]["evidence"]
        a, b = [g[0][0] for g in group]
        assert b - a > 6000 and evidence_score(group, [a]) == .5 and evidence_score(group,[b,a]) == 1


def test_three_formats_preserve_punctuation_across_pdf_page_breaks(tmp_path):
    turns, _ = scenario(24911, 100)
    lines = [f"{s}: {t}" for s, t in turns]
    txt = tmp_path / "dispatch.txt"
    txt.write_text("\n".join(lines))
    md = txt.with_suffix(".md")
    md.write_text("# Dispatch\n\n" + "\n".join(f"- **{s}:** {t}" for s, t in turns))
    pdf = txt.with_suffix(".pdf")
    pdf_transcript(pdf, lines)
    assert all(read_turns(p) == turns for p in (txt, md, pdf))


def test_blind_review_deduplicates_identical_system_outputs(tmp_path):
    claims = [{"text": "Sam approved it", "cites": ["1"]}]
    citations = {"1": "Sam: Approved."}
    sample = dict(question="Who approved it?", reference="Sam", facts=["Sam"], evidence=[[[1]]],
                  claims=claims, cited=citations, unguarded=claims, unguarded_citations=citations)
    for label in ("baseline", "meetai"):
        write(tmp_path / label / "q.json", sample)
    batch = packets(tmp_path)
    assert len(batch) == 1 and batch[0]["id"] == "q"
    assert all(label not in json.dumps(batch) for label in ("baseline", "meetai", "unguarded_citations"))


def test_resume_reuses_matching_files_and_detects_corruption(tmp_path, monkeypatch):
    import bench.phase3_checkpoint as cp

    monkeypatch.setattr(cp, "BASE", tmp_path)
    monkeypatch.setattr(cp, "protocol_id", lambda: "protocol-a")
    path = tmp_path / "size-5000"
    path.mkdir()
    output = path / "baseline.json"
    output.write_text('{"measured":true}')
    write(path / "done.json", {"batch":"size-5000", "protocol":"protocol-a", "files":{output.name:cp.sha(output)}})
    monkeypatch.setattr(cp, "run_module", lambda *args: pytest.fail("A finished batch ran twice"))
    assert cp.valid("size-5000")
    cp.run_batch("size-5000")
    output.write_text('{"measured":false}')
    assert not cp.valid("size-5000")


def test_resume_rejects_changed_protocol_or_missing_file(tmp_path, monkeypatch):
    import bench.phase3_checkpoint as cp

    monkeypatch.setattr(cp,"BASE",tmp_path)
    monkeypatch.setattr(cp,"protocol_id",lambda:"new")
    folder=tmp_path/"probe"
    write(folder/"done.json",{"batch":"probe","protocol":"old","files":{"probe.json":"missing"}})
    assert not cp.valid("probe")
    write(folder/"done.json",{"batch":"probe","protocol":"new","files":{"probe.json":"missing"}})
    assert not cp.valid("probe")
