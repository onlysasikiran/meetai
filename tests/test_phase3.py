import json

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
