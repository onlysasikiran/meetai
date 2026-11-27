"""Resolve stable IDs into local transcripts and upstream ground truth."""

import json
import re
from pathlib import Path
from zipfile import ZipFile

from .stress import stress

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".work"
RAW = ROOT / ".assets/raw"

# Evidence units: outer list = required groups, each group = alternative, sufficient sets of lines.
# ELITR references are deliberately hydrated from the authors' archive.
ELITR = {
    "001": [
        (3, "simple", [[[197], [202], [50]]]),
        (5, "simple", [[[822, 823]]]),
        (6, "simple", [[[975]]]),
        (7, "simple", [[[975, 976]]]),
    ],
    "003": [
        (2, "simple", [[[12], [186]]]),
        (3, "simple", [[[39], [40]]]),
        (4, "simple", [[[59], [86]]]),
        (5, "temporal", [[[144, 146], [140, 149]]]),
    ],
    "002": [
        (4, "temporal", [[[423, 424], [423, 425]]]),
        (5, "simple", [[[100], [186], [654]]]),
        (6, "simple", [[[110], [111], [112], [532]]]),
        (8, "simple", [[[273]]]),
    ],
    "004": [
        (2, "simple", [[[94]]]),
        (3, "temporal", [[[1324, 1325], [1324, 1327]]]),
        (5, "simple", [[[246, 252], [246, 253], [257, 258]]]),
        (7, "simple", [[[1168]]]),
    ],
}
# Reference-fact outlines help reviewers map independent clauses to the upstream summary.
QMS = {
    "Bed004": [
        (2, ["same-voice concern", "role distinction/demonstration"], [[[51]], [[48, 52]]]),
        (3, ["hire", "subject coordination", "graduate training"], [[[96]], [[114]], [[116]]]),
        (6, ["object-specific actions", "landmark example"], [[[223, 225]], [[225]]]),
    ],
    "ES2002a": [
        (2, ["manager", "designer", "marketing", "interface"], [[[3]], [[5]], [[7]], [[8]]]),
        (
            5,
            ["missing competitor context", "price comparison", "purchasing habits"],
            [[[171, 172]], [[174]], [[178]]],
        ),
    ],
    "Bed002": [
        (
            2,
            ["procedural questions", "deletion or identification", "curiosity", "request for purpose"],
            [[[15], [32]], [[36]], [[44, 46]], [[63, 65]]],
        ),
        (
            3,
            [
                "consent",
                "read digits",
                "microphone position for breathing",
                "transcript anonymity",
                "meeting query goal",
            ],
            [[[8], [24]], [[8], [20]], [[24]], [[33]], [[66]]],
        ),
    ],
    "Bmr005": [
        (
            3,
            [
                "recorded hours",
                "transcribed minutes",
                "more ICSI groups",
                "broadcast collaboration",
                "record shows at ICSI",
            ],
            [[[619], [620], [695]], [[736]], [[631, 633]], [[719]], [[725]]],
        ),
        (
            4,
            ["roughly 100-hour goal", "revise consent", "ten-session renewal"],
            [[[706]], [[694]], [[694]]],
        ),
    ],
}
SPLITS = {
    "tune": (["001", "003"], ["Bed004", "ES2002a"], [0, 1]),
    "checkpoint": (["002", "004"], ["Bed002", "Bmr005"], [2, 3]),
}


def save_transcript(name, turns):
    out = WORK / "transcripts" / (name + ".txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(f"{speaker}: {' '.join(body.split())}" for speaker, body in turns) + "\n"
    out.write_text(text)
    return out


def load(split):
    el, qm, sy = SPLITS[split]
    with ZipFile(RAW / "elqa.zip") as qa, ZipFile(RAW / "elcorpus.zip") as corp:
        qa_data = json.loads(qa.read("data/elitr-bench-qa_dev.json", pwd=b"utter"))["meetings"]
        for num in el:
            name = "meeting_en_dev_" + num
            (path,) = [n for n in corp.namelist() if f"en/dev/{name}/transcript_MAN" in n]
            source = corp.read(path).decode("utf-8-sig").splitlines()
            speaker, turns = "Transcript", []
            for line in source:
                found = re.match(r"^\(([^)]+)\)\s*(.*)$", line)
                if found:
                    speaker, line = found.groups()
                # One source line is one reference unit, with inherited speaker identity.
                turns.append((speaker, line.strip() or "[pause]"))
            questions = {int(q["id"]): q for m in qa_data if m["id"] == name for q in m["questions"]}
            cases = []
            for idx, category, evidence in ELITR[num]:
                q = questions[idx]
                cases.append(
                    {
                        "id": f"el-{num}-{idx}",
                        "question": q["question"],
                        "reference": q["groundtruth-answer"],
                        "facts": [q["groundtruth-answer"]],
                        "evidence": evidence,
                        "category": category,
                    }
                )
            yield name, "ELITR-Bench", save_transcript(name, turns), cases
    for name in qm:
        data = json.loads((RAW / (name + ".json")).read_text())
        turns = [(r["speaker"], r["content"] or "[pause]") for r in data["meeting_transcripts"]]
        cases = []
        for idx, facts, evidence in QMS[name]:
            q = data["specific_query_list"][idx - 1]
            cases.append(
                {
                    "id": f"qm-{name}-{idx}",
                    "question": q["query"],
                    "reference": q["answer"],
                    "facts": facts,
                    "evidence": evidence,
                    "category": "topic-summary",
                    "official_ranges": [[int(a) + 1, int(b) + 1] for a, b in q["relevant_text_span"]],
                }
            )
        yield name, "QMSum", save_transcript(name, turns), cases
    for n in sy:
        turns, cases = stress(n)
        yield f"stress-{n}", "MeetAI-Stress", save_transcript(f"stress-{n}", turns), cases


def meetingbank():
    with (RAW / "meetingbank.json").open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    # Fixed, outcome-independent size probes. Preserve each official agenda segment intact.
    for label, row in (
        ("longest", max(rows, key=lambda x: len(x["transcript"].split()))),
        ("first", rows[0]),
    ):
        sentences = re.split(r"(?<=[.!?])\s+", row["transcript"].strip())
        yield (
            label,
            row["uid"],
            save_transcript("bank-" + label, [("Transcript", s) for s in sentences]),
            row["summary"],
        )
