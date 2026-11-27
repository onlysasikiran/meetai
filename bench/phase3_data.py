"""Disjoint official test adapters and independent synthetic reference construction."""
import json
import random
import re
from zipfile import ZipFile

from .data import RAW, ROOT, save_transcript

PROTOCOL = json.loads((ROOT / "configs/phase3-protocol.json").read_text())
NAMES = {
    81037: ("Aster", "Aster Point", "Isha", "Omar", "April 6", "June 18", "Quartz", "Maple"),
    24911: ("Solstice", "Solstice Ridge", "Petra", "Felix", "May 9", "July 23", "Saffron", "Marble"),
}
DEPARTMENTS = ("facilities", "dispatch", "training", "inventory", "procurement", "support", "audit", "printing")
ACTIONS = ("reconciled the counts", "reviewed the drawings", "collected signatures", "filed the paperwork", "scheduled an inspection", "circulated the minutes")
PLACES = ("assembly bay", "overflow store", "trial site", "north depot", "customer office")


def scenario(seed, length):
    rng = random.Random(seed)
    project, other, old, owner, before, after, alias, gate = NAMES[seed]
    turns = []
    for i in range(length):
        team = rng.choice(DEPARTMENTS)
        body = f"Item {i + 1}: the {team} desk {rng.choice(ACTIONS)} for the {rng.choice(PLACES)} before closing the log."
        turns.append((rng.choice(("Clerk", "Coordinator", "Analyst", "Secretary")), body))
    where = [max(1, min(length, round(p * length))) for p in (.04, .16, .26, .42, .54, .72, .91, .96)]
    if seed == 81037:
        notes = [
            f"Our initial delivery owner for {project} was {old}. The planned deadline was {before}.",
            f"{project} is also known as {alias} in the dispatch records.",
            f"{other} uses a different sign-off called Ivory.",
            f"We considered canceling {project}; that proposal was rejected.",
            f"Dispatch requires the {gate} clearance for {alias} before release.",
            f"Approved handover: {owner} now owns delivery of {project}, replacing {old}.",
            f"The agreed delivery deadline for {project} has moved from {before} to {after}.",
            f"{other} still lists {old} as its delivery owner.",
        ]
        questions = [f"Who originally owned delivery of {project}?", f"Whose responsibility is {project} delivery now?",
                     f"Give the current delivery owner and deadline for {project}.",
                     f"What clearance does dispatch require before releasing {project}?", f"What was the approved travel allowance for {project}?"]
    else:
        notes = [
            f"The first delivery plan assigned {project} to {old} with a deadline of {before}.",
            f"The internal code name {alias} refers to {project}.",
            f"Do not confuse {other} with the transfer on today's agenda. Its gate is Indigo.",
            f"A draft retaining {old} on {project} was declined.",
            f"The required release gate for {alias} is {gate}.",
            f"The chair confirmed that {owner} takes over as delivery owner for {project}.",
            f"Decision adopted: the revised delivery deadline for {project} is {after}.",
            f"Budget questions for {other} stay with {old}.",
        ]
        questions = [f"Who did the first plan assign {project} delivery to?", f"Who took over ownership of {project} delivery?",
                     f"For {project}, identify the final delivery owner and revised deadline.",
                     f"Which release gate applies to {project}?", f"How much was allocated to catering for {project}?"]
    for i, note in zip(where, notes):
        turns[i - 1] = ("Chair", note)
    first, link, _, _, proof, transfer, revision, _ = where
    fields = [
        ("initial", "simple", [old], [[[first]]]), ("transfer", "temporal", [owner], [[[transfer]]]),
        ("final", "temporal", [owner, after], [[[transfer]], [[revision]]]),
        ("release", "multi-hop", [gate], [[[link]], [[proof]]]), ("absent", "unanswerable", [], []),
    ]
    cases = [{"id": f"s{seed}-{key}", "question": q, "category": cat, "facts": facts,
              "reference": ", ".join(facts) or "UNKNOWN", "evidence": refs}
             for q, (key, cat, facts, refs) in zip(questions, fields)]
    return turns, cases


def official():
    roster = PROTOCOL["heldout"]
    with ZipFile(RAW / "elqa.zip") as qa, ZipFile(RAW / "elcorpus.zip") as corp:
        rows = json.loads(qa.read("data/elitr-bench-qa_test2.json", pwd=b"utter"))["meetings"]
        for suffix in roster["ELITR-Bench"]["meetings"]:
            name = "meeting_en_test2_" + suffix
            (src,) = [n for n in corp.namelist() if f"en/test2/{name}/transcript_MAN" in n]
            speaker, turns = "Transcript", []
            for line in corp.read(src).decode("utf-8-sig").splitlines():
                found = re.match(r"^\(([^)]+)\)\s*(.*)$", line)
                if found:
                    speaker, line = found.groups()
                turns.append((speaker, line.strip() or "[pause]"))
            questions = next(x["questions"] for x in rows if x["id"] == name)
            cases = [{"id": f"et-{suffix}-{q['id']}", "question": q["question"], "reference": q["groundtruth-answer"]}
                     for q in questions if int(q["id"]) in roster["ELITR-Bench"]["question_ids"]]
            yield name, "ELITR-Bench", save_transcript(name, turns), cases
    for name in roster["QMSum"]["meetings"]:
        data = json.loads((RAW / (name + ".json")).read_text())
        turns = [(r["speaker"], r["content"] or "[pause]") for r in data["meeting_transcripts"]]
        cases = []
        for pos in roster["QMSum"]["specific_query_positions"]:
            q = data["specific_query_list"][pos-1]
            cases.append({"id": f"qt-{name}-{pos}", "question": q["query"], "reference": q["answer"],
                          "official_ranges": [[int(a)+1, int(b)+1] for a,b in q["relevant_text_span"]]})
        yield name, "QMSum", save_transcript(name, turns), cases


def heldout():
    gold = json.loads((ROOT / "configs/phase3-gold.json").read_text())
    for name, ds, path, cases in official():
        for q in cases:
            labels = gold[q["id"]]
            q.update(category=labels["category"], facts=labels.get("facts", [q["reference"]]), evidence=labels["evidence"])
        yield name, ds, path, cases
    for seed, length in zip(PROTOCOL["heldout"]["MeetAI-Stress"]["seeds"], (287, 917)):
        turns, cases = scenario(seed, length)
        name = f"test-s{seed}"
        yield name, "MeetAI-Stress", save_transcript(name, turns), cases


def bank_test():
    with (RAW / "meetingbank-test.json").open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for label, row in (("first", rows[0]), ("longest", max(rows, key=lambda r: len(r["transcript"].split())))):
        name = "test-bank-" + label
        turns = [("Transcript", s) for s in re.split(r"(?<=[.!?])\s+", row["transcript"].strip())]
        yield name, row["uid"], save_transcript(name, turns)
