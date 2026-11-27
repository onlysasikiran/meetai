"""Matched four-question development generation probe across every retrieval variant."""
import json
from pathlib import Path

from meetai.grounding import validated_answer
from meetai.linked_retrieval import retrieve
from meetai.models import Models, validate
from meetai.retrieval import Cache, Index, digest

from .data import ROOT, WORK, load
from .metrics import evidence_score, fingerprint
from .phase3_data import PROTOCOL
from .phase3_dev import OUT, config, evidence, leave_one_out, variants
from .run import write


def main():
    grid = variants()
    winner = json.loads((OUT / "selected.json").read_text())
    grid = {k: grid[k] for k in ("A0","A1","A2","A3","A4")}
    grid.update({"A5": winner}, **leave_one_out(winner))
    cfg = config("baseline")
    models = Models(ROOT / ".assets/models", cfg["threads"], cfg["window"])
    cache = Cache(WORK / "phase3-dev.sqlite")
    rows = []
    try:
        for name, ds, path, cases in load("tune"):
            indexes = {}
            for q in cases:
                if q["id"] not in PROTOCOL["development_generation_probe"]:
                    continue
                completions = {}
                for label, variant in grid.items():
                    system = variant.get("system", "baseline")
                    if system not in indexes:
                        indexes[system] = Index.build(path, models, variant, cache)
                    idx = indexes[system]
                    if system == "meetai":
                        ctx, ids, _, trace = retrieve(idx, q["question"], models, variant)
                    else:
                        ctx, ids, _ = evidence(idx, q["question"], models, variant)
                        trace = None
                    if ctx not in completions:
                        completions[ctx] = models.generate(q["question"], ctx, variant["max_output"])[0]
                    raw = completions[ctx]
                    pred = (validated_answer(raw, ids, idx, models, variant, q["question"], trace)
                            if system == "meetai" else validate(raw, set(ids)))
                    claims = pred["claims"]
                    lookup = {line.id: line.render() for line in idx.lines}
                    private = dict(question=q["question"], reference=q["reference"], facts=q["facts"],
                                   evidence=q["evidence"], claims=claims, cited={r:lookup[r] for c in claims for r in c["cites"]})
                    write(OUT / "private" / ("probe-" + label) / (q["id"] + ".json"), private)
                    rows.append({"id":q["id"], "dataset":ds, "category":q["category"], "meeting":name, "variant":label,
                                 "context_sha256":digest(ctx.encode()), "answer_digest":fingerprint(claims),
                                 "gold_units":len(q["facts"]), "claim_lines":len(claims), "abstained":not claims,
                                 "evidence_recall":evidence_score(q["evidence"],idx.complete_turns(ids))})
                print(q["id"], len(completions), "distinct generation prompts", flush=True)
                write(OUT / "probe.json", rows)
    finally:
        cache.close()


if __name__ == "__main__":
    main()
