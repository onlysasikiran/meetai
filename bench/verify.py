"""Rebuild every scored context and check frozen annotations against the source adapters."""

import argparse
import json
import re
from pathlib import Path

from meetai.models import Models
from meetai.retrieval import Index, digest

from .data import ROOT, WORK, load
from .metrics import evidence_score, fingerprint, quality


def norm(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/checkpoint.json")
    parser.add_argument("--labels", type=Path, default=ROOT / "results/claim_review.json")
    parser.add_argument("--out", type=Path, default=WORK / "validation.json")
    args = parser.parse_args()
    run = json.loads(args.run.read_text())
    reviews = json.loads(args.labels.read_text())
    frozen = ROOT / "configs/baseline.json"
    assert digest(frozen.read_bytes()) == run["config_sha256"]
    cfg = json.loads(frozen.read_text())
    models = Models(ROOT / ".assets/models", cfg["threads"], cfg["window"], vocab_only=True)
    rows = {r["id"]: r for r in run["queries"]}
    exact, abstentions, audited = [], [], []
    replayed = []
    for meeting, _, path, cases in load("checkpoint"):
        assert digest(path.read_bytes()) == run["ingestion"][meeting]["source_sha256"]
        idx = Index.build(path, models, cfg)
        for case in cases:
            key = case["id"]
            row = rows[key]
            raw = json.loads((WORK / "predictions" / (key + ".json")).read_text())
            assert row["answer_digest"] == fingerprint(raw["claims"])
            rank, _ = idx.rank(case["question"], models, cfg)
            ctx, ids = idx.pack(rank, cfg, models)
            assert ids == row["delivered_ids"], key
            assert len(models.tokens(ctx)) == row["context_tokens"] <= cfg["budget"]
            assert evidence_score(case["evidence"], idx.complete_turns(ids)) == row["evidence_recall"]
            assert all(set(c["cites"]) <= set(ids) for c in raw["claims"])
            actual = quality(row, reviews[key])["correctness"]
            if row["abstained"]:
                assert actual == float(not case["facts"])
                abstentions.append(key)
            elif len(raw["claims"]) == 1 and norm(raw["claims"][0]["text"]) == norm(case["reference"]):
                assert actual == 1.0
                exact.append(key)
            else:
                audited.append(key)
            replayed.append(key)
    assert set(replayed) == set(rows) == set(reviews)
    result = {
        "config_sha256": run["config_sha256"],
        "contexts_rebuilt": len(replayed),
        "matching_predictions_on_cold_repeat": len(rows),
        "deterministic_exact_answers": exact,
        "deterministic_abstentions": abstentions,
        "other_answers_scored_by_claim_review": audited,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
