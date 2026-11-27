import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from meetai.models import Models
from meetai.retrieval import Index

from .data import ROOT, load
from .metrics import evidence_score


def main():
    parser = argparse.ArgumentParser(description="Fit flat retrieval on tuning meetings only")
    parser.add_argument("--out", type=Path, default=ROOT / ".work/retune")
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((ROOT / "configs/protocol.json").read_text())
    base = protocol["base"]
    models = Models(ROOT / ".assets/models", base["threads"], base["window"], vocab_only=True)
    stats = defaultdict(lambda: defaultdict(list))
    for chunk_size in protocol["chunk_sizes"]:
        for name, ds, path, cases in load("tune"):
            cfg = dict(base, chunk_tokens=chunk_size)
            idx = Index.build(path, models, cfg)
            for case in cases:
                if not case["evidence"]:
                    continue
                ranks, _ = idx.rank(case["question"], models, cfg)
                for parent, top, budget in itertools.product(
                    protocol["parent_sizes"], protocol["top_ks"], protocol["budgets"]
                ):
                    key = (chunk_size, parent, top, budget)
                    ctx, ids = idx.pack(
                        ranks, dict(cfg, parent_chunks=parent, top_k=top, budget=budget), models
                    )
                    stats[key][ds].append(
                        (
                            evidence_score(case["evidence"], idx.complete_turns(ids)),
                            len(models.tokens(ctx)),
                        )
                    )
            print("Tuned", chunk_size, name, flush=True)
    rows = []
    for (chunk, parent, top, budget), groups in stats.items():
        recall = {ds: mean(v[0] for v in vals) for ds, vals in groups.items()}
        rows.append(
            {
                "chunk_tokens": chunk,
                "parent_chunks": parent,
                "top_k": top,
                "budget": budget,
                "mean_recall": mean(recall.values()),
                "recall_by_dataset": recall,
                "mean_tokens": mean(v[1] for vals in groups.values() for v in vals),
                "eligible_questions": sum(map(len, groups.values())),
            }
        )
    best = max(r["mean_recall"] for r in rows)
    eligible = [r for r in rows if r["mean_recall"] >= best - 0.02]
    winner = min(
        eligible,
        key=lambda r: (
            r["budget"],
            -r["mean_recall"],
            r["mean_tokens"],
            r["parent_chunks"],
            r["top_k"],
            r["chunk_tokens"],
        ),
    )
    frozen = dict(base, **{k: winner[k] for k in ("chunk_tokens", "parent_chunks", "top_k", "budget")})
    (out / "baseline.json").write_text(json.dumps(frozen, indent=2) + "\n")
    (out / "tuning.json").write_text(json.dumps({"winner": winner, "trials": rows}, indent=2) + "\n")
    print("Selected", winner, flush=True)


if __name__ == "__main__":
    main()
