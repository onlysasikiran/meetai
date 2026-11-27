import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .data import ROOT, WORK
from .metrics import quality, quantile, rollup


def summarize_run(run, reviews):
    queries = run["queries"]
    if set(reviews) != {r["id"] for r in queries} or len(reviews) != len(queries):
        raise ValueError("Exactly one digest-bound review per question is required")
    by_dataset, by_task = defaultdict(list), defaultdict(list)
    for row in queries:
        scored = dict(row, **quality(row, reviews[row["id"]]))
        by_dataset[row["dataset"]].append(scored)
        by_task[(row["dataset"], row["category"])].append(scored)
    dataset = {d: rollup(rs) for d, rs in by_dataset.items()}
    for d in {v["dataset"] for v in run["ingestion"].values()}:
        items = [v for v in run["ingestion"].values() if v["dataset"] == d]
        if any(v["cache_hit"] for v in items):
            raise ValueError("Ingestion throughput requires freshly built indexes")
        dataset.setdefault(d, {}).update(
            meetings_or_segments=len(items),
            ingested_tokens=sum(v["tokens"] for v in items),
            ingestion_tokens_s=sum(v["tokens"] for v in items) / sum(v["seconds"] for v in items),
        )
    all_cold = [v for r in queries for v in r["cold_repeats_s"]]
    all_warm = [v for r in queries for v in r["warm_s"]]
    sweep = defaultdict(list)
    for row in run["budget_sweep"]:
        sweep[(row["dataset"], row["budget"])].append(row)
    return {
        "dataset": dataset,
        "by_category": {f"{ds}/{kind}": rollup(rows) for (ds, kind), rows in sorted(by_task.items())},
        "budget_sweep": [
            {
                "dataset": ds,
                "budget": budget,
                "queries": len(rows),
                "mean_recall": mean(x["recall"] for x in rows),
                "mean_used_tokens": mean(x["tokens"] for x in rows),
            }
            for (ds, budget), rows in sorted(sweep.items())
        ],
        "performance": {
            "cold_queries": len(all_cold),
            "warm_queries": len(all_warm),
            "cold_p50_s": quantile(all_cold, 0.50),
            "cold_p95_s": quantile(all_cold, 0.95),
            "warm_p50_s": quantile(all_warm, 0.50),
            "warm_p95_s": quantile(all_warm, 0.95),
            "model_startup_s": run["model_startup_s"],
            "peak_rss_mib": run["peak_rss_mib"],
            "mean_first_pass_stage_s": {
                stage: mean(r["stage_s"][stage] for r in queries) for stage in queries[0]["stage_s"]
            },
        },
        "answers": {r["id"]: quality(r, reviews[r["id"]]) for r in queries},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Rescore recorded predictions using their audited, frozen claim labels"
    )
    parser.add_argument("--run", type=Path, default=ROOT / "results/checkpoint.json")
    parser.add_argument("--labels", type=Path, default=ROOT / "results/claim_review.json")
    parser.add_argument("--out", type=Path, default=WORK / "scores.json")
    args = parser.parse_args()
    scores = summarize_run(json.loads(args.run.read_text()), json.loads(args.labels.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scores, indent=2) + "\n")
    print(json.dumps({"dataset": scores["dataset"], "performance": scores["performance"]}, indent=2))


if __name__ == "__main__":
    main()
