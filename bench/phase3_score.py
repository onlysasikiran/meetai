"""Recompute held-out metrics and paired meeting-cluster intervals from reviewed atoms."""
import argparse
import json
import random
from collections import defaultdict
from math import ceil
from pathlib import Path
from statistics import mean, median

from .data import ROOT
from .metrics import fingerprint, quality, quantile, rollup
from .phase3_data import PROTOCOL
from .phase3_dev import OUT
from .run import write


def read(path):
    return json.loads(path.read_text())


def packets(private):
    batch = {}
    for path in sorted(private.glob("*/*.json")):
        item = read(path)
        for field, citations in (("claims", "cited"), ("unguarded", "unguarded_citations")):
            claims = item.get(field)
            if not claims:
                continue
            key = path.stem + "/" + fingerprint(claims)
            if key not in batch:
                batch[key] = {k: item[k] for k in ("question", "reference", "facts", "evidence")}
                batch[key].update(id=path.stem, key=key, claims=claims, cited=item[citations])
    values = list(batch.values())
    random.Random(5800).shuffle(values)
    return values


def review(row, labels):
    key = row["id"] + "/" + row["answer_digest"]
    marks = labels[key] if row["claim_lines"] else []
    return {**row, **quality(row, {"digest": row["answer_digest"], "atoms": marks})}


def quality_summary(rows):
    eligible = [r["evidence_recall"] for r in rows if r["evidence_recall"] is not None]
    return {
        "questions": len(rows), "correctness": mean(r["correctness"] for r in rows),
        "recall": mean(eligible) if eligible else None, "atoms": sum(r["atom_count"] for r in rows),
        "unsupported": sum(r["unsupported"] for r in rows),
        "citations_supported": sum(r["citations_supported"] for r in rows),
        "answered": sum(not r["abstained"] for r in rows),
    }


def complex_rows(rows):
    return [r for r in rows if r["category"] in PROTOCOL["complex_categories"]]


def summary(rows, run):
    result = rollup(rows)
    ingestion = list(run["ingestion"].values())
    times = [s for r in rows for s in r["cold_repeats_s"]]
    stages = sorted({name for r in rows for name in r["stage_s"]} - {"total_s"})
    result.update(
        cold_p50_s=median(times), cold_mean_s=mean(times), mean_tokens=mean(r["context_tokens"] for r in rows),
        ingestion_tokens_s=sum(i["tokens"] for i in ingestion) / sum(i["seconds"] for i in ingestion),
        ingestion_total_s=sum(i["seconds"] for i in ingestion), peak_rss_mib=run["peak_rss_mib"],
        stage_mean_s={name: mean(r["stage_s"].get(name, 0) for r in rows) for name in stages},
        real=rollup([r for r in rows if r["dataset"] != "MeetAI-Stress"]), complex=rollup(complex_rows(rows)),
        official_span_recall=mean(r["official_turn_recall"] for r in rows if "official_turn_recall" in r),
    )
    for field in ("dataset", "category"):
        result["per_" + field] = {
            k: rollup([r for r in rows if r[field] == k]) for k in sorted({r[field] for r in rows})
        }
    result["budget_sweep"] = {}
    for b in PROTOCOL["diagnostic_budgets"]:
        sweep = [s for s in run["budget_sweep"] if s["budget"] == b]
        result["budget_sweep"][str(b)] = {"recall": mean(s["recall"] for s in sweep), "mean_tokens": mean(s["tokens"] for s in sweep)}
    return result


def differences(left, right):
    a, b = rollup(left), rollup(right)
    comp = [mean(r["correctness"] for r in complex_rows(rows)) for rows in (left, right)]
    unsupported = (a["unsupported_claim_rate"], b["unsupported_claim_rate"])
    return {
        "correctness_pp": 100 * (b["answer_correctness"] - a["answer_correctness"]),
        "complex_correctness_pp": 100 * (comp[1] - comp[0]),
        "recall_pp": 100 * (b["evidence_recall"] - a["evidence_recall"]),
        "unsupported_pp": 100 * (unsupported[1] - unsupported[0]) if None not in unsupported else None,
        "cold_p95_relative_pct": 100 * (b["cold_p95_s"] / a["cold_p95_s"] - 1),
    }


def intervals(left, right):
    groups = defaultdict(list)
    indices = defaultdict(list)
    for n, row in enumerate(left):
        assert (row["id"], row["meeting"]) == (right[n]["id"], right[n]["meeting"])
        if row["meeting"] not in groups[row["dataset"]]:
            groups[row["dataset"]].append(row["meeting"])
        indices[row["meeting"]].append(n)
    rng = random.Random(PROTOCOL["bootstrap"]["seed"])
    samples = defaultdict(list)
    for _ in range(PROTOCOL["bootstrap"]["draws"]):
        chosen = [i for names in groups.values() for m in rng.choices(names, k=len(names)) for i in indices[m]]
        vals = differences([left[i] for i in chosen], [right[i] for i in chosen])
        for k, value in vals.items():
            if value is not None:
                samples[k].append(value)
    return {k: {"estimate": v, "lower": quantile(samples[k], .025), "upper": quantile(samples[k], .975),
                "valid_draws": len(samples[k])} for k, v in differences(left, right).items()}


def break_even(rows, runs):
    result = {}
    for meeting in sorted({r["meeting"] for r in rows["baseline"]}):
        ingest = [runs[k]["ingestion"][meeting]["seconds"] for k in ("baseline", "meetai")]
        added = ingest[1] - ingest[0]
        avg = {k: mean(s for r in group if r["meeting"] == meeting for s in r["cold_repeats_s"]) for k, group in rows.items()}
        saving = avg["baseline"] - avg["meetai"]
        result[meeting] = {"extra_ingest_s": added, "saved_s_per_cold_query": saving,
                           "min_queries_for_permanent_break_even": max(0, ceil(added / saving)) if saving > 0 else None,
                           "total_delta_meetai_minus_baseline_s": {str(n): added - n * saving for n in (1, 5, 20)}}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packet", action="store_true")
    ap.add_argument("--dir", type=Path, default=ROOT / "results/phase3")
    ap.add_argument("--reviews", type=Path, default=ROOT / "results/phase3/reviews.json")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.packet:
        write(args.out or OUT / "blind.json", packets(OUT / "private"))
        return
    labels = read(args.reviews)
    runs = {k: read(args.dir / (k + ".json")) for k in ("baseline", "meetai")}
    rows = {k: [review(r, labels) for r in v["queries"]] for k, v in runs.items()}
    result = {k: summary(rows[k], runs[k]) for k in rows}
    result["paired_intervals"] = intervals(rows["baseline"], rows["meetai"])
    shadow = [review(dict(r, answer_digest=r["without_validation_digest"], claim_lines=r["without_validation_lines"],
                          abstained=r["without_validation_lines"] == 0), labels) for r in runs["meetai"]["queries"]]
    result["validation_ablation"] = {"with": quality_summary(rows["meetai"]), "without": quality_summary(shadow)}
    probe = defaultdict(list)
    for row in read(args.dir / "probe.json"):
        probe[row["variant"]].append(review(row, labels))
    result["development_generation"] = {k: quality_summary(v) for k, v in probe.items()}
    result["amortization"] = break_even(rows, runs)
    write(args.out or args.dir / "scores.json", result)


if __name__ == "__main__":
    main()
