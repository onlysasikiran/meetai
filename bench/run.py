import argparse
import json
import os
import platform
import resource
import tempfile
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from meetai.models import Models
from meetai.pipeline import answer, summarize
from meetai.retrieval import Cache, Index, digest

from .data import ROOT, WORK, load, meetingbank
from .metrics import evidence_score, fingerprint


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def check_query(case, idx, cfg, models, cache, ds, meeting, full):
    start = perf_counter()
    pred = answer(idx, case["question"], models, cfg, cache)
    latency = perf_counter() - start
    if pred["cache_hit"] or pred["llm_calls"] != 1:
        raise RuntimeError("Cold workload reused a prediction")
    if pred["context_tokens"] > cfg["budget"]:
        raise RuntimeError("Evidence budget exceeded")
    row = {
        "id": case["id"],
        "dataset": ds,
        "meeting": meeting,
        "category": case["category"],
        "gold_units": len(case["facts"]),
        "answer_digest": fingerprint(pred["claims"]),
        "claim_lines": len(pred["claims"]),
        "abstained": pred["abstained"],
        "rejected_lines": len(pred["rejected"]),
        "context_tokens": pred["context_tokens"],
        "complete_turns": pred["complete_turns"],
        "delivered_ids": pred["ids"],
        "usage": pred["usage"],
        "evidence_recall": evidence_score(case["evidence"], pred["complete_turns"]),
        "cold_s": latency,
        "stage_s": pred["timings"],
        "llm_calls": pred["llm_calls"],
    }
    if case.get("official_ranges"):
        gold = {n for lo, hi in case["official_ranges"] for n in range(lo, hi + 1)}
        row["official_turn_recall"] = len(gold.intersection(pred["complete_turns"])) / len(gold)
    if full:
        # Upstream authors distribute some resources in protected archives. Keep verbatim data local.
        lookup = {l.id: l.render() for l in idx.lines}
        private = {
            "question": case["question"],
            "reference": case["reference"],
            "facts": case["facts"],
            "claims": pred["claims"],
            "rejected": pred["rejected"],
            "evidence": case["evidence"],
            "cited": {r: lookup[r] for c in pred["claims"] for r in c["cites"]},
        }
        write(WORK / "predictions" / (case["id"] + ".json"), private)
    return row


def main():
    parser = argparse.ArgumentParser(description="Run the frozen baseline checkpoint on CPU")
    parser.add_argument("--out", type=Path, default=WORK / "replay")
    out = parser.parse_args().out
    cfg_file = ROOT / "configs/baseline.json"
    cfg = json.loads(cfg_file.read_text())
    WORK.mkdir(exist_ok=True)
    run = {
        "config_sha256": digest(cfg_file.read_bytes()),
        "cold_passes": 2,
        "warm_passes": 5,
        "seed": 42,
        "runtime": {
            "platform": platform.platform(),
            "cpu": next(
                l.split(":", 1)[1].strip()
                for l in Path("/proc/cpuinfo").read_text().splitlines()
                if l.startswith("model name")
            ),
            "visible_cpus": len(os.sched_getaffinity(0)),
            "python": platform.python_version(),
            "ram_meminfo": Path("/proc/meminfo").read_text().splitlines()[0],
            "packages": {
                p: version(p)
                for p in ("llama-cpp-python", "onnxruntime", "tokenizers", "numpy", "pypdf")
            },
        },
        "ingestion": {},
        "queries": [],
        "budget_sweep": [],
    }
    t = perf_counter()
    models = Models(ROOT / ".assets/models", cfg["threads"], cfg["window"])
    run["model_startup_s"] = perf_counter() - t
    with tempfile.TemporaryDirectory(dir=WORK) as tmp:
        cache = Cache(Path(tmp) / "baseline.sqlite")
        try:
            for meeting, ds, path, cases in load("checkpoint"):
                idx = Index.build(path, models, cfg, cache)
                run["ingestion"][meeting] = {
                    **idx.ingest,
                    "dataset": ds,
                    "source_sha256": digest(path.read_bytes()),
                }
                rows = {}
                for pass_no in range(run["cold_passes"]):
                    with cache.db:
                        cache.db.execute("DELETE FROM answers")
                    for case in cases if pass_no == 0 else reversed(cases):
                        r = check_query(case, idx, cfg, models, cache, ds, meeting, pass_no == 0)
                        if pass_no == 0:
                            r["cold_repeats_s"] = [r["cold_s"]]
                            rows[r["id"]] = r
                        else:
                            first = rows[r["id"]]
                            if (
                                first["answer_digest"] != r["answer_digest"]
                                or first["delivered_ids"] != r["delivered_ids"]
                            ):
                                raise RuntimeError(f"Cold repeat diverged for {r['id']}")
                            first["cold_repeats_s"].append(r["cold_s"])
                        print("Cold", pass_no + 1, r["id"], round(r["cold_s"], 2), flush=True)
                for case in cases:
                    row = rows[case["id"]]
                    row["warm_s"] = []
                    for _ in range(run["warm_passes"]):
                        t = perf_counter()
                        cached = answer(idx, case["question"], models, cfg, cache)
                        row["warm_s"].append(perf_counter() - t)
                        assert (
                            cached["cache_hit"]
                            and cached["llm_calls"] == 0
                            and fingerprint(cached["claims"]) == row["answer_digest"]
                        )
                    if case["evidence"]:
                        ranks, _ = idx.rank(case["question"], models, cfg)
                        for budget in (2048, 4096, 8192):
                            ctx, ids = idx.pack(ranks, dict(cfg, budget=budget), models)
                            run["budget_sweep"].append(
                                {
                                    "id": case["id"],
                                    "dataset": ds,
                                    "budget": budget,
                                    "tokens": len(models.tokens(ctx)),
                                    "recall": evidence_score(case["evidence"], idx.complete_turns(ids)),
                                }
                            )
                    run["queries"].append(row)
                write(WORK / "run-progress.json", run)
            for label, uid, path, _ in meetingbank():
                idx = Index.build(path, models, cfg, cache)
                run["ingestion"]["bank-" + label] = {
                    **idx.ingest,
                    "dataset": "MeetingBank",
                    "uid": uid,
                    "source_sha256": digest(path.read_bytes()),
                }
                if label == "first":
                    t = perf_counter()
                    demo = summarize(idx, models, cfg)
                    run["meetingbank_summary"] = {
                        "uid": uid,
                        "llm_calls": demo["llm_calls"],
                        "claim_lines": len(demo["claims"]),
                        "abstained": demo["abstained"],
                        "seconds": perf_counter() - t,
                    }
                    write(WORK / "meetingbank_summary.json", demo)
                print("MeetingBank", label, idx.ingest, flush=True)
        finally:
            cache.close()
    run["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    write(out / "checkpoint.json", run)


if __name__ == "__main__":
    main()
