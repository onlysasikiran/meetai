"""Cold builds, independent inference misses, exact-cache hits and budget diagnostics."""
import argparse
import json
import os
import platform
import resource
import tempfile
from importlib.metadata import version
from time import perf_counter

from meetai.grounding import validated_answer
from meetai.models import Models
from meetai.pipeline import answer
from meetai.retrieval import Cache, Index, digest

from .data import ROOT, WORK
from .metrics import evidence_score, fingerprint
from .phase3_data import PROTOCOL, bank_test, heldout
from .phase3_dev import OUT, config, evidence
from .run import check_query, write


class CaptureModels(Models):
    def generate(self, *args, **kwargs):
        output = super().generate(*args, **kwargs)
        self.last_raw = output[0]
        return output


def runtime():
    return {"cpu": next(s.split(":", 1)[1].strip() for s in open("/proc/cpuinfo") if s.startswith("model name")),
            "cpus": len(os.sched_getaffinity(0)), "python": platform.python_version(),
            "packages": {p: version(p) for p in ("llama-cpp-python","onnxruntime","numpy","pypdf","tokenizers")},
            "omp": os.environ.get("OMP_NUM_THREADS"), "blas": os.environ.get("OPENBLAS_NUM_THREADS"),
            "hash_seed": os.environ.get("PYTHONHASHSEED"), "system": platform.platform()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("system", choices=("baseline", "meetai"))
    ap.add_argument("--meeting", help="Run one held-out meeting, or bank for the operational samples")
    args = ap.parse_args()
    name = args.system
    cfg = config(name)
    run = {"system": name, "config_sha256": digest((ROOT / f"configs/{name}.json").read_bytes()), "runtime": runtime(),
           "cold_passes": PROTOCOL["cold_passes"], "warm_passes": PROTOCOL["warm_passes"],
           "ingestion": {}, "queries": [], "budget_sweep": []}
    OUT.mkdir(parents=True, exist_ok=True)
    t = perf_counter()
    models = CaptureModels(ROOT / ".assets/models", cfg["threads"], cfg["window"])
    if name == "meetai":
        _ = models.nli
    run["startup_s"] = perf_counter() - t
    with tempfile.TemporaryDirectory(dir=WORK) as tmp:
        from pathlib import Path
        cache = Cache(Path(tmp) / "fresh.sqlite")
        try:
            for meeting, ds, path, cases in heldout():
                if args.meeting and args.meeting != meeting:
                    continue
                idx = Index.build(path, models, cfg, cache)
                assert not idx.ingest["cache_hit"]
                run["ingestion"][meeting] = dict(idx.ingest, dataset=ds, source_sha256=digest(path.read_bytes()))
                rows = {}
                for n in range(run["cold_passes"]):
                    with cache.db:
                        cache.db.execute("DELETE FROM answers")
                    for case in cases if n == 0 else reversed(cases):
                        key = case["id"]
                        r = check_query(case, idx, cfg, models, cache, ds, meeting, full=n == 0)
                        if n == 0:
                            rows[key] = r
                            r["cold_repeats_s"] = [r["cold_s"]]
                            origin = WORK / "predictions" / (key + ".json")
                            private = json.loads(origin.read_text())
                            if name == "meetai":
                                parsed = validated_answer(models.last_raw, r["delivered_ids"], idx, models, dict(cfg, semantic_validation=False))
                                private["unguarded"] = parsed["claims"]
                                lookup = {line.id: line.render() for line in idx.lines}
                                private["unguarded_citations"] = {ref: lookup[ref] for c in parsed["claims"] for ref in c["cites"]}
                                r["without_validation_digest"] = fingerprint(parsed["claims"])
                                r["without_validation_lines"] = len(parsed["claims"])
                            write(OUT / "private" / name / (key + ".json"), private)
                        else:
                            if (rows[key]["answer_digest"],rows[key]["delivered_ids"]) != (r["answer_digest"],r["delivered_ids"]):
                                raise RuntimeError("Nondeterministic cold output: " + key)
                            rows[key]["cold_repeats_s"].append(r["cold_s"])
                        print(name, meeting, n+1, key, round(r["cold_s"], 2), flush=True)
                for case in cases:
                    row = rows[case["id"]]
                    row["warm_s"] = []
                    for _ in range(run["warm_passes"]):
                        t = perf_counter()
                        hit = answer(idx, case["question"], models, cfg, cache)
                        row["warm_s"].append(perf_counter() - t)
                        assert hit["cache_hit"] and hit["llm_calls"] == 0 and fingerprint(hit["claims"]) == row["answer_digest"]
                    run["queries"].append(row)
                    if case["evidence"]:
                        for budget in PROTOCOL["diagnostic_budgets"]:
                            ctx, ids, _ = evidence(idx, case["question"], models, dict(cfg, budget=budget))
                            used = len(models.tokens(ctx))
                            assert used <= budget
                            run["budget_sweep"].append({"id": case["id"], "dataset": ds, "budget": budget,
                                                        "tokens": used, "recall": evidence_score(case["evidence"],idx.complete_turns(ids))})
                write(OUT / (name + "-progress.json"), run)
            bank_work = bank_test() if args.meeting in (None, "bank") else ()
            for meeting, uid, path in bank_work:
                idx = Index.build(path, models, cfg, cache)
                assert not idx.ingest["cache_hit"]
                run["ingestion"][meeting] = dict(idx.ingest, dataset="MeetingBank", uid=uid, source_sha256=digest(path.read_bytes()))
        finally:
            cache.close()
    if not run["ingestion"]:
        raise ValueError(f"Unknown held-out meeting: {args.meeting}")
    run["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    write(OUT / (name + ".json"), run)


if __name__ == "__main__":
    main()
