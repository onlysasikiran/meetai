"""Development-only component and candidate sweeps. No held-out labels are loaded here."""
import argparse
import json
from statistics import mean
from time import perf_counter

from meetai.linked_retrieval import retrieve
from meetai.models import Models
from meetai.retrieval import Cache, Index, digest

from .data import ROOT, WORK, load
from .metrics import evidence_score
from .run import write

OUT = WORK / "phase3"
CANDIDATES = ("A4", "compact", "windows", "compact-windows")


def config(name):
    return json.loads((ROOT / "configs" / (name + ".json")).read_text())


def variants():
    linked = config("meetai-phase2")
    return {
        "A0": config("baseline"),
        "A1": dict(linked, links=False, adaptive=False, coverage=False),
        "A2": dict(linked, adaptive=False, coverage=False),
        "A3": dict(linked, coverage=False),
        "A4": linked,
        "compact": dict(linked, turn_pool=20),
        "windows": dict(linked, dialogue_windows=True),
        "compact-windows": dict(linked, turn_pool=20, dialogue_windows=True),
    }


def leave_one_out(winner):
    result = {f"without_{key}": dict(winner, **{key: False}) for key in ("hierarchy", "links", "adaptive", "coverage")}
    if winner["turn_pool"] != 36:
        result["without_compact_pool"] = dict(winner, turn_pool=36)
    if winner.get("dialogue_windows"):
        result["without_windows"] = dict(winner, dialogue_windows=False)
    return result


def evidence(idx, question, models, cfg):
    if cfg.get("system") == "meetai":
        context, ids, stage, _ = retrieve(idx, question, models, cfg)
    else:
        ranks, stage = idx.rank(question, models, cfg)
        t = perf_counter()
        context, ids = idx.pack(ranks, cfg, models)
        stage["assembly_s"] = perf_counter() - t
    return context, ids, stage


def summarize(rows):
    datasets = sorted({r["dataset"] for r in rows})
    per_ds = {ds: mean(r["recall"] for r in rows if r["dataset"] == ds and r["recall"] is not None) for ds in datasets}
    return {
        "macro_recall": mean(per_ds.values()), "by_dataset": per_ds,
        "mean_tokens": mean(r["tokens"] for r in rows),
        "mean_retrieval_s": mean(sum(r["stage_s"].values()) for r in rows),
        "questions": len(rows),
    }


def sweep(models, configs):
    rows = {name: [] for name in configs}
    cache = Cache(WORK / "phase3-dev.sqlite")
    try:
        for name, ds, path, cases in load("tune"):
            indexes = {}
            for number, q in enumerate(cases):
                labels = list(configs)
                for label in labels[number % len(labels):] + labels[:number % len(labels)]:
                    cfg = configs[label]
                    key = cfg.get("system", "baseline")
                    if key not in indexes:
                        indexes[key] = Index.build(path, models, cfg, cache)
                    idx = indexes[key]
                    ctx, ids, timing = evidence(idx, q["question"], models, cfg)
                    rows[label].append({
                        "id": q["id"], "meeting": name, "dataset": ds, "tokens": len(models.tokens(ctx)),
                        "recall": evidence_score(q["evidence"], idx.complete_turns(ids)), "stage_s": timing,
                        "ids": ids,
                    })
            print("dev", name, flush=True)
    finally:
        cache.close()
    return rows


def main():
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("--out", type=type(ROOT), default=OUT)
    args = arg.parse_args()
    base = config("baseline")
    models = Models(ROOT / ".assets/models", base["threads"], base["window"], vocab_only=True)
    grid = variants()
    rows = sweep(models, grid)
    overview = {k: summarize(v) for k, v in rows.items()}
    best = max(overview[k]["macro_recall"] for k in CANDIDATES)
    eligible = [k for k in CANDIDATES if overview[k]["macro_recall"] >= best - .02]
    winner = min(eligible, key=lambda k: (overview[k]["mean_tokens"], grid[k]["turn_pool"], bool(grid[k].get("dialogue_windows")), k))
    frozen = grid[winner]
    write(args.out / "selected.json", frozen)
    write(args.out / "selection.json", {"winner": winner, "candidates": overview})
    print("selected", winner, overview[winner], flush=True)
    rows.update(sweep(models, leave_one_out(frozen)))
    write(args.out / "dev.json", {
        "selection": winner, "A5_equals": winner, "config_sha256": digest((args.out / "selected.json").read_bytes()),
        "summary": {k: summarize(v) for k, v in rows.items()}, "rows": rows,
    })


if __name__ == "__main__":
    main()
