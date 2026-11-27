"""Run Phase 2 functional acceptance checks on the real CPU models."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from meetai.minutes import whole_meeting
from meetai.models import Models
from meetai.pipeline import answer, summarize
from meetai.retrieval import Cache, Index, digest

from .data import ROOT, WORK, load
from .stress import stress


def exercise(folder, cfg, models, cache):
    results = {}

    def record(name, check, result, public_text=True):
        trace = result["trace"]
        row = {
            "passed": bool(check),
            "llm_calls": result["llm_calls"],
            "abstained": result["abstained"],
            "delivered_ids": result["ids"],
            "trace": trace,
        }
        row["citation_sets"] = [c["cites"] for c in result["claims"]]
        if public_text:
            row["claims"] = result["claims"]
        if "items" in result:
            row["items"] = result["items"]
        if "context_tokens" in result:
            row["evidence_tokens"] = result["context_tokens"]
            row["passed"] &= result["context_tokens"] <= trace["budget"]
        row["passed"] &= all(set(refs) <= set(row["delivered_ids"]) for refs in row["citation_sets"])
        results[name] = row
        print(("PASS " if row["passed"] else "FAIL ") + name, flush=True)

    for suffix in ("txt", "md", "pdf"):
        idx = Index.build(ROOT / f"examples/launch.{suffix}", models, cfg, cache)
        query = "Who is the current launch owner?"
        result = answer(idx, query, models, cfg, cache)
        record(
            "launch_" + suffix,
            any("priya" in c["text"].casefold() and "3" in c["cites"] for c in result["claims"]),
            result,
        )
    reuse = Index.build(ROOT / "examples/launch.pdf", models, cfg, cache)
    warm = answer(reuse, query, models, cfg, cache)
    record(
        "reopened_cache",
        reuse.ingest["graph_cache_hit"]
        and warm["cache_hit"]
        and warm["llm_calls"] == 0
        and warm["claims"] == result["claims"],
        warm,
    )

    turns, cases = stress(0)
    path = folder / "juniper.txt"
    path.write_text("\n".join(f"{s}: {t}" for s, t in turns))
    idx = Index.build(path, models, cfg, cache)
    for case in (cases[0], cases[2], cases[3], cases[4]):
        result = answer(idx, case["question"], models, cfg, cache)
        text = " ".join(c["text"] for c in result["claims"]).casefold()
        cites = {int(ref.split(".")[0]) for c in result["claims"] for ref in c["cites"]}
        meets = all(fact.casefold() in text for fact in case["facts"]) and all(
            any(set(option) <= cites for option in group) for group in case["evidence"]
        )
        record(case["id"], meets and (result["abstained"] == (not case["facts"])), result)

    idx = Index.build(ROOT / "examples/riverside.txt", models, cfg, cache)
    topic = answer(idx, "Summarize what was said about the vendor quote.", models, cfg, cache)
    record(
        "topic_summary",
        topic["trace"]["route"] == "topic"
        and any("4" in c["cites"] and "quote" in c["text"].casefold() for c in topic["claims"]),
        topic,
    )
    summary = summarize(idx, models, cfg)
    record("meeting_summary", len(summary["claims"]) >= 2, summary)
    short_summary = summarize(idx, models, dict(cfg, budget=45))
    record(
        "hierarchical_summary",
        short_summary["trace"]["initial_packets"] > 1
        and short_summary["llm_calls"] >= 3
        and bool(short_summary["claims"]),
        short_summary,
    )
    records = whole_meeting(idx, models, cfg, structured=True)
    kinds = {item["kind"] for item in records["items"]}
    action = any(
        (item.get("owner") or {}).get("value") == "Pablo"
        and (item.get("deadline") or {}).get("value") == "May 14"
        for item in records["items"]
    )
    record(
        "structured_minutes",
        action and {"decision", "action", "open_issue", "follow_up"} <= kinds,
        records,
    )

    # A real tuning meeting checks the external adapter and bounded generation. Record only IDs.
    name, _, path, cases = next(load("tune"))
    idx = Index.build(path, models, cfg, cache)
    result = answer(idx, cases[0]["question"], models, cfg, cache)
    record(
        "elitr_development_interface",
        idx.ingest["blocks"] > 1 and bool(result["ids"]) and result["llm_calls"] == 1,
        result,
        public_text=False,
    )
    results["elitr_development_interface"]["meeting"] = name
    results["elitr_development_interface"]["question_id"] = cases[0]["id"]
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=WORK / "phase2-functional.json")
    output = parser.parse_args().out
    config_path = ROOT / "configs/meetai.json"
    cfg = json.loads(config_path.read_text())
    models = Models(ROOT / ".assets/models", cfg["threads"], cfg["window"])
    WORK.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=WORK) as folder:
        cache = Cache(Path(folder) / "cache.sqlite")
        try:
            cases = exercise(Path(folder), cfg, models, cache)
        finally:
            cache.close()
    result = {
        "scope": "Functional checks on examples and development data",
        "config_sha256": digest(config_path.read_bytes()),
        "passed": sum(r["passed"] for r in cases.values()),
        "total": len(cases),
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    if result["passed"] != result["total"]:
        raise SystemExit("Functional checks failed. See " + str(output))


if __name__ == "__main__":
    main()
