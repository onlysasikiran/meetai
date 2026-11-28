"""Verify frozen code, matching source hashes, cache workloads and scale invariants."""
import argparse
from hashlib import sha256
from pathlib import Path

from .data import ROOT
from .metrics import fingerprint
from .phase3_data import PROTOCOL
from .phase3_score import read
from .run import write


def verify(directory):
    freeze = read(ROOT / "configs/phase3-freeze.json")
    correction = read(ROOT / "configs/phase3-harness-correction.json")
    for path, expected in freeze["files"].items():
        if path == correction["path"]:
            assert expected == correction["before_sha256"]
            expected = correction["after_sha256"]
        if path in correction.get("additional_files", {}):
            patch = correction["additional_files"][path]
            assert expected == patch["before_sha256"]
            expected = patch["after_sha256"]
        assert sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    runs = {k: read(directory / (k + ".json")) for k in ("baseline", "meetai")}
    base = runs["baseline"]
    targets = [(t, ".txt") for t in [*PROTOCOL["stress_targets"], "turns"]] + [(20000, ".md"), (20000, ".pdf")]
    scales = {}
    for key, run in runs.items():
        assert run["config_sha256"] == freeze["files"][f"configs/{key}.json"]
        assert run["runtime"] == base["runtime"]
        assert {m:(v["source_sha256"], v["tokens"], v["turns"]) for m,v in run["ingestion"].items()} == {
            m:(v["source_sha256"], v["tokens"], v["turns"]) for m,v in base["ingestion"].items()}
        assert not any(v["cache_hit"] for v in run["ingestion"].values())
        ids = [r["id"] for r in run["queries"]]
        assert ids == [r["id"] for r in base["queries"]] and len(set(ids)) == 19
        for r in run["queries"]:
            assert r["context_tokens"] <= 1024 and r["llm_calls"] == 1
            assert len(r["cold_repeats_s"]) == 2 and len(r["warm_s"]) == 5
        items = read(directory / ("scale-" + key + ".json"))
        keyed = {(i["target"],i["format"]): i for i in items}
        assert len(items) == len(keyed) == len(targets) and set(targets) == set(keyed)
        scales[key] = keyed
        for (target, ext), item in keyed.items():
            v = item["ingestion"]
            assert not v["cache_hit"] and len(item["qa"]) == 2
            if target == "turns":
                assert v["turns"] == PROTOCOL["separate_turns"]
            else:
                assert target <= v["tokens"] <= target + max(64,target*.001)
            for q in item["qa"]:
                assert fingerprint(q["claims"]) == q["answer_digest"] and q["tokens"] <= 1024
        for i in range(2):
            qa = [keyed[20000, ext]["qa"][i] for ext in (".txt",".md",".pdf")]
            assert len({(q["answer_digest"],tuple(q["ids"])) for q in qa}) == 1
    for key in targets:
        assert scales["baseline"][key]["source_sha256"] == scales["meetai"][key]["source_sha256"]
    return {"freeze_files_verified":len(freeze["files"]), "matched_primary_queries":19,
            "cold_calls":2*2*19, "exact_cache_hits":2*5*19, "matched_sources":len(base["ingestion"]),
            "scale_variants_per_system":len(targets), "scale_cold_queries":len(targets)*4,
            "largest_requested_tier":max(PROTOCOL["stress_targets"]), "separate_speaker_turns":PROTOCOL["separate_turns"],
            "formats":"equal normalized turns, delivered IDs and released answers for TXT, Markdown and text-layer PDF"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=ROOT / "results/phase3")
    args = parser.parse_args()
    result = verify(args.dir)
    write(args.dir / "verification.json", result)
    print(result)


if __name__ == "__main__":
    main()
