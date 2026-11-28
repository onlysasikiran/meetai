"""Run one complete matched benchmark batch, or validate and gather saved batches."""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from .data import ROOT
from .phase3_data import PROTOCOL


BASE = ROOT / "results/phase3/checkpoints"
WORK = ROOT / ".work/phase3"
SYSTEMS = ("baseline", "meetai")
QA = ["meeting_en_test2_" + n for n in PROTOCOL["heldout"]["ELITR-Bench"]["meetings"]]
QA += PROTOCOL["heldout"]["QMSum"]["meetings"]
QA += ["test-s" + str(s) for s in PROTOCOL["heldout"]["MeetAI-Stress"]["seeds"]]
SIZES = [str(n) for n in PROTOCOL["stress_targets"]] + ["turns"]
BATCHES = [*QA, "bank", *("size-" + n for n in SIZES), "probe", "tests"]


def write(path, value):
    from .run import write as write_json
    write_json(path, value)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def protocol_id():
    files = [*sorted((ROOT / "meetai").glob("*.py")), *sorted((ROOT / "bench").glob("phase3*.py")),
             ROOT / "bench/metrics.py", ROOT / "bench/run.py", ROOT / "requirements.lock",
             ROOT / "scripts/assets.py", *sorted((ROOT / "configs").glob("*.json")),
             ROOT / "configs/phase3-review-public.pem"]
    # Aggregation, scoring and publishing changes can be replayed from raw measurements.
    files = [p for p in files if p.name not in {"phase3_verify.py", "phase3_score.py", "phase3_checkpoint.py"}]
    return hashlib.sha256("".join(sha(p) for p in files).encode()).hexdigest()


def valid(batch):
    output = BASE / batch
    manifest = output / "done.json"
    if not manifest.exists():
        return False
    d = read(manifest)
    return d["protocol"] == protocol_id() and d["batch"] == batch and bool(d["files"]) and all(
        (output / p).is_file() and sha(output / p) == h for p, h in d["files"].items())


def run_module(module, *args):
    subprocess.run([sys.executable, "-m", module, *args], check=True)


def save_review(output):
    private = WORK / "private"
    if not private.is_dir():
        return
    data = WORK / "review.tar.gz"
    with tarfile.open(data, "w:gz") as arc:
        arc.add(private, arcname="private")
    subprocess.run(["openssl", "cms", "-encrypt", "-binary", "-aes-256-cbc", "-outform", "DER",
                    "-in", str(data), "-out", str(output / "review.cms"), str(ROOT / "configs/phase3-review-public.pem")], check=True)


def run_batch(batch):
    if valid(batch):
        print("Reusing completed batch", batch)
        return
    out = BASE / batch
    out.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    # Every CI batch has its own checkout. A local replay can reuse the same work directory.
    if (WORK / "private").exists():
        shutil.move(WORK / "private", WORK / ("private-before-" + batch + "-" + str(os.getpid())))
    if batch == "tests":
        test = subprocess.run([sys.executable, "-m", "pytest", "-q"], text=True, capture_output=True)
        (out / "pytest.txt").write_text(test.stdout + test.stderr)
        print(test.stdout)
        test.check_returncode()
    elif batch == "probe":
        shutil.copy2(ROOT / "configs/meetai.json", WORK / "selected.json")
        run_module("bench.phase3_probe")
        shutil.copy2(WORK / "probe.json", out / "probe.json")
        save_review(out)
    elif batch.startswith("size-"):
        tier = batch.removeprefix("size-")
        for system in SYSTEMS:
            run_module("bench.phase3_scale", system, "--tier", tier)
            shutil.copy2(WORK / f"scale-{system}-{tier}.json", out / f"{system}.json")
    else:
        for system in SYSTEMS:
            run_module("bench.phase3_run", system, "--meeting", batch)
            shutil.copy2(WORK / f"{system}.json", out / f"{system}.json")
        save_review(out)
    metadata = {
        "batch": batch, "protocol": protocol_id(), "runner": platform.platform(),
        "cpu": next(s.split(":",1)[1].strip() for s in Path("/proc/cpuinfo").read_text().splitlines() if s.startswith("model name")),
        "logical_cpus": len(os.sched_getaffinity(0)), "ram": Path("/proc/meminfo").read_text().splitlines()[0],
        "workflow_run": os.getenv("GITHUB_RUN_ID"), "source_commit": os.getenv("GITHUB_SHA"),
        "files": {p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!="done.json"},
    }
    write(out / "done.json", metadata)


def gather(dest):
    missing = [s for s in BATCHES if not valid(s)]
    if missing:
        raise ValueError("Missing or invalid batches: " + ", ".join(missing))
    for system in SYSTEMS:
        runs = {k:read(BASE / k / (system + ".json")) for k in [*QA, "bank"]}
        first = next(iter(runs.values()))
        combined = {k:first[k] for k in ("system","config_sha256","cold_passes","warm_passes")}
        combined.update(runtime={"pairing":"same runner for both systems within every meeting",
                                 "batches":{k:r["runtime"] for k,r in runs.items()}},
                        ingestion={k:v for r in runs.values() for k,v in r["ingestion"].items()},
                        queries=[q for r in runs.values() for q in r["queries"]],
                        budget_sweep=[q for r in runs.values() for q in r["budget_sweep"]],
                        startup_by_batch_s={k:r["startup_s"] for k,r in runs.items()},
                        peak_rss_mib=max(r["peak_rss_mib"] for r in runs.values()))
        write(dest / (system + ".json"), combined)
        write(dest / ("scale-" + system + ".json"), [q for tier in SIZES for q in read(BASE / ("size-"+tier) / (system+".json"))])
    shutil.copy2(BASE / "probe/probe.json", dest / "probe.json")
    write(dest / "batch-index.json",{b:read(BASE / b / "done.json") for b in BATCHES})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--batch", choices=BATCHES)
    g.add_argument("--pending", action="store_true")
    g.add_argument("--gather", type=Path)
    args = p.parse_args()
    if args.pending:
        print(json.dumps([b for b in BATCHES if not valid(b)]))
    elif args.gather:
        args.gather.mkdir(parents=True,exist_ok=True)
        gather(args.gather)
    else:
        run_batch(args.batch)


if __name__ == "__main__":
    main()
