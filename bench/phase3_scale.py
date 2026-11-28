"""Measure real tokenizer tiers, independent turn count, and text-layer input parity."""
import argparse
import json
import resource
import tempfile
from pathlib import Path
from time import perf_counter

from meetai.models import Models
from meetai.pipeline import answer
from meetai.retrieval import Cache, Index, digest
from meetai.text import read_turns

from .data import ROOT, WORK, save_transcript
from .metrics import evidence_score, fingerprint
from .phase3_data import PROTOCOL, scenario
from .phase3_dev import OUT, config
from .run import write


def pdf_transcript(path, lines):
    # A simple, multi-page, text-layer PDF. Each source turn fits on one landscape line.
    objects = []
    def add(data):
        objects.append(data.encode("ascii"))
        return len(objects)
    add("<< /Type /Catalog /Pages 2 0 R >>")
    add("")
    font = add("<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>")
    pages = []
    for start in range(0, len(lines), 44):
        commands = ["BT /F1 8 Tf 11 TL 18 575 Td"]
        for line in lines[start:start+44]:
            escaped = line.replace("\\","\\\\").replace("(", "\\(").replace(")", "\\)")
            commands.append(f"({escaped}) Tj T*")
        stream = "\n".join(commands + ["ET",""])
        content = add(f"<< /Length {len(stream)} >>\nstream\n{stream}endstream")
        pages.append(add(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 900 600] /Resources << /Font << /F1 {font} 0 R >> >> /Contents {content} 0 R >>"))
    objects[1] = f"<< /Type /Pages /Count {len(pages)} /Kids [{' '.join(f'{p} 0 R' for p in pages)}] >>".encode()
    buf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, obj in enumerate(objects,1):
        offsets.append(len(buf))
        buf.extend(f"{n} 0 obj\n".encode()+obj+b"\nendobj\n")
    xref = len(buf)
    buf.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode())
    for offset in offsets:
        buf.extend(f"{offset:010d} 00000 n \n".encode())
    buf.extend(f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    path.write_bytes(buf)


def tier(models, target=None, turns=None):
    n = turns or round(target / 23)
    for _ in range(8):
        rows, cases = scenario(81037, n)
        count = len(models.tokens("\n".join(f"{s}: {text}" for s,text in rows)))
        if turns or target <= count <= target + max(64,target*.001):
            return rows, cases, count
        step = round((target + 16 - count) * n / count)
        n += step or (1 if count < target else -1)
    raise ValueError("Could not calibrate tokenizer tier")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("system",choices=("baseline","meetai"))
    system = ap.parse_args().system
    cfg = config(system)
    models = Models(ROOT / ".assets/models", cfg["threads"], cfg["window"])
    if system == "meetai":
        _ = models.nli
    results = []
    for target in [*PROTOCOL["stress_targets"], "turns"]:
        rows, cases, count = (tier(models,turns=PROTOCOL["separate_turns"]) if target == "turns" else tier(models,target=target))
        txt = save_transcript("scale-" + str(target), rows)
        paths = [txt]
        if target == 20000:
            md, pdf = txt.with_suffix(".md"), txt.with_suffix(".pdf")
            md.write_text("# Dispatch meeting\n\n"+"\n".join(f"- **{s}:** {text}" for s,text in rows)+"\n")
            pdf_transcript(pdf, txt.read_text().splitlines())
            paths += [md,pdf]
        outputs = {}
        for path in paths:
            # Parse separately to test complete format parity. Its cost stays outside the ingest clock.
            assert read_turns(path) == rows
            with tempfile.TemporaryDirectory(dir=WORK) as temp:
                cache = Cache(Path(temp) / "scale.sqlite")
                try:
                    idx = Index.build(path, models, cfg, cache)
                    assert not idx.ingest["cache_hit"] and idx.tokens == count and len(idx.turn_parts) == len(rows)
                    record = {"target":target,"format":path.suffix,"ingestion":idx.ingest,"source_sha256":digest(path.read_bytes()),"qa":[]}
                    for q in (cases[2], cases[3]):
                        t = perf_counter()
                        pred = answer(idx, q["question"], models, cfg, cache)
                        elapsed = perf_counter() - t
                        assert len(pred["ids"]) == len(set(pred["ids"])) and pred["context_tokens"] <= cfg["budget"] and not pred["cache_hit"]
                        fprint = fingerprint(pred["claims"])
                        signature = (fprint, tuple(pred["ids"]))
                        if path == txt:
                            outputs[q["id"]] = signature
                        elif outputs[q["id"]] != signature:
                            raise ValueError("Format changed the answer or the delivered IDs")
                        record["qa"].append({"id":q["id"],"gold":q["facts"],"gold_evidence":q["evidence"],
                            "recall":evidence_score(q["evidence"],pred["complete_turns"]),"claims":pred["claims"],
                            "seconds":elapsed,"stage_s":pred["timings"],"tokens":pred["context_tokens"],"ids":pred["ids"],
                            "abstained":pred["abstained"],"answer_digest":fprint})
                    record["process_high_water_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                    results.append(record)
                    write(OUT / ("scale-" + system + ".json"), results)
                    print(system, target, path.suffix, count, len(rows), round(idx.ingest["seconds"],1), flush=True)
                finally:
                    cache.close()
            del idx
    assert len(results) == len(PROTOCOL["stress_targets"])+3


if __name__ == "__main__":
    main()
