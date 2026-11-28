"""Restore completed blinded-review packets without rerunning either RAG system."""
import argparse
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

from .phase3_checkpoint import BASE, BATCHES, WORK, valid


def restore(key, destination):
    count = 0
    destination.mkdir(parents=True, exist_ok=True)
    for batch in BATCHES:
        envelope = BASE / batch / "review.cms"
        if not envelope.is_file():
            continue
        if not valid(batch):
            raise ValueError(f"Invalid checkpoint for {batch}")
        with tempfile.TemporaryFile() as plain:
            subprocess.run(["openssl","cms","-decrypt","-binary","-inform","DER","-in",str(envelope),
                            "-inkey",str(key)],stdout=plain,check=True)
            plain.seek(0)
            with tarfile.open(fileobj=plain,mode="r:gz") as archive:
                for member in archive.getmembers():
                    p = Path(member.name)
                    if member.isdir():
                        continue
                    if (not member.isfile() or len(p.parts) != 3 or p.parts[0] != "private"
                            or ".." in p.parts or p.suffix != ".json"):
                        raise ValueError("Unexpected item in review packet")
                    content = archive.extractfile(member).read()
                    json.loads(content)
                    file = destination / p.parts[1] / p.parts[2]
                    if file.exists() and file.read_bytes() != content:
                        raise ValueError(f"Conflicting review record for {file.name}")
                    file.parent.mkdir(parents=True,exist_ok=True)
                    if not file.exists():
                        file.write_bytes(content)
                    count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=WORK / "private")
    args=parser.parse_args()
    print("Reviewed source records restored:", restore(args.key,args.out))
