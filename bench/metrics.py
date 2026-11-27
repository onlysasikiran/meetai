import hashlib
import json
from math import ceil
from statistics import mean


def fingerprint(claims):
    return hashlib.sha256(json.dumps(claims, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def evidence_score(groups, complete_turns):
    available = set(complete_turns)
    if not groups:
        return None
    return sum(any(set(option) <= available for option in group) for group in groups) / len(groups)


def quantile(values, fraction):
    return sorted(values)[max(0, ceil(len(values) * fraction) - 1)] if values else None


def divide(a, b):
    return a / b if b else None


def quality(row, labels):
    """atom = [1-based output line, matching gold unit or 0, source support, citation support]."""
    if labels["digest"] != row["answer_digest"]:
        raise ValueError(f"Prediction changed; a new claim review is required for {row['id']}")
    atoms = labels["atoms"]
    if {a[0] for a in atoms} != set(range(1, row["claim_lines"] + 1)):
        raise ValueError("Every released answer line requires an atom review")
    if any(
        len(a) != 4
        or not 0 <= a[1] <= row["gold_units"]
        or a[2] not in (0, 1)
        or a[3] not in (0, 1)
        or a[3] > a[2]
        or (a[1] and not a[2])
        for a in atoms
    ):
        raise ValueError("Invalid claim judgement")
    matched = len({a[1] for a in atoms if a[1]})
    gold = row["gold_units"]
    return {
        "correctness": (2 * matched / (len(atoms) + gold)) if gold else float(row["abstained"]),
        "atom_count": len(atoms),
        "unsupported": sum(1 - a[2] for a in atoms),
        "citations_supported": sum(a[3] for a in atoms),
        "matched": matched,
    }


def rollup(rows):
    recalls = [r["evidence_recall"] for r in rows if r["evidence_recall"] is not None]
    claims = sum(r["atom_count"] for r in rows)
    return {
        "questions": len(rows),
        "answer_correctness": mean(r["correctness"] for r in rows),
        "evidence_recall": mean(recalls) if recalls else None,
        "evidence_questions": len(recalls),
        "unsupported_claim_rate": divide(sum(r["unsupported"] for r in rows), claims),
        "unsupported": sum(r["unsupported"] for r in rows),
        "atomic_claims": claims,
        "citation_support_rate": divide(sum(r["citations_supported"] for r in rows), claims),
        "answered": sum(not r["abstained"] for r in rows),
        "cold_p95_s": quantile([s for r in rows for s in r["cold_repeats_s"]], 0.95),
        "warm_p95_s": quantile([s for r in rows for s in r["warm_s"]], 0.95),
    }
