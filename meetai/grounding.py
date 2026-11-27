"""Check a claim against the actual cited text. Classification can abstain in error."""

import re

from .linked_index import names
from .linked_retrieval import DRAFT, FACETS, REVISION, near_name
from .models import validate

NUMBER = re.compile(r"\b\d+(?:[.,:/-]\d+)*\b")


def hypothesis(text, question, focus=()):
    # A short name alone can be mentioned by an unrelated speaker. Test its proposed role.
    labeled = re.fullmatch(r"([^:]{1,55}):\s*(.{1,100})", text)
    if labeled and question:
        label, value = labeled.groups()
        scope = " for " + " and ".join(sorted(focus)) if focus else ""
        if FACETS["owner"].search(label):
            modifier = (
                "original" if re.search(r"\b(?:original|initial|former)", label, re.I) else "current"
            )
            role = "delivery owner" if "delivery" in (label + " " + question).casefold() else "owner"
            return f"{value} is the {modifier} {role}{scope}."
        if FACETS["deadline"].search(label):
            return f"The {label}{scope} is {value}."
    if len(text.split()) <= 3 and question:
        who = re.fullmatch(r"who\s+(.+?)\??", question.strip(), re.I)
        if who and not any(mark in text for mark in ".,!?"):
            return f"{text} {who[1]}."
        which = re.match(
            r"(?:which|what) ([a-z][a-z -]{0,38}?) (?:must|should|does|did|was|is|will|has|do)\b",
            question,
            re.I,
        )
        if which:
            slot = which[1].strip()
            value = re.sub(r"\s+" + re.escape(slot) + r"$", "", text, flags=re.I).strip() or text
            return f"The {slot} is {value}."
    return text


DATE = r"(?:[a-z]{3,9}\s+\d{1,2}(?:,?\s+\d{4})?|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}(?:/\d{2,4})?)"
DATE_CHANGE = re.compile(r"\bfrom (" + DATE + r") to (" + DATE + r")\b", re.I)


def verify_claims(claims, ids, index, models, cfg, question="", trace=None):
    trace = trace or {}
    delivered = {ref: index.lines[index.by_id[ref]] for ref in ids}
    accepted, rejected, work = [], [], []
    focus = set(trace.get("focus", ()))

    def reject(claim, reason):
        rejected.append({"text": claim["text"], "cites": claim["cites"], "reason": reason})

    for claim in claims:
        refs = list(dict.fromkeys(claim["cites"]))
        if not refs or any(r not in delivered for r in refs):
            reject(claim, "citation_outside_context")
            continue

        per_line_names = [names(delivered[r].speaker) | names(delivered[r].text) for r in refs]
        cited_names = set().union(*per_line_names)
        if any(near_name(entities, focus) for entities in per_line_names):
            reject(claim, "different_named_subject")
            continue
        if focus and not focus.intersection(cited_names):
            for alias, i in index.bridges(focus):
                bridge = index.lines[i].id
                if alias in cited_names and bridge in delivered and bridge not in refs:
                    refs.append(bridge)
                    break
            else:
                if trace.get("route") != "topic":
                    reject(claim, "missing_subject_evidence")
                    continue
        refs.sort(key=index.by_id.__getitem__)
        source = "\n".join(f"{delivered[r].speaker}: {delivered[r].text}" for r in refs)
        if not set(NUMBER.findall(claim["text"])) <= set(NUMBER.findall(source)):
            reject(claim, "number_missing_from_citations")
            continue
        statement = hypothesis(claim["text"], question, focus)
        later = []
        if trace.get("route") == "current" and focus:
            last = max(delivered[r].turn for r in refs)
            target = [
                p for key, p in FACETS.items() if key in trace.get("facets", []) and p.search(statement)
            ]
            later = [
                f"{line.speaker}: {line.text}"
                for line in delivered.values()
                if line.turn > last
                and focus & names(line.text)
                and REVISION.search(line.text)
                and not DRAFT.search(line.text)
                and any(p.search(line.text) for p in target)
            ]
        if trace.get("route") == "current" and "deadline" in trace.get("facets", []):
            revisions = [m for r in refs for m in DATE_CHANGE.finditer(delivered[r].text)]
            if any(
                m[1].casefold() in statement.casefold() and m[2].casefold() not in statement.casefold()
                for m in revisions
            ):
                reject(claim, "superseded_date")
                continue
        premises = [source]
        if len(refs) > 1:
            premises += [f"{delivered[r].speaker}: {delivered[r].text}" for r in refs]
        work.append(({"text": claim["text"], "cites": refs}, statement, premises, later))

    pairs = [(s, statement) for _, statement, premises, later in work for s in [*premises, *later]]
    scores = iter(models.support_scores(pairs) if pairs else [])
    for claim, _, premises, later in work:
        entailment = max(next(scores)[0] for _ in premises)
        conflicts = [next(scores)[1] for _ in later]
        if entailment < cfg["entailment_min"]:
            reject(claim, "insufficient_citation_support")
        elif any(score >= cfg["entailment_min"] for score in conflicts):
            reject(claim, "later_conflicting_evidence")
        else:
            accepted.append(claim)
    return accepted, rejected


def validated_answer(raw, ids, index, models, cfg, question="", trace=None):
    # Qwen can use suffix citations, or quote complete source lines with prefix IDs.
    lines = []
    for line in raw.strip().splitlines():
        line = re.sub(
            r"^\s*(?:[-*•]\s*)?(?:approved decisions?|current actions?|commitments?|outstanding issues?|follow[- ]ups?)\s*:\s*",
            "",
            line,
            flags=re.I,
        )
        if re.match(r"^\s*\[T", line):
            lines.extend(re.split(r"(?<=[.!?])\s+(?=\[T[0-9.]+\]\s+[A-Za-z])", line))
        else:
            lines.extend(
                re.sub(
                    r"(\[T[0-9.]+\])(?:\s*[,;.]\s*(?:and\s+)?|\s+and\s+)(?!and\s+\[T)(?=[A-Za-z])",
                    r"\1\n",
                    line,
                ).splitlines()
            )
    parsed = validate("\n".join(dict.fromkeys(lines)), set(ids))
    if not cfg.get("semantic_validation", True):
        return {**parsed, "validation": []}
    claims, checks = verify_claims(parsed["claims"], ids, index, models, cfg, question, trace)
    return {
        "claims": claims,
        "abstained": not claims,
        "rejected": parsed["rejected"] + [r["text"] for r in checks],
        "validation": checks,
    }
