"""Route a question, expand a bounded candidate set, and pack original turns."""

import re
from itertools import zip_longest
from math import log
from time import perf_counter

import numpy as np

from .linked_index import content_words, names

FACETS = {
    "owner": r"\b(?:owner|owned|owns?|accountable|responsib\w*|assign\w*|take[sn]? over|handed|lead)\b",
    "deadline": r"\b(?:deadline|date|due|when|schedul\w*|postpon\w*)\b",
    "dependency": r"\b(?:depends?|dependenc\w*|require\w*|prerequisit\w*|clear\w*|gate|blocked by)\b",
    "decision": r"\b(?:decid\w*|decision\w*|approv\w*|agreed|adopt\w*)\b",
}
FACETS = {name: re.compile(pattern, re.I) for name, pattern in FACETS.items()}
CURRENT = re.compile(r"\b(?:now|current\w*|latest|final|revis\w*|instead|ultimat\w*)\b", re.I)
HISTORY = re.compile(r"\b(?:initial\w*|original\w*|previous\w*|formerly|at first)\b", re.I)
EVOLUTION = re.compile(
    r"\b(?:evol\w*|timeline|chronolog\w*|how (?:did|has|have).{0,55}chang\w*)\b", re.I
)
TOPIC = re.compile(r"\b(?:summari[sz]\w*|summary|overview|discuss\w*|main (?:points|topics))\b", re.I)
REVISION = re.compile(
    r"\b(?:approved|confirm\w*|agreed|revis\w*|transfer\w*|handover|handed|instead|now|replace\w*|moved|no longer|take[sn]? over)\b",
    re.I,
)
DRAFT = re.compile(
    r"\b(?:draft|propos\w*|declin\w*|reject\w*|tentative|suggest\w*|hypothetical)\b", re.I
)


def interleave(*lists):
    return list(dict.fromkeys(x for row in zip_longest(*lists) for x in row if x is not None))


def pack_rows(index, priority, models, budget):
    chosen = []
    seen = set()
    for i in dict.fromkeys(priority):
        line = index.lines[i]
        key = (line.speaker, " ".join(line.text.casefold().split()))
        if key in seen:
            continue
        trial = sorted([*chosen, i])
        text = "\n".join(index.lines[j].render() for j in trial)
        if len(models.tokens(text)) <= budget:
            seen.add(key)
            chosen = trial
    return chosen


def near_name(line_names, focus):
    return not line_names.intersection(focus) and any(
        f" {target} " in f" {other} " for target in focus for other in line_names
    )


def retrieve(index, question, models, cfg):
    start = perf_counter()
    vector = models.dense.run(["Represent this sentence for searching relevant passages: " + question])[
        0
    ]
    seeds, times = index.rank(question, models, cfg, query_vector=vector)
    times["retrieval_s"] += perf_counter() - start - sum(times.values())
    start = perf_counter()
    seeds = seeds[: cfg["top_k"]]
    facets = {key for key, pattern in FACETS.items() if pattern.search(question)}
    focus = index.focus(question)
    bridges = index.bridges(focus)
    alias_names = {name for name, _ in bridges}
    seed_rows = [i for c in seeds for i in index.chunks[c]]
    query_terms = content_words(question)
    idf = {
        w: 1 + log((len(index.chunks) + 1) / (1 + len(index.postings.get(w, ())))) for w in query_terms
    }
    required = {w for w in query_terms if w in index.postings}
    initial = [index.lines[i].text for i in seed_rows]
    gaps = []
    if not any(
        (not focus or focus & names(text))
        and required <= content_words(text)
        and all(FACETS[f].search(text) for f in facets)
        for text in initial
    ):
        gaps.append("incomplete_seed")
    if any(w not in index.postings for w in query_terms):
        gaps.append("unmatched_query_term")
    if bridges and "dependency" in facets:
        gaps.append("linked_reference")
    route = "fact"
    if EVOLUTION.search(question):
        route = "timeline"
    elif HISTORY.search(question):
        route = "historical"
    elif CURRENT.search(question):
        route = "current"
    elif TOPIC.search(question):
        route = "topic"
    expand = bool(gaps) or route != "fact" or len(facets) > 1
    upper = cfg["budget"] if expand else min(cfg["simple_budget"], cfg["budget"])
    candidates = list(seeds)
    bridge_rows = []
    if expand:
        blocks = np.argsort(-(index.block_vectors @ vector), kind="stable")[:2]
        block_children = []
        for b in blocks:
            lo, hi = index.blocks[b]
            block_children.append(
                sorted(range(lo, hi), key=lambda c: (-float(index.vectors[c] @ vector), c))
            )
        parent = []
        for c in seeds:
            lo, hi = index.blocks[index.chunk_block[c]]
            parent.append(sorted(range(lo, hi), key=lambda j: (abs(c - j), j)))
        neighbors = interleave(*[[j for j, _ in index.links[c]] for c in seeds])
        bridge_rows = [i for _, i in bridges]
        linked = []
        for name in sorted(alias_names):
            terms = name.split()
            hits = min((index.postings.get(w, []) for w in terms), key=len)
            linked.extend(c for c, _ in hits if name in index.passages[c].casefold())
        linked = sorted(set(linked), key=lambda c: (-float(index.vectors[c] @ vector), c))
        remote = interleave(
            [index.line_chunk[i] for i in bridge_rows],
            linked,
            neighbors,
            interleave(*block_children),
            interleave(*parent),
        )
        candidates = list(dict.fromkeys(seeds + remote))[: cfg["expanded_chunks"]]

    info = {}
    for c in candidates:
        for i in index.chunks[c]:
            text = index.lines[i].text
            terms, entities = content_words(text), names(text)
            if near_name(entities, focus) and not entities.intersection(alias_names):
                continue
            matched = {f for f in facets if FACETS[f].search(text)}
            direct = bool(focus & entities)
            aliased = bool(alias_names & entities)
            weight = sum(idf[w] for w in terms & query_terms) / max(1, sum(idf.values()))
            weight += 0.42 * direct + 0.28 * aliased + 0.20 * len(matched)
            if route in {"current", "historical", "timeline"}:
                if route != "historical":
                    weight += 0.35 * bool(REVISION.search(text)) - 0.45 * bool(DRAFT.search(text))
                if route != "timeline":
                    weight += (0.50 if route == "historical" else -0.45) * bool(HISTORY.search(text))
            info[i] = (weight, matched, direct, aliased)
    ordered = sorted(info, key=lambda i: (-info[i][0], i))
    shortlist = list(dict.fromkeys([i for i in bridge_rows if i in info] + ordered))[: cfg["turn_pool"]]
    times["expansion_s"] = perf_counter() - start
    start = perf_counter()
    if shortlist:
        scores = models.cross.run([(question, index.lines[i].render()) for i in shortlist], pair=True)
        scores = 1 / (1 + np.exp(-np.clip(scores, -30, 30)))
        combined = {i: info[i][0] + float(s) for i, s in zip(shortlist, scores)}
        shortlist.sort(key=lambda i: (-combined[i], i))
    reranked_count = len(shortlist)
    times["turn_rerank_s"] = perf_counter() - start
    start = perf_counter()
    priority = []
    if alias_names and "dependency" in facets:
        priority.extend(i for i in shortlist if i in bridge_rows)
    for f in sorted(facets):
        pool = [i for i in shortlist if f in info[i][1]]
        scoped = [i for i in pool if info[i][2] or info[i][3]]
        pool = scoped or pool
        if route in {"current", "timeline"}:
            changes = [
                i
                for i in pool
                if REVISION.search(index.lines[i].text)
                and not DRAFT.search(index.lines[i].text)
                and not HISTORY.search(index.lines[i].text)
            ]
            if changes:
                priority.append(max(changes))
        if pool:
            priority.append(pool[0])
    if focus and route != "topic":
        scoped = [i for i in shortlist if info[i][2] or info[i][3]]
        unscoped = [i for i in shortlist if i not in scoped]
        if route == "current":
            # Withhold an explicitly historical row only when its requested fields have
            # later in-scope evidence. Keep unresolved parts of a partially revised plan.
            resolved = {
                f: max(
                    (
                        index.lines[j].turn
                        for j in scoped
                        if f in info[j][1]
                        and REVISION.search(index.lines[j].text)
                        and not DRAFT.search(index.lines[j].text)
                        and not HISTORY.search(index.lines[j].text)
                    ),
                    default=0,
                )
                for f in facets
            }
            scoped = [
                i
                for i in scoped
                if not (
                    HISTORY.search(index.lines[i].text)
                    and info[i][1]
                    and all(resolved[f] > index.lines[i].turn for f in info[i][1])
                )
            ]
        if scoped:
            priority = [i for i in priority if i in scoped]
            relevant = [i for i in scoped if info[i][1] or i in bridge_rows]
            if "linked_reference" in gaps:
                relevant = [i for i in relevant if i in bridge_rows or "dependency" in info[i][1]]
            anchors = relevant or scoped
            adjacent = [i for i in unscoped if any(abs(i - j) == 1 for j in anchors)]
            shortlist = anchors + adjacent[:2]
    if route == "topic":
        by_block = {}
        for i in shortlist:
            by_block.setdefault(index.chunk_block[index.line_chunk[i]], []).append(i)
        priority += interleave(*by_block.values())
    else:
        priority += shortlist
    # Stop once the requested local evidence is covered. Broad topic queries may use the full shortlist.
    cap = len(shortlist) if route == "topic" else (12 if expand else 4)
    choices = list(dict.fromkeys(priority))[:cap]
    rows = pack_rows(index, choices, models, upper)
    context = "\n".join(index.lines[i].render() for i in rows)
    trace = {
        "route": route,
        "expanded": expand,
        "gaps": gaps,
        "facets": sorted(facets),
        "focus": sorted(focus),
        "seed_chunks": seeds,
        "candidate_chunks": candidates,
        "blocks": [int(b) for b in sorted({index.chunk_block[c] for c in candidates})],
        "reranked_turns": reranked_count,
        "budget": upper,
    }
    times["assembly_s"] = perf_counter() - start
    return context, [index.lines[i].id for i in rows], times, trace
