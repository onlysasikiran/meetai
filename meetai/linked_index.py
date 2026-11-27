"""Contiguous topic blocks and bounded, nonlocal passage links for one transcript."""

import json
import re
from collections import Counter, defaultdict
from math import log

import numpy as np

from .retrieval import Index, digest, words

STOP = set(
    """
a about after all am an and any are as at be been before by can could did do does
for from give had has have how i if in into is it its me meeting of on or our
should so some tell that the their them then there these they this to us was we
were what when where which who whom why will with would you your
""".split()
)
NAME = re.compile(r"(?<![\w'-])[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*")
ALIAS = re.compile(
    r"\b(?:alias|aka|a\.k\.a\.|also (?:known|called|referred to) as|code[- ]?nam(?:e|ed)|internal(?:ly)? (?:name|called|known)|short for|shorthand for|refers? to)\b",
    re.I,
)
ALIAS_UNCERTAIN = re.compile(
    r"\b(?:not|never|might|could|no longer|draft|wrong|previously|formerly|propos\w*|suggest\w*|reject\w*|declin\w*)\b",
    re.I,
)
NAME_STOP = STOP | set(
    "approved confirmation committee draft friday january february march april may june july august september october november december monday tuesday wednesday thursday saturday sunday project team originally finally please confirmed correction decision action reminder update".split()
)


def names(text):
    result = set()
    for match in NAME.finditer(text):
        # Drop sentence starters and possessives, preserving complete names such as Orion West.
        parts = [p.casefold().removesuffix("'s").removesuffix("’s") for p in match[0].split()]
        while parts and parts[0] in NAME_STOP:
            parts.pop(0)
        if parts:
            result.add(" ".join(parts))
    return result


def content_words(text):
    return {w for w in words(text) if len(w) > 1 and w not in STOP}


def unit(vector):
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


class LinkedIndex(Index):
    @classmethod
    def build(cls, path, models, cfg, cache=None):
        from time import perf_counter

        start = perf_counter()
        index = super().build(path, models, cfg, cache)
        graph_options = {
            k: cfg[k]
            for k in (
                "linked_version",
                "block_min",
                "block_max",
                "topic_boundary",
                "semantic_floor",
                "link_degree",
            )
        }
        index.key = digest(json.dumps([index.key, graph_options], sort_keys=True).encode())
        row = None
        if cache:
            cache.db.execute(
                "CREATE TABLE IF NOT EXISTS meeting_graphs (key TEXT PRIMARY KEY, data TEXT)"
            )
            row = cache.db.execute(
                "SELECT data FROM meeting_graphs WHERE key=?", (index.key,)
            ).fetchone()
        if row:
            data = json.loads(row[0])
        else:
            data = index._organize(cfg)
            if cache:
                with cache.db:
                    cache.db.execute(
                        "INSERT OR REPLACE INTO meeting_graphs VALUES(?,?)",
                        (index.key, json.dumps(data)),
                    )
        index.blocks, index.links = data["blocks"], data["links"]
        index.aliases = data["aliases"]
        index.alias_targets = defaultdict(set)
        for left, right, _ in index.aliases:
            index.alias_targets[left].add(right)
            index.alias_targets[right].add(left)
        index.chunk_block = [b for b, (lo, hi) in enumerate(index.blocks) for _ in range(lo, hi)]
        index.line_chunk = [c for c, chunk in enumerate(index.chunks) for _ in chunk]
        index.block_vectors = np.asarray(
            [unit(index.vectors[lo:hi].mean(axis=0)) for lo, hi in index.blocks]
        )
        index.by_id = {line.id: i for i, line in enumerate(index.lines)}
        index.entities = set(data["entities"])
        index.ingest.update(
            graph_cache_hit=bool(row),
            blocks=len(index.blocks),
            links=sum(map(len, index.links)),
            seconds=perf_counter() - start,
        )
        return index

    def _organize(self, cfg):
        n = len(self.chunks)
        blocks, start = [], 0
        for i in range(1, n):
            size = i - start
            if size >= cfg["block_max"] or (
                size >= cfg["block_min"]
                and self.vectors[i - 1] @ self.vectors[i] < cfg["topic_boundary"]
            ):
                blocks.append([start, i])
                start = i
        blocks.append([start, n])

        entities, aliases = set(), []
        for i, line in enumerate(self.lines):
            found = names(line.text)
            entities.update(found)
            if len(found) == 2 and ALIAS.search(line.text) and not ALIAS_UNCERTAIN.search(line.text):
                left, right = sorted(found)
                aliases.append([left, right, i])

        # Fixed projection orderings supply semantic candidates even with no shared token.
        # Only a small window in each ordering is visited. No all-pairs matrix is stored.
        rng = np.random.default_rng(42)
        projections = self.vectors @ rng.standard_normal((self.vectors.shape[1], 6)).astype(np.float32)
        order = np.argsort(projections, axis=0, kind="stable")
        pos = np.argsort(order, axis=0, kind="stable")
        links = []
        for i, passage in enumerate(self.passages):
            overlap = Counter()
            terms = sorted(content_words(passage), key=lambda t: (len(self.postings[t]), t))
            for term in terms[:12]:
                hits = self.postings[term]
                if 1 < len(hits) <= min(24, max(3, n // 3)):
                    for j, _ in hits:
                        if abs(i - j) > 1:
                            overlap[j] += log((n + 1) / len(hits))
            candidates = set(overlap)
            for axis, p in enumerate(pos[i]):
                candidates.update(int(j) for j in order[max(0, p - 2) : p + 3, axis] if abs(i - j) > 1)
            ranked = []
            for j in candidates:
                cosine = float(self.vectors[i] @ self.vectors[j])
                if j in overlap or cosine >= cfg["semantic_floor"]:
                    weight = cosine + min(overlap[j], 12) / 12
                    ranked.append((-weight, j, "shared_term" if j in overlap else "semantic"))
            links.append([[j, kind] for _, j, kind in sorted(ranked)[: cfg["link_degree"]]])
        return {"blocks": blocks, "links": links, "aliases": aliases, "entities": sorted(entities)}

    def focus(self, question):
        tokens = words(question)
        found = set()
        i = 0
        while i < len(tokens):
            for stop in range(min(i + 8, len(tokens)), i, -1):
                candidate = " ".join(tokens[i:stop])
                if candidate in self.entities:
                    found.add(candidate)
                    i = stop
                    break
            else:
                i += 1
        return found

    def bridges(self, focus):
        return [
            (right if left in focus else left, row)
            for left, right, row in self.aliases
            if (left in focus or right in focus)
            and len(self.alias_targets[left]) == len(self.alias_targets[right]) == 1
        ]
