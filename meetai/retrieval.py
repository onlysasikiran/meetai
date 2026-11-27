import hashlib
import json
import re
import sqlite3
from collections import Counter
from dataclasses import asdict
from math import log
from time import perf_counter

import numpy as np

from .text import Line, make_chunks, read_turns, split_turns


def digest(value):
    return hashlib.sha256(value).hexdigest()


def words(text):
    return re.findall(r"\w+", text.casefold())


class Cache:
    def __init__(self, path):
        self.db = sqlite3.connect(path, timeout=30)
        self.db.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, data TEXT)")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS indexes (key TEXT PRIMARY KEY, meta TEXT, vectors BLOB)"
        )

    def close(self):
        self.db.close()


class Index:
    def __init__(self, lines, chunks, vectors):
        self.lines, self.chunks, self.vectors = lines, chunks, vectors
        self.passages = [
            "\n".join(f"{lines[i].speaker}: {lines[i].text}" for i in chunk) for chunk in chunks
        ]
        terms = [Counter(words(p)) for p in self.passages]
        self.lengths = np.asarray([sum(t.values()) for t in terms])
        self.avg = max(1.0, np.mean(self.lengths))
        self.postings = {}
        for idx, bag in enumerate(terms):
            for word, freq in bag.items():
                self.postings.setdefault(word, []).append((idx, freq))

    @classmethod
    def build(cls, path, models, cfg, cache=None):
        if cls is Index and cfg.get("system") == "meetai":
            from .linked_index import LinkedIndex

            return LinkedIndex.build(path, models, cfg, cache)
        start = perf_counter()
        raw = path.read_bytes()
        timed = cfg.get("system") == "meetai"
        key = digest(
            raw
            + (b"timestamp-parser-v1" if timed else b"")
            + repr(
                (path.suffix.lower(), cfg["chunk_tokens"], cfg["asset_id"], cfg["pipeline_version"])
            ).encode()
        )
        row = (
            cache.db.execute("SELECT meta,vectors FROM indexes WHERE key=?", (key,)).fetchone()
            if cache
            else None
        )
        if row:
            meta = json.loads(row[0])
            lines = [Line(**x) for x in meta["lines"]]
            index = cls(
                lines,
                meta["chunks"],
                np.frombuffer(row[1], dtype=np.float32).reshape(len(meta["chunks"]), -1),
            )
            count = meta["tokens"]
        else:
            turns = read_turns(path, timestamps=timed)
            count = len(models.tokens("\n".join(f"{s}: {t}" for s, t, *_ in turns)))
            lines = split_turns(turns, models.tokens, max(24, cfg["chunk_tokens"] - 16))
            chunks = make_chunks(lines, models.tokens, cfg["chunk_tokens"])
            passages = ["\n".join(f"{lines[i].speaker}: {lines[i].text}" for i in c) for c in chunks]
            vec = models.dense.run(passages)
            index = cls(lines, chunks, vec)
            if cache:
                meta = {"lines": [asdict(x) for x in lines], "chunks": chunks, "tokens": count}
                with cache.db:
                    cache.db.execute(
                        "INSERT OR REPLACE INTO indexes VALUES(?,?,?)",
                        (key, json.dumps(meta), vec.astype(np.float32).tobytes()),
                    )
        index.key, index.tokens = key, count
        index.turn_parts = {}
        for line in index.lines:
            index.turn_parts.setdefault(line.turn, set()).add(line.id)
        index.ingest = {
            "tokens": count,
            "seconds": perf_counter() - start,
            "cache_hit": bool(row),
            "turns": len(index.turn_parts),
            "chunks": len(index.chunks),
        }
        return index

    def rank(self, question, models, cfg, query_vector=None):
        start = perf_counter()
        scores = np.zeros(len(self.chunks), dtype=float)
        for word in sorted(set(words(question))):
            posting = self.postings.get(word, [])
            if not posting:
                continue
            idf = log(1 + (len(scores) - len(posting) + 0.5) / (len(posting) + 0.5))
            for idx, freq in posting:
                scores[idx] += (
                    idf * freq * 2.2 / (freq + 1.2 * (0.25 + 0.75 * self.lengths[idx] / self.avg))
                )
        if query_vector is None:
            query_vector = models.dense.run(
                ["Represent this sentence for searching relevant passages: " + question]
            )[0]
        dense = self.vectors @ query_vector
        pool = cfg["pool"]
        fused = np.zeros(len(scores))
        for values, sparse in ((scores, True), (dense, False)):
            order = np.argsort(-values, kind="stable")[:pool]
            if sparse:
                order = [i for i in order if values[i] > 0]
            for rank, i in enumerate(order, 1):
                fused[i] += 1 / (cfg["rrf_k"] + rank)
        candidates = np.argsort(-fused, kind="stable")[:pool]
        retrieved = perf_counter()
        rescores = models.cross.run([(question, self.passages[i]) for i in candidates], pair=True)
        ranked = candidates[np.argsort(-rescores, kind="stable")].tolist()
        return ranked, {"retrieval_s": retrieved - start, "rerank_s": perf_counter() - retrieved}

    def pack(self, ranks, cfg, models):
        selected, size = set(), 0
        group_size = cfg["parent_chunks"]
        for child in ranks[: cfg["top_k"]]:
            base = child // group_size * group_size
            neighbors = sorted(
                range(base, min(base + group_size, len(self.chunks))), key=lambda k: (abs(k - child), k)
            )
            for chunk in neighbors:
                for i in self.chunks[chunk]:
                    if i in selected:
                        continue
                    line_cost = len(models.tokens(self.lines[i].render() + "\n"))
                    if size + line_cost > cfg["budget"]:
                        continue
                    selected.add(i)
                    size += line_cost
        ordered = sorted(selected)
        render = lambda: "\n".join(self.lines[i].render() for i in ordered)
        text = render()
        # Final tokenizer pass includes newlines and token-boundary effects.
        while ordered and len(models.tokens(text)) > cfg["budget"]:
            ordered.pop()
            text = render()
        ids = [self.lines[i].id for i in ordered]
        return text, ids

    def complete_turns(self, ids):
        ids = set(ids)
        return [turn for turn, parts in self.turn_parts.items() if parts <= ids]
