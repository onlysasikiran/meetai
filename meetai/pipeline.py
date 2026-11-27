import json
from time import perf_counter

from .models import validate
from .retrieval import digest


def answer(index, question, models, cfg, cache=None, use_answer_cache=True):
    started = perf_counter()
    if not question.strip():
        raise ValueError("Question is empty")
    key = digest(json.dumps([index.key, cfg, question, "qa"], sort_keys=True).encode())
    if cache and use_answer_cache:
        row = cache.db.execute("SELECT data FROM answers WHERE key=?", (key,)).fetchone()
        if row:
            result = json.loads(row[0])
            result["cache_hit"] = True
            result["timings"] = {"total_s": perf_counter() - started}
            result["llm_calls"] = 0
            return result
    ranks, timings = index.rank(question, models, cfg)
    at = perf_counter()
    context, ids = index.pack(ranks, cfg, models)
    budget_used = len(models.tokens(context))
    timings["assembly_s"] = perf_counter() - at
    at = perf_counter()
    raw, usage = models.generate(question, context, cfg["max_output"])
    timings["generation_s"] = perf_counter() - at
    at = perf_counter()
    result = validate(raw, set(ids))
    result.update(
        {
            "ids": ids,
            "complete_turns": index.complete_turns(ids),
            "context_tokens": budget_used,
            "usage": usage,
            "llm_calls": 1,
            "cache_hit": False,
        }
    )
    timings["validation_s"] = perf_counter() - at
    result["timings"] = timings
    if cache and use_answer_cache:
        with cache.db:
            cache.db.execute("INSERT OR REPLACE INTO answers VALUES(?,?)", (key, json.dumps(result)))
    timings["total_s"] = perf_counter() - started
    return result


def summarize(index, models, cfg):
    """Conventional contiguous map/reduce; every retained citation survives each reduction."""
    prompt = "Summarize the key decisions, actions and outstanding issues across this meeting."
    budget = cfg["budget"]
    calls = 0

    def groups(rows):
        group = []
        for item in rows:
            candidate = group + [item]
            if group and len(models.tokens("\n".join(text for text, _ in candidate))) > budget:
                yield group
                group = []
            if len(models.tokens(item[0])) > budget:
                raise ValueError("A summary item exceeds the configured evidence budget")
            group.append(item)
        if group:
            yield group

    def summarize_group(batch):
        nonlocal calls
        allowed = {ref for _, refs in batch for ref in refs}
        context = "\n".join(text for text, _ in batch)
        raw, _ = models.generate(prompt, context, cfg["max_output"])
        calls += 1
        return validate(raw, allowed)["claims"]

    rows = [(line.render(), [line.id]) for line in index.lines]
    while rows:
        batches = list(groups(rows))
        claims = [c for batch in batches for c in summarize_group(batch)]
        if len(batches) == 1 or not claims:
            break
        rows = [(c["text"] + " " + " ".join(f"[T{r}]" for r in c["cites"]), c["cites"]) for c in claims]
        if len(list(groups(rows))) >= len(batches):
            raise ValueError("Summary reduction failed to shrink; increase the context budget")
    return {"claims": claims, "abstained": not bool(claims), "llm_calls": calls}
