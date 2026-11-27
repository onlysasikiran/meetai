# Phase 2 · MeetAI implementation

**Status:** Implemented and tested on a hosted Linux CPU. The frozen Phase 1 baseline and its scores are available in the same repository. [Phase 1 checkpoint report](https://github.com/onlysasikiran/meetai/blob/a5f555a608325cfbb8716f806c9e23933136fa94/PHASE_REPORT.md)

## Problem and completed work

A relevant chunk can contain a former owner, a similar project name or only half of an alias relationship. MeetAI reconstructs the evidence needed to answer one question and retains the original speaker-turn citations.

| Capability | Implementation |
| --- | --- |
| Hierarchy and links | Timestamp-aware turns and fragments form 160-token children. Adjacent children form topic blocks with centroid vectors. Each child has up to five nonlocal links selected using rare terms, projection neighborhoods and cosine similarity. Explicit alias declarations retain their source turn. |
| Adaptive retrieval | Start with the baseline hybrid search and reranker. Inspect missing terms, requested fields, temporal language and topic scope. Expand at most once into parent blocks, linked children and alias references. Rerank up to 36 turns across at most 12 candidate children. |
| Evidence assembly | Prioritize coverage of requested fields. For current-state queries, withhold explicitly historical evidence when later in-scope updates cover its requested fields. Deduplicate and render original turns in transcript order, with a 320-token simple or 1,024-token expanded ceiling. |
| Grounding | Check delivered IDs, numbers, named subjects, explicit date replacements and cited alias paths. A pinned local DeBERTa entailment classifier can reject unsupported claims. Later conflicting evidence is checked for scoped current-state questions. |
| Summaries and extraction | Topic questions use linked retrieval. Whole-meeting summaries select turns across blocks and reduce adjacent evidence packets using original source IDs. Pydantic and constrained JSON decoding produce decisions, actions, commitments, open issues and follow-ups. Each accepted owner and deadline has its own citations. |
| Persistence | SQLite stores embeddings, graph metadata and exact answers. Keys include content and configuration. A reopened index can answer a repeated question with zero generation calls. |

**Decisions.** Kept the Phase 1 Qwen weights, tokenizer, QA prompt, decoding settings and citation format. Graph construction uses existing embeddings and source text. A separate local classifier checks semantic support without another generative call. Routing rules, graph settings and the 0.60 entailment threshold are implementation defaults for Phase 3 to evaluate. [Configuration](configs/meetai.json) · [Research and dependencies](SOURCES.md)

## Executed results

**30 pytest tests passed. All 13 functional checks passed with the pinned CPU models.** The acceptance checks exercise interfaces and named expected outcomes. They are separate from the six-metric benchmark. [Per-case outputs, evidence IDs and retrieval traces](results/phase2-functional.json)

| Functional case | Observed result |
| --- | --- |
| Revised owner and deadline | On the 190-turn Juniper development fixture, returned Marek at T145 and May 9 at T170 using 94 evidence tokens. The historical query independently returned Sana at T8. |
| Distant alias and similar project | Returned Cedar. T23 establishes the internal name and T94 establishes the gate. The generator received 60 evidence tokens. |
| Missing information | Abstained on an unrecorded insurance premium. |
| Input formats and cache | Returned Priya at T3 from TXT, Markdown and text-layer PDF. Reopening the PDF index and repeating the question used zero generation calls. |
| Whole-meeting and topic summaries | The Riverside summary identified the launch decision, checklist and open quote. The topic query cited T4. A forced 45-token budget exercised two evidence packets and a final consolidation in three generation calls. |
| Typed minutes | Returned the approved decision, a checklist action assigned to Pablo and due May 14, an open vendor quote and its follow-up. The owner's and deadline's evidence both resolve to T2. |
| Upstream adapter | On ELITR development case `el-001-3`, the pipeline delivered 946 evidence tokens and abstained on an answerable question. This checks execution and exposes a real coverage limitation. |

The pytest suite also covers conditional expansion, semantic edges with disjoint wording, graph invalidation, ambiguous and negated aliases, conflicting dates, missing or mismatched citations, transcript fragments, timestamps, typed field rejection and multi-level summary reduction.

## Reproduction

```bash
bash scripts/setup.sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m bench.phase2_smoke
```

The smoke run writes a fresh report to `.work/phase2-functional.json`. Setup fetches the source assets and models with SHA-256 verification. Inference and these checks then run offline. The examples and runnable CLI commands are in the [README](README.md).

## Limitations

English naming and revision cues can miss implicit aliases, pronouns and informal corrections. Entailment scores can reject valid paraphrases or accept a plausible error. They assess the retrieved evidence and cannot detect an update the retriever missed. Timestamps are preserved for supported leading time formats, and ordering follows the supplied transcript. Global consolidation can omit details. Structured extraction returns up to eight records per final response, and an incomplete JSON response yields an explicit validation failure. PDF input requires a text layer.

The synthetic examples are development cases. The official ELITR and QMSum test splits, massive-context targets, matched latency and quality comparisons, ablations and target-laptop measurements remain for Phase 3.

## Phase 3 handoff

Start from this checkpoint. Keep the published baseline configuration, model bytes, evidence IDs, data splits and scoring rubric fixed. Compare both systems on the same hardware, with the same cache regimes. Track false abstentions and trace evidence loss before tuning. Profile the added turn reranking and verifier separately, then run the planned component ablations and held-out evaluation. [Benchmark protocol and six metrics](BENCHMARK.md)
