# MeetAI · Meeting transcript RAG

**Phase 1 delivers a frozen local baseline and the benchmark that later versions must beat.** It reads one English PDF, Markdown or TXT transcript, answers questions with speaker-turn citations, and summarizes the meeting. It runs on CPU without API keys.

## Problem statement

A meeting can assign a task, change its owner, and revise the deadline hundreds of turns later. A search for the project name can retrieve the original plan, a correction, and a different project with a similar name. The system must return the current answer and cite the lines that establish it.

In one checkpoint meeting, the baseline returned the revised deadline but named the former owner. Both required pieces of evidence were in its context. In another case it correctly named the new owner but cited an unrelated calibration update. These failures motivate measuring retrieval, answer quality and citation support separately.

## Solution and architecture

Phase 1 implements the conventional RAG comparator. Phase 2 is scoped to hierarchical organization, links between distant passages, and adaptive retrieval. Its performance is an open question.

| Stage | Implemented baseline |
| --- | --- |
| Ingest | Parse text, retain speaker and source IDs, split long turns, group turns into 160-token child chunks. |
| Retrieve | Combine BM25 and BGE-small cosine rankings using reciprocal rank fusion. MiniLM scores up to 12 candidates. |
| Assemble | Select the top 3 children and expand within two-child parents. Deduplicate and sort evidence chronologically within 1,024 Qwen tokens. |
| Answer | Qwen3-4B Q4_K_M answers locally. A validator checks that cited IDs occurred in the delivered evidence. Whole-meeting summaries use contiguous map/reduce. |
| Reuse | SQLite stores indexes and exact answers under content and configuration hashes. |

## Measured results

| Checkpoint workload | Questions | Answer correctness, factual-unit F1 | Evidence recall | Unsupported claims |
| --- | ---: | ---: | ---: | ---: |
| ELITR-Bench, real QA | 8 | 63.3% | 62.5% | 3/13 |
| QMSum, real topic summaries | 4 | 0.0% | 6.25% | 1/14 |
| MeetAI-Stress, synthetic QA | 10 | 65.0% | 87.5% | 1/8 |

On QMSum the baseline returned information about the meetings but missed all scored reference facts. For example, an answer about microphone quality omitted the requested setup guidance about consent, digit reading and microphone placement. This four-question result establishes a failure to investigate.

The benchmark measured **25.28 s P95** for 44 requests that required fresh inference, **0.166 ms P95** for 110 exact cache hits, and **5.00 GiB** peak process RAM. Eight fresh indexes processed **190,037 source tokens at 7,357 tokens/s**. The largest input was an **84,887-token MeetingBank agenda segment**. Hardware was a six-thread AMD EPYC CPU run. Measurements on the target Ryzen laptop remain open.

## Project record

| File | Contents |
| --- | --- |
| [Phase report](PHASE_REPORT.md) | Completed work, tests, decisions, limitations and Phase 2 handoff |
| [Benchmark protocol](BENCHMARK.md) | Data splits, six metrics, claim-review rules and reproduction commands |
| [Research and dependencies](SOURCES.md) | Research decisions, upstream data terms, model sources and pinned assets |
| [Frozen settings](configs/baseline.json) | Winning retrieval settings and generator configuration |
| [Tuning trials](results/tuning.json) | All 24 configurations and the selection outcome |
| [Checkpoint measurements](results/checkpoint.json) | Per-question evidence, latency, token use and ingestion measurements |
| [Quality scores](results/scores.json) | Dataset and category results for the six metrics |
| [Claim reviews](results/claim_review.json) · [Validation](results/validation.json) | Scoring decisions, prediction hashes and evidence replay |

## Local entry points

Setup and reproduction scripts are provided for future developers. The published measurements were executed in the project workspace.

```bash
bash scripts/setup.sh
.venv/bin/python -m meetai examples/launch.pdf --ask "Who owns the launch now?"
.venv/bin/python -m meetai examples/launch.md --summary
```
