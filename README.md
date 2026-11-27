# MeetAI · Meeting transcript RAG

MeetAI answers questions about one English meeting, summarizes discussions and extracts cited decisions and action items. It reads TXT, Markdown and text-layer PDF files and runs on CPU without an API key. **Phase 2 delivers the working MeetAI system alongside the frozen Phase 1 baseline.**

## Problem statement

A meeting can assign a task, change its owner, and revise the deadline hundreds of turns later. A search for the project name can retrieve the original plan, a correction, and a different project with a similar name. The system must return the current answer and cite the lines that establish it.

In one checkpoint meeting, the baseline returned the revised deadline but named the former owner. Both required pieces of evidence were in its context. In another case it correctly named the new owner but cited an unrelated calibration update. These failures motivate measuring retrieval, answer quality and citation support separately.

## Solution and architecture

Both systems use the same speaker turns, BGE-small embeddings, BM25, MiniLM reranker, Qwen3-4B generator and citation format.

| Stage | Standard baseline | MeetAI |
| --- | --- | --- |
| Organize | Fixed 160-token children with two-child parents | Group neighboring children into discussion blocks by embedding similarity. Build sparse cross-block links from shared terms and semantic neighbors. Record explicit, unambiguous aliases with source IDs. |
| Retrieve | Rerank 12 candidate passages and expand three children | Start with the same hybrid search. For missing evidence, revisions or topic questions, also inspect parent blocks, remote links and alias references. Rerank at most 36 source turns. |
| Assemble | Chronological evidence under 1,024 tokens | Prioritize complementary facts, filter similar project names, deduplicate and restore source order. The simple route has a 320-token ceiling and the expanded route has 1,024. |
| Validate | Check that cited turns were delivered | Also check numbers, named subjects and evidence for revisions. A local entailment model scores each claim against its citations. Unsupported claims can trigger abstention. |
| Summarize or extract | Contiguous map/reduce summaries | Select turns across discussion blocks, consolidate original citations hierarchically, and produce summaries or typed minutes with cited owners and deadlines. |
| Reuse | SQLite index and exact-answer cache | Also persist the hierarchy and links. The two systems have separate answer-cache keys. |

On the 190-turn Juniper development fixture, MeetAI returned **Marek** at turn 145 and **May 9** at turn 170. A separate question needed turn 23 to connect Juniper to its internal name and turn 94 to establish the **Cedar** gate. Both citations appeared in the answer. These are functional examples. The held-out comparison belongs to Phase 3. [Executed Phase 2 cases](results/phase2-functional.json)

## Datasets

| Dataset | Phase 1 use | Source |
| --- | --- | --- |
| ELITR-Bench | Factual meeting QA. English dev meetings 001 and 003 for tuning, 002 and 004 for the checkpoint. | [Questions and reference answers](https://github.com/utter-project/ELITR-Bench) · [Original transcripts](https://hdl.handle.net/11234/1-4692) |
| QMSum | Topic summaries and supporting turns. Training meetings Bed004 and ES2002a for tuning, validation meetings Bed002 and Bmr005 for the checkpoint. | [Dataset and annotations](https://github.com/Yale-LILY/QMSum) |
| MeetingBank | City council agenda segments from the validation split for ingestion measurements and summary testing. | [Dataset downloads](https://huggingface.co/datasets/huuuyeah/meetingbank) · [Project](https://meetingbank.github.io/) |
| MeetAI-Stress | Project-generated meetings with changed owners, revised dates, distant evidence and unanswerable questions. Scenarios 0–1 for tuning, 2–3 for the checkpoint. | [Generator and ground truth](bench/stress.py) |

The [asset manifest](configs/assets.json) records the original download URLs and SHA-256 checksums. The [benchmark protocol](BENCHMARK.md) lists the splits reserved for later evaluation. See [source and license notes](SOURCES.md) for reuse terms.

## Phase 1 baseline measurements

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
| [Phase report](PHASE_REPORT.md) | Phase 2 changes, functional tests, limitations and Phase 3 handoff |
| [MeetAI settings](configs/meetai.json) · [Functional results](results/phase2-functional.json) | Phase 2 settings and model outputs from the acceptance cases |
| [Benchmark protocol](BENCHMARK.md) | Data splits, six metrics, claim-review rules and reproduction commands |
| [Research and dependencies](SOURCES.md) | Research decisions, upstream data terms, model sources and pinned assets |
| [Frozen settings](configs/baseline.json) | Winning retrieval settings and generator configuration |
| [Tuning trials](results/tuning.json) | All 24 configurations and the selection outcome |
| [Checkpoint measurements](results/checkpoint.json) | Per-question evidence, latency, token use and ingestion measurements |
| [Quality scores](results/scores.json) | Dataset and category results for the six metrics |
| [Claim reviews](results/claim_review.json) · [Validation](results/validation.json) | Scoring decisions, prediction hashes and evidence replay |

## Local entry points

The CLI defaults to the frozen baseline for existing callers. Use `--system meetai` for linked retrieval. `--extract` selects MeetAI automatically.

```bash
bash scripts/setup.sh
.venv/bin/python -m meetai examples/launch.pdf --system meetai --ask "Who is the current launch owner?" --trace
.venv/bin/python -m meetai examples/riverside.txt --system meetai --summary
.venv/bin/python -m meetai examples/riverside.txt --extract
.venv/bin/python -m meetai examples/launch.md --system baseline --ask "Who is the current launch owner?"
```

Every released claim includes source IDs. The `evidence` object resolves them to the speaker, original text, turn number and timestamp when available. Extracted minutes can include decisions, actions, commitments, open issues and follow-ups. An absent or unsupported owner or deadline is returned as `null`.
