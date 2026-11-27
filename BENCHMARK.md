# Phase 1 benchmark protocol

## What is being measured

One English transcript per index. All neural inference runs in one local CPU process. The published checkpoint compares settings of a conventional flat RAG pipeline. The Phase 3 comparison will add the linked MeetAI pipeline.

| Split | ELITR-Bench | QMSum | MeetAI-Stress | MeetingBank |
| --- | --- | --- | --- | --- |
| Tune | English dev 001 and 003, 8 factual questions | Train Bed004 and ES2002a, 5 topic questions | Scenarios 0 and 1, 10 questions | — |
| Checkpoint | English dev 002 and 004, 8 factual questions | Validation Bed002 and Bmr005, 4 topic questions | Scenarios 2 and 3, 10 questions | First and longest-by-word-count validation agenda segments |
| Reserved for Phase 3 | Official English QA test2 | Official test | New meeting scenarios, wording and seeds | Test split |

The question roster, reference outlines and evidence groups live in [bench/data.py](bench/data.py). The synthetic generator records the source of each true fact independently of retrieval. MeetingBank supplies operational measurements and one ungraded summary. Each real QA dataset uses upstream human answers. ELITR source IDs count original transcript lines with inherited speaker identity. QMSum source IDs are the upstream utterance index plus one. Long utterances retain an ID for each fragment, for example T66.2. Some QMSum official spans omit material present elsewhere in the transcript. The checkpoint also stores recall over the entire official span as a separate diagnostic.

## Tuning and measurement

1. Fit on six tuning transcripts. Try two child sizes, two parent sizes, two retrieval depths and three evidence budgets. Indexes and reranker scores are reused within the sweep. The model never sees gold text. Two unanswerable questions have no gold evidence and are excluded from retrieval tuning.
2. Average evidence-group recall within each dataset, then give the three datasets equal weight. Select the smallest budget within two percentage points of the best result. The remaining tie-breakers are specified in [the protocol](configs/protocol.json). Freeze the winning settings before generating checkpoint answers.
3. Start the CPU models, create an empty SQLite database, and measure parsing, canonical token counting, chunking, embedding, sparse/vector index construction and persistence for each transcript. Time model startup separately.
4. Run two inference passes per checkpoint question, reversing the order on the second pass. Clear the answer cache between passes and reset the generator KV prefix for each miss. Require identical validated claims and delivered IDs. Replay each question five more times against the populated answer cache. Elapsed time covers the in-process request through the validated result, including a cache write on misses.
5. Inspect the returned statements against the upstream answer and original transcript. Store one row per atomic factual assertion. Score the exact predictions by digest so later output changes require a fresh review. Machine checks verify exact answers where applicable, citations and evidence budgets. The qualitative judgements were made by ChatGPT Work. A second human annotation is outstanding.
6. Report each dataset separately. Re-run retrieval with 2,048, 4,096 and 8,192 evidence-token ceilings and the frozen candidate count. This diagnostic measures delivered evidence only. Larger ceilings can leave room unused when the nominated passages already fit.

## The six primary metrics

| Metric | Calculation and interpretation |
| --- | --- |
| Answer correctness ↑ | Per-question factual-unit F1 = 2 × distinct matched reference units / (predicted atomic facts + reference units). Exact short answers score 1. For an unanswerable question, score 1 for abstention. Source-supported extra details can lower reference F1. Report the macro mean over questions. |
| Evidence recall ↑ | Fraction of required evidence groups delivered to the generator within its token budget, macro-averaged over eligible questions. A group can have alternative sufficient sets. All source turns in one alternative must be present. Split turns count when every fragment is present. The frozen curated alternatives can miss other valid supporting passages. |
| Unsupported claim rate ↓ | Total fabricated, contradicted or contextually obsolete asserted facts / all released factual assertions. Check against the full transcript. A fact can be supported while its specific citation is inadequate. Citation support and answer coverage are reported separately. Zero claims gives an undefined rate. |
| P95 query latency ↓ | Nearest-rank 95th percentile of wall time. Cold means loaded models, initialized index and an answer-cache miss. Warm means an exact answer-cache hit. Process startup is separately timed. P50 and first-pass component timings are also recorded. |
| Ingestion throughput ↑ | Sum of actual Qwen tokenizer counts for the normalized source texts / sum of fresh ingestion wall times, in tokens/s. Each freshly built index is written to SQLite. Dataset download, dataset-to-transcript conversion and one-time model loading are outside the clock. |
| Peak RAM ↓ | Linux process high-water RSS in MiB over the entire benchmark. Generation and both ONNX sessions share this process. The value includes model pages and the context cache. It is a workload-wide peak. |

### Claim review

In [claim_review.json](results/claim_review.json) an atom is an array of four integers. They give the output line number, matched reference unit (zero for an extra or erroneous fact), support anywhere in the source (0/1), and support in the cited fragment(s) (0/1). Split independently checkable statements. Ignore fillers, greetings, incomplete sentence tails and markup. A named person put forward as the answer counts as an assertion. Use the ordered reference outlines from the adapters. An ELITR short answer is one unit. Credit each gold unit once. A source-true extra remains in the precision denominator. Count a wrong current owner against support even if that person once owned the task.

The scorer rejects missing reviews, changed digests, and unchecked output lines. Reproduction writes the full original questions, references, accepted and rejected output, and cited evidence into the local predictions directory. The published traces carry IDs, model-token counts and fingerprints so the upstream protected question archive stays intact.

## Reproduction

The verified environment used 64-bit Linux and CPython 3.12. Native compilation needs GCC, G++, GNU Make or Ninja, and a Python installation with venv support. Setup downloads the pinned research data and model files. The query, tune, score and benchmark commands then work offline.

~~~bash
bash scripts/setup.sh
bash scripts/reproduce.sh
~~~

A fresh replay writes to .work/replay and a tuning rerun writes to .work/retune. The frozen files stay in configs and results. The recorded results can also be rescored without rerunning the LLM.

~~~bash
.venv/bin/python -m bench.score
~~~

Use the same model bytes, generator prompt, tokenizer, hardware class, thread settings, retrieval budgets, splits, cache regimes and scoring rules for Phase 3. Replay the baseline in the same environment as MeetAI before publishing a speed comparison. Regrade changed answers from either system using the same rubric, preferably blinded. Publish each system’s completeness and abstentions alongside unsupported claims.
