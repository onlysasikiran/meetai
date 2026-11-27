# Phase 1 · Foundation and baseline RAG

## Completed work

Built a CPU-only baseline for finding decisions, owners and deadlines in one meeting. Prepared meeting-disjoint tuning and checkpoint workloads, reference answers, sufficient evidence sets, and six evaluation metrics. Published the baseline code, benchmark, tuning trials, per-question traces, claim judgements and content hashes.

## Decisions

- **Comparator.** Conventional speaker-aware chunking, BM25 plus BGE-small retrieval, reciprocal rank fusion, MiniLM reranking, fixed parent expansion and chronological packing. Qwen3-4B Q4_K_M runs locally with greedy decoding. Source validation checks citation membership. Semantic support is assessed in the benchmark.
- **Freeze.** Select on tuning data only. Of 23 tuning questions, 21 had annotated evidence. Across 24 trials, the chosen configuration reached 70.42% dataset-macro evidence recall and averaged 835 evidence tokens. A 1,024-token ceiling tied the best 2,048-token result. The best child-only trial reached 60.42%. The frozen settings use 160-token children, parents of two children, and three selected children.
- **Separation.** Score 22 different questions on six meetings. Use MeetingBank for an additional operational workload. Official ELITR test2 and QMSum test remain available for Phase 3. Raw third-party data and weights are resolved from checksummed sources.
- **Scoring.** Grade reference completeness, truth in the full transcript, and support in the actual citation separately. ChatGPT Work reviewed the atomic statements. Exact answers and abstentions also have deterministic checks.

## Actual results

| Dataset | Answer F1 | Evidence recall | Unsupported / released factual claims | Claims supported by citations | Fresh P95 | Ingestion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ELITR-Bench · 8 questions | 63.3% | 62.5% | 3/13 | 5/13 | 28.44 s | 6,561 tokens/s |
| QMSum · 4 questions | 0.0% | 6.25% | 1/14 | 7/14 | 25.28 s | 6,858 tokens/s |
| MeetAI-Stress · 10 questions | 65.0% | 87.5% | 1/8 | 5/8 | 18.21 s | 8,629 tokens/s |
| MeetingBank · 2 agenda segments | — | — | — | — | — | 7,606 tokens/s |

**Whole workload.** Fresh-answer P50 was 17.54 s and P95 was 25.28 s. Exact cache-hit P95 was 0.166 ms. One-time model startup took 8.18 s. Peak process RSS was 5,117 MiB (5.00 GiB). Indexing took 25.83 s for 190,037 normalized source tokens. The largest individual segment had 84,887 tokens. The timing host exposed nine vCPUs on AMD EPYC 9V74 and used six inference threads, CPython 3.12.14 and Linux x86-64.

**Accuracy bottleneck.** Neither synthetic two-hop question earned answer credit. One answer explicitly gave a correct fact about the wrong project. On the four QMSum questions, mostly source-supported answers still omitted every scored reference unit. Fixed parent expansion can retrieve relevant discussion while missing the facts needed for a summary. The system correctly abstained on both unanswerable questions and also abstained on two answerable ones.

**Latency bottleneck.** On first-pass cache misses, generation averaged 17.83 s of the 17.95 s end-to-end time. Reranking averaged 0.089 s. Increasing just the evidence ceiling to 2K, 4K or 8K yielded identical recall at frozen retrieval depth because the nominated passages already fit.

## Test results and reproducibility

- All 22 repeated cold requests produced the same validated claims and cited context IDs. An additional retrieval replay rebuilt all 22 contexts and checked the token ceilings, gold-evidence recall and prediction fingerprints.
- **17 tests passed in 109.65 seconds** in a fresh Python environment built from a Git archive. The suite covers TXT, Markdown and text-layer PDF parsing, real-model answers across formats, long-turn fragmentation, chronological assembly, cache invalidation, citation rejection, metric edge cases and multi-pass summarization.
- The release pins the models, sources and Python environment. The setup procedure compiles llama.cpp for the host CPU. Benchmark reproduction commands and cache definitions are in [BENCHMARK.md](BENCHMARK.md).

## Limitations

The 22 questions are a development checkpoint. Each real QA dataset contributes two meetings. The two synthetic scoring meetings share templates with tuning, and their scores may overstate generalization. Evidence annotations are conservative, especially when only part of a long utterance was delivered. Claim judgements have one reviewer. The PDF parser requires a text layer. The measured workload used the hosted CPU. The 500K-token and 20K-turn targets, blinded audit and target-laptop performance remain future evaluations.

## Instructions for Phase 2

Use this repository and preserve the Phase 1 commit as the baseline reference. Add the proposed hierarchy, semantic links and adaptive evidence retrieval behind the same question, source-ID and answer interfaces. Start with the wrong-project and changed-owner examples in the Phase 1 traces. Keep the Qwen generator, tokenization and citation format shared. Run functional and integration tests for the new system. Reserve final held-out comparisons and ablations for Phase 3.
