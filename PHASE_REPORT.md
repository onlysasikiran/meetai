# Phase 3 · Benchmarking and optimization

**Problem.** A meeting can revise an owner or deadline and use a second name for the same project. Relevant passages may still leave the generator with an obsolete answer or a citation that mentions the right person without establishing the decision.

## Completed work

- Froze the standard RAG and selected the linked RAG configuration using six development meetings. The linked version now reranks 20 candidate turns and retains neighboring dialogue for suitable fact and topic questions. [Freeze](configs/phase3-freeze.json) · [Selections and measured ablations](results/phase3/dev.json)
- Measured a standalone Phase 3 baseline on an AMD CPU. It delivered 68.4% evidence recall on the held-out roster. The paired benchmark measures both systems on the same host within each batch. [Standalone raw measurements](results/phase3/baseline-amd-reference.json)
- Added separate GitHub checkpoints for eight held-out QA meetings, two MeetingBank operational inputs in a shared batch, seven scale workloads, a generation ablation and the test suite. Every completed batch records its source commit, CPU and file hashes. [Checkpoint folder](results/phase3/checkpoints) · [Hosted benchmark](https://github.com/onlysasikiran/meetai/actions/runs/46132175836)

## Development results

| Retrieval version | Equal-dataset evidence recall | Mean evidence tokens | Mean retrieval |
| --- | ---: | ---: | ---: |
| Standard RAG | 70.4% | 842 | 94 ms |
| Phase 2 MeetAI | 41.1% | 297 | 178 ms |
| Smaller reranking pool | 41.1% | 233 | 136 ms |
| Neighboring turns | 50.3% | 375 | 189 ms |
| Both changes, frozen MeetAI | 50.3% | 324 | 144 ms |

The 20-turn pool removed 16 reranking slots. Neighboring turns increased recall by 9.2 percentage points compared with Phase 2. Disabling coverage selection on the finalist raised recall to 63.1% but increased mean evidence from 324 to 509 tokens. The predeclared selection considered four full-architecture candidates. Leave-one-out variants diagnose tradeoffs and did not reopen selection. [Selection rules](configs/phase3-protocol.json)

## Checkpoint status and recovery

The paired held-out and scale batches publish as each measurement finishes. The final comparison requires all batches and scored output fingerprints. The local suite passed 35 tests, including restart, fingerprint, missing-file and corruption cases. The hosted run records the suite from its launch commit separately. Reviewed answers can be restored from encrypted packets. [Recovery instructions](results/phase3/checkpoints/README.md) · [Benchmark protocol and datasets](BENCHMARK.md)

The held-out sample contains nine real questions and ten synthetic questions. Timing is CPU-specific. Scoring uses one blinded Work reviewer. The independent Phase 4 audit is a later deliverable. [Phase 2 checkpoint](https://github.com/onlysasikiran/meetai/blob/a757327a82391773402626a3d5ea140d4de44612/PHASE_REPORT.md)
