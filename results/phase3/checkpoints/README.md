# Phase 3 recovery

Each subfolder holds one completed batch. A completion record includes hashes of every output and the inputs that define the experiment, plus the runner CPU and workflow run ID. Each QA or scale batch runs both systems sequentially on the same machine. Different meetings and sizes may run in parallel. A new workflow run skips intact batches with the same experiment fingerprint.

Only public metrics, hashes and synthetic outputs are stored as plain text. The temporary review packets contain the real model answers and citations. They are encrypted for the project reviewer. The private key is held separately in the user's dedicated MeetAI Phase 3 Recovery folder. The certificate in configs is public.

To resume, run the Phase 3 checkpoints workflow on the phase-3-benchmark branch. The status command also runs without model weights or third-party dependencies.

~~~bash
python3 -m bench.phase3_checkpoint --pending
~~~

The full Phase 1 baseline and Phase 2 acceptance checkpoints stay available in Git history. The standalone AMD baseline and development ablations predate these paired hosted runs. Changing an experiment input invalidates its prior checkpoint. Scoring and reporting can be repeated using saved predictions.

The restore command decrypts finished review packets into the ignored work directory. It checks the saved hashes and refuses conflicting records. Reviewers can restore batches as they arrive and rescore without generating answers again.

~~~bash
.venv/bin/python -m bench.phase3_restore --key /path/to/review-private.pem
.venv/bin/python -m bench.phase3_score --packet
.venv/bin/python -m bench.phase3_checkpoint --gather .work/phase3
~~~
