#!/usr/bin/env bash
set -euo pipefail
branch=phase-3-benchmark
folder="results/phase3/checkpoints/$MEETAI_BATCH"
test "$GITHUB_REF_NAME" = "$branch"
test -f "$folder/done.json"
dest="$RUNNER_TEMP/meetai-result-publish"
git config user.name 'github-actions[bot]'
git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
for attempt in {1..10}; do
    git fetch --no-tags origin "refs/heads/$branch:refs/remotes/origin/$branch"
    ref="refs/remotes/origin/$branch"
    if git show "$ref:$folder/done.json" > "$RUNNER_TEMP/prior-done.json" 2>/dev/null; then
        if python -c 'import json,sys; a,b=(json.load(open(p)) for p in sys.argv[1:]); sys.exit(a["protocol"] != b["protocol"])' "$folder/done.json" "$RUNNER_TEMP/prior-done.json"; then
            echo "The completed batch is already on GitHub."
            exit 0
        fi
    fi
    git worktree add --quiet --detach "$dest" "$ref"
    mkdir -p "$dest/$folder"
    cp -a "$folder/." "$dest/$folder/"
    git -C "$dest" add -- "$folder"
    git -C "$dest" commit --quiet -m "Checkpoint Phase 3 batch $MEETAI_BATCH"
    if git -C "$dest" push origin "HEAD:refs/heads/$branch"; then exit 0; fi
    git worktree remove --force "$dest"
    sleep $((attempt + RANDOM % 5))
done
exit 1
