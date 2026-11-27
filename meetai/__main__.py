import argparse
import json
from pathlib import Path

from .models import Models
from .pipeline import answer, summarize
from .retrieval import Cache, Index


def main():
    parser = argparse.ArgumentParser(description="Ask one local meeting transcript a question")
    parser.add_argument("transcript", type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--ask")
    action.add_argument("--summary", action="store_true")
    parser.add_argument("--config", default="configs/baseline.json", type=Path)
    parser.add_argument("--assets", default=".assets/models", type=Path)
    parser.add_argument("--cache", default=".work/cache.sqlite", type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    models = Models(args.assets, cfg["threads"], cfg["window"])
    cache = Cache(args.cache)
    try:
        idx = Index.build(args.transcript, models, cfg, cache)
        result = (
            summarize(idx, models, cfg) if args.summary else answer(idx, args.ask, models, cfg, cache)
        )
        cited = {ref for claim in result["claims"] for ref in claim["cites"]}
        public = {key: result[key] for key in ("claims", "abstained", "llm_calls")}
        public["evidence"] = {
            line.id: {"turn": line.turn, "speaker": line.speaker, "text": line.text}
            for line in idx.lines
            if line.id in cited
        }
        if "cache_hit" in result:
            public["cache_hit"] = result["cache_hit"]
        print(json.dumps(public, indent=2, ensure_ascii=False))
    finally:
        cache.close()


if __name__ == "__main__":
    main()
