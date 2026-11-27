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
    action.add_argument(
        "--extract",
        action="store_true",
        help="Extract cited decisions, actions, commitments, open issues and follow-ups with MeetAI",
    )
    parser.add_argument(
        "--system", choices=("baseline", "meetai"), help="Default: baseline, or meetai for --extract"
    )
    parser.add_argument(
        "--trace", action="store_true", help="Include evidence selection and validation diagnostics"
    )
    parser.add_argument("--config", type=Path, help="Override the selected system configuration")
    parser.add_argument("--assets", default=".assets/models", type=Path)
    parser.add_argument("--cache", default=".work/cache.sqlite", type=Path)
    args = parser.parse_args()
    system = args.system or ("meetai" if args.extract else "baseline")
    cfg = json.loads((args.config or Path(f"configs/{system}.json")).read_text())
    if args.system and args.system != cfg.get("system", "baseline"):
        parser.error("--system conflicts with the supplied configuration")
    if args.extract and cfg.get("system") != "meetai":
        parser.error("--extract requires the meetai system")
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    models = Models(args.assets, cfg["threads"], cfg["window"])
    cache = Cache(args.cache)
    try:
        idx = Index.build(args.transcript, models, cfg, cache)
        if args.extract:
            from .minutes import whole_meeting

            result = whole_meeting(idx, models, cfg, structured=True)
        elif args.summary:
            result = summarize(idx, models, cfg)
        else:
            result = answer(idx, args.ask, models, cfg, cache)
        cited = {ref for claim in result["claims"] for ref in claim["cites"]}
        public = {key: result[key] for key in ("claims", "abstained", "llm_calls")}
        public["evidence"] = {
            line.id: {
                "turn": line.turn,
                "speaker": line.speaker,
                "text": line.text,
                **({"timestamp": line.timestamp} if line.timestamp else {}),
            }
            for line in idx.lines
            if line.id in cited
        }
        if "items" in result:
            public["items"] = result["items"]
        if args.trace:
            public["trace"] = result.get("trace", {})
            public["validation"] = result.get("validation", result.get("rejected", []))
            public["delivered_ids"] = result.get("ids", [])
            public["timings"] = result.get("timings", {})
        if "cache_hit" in result:
            public["cache_hit"] = result["cache_hit"]
        print(json.dumps(public, indent=2, ensure_ascii=False))
    finally:
        cache.close()


if __name__ == "__main__":
    main()
