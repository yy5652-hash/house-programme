"""Run several different briefs against the live APIs and print one line per run, to see how steady the agent is.

    .venv/bin/python scripts/live_batch.py [runs-per-brief]

Each run costs about three model requests and 15-30 Qloo requests. No second opinion here.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from taste_mcp.agent import run_brief  # noqa: E402
from taste_mcp.app import load_env_file  # noqa: E402
from taste_mcp.llm import ModelError, default_model  # noqa: E402
from taste_mcp.qloo import QlooClient, QlooError  # noqa: E402

BRIEFS = [
    "We run a 40-seat natural wine bar in Lisbon. Regulars talk about Wes Anderson, Phoebe Bridgers and Sally Rooney novels. "
    "What should we programme this month, and who nearby should we team up with?",
    "Independent bookshop with a cafe corner in Brooklyn. Our customers love Studio Ghibli, Haruki Murakami and Mitski. "
    "Plan next month's events and suggest partners in the neighbourhood.",
    "Vinyl bar in Berlin Neukolln. The crowd is into Kraftwerk, David Lynch and Blade Runner. Give us listening nights and film picks.",
    "A climbing gym cafe in Denver. Members love Patagonia, Free Solo and the podcast Radiolab. What should we play, screen and stock?",
    "Small tea house in Kyoto popular with travellers who love Lost in Translation, Ryuichi Sakamoto and Banana Yoshimoto.",
]


def main() -> int:
    load_env_file(ROOT / ".env")
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    client, model = QlooClient(), default_model()
    print("brief | seconds | calls (failed) | kept/dropped | picks per section | places | unresolved seeds | caveats")
    for index, brief in enumerate(BRIEFS):
        for _ in range(repeats):
            started = time.time()
            try:
                out = run_brief(model, client, brief)
            except (ModelError, QlooError) as error:
                print(f"{index} | FAILED after {time.time() - started:.0f}s: {str(error)[:150]}")
                continue
            sections = out["brief"]["sections"]
            places = [p["name"] + (" @" + p.get("where", "?")) for s in sections if s.get("kind") == "place" for p in s["picks"]]
            unresolved = sorted({name for t in out["trace"] for name in []})
            stats = out["stats"]
            print(f"{index} | {time.time() - started:.0f}s | {stats['tool_calls']} ({stats['failed_calls']}) | {stats['kept']}/{stats['dropped']} | "
                  f"{[(s.get('kind'), len(s['picks'])) for s in sections]} | {places[:4]} | seeds {[s['name'] for s in out['seeds']]} | "
                  f"{len(out['brief'].get('caveats') or [])} caveats")
            for caveat in (out["brief"].get("caveats") or [])[:2]:
                print("      caveat:", caveat[:170])
            for entry in out["trace"]:
                if not entry["ok"]:
                    print("      failed call:", entry["tool"], str(entry["error"])[:140])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
