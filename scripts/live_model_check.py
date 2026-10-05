"""Check the configured model's tool use end to end against canned taste data (no Qloo key needed).

    .venv/bin/python scripts/live_model_check.py ["request text"]

Uses real model requests (about three per run). The taste results are sample data, not Qloo output.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from taste_mcp.agent import run_brief  # noqa: E402
from taste_mcp.app import load_env_file  # noqa: E402
from taste_mcp.llm import ModelError, default_model  # noqa: E402
from taste_mcp.qloo import QlooClient  # noqa: E402

DEFAULT = ("We run a small natural-wine bar in Lisbon. Our regulars love Wes Anderson films, Phoebe Bridgers and "
           "Sally Rooney novels. Plan this month for us.")


class Counting:
    def __init__(self, inner):
        self.inner, self.requests = inner, 0

    def step(self, *args, **kwargs):
        self.requests += 1
        return self.inner.step(*args, **kwargs)


def main() -> int:
    load_env_file(ROOT / ".env")
    import dev_fake
    client = QlooClient("sample-data-not-a-key", transport=dev_fake.transport)
    model = default_model()
    if not model.configured:
        print("no model credentials found in .env")
        return 2
    print("model:", model.model)
    counted, started = Counting(model), time.time()
    try:
        out = run_brief(counted, client, sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
    except ModelError as error:
        print("MODEL ERROR:", str(error)[:400])
        return 1
    print(f"seconds: {time.time() - started:.1f} | model requests: {counted.requests} | stats: {out['stats']}")
    for entry in out["trace"]:
        print("  step", entry["step"], entry["tool"], json.dumps(entry["args"], ensure_ascii=False)[:110], "->",
              "ok" if entry["ok"] else entry["error"], entry.get("returned"))
    brief = out["brief"]
    print("headline:", brief.get("headline"))
    print("read:", brief.get("audience_read"))
    for section in brief["sections"]:
        print(" ", section["title"], "|", section.get("kind"), "|", [(p["name"], p.get("affinity")) for p in section["picks"]])
        print("     why:", section["picks"][0]["why"][:160])
    print("unverified:", brief["unverified"])
    print("caveats:", brief.get("caveats"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
