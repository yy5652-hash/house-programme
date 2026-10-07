"""Run one programme end to end against the live Qloo API and the configured model, then the second opinion.

    .venv/bin/python scripts/live_run.py ["request text"]

Needs QLOO_API_KEY and a model key in .env. About four model requests and fifteen to twenty-five Qloo requests.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from taste_mcp.agent import run_brief  # noqa: E402
from taste_mcp.app import load_env_file  # noqa: E402
from taste_mcp.compare import compare  # noqa: E402
from taste_mcp.llm import ModelError, default_model  # noqa: E402
from taste_mcp.qloo import QlooClient, _urllib_transport  # noqa: E402

DEFAULT = ("We run a 40-seat natural wine bar in Lisbon. Regulars talk about Wes Anderson films, Phoebe Bridgers and "
           "Sally Rooney novels. What should we programme this month, and who nearby should we team up with?")


class Counted:
    def __init__(self, inner):
        self.inner, self.n = inner, 0

    def step(self, *args, **kwargs):
        self.n += 1
        return self.inner.step(*args, **kwargs)


def main() -> int:
    load_env_file(ROOT / ".env")
    calls = []

    def transport(url, headers, timeout):
        calls.append(url.split("?")[0].rsplit("/", 1)[-1])
        return _urllib_transport(url, headers, timeout)

    client, model = QlooClient(transport=transport), Counted(default_model())
    request = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    started = time.time()
    try:
        out = run_brief(model, client, request)
    except ModelError as error:
        print("MODEL ERROR:", str(error)[:400])
        return 1
    print(f"programme: {time.time() - started:.1f}s | model requests {model.n} | qloo requests {len(calls)} | stats {out['stats']}")
    for entry in out["trace"]:
        print("  step", entry["step"], entry["tool"], json.dumps(entry["args"], ensure_ascii=False)[:150], "->",
              "ok" if entry["ok"] else "ERROR " + str(entry["error"])[:120], entry.get("returned", ""))
    brief = out["brief"]
    print("seeds:", [s["name"] for s in out["seeds"]])
    print("headline:", brief.get("headline"))
    print("read:", brief.get("audience_read"))
    for section in brief["sections"]:
        print(f"\n  [{section.get('kind')}] {section['title']}")
        for pick in section["picks"]:
            drivers = ", ".join(f"{d['name']} {d['impact']}" for d in pick.get("drivers", [])[:3])
            print(f"    - {pick['name']} ({pick.get('detail', '')}) aff {pick.get('affinity')} | {pick.get('where', '')} | img {'y' if pick.get('image') else 'n'}")
            print(f"        why: {pick.get('why', '')[:170]}")
            print(f"        tags: {pick.get('tags')} | drivers: {drivers}")
    print("\nunverified:", brief["unverified"])
    print("caveats:", brief.get("caveats"))
    started = time.time()
    second = compare(model, client, request, out)
    print(f"\nsecond opinion: {time.time() - started:.1f}s | model requests {model.n} | qloo requests {len(calls)}")
    for row in second["rows"]:
        print("  ", row["kind"], "|", row["name"], "|", row["status"], row.get("affinity", ""), "|", row.get("resolved_as", ""))
    print("summary:", second["summary"], second.get("problem", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
