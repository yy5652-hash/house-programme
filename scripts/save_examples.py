"""Save one real run of each example on the page, so a visitor sees a full programme at once.

    .venv/bin/python scripts/save_examples.py [base-url]

Each example's text is read from the page itself, sent to the running app (the hosted demo by default), and the
programme and second-opinion events are written to src/taste_mcp/web/examples/<name>.json with the time of the run.
Nothing in those files is edited by hand; rerun this script to refresh them.
"""
import html
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "src/taste_mcp/web/index.html"
OUT = ROOT / "src/taste_mcp/web/examples"


def examples():
    for match in re.finditer(r'data-saved="([a-z]+)"\s+data-example="([^"]+)"', PAGE.read_text(encoding="utf-8")):
        yield match.group(1), html.unescape(match.group(2))


def run(base, text):
    request = urllib.request.Request(base.rstrip("/") + "/api/programme", data=json.dumps({"request": text}).encode(),
                                     headers={"Content-Type": "application/json"})
    started, events = time.monotonic(), []
    with urllib.request.urlopen(request, timeout=240) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events, round(time.monotonic() - started, 1)


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "https://house-programme.onrender.com"
    OUT.mkdir(parents=True, exist_ok=True)
    for name, text in examples():
        events, seconds = run(base, text)
        by_type = {event["type"]: event for event in events}
        if "programme" not in by_type:
            print(name, "FAILED:", by_type.get("error", {}).get("message", "no programme"))
            continue
        saved = {"saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "source": base, "request": text, "seconds": seconds,
                 "calls": sum(1 for event in events if event["type"] == "call"), "programme": by_type["programme"],
                 "comparison": by_type.get("comparison")}
        (OUT / f"{name}.json").write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")
        stats = by_type["programme"].get("stats", {})
        print(name, seconds, "s,", saved["calls"], "calls, stats", {k: stats[k] for k in list(stats)[:4]}, "| second opinion:", bool(saved["comparison"] and saved["comparison"].get("rows")))
        time.sleep(8)                                                  # the free model tier counts requests per minute
