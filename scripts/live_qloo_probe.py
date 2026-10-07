"""Probe the live Qloo hackathon API and print response shapes. Needs QLOO_API_KEY in .env. Read-only."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from taste_mcp.app import load_env_file  # noqa: E402
from taste_mcp.qloo import QlooClient, QlooError, summarize_entities  # noqa: E402

load_env_file(ROOT / ".env")
client = QlooClient()


def shape(value, depth=0):
    if isinstance(value, dict):
        if depth >= 3:
            return "{..}"
        return "{" + ", ".join(f"{k}: {shape(v, depth + 1)}" for k, v in list(value.items())[:16]) + "}"
    if isinstance(value, list):
        return f"[{shape(value[0], depth + 1) if value else ''}]x{len(value)}"
    return type(value).__name__


def attempt(label, call):
    started = time.time()
    try:
        out = call()
        print(f"\n== {label}  ({time.time() - started:.1f}s)")
        return out
    except QlooError as error:
        print(f"\n== {label}  FAILED {error.status}: {error.body[:260]}")
    except ValueError as error:
        print(f"\n== {label}  REFUSED LOCALLY: {error}")
    return None


def first_id(name, types=None):
    response = client.search(name, types=types, take=1)
    return response["results"][0]["entity_id"] if response.get("results") else None


if __name__ == "__main__":
    wes = first_id("Wes Anderson")
    phoebe = first_id("Phoebe Bridgers", ["artist"])
    rooney = first_id("Sally Rooney")
    print("seed ids:", bool(wes), bool(phoebe), bool(rooney))
    seeds = [s for s in (wes, phoebe, rooney) if s]

    for kind, extra in (("movie", {}), ("artist", {}), ("book", {}), ("podcast", {}), ("brand", {}), ("tv_show", {}),
                        ("place", {"location": "Lisbon"})):
        out = attempt(f"insights {kind} {extra or ''}",
                      lambda: client.insights(kind, entities=seeds, location_query=extra.get("location"), take=4))
        if not out:
            continue
        if kind == "movie":
            print("top-level:", shape(out)[:500])
            entity = (out.get("results") or {}).get("entities", [{}])[0]
            print("entity keys:", list(entity.keys()))
            print("query:", json.dumps(entity.get("query"))[:400])
            print("properties keys:", list((entity.get("properties") or {}).keys())[:25])
            print("tags[0]:", json.dumps((entity.get("tags") or [None])[0])[:200])
        for item in summarize_entities(out)[:4]:
            print("  ", json.dumps({k: item.get(k) for k in ("name", "type", "affinity", "where", "drivers")}, ensure_ascii=False)[:260])

    out = attempt("tags 'natural wine'", lambda: client.tags("natural wine", take=5))
    if out:
        print("shape:", shape(out)[:300])
        print("  ", json.dumps((out.get("results") or {}).get("tags", out.get("results"))[:3] if out.get("results") else None)[:400])
    out = attempt("audiences (hobbies)", lambda: client.audiences(["urn:audience:hobbies_and_interests"], take=5))
    if out:
        print("shape:", shape(out)[:300])
        print("  ", json.dumps(((out.get("results") or {}).get("audiences") or [])[:3])[:400])
