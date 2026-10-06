"""Run the web app against canned data so the page can be worked on without any keys.

    .venv/bin/python scripts/dev_fake.py   ->  http://127.0.0.1:7861

Everything shown in this mode is sample data, not a real taste-graph result.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn  # noqa: E402

from taste_mcp.app import create_app  # noqa: E402
from taste_mcp.qloo import QlooClient  # noqa: E402

SEEDS = {"Wes Anderson": ("E-WES", "person"), "Phoebe Bridgers": ("E-PB", "artist"), "Sally Rooney": ("E-SR", "person")}
SAMPLE = {
    "artist": [("Big Thief", 0.94, ["indie folk", "intimate"], "E-PB"), ("Sufjan Stevens", 0.91, ["chamber pop"], "E-PB"),
               ("Faye Webster", 0.88, ["soft rock"], "E-PB")],
    "movie": [("Frances Ha", 0.92, ["black and white", "coming of age"], "E-WES"), ("The Worst Person in the World", 0.9, ["romance"], "E-SR")],
    "book": [("My Year of Rest and Relaxation", 0.89, ["literary fiction"], "E-SR"), ("Trick Mirror", 0.86, ["essays"], "E-SR")],
    "place": [("Pensão Amor", 0.93, ["cocktail bar", "eclectic"], "E-WES"), ("Livraria Ler Devagar", 0.9, ["bookshop"], "E-SR")],
}


def transport(url, headers, timeout):
    from urllib.parse import parse_qsl, urlparse
    parsed = urlparse(url)
    params = dict(parse_qsl(parsed.query))
    time.sleep(0.35)
    if parsed.path == "/search":
        query = params.get("query", "")
        hit = SEEDS.get(query)
        for kind, rows in SAMPLE.items():          # sample picks can be looked up too, so the comparison has something to find
            for i, (name, *_rest) in enumerate(rows):
                if name.lower() == query.lower():
                    hit = (f"S-{kind}-{i}", kind)
                    query = name
        body = {"results": [{"entity_id": hit[0], "name": query, "types": [f"urn:entity:{hit[1]}"]}] if hit else []}
    else:
        kind = params.get("filter.type", "").split(":")[-1]
        wanted = set(filter(None, params.get("filter.results.entities", "").split(",")))
        body = {"results": {"entities": [entity for entity in [
            {"entity_id": f"S-{kind}-{i}", "name": name, "type": f"urn:entity:{kind}", "tags": [f"urn:tag:x:{t.replace(' ', '_')}" for t in tags],
             "properties": {"geocode": {"name": "Lisbon", "country_code": "PT"}} if kind == "place" else {},
             "query": {"affinity": score, "explainability": {driver: 0.8}}}
            for i, (name, score, tags, driver) in enumerate(SAMPLE.get(kind, []))] if not wanted or entity["entity_id"] in wanted]}}
    return 200, {}, json.dumps(body).encode()


class SampleModel:
    configured = True

    def step(self, system, messages, tools, **_):
        time.sleep(0.5)
        if "no tools" in system:      # the unaided second opinion
            return {"text": json.dumps({"picks": [
                {"kind": "artist", "name": "Big Thief"}, {"kind": "artist", "name": "Bon Iver"}, {"kind": "artist", "name": "Mitski"},
                {"kind": "movie", "name": "Frances Ha"}, {"kind": "movie", "name": "The Grand Budapest Hotel"},
                {"kind": "book", "name": "Normal People"}, {"kind": "place", "name": "Time Out Market"}]}), "calls": []}
        turns = sum(1 for m in messages if m["role"] == "tool")
        if turns == 0:
            return {"text": None, "calls": [{"name": "find_entities", "args": {"names": list(SEEDS) + ["Some Obscure Zine"]}}]}
        kinds = ["artist", "movie", "book", "place"]
        if turns <= len(kinds):
            kind = kinds[turns - 1]
            args = {"target_type": kind, "entity_ids": [v[0] for v in SEEDS.values()], "take": 4}
            if kind == "place":
                args["city"] = "Lisbon"
            return {"text": None, "calls": [{"name": "recommend", "args": args}]}
        sections = [{"title": title, "kind": kind, "picks": [{"name": n, "why": f"Sample reason for {n}."} for n, *_ in SAMPLE[kind]]}
                    for title, kind in (("On the speakers", "artist"), ("Film night", "movie"), ("Book club", "book"), ("Neighbours to team up with", "place"))]
        sections[0]["picks"].append({"name": "A Band The Model Made Up", "why": "This one is not in the graph."})
        return {"text": json.dumps({"headline": "Soft light, sad songs, good wine (sample data)",
                                    "audience_read": "Sample text: a crowd that likes its melancholy well composed and its jokes deadpan.",
                                    "sections": sections, "caveats": ["'Some Obscure Zine' could not be found in the taste graph."]}), "calls": []}


if __name__ == "__main__":
    uvicorn.run(create_app(QlooClient("sample", transport=transport), SampleModel()), host="127.0.0.1", port=7861, log_level="warning")
