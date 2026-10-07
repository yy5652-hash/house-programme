"""Offline tests for the unaided-model comparison."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient  # noqa: E402

from taste_mcp.agent import run_brief  # noqa: E402
from taste_mcp.app import create_app  # noqa: E402
from taste_mcp.compare import compare, unaided_picks  # noqa: E402
from taste_mcp.qloo import QlooClient  # noqa: E402
from test_agent import BRIDGE_CALL, ScriptedModel, final  # noqa: E402
from test_app import events_of  # noqa: E402
from test_taste_mcp import KEY, FakeTransport, insights_route, search_route  # noqa: E402

CANDIDATES = {"Bar A": "P-2", "Bar B": "P-3", "Pensao Amor": "P-1"}


def search(params):
    if params["query"] in CANDIDATES:
        name = "Pensão Amor" if params["query"] == "Pensao Amor" else params["query"]
        return 200, {}, {"results": [{"entity_id": CANDIDATES[params["query"]], "name": name, "types": ["urn:entity:place"]}]}
    return search_route(params)


def insights(params):
    if "filter.results.entities" in params:      # scoring call: P-2 has an affinity, P-3 does not; P-99 was never asked about
        return 200, {}, {"results": {"entities": [
            {"entity_id": "P-2", "name": "Bar A", "type": "urn:entity:place", "query": {"affinity": 0.41}},
            {"entity_id": "P-99", "name": "Stray", "type": "urn:entity:place", "query": {"affinity": 0.99}},
        ]}}
    return insights_route(params)


def unaided(*names):
    return {"text": json.dumps({"picks": [{"kind": "place", "name": n} for n in names]}), "calls": []}


def setup(replies, routes=None):
    transport = FakeTransport(routes=routes or {"/search": search, "/v2/insights": insights})
    return ScriptedModel(replies), QlooClient(KEY, transport=transport, max_retries=0), transport


class CompareTests(unittest.TestCase):
    def test_each_unaided_suggestion_is_looked_up_and_scored_against_the_same_signals(self):
        model, client, transport = setup([BRIDGE_CALL, final(["Pensão Amor"]), unaided("Pensao Amor", "Bar A", "Bar B", "Ghost Bar")])
        result = run_brief(model, client, "Regulars love Wes Anderson. We are in Lisbon.")
        self.assertEqual(result["seeds"], [{"id": "E-WES", "name": "Wes Anderson"}])

        out = compare(model, client, "Regulars love Wes Anderson. We are in Lisbon.", result, per_kind=4)
        by_name = {row["name"]: row for row in out["rows"]}
        self.assertEqual(by_name["Pensao Amor"], {"kind": "place", "name": "Pensao Amor", "resolved_as": "Pensão Amor",
                                                  "status": "in_programme", "affinity": 0.93})
        self.assertEqual(by_name["Bar A"], {"kind": "place", "name": "Bar A", "status": "scored", "affinity": 0.41,
                                            "below_programme": True})
        self.assertEqual(by_name["Bar B"]["status"], "no_affinity")
        self.assertEqual(by_name["Ghost Bar"]["status"], "not_in_graph")
        self.assertEqual(out["summary"], {"suggested": 4, "in_programme": 1, "scored": 1, "no_affinity": 1, "not_in_graph": 1,
                                          "unchecked": 0, "scored_below_programme": 1, "programme_floors": {"place": 0.93}})
        scoring = [c for c in transport.calls if "filter.results.entities" in c["params"]][0]["params"]
        self.assertEqual(scoring["signal.interests.entities"], "E-WES")
        self.assertEqual(scoring["filter.results.entities"], "P-1,P-2,P-3")
        self.assertEqual(model.seen[-1]["tools"], [])          # the second opinion really had no tools

    def test_a_failing_lookup_marks_rows_unchecked_instead_of_guessing(self):
        def broken(params):
            return (500, {}, {"error": "boom"}) if "filter.results.entities" in params else insights_route(params)
        model, client, _ = setup([BRIDGE_CALL, final(["Pensão Amor"]), unaided("Bar A", "Pensao Amor")],
                                 routes={"/search": search, "/v2/insights": broken})
        result = run_brief(model, client, "x")
        out = compare(model, client, "x", result)
        self.assertEqual({r["name"]: r["status"] for r in out["rows"]}, {"Bar A": "unchecked", "Pensao Amor": "in_programme"})
        self.assertIn("HTTP 500", out["problem"])

    def test_unaided_picks_are_deduplicated_capped_and_limited_to_the_asked_kinds(self):
        reply = {"text": json.dumps({"picks": [{"kind": "place", "name": "A"}, {"kind": "place", "name": "a"},
                                               {"kind": "place", "name": "B"}, {"kind": "place", "name": "C"},
                                               {"kind": "movie", "name": "M"}, {"kind": "place", "name": ""}, "junk"]}), "calls": []}
        self.assertEqual(unaided_picks(ScriptedModel([reply]), "x", ["place"], per_kind=2),
                         [{"kind": "place", "name": "A"}, {"kind": "place", "name": "B"}])

    def test_nothing_to_compare_makes_no_model_call(self):
        model = ScriptedModel([])
        self.assertEqual(compare(model, None, "x", {"brief": {"sections": []}, "seeds": []}), {"rows": [], "summary": None})
        self.assertEqual(model.seen, [])


class AppComparisonTests(unittest.TestCase):
    def test_comparison_is_streamed_after_the_programme(self):
        model, client, _ = setup([BRIDGE_CALL, final(["Pensão Amor"]), unaided("Bar A")])
        events = events_of(TestClient(create_app(client, model)).post("/api/programme", json={"request": "Regulars love Wes Anderson."}))
        self.assertEqual([e["type"] for e in events], ["call", "result", "done", "programme", "comparing", "comparison"])
        self.assertEqual(events[-1]["rows"][0]["status"], "scored")

    def test_a_broken_comparison_never_spoils_the_programme(self):
        model, client, _ = setup([BRIDGE_CALL, final(["Pensão Amor"])])        # no reply left for the second opinion
        with self.assertLogs("taste_mcp", level="ERROR"):
            events = events_of(TestClient(create_app(client, model)).post("/api/programme", json={"request": "Regulars love Wes Anderson."}))
        self.assertEqual([e["type"] for e in events], ["call", "result", "done", "programme", "comparing", "comparison"])
        self.assertEqual(events[-1]["rows"], [])
        self.assertIn("unexpectedly", events[-1]["problem"])


if __name__ == "__main__":
    unittest.main()
