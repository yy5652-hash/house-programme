"""Offline tests. Responses are shaped like the Qloo reference docs; no network and no API key needed."""
import json
import sys
import unittest
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from taste_mcp.qloo import QlooClient, QlooError, entity_urn, summarize_entities  # noqa: E402
from taste_mcp.tools import safe_call, taste_bridge  # noqa: E402

KEY = "test-key-not-real"


class FakeTransport:
    """Records calls and replays queued (status, headers, json) responses, or routes by path."""

    def __init__(self, routes=None, queue=None):
        self.routes = routes or {}
        self.queue = list(queue or [])
        self.calls = []

    def __call__(self, url, headers, timeout):
        parsed = urllib.parse.urlparse(url)
        params = dict(urllib.parse.parse_qsl(parsed.query))
        self.calls.append({"url": url, "path": parsed.path, "params": params, "headers": dict(headers)})
        if self.queue:
            status, response_headers, payload = self.queue.pop(0)
        else:
            status, response_headers, payload = self.routes[parsed.path](params)
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        return status, response_headers, body


def search_route(params):
    known = {
        "Wes Anderson": {"entity_id": "E-WES", "name": "Wes Anderson", "types": ["urn:entity:person"], "popularity": 0.97},
        "Phoebe Bridgers": {"entity_id": "E-PHOEBE", "name": "Phoebe Bridgers", "types": ["urn:entity:artist"]},
    }
    hit = known.get(params["query"])
    return 200, {}, {"results": [hit] if hit else []}


def insights_route(params):
    return 200, {}, {
        "success": True,
        "results": {"entities": [{
            "entity_id": "P-1",
            "name": "Pensão Amor",
            "type": "urn:entity:place",
            "subtype": "urn:entity:place:bar",
            "properties": {
                "description": "Former boarding house turned bar. " * 12,
                "geocode": {"name": "Lisbon", "country_code": "PT"},
                "popularity": 0.81,
            },
            "tags": ["urn:tag:genre:place:bar:cocktail_bar", {"name": "Eclectic", "tag_id": "urn:tag:style:qloo:eclectic"}],
            "query": {"affinity": 0.93, "explainability": {"E-WES": 0.7, "E-PHOEBE": 0.3}},
        }]},
    }


class ClientTests(unittest.TestCase):
    def test_insights_request_is_built_from_documented_parameters(self):
        transport = FakeTransport(routes={"/v2/insights": insights_route})
        client = QlooClient(KEY, transport=transport)
        client.insights("movie", entities=["A", "B"], location_query="Lisbon", take=5,
                        extra={"filter.release_year.min": 2015})
        call = transport.calls[0]
        self.assertEqual(call["path"], "/v2/insights")
        self.assertEqual(call["params"]["filter.type"], "urn:entity:movie")
        self.assertEqual(call["params"]["signal.interests.entities"], "A,B")
        self.assertEqual(call["params"]["filter.location.query"], "Lisbon")
        self.assertEqual(call["params"]["filter.release_year.min"], "2015")
        self.assertEqual(call["params"]["feature.explainability"], "true")
        self.assertEqual(call["headers"]["X-Api-Key"], KEY)
        self.assertNotIn(KEY, call["url"])
        self.assertTrue(call["url"].startswith("https://hackathon.api.qloo.com/"))

    def test_undocumented_parameter_is_rejected_before_any_request(self):
        transport = FakeTransport(routes={"/v2/insights": insights_route})
        client = QlooClient(KEY, transport=transport)
        with self.assertRaises(ValueError):
            client.insights("movie", entities=["A"], extra={"fliter.tags": "x"})
        self.assertEqual(transport.calls, [])

    def test_insights_needs_a_signal_or_filter(self):
        client = QlooClient(KEY, transport=FakeTransport())
        with self.assertRaises(ValueError):
            client.insights("movie")

    def test_entity_types(self):
        self.assertEqual(entity_urn("tv_show"), "urn:entity:tv_show")
        self.assertEqual(entity_urn("urn:entity:place"), "urn:entity:place")
        self.assertEqual(entity_urn("urn:heatmap"), "urn:heatmap")
        with self.assertRaises(ValueError):
            entity_urn("restaurant")

    def test_missing_key_fails_without_calling_the_api(self):
        transport = FakeTransport()
        with self.assertRaises(QlooError):
            QlooClient("", transport=transport).search("Wes Anderson")
        self.assertEqual(transport.calls, [])

    def test_rate_limit_is_retried_then_succeeds(self):
        slept = []
        transport = FakeTransport(queue=[(429, {"Retry-After": "1"}, {}), (200, {}, {"results": []})])
        client = QlooClient(KEY, transport=transport, sleep=slept.append)
        self.assertEqual(client.search("anything"), {"results": []})
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(slept, [1.0])

    def test_persistent_error_surfaces_status_and_body(self):
        transport = FakeTransport(queue=[(500, {}, b"boom")] * 3)
        client = QlooClient(KEY, transport=transport, sleep=lambda _: None)
        with self.assertRaises(QlooError) as caught:
            client.search("anything")
        self.assertEqual(caught.exception.status, 500)
        self.assertEqual(caught.exception.body, "boom")
        self.assertEqual(len(transport.calls), 3)

    def test_client_error_is_not_retried(self):
        transport = FakeTransport(queue=[(403, {}, {"error": "forbidden"})])
        client = QlooClient(KEY, transport=transport, sleep=lambda _: None)
        with self.assertRaises(QlooError) as caught:
            client.search("anything")
        self.assertEqual(caught.exception.status, 403)
        self.assertEqual(len(transport.calls), 1)

    def test_repr_does_not_leak_the_key(self):
        self.assertNotIn(KEY, repr(QlooClient(KEY)))


class ShapingTests(unittest.TestCase):
    def test_summary_keeps_evidence_and_trims_noise(self):
        _, _, payload = insights_route({})
        (pick,) = summarize_entities(payload)
        self.assertEqual(pick["id"], "P-1")
        self.assertEqual(pick["type"], "place")
        self.assertEqual(pick["subtype"], "bar")
        self.assertEqual(pick["affinity"], 0.93)
        self.assertEqual(pick["popularity"], 0.81)
        self.assertEqual(pick["where"], "Lisbon, PT")
        self.assertEqual(pick["tags"], ["cocktail bar", "Eclectic"])
        self.assertLessEqual(len(pick["description"]), 241)
        self.assertEqual(pick["drivers"][0], {"signal": "E-WES", "impact": 0.7})

    def test_empty_or_odd_responses_do_not_crash(self):
        self.assertEqual(summarize_entities({}), [])
        self.assertEqual(summarize_entities({"results": {"entities": []}}), [])
        self.assertEqual(summarize_entities({"results": [{"name": "X"}]}), [{"name": "X"}])


class BridgeTests(unittest.TestCase):
    def make(self):
        transport = FakeTransport(routes={"/search": search_route, "/v2/insights": insights_route})
        return QlooClient(KEY, transport=transport), transport

    def test_bridge_resolves_seeds_then_asks_for_insights(self):
        client, transport = self.make()
        result = taste_bridge(client, ["Wes Anderson", "Phoebe Bridgers", "Nonexistent Thing"], "place", city="Lisbon")
        self.assertEqual([c["path"] for c in transport.calls], ["/search", "/search", "/search", "/v2/insights"])
        self.assertEqual(transport.calls[-1]["params"]["signal.interests.entities"], "E-WES,E-PHOEBE")
        self.assertEqual(result["seeds"]["unresolved"], ["Nonexistent Thing"])
        self.assertEqual([r["asked"] for r in result["seeds"]["resolved"]], ["Wes Anderson", "Phoebe Bridgers"])
        driver = result["picks"][0]["drivers"][0]
        self.assertEqual((driver["signal"], driver["name"]), ("E-WES", "Wes Anderson"))

    def test_bridge_stops_when_no_seed_resolves(self):
        client, transport = self.make()
        result = taste_bridge(client, ["Nonexistent Thing"], "place")
        self.assertEqual(result["picks"], [])
        self.assertIn("none of the seeds", result["note"])
        self.assertEqual([c["path"] for c in transport.calls], ["/search"])

    def test_safe_call_reports_errors_as_data(self):
        client = QlooClient(KEY, transport=FakeTransport(queue=[(401, {}, {"error": "bad key"})]), sleep=lambda _: None)
        outcome = safe_call(taste_bridge, client, ["Wes Anderson"], "place")
        self.assertFalse(outcome["ok"])
        self.assertEqual(outcome["status"], 401)
        self.assertEqual(safe_call(taste_bridge, client, [], "place")["error"], "give at least one seed name or audience id")


if __name__ == "__main__":
    unittest.main()
