"""Offline tests for the web app: scripted model, fake taste-graph transport."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient  # noqa: E402

from taste_mcp.app import RateLimiter, create_app  # noqa: E402
from taste_mcp.qloo import QlooClient  # noqa: E402
from test_agent import BRIDGE_CALL, ScriptedModel, client_and_transport, final  # noqa: E402

ASK = {"request": "Regulars love Wes Anderson. We are in Lisbon.", "compare": False}


KeyedModel = ScriptedModel  # a scripted model needs no key, so it always counts as configured


def events_of(response):
    return [json.loads(line[6:]) for line in response.text.split("\n\n") if line.startswith("data: ")]


class AppTests(unittest.TestCase):
    def make(self, replies, limiter=None):
        client, transport = client_and_transport()
        return TestClient(create_app(client, KeyedModel(replies), limiter)), transport

    def test_page_and_health(self):
        web, _ = self.make([])
        page = web.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("House Programme", page.text)
        self.assertEqual(web.get("/api/health").json(), {"taste_graph": True, "model": True})

    def test_programme_streams_steps_then_the_grounded_result(self):
        web, transport = self.make([BRIDGE_CALL, final(["Pensão Amor", "Made Up Place"])])
        response = web.post("/api/programme", json=ASK)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        events = events_of(response)
        self.assertEqual([e["type"] for e in events], ["call", "result", "done", "programme"])
        last = events[-1]
        self.assertEqual([p["name"] for p in last["brief"]["sections"][0]["picks"]], ["Pensão Amor"])
        self.assertEqual(last["brief"]["unverified"][0]["name"], "Made Up Place")
        self.assertIn("seconds", last["stats"])
        self.assertNotIn("test-key-not-real", response.text)          # the taste-graph key never reaches the browser
        self.assertEqual(len(transport.calls), 2)

    def test_model_failure_is_reported_without_internals(self):
        web, _ = self.make([{"text": "no json", "calls": []}])
        events = events_of(web.post("/api/programme", json=ASK))
        self.assertEqual(events[-1]["type"], "error")
        self.assertIn("usable brief", events[-1]["message"])

    def test_input_limits_and_rate_limit(self):
        web, _ = self.make([BRIDGE_CALL, final([])], limiter=RateLimiter(1, 600))
        self.assertEqual(web.post("/api/programme", json={"request": "hi"}).status_code, 422)
        self.assertEqual(web.post("/api/programme", json={"request": "x" * 601}).status_code, 422)
        self.assertEqual(web.post("/api/programme", json=ASK).status_code, 200)
        limited = web.post("/api/programme", json=ASK)
        self.assertEqual(limited.status_code, 429)
        self.assertIn("Too many", limited.json()["error"])

    def test_missing_credentials_are_reported_not_hidden(self):
        web = TestClient(create_app(QlooClient(""), KeyedModel([])))
        self.assertEqual(web.get("/api/health").json(), {"taste_graph": False, "model": True})
        refused = web.post("/api/programme", json=ASK)
        self.assertEqual(refused.status_code, 503)
        self.assertIn("taste_graph", refused.json()["error"])

    def test_rate_limiter_window(self):
        now = [0.0]
        limiter = RateLimiter(2, 10, clock=lambda: now[0])
        self.assertEqual([limiter.allow("a"), limiter.allow("a"), limiter.allow("a"), limiter.allow("b")], [True, True, False, True])
        now[0] = 11
        self.assertTrue(limiter.allow("a"))


if __name__ == "__main__":
    unittest.main()
