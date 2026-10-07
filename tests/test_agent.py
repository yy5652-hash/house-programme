"""Offline tests for the registry, the model adapter and the agent loop. No network, no keys."""
import json
import os
import sys
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from taste_mcp.agent import run_brief  # noqa: E402
from taste_mcp.llm import GeminiModel, ModelError, OpenAICompatModel, default_model, extract_json  # noqa: E402
from taste_mcp.qloo import QlooClient  # noqa: E402
from taste_mcp.registry import TOOLS, call_tool  # noqa: E402
from test_taste_mcp import KEY, FakeTransport, insights_route, search_route  # noqa: E402


class ScriptedModel:
    """Replays a fixed list of replies and records what it was shown."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def step(self, system, messages, tools, **_):
        self.seen.append({"messages": json.loads(json.dumps(messages)), "tools": [t.name for t in tools]})
        return self.replies.pop(0)


def client_and_transport():
    transport = FakeTransport(routes={"/search": search_route, "/v2/insights": insights_route})
    return QlooClient(KEY, transport=transport), transport


BRIDGE_CALL = {"text": None, "calls": [{"name": "bridge_tastes",
                                        "args": {"seeds": ["Wes Anderson"], "target_type": "place", "city": "Lisbon"}}]}


def final(picks, caveats=()):
    return {"text": json.dumps({"headline": "A night out", "audience_read": "Whimsy and melancholy.",
                                "sections": [{"title": "Where to send them", "kind": "place",
                                              "picks": [{"name": n, "why": "fits"} for n in picks]}],
                                "caveats": list(caveats)}), "calls": []}


class AgentLoopTests(unittest.TestCase):
    def test_picks_are_kept_only_when_a_tool_returned_them(self):
        client, transport = client_and_transport()
        model = ScriptedModel([BRIDGE_CALL, final(["pensão amor", "Invented Speakeasy"])])
        events = []
        result = run_brief(model, client, "Regulars love Wes Anderson. We are in Lisbon.", on_event=events.append)

        picks = result["brief"]["sections"][0]["picks"]
        self.assertEqual([p["name"] for p in picks], ["Pensão Amor"])           # canonical name restored
        self.assertEqual(picks[0]["affinity"], 0.93)                             # evidence attached
        self.assertEqual(picks[0]["drivers"][0]["name"], "Wes Anderson")
        self.assertEqual(result["brief"]["unverified"], [{"name": "Invented Speakeasy", "section": "Where to send them"}])
        self.assertEqual(result["stats"]["kept"], 1)
        self.assertEqual(result["stats"]["dropped"], 1)
        self.assertEqual(result["trace"], [{"step": 1, "tool": "bridge_tastes", "ok": True, "error": None, "returned": 1,
                                            "args": BRIDGE_CALL["calls"][0]["args"]}])
        self.assertEqual([e["type"] for e in events], ["call", "result", "done"])
        self.assertEqual([c["path"] for c in transport.calls], ["/search", "/v2/insights"])
        # the tool result was fed back to the model before it answered
        self.assertEqual(model.seen[1]["messages"][-1]["role"], "tool")

    def test_provider_data_on_a_model_turn_is_passed_back_untouched(self):
        client, _ = client_and_transport()
        model = ScriptedModel([{**BRIDGE_CALL, "raw": [{"opaque": 1}]}, final(["Pensão Amor"])])
        run_brief(model, client, "x")
        self.assertEqual(model.seen[1]["messages"][1]["raw"], [{"opaque": 1}])

    def test_bad_tool_calls_come_back_as_errors_and_the_loop_continues(self):
        client, transport = client_and_transport()
        model = ScriptedModel([
            {"text": None, "calls": [{"name": "teleport", "args": {}},
                                     {"name": "recommend", "args": {"target_type": "place", "radius": 5}}]},
            final([], caveats=["nothing could be looked up"]),
        ])
        result = run_brief(model, client, "anything")
        self.assertEqual([t["ok"] for t in result["trace"]], [False, False])
        self.assertIn("unknown tool", result["trace"][0]["error"])
        self.assertIn("does not take ['radius']", result["trace"][1]["error"])
        self.assertEqual(result["stats"]["failed_calls"], 2)
        self.assertEqual(result["brief"]["sections"], [])
        self.assertEqual(transport.calls, [])

    def test_step_budget_forces_an_answer_without_tools(self):
        client, _ = client_and_transport()
        model = ScriptedModel([BRIDGE_CALL, BRIDGE_CALL, final(["Pensão Amor"])])
        result = run_brief(model, client, "x", max_steps=2)
        self.assertEqual(result["stats"]["tool_calls"], 2)
        self.assertEqual(model.seen[-1]["tools"], [])
        self.assertIn("final JSON", model.seen[-1]["messages"][-1]["text"])
        self.assertEqual(result["stats"]["kept"], 1)

    def test_drivers_are_shown_by_name_when_recommend_returns_ids(self):
        client, _ = client_and_transport()
        model = ScriptedModel([
            {"text": None, "calls": [{"name": "find_entities", "args": {"names": ["Wes Anderson"]}}]},
            {"text": None, "calls": [{"name": "recommend", "args": {"target_type": "place", "entity_ids": ["E-WES"]}}]},
            final(["Pensão Amor"]),
        ])
        pick = run_brief(model, client, "x")["brief"]["sections"][0]["picks"][0]
        # E-WES was resolved earlier so it gets its name; E-PHOEBE was never resolved and is left out
        self.assertEqual(pick["drivers"], [{"signal": "E-WES", "impact": 0.7, "name": "Wes Anderson"}])

    def test_ids_the_tools_never_returned_are_refused_before_reaching_the_api(self):
        client, transport = client_and_transport()
        model = ScriptedModel([
            {"text": None, "calls": [{"name": "recommend", "args": {"target_type": "place", "entity_ids": ["MADE-UP-1"],
                                                                     "filter_tag_ids": ["tag_123"]}}]},
            {"text": None, "calls": [{"name": "find_entities", "args": {"names": ["Wes Anderson"]}}]},
            {"text": None, "calls": [{"name": "recommend", "args": {"target_type": "place", "entity_ids": ["E-WES"]}}]},
            final(["Pensão Amor"]),
        ])
        result = run_brief(model, client, "x")
        self.assertEqual([t["ok"] for t in result["trace"]], [False, True, True])
        self.assertIn("did not come from a tool result", result["trace"][0]["error"])
        self.assertIn("MADE-UP-1", result["trace"][0]["error"])
        self.assertEqual([c["path"] for c in transport.calls], ["/search", "/v2/insights"])     # the bad call cost nothing
        self.assertEqual(result["seeds"], [{"id": "E-WES", "name": "Wes Anderson"}])              # and left no trace in the seeds

    def test_tags_are_reference_data_not_things_to_programme(self):
        def tags_route(params):
            return 200, {}, {"results": {"tags": [{"id": "urn:tag:genre:place:record_store", "name": "Record store",
                                                   "type": "urn:tag:genre:place", "parents": [{"type": "urn:entity:place"}]}]}}
        transport = FakeTransport(routes={"/search": search_route, "/v2/insights": insights_route, "/v2/tags": tags_route})
        client = QlooClient(KEY, transport=transport)
        model = ScriptedModel([
            {"text": None, "calls": [{"name": "find_tags", "args": {"query": "record store"}}, BRIDGE_CALL["calls"][0]]},
            {"text": None, "calls": [{"name": "recommend", "args": {"target_type": "place", "entity_ids": ["E-WES"], "city": "Lisbon",
                                                                     "filter_tag_ids": ["urn:tag:genre:place:record_store"]}}]},
            final(["Pensão Amor", "Record store"]),
        ])
        result = run_brief(model, client, "x")
        self.assertEqual([t["ok"] for t in result["trace"]], [True, True, True])
        self.assertEqual(transport.calls[-1]["params"]["filter.tags"], "urn:tag:genre:place:record_store")
        self.assertEqual(result["brief"]["unverified"], [{"name": "Record store", "section": "Where to send them"}])

    def test_unusable_final_answer_is_an_error(self):
        client, _ = client_and_transport()
        with self.assertRaises(ModelError):
            run_brief(ScriptedModel([{"text": "Sorry, I cannot help with that.", "calls": []}]), client, "x")


class RegistryTests(unittest.TestCase):
    def test_argument_checks(self):
        client, transport = client_and_transport()
        self.assertIn("missing required ['names']", call_tool(client, "find_entities", {})["error"])
        self.assertIn("does not take", call_tool(client, "find_tags", {"query": "wine", "limit": 3})["error"])
        self.assertEqual(transport.calls, [])

    def test_score_candidates_restricts_results_to_the_candidates(self):
        client, transport = client_and_transport()
        outcome = call_tool(client, "score_candidates",
                            {"target_type": "place", "entity_ids": ["E-WES"], "candidate_ids": ["P-1", "P-2"]})
        self.assertTrue(outcome["ok"])
        params = transport.calls[0]["params"]
        self.assertEqual(params["filter.results.entities"], "P-1,P-2")
        self.assertEqual(params["signal.interests.entities"], "E-WES")

    def test_every_tool_declares_a_schema_with_required_fields_present(self):
        for tool in TOOLS:
            self.assertEqual(tool.parameters["type"], "object", tool.name)
            for field in tool.parameters.get("required", []):
                self.assertIn(field, tool.parameters["properties"], tool.name)


class GeminiAdapterTests(unittest.TestCase):
    def make(self, replies, model="gemini-test"):
        calls = []

        def http(method, url, headers, body, timeout):
            calls.append({"method": method, "url": url, "headers": headers, "body": body})
            return replies.pop(0)

        return GeminiModel("secret-key", model, http=http), calls

    def test_request_and_reply_translation(self):
        parts = [
            {"text": "Looking that up."},
            {"functionCall": {"name": "find_tags", "args": {"query": "natural wine"}}, "thoughtSignature": "sig-2"},
        ]
        earlier = [{"functionCall": {"name": "find_tags", "args": {"query": "x"}}, "thoughtSignature": "sig-1"}]
        model, calls = self.make([{"candidates": [{"content": {"parts": parts}}]}])
        out = model.step("SYS", [
            {"role": "user", "text": "hi"},
            {"role": "model", "text": None, "calls": [{"name": "find_tags", "args": {"query": "x"}}], "raw": earlier},
            {"role": "tool", "results": [{"name": "find_tags", "response": {"ok": True}}]},
        ], TOOLS)
        self.assertEqual(out, {"text": "Looking that up.", "calls": [{"name": "find_tags", "args": {"query": "natural wine"}}],
                               "raw": parts})
        # the earlier model turn goes back exactly as Gemini sent it, signature included
        self.assertEqual(calls[0]["body"]["contents"][1]["parts"], earlier)
        sent = calls[0]
        self.assertTrue(sent["url"].endswith("/models/gemini-test:generateContent"))
        self.assertNotIn("secret-key", sent["url"])
        self.assertEqual(sent["headers"]["x-goog-api-key"], "secret-key")
        self.assertEqual(sent["body"]["system_instruction"]["parts"][0]["text"], "SYS")
        self.assertEqual([c["role"] for c in sent["body"]["contents"]], ["user", "model", "user"])
        self.assertIn("functionResponse", sent["body"]["contents"][2]["parts"][0])
        declared = sent["body"]["tools"][0]["functionDeclarations"]
        self.assertEqual(len(declared), len(TOOLS))
        self.assertEqual(declared[0]["parameters"]["type"], "OBJECT")
        self.assertEqual(declared[0]["parameters"]["properties"]["names"]["type"], "ARRAY")

    def test_model_is_chosen_from_what_the_key_can_use(self):
        listing = {"models": [
            {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-3.0-flash-preview", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-3.0-flash", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-3.0-flash-lite", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/embedding-9", "supportedGenerationMethods": ["embedContent"]},
        ]}
        model, _ = self.make([listing], model=None)
        self.assertEqual(model.model, "gemini-3.0-flash-lite")           # free-tier quota is far larger on lite
        without_lite = {"models": [m for m in listing["models"] if "lite" not in m["name"]]}
        model, _ = self.make([without_lite], model=None)
        self.assertEqual(model.model, "gemini-3.0-flash")

    def test_missing_key_and_empty_reply(self):
        with self.assertRaises(ModelError):
            GeminiModel("", "m").step("s", [{"role": "user", "text": "x"}], [])
        model, _ = self.make([{"candidates": []}])
        with self.assertRaises(ModelError):
            model.step("s", [{"role": "user", "text": "x"}], [])
        model, _ = self.make([{"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}}])
        with self.assertRaisesRegex(ModelError, "declined this request"):
            model.step("s", [{"role": "user", "text": "x"}], [])
        self.assertNotIn("secret-key", repr(model))


class OpenAICompatAdapterTests(unittest.TestCase):
    def make(self, replies):
        calls = []

        def http(method, url, headers, body, timeout):
            calls.append({"method": method, "url": url, "headers": headers, "body": body})
            return replies.pop(0)

        return OpenAICompatModel("secret-key", "some-model", "https://llm.example/v1/", http=http), calls

    def test_request_and_reply_translation(self):
        reply = {"choices": [{"message": {"content": " Looking that up. ", "tool_calls": [
            {"id": "x1", "type": "function", "function": {"name": "find_tags", "arguments": '{"query": "natural wine"}'}},
            {"id": "x2", "type": "function", "function": {"name": "find_tags", "arguments": "{not json"}},
        ]}}]}
        model, calls = self.make([reply])
        out = model.step("SYS", [
            {"role": "user", "text": "hi"},
            {"role": "model", "text": None, "calls": [{"name": "find_tags", "args": {"query": "x"}},
                                                      {"name": "list_audiences", "args": {}}]},
            {"role": "tool", "results": [{"name": "find_tags", "response": {"ok": True}},
                                         {"name": "list_audiences", "response": {"ok": False}}]},
        ], TOOLS)
        self.assertEqual(out, {"text": "Looking that up.", "calls": [{"name": "find_tags", "args": {"query": "natural wine"}},
                                                                     {"name": "find_tags", "args": {}}]})
        sent = calls[0]
        self.assertEqual(sent["url"], "https://llm.example/v1/chat/completions")
        self.assertEqual(sent["headers"]["Authorization"], "Bearer secret-key")
        messages = sent["body"]["messages"]
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "tool", "tool"])
        issued = [c["id"] for c in messages[2]["tool_calls"]]
        self.assertEqual([m["tool_call_id"] for m in messages[3:]], issued)      # results answer the calls in order
        self.assertEqual(len(set(issued)), 2)
        self.assertEqual(json.loads(messages[2]["tool_calls"][0]["function"]["arguments"]), {"query": "x"})
        self.assertEqual(json.loads(messages[4]["content"]), {"ok": False})
        self.assertEqual(sent["body"]["tools"][0]["function"]["parameters"]["type"], "object")
        self.assertNotIn("secret-key", repr(model))

    def test_needs_key_model_and_base_url(self):
        self.assertFalse(OpenAICompatModel("k", "m", "").configured)
        with self.assertRaises(ModelError):
            OpenAICompatModel("k", "", "https://llm.example/v1").step("s", [{"role": "user", "text": "x"}], [])
        model, _ = self.make([{"choices": []}])
        with self.assertRaises(ModelError):
            model.step("s", [{"role": "user", "text": "x"}], [])

    def test_default_model_follows_the_credentials_present(self):
        other = {"LLM_API_KEY": "k", "LLM_BASE_URL": "https://llm.example/v1", "LLM_MODEL": "m"}
        with mock.patch.dict(os.environ, other, clear=True):
            self.assertIsInstance(default_model(), OpenAICompatModel)
        with mock.patch.dict(os.environ, {**other, "GEMINI_API_KEY": "g"}, clear=True):
            self.assertIsInstance(default_model(), GeminiModel)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(default_model().configured)


class ExtractJsonTests(unittest.TestCase):
    def test_tolerates_fences_prose_and_braces_inside_strings(self):
        self.assertEqual(extract_json('Here you go:\n```json\n{"a": {"b": "}"}}\n```\nEnjoy'), {"a": {"b": "}"}})
        self.assertEqual(extract_json('prefix {"a": 1} suffix {"b": 2}'), {"a": 1})
        for bad in ("", "no json here", '{"a": 1'):
            with self.assertRaises(ValueError):
                extract_json(bad)


if __name__ == "__main__":
    unittest.main()
