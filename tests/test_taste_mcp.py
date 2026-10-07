"""Offline tests. Responses are shaped like the Qloo reference docs; no network and no API key needed."""
import json
import sys
import unittest
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from taste_mcp.qloo import (QlooClient, QlooError, entity_urn, summarize_audiences, summarize_entities,  # noqa: E402
                            summarize_entity, summarize_tags, thumbnail)
from taste_mcp.tools import in_city, resolve_entities, safe_call, taste_bridge  # noqa: E402

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


LIVE_MOVIE = {   # trimmed from a real /v2/insights response, 2026-10-07
    "name": "I'm Thinking of Ending Things", "type": "urn:entity", "subtype": "urn:entity:movie",
    "entity_id": "M-1", "popularity": 0.93, "disambiguation": "2020, Charlie Kaufman",
    "tags": [{"id": "urn:tag:keyword:media:identity", "name": "Identity", "type": "urn:tag:keyword:media"},
             {"id": "urn:tag:keyword:media:billboard", "name": "Billboard", "type": "urn:tag:keyword:media"},
             {"id": "urn:tag:plot:media:psychological", "name": "Psychological", "type": "urn:tag:plot:media"},
             {"id": "urn:tag:genre:media:drama", "name": "Drama", "type": "urn:tag:genre:media"},
             {"id": "urn:tag:genre:media:drama2", "name": "drama", "type": "urn:tag:genre:media"}],
    "properties": {"image": {"url": "https://img.example/poster.jpg"}, "release_year": 2020,
                   "description": "A long description. " * 30,
                   "short_descriptions": [{"value": "Eine junge Frau reist.", "languages": ["de"]},
                                          {"value": "A young woman travels with her new boyfriend.", "languages": ["en"]}]},
    "query": {"affinity": 0.9225934599458028, "measurements": {"audience_growth": 0},
              "explainability": {"signal.interests.entities": [{"entity_id": "E-PHOEBE", "score": 0.2955},
                                                               {"entity_id": "E-WES", "score": 0.3579}]}},
}
LIVE_PLACE = {
    "name": "Amor Records", "type": "urn:entity", "subtype": "urn:entity:place", "entity_id": "P-9",
    "disambiguation": "R. Frei Francisco Foreiro 2A 1150-166 Lisboa Portugal", "tags": [],
    "properties": {"images": [{"url": "https://img.example/shop.jpg", "type": "urn:image:place:unknown"}],
                   "neighborhood": "Anjos", "website": "http://amor.example/", "business_rating": 4.800000190734863,
                   "geocode": {"city": "Lisbon", "name": None, "country_code": "PT", "admin1_region": None}},
    "query": {"affinity": 0.758},
}


class LiveShapeTests(unittest.TestCase):
    def test_insights_entity_as_the_live_api_returns_it(self):
        movie = summarize_entity(LIVE_MOVIE)
        self.assertEqual((movie["type"], movie["affinity"], movie["detail"]), ("movie", 0.923, "2020, Charlie Kaufman"))
        self.assertNotIn("subtype", movie)
        self.assertEqual(movie["tags"], ["Drama", "Psychological", "Identity", "Billboard"])      # descriptive families first, no repeats
        self.assertEqual(movie["description"], "A young woman travels with her new boyfriend.")  # English short text wins
        self.assertEqual(movie["image"], "https://img.example/poster.jpg")
        self.assertEqual(movie["drivers"], [{"signal": "E-WES", "impact": 0.358}, {"signal": "E-PHOEBE", "impact": 0.295}])

    def test_place_fields(self):
        place = summarize_entity(LIVE_PLACE)
        self.assertEqual(place["where"], "Anjos, Lisbon, PT")
        self.assertEqual((place["image"], place["website"], place["rating"]), ("https://img.example/shop.jpg", "http://amor.example/", 4.8))
        self.assertNotIn("drivers", place)

    def test_thumbnails_ask_each_image_host_for_a_small_version(self):
        cases = {
            "https://lastfm.freetls.fastly.net/i/u/770x0/abc.jpg": "https://lastfm.freetls.fastly.net/i/u/174s/abc.jpg",
            "https://m.media-amazon.com/images/M/MV5Bxyz@._V1_.jpg": "https://m.media-amazon.com/images/M/MV5Bxyz@._V1_UX128_.jpg",
            "https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1446469353i/22822858.jpg":
                "https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1446469353i/22822858._SX98_.jpg",
            "https://lh5.googleusercontent.com/p/AF1Qip=w1900-h1900-k-no": "https://lh5.googleusercontent.com/p/AF1Qip=w160-h160-k-no",
            "https://img.example/poster.jpg": "https://img.example/poster.jpg",
        }
        for url, small in cases.items():
            self.assertEqual(thumbnail(url), small)
        self.assertIsNone(thumbnail(None))
        self.assertEqual(summarize_entity(LIVE_PLACE)["thumb"], "https://img.example/shop.jpg")

    def test_tags_that_say_nothing_are_left_off(self):
        entity = {"name": "X", "entity_id": "1", "tags": [{"name": "Place", "type": "urn:tag:category:place"},
                                                          {"name": "Record store", "type": "urn:tag:category:place"},
                                                          {"name": "Shopping", "type": "urn:tag:genre:place"}]}
        self.assertEqual(summarize_entity(entity)["tags"], ["Record store"])

    def test_search_hits_take_their_kind_from_types(self):
        hit = summarize_entity({"name": "Wes Anderson", "entity_id": "E-WES", "types": ["urn:entity:author"], "popularity": 0.97})
        self.assertEqual(hit["type"], "author")

    def test_tags_and_audiences_are_reduced_to_what_a_call_needs(self):
        tags = summarize_tags({"results": {"tags": [{"id": "urn:tag:genre:place:record_store", "name": "Record store",
                                                     "type": "urn:tag:genre:place", "parents": [{"type": "urn:entity:place"}],
                                                     "popularity": 0.9, "properties": {}, "tags": []}, {"name": "no id"}]}})
        self.assertEqual(tags, [{"id": "urn:tag:genre:place:record_store", "name": "Record store", "family": "genre", "applies_to": ["place"]}])
        audiences = summarize_audiences({"results": {"audiences": [{"id": "urn:audience:hobbies_and_interests:vinyl", "name": "Vinyl",
                                                                    "entity_id": "X", "parents": []}]}})
        self.assertEqual(audiences, [{"id": "urn:audience:hobbies_and_interests:vinyl", "name": "Vinyl"}])


def place(name, city, where_name=None, affinity=0.8):
    return {"entity_id": "P-" + name, "name": name, "type": "urn:entity", "subtype": "urn:entity:place",
            "properties": {"geocode": {"city": city, "name": where_name, "country_code": "PT"}}, "query": {"affinity": affinity}}


class ResolveTests(unittest.TestCase):
    HITS = {
        "Patagonia": [("L-1", "Patagonia", "locality"), ("B-1", "Patagonia", "brand")],
        "Banana Yoshiconductor": [("X-1", "Banana Island", "movie")],
        "Ghibli": [("G-1", "Studio Ghibli", "brand")],
    }

    def client(self):
        def search(params):
            return 200, {}, {"results": [{"entity_id": i, "name": n, "types": [f"urn:entity:{k}"]}
                                         for i, n, k in self.HITS.get(params["query"], [])]}
        return QlooClient(KEY, transport=FakeTransport(routes={"/search": search}))

    def test_a_same_named_thing_wins_over_a_place_when_no_kind_was_asked_for(self):
        found = resolve_entities(self.client(), ["Patagonia"])["resolved"][0]
        self.assertEqual((found["id"], found["type"]), ("B-1", "brand"))
        self.assertEqual(found["alternatives"], ["Patagonia (locality)"])

    def test_an_asked_kind_is_respected(self):
        found = resolve_entities(self.client(), ["Patagonia"], ["locality"])["resolved"][0]
        self.assertEqual(found["id"], "L-1")

    def test_a_nearest_match_with_a_different_name_is_a_miss(self):
        out = resolve_entities(self.client(), ["Banana Yoshiconductor", "Ghibli"])
        self.assertEqual(out["unresolved"], ["Banana Yoshiconductor"])
        self.assertEqual([r["name"] for r in out["resolved"]], ["Studio Ghibli"])


class VenueTests(unittest.TestCase):
    TAGS = {"bookstore": [("urn:tag:nearby_attraction:qloo:bookstore", "Bookstore"), ("urn:tag:category:place:book_store", "Book store"),
                          ("urn:tag:genre:place:book_store", "Book store")],
            "wine bar": [("urn:tag:genre:place:restaurant:wine_bar", "Wine bar"), ("urn:tag:dining_option:qloo:wine_bar", "Wine bar")],
            "zeppelin hangar": []}

    def routes(self, entities):
        def tags(params):
            found = self.TAGS.get(params["filter.query"], [])
            return 200, {}, {"results": {"tags": [{"id": i, "name": n, "type": i.rsplit(":", 1)[0], "parents": [{"type": "urn:entity:place"}]}
                                                  for i, n in found]}}
        return {"/search": search_route, "/v2/tags": tags, "/v2/insights": lambda params: (200, {}, {"results": {"entities": entities}})}

    def test_category_is_resolved_by_name_and_results_outside_the_city_are_dropped(self):
        transport = FakeTransport(routes=self.routes([place("Surf Hotel", "Roliça"), place("Livraria A", "Lisbon"),
                                                      place("Livraria B", None, where_name="Lisbon"), place("Casa", "Cascais")]))
        result = taste_bridge(QlooClient(KEY, transport=transport), ["Wes Anderson"], "place", city="Lisbon", category="bookshop", take=3)
        self.assertEqual([p["name"] for p in result["picks"]], ["Livraria A", "Livraria B"])
        self.assertEqual(result["category"], {"id": "urn:tag:category:place:book_store", "name": "Book store", "asked": "bookshop"})
        self.assertEqual(result["left_out"], "2 results were outside Lisbon and were dropped")
        tags_call, insights_call = transport.calls[-2]["params"], transport.calls[-1]["params"]
        self.assertEqual(tags_call["filter.query"], "bookstore")                 # the alias, not the word the host used
        self.assertEqual(insights_call["filter.tags"], "urn:tag:category:place:book_store")
        self.assertEqual(insights_call["take"], "12")                            # asks for more, because some will be dropped

    def test_a_looser_tag_is_used_when_no_plain_venue_tag_exists(self):
        transport = FakeTransport(routes=self.routes([place("Bar X", "Lisbon")]))
        result = taste_bridge(QlooClient(KEY, transport=transport), ["Wes Anderson"], "place", city="Lisbon", category="wine bar")
        self.assertEqual(result["category"]["id"], "urn:tag:genre:place:restaurant:wine_bar")

    def test_an_unknown_category_is_reported_and_costs_no_recommendation_call(self):
        transport = FakeTransport(routes=self.routes([]))
        result = taste_bridge(QlooClient(KEY, transport=transport), ["Wes Anderson"], "place", city="Lisbon", category="zeppelin hangar")
        self.assertEqual(result["picks"], [])
        self.assertIn("no venue category called 'zeppelin hangar'", result["note"])
        self.assertEqual([c["path"] for c in transport.calls], ["/search", "/v2/tags"])

    def test_city_filter_ignores_accents_and_case(self):
        kept = in_city([{"name": "A", "city": "São Paulo"}, {"name": "B", "where": "Pinheiros, Sao Paulo, BR"}, {"name": "C", "city": "Santos"}],
                       "sao paulo", take=5)
        self.assertEqual(([p["name"] for p in kept["picks"]], kept["outside_city"]), (["A", "B"], 1))

    def test_when_no_address_matches_the_results_are_kept_and_flagged(self):
        transport = FakeTransport(routes=self.routes([place("Livraria C", "Lisboa"), place("Livraria D", "Lisboa")]))
        result = taste_bridge(QlooClient(KEY, transport=transport), ["Wes Anderson"], "place", city="Lisbon", category="bookshop")
        self.assertEqual(len(result["picks"]), 2)
        self.assertIn("could not confirm", result["left_out"])


if __name__ == "__main__":
    unittest.main()
