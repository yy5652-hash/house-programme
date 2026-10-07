"""One tool registry shared by the MCP server and the built-in agent.

Each tool has a JSON-schema description (so any function-calling model can use it) and a callable that takes the
Qloo client plus keyword arguments. Keeping a single list means the hosted demo and third-party MCP clients see
exactly the same capabilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from .qloo import QlooClient, summarize_audiences, summarize_entities, summarize_tags
from .tools import in_city, resolve_entities, safe_call, taste_bridge

ENTITY_KINDS = ["artist", "book", "brand", "destination", "movie", "person", "place", "podcast", "tv_show", "video_game"]
AUDIENCE_CATEGORIES = [
    "communities", "global_issues", "hobbies_and_interests", "investing_interests", "leisure", "life_stage",
    "lifestyle_preferences_beliefs", "political_preferences", "professional_area", "spending_habits",
]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    run: Callable[..., Any]


def _strings(description: str) -> dict:
    return {"type": "array", "items": {"type": "string"}, "description": description}


def score_candidates(client: QlooClient, target_type: str, entity_ids: List[str], candidate_ids: List[str]) -> list:
    """Affinity of specific candidates for an audience defined by ``entity_ids``.

    Used to check picks that did not come from the taste graph, for example a model's unaided suggestions.
    """
    if not candidate_ids:
        return []
    response = client.insights(
        target_type, entities=entity_ids, take=min(len(candidate_ids), 50), explain=False,
        extra={"filter.results.entities": candidate_ids},
    )
    return summarize_entities(response)


def _recommend(client: QlooClient, target_type, entity_ids, tag_ids, audience_ids, city, filter_tag_ids, take: int) -> list:
    local_only = bool(city) and target_type == "place"
    picks = summarize_entities(client.insights(
        target_type, entities=entity_ids, tags=tag_ids, audiences=audience_ids, location_query=city,
        take=min(take * 4, 50) if local_only else take,
        extra={"filter.tags": filter_tag_ids} if filter_tag_ids else None))
    return in_city(picks, city, take)["picks"] if local_only else picks


TOOLS: List[Tool] = [
    Tool(
        "find_entities",
        "Resolve names of films, artists, brands, places, books, shows, podcasts, games or people to taste-graph ids. "
        "Always do this before using a name as a signal. Names that cannot be resolved are returned, not dropped.",
        {"type": "object", "properties": {
            "names": _strings("Names to look up, one per item."),
            "types": {"type": "array", "items": {"type": "string", "enum": ENTITY_KINDS},
                      "description": "Optional entity kinds to narrow the search."},
        }, "required": ["names"]},
        lambda client, names, types=None: resolve_entities(client, names, types),
    ),
    Tool(
        "find_tags",
        "Search taste-graph tags such as genres, cuisines, styles and venue categories. Returns tag ids to use as "
        "signals (tag_ids) or to narrow results (filter_tag_ids); applies_to says which kinds a tag belongs to.",
        {"type": "object", "properties": {
            "query": {"type": "string", "description": "Word or phrase, e.g. 'natural wine' or 'shoegaze'."},
            "take": {"type": "integer", "description": "How many tags to return (default 10)."},
        }, "required": ["query"]},
        lambda client, query, take=10: summarize_tags(client.tags(query, take=int(take))),
    ),
    Tool(
        "list_audiences",
        "List audience segments in one category. Returns audience ids for use as signals.",
        {"type": "object", "properties": {
            "category": {"type": "string", "enum": AUDIENCE_CATEGORIES},
            "take": {"type": "integer"},
        }, "required": ["category"]},
        lambda client, category, take=50: summarize_audiences(client.audiences([f"urn:audience:{category}"], take=int(take))),
    ),
    Tool(
        "recommend",
        "Taste-ranked entities of one kind for resolved signals. Each result carries an affinity score and, when "
        "entity or tag signals are given, which signal drove it. Use city to restrict places and destinations, and "
        "filter_tag_ids to keep only results of one category (a city alone returns every kind of venue).",
        {"type": "object", "properties": {
            "target_type": {"type": "string", "enum": ENTITY_KINDS},
            "entity_ids": _strings("Entity ids from find_entities."),
            "tag_ids": _strings("Tag ids from find_tags, used as taste signals."),
            "audience_ids": _strings("Audience ids from list_audiences."),
            "city": {"type": "string", "description": "City or locality name, for places only."},
            "filter_tag_ids": _strings("Tag ids every result must carry, e.g. a venue category."),
            "take": {"type": "integer", "description": "How many results (default 8, max 50)."},
        }, "required": ["target_type"]},
        lambda client, target_type, entity_ids=None, tag_ids=None, audience_ids=None, city=None, filter_tag_ids=None, take=8:
            _recommend(client, target_type, entity_ids, tag_ids, audience_ids, city, filter_tag_ids, int(take)),
    ),
    Tool(
        "bridge_tastes",
        "Shortcut that needs no ids: resolve free-text seeds and recommend one kind of thing in a single call. Shows "
        "what each seed resolved to and the per-seed impact on every pick. For places give the city and a category; "
        "results outside the city are dropped.",
        {"type": "object", "properties": {
            "seeds": _strings("Things the audience already loves, by name."),
            "target_type": {"type": "string", "enum": ENTITY_KINDS},
            "city": {"type": "string", "description": "City name, for places."},
            "category": {"type": "string", "description": "Kind of venue to keep, in plain words: record store, bookstore, "
                                                          "movie theater, art gallery, cafe, wine bar, live music venue. Places only."},
            "take": {"type": "integer"},
        }, "required": ["seeds", "target_type"]},
        lambda client, seeds, target_type, city=None, category=None, take=8:
            taste_bridge(client, seeds, target_type, city=city, category=category, take=int(take)),
    ),
    Tool(
        "score_candidates",
        "Check how well specific candidates fit an audience. Give resolved audience entity ids and resolved candidate "
        "ids; returns the candidates the taste graph recognises for that audience, with affinity.",
        {"type": "object", "properties": {
            "target_type": {"type": "string", "enum": ENTITY_KINDS},
            "entity_ids": _strings("Audience signal ids."),
            "candidate_ids": _strings("Candidate entity ids to score."),
        }, "required": ["target_type", "entity_ids", "candidate_ids"]},
        lambda client, target_type, entity_ids, candidate_ids:
            score_candidates(client, target_type, entity_ids, candidate_ids),
    ),
]

TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}


def call_tool(client: QlooClient, name: str, arguments: Dict[str, Any]) -> dict:
    """Run a registered tool; unknown tools and bad arguments come back as data, never as exceptions."""
    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        return {"ok": False, "error": f"unknown tool {name!r}; available: {sorted(TOOLS_BY_NAME)}"}
    allowed = set(tool.parameters.get("properties", {}))
    unexpected = sorted(set(arguments) - allowed)
    if unexpected:
        return {"ok": False, "error": f"{name} does not take {unexpected}; allowed: {sorted(allowed)}"}
    missing = [key for key in tool.parameters.get("required", []) if arguments.get(key) in (None, "", [])]
    if missing:
        return {"ok": False, "error": f"{name} is missing required {missing}"}
    try:
        return safe_call(tool.run, client, **arguments)
    except TypeError as error:
        return {"ok": False, "error": f"{name}: {error}"}
