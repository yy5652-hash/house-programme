"""Agent-facing tools built on the Qloo client.

Every tool returns plain dicts that carry the evidence for each pick (affinity, the signals that drove it,
the ids that were resolved) so an agent can cite what the taste graph said rather than paraphrase it.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Mapping, Optional

from .qloo import QlooClient, QlooError, summarize_entities, summarize_tags

# What a host says and what the tag is called.
_CATEGORY_ALIASES = {"bookshop": "bookstore", "book shop": "bookstore", "record shop": "record store", "cinema": "movie theater",
                     "gallery": "art gallery", "coffee house": "coffee shop", "music venue": "live music venue"}
_VENUE_TAG = re.compile(r"^urn:tag:(genre|category):place:[a-z0-9_]+$")


def _plain(text: object) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(text or "")) if not unicodedata.combining(c)).lower().strip()


def venue_category(client: QlooClient, name: str) -> Optional[dict]:
    """The tag for a kind of venue ("record store", "bookshop", "wine bar"), or None when the graph has no such category."""
    wanted = _CATEGORY_ALIASES.get(_plain(name), name.strip())
    tags = [tag for tag in summarize_tags(client.tags(wanted, take=10)) if "place" in tag["applies_to"]]
    exact = [tag for tag in tags if _VENUE_TAG.match(tag["id"])]
    loose = [tag for tag in tags if tag["family"] in ("genre", "category")]
    chosen = (exact or loose or [None])[0]
    return {"id": chosen["id"], "name": chosen["name"], "asked": name} if chosen else None


def in_city(places: List[dict], city: str, take: int) -> dict:
    """Keep the places that are actually in ``city``.

    The API reads a city name as its wider region (asking for Lisbon also returns the coast an hour away), so the
    results are checked against each place's own address fields.
    """
    wanted = _plain(city)
    local = [p for p in places if wanted and (_plain(p.get("city")) == wanted or wanted in _plain(p.get("where")).split(", "))]
    if places and not local:
        # Nothing matched by name, which more likely means a spelling difference than an empty city: keep the results.
        return {"picks": places[:take], "outside_city": 0, "unconfirmed": True}
    return {"picks": local[:take], "outside_city": len(places) - len(local)}


def resolve_entities(client: QlooClient, names: Iterable[str], types: Optional[Iterable[str]] = None) -> dict:
    """Resolve free-text names to Qloo entity ids, keeping the misses visible."""
    resolved, unresolved = [], []
    for name in names:
        name = name.strip()
        if not name:
            continue
        hits = summarize_entities(client.search(name, types=types, take=3))
        if hits:
            resolved.append({"asked": name, **hits[0], "alternatives": [h.get("name") for h in hits[1:]]})
        else:
            unresolved.append(name)
    return {"resolved": resolved, "unresolved": unresolved}


def taste_bridge(
    client: QlooClient,
    seeds: List[str],
    target_type: str,
    *,
    seed_types: Optional[Iterable[str]] = None,
    city: Optional[str] = None,
    category: Optional[str] = None,
    audiences: Optional[Iterable[str]] = None,
    take: int = 8,
    extra: Optional[Mapping[str, object]] = None,
) -> dict:
    """Cross-domain recommendation: things of ``target_type`` that people who like ``seeds`` also like.

    Example: seeds ``["Wes Anderson", "Phoebe Bridgers"]``, target ``"place"``, city ``"Lisbon"``, category
    ``"record store"``. Returns the picks with affinity and per-seed impact, plus exactly which entities the seeds
    resolved to and which category tag was applied.
    """
    if not seeds and not audiences:
        raise ValueError("give at least one seed name or audience id")
    lookup = resolve_entities(client, seeds, seed_types)
    seed_ids = [r["id"] for r in lookup["resolved"] if r.get("id")]
    if seeds and not seed_ids:
        return {"picks": [], "seeds": lookup, "note": "none of the seeds could be resolved; nothing was recommended"}

    params = dict(extra or {})
    category_tag = None
    if category:
        category_tag = venue_category(client, category)
        if category_tag is None:
            return {"picks": [], "seeds": lookup, "category": {"asked": category, "id": None},
                    "note": f"the taste graph has no venue category called {category!r}; try another word for it"}
        params["filter.tags"] = [category_tag["id"]]
    local_only = bool(city) and target_type == "place"
    response = client.insights(
        target_type,
        entities=seed_ids,
        audiences=list(audiences) if audiences else None,
        location_query=city,
        take=min(take * 4, 50) if local_only else take,
        extra=params or None,
    )
    picks = summarize_entities(response)
    outside, unconfirmed = 0, False
    if local_only:
        kept = in_city(picks, city, take)
        picks, outside, unconfirmed = kept["picks"], kept["outside_city"], bool(kept.get("unconfirmed"))
    names_by_id = {r["id"]: r.get("name") for r in lookup["resolved"]}
    for pick in picks:
        for driver in pick.get("drivers", []):
            if isinstance(driver, dict) and driver.get("signal") in names_by_id:
                driver["name"] = names_by_id[driver["signal"]]
    out = {
        "target_type": target_type,
        "city": city,
        "seeds": lookup,
        "picks": picks,
        "note": None if picks else "the taste graph returned no results for these signals and filters",
    }
    if category_tag:
        out["category"] = category_tag
    if outside:
        out["left_out"] = f"{outside} results were outside {city} and were dropped"
    if unconfirmed:
        out["left_out"] = f"could not confirm from the addresses which results are inside {city}; check before relying on them"
    return out


def safe_call(function, *args, **kwargs) -> dict:
    """Run a tool and turn failures into data the agent can read and act on."""
    try:
        return {"ok": True, "result": function(*args, **kwargs)}
    except QlooError as error:
        return {"ok": False, "error": str(error), "status": error.status, "detail": error.body[:300]}
    except ValueError as error:
        return {"ok": False, "error": str(error)}
