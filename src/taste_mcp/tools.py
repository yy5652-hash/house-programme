"""Agent-facing tools built on the Qloo client.

Every tool returns plain dicts that carry the evidence for each pick (affinity, the signals that drove it,
the ids that were resolved) so an agent can cite what the taste graph said rather than paraphrase it.
"""
from __future__ import annotations

from typing import Iterable, List, Mapping, Optional

from .qloo import QlooClient, QlooError, summarize_entities


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
    audiences: Optional[Iterable[str]] = None,
    take: int = 8,
    extra: Optional[Mapping[str, object]] = None,
) -> dict:
    """Cross-domain recommendation: things of ``target_type`` that people who like ``seeds`` also like.

    Example: seeds ``["Wes Anderson", "Phoebe Bridgers"]``, target ``"place"``, city ``"Lisbon"``.
    Returns the picks with affinity and per-seed impact, plus exactly which entities the seeds resolved to.
    """
    if not seeds and not audiences:
        raise ValueError("give at least one seed name or audience id")
    lookup = resolve_entities(client, seeds, seed_types)
    seed_ids = [r["id"] for r in lookup["resolved"] if r.get("id")]
    if seeds and not seed_ids:
        return {"picks": [], "seeds": lookup, "note": "none of the seeds could be resolved; nothing was recommended"}

    response = client.insights(
        target_type,
        entities=seed_ids,
        audiences=list(audiences) if audiences else None,
        location_query=city,
        take=take,
        extra=extra,
    )
    picks = summarize_entities(response)
    names_by_id = {r["id"]: r.get("name") for r in lookup["resolved"]}
    for pick in picks:
        for driver in pick.get("drivers", []):
            if isinstance(driver, dict) and driver.get("signal") in names_by_id:
                driver["name"] = names_by_id[driver["signal"]]
    return {
        "target_type": target_type,
        "city": city,
        "seeds": lookup,
        "picks": picks,
        "note": None if picks else "the taste graph returned no results for these signals and filters",
    }


def safe_call(function, *args, **kwargs) -> dict:
    """Run a tool and turn failures into data the agent can read and act on."""
    try:
        return {"ok": True, "result": function(*args, **kwargs)}
    except QlooError as error:
        return {"ok": False, "error": str(error), "status": error.status, "detail": error.body[:300]}
    except ValueError as error:
        return {"ok": False, "error": str(error)}
