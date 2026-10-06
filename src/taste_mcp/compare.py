"""Second opinion: what the model would suggest on its own, checked against the taste graph.

The programme itself only contains what the graph returned. This asks the same model the same question with no tools,
then looks every one of its suggestions up: is it in the graph at all, and how strongly does the graph tie it to this
crowd compared with what made the programme?
"""
from __future__ import annotations

from typing import Any, Dict, List

from .llm import extract_json
from .qloo import QlooClient, QlooError
from .registry import ENTITY_KINDS, score_candidates
from .tools import resolve_entities

UNAIDED_PROMPT = (
    "A venue host describes their regulars. Using only what you already know, with no tools, name things this crowd "
    "would probably also love. Answer with JSON only: "
    '{{"picks": [{{"kind": "<one of: {kinds}>", "name": "<proper name>"}}]}}. Give at most {n} per kind.'
)


def unaided_picks(model: Any, request: str, kinds: List[str], per_kind: int = 3) -> List[dict]:
    reply = model.step(UNAIDED_PROMPT.format(kinds=", ".join(kinds), n=per_kind), [{"role": "user", "text": request.strip()}], [])
    data = extract_json(reply.get("text") or "")
    picks, seen, count = [], set(), {kind: 0 for kind in kinds}
    for item in (data.get("picks") if isinstance(data, dict) else None) or []:
        if not isinstance(item, dict):
            continue
        kind, name = item.get("kind"), str(item.get("name") or "").strip()
        if kind in count and name and name.lower() not in seen and count[kind] < per_kind:
            seen.add(name.lower())
            count[kind] += 1
            picks.append({"kind": kind, "name": name})
    return picks


def compare(model: Any, client: QlooClient, request: str, result: dict, *, per_kind: int = 3, max_kinds: int = 4) -> dict:
    """Return one row per unaided suggestion with a status:

    in_programme   the graph chose it too
    scored         in the graph, with an affinity for this crowd (``affinity`` set)
    no_affinity    in the graph, but it returned no affinity for this crowd
    not_in_graph   the name could not be found
    unchecked      the lookup failed, or there were no resolved signals to score against
    """
    kinds: List[str] = []
    programme: Dict[str, Any] = {}
    for section in result["brief"].get("sections") or []:
        kind = section.get("kind")
        if kind in ENTITY_KINDS and kind not in kinds:
            kinds.append(kind)
        for pick in section.get("picks") or []:
            programme[str(pick.get("name", "")).lower()] = pick.get("affinity")
    kinds = kinds[:max_kinds]
    if not kinds:
        return {"rows": [], "summary": None}

    seed_ids = [seed["id"] for seed in result.get("seeds") or [] if seed.get("id")]
    affinities = [a for a in programme.values() if isinstance(a, (int, float))]
    floor = min(affinities) if affinities else None
    picks = unaided_picks(model, request, kinds, per_kind)
    rows: List[dict] = []
    problem = None
    for kind in kinds:
        names = [p["name"] for p in picks if p["kind"] == kind]
        if not names:
            continue
        found: Dict[str, dict] = {}
        scored: Dict[str, Any] = {}
        failed = False
        try:
            found = {r["asked"]: r for r in resolve_entities(client, names, [kind])["resolved"]}
            ids = [r["id"] for r in found.values() if r.get("id")]
            if ids and seed_ids:
                scored = {e["id"]: e.get("affinity") for e in score_candidates(client, kind, seed_ids, ids) if e.get("id") in ids}
        except (QlooError, ValueError) as error:
            failed, problem = True, str(error)[:160]
        for name in names:
            hit = found.get(name)
            row = {"kind": kind, "name": name}
            canonical = (hit or {}).get("name") or name
            if hit and canonical != name:
                row["resolved_as"] = canonical
            if name.lower() in programme or canonical.lower() in programme:
                row["status"] = "in_programme"
                affinity = programme.get(canonical.lower(), programme.get(name.lower()))
                if isinstance(affinity, (int, float)):
                    row["affinity"] = affinity
            elif failed:
                row["status"] = "unchecked"
            elif not hit:
                row["status"] = "not_in_graph"
            elif not seed_ids:
                row["status"] = "unchecked"
            elif isinstance(scored.get(hit.get("id")), (int, float)):
                row["status"], row["affinity"] = "scored", scored[hit["id"]]
            else:
                row["status"] = "no_affinity"
            rows.append(row)

    counts = {status: sum(1 for r in rows if r["status"] == status)
              for status in ("in_programme", "scored", "no_affinity", "not_in_graph", "unchecked")}
    below = sum(1 for r in rows if r["status"] == "scored" and floor is not None and r["affinity"] < floor)
    out = {"rows": rows, "summary": {"suggested": len(rows), **counts, "scored_below_programme": below, "programme_floor": floor}}
    if problem:
        out["problem"] = problem
    return out
