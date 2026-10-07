"""The built-in agent: plans taste-graph lookups, then writes a programme whose every pick is backed by evidence.

Two guarantees the loop enforces itself rather than trusting the model:
  * every tool call is recorded with its arguments and outcome, so the page can show the work;
  * a pick that never appeared in a tool result is removed from the answer and reported as unverified.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Sequence

from .llm import ModelError, extract_json
from .qloo import QlooClient
from .registry import TOOLS, Tool, call_tool

SYSTEM_PROMPT = """You programme culture for a real room: a bar, bookshop, cafe, pop-up or community space.
The host tells you what their regulars already love and where they are. You answer with a short programme the host
can act on this week.

How to work:
1. Resolve the things the audience loves with find_entities (or bridge_tastes) before using them as signals.
2. Ask the taste graph, one kind of thing at a time, what this audience also loves: music (artist), a film to screen
   (movie), a book for the club (book), a podcast to sponsor or play (podcast), brands to stock or partner with
   (brand) and, when a city is given, nearby places to cross-promote with (place). Choose the kinds that fit the request.
   Ask for all of those kinds in the same turn: issue the tool calls together rather than one per turn.
   Use all of the crowd's favourites together as the signals for every kind, not one favourite per kind: what the
   same people love across domains is exactly what the graph knows and a single-domain lookup does not.
   For places, a city alone returns every kind of venue (hotels, surf schools, shops). Decide which two or three
   kinds of neighbour would suit this host (record store, bookstore, movie theater, art gallery, cafe, wine bar, live
   music venue ...) and make one bridge_tastes call per kind with city and category.
   Names that came back from the taste graph are already verified; do not look them up again.
   Never make up an id. An id is only valid if a tool returned it earlier in this conversation; a call that needs
   ids you do not have yet has to wait for the next turn. bridge_tastes takes names, so it can go in the first turn.
3. Use only names that came back from a tool. Never add a pick from memory. If the graph returns nothing for a kind,
   say so in caveats instead of filling the gap.
4. When a seed could not be resolved or resolved to the wrong thing, say so in caveats. Copy the favourites exactly as
   the host wrote them; do not alter or complete a name.
5. Unless the host asks for something narrower, a programme always has music (artist), a film night (movie) and a book
   (book); add podcasts, brands and neighbours when they fit.

Finish with JSON only, no prose around it:
{"headline": "one line the host could put on a poster",
 "audience_read": "two sentences on what ties this crowd together, using the tags and drivers you saw",
 "sections": [{"title": "short heading", "kind": "artist|movie|book|podcast|brand|place|tv_show|video_game|destination",
               "picks": [{"name": "exact name from a tool result", "why": "one sentence, concrete, mentions the driver"}]}],
 "caveats": ["anything unresolved, thin or uncertain"]}
Give three or four picks per section when the graph returned that many, and include every kind that fits the request
and returned results. Section titles are plain headings a host would use ("On the speakers",
"Film night", "Book club", "Neighbours to team up with"), without the kind in brackets."""

Event = Dict[str, Any]


def _compact(value: Any, limit: int = 6000) -> Any:
    """Tool results can be large; cap what is fed back to the model without breaking JSON structure."""
    text = json.dumps(value, ensure_ascii=False)
    if len(text) <= limit:
        return value
    if isinstance(value, dict) and isinstance(value.get("result"), list):
        trimmed = dict(value)
        items = list(value["result"])
        while items and len(json.dumps({**trimmed, "result": items}, ensure_ascii=False)) > limit:
            items.pop()
        trimmed["result"] = items
        trimmed["truncated"] = True
        return trimmed
    return {"ok": value.get("ok", True) if isinstance(value, dict) else True, "truncated": True, "preview": text[:limit]}


_NOT_FOR_THE_MODEL = {"image", "thumb", "website", "popularity", "alternatives", "city"}


def _slim(value: Any) -> Any:
    """Drop fields the page needs but the model does not, before a tool result goes back into the conversation."""
    if isinstance(value, dict):
        return {k: _slim(v) for k, v in value.items() if k not in _NOT_FOR_THE_MODEL}
    if isinstance(value, list):
        return [_slim(v) for v in value]
    return value


_ID_ARGUMENTS = ("entity_ids", "candidate_ids", "tag_ids", "filter_tag_ids", "audience_ids")


def _collect_ids(value: Any, into: set) -> None:
    """Remember every id a tool has returned, so later calls can be checked against them."""
    if isinstance(value, dict):
        if isinstance(value.get("id"), str):
            into.add(value["id"])
        for item in value.values():
            _collect_ids(item, into)
    elif isinstance(value, list):
        for item in value:
            _collect_ids(item, into)


def _invented_ids(arguments: dict, known: set) -> List[str]:
    """Ids in a call that no tool has returned. Models sometimes make ids up when they skip the lookup."""
    return [i for key in _ID_ARGUMENTS for i in (arguments.get(key) or []) if i not in known]


def _is_reference(value: dict) -> bool:
    """Tags and audiences have names too, but they are not things a host can programme."""
    return str(value.get("id", "")).startswith(("urn:tag:", "urn:audience:"))


def _collect_evidence(value: Any, into: Dict[str, dict]) -> None:
    """Index every named entity in a tool result by lower-cased name."""
    if isinstance(value, dict):
        name = value.get("name")
        if isinstance(name, str) and name.strip() and ("id" in value or "affinity" in value) and not _is_reference(value):
            key = name.strip().lower()
            known = into.setdefault(key, {"name": name.strip()})
            for field in ("id", "type", "affinity", "popularity", "where", "tags", "drivers", "detail", "image", "thumb", "website", "rating"):
                if value.get(field) is not None and field not in known:
                    known[field] = value[field]
        for item in value.values():
            _collect_evidence(item, into)
    elif isinstance(value, list):
        for item in value:
            _collect_evidence(item, into)


def _ground(brief: dict, evidence: Dict[str, dict]) -> dict:
    """Attach evidence to each pick; move anything the tools never returned into ``unverified``."""
    kept_total, dropped = 0, []
    sections = []
    used = set()                                   # a thing appears once, in the first section that names it
    names_by_id = {proof["id"]: proof["name"] for proof in evidence.values() if proof.get("id")}
    for section in brief.get("sections") or []:
        picks = []
        for pick in section.get("picks") or []:
            name = str(pick.get("name", "")).strip()
            proof = evidence.get(name.lower())
            if not proof:
                dropped.append({"name": name, "section": section.get("title")})
                continue
            if proof["name"].lower() in used:
                continue
            used.add(proof["name"].lower())
            extra = {k: proof[k] for k in ("id", "affinity", "where", "tags", "detail", "image", "thumb", "website", "rating") if k in proof}
            drivers = [  # show the seed's name, not its id; drop drivers that cannot be named
                {**d, "name": d.get("name") or names_by_id.get(d.get("signal"))}
                for d in proof.get("drivers", []) if isinstance(d, dict)
            ]
            drivers = [d for d in drivers if d["name"]]
            if drivers:
                extra["drivers"] = drivers
            picks.append({**pick, "name": proof["name"], **extra})
        kept_total += len(picks)
        if picks:
            sections.append({**section, "picks": picks})
    return {**brief, "sections": sections, "unverified": dropped,
            "grounding": {"kept": kept_total, "dropped": len(dropped)}}


_STAPLES = ("artist", "movie", "book")      # what a venue can always programme: music, a film night, a book


def _missing_staples(final_text: Optional[str], request: str) -> List[str]:
    """Staple kinds the draft answer left out, unless the host's own words show they did not want them."""
    try:
        draft = extract_json(final_text or "")
    except ValueError:
        return []
    kinds = {section.get("kind") for section in (draft.get("sections") or []) if isinstance(section, dict)} if isinstance(draft, dict) else set()
    wording = request.lower()
    if any(phrase in wording for phrase in ("only ", "just ", "nothing but")):
        return []
    return [kind for kind in _STAPLES if kind not in kinds]


def run_brief(
    model: Any,
    client: QlooClient,
    request: str,
    *,
    tools: Sequence[Tool] = TOOLS,
    max_steps: int = 10,
    on_event: Optional[Callable[[Event], None]] = None,
) -> dict:
    """Run the agent on one request. Returns the grounded brief, the full trace and summary counts."""
    emit = on_event or (lambda event: None)
    messages: List[dict] = [{"role": "user", "text": request.strip()}]
    trace: List[dict] = []
    evidence: Dict[str, dict] = {}
    signal_ids: List[str] = []          # entity ids the run used as audience signals
    known_ids: set = set()              # every id a tool has returned so far
    final_text: Optional[str] = None

    nudged = False
    for step in range(max_steps):
        reply = model.step(SYSTEM_PROMPT, messages, tools)
        calls = reply.get("calls") or []
        if not calls:
            missing = [] if nudged else _missing_staples(reply.get("text"), request)
            if missing and step < max_steps - 1:
                # The model sometimes answers one part of the brief and stops. Ask once for the rest.
                nudged = True
                messages.append({"role": "model", "text": reply.get("text"), **({"raw": reply["raw"]} if reply.get("raw") else {})})
                messages.append({"role": "user", "text": (
                    f"The programme has no section for: {', '.join(missing)}. Ask the taste graph for those kinds too, using all of "
                    "the crowd's favourites as signals, then give the complete final JSON.")})
                continue
            final_text = reply.get("text")
            break
        turn = {"role": "model", "text": reply.get("text"), "calls": calls}
        if reply.get("raw"):
            turn["raw"] = reply["raw"]
        messages.append(turn)
        results = []
        for call in calls:
            emit({"type": "call", "step": step + 1, "tool": call["name"], "args": call.get("args", {})})
            arguments = dict(call.get("args") or {})
            invented = _invented_ids(arguments, known_ids)
            if invented:
                outcome = {"ok": False, "error": f"these ids did not come from a tool result: {invented[:4]}. Look names up "
                                                 "with find_entities or find_tags first and use the ids they return."}
            else:
                outcome = call_tool(client, call["name"], arguments)
            _collect_evidence(outcome, evidence)
            _collect_ids(outcome, known_ids)
            result = outcome.get("result")
            used: List[str] = []
            if outcome.get("ok") and call["name"] in ("recommend", "score_candidates"):
                used = list(arguments.get("entity_ids") or [])
            elif outcome.get("ok") and call["name"] == "bridge_tastes" and isinstance(result, dict):
                used = [r.get("id") for r in (result.get("seeds") or {}).get("resolved", [])]
            signal_ids.extend(i for i in used if i and i not in signal_ids)
            entry = {"step": step + 1, "tool": call["name"], "args": call.get("args", {}), "ok": bool(outcome.get("ok")),
                     "error": outcome.get("error")}
            if isinstance(result, list):
                entry["returned"] = len(result)
            elif isinstance(result, dict) and isinstance(result.get("picks"), list):
                entry["returned"] = len(result["picks"])
            trace.append(entry)
            emit({"type": "result", **entry})
            results.append({"name": call["name"], "response": _compact(_slim(outcome))})
        messages.append({"role": "tool", "results": results})
    else:
        # Out of steps: ask once more for the answer with tools switched off.
        messages.append({"role": "user", "text": "Stop looking things up. Write the final JSON now from what you have."})
        final_text = model.step(SYSTEM_PROMPT, messages, []).get("text")

    try:
        brief = extract_json(final_text or "")
        if not isinstance(brief, dict):
            raise ValueError("final answer was not a JSON object")
    except ValueError as error:
        raise ModelError(f"the model did not return a usable brief: {error}") from error

    grounded = _ground(brief, evidence)
    emit({"type": "done", "grounding": grounded["grounding"]})
    names = {proof["id"]: proof["name"] for proof in evidence.values() if proof.get("id")}
    return {
        "brief": grounded,
        "trace": trace,
        "seeds": [{"id": i, "name": names.get(i)} for i in signal_ids],
        "stats": {"tool_calls": len(trace), "failed_calls": sum(1 for t in trace if not t["ok"]),
                  "entities_seen": len(evidence), **grounded["grounding"]},
    }
