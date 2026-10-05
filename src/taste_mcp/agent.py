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
3. Use only names that came back from a tool. Never add a pick from memory. If the graph returns nothing for a kind,
   say so in caveats instead of filling the gap.
4. When a seed could not be resolved or resolved to the wrong thing, say so in caveats.

Finish with JSON only, no prose around it:
{"headline": "one line the host could put on a poster",
 "audience_read": "two sentences on what ties this crowd together, using the tags and drivers you saw",
 "sections": [{"title": "short heading", "kind": "artist|movie|book|podcast|brand|place|tv_show|video_game|destination",
               "picks": [{"name": "exact name from a tool result", "why": "one sentence, concrete, mentions the driver"}]}],
 "caveats": ["anything unresolved, thin or uncertain"]}
Keep to at most four picks per section."""

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


def _collect_evidence(value: Any, into: Dict[str, dict]) -> None:
    """Index every named entity in a tool result by lower-cased name."""
    if isinstance(value, dict):
        name = value.get("name")
        if isinstance(name, str) and name.strip() and ("id" in value or "affinity" in value):
            key = name.strip().lower()
            known = into.setdefault(key, {"name": name.strip()})
            for field in ("id", "type", "affinity", "popularity", "where", "tags", "drivers"):
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
    names_by_id = {proof["id"]: proof["name"] for proof in evidence.values() if proof.get("id")}
    for section in brief.get("sections") or []:
        picks = []
        for pick in section.get("picks") or []:
            name = str(pick.get("name", "")).strip()
            proof = evidence.get(name.lower())
            if not proof:
                dropped.append({"name": name, "section": section.get("title")})
                continue
            extra = {k: proof[k] for k in ("id", "affinity", "where", "tags") if k in proof}
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
    final_text: Optional[str] = None

    for step in range(max_steps):
        reply = model.step(SYSTEM_PROMPT, messages, tools)
        calls = reply.get("calls") or []
        if not calls:
            final_text = reply.get("text")
            break
        turn = {"role": "model", "text": reply.get("text"), "calls": calls}
        if reply.get("raw"):
            turn["raw"] = reply["raw"]
        messages.append(turn)
        results = []
        for call in calls:
            emit({"type": "call", "step": step + 1, "tool": call["name"], "args": call.get("args", {})})
            outcome = call_tool(client, call["name"], dict(call.get("args") or {}))
            _collect_evidence(outcome, evidence)
            entry = {"step": step + 1, "tool": call["name"], "args": call.get("args", {}), "ok": bool(outcome.get("ok")),
                     "error": outcome.get("error")}
            result = outcome.get("result")
            if isinstance(result, list):
                entry["returned"] = len(result)
            elif isinstance(result, dict) and isinstance(result.get("picks"), list):
                entry["returned"] = len(result["picks"])
            trace.append(entry)
            emit({"type": "result", **entry})
            results.append({"name": call["name"], "response": _compact(outcome)})
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
    return {
        "brief": grounded,
        "trace": trace,
        "stats": {"tool_calls": len(trace), "failed_calls": sum(1 for t in trace if not t["ok"]),
                  "entities_seen": len(evidence), **grounded["grounding"]},
    }
