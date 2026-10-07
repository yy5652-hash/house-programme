"""Thin client for the Qloo hackathon API.

Only documented endpoints are used: ``/search``, ``/v2/tags``, ``/v2/audiences`` and ``/v2/insights``.
The API silently ignores parameters it does not know, so parameter names are checked here first:
a typo should fail loudly instead of quietly returning unfiltered results.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Iterable, Mapping, Optional

DEFAULT_BASE_URL = "https://hackathon.api.qloo.com"

# The hackathon guide and the Insights reference spell two types differently; accept both.
ENTITY_TYPES = {
    "artist", "book", "brand", "destination", "locality", "movie", "person",
    "place", "podcast", "tv_show", "video_game", "videogame",
}
SPECIAL_FILTER_TYPES = {"urn:heatmap", "urn:tag", "urn:demographics"}

_PARAM_PREFIXES = (
    "filter.", "signal.", "bias.", "diversify.", "backfill.", "feature.",
    "operator.", "output.", "heatmap.",
)
_PARAM_NAMES = {"take", "page", "offset", "sort_by", "query", "types"}

Transport = Callable[[str, Mapping[str, str], float], "tuple[int, Mapping[str, str], bytes]"]


class QlooError(RuntimeError):
    """Raised for configuration problems and non-2xx API responses."""

    def __init__(self, message: str, status: Optional[int] = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


def entity_urn(kind: str) -> str:
    """``"movie"`` -> ``"urn:entity:movie"``; full URNs pass through unchanged."""
    if kind.startswith("urn:"):
        if kind in SPECIAL_FILTER_TYPES or kind.split(":")[-1] in ENTITY_TYPES:
            return kind
        raise ValueError(f"unknown filter type {kind!r}")
    if kind not in ENTITY_TYPES:
        raise ValueError(f"unknown entity type {kind!r}; expected one of {sorted(ENTITY_TYPES)}")
    return f"urn:entity:{kind}"


def _encode(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, set)):
        return ",".join(_encode(v) for v in value)
    return str(value)


def _check_param(name: str) -> None:
    if name in _PARAM_NAMES or name.startswith(_PARAM_PREFIXES):
        return
    raise ValueError(f"{name!r} is not a documented Qloo parameter; the API would ignore it silently")


def _urllib_transport(url: str, headers: Mapping[str, str], timeout: float):
    request = urllib.request.Request(url, headers=dict(headers), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers or {}), error.read()


class QlooClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        *,
        transport: Optional[Transport] = None,
        timeout: float = 20.0,
        max_retries: int = 2,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._api_key = api_key if api_key is not None else os.environ.get("QLOO_API_KEY", "")
        self.base_url = (base_url or os.environ.get("QLOO_API_URL") or DEFAULT_BASE_URL).rstrip("/")
        self._transport = transport or _urllib_transport
        self._timeout = timeout
        self._max_retries = max_retries
        self._sleep = sleep

    def __repr__(self) -> str:  # never show the key
        return f"QlooClient(base_url={self.base_url!r}, key={'set' if self._api_key else 'missing'})"

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    # -- low level -------------------------------------------------------------------------
    def get(self, path: str, params: Mapping[str, Any]) -> dict:
        if not self._api_key:
            raise QlooError("QLOO_API_KEY is not set")
        clean = {}
        for name, value in params.items():
            if value is None or value == "" or value == [] or value == ():
                continue
            _check_param(name)
            clean[name] = _encode(value)
        url = f"{self.base_url}{path}?{urllib.parse.urlencode(clean)}"
        headers = {"X-Api-Key": self._api_key, "Accept": "application/json"}

        attempt = 0
        while True:
            status, response_headers, body = self._transport(url, headers, self._timeout)
            if 200 <= status < 300:
                try:
                    return json.loads(body.decode("utf-8"))
                except (ValueError, UnicodeDecodeError) as error:
                    raise QlooError(f"{path}: response was not JSON", status, body[:300].decode("utf-8", "replace")) from error
            retryable = status == 429 or status >= 500
            if retryable and attempt < self._max_retries:
                retry_after = {k.lower(): v for k, v in response_headers.items()}.get("retry-after")
                try:
                    delay = float(retry_after) if retry_after else 0.0
                except ValueError:
                    delay = 0.0
                self._sleep(min(max(delay, 0.5 * 2 ** attempt), 10.0))
                attempt += 1
                continue
            raise QlooError(f"{path}: HTTP {status}", status, body[:500].decode("utf-8", "replace"))

    # -- endpoints -------------------------------------------------------------------------
    def search(self, query: str, types: Optional[Iterable[str]] = None, take: int = 5) -> dict:
        """Look up entity ids by name."""
        if not query.strip():
            raise ValueError("search query is empty")
        return self.get("/search", {
            "query": query,
            "types": [entity_urn(t) for t in types] if types else None,
            "take": take,
        })

    def tags(self, query: str, tag_types: Optional[Iterable[str]] = None, take: int = 10) -> dict:
        """Find tag ids to use in ``filter.tags`` / ``signal.interests.tags``."""
        return self.get("/v2/tags", {
            "filter.query": query,
            "filter.tag.types": list(tag_types) if tag_types else None,
            "take": take,
        })

    def audiences(self, parent_types: Optional[Iterable[str]] = None, take: int = 50) -> dict:
        """List audience ids, optionally limited to categories such as ``urn:audience:hobbies_and_interests``."""
        return self.get("/v2/audiences", {
            "filter.parents.types": list(parent_types) if parent_types else None,
            "take": take,
        })

    def insights(
        self,
        filter_type: str,
        *,
        entities: Optional[Iterable[str]] = None,
        tags: Optional[Iterable[str]] = None,
        audiences: Optional[Iterable[str]] = None,
        location_query: Optional[str] = None,
        take: int = 10,
        explain: bool = True,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> dict:
        """Taste-ranked entities of ``filter_type`` for the given signals.

        ``location_query`` is applied as ``filter.location.query`` (restrict results to a place).
        ``extra`` carries any other documented parameter, for example ``{"filter.release_year.min": 2015}``.
        """
        params: dict = {
            "filter.type": entity_urn(filter_type),
            "signal.interests.entities": list(entities) if entities else None,
            "signal.interests.tags": list(tags) if tags else None,
            "signal.demographics.audiences": list(audiences) if audiences else None,
            "filter.location.query": location_query,
            "take": take,
            "feature.explainability": explain if (entities or tags) else None,
        }
        if extra:
            params.update(extra)
        has_signal_or_filter = any(
            value not in (None, "", [], ()) for name, value in params.items()
            if name not in ("filter.type", "take", "feature.explainability", "page", "offset", "sort_by")
        )
        if not has_signal_or_filter:
            raise ValueError("insights needs at least one signal or filter besides filter.type")
        return self.get("/v2/insights", params)


# -- response shaping --------------------------------------------------------------------------
# An entity carries dozens to hundreds of tags. These tag families describe what a thing is like; the rest
# (plot keywords, instruments, nearby attractions, payment options ...) are noise on a programme card.
_TAG_FAMILIES = ("genre", "subgenre", "category", "style", "theme", "ambience", "plot", "critical_descriptor",
                 "emotional_tone", "lifestyle", "aesthetic_property", "cuisine", "setting", "decor")


def _tag_name(tag: Any) -> str:
    if isinstance(tag, Mapping):
        return str(tag.get("name") or tag.get("id") or tag.get("tag_id") or "")
    return str(tag).split(":")[-1].replace("_", " ")


def _tag_family(tag: Any) -> str:
    raw = (tag.get("type") or tag.get("id") or tag.get("tag_id") or "") if isinstance(tag, Mapping) else str(tag)
    parts = raw.split(":")
    return parts[2] if len(parts) > 2 else ""


def _pick_tags(tags: Any, limit: int) -> list:
    """Up to ``limit`` distinct tag names, descriptive families first, original order within a family."""
    ranked = sorted(enumerate(tags or []), key=lambda pair: (
        _TAG_FAMILIES.index(_tag_family(pair[1])) if _tag_family(pair[1]) in _TAG_FAMILIES else len(_TAG_FAMILIES), pair[0]))
    names, seen = [], set()
    for _, tag in ranked:
        name = _tag_name(tag)
        if name and name.lower() not in seen and name.lower() not in _DULL_TAGS:
            seen.add(name.lower())
            names.append(name)
        if len(names) == limit:
            break
    return names


def _kind_and_subtype(entity: Mapping[str, Any]) -> tuple:
    """Insights results say ``type: urn:entity`` and put the kind in ``subtype``; search results use ``types``."""
    for raw in (entity.get("subtype"), entity.get("type"), *(entity.get("types") or [])):
        if isinstance(raw, str) and raw.startswith("urn:entity:"):
            parts = raw.split(":")
            return parts[2], (parts[3] if len(parts) > 3 else None)
    return None, None


def _drivers(explainability: Any) -> list:
    """Which signals pulled this entity up, strongest first.

    Live shape: ``{"signal.interests.entities": [{"entity_id": ..., "score": ...}], "signal.interests.tags": [...]}``.
    A flat ``{id: score}`` mapping is accepted as well.
    """
    found = []
    if isinstance(explainability, Mapping):
        for key, value in explainability.items():
            if isinstance(value, (int, float)):
                found.append((key, float(value)))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Mapping) and isinstance(item.get("score"), (int, float)):
                        signal = item.get("entity_id") or item.get("tag_id") or item.get("id")
                        if signal:
                            found.append((signal, float(item["score"])))
    found.sort(key=lambda pair: -pair[1])
    return [{"signal": signal, "impact": round(score, 3)} for signal, score in found[:5]]


# Tags that say nothing on a card.
_DULL_TAGS = {"place", "shopping", "food", "establishment", "point of interest", "store", "podcasts"}

# The image urls point at originals (posters of 8000 x 12000 pixels exist). Each host has its own way to ask for a small one.
_THUMBNAIL_RULES = (
    (re.compile(r"^(https://lastfm\.freetls\.fastly\.net/i/u/)[^/]+(/.+)$"), r"\g<1>174s\g<2>"),
    (re.compile(r"^(https://m\.media-amazon\.com/images/.+\._V1_)(\.\w+)$"), r"\g<1>UX128_\g<2>"),
    (re.compile(r"^(https://images-na\.ssl-images-amazon\.com/images/S/compressed\.photo\.goodreads\.com/.+/\d+)(\.\w+)$"),
     r"\g<1>._SX98_\g<2>"),
    (re.compile(r"^(https://lh\d\.googleusercontent\.com/.+=)w\d+-h\d+(.*)$"), r"\g<1>w160-h160\g<2>"),
)


def thumbnail(url: Optional[str]) -> Optional[str]:
    """A small version of a known image url; unknown hosts are returned unchanged."""
    if not url:
        return None
    for pattern, replacement in _THUMBNAIL_RULES:
        if pattern.match(url):
            return pattern.sub(replacement, url)
    return url


def _english(values: Any) -> str:
    for item in values or []:
        if isinstance(item, Mapping) and item.get("value") and "en" in (item.get("languages") or ["en"]):
            return str(item["value"])
    return ""


def summarize_entity(entity: Mapping[str, Any], max_tags: int = 6) -> dict:
    """Reduce one API entity to the fields an agent needs to cite it and a page needs to show it."""
    properties = entity.get("properties") or {}
    geocode = properties.get("geocode") or {}
    query = entity.get("query") or {}
    kind, subtype = _kind_and_subtype(entity)
    description = (_english(properties.get("short_descriptions")) or properties.get("description")
                   or properties.get("short_description") or "").strip()
    image = (properties.get("image") or {}).get("url") if isinstance(properties.get("image"), Mapping) else None
    if not image:
        image = next((i.get("url") for i in properties.get("images") or [] if isinstance(i, Mapping) and i.get("url")), None)
    affinity = query.get("affinity")
    place_parts = (properties.get("neighborhood"), geocode.get("city") or geocode.get("name"),
                   geocode.get("admin1_region"), geocode.get("country_code"))
    where, seen = [], set()
    for part in place_parts:
        if part and str(part).lower() not in seen:
            seen.add(str(part).lower())
            where.append(str(part))
    rating = properties.get("business_rating")
    summary = {
        "id": entity.get("entity_id") or entity.get("id"),
        "name": entity.get("name"),
        "type": kind,
        "subtype": subtype,
        "affinity": round(affinity, 3) if isinstance(affinity, (int, float)) else None,
        "popularity": entity.get("popularity", properties.get("popularity")),
        "detail": (entity.get("disambiguation") or "").strip() or None,
        "where": ", ".join(where) or None,
        "city": geocode.get("city"),
        "tags": _pick_tags(entity.get("tags"), max_tags),
        "description": (description[:240] + "…") if len(description) > 240 else (description or None),
        "image": image,
        "thumb": thumbnail(image),
        "website": properties.get("website"),
        "rating": round(rating, 1) if isinstance(rating, (int, float)) else None,
        "drivers": _drivers(query.get("explainability")),
    }
    return {k: v for k, v in summary.items() if v not in (None, [], "")}


def summarize_entities(response: Mapping[str, Any], max_tags: int = 6) -> list:
    results = response.get("results") or {}
    entities = results.get("entities") if isinstance(results, Mapping) else results
    return [summarize_entity(e, max_tags) for e in (entities or [])]


def summarize_tags(response: Mapping[str, Any]) -> list:
    tags = (response.get("results") or {}).get("tags") or []
    return [{"id": t.get("id"), "name": t.get("name"), "family": _tag_family(t),
             "applies_to": [str(p.get("type", "")).split(":")[-1] for p in t.get("parents") or []][:3]}
            for t in tags if isinstance(t, Mapping) and t.get("id")]


def summarize_audiences(response: Mapping[str, Any]) -> list:
    audiences = (response.get("results") or {}).get("audiences") or []
    return [{"id": a.get("id"), "name": a.get("name")} for a in audiences if isinstance(a, Mapping) and a.get("id")]
