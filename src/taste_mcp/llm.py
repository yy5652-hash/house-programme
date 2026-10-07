"""Model adapters for the built-in agent.

The agent only needs one operation: given a system prompt, the conversation so far and the tool list, return either
tool calls or text. ``GeminiModel`` implements that over the Gemini REST API and ``OpenAICompatModel`` over any
provider that speaks the OpenAI chat-completions protocol, both with the standard library only. Tests use a scripted
stand-in. ``default_model`` picks whichever one has credentials.

Conversation format (provider-neutral):
    {"role": "user", "text": "..."}
    {"role": "model", "text": "...", "calls": [{"name": "...", "args": {...}}], "raw": <opaque, optional>}
    {"role": "tool", "results": [{"name": "...", "response": {...}}]}

``raw`` is whatever the provider needs to see again verbatim on the next request (Gemini's thought signatures); the
agent passes it back untouched.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence

from .registry import Tool

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"


class ModelError(RuntimeError):
    """``status`` is the HTTP status when there was one. ``busy`` means another model, or the same one later, may work."""

    def __init__(self, message: str, status: Optional[int] = None, busy: bool = False):
        super().__init__(message)
        self.status = status
        self.busy = busy

    @property
    def public(self) -> str:
        """What a visitor should read: no provider payloads."""
        if self.busy:
            return "The language model is busy right now. Please try again in a minute."
        return str(self)[:300]


def _to_gemini_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """JSON Schema -> the subset Gemini accepts for function parameters (upper-case type names)."""
    out: Dict[str, Any] = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            out["type"] = value.upper()
        elif key == "properties":
            out["properties"] = {name: _to_gemini_schema(sub) for name, sub in value.items()}
        elif key == "items":
            out["items"] = _to_gemini_schema(value)
        elif key in ("description", "enum", "required"):
            out[key] = value
    return out


def _http_json(method: str, url: str, headers: Dict[str, str], body: Optional[dict], timeout: float,
               *, retries: int = 1, sleep: Callable[[float], None] = time.sleep) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    attempt = 0
    while True:
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")
            daily = error.code == 429 and "PerDay" in detail
            busy = error.code in (429, 500, 503)
            # One short wait covers a per-minute limit or a blip; a spent daily quota will not recover, so do not wait on it.
            if busy and attempt < retries and not daily:
                hinted = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', detail)
                try:
                    delay = float(hinted.group(1)) + 1 if hinted else float(error.headers.get("Retry-After") or 0)
                except (TypeError, ValueError):
                    delay = 0.0
                if delay <= 20:
                    sleep(max(delay, 2.0))
                    attempt += 1
                    continue
            if daily:
                raise ModelError("the model's free daily quota is used up", error.code, busy=True) from error
            raise ModelError(f"model API returned HTTP {error.code}: {detail[:400]}", error.code, busy=busy) from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise ModelError(f"model API call failed: {error}", busy=True) from error
        except ValueError as error:
            raise ModelError(f"model API call failed: {error}") from error


def _version_key(name: str):
    numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", name)]
    return (numbers[0] if numbers else 0.0, "preview" not in name and "exp" not in name, -len(name))


class GeminiModel:
    """Gemini over REST. The API key is read from ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) and never logged.

    Unless a model is pinned, the adapter keeps a short list of models the key can use and moves to the next one when
    the current one is overloaded or out of quota. Each model has its own free quota, and a conversation started on one
    continues on another (checked against the live API, thought signatures included).
    """

    SLOW_SECONDS = 12.0     # a model that takes longer than this for one request is treated as having a bad spell

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, *, timeout: float = 30.0,
                 http: Callable[..., dict] = _http_json, clock: Callable[[], float] = time.monotonic):
        self._key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
        pinned = model or os.environ.get("GEMINI_MODEL") or ""
        self._models: List[str] = [pinned] if pinned else []
        self._current = 0
        self._timeout = timeout
        self._http = http
        self._clock = clock

    def __repr__(self) -> str:
        return f"GeminiModel(model={(self._models[self._current] if self._models else 'auto')!r}, key={'set' if self._key else 'missing'})"

    @property
    def configured(self) -> bool:
        return bool(self._key)

    @property
    def model(self) -> str:
        """The model the next request will try first."""
        if not self._models:
            self._models = self._list_models()
        return self._models[self._current]

    def _headers(self) -> Dict[str, str]:
        if not self._key:
            raise ModelError("GEMINI_API_KEY is not set")
        return {"x-goog-api-key": self._key, "Content-Type": "application/json"}

    def _list_models(self) -> List[str]:
        """Text models the key can use, in the order to try them, so the app survives retirements and busy spells.

        ``flash-lite`` comes first: on the free tier it allows hundreds of requests a day where ``flash`` allows twenty.
        """
        listing = self._http("GET", f"{GEMINI_BASE}/models?pageSize=200", self._headers(), None, self._timeout)
        names = [
            m["name"].split("/", 1)[-1] for m in listing.get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
        ]
        ordered: List[str] = []
        for pattern in (r"gemini-\d+(\.\d+)?-flash-lite", r"gemini-\d+(\.\d+)?-flash"):
            ordered += sorted((n for n in names if re.fullmatch(pattern, n)), key=_version_key, reverse=True)[:2]
        if not ordered and names:
            ordered = [max(names, key=_version_key)]
        if not ordered:
            raise ModelError("no model that supports generateContent is available to this key")
        return ordered

    def _generate(self, body: dict) -> dict:
        """Try the current model, then the others in order; stay on whichever one answered, unless it was slow."""
        self.model                                # makes sure the list is loaded
        last: Optional[ModelError] = None
        for offset in range(len(self._models)):
            index = (self._current + offset) % len(self._models)
            url = f"{GEMINI_BASE}/models/{self._models[index]}:generateContent"
            started = self._clock()
            try:
                reply = self._http("POST", url, self._headers(), body, self._timeout)
            except ModelError as error:
                if not error.busy:
                    raise
                last = error
                continue
            slow = self._clock() - started > self.SLOW_SECONDS
            self._current = (index + 1) % len(self._models) if slow else index   # an answer, but start elsewhere next time
            return reply
        raise last if last else ModelError("no model is configured")

    @staticmethod
    def _contents(messages: Sequence[dict]) -> List[dict]:
        contents = []
        for message in messages:
            role = message["role"]
            if role == "user":
                contents.append({"role": "user", "parts": [{"text": message["text"]}]})
            elif role == "model":
                parts = list(message.get("raw") or [])  # carries the thought signatures Gemini insists on getting back
                if not parts:
                    if message.get("text"):
                        parts.append({"text": message["text"]})
                    for call in message.get("calls", []):
                        parts.append({"functionCall": {"name": call["name"], "args": call.get("args", {})}})
                contents.append({"role": "model", "parts": parts})
            elif role == "tool":
                contents.append({"role": "user", "parts": [
                    {"functionResponse": {"name": r["name"], "response": r["response"]}} for r in message["results"]
                ]})
        return contents

    def step(self, system: str, messages: Sequence[dict], tools: Sequence[Tool], *, temperature: float = 0.3) -> dict:
        body: Dict[str, Any] = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": self._contents(messages),
            "generationConfig": {"temperature": temperature},
        }
        if tools:
            body["tools"] = [{"functionDeclarations": [
                {"name": t.name, "description": t.description, "parameters": _to_gemini_schema(t.parameters)} for t in tools
            ]}]
        reply = self._generate(body)
        candidates = reply.get("candidates") or []
        if not candidates:
            reason = (reply.get("promptFeedback") or {}).get("blockReason")
            if reason:
                raise ModelError(f"the model declined this request ({reason}); try wording it differently")
            raise ModelError("the model returned an empty reply")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if "text" in p and not p.get("thought")).strip()
        calls = [{"name": p["functionCall"]["name"], "args": dict(p["functionCall"].get("args") or {})}
                 for p in parts if "functionCall" in p]
        return {"text": text or None, "calls": calls, "raw": parts}


class OpenAICompatModel:
    """Any provider with an OpenAI-style ``/chat/completions`` endpoint.

    Configured with ``LLM_API_KEY``, ``LLM_BASE_URL`` (for example ``https://api.groq.com/openai/v1``) and ``LLM_MODEL``.
    The key is never logged.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, base_url: Optional[str] = None, *,
                 timeout: float = 60.0, http: Callable[..., dict] = _http_json):
        self._key = api_key or os.environ.get("LLM_API_KEY") or ""
        self.model = model or os.environ.get("LLM_MODEL") or ""
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or "").rstrip("/")
        self._timeout = timeout
        self._http = http

    def __repr__(self) -> str:
        return (f"OpenAICompatModel(base_url={self.base_url!r}, model={self.model!r}, "
                f"key={'set' if self._key else 'missing'})")

    @property
    def configured(self) -> bool:
        return bool(self._key and self.model and self.base_url)

    @staticmethod
    def _messages(system: str, messages: Sequence[dict]) -> List[dict]:
        """The neutral format has no call ids; they are made up per turn and matched to results by position."""
        out: List[dict] = [{"role": "system", "content": system}]
        last_ids: List[str] = []
        for turn, message in enumerate(messages):
            role = message["role"]
            if role == "user":
                out.append({"role": "user", "content": message["text"]})
            elif role == "model":
                calls = message.get("calls", [])
                last_ids = [f"call_{turn}_{n}" for n in range(len(calls))]
                entry: Dict[str, Any] = {"role": "assistant", "content": message.get("text") or None}
                if calls:
                    entry["tool_calls"] = [
                        {"id": call_id, "type": "function",
                         "function": {"name": call["name"], "arguments": json.dumps(call.get("args", {}), ensure_ascii=False)}}
                        for call_id, call in zip(last_ids, calls)
                    ]
                out.append(entry)
            elif role == "tool":
                for n, result in enumerate(message["results"]):
                    call_id = last_ids[n] if n < len(last_ids) else f"call_{turn}_{n}"
                    out.append({"role": "tool", "tool_call_id": call_id,
                                "content": json.dumps(result["response"], ensure_ascii=False)})
        return out

    def step(self, system: str, messages: Sequence[dict], tools: Sequence[Tool], *, temperature: float = 0.3) -> dict:
        if not self.configured:
            raise ModelError("LLM_API_KEY, LLM_BASE_URL and LLM_MODEL must all be set")
        body: Dict[str, Any] = {"model": self.model, "messages": self._messages(system, messages),
                                "temperature": temperature}
        if tools:
            body["tools"] = [{"type": "function", "function": {
                "name": t.name, "description": t.description, "parameters": t.parameters}} for t in tools]
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        reply = self._http("POST", f"{self.base_url}/chat/completions", headers, body, self._timeout)
        choices = reply.get("choices") or []
        if not choices:
            raise ModelError(f"model returned no choices: {json.dumps(reply.get('error', {}))[:200]}")
        message = choices[0].get("message") or {}
        calls = []
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            raw = function.get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except ValueError:
                args = {}  # the registry reports the missing arguments back to the model
            calls.append({"name": function.get("name", ""), "args": args if isinstance(args, dict) else {}})
        text = message.get("content")
        return {"text": text.strip() or None if isinstance(text, str) else None, "calls": calls}


def default_model():
    """Gemini when its key is set, otherwise an OpenAI-compatible provider configured through ``LLM_*``."""
    gemini = GeminiModel()
    if gemini.configured:
        return gemini
    other = OpenAICompatModel()
    return other if other.configured else gemini


def extract_json(text: str) -> Any:
    """Parse the first JSON object in a model reply, tolerating code fences and surrounding prose."""
    if not text:
        raise ValueError("empty reply")
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("{")
    if start < 0:
        raise ValueError("no JSON object in reply")
    depth, in_string, escaped = 0, False, False
    for index in range(start, len(candidate)):
        char = candidate[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(candidate[start:index + 1])
    raise ValueError("unterminated JSON object in reply")
