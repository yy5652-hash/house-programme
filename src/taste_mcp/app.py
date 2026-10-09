"""Web app for the hosted demo: one page, one streaming endpoint.

Keys stay on the server (``QLOO_API_KEY`` and the model key). The browser only ever sees the agent's events and the
finished programme.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .agent import run_brief
from .compare import compare
from .llm import ModelError, default_model
from .qloo import QlooClient, QlooError

WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_REQUEST_CHARS = 600
RATE_LIMIT = 8          # programmes per client
RATE_WINDOW = 600.0     # per ten minutes


def load_env_file(path: Path) -> None:
    """Minimal .env reader for local runs; real environments set variables directly."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if value.strip() and key.strip() not in os.environ:
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


class BriefRequest(BaseModel):
    request: str = Field(min_length=8, max_length=MAX_REQUEST_CHARS)
    compare: bool = True     # also ask the model unaided and check its suggestions against the graph


class RateLimiter:
    def __init__(self, limit: int, window: float, clock=time.monotonic):
        self._limit, self._window, self._clock = limit, window, clock
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, who: str) -> bool:
        now = self._clock()
        with self._lock:
            hits = self._hits[who]
            while hits and now - hits[0] > self._window:
                hits.popleft()
            if len(hits) >= self._limit:
                return False
            hits.append(now)
            return True


def create_app(client: QlooClient = None, model=None, limiter: RateLimiter = None) -> FastAPI:
    load_env_file(Path.cwd() / ".env")
    qloo = client or QlooClient()
    llm = model or default_model()
    limits = limiter or RateLimiter(RATE_LIMIT, RATE_WINDOW)
    app = FastAPI(title="House Programme", docs_url=None, redoc_url=None)

    def ready() -> dict:
        return {"taste_graph": bool(getattr(qloo, "configured", True)), "model": bool(getattr(llm, "configured", True))}

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/health")
    def health():
        return ready()

    @app.get("/api/examples/{name}")
    def example(name: str):
        """A saved real run of one of the page's examples (written by scripts/save_examples.py)."""
        path = WEB_DIR / "examples" / f"{name}.json"
        if not name.isalpha() or not name.islower() or not path.is_file():
            raise HTTPException(status_code=404, detail="No saved run by that name.")
        return FileResponse(path, media_type="application/json")

    @app.post("/api/programme")
    def programme(body: BriefRequest, http: Request):
        state = ready()
        if not all(state.values()):
            missing = [name for name, ok in state.items() if not ok]
            raise HTTPException(503, f"server is missing credentials for: {', '.join(missing)}")
        who = (http.headers.get("x-forwarded-for") or (http.client.host if http.client else "?")).split(",")[0].strip()
        if not limits.allow(who):
            raise HTTPException(429, "Too many programmes from this address; try again in a few minutes.")

        events: "queue.Queue" = queue.Queue()

        def work():
            started = time.monotonic()
            try:
                result = run_brief(llm, qloo, body.request, on_event=events.put)
                result["stats"]["seconds"] = round(time.monotonic() - started, 1)
                events.put({"type": "programme", **result})
                if body.compare:
                    events.put({"type": "comparing"})
                    try:
                        events.put({"type": "comparison", **compare(llm, qloo, body.request, result)})
                    except (ModelError, QlooError, ValueError) as error:
                        events.put({"type": "comparison", "rows": [], "summary": None, "problem": str(error)[:200]})
                    except Exception:  # the programme is already delivered; never turn it into a failure
                        import logging
                        logging.getLogger("taste_mcp").exception("comparison failed")
                        events.put({"type": "comparison", "rows": [], "summary": None, "problem": "it failed unexpectedly"})
            except ModelError as error:
                events.put({"type": "error", "message": error.public})
            except QlooError as error:
                events.put({"type": "error", "message": str(error)[:300]})
            except Exception:  # keep details out of the browser; the server log has the traceback
                import logging
                logging.getLogger("taste_mcp").exception("programme failed")
                events.put({"type": "error", "message": "Something went wrong while building the programme."})
            finally:
                events.put(None)

        threading.Thread(target=work, daemon=True).start()

        def stream():
            while True:
                event = events.get()
                if event is None:
                    break
                yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @app.exception_handler(HTTPException)
    def http_error(_, error: HTTPException):
        return JSONResponse({"error": error.detail}, status_code=error.status_code)

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))


if __name__ == "__main__":
    main()
