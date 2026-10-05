# House Programme

Programme a real room around what its regulars already love.

A host of a bar, bookshop, cafe or pop-up describes their crowd and their city. An agent asks
[Qloo's taste graph](https://docs.qloo.com/) what that crowd also loves, across music, film, books, podcasts, brands
and nearby places, and returns a short programme. Every pick carries its evidence: the affinity score and which of
the crowd's favourites drove it.

Built for the Qloo Agentic Hackathon.

**Status (5 October 2026):** work in progress. The agent loop has been run end to end with a real model against sample
taste data. The Qloo client follows the published API reference and is covered by offline tests; it has not yet been
run against the live Qloo API, because the hackathon key has not arrived.

## What is in here

| Part | File | What it does |
|---|---|---|
| Qloo client | `src/taste_mcp/qloo.py` | Standard-library client for `/search`, `/v2/tags`, `/v2/audiences`, `/v2/insights`. Rejects undocumented parameter names before sending, because the API ignores them silently. |
| Tool registry | `src/taste_mcp/registry.py` | Six tools with JSON schemas and argument checks, shared by everything below. |
| MCP server | `src/taste_mcp/server.py` | Exposes the same six tools to any MCP client over stdio. |
| Agent | `src/taste_mcp/agent.py` | Tool-calling loop with a step budget and a full trace. |
| Model adapter | `src/taste_mcp/llm.py` | Gemini over REST, or any OpenAI-style provider, with function calling. Standard library only. |
| Web app | `src/taste_mcp/app.py`, `src/taste_mcp/web/index.html` | One page; the agent's steps stream in as it works. |

## Why the taste graph, and not just a language model

A language model asked for "a film night for people who like Phoebe Bridgers" answers from what is written about her.
The taste graph answers from what people who like her also turn out to like, across domains. The agent is not allowed
to mix the two up: after the model writes its programme, the loop checks every pick against the tool results and
**removes anything the taste graph never returned**, listing it as unverified. The page shows how many picks were kept
and how many were removed.

## Run it

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env        # then fill in QLOO_API_KEY and a model key
.venv/bin/house-programme   # http://127.0.0.1:7860
```

The model can be Gemini (`GEMINI_API_KEY`) or any provider with an OpenAI-style chat-completions endpoint
(`LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`). With Gemini the app picks the newest `flash-lite` the key can use: one
programme costs about three model requests, and on the free tier `flash-lite` allows hundreds of requests a day where
`flash` allows twenty. Set `GEMINI_MODEL` to override.

No keys yet? `scripts/dev_fake.py` serves the page against canned sample data on port 7861. With a model key but no
Qloo key, `scripts/live_model_check.py` runs the real model through the whole loop against that same sample data.

Tests are offline and need no keys:

```bash
.venv/bin/python -m unittest discover -s tests
```

## Use the tools from your own agent (MCP)

```json
{
  "mcpServers": {
    "taste": { "command": "taste-mcp", "env": { "QLOO_API_KEY": "..." } }
  }
}
```

Tools: `find_entities`, `find_tags`, `list_audiences`, `recommend`, `bridge_tastes`, `score_candidates`.

## Deploy

The `Dockerfile` listens on port 7860, which is what a Docker Space on Hugging Face expects. Set `QLOO_API_KEY` and
the model key as secrets on the host; they are read on the server and never sent to the browser.

## Licence

MIT. See `LICENSE`.
