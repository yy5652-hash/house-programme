---
title: House Programme
emoji: 🎟️
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Programme a real room from Qloo's taste graph
---

# House Programme

Programme a bar, bookshop or cafe around what its regulars already love. An agent asks Qloo's taste graph what that
crowd also loves and returns a short programme; every pick shows its affinity score and what drove it.

Source and documentation: https://github.com/yy5652-hash/house-programme

The Space needs two secrets: `QLOO_API_KEY` and `GEMINI_API_KEY` (or `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`).
