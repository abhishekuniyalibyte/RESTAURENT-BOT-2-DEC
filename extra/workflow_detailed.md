# Milestone 1 — High-Level Architecture

## High-level architecture (components & responsibilities)

1. **Frontend (website)**

   * Presents chat UI (websocket or HTTP polling). Sends user messages to Bot API. Renders streaming responses. (Handled by frontend team.)

2. **Bot Service (your code; Python CLI for development)** — *stateless or small session cache*:

   * Message preprocessor, intent/entity extractor (lightweight rules + LLM), dialog manager (state machine + slot filling), RAG retriever, LLM prompt manager, tool/function caller (to call ordering, menu, payment functions), response post-processor (personality/phrasing), and logging/metrics.

3. **Django Backend (backend dev)** — *stateful*:

   * User accounts, menus, prices, inventory, order creation, payment, POS integration, notifications, and webhooks. Exposes REST/GraphQL endpoints for bot to call.

4. **Vector DB for RAG** (Qdrant / Pinecone / Weaviate / Milvus) — stores menu descriptions, policies, FAQs, images metadata and sample dialogs.

5. **Embedding service** — compute embeddings for documents and short context; can use Sentence-Transformers locally or managed embedding APIs (OpenAI, Hugging Face). (You can also check for Groq-provided embedding models if needed.)

6. **Observability & Logging** — Sentry, Prometheus/Grafana, ELK/CloudWatch.

7. **Optional**: Payment provider (Stripe/Paytm), Analytics (Mixpanel), Email/SMS gateways.

---

# Milestone 2 — Tools, Libraries & Development Stack

## Tools / Libraries (recommended)

### Core (Python)

* `groq` SDK (official) or REST usage. Use environment variable `GROQ_API_KEY`. ([GroqCloud][3])
* HTTP client: `httpx` (async), `requests` (sync)
* Websocket (if streaming to frontend directly): `websockets` or Django Channels on backend
* Async orchestration: `asyncio`, `anyio`
* CLI dev tooling: `typer` or `argparse` for CLI commands
* Logging: `structlog` or Python `logging`
* Rate-limiting / throttling: `aio-limiter` or custom token bucket
* Serialization/validation: `pydantic` v1/v2 (fast data models)

### Conversation / NLU / RAG

* Intent/slot extraction: small rule set + LLM fallback; optional: `rasa`, `snips`
* Vector DB: Qdrant / Pinecone / Weaviate
* Embeddings: sentence-transformers or managed embeddings
* Retrieval toolkit: LlamaIndex or LangChain (or your custom version)

### Dev / Infra

* Docker
* GitHub Actions / GitLab CI
* Secrets manager
* Monitoring stack

---

# Milestone 3 — Conversation Flow & Dialog Manager

## Conversation flow & dialog manager (detailed)

### Primary flows

1. Greeting / small talk
2. Browse menu
3. Mood-based recommendation
4. Order flow (transactional)
5. Services info / FAQs via RAG
6. Fallback & escalation

### Dialog manager approach

* FSM for ordering
* LLM for free-form content + intent extraction
* Flow:

  1. Receive user text
  2. Intent/entity extraction
  3. FSM for transactional intents
  4. RAG + LLM for informational intents

### Mood → food mapping (example)

* Happy → shareable plates, desserts
* Cozy/Comfort → soups, sandwiches, rice bowls
* Adventurous → specials, fusion
* Stressed → mild, soothing dishes

---

# Milestone 4 — RAG (Retrieval-Augmented Generation)

## RAG (Retrieval-Augmented Generation)

* Index: menus, dish descriptions, allergies, policies, FAQs
* Embeddings: updated when menu changes
* Retrieval: top-k
* Prompting: include only retrieved docs, avoid long context

---

# Milestone 5 — Tool / Function Calling

## Tool / function calling (recommended)

* Use schema-based function calling patterns
* Examples:

  * `list_menu`
  * `get_item_details`
  * `add_to_cart`
  * `create_order`

Groq supports tool calling patterns.

---

# Milestone 6 — Prompt Engineering & Safety

## Prompt engineering (templates & safety)

* Two-layer prompting: system + user
* System rules: concise, confirm orders, never invent prices
* Include retrieved documents only
* Safety: ask clarifying questions when confidence < 95%

---

# Milestone 7 — CLI Development Prototype

## Sample terminal (CLI) Python snippet — Groq chat streaming (development)

```python
# file: cli_chat.py
import os
from groq import Groq

def stream_chat(messages, model="meta-llama/llama-4-maverick-17b-128e-instruct"):
    # Ensure env var: export GROQ_API_KEY=...
    client = Groq()
    # Note: 'stream=True' yields an iterator of chunks
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_completion_tokens=800,
        top_p=1.0,
        stream=True,
    )
    try:
        for chunk in completion:
            delta = chunk.choices[0].delta
            text = delta.content or ""
            print(text, end="", flush=True)
    except KeyboardInterrupt:
        print("\n[stream interrupted]")

if __name__ == "__main__":
    # Example system + user messages
    system = {"role":"system","content":"You are a friendly restaurant assistant. Keep responses brief and helpful."}
    user = {"role":"user","content":"I'm in a cozy mood and want vegetarian suggestions for dinner."}
    stream_chat([system, user])
```

---

# Milestone 8 — Backend Integration

## Integration points with Django backend

1. Auth & user linking
2. Menu fetch
3. Order creation
4. Payment
5. Webhooks

---

# Milestone 9 — Testing, Metrics & Monitoring

## Testing, metrics & monitoring

* Unit tests
* Integration tests
* A/B testing
* Metrics: order success rate, fallback rate, cost, tokens
* Observability stack

---

# Milestone 10 — Security, Privacy & Compliance

## Security, privacy & compliance

* Store PII only in Django
* HTTPS and secret rotation
* PCI compliance delegates to provider
* Input sanitization

---

# Milestone 11 — Cost & Rate-Limit Planning

## Cost & rate-limit considerations (Groq specifics)

* Model pricing & limits
* Efficient prompt strategy
* Token cost monitoring
* Caching frequent replies

---

# Milestone 12 — Implementation Roadmap

## Implementation roadmap (milestones)

1. M0 — project setup, env, API keys, Groq quickstart
2. M1 — core CLI bot service + basic menu fetch
3. M2 — FSM ordering + Django integration
4. M3 — RAG + vector DB
5. M4 — mood mapping + A/B testing
6. M5 — monitoring, rate limiting, security, production deployment
