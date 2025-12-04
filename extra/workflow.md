# Milestone-Based Workflow

## **Milestone 1 — Core Setup (8 hours)**

**Objectives:**

* Set up Python environment
* Connect to Groq API (Llama 4 Maverick)
* Build a simple CLI chatbot loop (terminal-based)

**Tasks:**

1. Project skeleton + folder structure (1 hr)
2. Groq API integration + test prompt (2 hrs)
3. Basic CLI chat interface (2 hrs)
4. Logging & environment setup (.env, config) (1 hr)
5. Minimal conversation prompt (2 hrs)

---

## **Milestone 2 — Intent + Basic Conversation Flow (10 hours)**

**Objectives:**

* Detect user intents (menu request, diet type, mood, order start)
* Build simple conversation logic

**Tasks:**

1. Lightweight intent extraction (LLM prompt or rules) (4 hrs)
2. Handlers for: greetings, veg/vegan/non-veg requests, mood asking (4 hrs)
3. Testing and refinement (2 hrs)

---

## **Milestone 3 — Django Integration (Menu + Ordering) (14 hours)**

**Objectives:**

* Connect bot to Django backend APIs
* Fetch menu + item details
* Start order creation flow

**Tasks:**

1. API wrapper (Python requests/httpx) for Django endpoints (4 hrs)
2. Menu fetching + formatting + LLM summarization (4 hrs)
3. Order flow: selecting item → quantity → confirmation → send payload to Django (4 hrs)
4. Fixes + integration testing (2 hrs)

---

## **Milestone 4 — Mood-Based Recommendation (6 hours)**

**Objectives:**

* Ask user mood
* Map mood → dish recommendations
* Generate natural suggestions

**Tasks:**

1. Simple mood → dish mapping config (2 hrs)
2. Connect mapping to Django menu data (2 hrs)
3. LLM-based conversational explanation (2 hrs)

---

## **Milestone 5 — Final Conversation Polish (6 hours)**

**Objectives:**

* Make bot responses feel human-like
* Add clarifying questions, small talk, service info

**Tasks:**

1. Improve system prompt + tone (2 hrs)
2. Add small talk + fallback responses (2 hrs)
3. Quick usability testing (2 hrs)

---

## **Milestone 6 — Frontend Integration Prep (4 hours)**

**Objectives:**

* Prepare bot for merging with Django/backend
* Ensure message structure is compatible with frontend chat UI

**Tasks:**

1. Define message format (JSON) for frontend (1 hr)
2. Add output wrapper (1 hr)
3. Basic API endpoint outline for backend dev (2 hrs)

---

# **Total Estimated Hours: 48 hours**
