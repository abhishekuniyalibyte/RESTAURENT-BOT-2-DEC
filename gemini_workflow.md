# Project Workflow: AI Concierge for Restaurant Website

## 1. Project Overview
**Goal:** Develop a conversational AI agent ("Leo") that functions as a virtual waiter. The agent will engage users naturally, analyze their mood to recommend specific menu items, answer service queries, and structure orders for the backend system.

**Deployment Context:**
* **Interface:** Command Line Interface (Terminal) for development/testing.
* **Production Goal:** Integration into a Django-based restaurant website.
* **Core Feature:** Mood-based Food Recommendation Engine using RAG (Retrieval-Augmented Generation).

---

## 2. Technical Stack & Tools

### AI & Machine Learning Core
* **Large Language Model (LLM):** Llama 4 Maverick (via Groq API).
    * *Reasoning:* Selected for high-speed inference and "Mixture of Experts" architecture, balancing cost (Free Tier) and performance.
* **Vector Database:** ChromaDB.
    * *Reasoning:* Local, open-source database to store menu items as mathematical vectors for semantic search.
* **Embedding Model:** `all-MiniLM-L6-v2` (via Sentence-Transformers).
    * *Reasoning:* Lightweight, local model to convert text to vectors without API latency or cost.

### Infrastructure & Environment
* **Language:** Python 3.10+.
* **Environment Management:** `.env` for secure API key storage.
* **Backend Framework:** Django (Managed by Backend Developer).

---

## 3. System Architecture

The system follows a **Retrieval-Augmented Generation (RAG)** pattern.

### The Data Flow
1.  **User Input:** The customer types a message (e.g., "I had a rough day, I need food").
2.  **Semantic Search (Retrieval):**
    * The system converts the input into a vector embedding.
    * It queries ChromaDB for menu items associated with tags like "comfort," "warm," or "sad."
3.  **Context Construction:**
    * Relevant menu items (e.g., "Truffle Mac & Cheese") are retrieved.
    * These items are injected into a hidden system instruction block.
4.  **LLM Processing (Generation):**
    * Llama 4 Maverick receives the conversation history + the retrieved menu items + the "Waiter Persona."
    * It generates a human-like response tailored to the user's mood.
5.  **Action Extraction:**
    * If the user places an order, the AI generates a structured data object (JSON format).
    * This object is parsed and prepared for the Django Backend.

---

## 4. Detailed Workflow Steps

### Phase 1: Knowledge Base Initialization (Data Ingestion)
* **Objective:** Teach the AI the menu without hard-coding it into the prompt.
* **Process:**
    1.  Receive raw menu data (Name, Price, Ingredients, Mood Tags) from the database schema.
    2.  Process this text through the Embedding Model to create vector representations.
    3.  Store vectors in the ChromaDB collection `restaurant_menu`.
    4.  *Outcome:* The AI now "knows" the menu conceptually (e.g., it knows a burger is "heavy/dinner" food).

### Phase 2: The Agent Persona & Logic
* **Objective:** Define how the bot behaves.
* **Persona Definition:** The "System Prompt" configures the AI as a charismatic waiter. It is strictly instructed to:
    * Ask for mood if unknown.
    * Never hallucinate menu items (stick to retrieved context).
    * Adopt a tone that matches the user (empathetic for sad users, energetic for happy users).

### Phase 3: The Runtime Loop (Terminal Interface)
* **Objective:** Handle the live conversation.
* **Step-by-Step Logic:**
    1.  **Listen:** Wait for user text input.
    2.  **Retrieve:** Query the vector database for the top 3 most relevant dishes based on the input text.
    3.  **Augment:** Create a prompt that looks like: *"User said X. Here are the 3 dishes we have that match X. Respond as the waiter."*
    4.  **Generate:** Send payload to Groq API.
    5.  **Display:** Print the AI's textual response to the terminal.

### Phase 4: Backend Handoff (Integration)
* **Objective:** Connect the AI logic to the Django transaction system.
* **Mechanism:** Structured Output Parsing.
    * The AI is instructed to output a specific data format (JSON) when a transaction is confirmed.
    * *Example Payload Structure:* Action Type (Order), Item IDs, Total Price.
    * The Python script detects this payload, strips it from the chat view, and (conceptually) sends it to the Django API endpoint.

---

## 5. Roles & Responsibilities

### AI Engineer (You)
* Prompt Engineering (Persona design, safety rails).
* RAG Pipeline implementation (ChromaDB setup, Embedding logic).
* LLM Integration (Groq API connection).
* Conversation Logic (State management in Python).

### Backend Developer (Django)
* Providing the "Source of Truth" menu data (SQL Database).
* Creating API Endpoints to receive the Order Payload from the AI.
* Handling payment processing and order tickets based on the AI's data.
