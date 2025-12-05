import os
import json
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict
from pathlib import Path
import re

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer, util

load_dotenv()

# ============================================
# Configuration
# ============================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMBEDDINGS_PATH = Path("menu_embeddings.npy")
CHUNKS_PATH = Path("text_chunks.json")
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# ============================================
# Global state (loaded once)
# ============================================
_embed_model = None
_embeddings = None
_text_chunks = None
_groq_client = None


# Common restaurant typos / aliases
COMMON_TYPO_MAP = {
    "desert": "dessert",
    "deserts": "desserts",
    # you can add more later here...
}
COMMON_TYPO_MAP = {
    "desert": "dessert",
    "deserts": "desserts",
    # you can add more later here...
}

def build_search_items_reply(
    user_query: str,
    normalized_term: str,
    retrieved_items: List[Dict[str, any]],
) -> str:
    """
    Build a clean, bullet-style reply for SEARCH_ITEM queries
    using the semantic search results.
    """
    # Try to find a primary category from the results
    primary_category = None
    for item in retrieved_items:
        parsed = item.get("parsed") or {}
        cat = parsed.get("category")
        if cat:
            primary_category = cat
            break

    # Decide heading text
    heading_term = primary_category or (normalized_term.title() if normalized_term else None)

    lines: List[str] = []
    if heading_term:
        lines.append(f"Here are some {heading_term} options from our menu:")
    else:
        lines.append("Here are some items from our menu:")

    # Group items by category for nicer layout
    items_by_category: Dict[str, List[Dict[str, str]]] = {}
    for item in retrieved_items:
        parsed = item.get("parsed") or {}
        name = parsed.get("name") or ""
        price = parsed.get("price") or ""
        category = parsed.get("category") or ""
        if not name:
            continue
        items_by_category.setdefault(category, []).append(
            {"name": name, "price": price}
        )

    # Build bullet lines
    for category, items in items_by_category.items():
        lines.append("")  # blank line before each category
        if category:
            lines.append(f"**{category}**")
        for it in items:
            price = it["price"]
            if price:
                lines.append(f"• {it['name']} — ₹{price}")
            else:
                lines.append(f"• {it['name']}")

    lines.append("")
    lines.append("Tell me which one you'd like to add!")

    return "\n".join(lines)


def normalize_search_term(term: str) -> str:
    """
    Normalize common user typos / variants for menu search.
    E.g. 'desert' -> 'dessert' in restaurant context.
    """
    if not term:
        return term

    original = term

    # 1) Lowercase + strip spaces
    t = term.strip().lower()

    # 2) Remove punctuation like ?, !, . , etc.
    t = re.sub(r"[^a-z0-9\s]", "", t)

    # 3) Collapse multiple spaces
    t = re.sub(r"\s+", " ", t).strip()

    # 4) Desert → dessert (handle phrases, not just exact word)
    #    e.g. "desert", "in desert", "what desert" → "dessert"
    if "desert" in t and "dessert" not in t:
        t = "dessert"

    # 5) Apply explicit typo map if we still want it
    t = COMMON_TYPO_MAP.get(t, t)

    print(f"[RAG] normalize_search_term: '{original}' -> '{t}'")
    return t


# Track last-known modification times so we can hot-reload embeddings
_emb_last_mtime: Optional[float] = None
_chunks_last_mtime: Optional[float] = None


def load_rag_system():
    """
    Load embeddings, chunks, and models (called once on startup
    or after a forced reload).
    """
    global _embed_model, _embeddings, _text_chunks, _groq_client
    global _emb_last_mtime, _chunks_last_mtime

    # If everything is already loaded, don't do the work again.
    if _embed_model is not None and _embeddings is not None and _text_chunks is not None:
        return

    print("[RAG] Loading embedding model and data...")

    # 1) Load sentence transformer (kept in memory; doesn't change when files change)
    if _embed_model is None:
        _embed_model = SentenceTransformer(MODEL_NAME)

    # 2) Load embeddings from disk
    if EMBEDDINGS_PATH.exists():
        _embeddings = np.load(EMBEDDINGS_PATH)
        _emb_last_mtime = EMBEDDINGS_PATH.stat().st_mtime
        print(f"[RAG] Loaded embeddings: {_embeddings.shape}")
    else:
        raise FileNotFoundError(f"Embeddings not found at {EMBEDDINGS_PATH}")

    # 3) Load text chunks from disk
    if CHUNKS_PATH.exists():
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            _text_chunks = json.load(f)
        _chunks_last_mtime = CHUNKS_PATH.stat().st_mtime
        print(f"[RAG] Loaded {len(_text_chunks)} text chunks")
    else:
        raise FileNotFoundError(f"Text chunks not found at {CHUNKS_PATH}")

    # 4) Initialize Groq client
    if GROQ_API_KEY:
        if _groq_client is None:
            _groq_client = Groq(api_key=GROQ_API_KEY)
        print("[RAG] Groq client initialized")
    else:
        print("[RAG] Warning: GROQ_API_KEY not set. LLM features disabled.")


def reload_rag_system():
    """
    Force reload of embeddings and chunks from disk
    in *this* process.

    Used by Celery or manually when you know the files changed.
    """
    global _embed_model, _embeddings, _text_chunks
    global _emb_last_mtime, _chunks_last_mtime

    # We keep the model object (it's big, doesn't depend on menu)
    # and only force re-load of embeddings + chunks.
    _embeddings = None
    _text_chunks = None
    _emb_last_mtime = None
    _chunks_last_mtime = None

    load_rag_system()


def ensure_latest_embeddings():
    """
    Ensure this process has the latest version of embeddings / text chunks.

    Called at the start of semantic_search(). If Celery has regenerated
    menu_embeddings.npy / text_chunks.json, this will detect the changed
    modification times and reload the arrays into the Django process.
    """
    global _embed_model, _embeddings, _text_chunks
    global _emb_last_mtime, _chunks_last_mtime

    # Make sure we have at least one load
    if _embed_model is None or _embeddings is None or _text_chunks is None:
        load_rag_system()
        return

    # If files are missing, nothing to do
    if not EMBEDDINGS_PATH.exists() or not CHUNKS_PATH.exists():
        return

    current_emb_mtime = EMBEDDINGS_PATH.stat().st_mtime
    current_chunks_mtime = CHUNKS_PATH.stat().st_mtime

    # First time we see mtimes – just record and continue
    if _emb_last_mtime is None or _chunks_last_mtime is None:
        _emb_last_mtime = current_emb_mtime
        _chunks_last_mtime = current_chunks_mtime
        return

    # If nothing changed, we’re good
    if (
        current_emb_mtime == _emb_last_mtime
        and current_chunks_mtime == _chunks_last_mtime
    ):
        return

    # Files changed on disk → reload arrays only
    print("[RAG] Detected updated embeddings on disk; reloading in this process...")

    _embeddings = np.load(EMBEDDINGS_PATH)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        _text_chunks = json.load(f)

    _emb_last_mtime = current_emb_mtime
    _chunks_last_mtime = current_chunks_mtime

    print(
        f"[RAG] Reloaded embeddings: {_embeddings.shape}, "
        f"chunks: {len(_text_chunks)}"
    )


def semantic_search(query: str, top_k: int = 5) -> List[Dict[str, any]]:
    """
    Search for menu items using semantic similarity.
    Returns list of dicts with 'text', 'score', 'parsed' info.
    """
    # 🔁 Make sure this Django process sees the latest files that Celery wrote
    ensure_latest_embeddings()

    # By here we know _embed_model, _embeddings, _text_chunks are all loaded
    query_emb = _embed_model.encode(query, convert_to_numpy=True)
    scores = util.cos_sim(query_emb, _embeddings)[0]

    # Get top_k results
    top_indices = scores.argsort(descending=True)[:top_k].tolist()

    results = []
    for idx in top_indices:
        text = _text_chunks[idx]
        score = float(scores[idx])

        # Parse the text chunk: "Category: X. Item: Y. Price: Z"
        parsed = parse_chunk_text(text)

        results.append(
            {
                "text": text,
                "score": score,
                "parsed": parsed,
            }
        )

    return results

def parse_chunk_text(chunk: str) -> Dict[str, str]:
    """
    Parse chunk like 'Category: Breads. Item: Butter Naan. Price: 50'
    Returns dict with 'category', 'name', 'price'
    """
    parts = {}
    for segment in chunk.split(". "):
        if ": " in segment:
            key, value = segment.split(": ", 1)
            parts[key.lower()] = value
    
    return {
        "category": parts.get("category", ""),
        "name": parts.get("item", ""),
        "price": parts.get("price", "")
    }

@dataclass
class ChatbotResult:
    """
    Unified result format for chatbot responses.
    """
    intent: str  # ADD_ITEM, REMOVE_ITEM, SHOW_CART, SHOW_MENU, CLEAR_CART, CONFIRM_ORDER, HELP
    reply: str
    item_id: Optional[int] = None
    quantity: int = 1
    item_name: Optional[str] = None
    confidence: float = 1.0  # New: confidence score
    suggestions: Optional[List[Dict[str, str]]] = None  # NEW: menu suggestions for UI


def classify_intent_with_llm(message: str) -> Dict[str, any]:
    """
    Use Groq LLM to classify user intent and extract entities.
    Returns dict with: intent, item_name, quantity, confidence
    """
    if _groq_client is None:
        load_rag_system()
    
    if not _groq_client:
        # Fallback to rule-based if no LLM
        return {"intent": "HELP", "confidence": 0.5}
    
    prompt = f"""You are an intelligent restaurant ordering assistant. Your task is to analyze the user's message and extract their intent and any relevant details.

AVAILABLE INTENTS:
1. ADD_ITEM - User wants to add food to their cart (e.g., "I want...", "add...", "get me...", "I'll have...")
2. REMOVE_ITEM - User wants to remove food from cart (e.g., "remove...", "delete...", "take out...")
3. SHOW_CART - User wants to see their current order (e.g., "my cart", "what's in my order", "show order")
4. SHOW_MENU - User wants to see the COMPLETE menu with NO specific focus (ONLY: "menu", "show menu", "see menu")
5. CLEAR_CART - User wants to empty their entire cart (e.g., "clear cart", "start over", "cancel order")
6. CONFIRM_ORDER - User is ready to place the order (e.g., "confirm", "place order", "checkout", "I'm done")
7. SEARCH_ITEM - User is asking about specific items, categories, or types WITHOUT wanting to add (e.g., "do you have...", "what X do you have", "tell me about...")
8. HELP - User needs assistance or message is unclear

CRITICAL DISTINCTION:
- "show menu" / "menu" = SHOW_MENU (wants to see everything)
- "what do you have in desserts?" = SEARCH_ITEM with item_name="desserts"
- "what vegetarian dishes?" = SEARCH_ITEM with item_name="vegetarian"
- "what's in your breads section?" = SEARCH_ITEM with item_name="breads"
- "do you have biryani?" = SEARCH_ITEM with item_name="biryani"

INSTRUCTIONS:
- Extract the dish name or category exactly as mentioned (lowercase)
- For quantity, look for numbers or words like "two", "three" (default is 1)
- Be smart: "I want biryani" is ADD_ITEM, but "what desserts do you have?" is SEARCH_ITEM
- For greetings or unclear messages, use HELP
- If asking ABOUT items/categories (not requesting to add), use SEARCH_ITEM

USER MESSAGE: "{message}"

RESPOND WITH ONLY THIS JSON FORMAT (no markdown, no extra text):
{{"intent": "ADD_ITEM", "item_name": "paneer tikka", "quantity": 2}}

EXAMPLES:
Input: "add 2 butter naan"
Output: {{"intent": "ADD_ITEM", "item_name": "butter naan", "quantity": 2}}

Input: "I want three masala dosa"
Output: {{"intent": "ADD_ITEM", "item_name": "masala dosa", "quantity": 3}}

Input: "get me some biryani"
Output: {{"intent": "ADD_ITEM", "item_name": "biryani", "quantity": 1}}

Input: "menu"
Output: {{"intent": "SHOW_MENU", "item_name": null, "quantity": 1}}

Input: "show me the menu"
Output: {{"intent": "SHOW_MENU", "item_name": null, "quantity": 1}}

Input: "what do you have in desserts?"
Output: {{"intent": "SEARCH_ITEM", "item_name": "desserts", "quantity": 1}}

Input: "what desserts do you have?"
Output: {{"intent": "SEARCH_ITEM", "item_name": "desserts", "quantity": 1}}

Input: "tell me about your breads"
Output: {{"intent": "SEARCH_ITEM", "item_name": "breads", "quantity": 1}}

Input: "what vegetarian options?"
Output: {{"intent": "SEARCH_ITEM", "item_name": "vegetarian", "quantity": 1}}

Input: "what's in my cart?"
Output: {{"intent": "SHOW_CART", "item_name": null, "quantity": 1}}

Input: "remove paneer tikka"
Output: {{"intent": "REMOVE_ITEM", "item_name": "paneer tikka", "quantity": 1}}

Input: "clear my order"
Output: {{"intent": "CLEAR_CART", "item_name": null, "quantity": 1}}

Input: "place the order"
Output: {{"intent": "CONFIRM_ORDER", "item_name": null, "quantity": 1}}

Input: "do you have rajma chawal?"
Output: {{"intent": "SEARCH_ITEM", "item_name": "rajma chawal", "quantity": 1}}

Input: "is paneer available?"
Output: {{"intent": "SEARCH_ITEM", "item_name": "paneer", "quantity": 1}}

Input: "I want something spicy"
Output: {{"intent": "SEARCH_ITEM", "item_name": "spicy", "quantity": 1}}

Input: "what do you have?"
Output: {{"intent": "SEARCH_ITEM", "item_name": "dishes", "quantity": 1}}

Input: "hello"
Output: {{"intent": "HELP", "item_name": null, "quantity": 1}}

Input: "what can you do?"
Output: {{"intent": "HELP", "item_name": null, "quantity": 1}}

NOW ANALYZE THE USER MESSAGE AND RESPOND WITH JSON ONLY:"""
    
    try:
        response = _groq_client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=250
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Clean JSON response - handle markdown code blocks
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
        
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        # Parse JSON
        result = json.loads(response_text)
        
        # Validate required fields
        if "intent" not in result:
            result["intent"] = "HELP"
        if "quantity" not in result:
            result["quantity"] = 1
        
        # Set confidence based on intent clarity
        result["confidence"] = 0.9
        
        print(f"[RAG] LLM parsed intent: {result['intent']}, item: {result.get('item_name')}, qty: {result.get('quantity')}")
        
        return result
        
    except json.JSONDecodeError as e:
        print(f"[RAG] JSON parsing error: {e}")
        print(f"[RAG] Raw response: {response_text}")
        return {"intent": "HELP", "confidence": 0.3}
    except Exception as e:
        print(f"[RAG] LLM classification error: {e}")
        return {"intent": "HELP", "confidence": 0.3}


def generate_conversational_response(user_query: str, retrieved_items: List[Dict[str, any]]) -> str:
    """
    Generate natural conversational responses using LLM with retrieved menu context.
    Similar to qa_menu.py's ask_llm function.
    """
    if not _groq_client:
        # Fallback to simple response if no LLM
        return "I found some items that might interest you."
    
    # Build context string from retrieved items
    context_lines = []
    for item in retrieved_items:
        text = item.get("text", "")
        if text:
            context_lines.append(f"- {text}")
    
    context_string = "\n".join(context_lines)
    
    system_prompt = (
        "You are a friendly and helpful restaurant assistant. "
        "Answer questions about our menu naturally and conversationally. "
        "Be warm, welcoming, and enthusiastic about our dishes. "
        "When the user asks about a category (like desserts, breads, vegetarian), describe the options in that category. "
        "Always mention prices when describing items. "
        "If describing dishes, be appetizing and helpful. "
        "Keep responses concise (2-4 sentences) unless more detail is requested. "
        "If the user seems interested in an item, you can suggest they add it to their cart. "
        "Focus ONLY on the items provided in the context - don't make up items that aren't listed."
    )
    
    user_prompt = (
        f"User question: {user_query}\n\n"
        f"Relevant menu items:\n{context_string}\n\n"
        "Provide a natural, friendly response based on the menu information above."
    )
    
    try:
        response = _groq_client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.6,
            max_tokens=350
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"[RAG] Conversational response error: {e}")
        # Fallback to listing items
        item_names = [item["parsed"]["name"] for item in retrieved_items if item.get("parsed", {}).get("name")]
        if item_names:
            return f"I found these items: {', '.join(item_names)}."
        return "I found some items that might interest you."


def parse_message(message: str, restaurant_menu_items=None) -> ChatbotResult:
    """
    Main entry point: parse user message using AI.
    
    Args:
        message: User's input text
        restaurant_menu_items: Optional QuerySet of MenuItem objects for name matching
    
    Returns:
        ChatbotResult with intent and extracted info
    """
    text = (message or "").strip()
    
    if not text:
        return ChatbotResult(
            intent="HELP",
            reply="Please type something like 'menu', 'add butter naan', or 'show cart'.",
            confidence=1.0
        )
    
    # Use LLM to classify intent
    llm_result = classify_intent_with_llm(text)
    intent = llm_result.get("intent", "HELP")
    item_name_raw = llm_result.get("item_name")
    quantity = llm_result.get("quantity", 1)
    confidence = llm_result.get("confidence", 0.5)
    
    # ============================================
    # Intent: SHOW_CART
    # ============================================
    if intent == "SHOW_CART":
        return ChatbotResult(
            intent="SHOW_CART",
            reply="Here is your current cart:",
            confidence=confidence
        )
    
    # ============================================
    # Intent: SHOW_MENU
    # ============================================
    if intent == "SHOW_MENU":
        return ChatbotResult(
            intent="SHOW_MENU",
            reply="Here are items from our menu:",
            confidence=confidence
        )
    
    # ============================================
    # Intent: CLEAR_CART
    # ============================================
    if intent == "CLEAR_CART":
        return ChatbotResult(
            intent="CLEAR_CART",
            reply="Okay, I'll clear your cart. ✅",
            confidence=confidence
        )
    
    # ============================================
    # Intent: CONFIRM_ORDER
    # ============================================
    if intent == "CONFIRM_ORDER":
        return ChatbotResult(
            intent="CONFIRM_ORDER",
            reply="Confirming your order now... ✅",
            confidence=confidence
        )
    
        # ============================================
    # Intent: SEARCH_ITEM (semantic search only, no cart changes)
    # ============================================
    # if intent == "SEARCH_ITEM" and item_name_raw:
    #     print(f"[RAG] Search Item triggered for: {item_name_raw}")
    #     search_results = semantic_search(item_name_raw, top_k=5)

    #     if not search_results:
    #         return ChatbotResult(
    #             intent="SEARCH_ITEM",
    #             reply=f"I couldn't find anything related to '{item_name_raw}'. Try 'menu' to see all dishes.",
    #             confidence=0.0,
    #         )

    #     # Check if best match is too low - means item doesn't exist
    #     best_score = search_results[0]["score"]
    #     if best_score < 0.4:
    #         return ChatbotResult(
    #             intent="SEARCH_ITEM",
    #             reply=f"Sorry, we don't have '{item_name_raw}' on our menu. Would you like to see what we do have? Just type 'menu'.",
    #             confidence=best_score,
    #         )

    #     # Use LLM to generate natural conversational response
    #     conversational_reply = generate_conversational_response(text, search_results)

    #     return ChatbotResult(
    #         intent="SEARCH_ITEM",
    #         reply=conversational_reply,
    #         confidence=best_score,
    #     )

        # ============================================
    # Intent: SEARCH_ITEM (semantic + category detection)
    # ============================================
    if intent == "SEARCH_ITEM" and item_name_raw:
        print(f"[RAG] Search Item triggered for: {item_name_raw}")

        # ✅ Normalize common typos like 'desert' -> 'dessert'
        normalized_term = normalize_search_term(item_name_raw)
        print(f"[RAG] Normalized search term: {normalized_term}")

        # ✅ 1️⃣ Try direct category match first
        all_categories = {
            parse_chunk_text(c)["category"].lower()
            for c in _text_chunks
            if parse_chunk_text(c)["category"]
        }

        # Normalize plural forms (desserts → dessert)
        cat_query = normalized_term.lower().rstrip("s")
        cat_match = None
        for cat in all_categories:
            if cat == normalized_term.lower() or cat.rstrip("s") == cat_query:
                cat_match = cat
                break

        if cat_match:
            print(f"[RAG] Direct category match: {cat_match}")
            matched_items = [
                {"text": c, "score": 1.0, "parsed": parse_chunk_text(c)}
                for c in _text_chunks
                if parse_chunk_text(c)["category"].lower() == cat_match
            ]

            reply_text = build_search_items_reply(
                user_query=text,
                normalized_term=cat_match,
                retrieved_items=matched_items,
            )

            suggestions = [
                {
                    "name": i["parsed"]["name"],
                    "price": i["parsed"]["price"],
                    "category": i["parsed"]["category"],
                }
                for i in matched_items
            ]

            return ChatbotResult(
                intent="SEARCH_ITEM",
                reply=reply_text,
                confidence=1.0,
                suggestions=suggestions,
            )

        # ✅ 2️⃣ If no category match, fall back to semantic search
        search_results = semantic_search(normalized_term, top_k=5)

        if not search_results:
            return ChatbotResult(
                intent="SEARCH_ITEM",
                reply=f"I couldn't find anything related to '{item_name_raw}'. Try 'menu' to see all dishes.",
                confidence=0.0,
            )

        best_score = search_results[0]["score"]
        if best_score < 0.4:
            return ChatbotResult(
                intent="SEARCH_ITEM",
                reply=f"Sorry, we don't have '{item_name_raw}' on our menu. Would you like to see what we do have? Just type 'menu'.",
                confidence=best_score,
            )

        # 🔹 Build bullet-style reply (your existing nice text)
        reply_text = build_search_items_reply(text, normalized_term, search_results)

        # 🔹 Build structured suggestions for frontend
        suggestions: List[Dict[str, str]] = []
        for item in search_results:
            parsed = item.get("parsed") or {}
            name = parsed.get("name") or ""
            price = parsed.get("price") or ""
            category = parsed.get("category") or ""
            if not name:
                continue
            suggestions.append(
                {"name": name, "price": price, "category": category}
            )

        return ChatbotResult(
            intent="SEARCH_ITEM",
            reply=reply_text,
            confidence=best_score,
            suggestions=suggestions,
        )




    # ============================================
    # Intent: ADD_ITEM (with semantic search)
    # ============================================
    # if intent == "ADD_ITEM" and item_name_raw:
    #     # Use semantic search to find best matching menu item
    #     search_results = semantic_search(item_name_raw, top_k=3)
        
    #     if not search_results:
    #         return ChatbotResult(
    #             intent="HELP",
    #             reply=f"I couldn't find any dishes matching '{item_name_raw}'. Try 'menu' to see options.",
    #             confidence=0.0
    #         )
        
    #     # Get best match
    #     best_match = search_results[0]
    #     matched_name = best_match["parsed"]["name"]
    #     match_score = best_match["score"]
        
    #     # If confidence is too low, the item doesn't exist or ask for clarification
    #     if match_score < 0.4:
    #         return ChatbotResult(
    #             intent="HELP",
    #             reply=f"Sorry, I couldn't find '{item_name_raw}' on our menu. Type 'menu' to see all available dishes.",
    #             confidence=match_score
    #         )
    #     elif match_score < 0.6:
    #         # Medium confidence - ask for clarification
    #         alternatives = [r["parsed"]["name"] for r in search_results[:3]]
    #         return ChatbotResult(
    #             intent="HELP",
    #             reply=f"I'm not sure about '{item_name_raw}'. Did you mean: {', '.join(alternatives)}?",
    #             confidence=match_score
    #         )
        
    #     return ChatbotResult(
    #         intent="ADD_ITEM",
    #         reply=f"Adding {quantity} × {matched_name} to your cart...",
    #         item_name=matched_name,
    #         quantity=quantity,
    #         confidence=match_score
    #     )
    

    if intent == "ADD_ITEM" and item_name_raw:
        search_results = semantic_search(item_name_raw, top_k=3)

        if not search_results:
            return ChatbotResult(
                intent="HELP",
                reply=f"I couldn't find any dishes matching '{item_name_raw}'. Try 'menu' to see options.",
                confidence=0.0
            )

        best_match = search_results[0]
        matched_name = best_match["parsed"]["name"]
        match_score = best_match["score"]

        # ✅ NEW: if the name exactly matches, trust it fully
        if matched_name and matched_name.strip().lower() == item_name_raw.strip().lower():
            return ChatbotResult(
                intent="ADD_ITEM",
                reply=f"Adding {quantity} × {matched_name} to your cart...",
                item_name=matched_name,
                quantity=quantity,
                confidence=1.0,
            )

        # OLD threshold logic (you can still tweak numbers)
        if match_score < 0.4:
            return ChatbotResult(
                intent="HELP",
                reply=f"Sorry, I couldn't find '{item_name_raw}' on our menu. Type 'menu' to see all available dishes.",
                confidence=match_score
            )
        elif match_score < 0.6:
            alternatives = [r["parsed"]["name"] for r in search_results[:3]]
            return ChatbotResult(
                intent="HELP",
                reply=f"I'm not sure about '{item_name_raw}'. Did you mean: {', '.join(alternatives)}?",
                confidence=match_score
            )

        return ChatbotResult(
            intent="ADD_ITEM",
            reply=f"Adding {quantity} × {matched_name} to your cart...",
            item_name=matched_name,
            quantity=quantity,
            confidence=match_score
        )

    # ============================================
    # Intent: REMOVE_ITEM
    # ============================================
    if intent == "REMOVE_ITEM" and item_name_raw:
        # Use semantic search for removal too
        search_results = semantic_search(item_name_raw, top_k=1)
        
        if search_results:
            matched_name = search_results[0]["parsed"]["name"]
            return ChatbotResult(
                intent="REMOVE_ITEM",
                reply=f"Removing {quantity} × {matched_name} from cart...",
                item_name=matched_name,
                quantity=quantity,
                confidence=search_results[0]["score"]
            )
    
    # ============================================
    # Default: HELP (with smart conversational handling)
    # ============================================
    
    # If the message seems like a menu question but wasn't classified correctly,
    # try to give a conversational answer
    if len(text.split()) > 2:  # More than 2 words suggests a real question
        search_results = semantic_search(text, top_k=5)
        if search_results and search_results[0]["score"] >= 0.3:
            # There's some relevant context, generate conversational response
            conversational_reply = generate_conversational_response(text, search_results)
            return ChatbotResult(
                intent="HELP",
                reply=conversational_reply,
                confidence=0.6
            )
    



    # Default help message
    return ChatbotResult(


        intent="HELP",
        reply=(
            "I can help you with:\n"
            "• 'menu' - see available dishes\n"
            "• 'add butter naan' or 'add 2 paneer tikka'\n"
            "• 'cart' - view your order\n"
            "• 'remove [item]'\n"
            "• 'clear' or 'confirm'\n\n"
            "You can also ask me questions about our menu!"
        ),
        confidence=0.5
    )


# ============================================
# Initialize on module import
# ============================================
try:
    load_rag_system()
except Exception as e:
    print(f"[RAG] Warning: Could not load RAG system: {e}")
    print("[RAG] Chatbot will use fallback mode.")