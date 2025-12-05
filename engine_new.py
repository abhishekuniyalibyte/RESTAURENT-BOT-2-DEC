import os
import json
import pickle
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict
from pathlib import Path
import re

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

# ============================================
# Configuration
# ============================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMBEDDINGS_PATH = Path("media/embeddings/restaurant_1_menu_embeddings.pkl")
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# ============================================
# Global state (loaded once)
# ============================================
_embed_model = None
_embeddings = None
_metadata = None
_groq_client = None

# Common restaurant typos / aliases
COMMON_TYPO_MAP = {
    "desert": "dessert",
    "deserts": "desserts",
}

# Track last-known modification times
_emb_last_mtime: Optional[float] = None


@dataclass
class ChatbotResult:
    """
    Unified result format for chatbot responses.
    """
    intent: str  # ADD_ITEM, REMOVE_ITEM, SHOW_CART, SHOW_MENU, CLEAR_CART, CONFIRM_ORDER, HELP, SEARCH_ITEM
    reply: str
    item_id: Optional[int] = None
    quantity: int = 1
    item_name: Optional[str] = None
    confidence: float = 1.0
    suggestions: Optional[List[Dict[str, str]]] = None


def load_rag_system():
    """
    Load embeddings, metadata, and models (called once on startup).
    """
    global _embed_model, _embeddings, _metadata, _groq_client
    global _emb_last_mtime

    if _embed_model is not None and _embeddings is not None and _metadata is not None:
        return

    print("[RAG] Loading embedding model and data...")

    # 1) Load sentence transformer
    if _embed_model is None:
        _embed_model = SentenceTransformer(MODEL_NAME, device="cpu")

    # 2) Load embeddings from pickle (matching chatbot.py structure)
    if EMBEDDINGS_PATH.exists():
        with open(EMBEDDINGS_PATH, 'rb') as f:
            data = pickle.load(f)
            _embeddings = data['embeddings']
            _metadata = data['metadata']
        
        _emb_last_mtime = EMBEDDINGS_PATH.stat().st_mtime
        print(f"[RAG] Loaded embeddings: {len(_embeddings)} items")
    else:
        raise FileNotFoundError(f"Embeddings not found at {EMBEDDINGS_PATH}")

    # 3) Initialize Groq client
    if GROQ_API_KEY:
        if _groq_client is None:
            _groq_client = Groq(api_key=GROQ_API_KEY)
        print("[RAG] Groq client initialized")
    else:
        print("[RAG] Warning: GROQ_API_KEY not set. LLM features disabled.")


def reload_rag_system():
    """Force reload of embeddings and metadata from disk."""
    global _embed_model, _embeddings, _metadata
    global _emb_last_mtime

    _embeddings = None
    _metadata = None
    _emb_last_mtime = None

    load_rag_system()


def ensure_latest_embeddings():
    """Ensure this process has the latest version of embeddings."""
    global _embed_model, _embeddings, _metadata
    global _emb_last_mtime

    if _embed_model is None or _embeddings is None or _metadata is None:
        load_rag_system()
        return

    if not EMBEDDINGS_PATH.exists():
        return

    current_emb_mtime = EMBEDDINGS_PATH.stat().st_mtime

    if _emb_last_mtime is None:
        _emb_last_mtime = current_emb_mtime
        return

    if current_emb_mtime == _emb_last_mtime:
        return

    print("[RAG] Detected updated embeddings on disk; reloading...")
    
    with open(EMBEDDINGS_PATH, 'rb') as f:
        data = pickle.load(f)
        _embeddings = data['embeddings']
        _metadata = data['metadata']
    
    _emb_last_mtime = current_emb_mtime
    print(f"[RAG] Reloaded embeddings: {len(_embeddings)} items")


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def semantic_search(query: str, top_k: int = 5) -> List[Dict[str, any]]:
    """
    Search for menu items using semantic similarity.
    Returns list of dicts with metadata and score.
    """
    ensure_latest_embeddings()

    query_emb = _embed_model.encode(query)
    
    similarities = []
    for idx, embedding in enumerate(_embeddings):
        sim = cosine_similarity(query_emb, embedding)
        similarities.append((idx, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    results = []
    for idx, sim in similarities[:top_k]:
        meta = _metadata[idx]
        
        # Extract relevant fields from metadata
        name = meta.get('name', '')
        price = meta.get('price', '')
        category = meta.get('category', '')
        
        # Also check original_data if available
        if meta.get('original_data'):
            orig = meta['original_data']
            if not name:
                name = orig.get('name', '')
            if not price:
                price = orig.get('price', '')
            if not category:
                category = orig.get('category', '')
        
        results.append({
            'metadata': meta,
            'score': float(sim),
            'parsed': {
                'name': name,
                'price': price,
                'category': category
            }
        })
    
    return results


def normalize_search_term(term: str) -> str:
    """Normalize common user typos/variants for menu search."""
    if not term:
        return term

    original = term
    t = term.strip().lower()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    
    if "desert" in t and "dessert" not in t:
        t = "dessert"
    
    t = COMMON_TYPO_MAP.get(t, t)
    
    print(f"[RAG] normalize_search_term: '{original}' -> '{t}'")
    return t


def build_search_items_reply(
    user_query: str,
    normalized_term: str,
    retrieved_items: List[Dict[str, any]],
) -> str:
    """Build a clean, bullet-style reply for SEARCH_ITEM queries."""
    primary_category = None
    for item in retrieved_items:
        parsed = item.get("parsed") or {}
        cat = parsed.get("category")
        if cat:
            primary_category = cat
            break

    heading_term = primary_category or (normalized_term.title() if normalized_term else None)

    lines: List[str] = []
    if heading_term:
        lines.append(f"Here are some {heading_term} options from our menu:")
    else:
        lines.append("Here are some items from our menu:")

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

    for category, items in items_by_category.items():
        lines.append("")
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


def classify_intent_with_llm(message: str) -> Dict[str, any]:
    """Use Groq LLM to classify user intent and extract entities."""
    if _groq_client is None:
        load_rag_system()
    
    if not _groq_client:
        return {"intent": "HELP", "confidence": 0.5}
    
    prompt = f"""You are an intelligent restaurant ordering assistant. Analyze the user's message and extract their intent and details.

AVAILABLE INTENTS:
1. ADD_ITEM - User wants to add food (e.g., "I want...", "add...", "get me...")
2. REMOVE_ITEM - User wants to remove food (e.g., "remove...", "delete...")
3. SHOW_CART - User wants to see current order (e.g., "my cart", "show order")
4. SHOW_MENU - User wants to see COMPLETE menu (ONLY: "menu", "show menu")
5. CLEAR_CART - User wants to empty cart (e.g., "clear cart", "start over")
6. CONFIRM_ORDER - User ready to place order (e.g., "confirm", "checkout")
7. SEARCH_ITEM - User asking about items WITHOUT adding (e.g., "do you have...", "what X do you have")
8. HELP - User needs assistance or unclear message

CRITICAL:
- "show menu" = SHOW_MENU
- "what desserts?" = SEARCH_ITEM with item_name="desserts"
- "do you have biryani?" = SEARCH_ITEM with item_name="biryani"
- "I want biryani" = ADD_ITEM

USER MESSAGE: "{message}"

RESPOND WITH ONLY JSON (no markdown):
{{"intent": "ADD_ITEM", "item_name": "paneer tikka", "quantity": 2}}

EXAMPLES:
"add 2 butter naan" → {{"intent": "ADD_ITEM", "item_name": "butter naan", "quantity": 2}}
"menu" → {{"intent": "SHOW_MENU", "item_name": null, "quantity": 1}}
"what desserts do you have?" → {{"intent": "SEARCH_ITEM", "item_name": "desserts", "quantity": 1}}
"show cart" → {{"intent": "SHOW_CART", "item_name": null, "quantity": 1}}
"remove paneer" → {{"intent": "REMOVE_ITEM", "item_name": "paneer", "quantity": 1}}
"confirm order" → {{"intent": "CONFIRM_ORDER", "item_name": null, "quantity": 1}}

NOW ANALYZE:"""
    
    try:
        response = _groq_client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=250
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Clean JSON response
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
        
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        result = json.loads(response_text)
        
        if "intent" not in result:
            result["intent"] = "HELP"
        if "quantity" not in result:
            result["quantity"] = 1
        
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
    """Generate natural conversational responses using LLM."""
    if not _groq_client:
        return "I found some items that might interest you."
    
    context_lines = []
    for item in retrieved_items:
        parsed = item.get("parsed", {})
        name = parsed.get("name", "")
        price = parsed.get("price", "")
        category = parsed.get("category", "")
        
        if name:
            line = f"- {name}"
            if category:
                line += f" ({category})"
            if price:
                line += f" - ₹{price}"
            context_lines.append(line)
    
    context_string = "\n".join(context_lines)
    
    system_prompt = (
        "You are a friendly restaurant assistant. "
        "Answer questions naturally and warmly. "
        "When describing items, mention prices. "
        "Keep responses concise (2-4 sentences). "
        "Focus ONLY on items in the context."
    )
    
    user_prompt = (
        f"User question: {user_query}\n\n"
        f"Available menu items:\n{context_string}\n\n"
        "Provide a natural, friendly response."
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
        item_names = [item["parsed"]["name"] for item in retrieved_items if item.get("parsed", {}).get("name")]
        if item_names:
            return f"I found these items: {', '.join(item_names)}."
        return "I found some items that might interest you."


def parse_message(message: str, restaurant_menu_items=None) -> ChatbotResult:
    """
    Main entry point: parse user message using AI.
    
    Args:
        message: User's input text
        restaurant_menu_items: Optional QuerySet of MenuItem objects
    
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
    # Intent: SEARCH_ITEM
    # ============================================
    if intent == "SEARCH_ITEM" and item_name_raw:
        print(f"[RAG] Search Item triggered for: {item_name_raw}")
        
        normalized_term = normalize_search_term(item_name_raw)
        print(f"[RAG] Normalized search term: {normalized_term}")
        
        # Try semantic search
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
        
        reply_text = build_search_items_reply(text, normalized_term, search_results)
        
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
    # Intent: ADD_ITEM
    # ============================================
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
        
        # Exact match check
        if matched_name and matched_name.strip().lower() == item_name_raw.strip().lower():
            return ChatbotResult(
                intent="ADD_ITEM",
                reply=f"Adding {quantity} × {matched_name} to your cart...",
                item_name=matched_name,
                quantity=quantity,
                confidence=1.0,
            )
        
        # Threshold checks
        if match_score < 0.4:
            return ChatbotResult(
                intent="HELP",
                reply=f"Sorry, I couldn't find '{item_name_raw}' on our menu. Type 'menu' to see all available dishes.",
                confidence=match_score
            )
        elif match_score < 0.6:
            alternatives = [r["parsed"]["name"] for r in search_results[:3] if r["parsed"]["name"]]
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
    # Default: HELP
    # ============================================
    if len(text.split()) > 2:
        search_results = semantic_search(text, top_k=5)
        if search_results and search_results[0]["score"] >= 0.3:
            conversational_reply = generate_conversational_response(text, search_results)
            return ChatbotResult(
                intent="HELP",
                reply=conversational_reply,
                confidence=0.6
            )
    
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