# **AI Restaurant Chatbot Workflow**

## **1. System Architecture Overview**

### Components

1. **Frontend (Restaurant Website)**

   * Chat UI embedded in the site (HTML/JS)
   * Sends user messages to Django backend

2. **Backend (Handled by Django Developer)**

   * REST API endpoints for chat, menu, ordering, services
   * Database for menu, orders, and user context
   * Session management

3. **AI Layer (Your Python Terminal Application)**

   * Uses Groq API with Llama 4 Maverick
   * Responsible for natural conversation, mood detection, recommendations
   * Communicates with Django via REST API calls

---

## **2. Workflow Breakdown**

### **Step 1: Input Capture**

* Django receives message from frontend
* Django sends text to your Python bot
* Bot starts or continues the conversation session
* Bot stores short-term conversation memory

**Tools/Libraries:**
Python, `requests`, optional `rich` for terminal formatting

---

### **Step 2: Preprocessing & Intent Detection**

* Extract user mood (happy, sad, stressed, bored, adventurous, neutral)
* Determine intent (menu inquiry, recommendations, order creation, services information, small talk)
* Use structured prompting for Llama 4 Maverick to handle conversational flow

**Tools/Libraries:**
Groq API (Llama 4 Maverick), optional spaCy, regex

---

### **Step 3: Mood-Based Recommendation Engine**

* Retrieve menu from Django backend
* Combine rule-based and LLM-based suggestions
* Consider mood → dish mapping (comfort food, spicy items, desserts, chef specials)

**Tools/Libraries:**
Custom Python logic, optional embeddings (sentence-transformers)

---

### **Step 4: Backend Communication Layer**

Your bot will interact with Django using REST endpoints such as:

1. **Get Menu:** `/api/menu/`
2. **Create Order:** `/api/order/`
3. **Get Services:** `/api/services/`
4. **Conversation/State:** `/api/user-state/`

**Features:**

* Fetch updated menu
* Submit user order
* Retrieve service details
* Maintain user session state

**Tools/Libraries:**
`requests`, optional `pydantic` for schema validation

---

### **Step 5: LLM Response Generation**

Using Groq API (Llama 4 Maverick):

* Generate human-like responses
* Maintain friendly restaurant staff persona
* Use conversation history to maintain context
* Avoid bot-like tone
* Integrate recommendations, ordering steps, and service information

**Tools/Libraries:**
Groq Python SDK

---

### **Step 6: Conversation Memory & State Management**

Track:

* User mood
* Order progress
* Selected menu items
* User preferences
* Recent conversation history

**Implementation Options:**

* Basic in-memory store (for terminal use)
* Redis for production
* Django session sync if needed

---

### **Step 7: Response Delivery**

Your bot sends structured output back to Django, including:

* Human-like text message
* Identified intent
* Detected mood
* Recommended dishes
* Order/action metadata
* Updated conversation state

Django returns simplified plain text to the frontend chat UI.

---

## **3. Technical Stack Summary**

### **AI/LLM Layer**

* Groq API
* Llama 4 Maverick model (fast, free inference)

### **Python Environment**

* Groq SDK
* Requests
* Optional extras:

  * Pydantic
  * Redis
  * Rich
  * Python-dotenv

### **Backend**

* Django REST Framework
* Menu/order management
* Services API
* Conversation/session API

---

## **4. Recommended Project Structure**

```
restaurant_ai_bot/
│
├── bot/
│   ├── core/
│   │   ├── llm_client.py
│   │   ├── mood_engine.py
│   │   ├── intent_classifier.py
│   │   └── recommend.py
│   ├── services/
│   │   ├── django_api.py
│   └── main.py
│
├── config/
│   └── settings.py
└── requirements.txt
```

---

## **5. Example Conversation Flow (Conceptual)**

1. User says: “I’m feeling sad today.”
2. Bot identifies mood: Sad
3. Bot recommends warm soups, desserts, comfort food
4. User selects an item
5. Bot sends order request to Django backend
6. Django confirms order ID
7. Bot responds with order confirmation message

---

## **6. Implementation Roadmap**

1. Build Groq LLM wrapper
2. Create mood detection prompt
3. Create intent classifier prompt
4. Build recommendation logic
5. Integrate Django endpoints
6. Add conversation memory
7. Test with real menu data
8. Integrate with frontend via Django
