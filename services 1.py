# chatbot/services.py (AI-powered version)
from django.db import transaction
from django.shortcuts import get_object_or_404

from restaurants.models import Restaurant
from menu.models import MenuItem
from orders.models import Order, OrderItem
from .engine import ChatbotResult


def get_or_create_open_order(restaurant: Restaurant, session_id: str) -> Order:
    """Get or create a pending order for this session."""
    order, _ = Order.objects.get_or_create(
        restaurant=restaurant,
        status=Order.OrderStatus.PENDING,
        session_id=session_id,
        defaults={
            "order_type": Order.OrderType.TAKEAWAY,
            "source": "chatbot",
        },
    )
    return order


def find_menu_item_by_name(restaurant: Restaurant, item_name: str) -> MenuItem:
    """
    Find MenuItem by name using fuzzy matching.
    Raises MenuItem.DoesNotExist if not found.
    """
    # Try exact match first
    try:
        return MenuItem.objects.get(
            restaurant=restaurant,
            name__iexact=item_name
        )
    except MenuItem.DoesNotExist:
        pass
    
    # Try contains match
    qs = MenuItem.objects.filter(
        restaurant=restaurant,
        name__icontains=item_name
    )
    
    if qs.exists():
        return qs.first()
    
    raise MenuItem.DoesNotExist(f"No menu item found matching: {item_name}")


def apply_intent(restaurant: Restaurant, session_id: str, result: ChatbotResult):
    """
    Takes ChatbotResult from AI engine, performs DB actions,
    and ALWAYS returns (reply_text, order, extra_dict).
    """
    order = get_or_create_open_order(restaurant, session_id)

    # ============================================
    # SHOW_CART
    # ============================================
    if result.intent == "SHOW_CART":
        if not order.items.exists():
            return "Your cart is empty.", order, {}

        lines = []
        for item in order.items.select_related("menu_item"):
            lines.append(f"{item.quantity} × {item.name} — ₹{item.total_price}")

        reply = "Here is your cart:\n" + "\n".join(lines) + f"\nTotal: ₹{order.total}"
        return reply, order, {}

    # ============================================
    # SHOW_MENU
    # ============================================
    if result.intent == "SHOW_MENU":
        qs = MenuItem.objects.filter(
            restaurant=restaurant,
            available=True
        ).order_by("category", "name")[:50]

        if not qs:
            return "This restaurant has no menu items yet.", order, {}

        lines = []
        current_category = None
        suggestions = []

        for m in qs:
            if m.category != current_category:
                current_category = m.category
                lines.append(f"\n**{current_category}**")
            lines.append(f"• {m.name} — ₹{m.price}")

            suggestions.append(
                {
                    "id": m.id,
                    "name": m.name,
                    "price": str(m.price),
                    "category": m.category or "",
                }
            )

        reply = (
            "Here's our menu:\n"
            + "\n".join(lines)
            + "\n\nJust tap an item or tell me what you'd like to add!"
        )
        return reply, order, {"menu_items": suggestions}

    # ============================================
    # HELP
    # ============================================
    if result.intent == "HELP":
        return result.reply, order, {}

    # ============================================
    # CLEAR_CART
    # ============================================
    if result.intent == "CLEAR_CART":
        order.items.all().delete()
        order.recalc_totals()
        return "✅ Your cart has been cleared.", order, {}

    # ============================================
    # CONFIRM_ORDER
    # ============================================
    if result.intent == "CONFIRM_ORDER":
        if not order.items.exists():
            return "Your cart is empty. Add some items before confirming.", order, {}

        order.status = Order.OrderStatus.CONFIRMED
        order.save(update_fields=["status"])

        return (
            f"✅ Order #{order.id} confirmed! Total: ₹{order.total}\n\nThank you for your order!",
            order,
            {},
        )

    # ============================================
    # SEARCH_ITEM
    # ============================================
    if result.intent == "SEARCH_ITEM":
        extra = {}
        if getattr(result, "suggestions", None):
            extra["menu_items"] = result.suggestions
        return result.reply, order, extra

    # ============================================
    # ADD_ITEM
    # ============================================
    if result.intent == "ADD_ITEM":
        if not result.item_name:
            return (
                "I couldn't figure out what to add. Try 'add butter naan' or 'menu'.",
                order,
                {},
            )

        try:
            menu_item = find_menu_item_by_name(restaurant, result.item_name)
        except MenuItem.DoesNotExist:
            similar_items = MenuItem.objects.filter(
                restaurant=restaurant,
                name__icontains=result.item_name.split()[0]
            )[:3]

            if similar_items:
                suggestions = ", ".join([item.name for item in similar_items])
                return (
                    f"I couldn't find '{result.item_name}'. Did you mean: {suggestions}?",
                    order,
                    {},
                )
            else:
                return (
                    f"Sorry, '{result.item_name}' is not on our menu. Type 'menu' to see options.",
                    order,
                    {},
                )

        # 🔹 Quantity validation – only whole items allowed
        raw_qty = getattr(result, "quantity", 1)

        try:
            qty_to_add = int(raw_qty)
        except (TypeError, ValueError):
            return (
                "I can only add whole items. For example: 'add 2 butter naan'.",
                order,
                {},
            )

        if qty_to_add < 1:
            return (
                "Quantity must be at least 1 full item. Try 'add 1 butter naan'.",
                order,
                {},
            )

        with transaction.atomic():
            oi, created = OrderItem.objects.get_or_create(
                order=order,
                menu_item=menu_item,
                defaults={
                    "name": menu_item.name,
                    "quantity": qty_to_add,
                    "unit_price": menu_item.price,
                    "total_price": menu_item.price * qty_to_add,
                },
            )

            if not created:
                oi.quantity += qty_to_add
                oi.total_price = oi.unit_price * oi.quantity
                oi.save(update_fields=["quantity", "total_price"])

            order.recalc_totals()

        confidence_emoji = "✅" if result.confidence > 0.7 else "👍"
        reply = (
            f"{confidence_emoji} Added {qty_to_add} × {menu_item.name} to your cart.\n"
            f"Current total: ₹{order.total}"
        )
        return reply, order, {}


    # ============================================
    # REMOVE_ITEM
    # ============================================
        # ============================================
    # REMOVE_ITEM
    # ============================================
    if result.intent == "REMOVE_ITEM":
        if not order.items.exists():
            return "Your cart is already empty.", order, {}

        if not result.item_name:
            return "Which item would you like to remove?", order, {}

        try:
            menu_item = find_menu_item_by_name(restaurant, result.item_name)
            oi = OrderItem.objects.get(
                order=order,
                menu_item=menu_item,
            )
        except (MenuItem.DoesNotExist, OrderItem.DoesNotExist):
            return f"'{result.item_name}' is not in your cart.", order, {}

        raw_qty = getattr(result, "quantity", 1)

        # 🔹 CASE 1: LLM ne fraction diya (e.g. 0.5 for "half")
        qty_to_remove = None
        try:
            # float / int / str → number convert karne ki koshish
            numeric_qty = float(raw_qty)
        except (TypeError, ValueError):
            numeric_qty = None

        if numeric_qty is not None and 0 < numeric_qty < 1:
            # e.g. 0.5 => half of current quantity
            fraction = numeric_qty
            current_qty = oi.quantity

            qty_to_remove = int(current_qty * fraction)
            if qty_to_remove < 1:
                qty_to_remove = 1

        # 🔹 CASE 2: normal "remove 2" / "remove 10"
        if qty_to_remove is None:
            try:
                qty_to_remove = int(raw_qty)
            except (TypeError, ValueError):
                return (
                    "I can only remove whole items. Try 'remove 1 Brownie Banana Sundae'.",
                    order,
                    {},
                )

        if qty_to_remove < 1:
            return (
                "Quantity must be at least 1 full item. For example: 'remove 1 Brownie Banana Sundae'.",
                order,
                {},
            )

        # ✅ yahan se qty_to_remove hamesha positive int hai
        if qty_to_remove >= oi.quantity:
            oi.delete()
            msg = f"Removed {oi.name} from your cart."
        else:
            oi.quantity -= qty_to_remove
            oi.total_price = oi.unit_price * oi.quantity
            oi.save(update_fields=["quantity", "total_price"])
            msg = f"Removed {qty_to_remove} × {oi.name} from your cart."

        order.recalc_totals()
        msg += f" Current total: ₹{order.total}"
        return msg, order, {}

    # ============================================
    # Fallback
    # ============================================
    return (
        "I'm not sure what to do. Try 'menu', 'cart', 'add [item]', or 'confirm'.",
        order,
        {},
    )