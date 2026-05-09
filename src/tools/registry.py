import logging

log = logging.getLogger("tools")

def get_order_status(order_id: str):
    """Checks the status of a customer order."""
    log.info(f"🛠️ Executing get_order_status for {order_id}")
    # Mock data
    orders = {
        "12345": "Shipped - arriving tomorrow",
        "67890": "Processing - will ship in 2 days",
    }
    return orders.get(order_id, "Order not found. Please verify the ID.")

def send_sms(phone_number: str, message: str):
    """Sends an SMS message to a specific phone number."""
    log.info(f"🛠️ Executing send_sms to {phone_number}: {message}")
    # Mock success
    return f"Successfully sent SMS to {phone_number}"

# Tool Definitions for Gemma-4
# Note: Gemma-4 usually expects a list of tool dictionaries
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Get the current status of a customer order by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The 5-digit order ID."}
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_sms",
            "description": "Send a text message to a user's phone number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone_number": {"type": "string", "description": "The recipient's phone number."},
                    "message": {"type": "string", "description": "The content of the SMS."}
                },
                "required": ["phone_number", "message"]
            }
        }
    }
]

# Mapping names to actual functions
TOOL_MAP = {
    "get_order_status": get_order_status,
    "send_sms": send_sms,
}
