import logging
import asyncio
from shared.identity import fetch_user_conversations
from shared.models import UNRESOLVED_PROFILE

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

async def get_recent_activity(user_id: str = None):
    """
    Fetch the user's most recent Project Echo conversations.
    Useful when the user asks about their chat history or active sessions.
    """
    if not user_id or user_id == "":
        return "Error: User must be identified before checking activity."
    
    chats = await fetch_user_conversations(user_id, limit=3)
    if not chats:
        return "No recent conversation history found in Project Echo."
    
    results = []
    for c in chats:
        results.append(f"- {c['title']} (Status: {c['status']}, Created: {c['created_at'].strftime('%Y-%m-%d')})")
    
    return "Here are the most recent conversations from Project Echo:\n" + "\n".join(results)

async def query_project_echo(query: str):
    """
    Query the local user's data from Project Echo to answer questions.
    """
    from shared.identity import get_local_user_token, perform_api_query
    
    token = get_local_user_token()
    if not token:
        return "Error: Could not retrieve the local user's access token. The local user needs to log into the app first."
        
    result = await perform_api_query(query, token)
    return result

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
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_activity",
            "description": "Fetch the user's most recent Project Echo conversations and chat history.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_project_echo",
            "description": "Look up the owner's personal details, business offerings, pricing packages, or identity from Project Echo to answer questions on their behalf.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The question or search query to find in the local user's data."}
                },
                "required": ["query"]
            }
        }
    }
]

# Mapping names to actual functions
TOOL_MAP = {
    "get_order_status": get_order_status,
    "send_sms": send_sms,
    "get_recent_activity": get_recent_activity,
    "query_project_echo": query_project_echo,
}
