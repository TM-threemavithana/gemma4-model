import asyncio
import logging
from config import IDENTITY_FETCH_TIMEOUT_S
from shared.db import get_pool
from shared.models import UserProfile, UNRESOLVED_PROFILE

logger = logging.getLogger(__name__)

# Only these columns are fetched from Project Echo — security boundary
_COLUMNS = """
    u.user_id,
    u.username,
    u.email,
    u.is_active,
    TO_CHAR(u.created_at, 'YYYY-MM-DD') AS member_since
"""

async def get_full_user_context(space_id: str) -> UserProfile:
    """
    Identifies a user based on their 'Local Space' connection ID (IP/SIP).
    This replaces all phone-number based logic.
    """
    if not space_id or space_id == "unknown":
        return UNRESOLVED_PROFILE

    # Map the Space ID (e.g. IP address) to a User ID via the linked_spaces table
    query = f"""
        SELECT {_COLUMNS}
        FROM users u
        JOIN linked_spaces ls ON u.user_id = ls.user_id
        WHERE ls.space_id = $1
        LIMIT 1
    """
    
    try:
        async with asyncio.timeout(IDENTITY_FETCH_TIMEOUT_S):
            async with get_pool().acquire() as conn:
                row = await conn.fetchrow(query, space_id)

        if row:
            return UserProfile(
                user_id      = str(row["user_id"]),
                username     = row["username"],
                email        = row["email"],
                is_active    = bool(row["is_active"]),
                member_since = row["member_since"],
            )
        
        logger.info("Connection from unknown Space ID: %s (Acting as Receptionist)", space_id)
        return UNRESOLVED_PROFILE

    except Exception as exc:
        logger.error("Space identity resolution failed for %s: %s", space_id, exc)
        return UNRESOLVED_PROFILE

async def fetch_user_conversations(user_id: str = None, token: str = None, limit: int = 3):
    """
    Fetches the most recent conversation titles from Project Echo.
    Supports both internal User ID (Local Space) and Access Token (Outside World).
    """
    if token:
        # In a real production system, you would call the Project Echo API 
        # using the Bearer Token. For now, we simulate the fetch.
        logger.info("Fetching data for Outside User via Access Token")
        return [{"title": "Cloud Chat 1", "status": "active", "created_at": "2026-05-10"}]

    if not user_id: return []
    
    query = """
        SELECT title, status, created_at 
        FROM conversations 
        WHERE user_id = $1 
        ORDER BY created_at DESC 
        LIMIT $2
    """
    try:
        async with asyncio.timeout(IDENTITY_FETCH_TIMEOUT_S):
            async with get_pool().acquire() as conn:
                rows = await conn.fetch(query, int(user_id), limit)
                return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"Failed to fetch conversations for {user_id}: {e}")
        return []

async def perform_api_login(username, password):
    """
    Proxies a verbal login to the Main System (Project Echo) API.
    Returns (Success, Token/Error)
    """
    import httpx
    # In production, this URL should be in your .env
    LOGIN_URL = "http://localhost:8000/api/login" 
    
    try:
        async with httpx.AsyncClient() as client:
            # We send credentials exactly as the main system expects them
            response = await client.post(LOGIN_URL, json={
                "username": username,
                "password": password
            }, timeout=IDENTITY_FETCH_TIMEOUT_S)
            
            if response.status_code == 200:
                data = response.json()
                return True, data.get("access_token")
            else:
                return False, "Invalid credentials"
    except Exception as e:
        logger.error(f"API Login failed: {e}")
        return False, "Main system connection error"

async def resolve_by_token(token: str) -> UserProfile:
    """
    Decodes a token or calls /me endpoint to get the user's identity.
    For now, we simulate the successful resolution.
    """
    # Simulate a successful profile fetch for the demo
    return UserProfile(
        user_id="outside_user",
        username="Authenticated User",
        email="user@echo.cloud",
        is_active=True,
        member_since="2026-05-10"
    )
