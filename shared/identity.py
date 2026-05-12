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

import os

def get_local_user_token() -> str | None:
    """Reads the stored access token from the local mobile device."""
    token_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "local_token.txt")
    try:
        with open(token_path, "r") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Failed to read local token from {token_path}: {e}")
        return None

async def perform_api_query(query: str, token: str) -> str:
    """
    Proxies a query to the Project Echo API using the local user's token.
    Extracts the specific document data as requested.
    """
    import httpx
    QUERY_URL = "http://localhost:8050/api/v1/chat/query"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                QUERY_URL, 
                data={"content": query},
                headers={"Authorization": f"Bearer {token}"},
                timeout=10.0
            )
            
            if response.status_code == 200:
                data = response.json()
                try:
                    documents = data.get("results", {}).get("documents", [])
                    if documents and len(documents) > 0 and len(documents[0]) > 0:
                        return str(documents[0][0])
                    else:
                        return "No relevant data found in Project Echo."
                except Exception as e:
                    logger.error(f"Failed to parse query response: {e}")
                    return "Error extracting document data."
            else:
                logger.error(f"Query failed {response.status_code}: {response.text}")
                return "Failed to query the database."
    except Exception as e:
        logger.error(f"API Query failed: {e}")
        return "Main system connection error."
