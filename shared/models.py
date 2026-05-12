from dataclasses import dataclass
from typing import Optional

@dataclass
class UserProfile:
    # Identity (Project Echo Schema)
    user_id:      str
    username:     str
    email:        str
    is_active:    bool

    # Operational metadata
    member_since: Optional[str]  # ISO date string

    # Fetch state — set by identity.py, not from DB
    fetch_error:  Optional[str]  = None


UNRESOLVED_PROFILE = UserProfile(
    user_id="",
    username="",
    email="",
    is_active=False,
    member_since=None,
    fetch_error="space_not_found",
)
