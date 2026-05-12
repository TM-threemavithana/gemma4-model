import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env from the same directory as this file
load_dotenv(Path(__file__).parent / ".env")

PG_HOST     = os.environ.get("PG_HOST", "localhost")
PG_PORT     = int(os.getenv("PG_PORT", "5432"))
PG_DATABASE = os.environ.get("PG_DATABASE", "main_system")
PG_USER     = os.environ.get("PG_USER", "postgres")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "")
PG_MIN_POOL = int(os.getenv("PG_MIN_POOL", "2"))
PG_MAX_POOL = int(os.getenv("PG_MAX_POOL", "10"))

IDENTITY_FETCH_TIMEOUT_S = float(os.getenv("IDENTITY_FETCH_TIMEOUT_S", "4.0"))
