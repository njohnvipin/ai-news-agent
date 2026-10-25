"""
config.py — Loads settings from .env and sets up logging.

Only GROQ_API_KEY is required for everything. Email settings are only
required when you actually send an email (agent mode), so the Streamlit UI
and the API work without any Gmail setup.
"""
import logging
import os

from dotenv import load_dotenv

# override=True: values in .env win over any old variables already set
# in your system environment (common source of "I changed .env but
# nothing happened" bugs on Windows).
load_dotenv(override=True)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ── LLM ──────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "openai/gpt-oss-120b"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME") or DEFAULT_MODEL

if not GROQ_API_KEY:
    raise EnvironmentError(
        "Missing GROQ_API_KEY in .env — get a free key at https://console.groq.com"
    )

# ── Email (optional, only needed for agent mode) ─────────────────────────────
EMAIL        = os.getenv("EMAIL")
APP_PASSWORD = os.getenv("APP_PASSWORD")
RECIPIENTS   = [r.strip() for r in os.getenv("RECIPIENTS", "").split(",") if r.strip()]


def missing_email_settings() -> list[str]:
    """Return the names of any email settings that are not filled in."""
    values = {"EMAIL": EMAIL, "APP_PASSWORD": APP_PASSWORD, "RECIPIENTS": RECIPIENTS}
    return [name for name, value in values.items() if not value]
