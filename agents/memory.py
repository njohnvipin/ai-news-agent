"""
memory.py — Long-term memory for the digest agent.

Every digest that a human approves and sends is saved to data/digests.json.
The agent can read recent ones with the get_previous_digests tool, so it can
point out continuing trends and avoid repeating yesterday's briefing.

(Short-term memory, i.e. the conversation within one run, is handled by the
LangGraph checkpointer in news_agent.py.)
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List

from config import DATA_DIR

logger = logging.getLogger(__name__)

DIGEST_FILE: Path = DATA_DIR / "digests.json"
MAX_STORED_DIGESTS = 30


def _read_all() -> List[dict]:
    if not DIGEST_FILE.exists():
        return []
    try:
        data = json.loads(DIGEST_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not read %s (%s); starting with empty memory", DIGEST_FILE, e)
        return []


def save_digest(insights: str, headlines: List[str]) -> None:
    """Remember a sent digest (keeps the newest MAX_STORED_DIGESTS)."""
    digests = _read_all()
    digests.append({
        "sent_at": datetime.now().isoformat(timespec="minutes"),
        "insights": insights.strip(),
        "headlines": headlines[:10],
    })
    DIGEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    DIGEST_FILE.write_text(json.dumps(digests[-MAX_STORED_DIGESTS:], indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Digest saved to long-term memory (%s)", DIGEST_FILE)


def load_recent_digests(limit: int = 3) -> List[dict]:
    """The most recent digests, newest first."""
    return list(reversed(_read_all()))[:limit]
