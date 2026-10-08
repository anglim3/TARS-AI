"""Short-term voice conversation memory for xAI realtime turns.

Keeps recent user/assistant transcripts for session_ttl_sec (default 300s)
and injects them into the next realtime session so a wake within the window
continues the same conversation. Mic is not held between turns.
"""
from __future__ import annotations

import threading
import time
from typing import List, Dict, Optional

_lock = threading.Lock()
_turns: List[Dict] = []
_last_ts: float = 0.0
_ttl: float = 300.0


def configure(ttl_sec: float = 300.0):
    global _ttl
    with _lock:
        _ttl = max(30.0, float(ttl_sec))


def _purge(now: Optional[float] = None):
    global _turns, _last_ts
    now = time.time() if now is None else now
    if _last_ts and (now - _last_ts) > _ttl:
        _turns = []
        _last_ts = 0.0
        return
    cutoff = now - _ttl
    _turns = [t for t in _turns if t.get("ts", 0) >= cutoff]
    if not _turns:
        _last_ts = 0.0


def clear():
    global _turns, _last_ts
    with _lock:
        _turns = []
        _last_ts = 0.0


def add_turn(user_text: str, assistant_text: str):
    """Record one completed voice turn."""
    global _last_ts, _turns
    user_text = (user_text or "").strip()
    assistant_text = (assistant_text or "").strip()
    if not user_text and not assistant_text:
        return
    now = time.time()
    with _lock:
        _purge(now)
        if user_text:
            _turns.append({"role": "user", "text": user_text, "ts": now})
        if assistant_text:
            _turns.append({"role": "assistant", "text": assistant_text, "ts": now})
        if len(_turns) > 24:
            _turns = _turns[-24:]
        _last_ts = now


def snapshot():
    """Return (age_sec or None, turns_copy)."""
    now = time.time()
    with _lock:
        _purge(now)
        if not _turns or not _last_ts:
            return None, []
        return now - _last_ts, list(_turns)


def context_for_prompt(max_turns: int = 12) -> str:
    """Plain-text recent dialogue for session instructions."""
    age, turns = snapshot()
    if not turns:
        return ""
    lines = []
    for t in turns[-max_turns:]:
        who = "User" if t["role"] == "user" else "TARS"
        lines.append(f"{who}: {t['text']}")
    return "\n".join(lines)


def conversation_items(max_turns: int = 12) -> List[dict]:
    """Realtime conversation.item.create payloads for prior turns."""
    age, turns = snapshot()
    items = []
    for t in turns[-max_turns:]:
        role = "user" if t["role"] == "user" else "assistant"
        if role == "user":
            content = [{"type": "input_text", "text": t["text"]}]
        else:
            content = [{"type": "text", "text": t["text"]}]
        items.append({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": role,
                "content": content,
            },
        })
    return items
