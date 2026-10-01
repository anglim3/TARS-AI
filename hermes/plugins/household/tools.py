"""Hermes tool handlers for the house, the calendar, and the task list.

Secrets are read from the environment at call time. This file has names
only. A missing name returns one spoken error and does not open a socket.
"""

import os
from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None

HA_TIMEOUT = 15
TODOIST_API = "https://api.todoist.com/api/v1"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
CALENDAR_API = "https://www.googleapis.com/calendar/v3/calendars"


def _env(name):
    return (os.environ.get(name) or "").strip()


def _http():
    import requests
    return requests


def _speech(text):
    return str(text or "").strip()


def home(params, http=None, **kwargs):
    """POST the user's phrase to Home Assistant conversation."""
    del kwargs
    phrase = _speech((params or {}).get("text") or (params or {}).get("phrase"))
    if not phrase:
        return "No home command to send."
    token = _env("HA_TOKEN")
    base = _env("HA_URL").rstrip("/")
    if not token or not base:
        return "Home Assistant is not configured."
    client = http or _http()
    try:
        response = client.post(
            f"{base}/api/conversation/process",
            json={"text": phrase},
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            timeout=HA_TIMEOUT,
        )
    except Exception:
        return "Home Assistant did not answer."
    if getattr(response, "ok", False) is not True:
        return "Home Assistant returned an error."
    try:
        body = response.json()
        said = body["response"]["speech"]["plain"]["speech"]
    except Exception:
        return "Home Assistant returned an error."
    if not said:
        return "Home Assistant returned an error."
    return _speech(said)


def _zone():
    name = _env("CALENDAR_TIMEZONE") or "UTC"
    if ZoneInfo is None:
        return timezone.utc
    try:
        return ZoneInfo(name)
    except Exception:
        return timezone.utc


def _calendar_token(http):
    client_id = _env("GOOGLE_OAUTH_CLIENT_ID")
    client_secret = _env("GOOGLE_OAUTH_CLIENT_SECRET")
    refresh = _env("GOOGLE_OAUTH_REFRESH_TOKEN")
    if not client_id or not client_secret or not refresh:
        return None
    response = http.post(
        GOOGLE_TOKEN_URL,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh,
            "grant_type": "refresh_token",
        },
        timeout=HA_TIMEOUT,
    )
    if getattr(response, "ok", False) is not True:
        return None
    return (response.json() or {}).get("access_token") or None


def _calendar_id():
    return _env("CALENDAR_ID") or "primary"


def calendar_agenda(params, http=None, **kwargs):
    """Remaining events today, or tomorrow when the call says so."""
    del kwargs
    when = _speech((params or {}).get("when") or "today").lower()
    client = http or _http()
    token = _calendar_token(client) if (_env("GOOGLE_OAUTH_CLIENT_ID") and _env("GOOGLE_OAUTH_REFRESH_TOKEN")) else None
    if not token:
        return "Google Calendar is not configured."
    zone = _zone()
    now = datetime.now(zone)
    start_day = now
    if when == "tomorrow":
        start_day = now + timedelta(days=1)
        window_start = start_day.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        window_start = now
    window_end = start_day.replace(hour=23, minute=59, second=59, microsecond=0)
    url = f"{CALENDAR_API}/{_calendar_id()}/events"
    try:
        response = client.get(
            url,
            params={
                "timeMin": window_start.isoformat(),
                "timeMax": window_end.isoformat(),
                "singleEvents": "true",
                "orderBy": "startTime",
            },
            headers={"Authorization": f"Bearer {token}"},
            timeout=HA_TIMEOUT,
        )
    except Exception:
        return "Calendar did not answer."
    if getattr(response, "ok", False) is not True:
        return "Calendar returned an error."
    items = (response.json() or {}).get("items") or []
    if not items:
        label = "tomorrow" if when == "tomorrow" else "the rest of today"
        return f"Nothing on your calendar for {label}."
    lines = []
    for item in items[:5]:
        title = item.get("summary") or "untitled"
        start = (item.get("start") or {}).get("dateTime") or (item.get("start") or {}).get("date") or ""
        lines.append(f"{title} at {start}" if start else title)
    return "; ".join(lines)


def calendar_add(params, http=None, **kwargs):
    """Create one event and speak the time the API stored."""
    del kwargs
    title = _speech((params or {}).get("title"))
    start_raw = _speech((params or {}).get("start"))
    try:
        minutes = int((params or {}).get("duration_minutes") or 30)
    except (TypeError, ValueError):
        minutes = 30
    if not title or not start_raw:
        return "Need a title and a start time to add a calendar event."
    client = http or _http()
    token = _calendar_token(client) if _env("GOOGLE_OAUTH_REFRESH_TOKEN") else None
    if not token:
        return "Google Calendar is not configured."
    zone = _zone()
    try:
        start = datetime.fromisoformat(start_raw)
    except ValueError:
        return "Could not read that start time."
    if start.tzinfo is None:
        start = start.replace(tzinfo=zone)
    end = start + timedelta(minutes=max(minutes, 1))
    url = f"{CALENDAR_API}/{_calendar_id()}/events"
    try:
        response = client.post(
            url,
            json={
                "summary": title,
                "start": {"dateTime": start.isoformat(), "timeZone": _env("CALENDAR_TIMEZONE") or "UTC"},
                "end": {"dateTime": end.isoformat(), "timeZone": _env("CALENDAR_TIMEZONE") or "UTC"},
            },
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            timeout=HA_TIMEOUT,
        )
    except Exception:
        return "Calendar did not answer."
    if getattr(response, "ok", False) is not True:
        return "Calendar did not accept that event."
    stored = ((response.json() or {}).get("start") or {}).get("dateTime") or start.isoformat()
    return f"On your calendar at {stored}: {title}."


def _todoist_headers():
    token = _env("TODOIST_API_TOKEN")
    if not token:
        return None
    return {"Authorization": f"Bearer {token}"}


def tasks_list(params, http=None, **kwargs):
    del kwargs
    headers = _todoist_headers()
    if not headers:
        return "Todoist is not configured."
    when = _speech((params or {}).get("filter") or (params or {}).get("when") or "").lower()
    query = {}
    if when in ("today", "inbox") or when:
        query["filter"] = when or "today"
    client = http or _http()
    try:
        response = client.get(f"{TODOIST_API}/tasks", params=query, headers=headers, timeout=HA_TIMEOUT)
    except Exception:
        return "Todoist did not answer."
    if getattr(response, "ok", False) is not True:
        return "Todoist returned an error."
    tasks = response.json() or []
    if isinstance(tasks, dict):
        tasks = tasks.get("results") or tasks.get("items") or []
    if not tasks:
        return "No matching tasks."
    names = [item.get("content") or "untitled" for item in tasks[:8]]
    return "Tasks: " + "; ".join(names)


def tasks_add(params, http=None, **kwargs):
    del kwargs
    headers = _todoist_headers()
    if not headers:
        return "Todoist is not configured."
    content = _speech((params or {}).get("content"))
    if not content:
        return "Need the task to add."
    body = {"content": content}
    due = _speech((params or {}).get("due"))
    if due:
        body["due_string"] = due
    client = http or _http()
    headers = dict(headers)
    headers["Content-Type"] = "application/json"
    try:
        response = client.post(f"{TODOIST_API}/tasks", json=body, headers=headers, timeout=HA_TIMEOUT)
    except Exception:
        return "Todoist did not answer."
    if getattr(response, "ok", False) is not True:
        return "Todoist did not accept that task."
    saved = (response.json() or {}).get("content") or content
    if due:
        return f"Added: {saved}, due {due}."
    return f"Added: {saved}."


def tasks_complete(params, http=None, **kwargs):
    """Close one task when the spoken title matches exactly one active task."""
    del kwargs
    headers = _todoist_headers()
    if not headers:
        return "Todoist is not configured."
    title = _speech((params or {}).get("title") or (params or {}).get("content")).lower()
    if not title:
        return "Need the task to close."
    client = http or _http()
    try:
        listed = client.get(f"{TODOIST_API}/tasks", headers=headers, timeout=HA_TIMEOUT)
    except Exception:
        return "Todoist did not answer."
    if getattr(listed, "ok", False) is not True:
        return "Todoist returned an error."
    tasks = listed.json() or []
    if isinstance(tasks, dict):
        tasks = tasks.get("results") or tasks.get("items") or []
    matches = []
    for item in tasks:
        content = (item.get("content") or "").strip()
        if title == content.lower() or title in content.lower():
            matches.append(item)
    if not matches:
        return "No matching task."
    if len(matches) > 1:
        names = "; ".join(item.get("content") or "untitled" for item in matches[:5])
        return f"More than one match: {names}."
    task = matches[0]
    task_id = task.get("id")
    try:
        closed = client.post(f"{TODOIST_API}/tasks/{task_id}/close", headers=headers, timeout=HA_TIMEOUT)
    except Exception:
        return "Todoist did not answer."
    if getattr(closed, "ok", False) is not True:
        return "Todoist did not close that task."
    due = task.get("due") or {}
    if due.get("is_recurring"):
        return f"Closed: {task.get('content')}. It repeats, so the next one is still open."
    return f"Closed: {task.get('content')}."


_NO_INVENT = (
    "Do not invent temperatures, agendas, or whether a device is on. "
    "Call the tool, then speak only from its result."
)

TOOLS = [
    {
        "name": "home",
        "description": "Send the user's words, unchanged, to Home Assistant. " + _NO_INVENT,
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The user's phrase, unchanged."},
            },
            "required": ["text"],
        },
    },
    {
        "name": "calendar_agenda",
        "description": "List remaining events today, or tomorrow when asked. " + _NO_INVENT,
        "parameters": {
            "type": "object",
            "properties": {
                "when": {"type": "string", "description": "today or tomorrow."},
            },
        },
    },
    {
        "name": "calendar_create",
        "description": "Add one calendar event. Speak the time the API stored. " + _NO_INVENT,
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "start": {"type": "string", "description": "ISO start time."},
                "duration_minutes": {"type": "integer"},
            },
            "required": ["title", "start"],
        },
    },
    {
        "name": "tasks_list",
        "description": "List active Todoist tasks. Default filter is today when the user says today. " + _NO_INVENT,
        "parameters": {
            "type": "object",
            "properties": {
                "filter": {"type": "string", "description": "today, inbox, or a filter the user named."},
            },
        },
    },
    {
        "name": "tasks_add",
        "description": "Add a Todoist task, with a due string when the user gave one. " + _NO_INVENT,
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "due": {"type": "string"},
            },
            "required": ["content"],
        },
    },
    {
        "name": "tasks_complete",
        "description": "Close one Todoist task when exactly one title matches. Two matches: say them and do not close. " + _NO_INVENT,
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
            },
            "required": ["title"],
        },
    },
]
