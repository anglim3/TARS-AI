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



def _phrase_words(phrase):
    low = phrase.lower().replace("'", "").replace("\u2019", "")
    return [word for word in low.replace("-", " ").split() if word]


_GENERIC_WORDS = {
    "light", "lights", "lamp", "lamps", "the", "a", "an", "all", "every",
    "my", "our", "on", "off", "turn", "please", "switch", "switches",
}


def _name_tokens(name):
    folded = name.lower().replace("'", "").replace("\u2019", "").replace("-", " ")
    return [word for word in folded.split() if word not in _GENERIC_WORDS and len(word) >= 3]


def _token_hit(phrase_words, token):
    for word in phrase_words:
        if word == token:
            return True
        if len(word) >= 4 and (token.startswith(word) or word.startswith(token)):
            return True
    return False


def _entities(states, prefix):
    found = []
    for item in states:
        entity_id = str(item.get("entity_id", ""))
        if entity_id.startswith(prefix):
            found.append(item)
    return found


def _named_matches(phrase, items):
    words = _phrase_words(phrase)
    matched = []
    for item in items:
        name = str((item.get("attributes") or {}).get("friendly_name") or "")
        if any(_token_hit(words, token) for token in _name_tokens(name)):
            matched.append(item)
    return matched


def _light_targets(phrase, states):
    """Named lights alone, or every light when the phrase just says the lights."""
    lights = _entities(states, "light.")
    named = _named_matches(phrase, lights)
    if named:
        return [item["entity_id"] for item in named]
    if "light" in phrase.lower():
        return [item["entity_id"] for item in lights]
    return []


def _friendly(states, entity_id):
    for item in states:
        if item.get("entity_id") == entity_id:
            name = str((item.get("attributes") or {}).get("friendly_name") or "").strip()
            if name:
                return name
    return entity_id


def _service_verb(phrase):
    low = phrase.lower()
    padded = f" {low}"
    if " off" in padded or low.startswith("off"):
        return "turn_off", "off"
    if " on" in padded or low.startswith("on"):
        return "turn_on", "on"
    return None, None


def _control_fallback(phrase, client, base, token):
    service, verb = _service_verb(phrase)
    if service is None:
        return None
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        listed = client.get(f"{base}/api/states", headers=headers, timeout=HA_TIMEOUT)
        states = listed.json() if getattr(listed, "ok", False) else []
    except Exception:
        return None
    if not isinstance(states, list):
        return None
    lights = _entities(states, "light.")
    switches = _entities(states, "switch.")
    light_ids = _light_targets(phrase, states)
    switch_ids = []
    if not light_ids:
        switch_ids = [item["entity_id"] for item in _named_matches(phrase, switches)]
    targets = light_ids or switch_ids
    if not targets:
        return None
    domain = "light" if light_ids else "switch"
    try:
        done = client.post(
            f"{base}/api/services/{domain}/{service}",
            json={"entity_id": targets},
            headers=headers,
            timeout=HA_TIMEOUT,
        )
    except Exception:
        return None
    if getattr(done, "ok", False) is not True:
        return None
    every_light = light_ids and len(light_ids) == len(lights) and lights
    if every_light:
        return f"Turned the lights {verb}."
    if len(targets) == 1:
        return f"Turned {_friendly(states, targets[0])} {verb}."
    names = ", ".join(_friendly(states, entity_id) for entity_id in targets)
    return f"Turned {names} {verb}."


def device_roster(http=None):
    """One sentence of live light and switch names. Empty when Home Assistant is down."""
    token = _env("HA_TOKEN")
    base = _env("HA_URL").rstrip("/")
    if not token or not base:
        return ""
    client = http or _http()
    headers = {"Authorization": f"Bearer {token}"}
    try:
        listed = client.get(f"{base}/api/states", headers=headers, timeout=HA_TIMEOUT)
    except Exception:
        return ""
    if getattr(listed, "ok", False) is not True:
        return ""
    try:
        states = listed.json()
    except Exception:
        return ""
    if not isinstance(states, list):
        return ""
    bits = []
    for item in states:
        entity_id = str(item.get("entity_id", ""))
        if entity_id.startswith("light."):
            domain = "light"
        elif entity_id.startswith("switch."):
            domain = "switch"
        else:
            continue
        name = str((item.get("attributes") or {}).get("friendly_name") or entity_id)
        bits.append(f"{name} ({domain}, {item.get('state')})")
    if not bits:
        return ""
    return (
        "Known devices: " + "; ".join(bits) + ". "
        "The lights means every light. "
        "A named device is controlled alone."
    )


def home(params, http=None, **kwargs):
    """Control an on/off request from the live entity list, or ask conversation."""
    del kwargs
    phrase = _speech((params or {}).get("text") or (params or {}).get("phrase"))
    if not phrase:
        return "No home command to send."
    token = _env("HA_TOKEN")
    base = _env("HA_URL").rstrip("/")
    if not token or not base:
        return "Home Assistant is not configured."
    client = http or _http()
    # Conversation often returns a non-error sentence that never switched
    # the lights. On and off are decided from the live entity list first.
    direct = _control_fallback(phrase, client, base, token)
    if direct:
        return direct
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        response = client.post(
            f"{base}/api/conversation/process",
            json={"text": phrase},
            headers=headers,
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


def _clock(start, zone):
    """Speak a calendar timestamp in the calendar zone, not raw UTC."""
    if "T" not in (start or ""):
        return start or ""
    try:
        dt = datetime.fromisoformat(start.replace("Z", "+00:00")).astimezone(zone)
    except ValueError:
        return start
    hour = dt.strftime("%I").lstrip("0")
    minute = dt.strftime("%M")
    ampm = dt.strftime("%p").lower()
    if minute == "00":
        return f"{hour}{ampm}"
    return f"{hour}:{minute}{ampm}"


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
    label = "tomorrow" if when == "tomorrow" else "the rest of today"
    if not items and when != "tomorrow":
        start_day = now + timedelta(days=1)
        window_start = start_day.replace(hour=0, minute=0, second=0, microsecond=0)
        window_end = start_day.replace(hour=23, minute=59, second=59, microsecond=0)
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
            response = None
        if response is not None and getattr(response, "ok", False) is True:
            later = (response.json() or {}).get("items") or []
            if later:
                items = later
                label = "tomorrow"
    if not items:
        return f"Nothing on your calendar for {label}."
    lines = []
    for item in items[:5]:
        title = item.get("summary") or "untitled"
        start = (item.get("start") or {}).get("dateTime") or (item.get("start") or {}).get("date") or ""
        spoken = _clock(start, zone) if start else ""
        lines.append(f"{title} at {spoken}" if spoken else title)
    prefix = "Tomorrow: " if label == "tomorrow" and when != "tomorrow" else ""
    return prefix + "; ".join(lines)


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
        "description": "Send the user's words, unchanged, to Home Assistant. The lights means every light. A named device is controlled alone. " + _NO_INVENT,
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
