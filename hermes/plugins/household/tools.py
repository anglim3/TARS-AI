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
    """Control an on/off request from the live entity list, or ask conversation.

    The realtime model is told to call ha_call_service first and to use this
    tool only as a fallback. On and off are still decided from the live entity
    list before conversation, because conversation often returns a non-error
    sentence that never switched the lights.
    """
    del kwargs
    phrase = _speech((params or {}).get("text") or (params or {}).get("phrase"))
    if not phrase:
        return "No home command to send."
    token = _env("HA_TOKEN")
    base = _env("HA_URL").rstrip("/")
    if not token or not base:
        return "Home Assistant is not configured."
    client = http or _http()
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


_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
CALENDAR_MAX_DAYS = 31


def _parse_day(text, today):
    """YYYY-MM-DD, today, tomorrow, or a weekday name (next occurrence, today counts). None if unknown."""
    text = _speech(text).lower()
    if not text:
        return None
    if text == "today":
        return today
    if text == "tomorrow":
        return today + timedelta(days=1)
    if text in _WEEKDAYS:
        return today + timedelta(days=(_WEEKDAYS.index(text) - today.weekday()) % 7)
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _day_label(day):
    return f"{day.strftime('%A')}, {day.strftime('%B')} {day.day}, {day.year} ({day.isoformat()})"


def _calendar_events(client, token, window_start, window_end):
    response = client.get(
        f"{CALENDAR_API}/{_calendar_id()}/events",
        params={
            "timeMin": window_start.isoformat(),
            "timeMax": window_end.isoformat(),
            "singleEvents": "true",
            "orderBy": "startTime",
            "maxResults": "50",
        },
        headers={"Authorization": f"Bearer {token}"},
        timeout=HA_TIMEOUT,
    )
    if getattr(response, "ok", False) is not True:
        return None
    return (response.json() or {}).get("items") or []


def _event_line(item, zone, with_day, window_start=None):
    title = item.get("summary") or "untitled"
    start = (item.get("start") or {}).get("dateTime") or (item.get("start") or {}).get("date") or ""
    if not start:
        return title
    if "T" in start:
        try:
            dt = datetime.fromisoformat(start.replace("Z", "+00:00")).astimezone(zone)
        except ValueError:
            return f"{title} at {start}"
        day = dt.strftime("%a %b ") + str(dt.day)
        if window_start is not None and dt < window_start:
            return f"{title} (continuing, started {day} at {_clock(start, zone)})"
        return f"{day}: {title} at {_clock(start, zone)}" if with_day else f"{title} at {_clock(start, zone)}"
    try:
        d = datetime.strptime(start[:10], "%Y-%m-%d").date()
        day = d.strftime("%a %b ") + str(d.day)
        if window_start is not None and d < window_start.date():
            return f"{title} (all day, continuing, started {day})"
    except ValueError:
        day = start
    return f"{day}: {title} (all day)" if with_day else f"{title} (all day)"


def calendar_agenda(params, http=None, **kwargs):
    """Events for one day or a date range. The reply names the exact dates queried.

    params: date (YYYY-MM-DD or weekday name), or start/end (YYYY-MM-DD, inclusive),
    or start + days; legacy when=today|tomorrow (today = rest of today, falls back to tomorrow).
    """
    del kwargs
    params = params or {}
    client = http or _http()
    token = _calendar_token(client) if (_env("GOOGLE_OAUTH_CLIENT_ID") and _env("GOOGLE_OAUTH_REFRESH_TOKEN")) else None
    if not token:
        return "Google Calendar is not configured."
    zone = _zone()
    now = datetime.now(zone)
    today = now.date()
    date_arg = params.get("date")
    start_arg = params.get("start")
    end_arg = params.get("end")
    days_arg = params.get("days")
    when = _speech(params.get("when") or "").lower()
    explicit = bool(date_arg or start_arg or end_arg or days_arg) or (when and when not in ("today", "tomorrow"))
    phrase = when.replace("the ", "").strip()
    if phrase in ("this weekend", "weekend", "next weekend", "this week", "week", "next week") and not (date_arg or start_arg):
        wd = today.weekday()
        if phrase in ("this weekend", "weekend"):
            sat = today if wd == 5 else today + timedelta(days=(5 - wd) % 7)
            start_arg = (today if wd == 6 else sat).isoformat()
            end_arg = (today if wd == 6 else sat + timedelta(days=1)).isoformat()
        elif phrase == "next weekend":
            sat = today + timedelta(days=(5 - wd) % 7 or 7)
            if wd in (5, 6):
                sat = today + timedelta(days=(12 - wd))
            start_arg, end_arg = sat.isoformat(), (sat + timedelta(days=1)).isoformat()
        elif phrase in ("this week", "week"):
            start_arg, end_arg = today.isoformat(), (today + timedelta(days=6 - wd)).isoformat()
        else:
            mon = today + timedelta(days=7 - wd)
            start_arg, end_arg = mon.isoformat(), (mon + timedelta(days=6)).isoformat()
        when = ""
    if explicit:
        first = _parse_day(date_arg or start_arg or (when if not (date_arg or start_arg) else ""), today)
        if first is None and not (date_arg or start_arg) and (end_arg or days_arg):
            first = today
        if first is None:
            return "Give the day as YYYY-MM-DD. Today is " + _day_label(today) + "."
        last = first
        if end_arg and not date_arg:
            last = _parse_day(end_arg, today)
            if last is None:
                return "Give the end day as YYYY-MM-DD."
        elif days_arg and not date_arg:
            try:
                count = max(1, int(days_arg))
            except (TypeError, ValueError):
                count = 1
            last = first + timedelta(days=count - 1)
        if last < first:
            first, last = last, first
        if (last - first).days >= CALENDAR_MAX_DAYS:
            last = first + timedelta(days=CALENDAR_MAX_DAYS - 1)
        window_start = datetime(first.year, first.month, first.day, tzinfo=zone)
        window_end = datetime(last.year, last.month, last.day, 23, 59, 59, tzinfo=zone)
        if first == last:
            label = _day_label(first)
        else:
            label = f"{_day_label(first)} through {_day_label(last)}"
        try:
            items = _calendar_events(client, token, window_start, window_end)
        except Exception:
            return "Calendar did not answer."
        if items is None:
            return "Calendar returned an error."
        if not items:
            return f"Calendar for {label}: nothing scheduled."
        lines = [_event_line(item, zone, first != last, window_start) for item in items[:12]]
        more = f" (+{len(items) - 12} more)" if len(items) > 12 else ""
        return f"Calendar for {label}: " + "; ".join(lines) + more

    # Legacy today/tomorrow behavior, now naming the date queried.
    if when == "tomorrow":
        day = today + timedelta(days=1)
        window_start = datetime(day.year, day.month, day.day, tzinfo=zone)
    else:
        day = today
        window_start = now
    window_end = datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=zone)
    try:
        items = _calendar_events(client, token, window_start, window_end)
    except Exception:
        return "Calendar did not answer."
    if items is None:
        return "Calendar returned an error."
    label = f"tomorrow, {_day_label(day)}" if when == "tomorrow" else f"the rest of today, {_day_label(day)}"
    if not items and when != "tomorrow":
        nxt = today + timedelta(days=1)
        try:
            later = _calendar_events(
                client, token,
                datetime(nxt.year, nxt.month, nxt.day, tzinfo=zone),
                datetime(nxt.year, nxt.month, nxt.day, 23, 59, 59, tzinfo=zone),
            ) or []
        except Exception:
            later = []
        if later:
            lines = [_event_line(item, zone, False) for item in later[:5]]
            return f"Nothing more today. Tomorrow, {_day_label(nxt)}: " + "; ".join(lines)
    if not items:
        return f"Nothing on your calendar for {label}."
    lines = [_event_line(item, zone, False) for item in items[:5]]
    return f"Calendar for {label}: " + "; ".join(lines)


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
        "description": (
            "Read events from the calendar for any day or date range (read-only). "
            "One day: date=YYYY-MM-DD. Range: start=YYYY-MM-DD and end=YYYY-MM-DD (inclusive), or start plus days. "
            "when=today (rest of today) or when=tomorrow still work. Resolve weekday words like Wednesday, next Friday, "
            "this weekend, this week to real dates from today's date in your instructions. "
            "The result names the exact day and date queried; say that day and date. " + _NO_INVENT
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "One day, YYYY-MM-DD."},
                "start": {"type": "string", "description": "Range start, YYYY-MM-DD."},
                "end": {"type": "string", "description": "Range end, YYYY-MM-DD, inclusive."},
                "days": {"type": "integer", "description": "Number of days from start (default 1, max 31)."},
                "when": {"type": "string", "description": "today, tomorrow, a weekday name, this weekend, next weekend, this week, or next week."},
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


# ---------------------------------------------------------------------------
# Direct Home Assistant control: states, services, service calls.
# Names and entity ids come from live HA state. Secrets are read from the
# environment at call time and are never logged or returned.
# ---------------------------------------------------------------------------

import json as _json
import re as _re
import time as _time

HA_DIRECT_TIMEOUT = 5
HA_STATE_TTL = 3.0
_STATE_CACHE = {"at": 0.0, "states": None}
_SERVICE_CACHE = {"at": 0.0, "services": None}
_SLUG = _re.compile(r"^[a-z0-9_]+$")
_ENTITY = _re.compile(r"^[a-z0-9_]+\.[a-z0-9_]+$")
# Voice may never open locks or disarm alarms.
_BLOCKED_SERVICES = {
    ("lock", "unlock"),
    ("lock", "open"),
    ("alarm_control_panel", "alarm_disarm"),
}
# These domains need explicitly named entity ids (no area, no "all").
_EXPLICIT_DOMAINS = {"lock", "alarm_control_panel"}
_KEY_ATTRS = (
    "brightness", "color_mode", "rgb_color", "hs_color", "color_temp_kelvin",
    "supported_color_modes", "min_color_temp_kelvin", "max_color_temp_kelvin",
    "effect", "temperature", "current_temperature", "target_temp_high",
    "target_temp_low", "hvac_modes", "hvac_action", "fan_mode", "preset_mode",
    "volume_level", "is_volume_muted", "media_title", "source",
    "current_position", "percentage", "unit_of_measurement", "device_class",
)
_ROSTER_DOMAINS = (
    "light", "switch", "fan", "cover", "climate", "media_player", "scene",
    "script", "lock", "input_boolean", "vacuum", "humidifier", "water_heater",
    "siren", "valve", "alarm_control_panel",
)


def _ha_log(text):
    try:
        from modules.module_messageQue import queue_message
        queue_message(text)
    except Exception:
        print(text, flush=True)


def _ha_config():
    token = _env("HA_TOKEN")
    base = _env("HA_URL").rstrip("/")
    return base, token


def _ha_headers(token):
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _fold(text):
    return str(text or "").lower().replace("\u2019", "'").replace("\u2018", "'")


def _ha_all_states(client=None, fresh=False):
    """Every HA state, cached for a few seconds. None when HA is unreachable."""
    now = _time.monotonic()
    if not fresh and _STATE_CACHE["states"] is not None and now - _STATE_CACHE["at"] < HA_STATE_TTL:
        return _STATE_CACHE["states"]
    base, token = _ha_config()
    if not base or not token:
        return None
    client = client or _http()
    try:
        listed = client.get(f"{base}/api/states", headers=_ha_headers(token), timeout=HA_DIRECT_TIMEOUT)
    except Exception:
        return None
    if getattr(listed, "ok", False) is not True:
        return None
    try:
        states = listed.json()
    except Exception:
        return None
    if not isinstance(states, list):
        return None
    _STATE_CACHE["states"] = states
    _STATE_CACHE["at"] = now
    return states


def _compact_state(item):
    attrs = item.get("attributes") or {}
    row = {
        "entity_id": item.get("entity_id"),
        "name": attrs.get("friendly_name") or item.get("entity_id"),
        "state": item.get("state"),
    }
    for key in _KEY_ATTRS:
        value = attrs.get(key)
        if value is None or value == [] or value == "":
            continue
        if key == "brightness":
            try:
                row["brightness_pct"] = round(int(value) * 100 / 255)
            except (TypeError, ValueError):
                pass
            continue
        if key == "hs_color" and isinstance(value, (list, tuple)):
            value = [round(float(v), 1) for v in value]
        row[key] = value
    return row


def _color_support(item):
    modes = (item.get("attributes") or {}).get("supported_color_modes") or []
    if any(mode in modes for mode in ("hs", "rgb", "rgbw", "rgbww", "xy")):
        return "color"
    if "color_temp" in modes:
        return "white-temp"
    if "brightness" in modes:
        return "dim-only"
    if modes:
        return "on-off"
    return ""


def ha_states(params, http=None, **kwargs):
    """Compact live states, optionally filtered by domain and a search string."""
    del kwargs
    params = params or {}
    domain = _fold(params.get("domain")).strip()
    search = _fold(params.get("search")).strip()
    if not all(_ha_config()):
        return "Home Assistant is not configured."
    states = _ha_all_states(http)
    if states is None:
        return "Home Assistant did not answer."
    rows = []
    for item in states:
        entity_id = str(item.get("entity_id", ""))
        if domain and not entity_id.startswith(domain + "."):
            continue
        if search:
            name = _fold((item.get("attributes") or {}).get("friendly_name"))
            words = [w for w in _re.split(r"[\s_.]+", search) if w]
            hay = f"{entity_id} {name}"
            if not all(w in hay or w.rstrip("s") in hay for w in words):
                continue
        rows.append(_compact_state(item))
        if len(rows) >= 60:
            break
    return _json.dumps({"count": len(rows), "entities": rows}, ensure_ascii=False)


def ha_services(params, http=None, **kwargs):
    """Services and field names for one domain, from /api/services."""
    del kwargs
    domain = _fold((params or {}).get("domain")).strip()
    base, token = _ha_config()
    if not base or not token:
        return "Home Assistant is not configured."
    now = _time.monotonic()
    services = _SERVICE_CACHE["services"]
    if services is None or now - _SERVICE_CACHE["at"] > 300:
        client = http or _http()
        try:
            listed = client.get(f"{base}/api/services", headers=_ha_headers(token), timeout=HA_DIRECT_TIMEOUT)
            services = listed.json() if getattr(listed, "ok", False) is True else None
        except Exception:
            services = None
        if not isinstance(services, list):
            return "Home Assistant did not answer."
        _SERVICE_CACHE["services"] = services
        _SERVICE_CACHE["at"] = now
    if not domain:
        return _json.dumps({"domains": sorted(str(d.get("domain")) for d in services)})
    for entry in services:
        if entry.get("domain") != domain:
            continue
        out = {}
        for name, spec in (entry.get("services") or {}).items():
            fields = (spec or {}).get("fields") or {}
            flat = []
            for field, info in fields.items():
                if isinstance(info, dict) and isinstance(info.get("fields"), dict):
                    flat.extend(info["fields"].keys())
                else:
                    flat.append(field)
            out[name] = flat
        return _json.dumps({"domain": domain, "services": out})
    return f"No services for domain {domain}."


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    return []


def _as_dict(value):
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = _json.loads(value)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def ha_call_service(params, http=None, **kwargs):
    """POST /api/services/{domain}/{service}, then return the targets' new states."""
    del kwargs
    params = params or {}
    domain = _fold(params.get("domain")).strip()
    service = _fold(params.get("service")).strip()
    if "." in service and not domain:
        domain, service = service.split(".", 1)
    if not _SLUG.match(domain or "") or not _SLUG.match(service or ""):
        return "Need a valid domain and service, for example light and turn_on."
    if (domain, service) in _BLOCKED_SERVICES:
        return f"{domain}.{service} is not allowed by voice. Do it from the Home Assistant app."
    data = _as_dict(params.get("data"))
    entity_ids = _as_list(params.get("entity_ids") or params.get("entity_id"))
    entity_ids += [e for e in _as_list(data.pop("entity_id", None)) if e not in entity_ids]
    areas = _as_list(params.get("area") or params.get("area_id"))
    areas += [a for a in _as_list(data.pop("area_id", None)) if a not in areas]
    entity_ids = [e.lower() for e in entity_ids]
    bad = [e for e in entity_ids if e != "all" and not _ENTITY.match(e)]
    if bad:
        return "Use entity ids like light.kitchen, not names. Bad: " + ", ".join(bad[:5])
    target_domains = {e.split(".", 1)[0] for e in entity_ids if e != "all"}
    if (domain in _EXPLICIT_DOMAINS or target_domains & _EXPLICIT_DOMAINS) and (areas or "all" in entity_ids or not entity_ids):
        return "Locks and alarms need explicitly named entity ids."
    if any(e.startswith("lock.") for e in entity_ids) and service in ("unlock", "open"):
        return "Unlocking is not allowed by voice."
    if any(e.startswith("alarm_control_panel.") for e in entity_ids) and service == "alarm_disarm":
        return "Disarming is not allowed by voice."
    if isinstance(data.get("color_name"), str):
        data["color_name"] = data["color_name"].strip().lower()
    base, token = _ha_config()
    if not base or not token:
        return "Home Assistant is not configured."
    client = http or _http()
    known = _ha_all_states(client)
    if known is not None and entity_ids:
        ids = {str(item.get("entity_id")) for item in known}
        unknown = [e for e in entity_ids if e != "all" and e not in ids]
        if unknown:
            return "Unknown entity ids: " + ", ".join(unknown[:5]) + ". Call ha_states to look them up."
    before = {}
    if known is not None:
        for item in known:
            if item.get("entity_id") in entity_ids:
                before[item["entity_id"]] = _compact_state(item)
    body = dict(data)
    if entity_ids:
        body["entity_id"] = entity_ids if len(entity_ids) > 1 else entity_ids[0]
    if areas:
        body["area_id"] = areas if len(areas) > 1 else areas[0]
    _ha_log(
        f"INFO: HA call {domain}.{service} entities={','.join(entity_ids) or '-'}"
        f" areas={','.join(areas) or '-'} data_keys={','.join(sorted(data)) or '-'}"
    )
    try:
        response = client.post(
            f"{base}/api/services/{domain}/{service}",
            json=body,
            headers=_ha_headers(token),
            timeout=HA_DIRECT_TIMEOUT,
        )
    except Exception:
        _ha_log(f"WARN: HA call {domain}.{service} failed: no answer")
        return "Home Assistant did not answer. Nothing confirmed."
    _STATE_CACHE["states"] = None
    if getattr(response, "ok", False) is not True:
        code = getattr(response, "status_code", "?")
        detail = ""
        try:
            detail = str(response.json().get("message") or "")[:160]
        except Exception:
            detail = str(getattr(response, "text", "") or "")[:160]
        _ha_log(f"WARN: HA call {domain}.{service} HTTP {code}")
        return f"Home Assistant rejected {domain}.{service} (HTTP {code}). {detail}".strip()
    changed = []
    try:
        changed = response.json() or []
    except Exception:
        changed = []
    # Device state can land a moment after the call returns; re-read targets.
    results = []
    settled = True
    if entity_ids and "all" not in entity_ids:
        # Poll until every target reports a change (devices like LIFX lag), max ~2.5s.
        deadline = _time.monotonic() + 2.5
        while True:
            _time.sleep(0.4)
            fresh = _ha_all_states(client, fresh=True) or []
            by_id = {str(item.get("entity_id")): item for item in fresh}
            results = [_compact_state(by_id[e]) for e in entity_ids if e in by_id]
            settled = all(row != before.get(row["entity_id"]) for row in results)
            if settled or _time.monotonic() >= deadline:
                break
    else:
        for item in changed if isinstance(changed, list) else []:
            if isinstance(item, dict) and item.get("entity_id"):
                results.append(_compact_state(item))
        results = results[:30]
    return _json.dumps(
        {"ok": True, "called": f"{domain}.{service}", "data": data, "results": results,
         **({} if settled else {"note": "some targets show no change yet (already in that state, or still updating)"})},
        ensure_ascii=False,
    )


def ha_roster(http=None):
    """Compact live roster for the realtime instructions. Empty when HA is down."""
    states = _ha_all_states(http)
    if not states:
        return ""
    lines = []
    for item in states:
        entity_id = str(item.get("entity_id", ""))
        domain = entity_id.split(".", 1)[0]
        if domain not in _ROSTER_DOMAINS:
            continue
        name = (item.get("attributes") or {}).get("friendly_name") or entity_id
        bits = [entity_id, f'"{name}"', str(item.get("state"))]
        if domain == "light":
            support = _color_support(item)
            if support:
                bits.append(support)
        lines.append(" | ".join(bits))
        if len(lines) >= 80:
            break
    if not lines:
        return ""
    return "Live Home Assistant devices (entity_id | name | state | color support):\n" + "\n".join(lines)


_HA_DIRECT_TOOLS = [
    {
        "name": "ha_states",
        "description": (
            "Read live Home Assistant entity states: entity_id, name, state, and key attributes "
            "(brightness_pct, rgb_color, hs_color, color_temp_kelvin, supported_color_modes, temperature, etc.). "
            "Filter by domain (light, switch, climate, media_player, ...) and/or a search string matched against id and name."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Optional domain, e.g. light."},
                "search": {"type": "string", "description": "Optional words to match in entity id or name, e.g. lamp."},
            },
        },
    },
    {
        "name": "ha_call_service",
        "description": (
            "Directly control Home Assistant devices: POST /api/services/{domain}/{service} and return the targets' resulting states. "
            "Examples: light.turn_on with data {color_name:'red'} or {rgb_color:[255,0,0]}, {brightness_pct:40}, "
            "{color_temp_kelvin:2700}, {transition:2}; light.turn_off; switch.turn_on/turn_off/toggle; "
            "climate.set_temperature {temperature:70}; climate.set_hvac_mode; media_player.volume_set {volume_level:0.3}; "
            "media_player.media_pause; scene.turn_on; script.turn_on; cover.open_cover; fan.set_percentage. "
            "Pass exact entity_ids from the roster or ha_states. Unlock and disarm are refused."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Service domain, e.g. light."},
                "service": {"type": "string", "description": "Service name, e.g. turn_on."},
                "entity_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Exact entity ids to target, e.g. [\"light.x\", \"light.y\"].",
                },
                "area": {"type": "string", "description": "Optional Home Assistant area id instead of entity ids."},
                "data": {
                    "type": "object",
                    "description": "Service data, e.g. {\"color_name\": \"red\", \"brightness_pct\": 80}.",
                },
            },
            "required": ["domain", "service"],
        },
    },
    {
        "name": "ha_services",
        "description": "List Home Assistant services and their field names for one domain, to discover capabilities. No domain lists all domains.",
        "parameters": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Domain, e.g. light or climate."},
            },
        },
    },
]

TOOLS.extend(_HA_DIRECT_TOOLS)


# ---------------------------------------------------------------------------
# Weather: Open-Meteo forecast + geocoding (no API key).
# Home location comes from Home Assistant /api/config (lat/lon only, never
# logged). Results are cached for WEATHER_TTL seconds per location.
# ---------------------------------------------------------------------------

WEATHER_API = "https://api.open-meteo.com/v1/forecast"
GEOCODE_API = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_TIMEOUT = 5
WEATHER_TTL = 600.0
WEATHER_HOME_TZ = "America/New_York"
WEATHER_DAYS = 8  # today + 7
# module_xai re-executes this file for every tool call, so the caches live
# on a process-wide holder instead of in module globals.
import sys as _sys
import types as _types
_WX = _sys.modules.get("_household_weather_cache")
if _WX is None:
    _WX = _types.ModuleType("_household_weather_cache")
    _WX.forecast, _WX.geo, _WX.home = {}, {}, {"at": 0.0, "coords": None}
    _sys.modules["_household_weather_cache"] = _WX
_WEATHER_CACHE = _WX.forecast
_GEO_CACHE = _WX.geo
_HOME_CACHE = _WX.home

_WMO = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "light freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "light freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "heavy showers",
    85: "light snow showers", 86: "heavy snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail", 99: "severe thunderstorms with hail",
}

_US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "new hampshire",
    "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee",
    "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
}

_HOME_WORDS = {"", "home", "here", "house", "my house", "the house", "outside", "local"}


def _wmo(code):
    try:
        return _WMO.get(int(code), "unknown conditions")
    except (TypeError, ValueError):
        return "unknown conditions"


def _deg(value):
    try:
        return f"{round(float(value))} degrees"
    except (TypeError, ValueError):
        return "unknown"


def _inches(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v < 0.005:
        return None
    return f"{v:.2f} inches" if v >= 0.1 else f"{v:.2f} inches"


def _home_coords(client):
    now = _time.time()
    if _HOME_CACHE["coords"] and now - _HOME_CACHE["at"] < 6 * 3600:
        return _HOME_CACHE["coords"]
    coords = None
    base, token = _ha_config()
    if base and token:
        try:
            r = client.get(f"{base}/api/config", headers=_ha_headers(token), timeout=WEATHER_TIMEOUT)
            if getattr(r, "ok", False):
                cfg = r.json() or {}
                lat, lon = cfg.get("latitude"), cfg.get("longitude")
                if lat is not None and lon is not None:
                    coords = (float(lat), float(lon), "home", cfg.get("time_zone") or WEATHER_HOME_TZ)
        except Exception:
            coords = None
    if coords is None:
        lat, lon = _env("WEATHER_LATITUDE"), _env("WEATHER_LONGITUDE")
        try:
            if lat and lon:
                coords = (float(lat), float(lon), "home", WEATHER_HOME_TZ)
        except ValueError:
            coords = None
    if coords is not None:
        _HOME_CACHE.update(at=now, coords=coords)
    return coords


def _geocode(client, place):
    key = _fold(place).strip()
    if key in _GEO_CACHE:
        return _GEO_CACHE[key]
    parts = [p.strip() for p in _re.split(r",", key) if p.strip()]
    name = parts[0] if parts else key
    qualifier = " ".join(parts[1:]).strip()
    if not qualifier:
        words = name.split()
        if len(words) > 1 and words[-1] in _US_STATES:
            name, qualifier = " ".join(words[:-1]), words[-1]
    qualifier = _US_STATES.get(qualifier, qualifier)
    r = client.get(
        GEOCODE_API,
        params={"name": name, "count": 10, "language": "en", "format": "json"},
        timeout=WEATHER_TIMEOUT,
    )
    if not getattr(r, "ok", False):
        return None
    results = (r.json() or {}).get("results") or []
    if not results:
        return None
    pick = None
    if qualifier:
        for item in results:
            hay = _fold(" ".join(str(item.get(k) or "") for k in ("admin1", "admin2", "country", "country_code")))
            if qualifier in hay:
                pick = item
                break
    if pick is None:
        us = [i for i in results if (i.get("country_code") or "").upper() == "US"]
        pick = us[0] if (us and not qualifier) else results[0]
    label = ", ".join(x for x in (pick.get("name"), pick.get("admin1") or pick.get("country")) if x)
    found = (float(pick["latitude"]), float(pick["longitude"]), label, pick.get("timezone") or WEATHER_HOME_TZ)
    _GEO_CACHE[key] = found
    return found


def _forecast(client, lat, lon, tz):
    key = (round(lat, 3), round(lon, 3), tz)
    hit = _WEATHER_CACHE.get(key)
    now = _time.time()
    if hit and now - hit[0] < WEATHER_TTL:
        return hit[1]
    r = client.get(
        WEATHER_API,
        params={
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,"
                       "wind_speed_10m,wind_gusts_10m,precipitation",
            "hourly": "temperature_2m,precipitation_probability,precipitation,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,"
                     "precipitation_sum,wind_speed_10m_max,wind_gusts_10m_max",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
            "timezone": tz,
            "forecast_days": WEATHER_DAYS,
        },
        timeout=WEATHER_TIMEOUT,
    )
    if not getattr(r, "ok", False):
        return None
    data = r.json() or {}
    _WEATHER_CACHE[key] = (now, data)
    return data


def _day_condition(daily, i, hourly):
    """Typical daytime sky (7am-9pm) from hourly codes; the daily code is only the worst hour."""
    d = datetime.strptime(daily["time"][i], "%Y-%m-%d")
    idx = _hour_window(hourly or {}, d + timedelta(hours=7), d + timedelta(hours=21))
    codes = [hourly["weather_code"][k] for k in idx if hourly["weather_code"][k] is not None]
    if not codes:
        return _wmo(daily["weather_code"][i])
    wet = [c for c in codes if c >= 51]
    if len(wet) >= 2:
        return _wmo(max(set(wet), key=wet.count))
    return _wmo(max(set(codes), key=codes.count))


def _daily_line(daily, i, hourly=None):
    d = datetime.strptime(daily["time"][i], "%Y-%m-%d").date()
    bits = [
        _day_condition(daily, i, hourly),
        f"high {_deg(daily['temperature_2m_max'][i])}",
        f"low {_deg(daily['temperature_2m_min'][i])}",
    ]
    pop = daily.get("precipitation_probability_max", [None] * (i + 1))[i]
    if pop is not None:
        bits.append(f"{int(pop)} percent chance of precipitation")
    amt = _inches(daily.get("precipitation_sum", [0] * (i + 1))[i])
    if amt:
        bits.append(f"about {amt}")
    wind = daily.get("wind_speed_10m_max", [None] * (i + 1))[i]
    gust = daily.get("wind_gusts_10m_max", [None] * (i + 1))[i]
    if wind is not None:
        w = f"wind up to {round(wind)} mph"
        if gust is not None and gust >= wind + 8:
            w += f", gusts {round(gust)}"
        bits.append(w)
    return d, ", ".join(bits)


def _hour_window(hourly, start, end):
    """Indices of hourly rows with start <= time < end (naive local datetimes)."""
    out = []
    for i, t in enumerate(hourly.get("time") or []):
        try:
            dt = datetime.strptime(t, "%Y-%m-%dT%H:%M")
        except ValueError:
            continue
        if start <= dt < end:
            out.append(i)
    return out


def _window_line(hourly, idx):
    temps = [hourly["temperature_2m"][i] for i in idx if hourly["temperature_2m"][i] is not None]
    pops = [hourly["precipitation_probability"][i] or 0 for i in idx]
    amts = [hourly["precipitation"][i] or 0 for i in idx]
    winds = [hourly["wind_speed_10m"][i] or 0 for i in idx]
    codes = [hourly["weather_code"][i] for i in idx if hourly["weather_code"][i] is not None]
    wet = [c for c in codes if c >= 45]
    main = max(set(wet or codes), key=(wet or codes).count) if codes else None
    bits = [_wmo(main)]
    if temps:
        bits.append(f"{_deg(max(temps))} falling to {_deg(min(temps))}" if temps[0] >= temps[-1]
                    else f"{_deg(min(temps))} rising to {_deg(max(temps))}")
    bits.append(f"{int(max(pops)) if pops else 0} percent chance of precipitation")
    amt = _inches(sum(amts))
    if amt:
        bits.append(f"about {amt}")
    if winds:
        bits.append(f"wind up to {round(max(winds))} mph")
    return ", ".join(bits)


_PARTS = {"morning": (6, 12), "afternoon": (12, 18), "evening": (18, 22), "night": (18, 30), "tonight": (18, 30)}


def weather(params, http=None, **kwargs):
    """Forecast for home (Home Assistant location) or a named place.

    params: when = now | today | tonight | tomorrow | <weekday> | this weekend |
    next weekend | this week | next 24 hours | YYYY-MM-DD, optionally with
    morning/afternoon/evening/night; date = YYYY-MM-DD; location = place name.
    """
    del kwargs
    params = params or {}
    client = http or _http()
    place = _speech(params.get("location") or "")
    try:
        if _fold(place) in _HOME_WORDS:
            loc = _home_coords(client)
            if loc is None:
                return "Home location is not available from Home Assistant."
        else:
            loc = _geocode(client, place)
            if loc is None:
                return f"Could not find a place called {place}."
        lat, lon, label, tz = loc
        data = _forecast(client, lat, lon, tz)
    except Exception:
        return "The weather service did not answer."
    if not data or "daily" not in data:
        return "The weather service returned an error."
    where = "at home" if label == "home" else f"in {label}"
    daily, hourly, cur = data["daily"], data.get("hourly") or {}, data.get("current") or {}
    days = [datetime.strptime(t, "%Y-%m-%d").date() for t in daily.get("time") or []]
    try:
        zone = ZoneInfo(tz) if ZoneInfo else timezone.utc
    except Exception:
        zone = timezone.utc
    now = datetime.now(zone).replace(tzinfo=None)
    # Day words (today, tomorrow, Friday) are the user's days, in the home zone.
    try:
        today = datetime.now(ZoneInfo(WEATHER_HOME_TZ)).date() if ZoneInfo else now.date()
    except Exception:
        today = now.date()

    when = _fold(params.get("when") or "").replace("the ", "").strip()
    date_arg = _speech(params.get("date") or "")
    part = None
    for word in ("morning", "afternoon", "evening", "tonight", "night"):
        if when.endswith(word):
            part = word
            when = when[: -len(word)].strip() or ("today" if word != "tonight" else "today")
            break
    if when in ("", "current", "currently", "right now", "outside") and not date_arg and part is None:
        when = "now"

    def day_index(d):
        return days.index(d) if d in days else None

    if when == "now":
        bits = [
            f"Right now {where}: {_wmo(cur.get('weather_code'))}, {_deg(cur.get('temperature_2m'))}",
        ]
        feels = cur.get("apparent_temperature")
        if feels is not None and abs(feels - (cur.get("temperature_2m") or feels)) >= 3:
            bits.append(f"feels like {_deg(feels)}")
        if cur.get("wind_speed_10m") is not None:
            bits.append(f"wind {round(cur['wind_speed_10m'])} mph")
        if cur.get("relative_humidity_2m") is not None:
            bits.append(f"humidity {int(cur['relative_humidity_2m'])} percent")
        line = ", ".join(bits) + "."
        i = day_index(today)
        if i is not None:
            d, summary = _daily_line(daily, i, hourly)
            line += f" Today, {_day_label(d)}: {summary}."
        nxt = _hour_window(hourly, now, now + timedelta(hours=12))
        wet = [k for k in nxt if (hourly["precipitation_probability"][k] or 0) >= 40]
        if wet:
            first = datetime.strptime(hourly["time"][wet[0]], "%Y-%m-%dT%H:%M")
            line += f" Precipitation likely from about {first.strftime('%I %p').lstrip('0').lower()}."
        return line

    if when in ("next 24 hours", "24 hours", "hourly", "next few hours", "rest of day", "rest of today"):
        hours = 24 if "24" in when or when == "hourly" else 12
        idx = _hour_window(hourly, now.replace(minute=0), now + timedelta(hours=hours))
        if not idx:
            return "No hourly forecast available."
        steps = []
        for k in idx[::3]:
            t = datetime.strptime(hourly["time"][k], "%Y-%m-%dT%H:%M")
            steps.append(f"{t.strftime('%I %p').lstrip('0').lower()} {_deg(hourly['temperature_2m'][k])} "
                         f"{_wmo(hourly['weather_code'][k])} ({int(hourly['precipitation_probability'][k] or 0)} percent)")
        return f"Next {hours} hours {where} from {_day_label(today)}: {_window_line(hourly, idx)}. By hour: " + "; ".join(steps) + "."

    span = None
    if when in ("this weekend", "weekend", "next weekend", "this week", "week", "next 7 days", "7 days", "week ahead", "next week"):
        wd = today.weekday()
        if when in ("this weekend", "weekend"):
            sat = today if wd == 5 else today + timedelta(days=(5 - wd) % 7)
            span = [today] if wd == 6 else [sat, sat + timedelta(days=1)]
        elif when == "next weekend":
            sat = today + timedelta(days=(5 - wd) % 7 or 7)
            if wd in (5, 6):
                sat = today + timedelta(days=12 - wd)
            span = [sat, sat + timedelta(days=1)]
        elif when in ("this week", "week"):
            span = [today + timedelta(days=k) for k in range(0, 7 - wd)]
        else:
            span = [today + timedelta(days=k) for k in range(0, 7)]
        span = [d for d in span if d in days]
        if not span:
            return f"The forecast only covers {_day_label(days[0])} through {_day_label(days[-1])}."
        lines = []
        for d in span:
            _, summary = _daily_line(daily, days.index(d), hourly)
            lines.append(f"{d.strftime('%A %B')} {d.day}: {summary}")
        return f"Forecast {where}: " + "; ".join(lines) + "."

    if part == "tonight" and not date_arg and when in ("today", "tonight"):
        when = "today"
    target = _parse_day(date_arg or when or "today", today)
    if target is None and when == "tonight":
        target, part = today, "tonight"
    if target is None:
        return "Give the day as YYYY-MM-DD, a weekday, today, tonight, or tomorrow. Today is " + _day_label(today) + "."
    i = day_index(target)
    if i is None:
        return f"The forecast only covers {_day_label(days[0])} through {_day_label(days[-1])}."
    if part:
        h0, h1 = _PARTS[part]
        start = datetime(target.year, target.month, target.day) + timedelta(hours=h0)
        end = datetime(target.year, target.month, target.day) + timedelta(hours=h1)
        if part in ("tonight", "night") and target == now.date() and now.hour < 5:
            # After midnight, tonight means the rest of this night.
            start, end = now, datetime(target.year, target.month, target.day, 7)
        if target == now.date() and start < now:
            start = now.replace(minute=0, second=0, microsecond=0)
        idx = _hour_window(hourly, start, end)
        if not idx:
            return f"No hourly forecast for {part} on {_day_label(target)}."
        name = "tonight" if part in ("tonight", "night") else part
        return f"{_day_label(target)}, {name}, {where}: {_window_line(hourly, idx)}."
    d, summary = _daily_line(daily, i, hourly)
    return f"Weather {where} for {_day_label(d)}: {summary}."


TOOLS.append({
    "name": "weather",
    "description": (
        "Live weather forecast from Open-Meteo: current conditions, next 24 hours, and daily up to 7 days, "
        "in Fahrenheit, mph, and inches. Default location is home. when = now, today, tonight, tomorrow, "
        "a weekday name, this weekend, next weekend, this week, next 24 hours, or YYYY-MM-DD, optionally "
        "ending in morning, afternoon, evening, or night (e.g. 'tomorrow morning'). location = a city "
        "name only when the user names a place. The result names the exact day and date; say it. "
        "Never guess the weather. " + _NO_INVENT
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "when": {"type": "string", "description": "now, today, tonight, tomorrow, Friday, this weekend, next 24 hours, tomorrow morning, ..."},
            "date": {"type": "string", "description": "One day, YYYY-MM-DD (overrides when's day)."},
            "location": {"type": "string", "description": "Place name, e.g. 'Denver, CO'. Omit for home."},
        },
    },
})
