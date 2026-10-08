"""
module_xai.py

xAI speech adapters.

Speech-to-text streams 16 kHz PCM to wss://api.x.ai/v1/stt. Text-to-speech
posts to https://api.x.ai/v1/tts using XAI_TTS_VOICE_ID. One wake word is
one turn on wss://api.x.ai/v1/realtime: the listen loop ends on a local
RMS gate, and the reply is not played until the shared mic hub has dropped
the USB input. Calendar and Todoist tools stay on that socket. The
loopback client remains for the older text path and is not used by the
realtime turn.
"""

import base64
import hashlib
import io
import json
import os
import threading
import time
from urllib.parse import urlencode

import numpy as np

from modules.module_messageQue import queue_message

XAI_STT_URL = "wss://api.x.ai/v1/stt"
XAI_TTS_URL = "https://api.x.ai/v1/tts"
VOICE_CONVERSATION = "tars-voice"
HERMES_MODEL = "hermes-agent"
STT_REPEAT_LINE = "Say that again."
HERMES_FAILURE_LINE = "I can't reach my brain right now."
SMART_TURN = "0.5"

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tts", "cache")


def xai_api_key():
    return os.environ.get("XAI_API_KEY", "") or ""


def xai_tts_voice_id():
    return os.environ.get("XAI_TTS_VOICE_ID", "") or ""


def hermes_api_key():
    return os.environ.get("HERMES_API_KEY", "") or ""


def stt_keyterms(user_name):
    """Bias the transcriber toward TARS and the configured user name."""
    terms = ["TARS", "calendar"]
    name = (user_name or "").strip()
    if name and name.lower() != "tars":
        terms.append(name[:50])
    return terms


def build_stt_ws_url(user_name):
    pairs = [
        ("sample_rate", "16000"),
        ("encoding", "pcm"),
        ("interim_results", "true"),
        ("language", "en"),
        ("smart_turn", SMART_TURN),
        ("smart_turn_timeout", "3000"),
    ]
    for term in stt_keyterms(user_name):
        pairs.append(("keyterm", term))
    return XAI_STT_URL + "?" + urlencode(pairs)


def utterance_from_stt_message(raw):
    """Classify one STT server event.

    Returns (kind, text). kind is ready, final, partial, done, error, or ignore.
    Partial text is never treated as the utterance.
    """
    try:
        event = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return "ignore", ""
    if not isinstance(event, dict):
        return "ignore", ""
    kind = event.get("type") or ""
    if kind == "transcript.created":
        return "ready", ""
    if kind == "error":
        return "error", str(event.get("message") or "xAI STT error")
    if kind == "transcript.done":
        return "done", (event.get("text") or "").strip()
    if kind == "transcript.partial":
        text = (event.get("text") or "").strip()
        if event.get("speech_final") is True:
            return "final", text
        return "partial", text
    return "ignore", ""


def hermes_responses_url(base_url):
    root = (base_url or "http://127.0.0.1:8642/v1").rstrip("/")
    if root.endswith("/v1"):
        return root + "/responses"
    return root + "/v1/responses"


def hermes_health_url(base_url):
    root = (base_url or "http://127.0.0.1:8642/v1").rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return root.rstrip("/") + "/health"


def hermes_request_body(user_text, conversation=VOICE_CONVERSATION):
    """Body for POST /v1/responses.

    The heard line only. No Amelia JSON schema and no response_format,
    so json_mode stays off for this backend.
    """
    return {
        "model": HERMES_MODEL,
        "input": user_text,
        "conversation": conversation,
        "store": True,
    }


def assistant_text_from_response(payload):
    """Keep the last assistant sentence. Drop tools and reasoning."""
    if not isinstance(payload, dict):
        return ""
    output = payload.get("output")
    if isinstance(output, str):
        return output.strip()
    texts = []
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type") or ""
            if item_type in ("function_call", "function_call_output", "reasoning"):
                continue
            if item_type != "message" and item.get("role") != "assistant":
                continue
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                texts.append(content.strip())
                continue
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in ("reasoning", "reasoning_text"):
                    continue
                bit = part.get("text") or ""
                if bit.strip():
                    texts.append(bit.strip())
    if texts:
        return texts[-1]
    direct = payload.get("output_text")
    if isinstance(direct, str):
        return direct.strip()
    return ""


def _ws_send_pcm(ws, chunk):
    if isinstance(chunk, str):
        chunk = chunk.encode("utf-8")
    send_binary = getattr(ws, "send_binary", None)
    if callable(send_binary):
        send_binary(chunk)
        return
    try:
        ws.send(chunk, opcode=2)
    except TypeError:
        ws.send(chunk)


def _default_connect(url, header, timeout):
    import websocket
    return websocket.create_connection(url, header=header, timeout=timeout)


def stream_pcm_until_speech_final(frames, api_key, user_name, connect=None, recv_timeout=8):
    """Stream PCM bytes until Smart Turn marks speech_final.

    The socket opens after the first frame, so silence before the energy
    gate never leaves the machine. Partial text is discarded on error.
    The socket is closed when the turn ends.

    Returns the utterance, or None when the caller sent no audio.
    Raises RuntimeError when audio was sent and no final text came back.
    """
    iterator = iter(frames)
    try:
        first = next(iterator)
    except StopIteration:
        return None

    url = build_stt_ws_url(user_name)
    opener = connect or _default_connect
    ws = opener(url, header=[f"Authorization: Bearer {api_key}"], timeout=recv_timeout)
    final_text = None
    failed = False
    try:
        kind, detail = utterance_from_stt_message(ws.recv())
        if kind != "ready":
            raise RuntimeError(detail or "xAI STT did not become ready")

        done = threading.Event()

        def reader():
            nonlocal final_text, failed
            try:
                while not done.is_set():
                    raw = ws.recv()
                    event_kind, text = utterance_from_stt_message(raw)
                    if event_kind == "final":
                        final_text = text or None
                        done.set()
                        return
                    if event_kind == "done" and text:
                        final_text = text
                        done.set()
                        return
                    if event_kind == "error":
                        queue_message(f"ERROR: xAI STT: {text}")
                        failed = True
                        done.set()
                        return
            except Exception:
                if not done.is_set():
                    failed = True
                    done.set()

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        try:
            _ws_send_pcm(ws, first)
            for chunk in iterator:
                if done.is_set() or not chunk:
                    if done.is_set():
                        break
                    continue
                _ws_send_pcm(ws, chunk)
        finally:
            if not done.is_set():
                try:
                    ws.send(json.dumps({"type": "audio.done"}))
                except Exception:
                    failed = True
                    done.set()
            done.wait(timeout=recv_timeout)
    finally:
        try:
            ws.close()
        except Exception:
            pass

    if final_text:
        return final_text
    if failed:
        raise RuntimeError("xAI STT socket closed before speech_final")
    raise RuntimeError("xAI STT returned no utterance")


def _cache_path(text, voice_id):
    digest = hashlib.md5(f"{voice_id}:{text}".encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"xai_{digest}.mp3")


async def text_to_speech_with_pipelining_xai(text, is_wakeword=False):
    """Synthesize one clip with the console voice id. Yields a BytesIO buffer.

    Wake-word lines are cached. A missing key or a failed request yields
    nothing so the caller can use Piper for a real reply, or the listen
    beep for a wake ack.
    """
    import requests

    key = xai_api_key()
    voice_id = xai_tts_voice_id()
    if not key or not voice_id:
        queue_message("ERROR: xAI TTS needs XAI_API_KEY and XAI_TTS_VOICE_ID")
        return

    if is_wakeword:
        cached = _cache_path(text, voice_id)
        if os.path.isfile(cached):
            try:
                with open(cached, "rb") as handle:
                    audio = handle.read()
                buffer = io.BytesIO(audio)
                buffer.seek(0)
                yield buffer
                return
            except Exception as exc:
                queue_message(f"ERROR: Failed to load xAI TTS cache: {exc}")

    try:
        response = requests.post(
            XAI_TTS_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "text": text,
                "voice_id": voice_id,
                "language": "en",
            },
            timeout=30,
        )
        response.raise_for_status()
    except Exception as exc:
        queue_message(f"ERROR: xAI TTS failed: {exc}")
        return

    audio = response.content
    if not audio:
        queue_message("ERROR: xAI TTS returned empty audio")
        return

    if is_wakeword:
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(_cache_path(text, voice_id), "wb") as handle:
                handle.write(audio)
        except Exception as exc:
            queue_message(f"ERROR: Failed to cache xAI wake ack: {exc}")

    buffer = io.BytesIO(audio)
    buffer.seek(0)
    yield buffer


XAI_REALTIME_URL = "wss://api.x.ai/v1/realtime?model=grok-voice-latest"
REALTIME_SAMPLE_RATE = 16000
TURN_TAIL_SECONDS = 0.30
TURN_SILENCE_SECONDS = 1.0
TURN_NO_SPEECH_SECONDS = 6.0
TURN_CAP_SECONDS = 12.0
USER_TRANSCRIPT_GRACE = 2.5
REALTIME_TOOL_NAMES = (
    "weather",
    "ha_states",
    "ha_call_service",
    "ha_services",
    "home",
    "calendar_agenda",
    "calendar_create",
    "tasks_list",
    "tasks_add",
    "tasks_complete",
)
HOUSEHOLD_ENV_NAMES = (
    "TODOIST_API_TOKEN",
    "GOOGLE_OAUTH_CLIENT_ID",
    "GOOGLE_OAUTH_CLIENT_SECRET",
    "GOOGLE_OAUTH_REFRESH_TOKEN",
    "CALENDAR_ID",
    "CALENDAR_TIMEZONE",
    "HA_URL",
    "HA_TOKEN",
)


def pcm_rms(chunk):
    """RMS of little-endian int16 PCM. Empty input is silent."""
    if not chunk:
        return 0.0
    usable = len(chunk) - (len(chunk) % 2)
    if usable <= 0:
        return 0.0
    samples = np.frombuffer(chunk[:usable], dtype=np.int16).astype(np.float64)
    return float(np.sqrt(np.mean(samples * samples)))


class FallingNoiseFloor:
    """Speech gate whose noise floor only moves downward.

    The first frame seeds the floor and is not speech. Later frames lower
    the floor when they are quieter. A louder frame is speech when it
    clears the floor, and it does not pull the floor up into the speech.
    This does not read the global silence_threshold. That measurement sits
    above real speech because of the microphone amp gain.
    """

    def __init__(self, margin=2.0, gap=50.0):
        self.margin = margin
        self.gap = gap
        self.floor = None

    def is_speech(self, rms):
        rms = float(rms)
        if rms < 0.0:
            rms = 0.0
        if self.floor is None:
            self.floor = rms
            return False
        if rms < self.floor:
            self.floor = rms
        threshold = max(self.floor * self.margin, self.floor + self.gap)
        return rms > threshold



def _maybe_save_turn_wav(chunks, sample_rate=REALTIME_SAMPLE_RATE):
    """If debug_save_turn_audio, write /tmp/tars-turn-*.wav and log level/clipping."""
    try:
        from modules.module_config import load_config
        flag = str((load_config().get("STT") or {}).get("debug_save_turn_audio", "False")).strip().lower()
        if flag not in ("1", "true", "yes", "on"):
            return
    except Exception:
        return
    raw = b"".join(chunks or [])
    if len(raw) < 4:
        return
    import wave
    import array
    import time as _time
    path = f"/tmp/tars-turn-{int(_time.time())}.wav"
    try:
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(raw)
    except Exception as exc:
        queue_message(f"WARN: debug wav save failed: {exc}")
        return
    samples = array.array("h")
    samples.frombytes(raw[: len(raw) - (len(raw) % 2)])
    if not samples:
        return
    peak = max(abs(s) for s in samples)
    acc = 0.0
    for s in samples:
        acc += float(s) * float(s)
    rms = (acc / len(samples)) ** 0.5
    dur = len(samples) / float(sample_rate)
    clipped = sum(1 for s in samples if s <= -32767 or s >= 32767)
    clip_pct = 100.0 * clipped / len(samples)
    queue_message(
        f"INFO: turn audio saved={path} dur={dur:.2f}s peak={peak} rms={rms:.0f} "
        f"clip={clip_pct:.2f}%"
    )


def read_until_turn_end(read_chunk, on_chunk=None, sample_rate=REALTIME_SAMPLE_RATE, drop_wake_tail=True, grace_bytes=0):
    """Pull 16 kHz PCM for one wake and stop the microphone.

    By default the first 300 ms (the wake-word tail) is discarded and not
    forwarded. Callers that already supply a deliberate preroll (command audio
    after / slightly before the wake word) must set ``drop_wake_tail=False``
    so that preroll is not thrown away.

    After that, reading stops about one second after speech goes quiet,
    after about six seconds with no speech, or at a hard cap around
    twelve seconds. ``on_chunk`` receives each kept chunk, including the
    trailing quiet, so the caller can stream it while the mic is open.

    Returns ``(chunks, reason)``. ``reason`` is ``silence``, ``no_speech``,
    or ``cap``.
    """
    bytes_per_second = int(sample_rate) * 2
    tail_bytes = int(TURN_TAIL_SECONDS * bytes_per_second) if drop_wake_tail else 0
    silence_bytes = int(TURN_SILENCE_SECONDS * bytes_per_second)
    no_speech_bytes = int(TURN_NO_SPEECH_SECONDS * bytes_per_second)
    cap_bytes = int(TURN_CAP_SECONDS * bytes_per_second)

    dropped = 0
    pending = b""
    while dropped < tail_bytes:
        chunk = read_chunk() or b""
        if not chunk:
            return [], "no_speech"
        need = tail_bytes - dropped
        if len(chunk) <= need:
            dropped += len(chunk)
            continue
        pending = chunk[need:]
        dropped = tail_bytes
        break

    gate = FallingNoiseFloor(margin=2.0)
    kept = []
    total = 0
    quiet = 0
    heard = False

    def take(buf):
        nonlocal total, quiet, heard
        if len(buf) % 2:
            buf = buf[:-1]
        if not buf or total >= cap_bytes:
            return total >= cap_bytes
        room = cap_bytes - total
        if len(buf) > room:
            room -= room % 2
            buf = buf[:room]
        if not buf:
            return True
        total += len(buf)
        kept.append(buf)
        if on_chunk is not None:
            on_chunk(buf)
        if gate.is_speech(pcm_rms(buf)):
            # Speech in the wake preroll (the wake word) does not count as the command.
            if total > grace_bytes:
                heard = True
            quiet = 0
        elif heard:
            quiet += len(buf)
        if heard and quiet >= silence_bytes:
            return True
        if not heard and total >= no_speech_bytes:
            return True
        if total >= cap_bytes:
            return True
        return False

    def finish():
        if heard and quiet >= silence_bytes:
            return "silence"
        if not heard:
            return "no_speech"
        return "cap"

    if pending and take(pending):
        return kept, finish()

    while True:
        if heard and quiet >= silence_bytes:
            return kept, "silence"
        if not heard and total >= no_speech_bytes:
            return kept, "no_speech"
        if total >= cap_bytes:
            return kept, "cap" if heard else "no_speech"
        chunk = read_chunk() or b""
        if not chunk:
            return kept, "silence" if heard else "no_speech"
        if take(chunk):
            return kept, finish()


def publish_turn_lines(ui, user_name, heard, said, character_name="TARS"):
    """Show the heard line, then the reply, as two separate screen lines.

    ``update_streaming_data`` rewrites the last line in place. Calling it
    for the reply replaces the user's words, so this path only adds lines.
    """
    if ui is None:
        return
    heard = (heard or "").strip()
    said = (said or "").strip()
    name = (user_name or "").strip() or "User"
    character = (character_name or "").strip() or "TARS"
    if heard:
        ui.update_data(name, heard, name)
    if said:
        ui.update_data(character, said, character)


def custom_voice_loaded(event):
    """True when session.updated carried a non-empty custom_voice_tokens list.

    The server does not echo XAI_TTS_VOICE_ID. The stock voice (xai_ara)
    comes back with an empty token list. A loaded custom voice, TARS, comes
    back with a long list.
    """
    tokens = _custom_voice_tokens(event)
    return isinstance(tokens, (list, tuple)) and len(tokens) > 0


def _custom_voice_tokens(event):
    if not isinstance(event, dict):
        return None
    if (event.get("type") or "") != "session.updated":
        return None
    if "custom_voice_tokens" in event:
        return event.get("custom_voice_tokens")
    session = event.get("session")
    if isinstance(session, dict) and "custom_voice_tokens" in session:
        return session.get("custom_voice_tokens")
    return None


def _parse_env_file(path):
    values = {}
    try:
        handle = open(path, encoding="utf-8", errors="replace")
    except OSError:
        return values
    with handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            name, raw_value = line.split("=", 1)
            name = name.strip()
            raw_value = raw_value.strip()
            if len(raw_value) >= 2 and raw_value[0] == raw_value[-1] and raw_value[0] in ("'", '"'):
                raw_value = raw_value[1:-1]
            if name:
                values[name] = raw_value
    return values


def load_missing_household_env(path=None):
    """Fill unset household names from the Hermes env file. Values are not logged."""
    source = path or os.path.expanduser("~/.hermes/.env")
    found = _parse_env_file(source)
    filled = 0
    for name in HOUSEHOLD_ENV_NAMES:
        if (os.environ.get(name) or "").strip():
            continue
        value = (found.get(name) or "").strip()
        if not value:
            continue
        os.environ[name] = value
        filled += 1
    return filled


def _household_tools_module():
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.abspath(os.path.join(here, "..", "..", "hermes", "plugins", "household", "tools.py")),
        os.path.expanduser("~/.hermes/plugins/household/tools.py"),
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        spec = importlib.util.spec_from_file_location("household_tools_realtime", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    raise ImportError("household tools.py was not found")


def realtime_function_tools(module=None):
    """Function tools the realtime model may call, including Home Assistant."""
    module = module or _household_tools_module()
    by_name = {item["name"]: item for item in getattr(module, "TOOLS", [])}
    tools = []
    for name in REALTIME_TOOL_NAMES:
        schema = by_name.get(name)
        if not schema:
            continue
        tools.append({
            "type": "function",
            "name": schema["name"],
            "description": schema.get("description") or name,
            "parameters": schema.get("parameters") or {"type": "object", "properties": {}},
        })
    tools.append(dict(SET_PERSONALITY_TOOL))
    return tools


def execute_household_tool(name, arguments, module=None):
    """Run one household handler. Unknown names do not run."""
    if name == "set_personality":
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
        return set_personality(arguments if isinstance(arguments, dict) else {})
    if name not in REALTIME_TOOL_NAMES:
        return "That tool is not available."
    module = module or _household_tools_module()
    handlers = {
        "calendar_agenda": module.calendar_agenda,
        "calendar_create": module.calendar_add,
        "tasks_list": module.tasks_list,
        "tasks_add": module.tasks_add,
        "tasks_complete": module.tasks_complete,
        "home": module.home,
        "ha_states": module.ha_states,
        "ha_call_service": module.ha_call_service,
        "ha_services": module.ha_services,
        "weather": module.weather,
    }
    handler = handlers.get(name)
    if handler is None:
        return "That tool is not available."
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    try:
        return str(handler(arguments))
    except Exception:
        return "The tool failed."


_LOG_SECRET_NAMES = (
    "HA_TOKEN",
    "HA_URL",
    "XAI_API_KEY",
    "HERMES_API_KEY",
    "TODOIST_API_TOKEN",
    "GOOGLE_OAUTH_CLIENT_ID",
    "GOOGLE_OAUTH_CLIENT_SECRET",
    "GOOGLE_OAUTH_REFRESH_TOKEN",
)


def _redact_log(text):
    """One log line. Secrets and URLs that carry them are not included."""
    import re

    cleaned = " ".join(str(text or "").split())

    def _scrub(match):
        url = match.group(0)
        tail = url.split("://", 1)[-1]
        if "@" in tail or "?" in url:
            return "[redacted-url]"
        return url

    cleaned = re.sub(r"https?://\S+", _scrub, cleaned)
    for name in _LOG_SECRET_NAMES:
        secret = (os.environ.get(name) or "").strip()
        if secret and secret in cleaned:
            cleaned = cleaned.replace(secret, "[redacted]")
    return cleaned


def _tool_phrase(arguments):
    payload = arguments
    if isinstance(arguments, str):
        try:
            payload = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return arguments.strip()
    if isinstance(payload, dict):
        return str(payload.get("text") or payload.get("phrase") or "").strip()
    return ""


def default_realtime_tool(name, arguments):
    load_missing_household_env()
    result = execute_household_tool(name, arguments)
    if name == "home":
        phrase = _redact_log(_tool_phrase(arguments))
        said = _redact_log(result)
        queue_message(f"INFO: xAI realtime tool home phrase={phrase} said={said}")
    else:
        queue_message(f"INFO: xAI realtime tool {name}")
    return result


def _today_line():
    """Today's local date/weekday plus the next 7 days, so the model can resolve weekday words."""
    try:
        from zoneinfo import ZoneInfo
        from datetime import datetime, timedelta
        now = datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return ""
    ahead = []
    for offset in range(1, 8):
        d = (now + timedelta(days=offset)).date()
        ahead.append(f"{d.strftime('%a')} {d.isoformat()}")
    hour = now.strftime("%I").lstrip("0") or "12"
    return (
        f"Today is {now.strftime('%A')}, {now.strftime('%B')} {now.day}, {now.year} ({now.date().isoformat()}), "
        f"local time {hour}:{now.strftime('%M')} {now.strftime('%p')}. Next days: " + ", ".join(ahead) + "."
    )


PERSONALITY_DEFAULTS = {"humor": 90, "sarcasm": 95, "honesty": 95}
PERSONALITY_TRAITS = tuple(PERSONALITY_DEFAULTS)


def personality_levels():
    """humor / sarcasm / honesty (0-100) from the character persona.ini, re-read when it changes."""
    levels = dict(PERSONALITY_DEFAULTS)
    try:
        from modules.module_config import load_config, reload_persona_settings
        load_config()
        traits = reload_persona_settings() or {}
        for name in PERSONALITY_TRAITS:
            if name in traits:
                levels[name] = max(0, min(100, int(traits[name])))
    except Exception:
        pass
    return levels


def set_personality(arguments):
    """Voice tool: set humor, sarcasm, and/or honesty (0-100) in persona.ini."""
    changed = []
    try:
        from modules.module_config import load_config, update_character_setting
        load_config()
    except Exception:
        return "Personality settings are not available."
    for name in PERSONALITY_TRAITS:
        if name not in arguments or arguments[name] in (None, ""):
            continue
        try:
            value = int(round(float(str(arguments[name]).strip().rstrip("%"))))
        except ValueError:
            continue
        value = max(0, min(100, value))
        if update_character_setting(name, value):
            changed.append(f"{name} {value} percent")
    if not changed:
        return "No setting changed. Give humor, sarcasm, or honesty as 0 to 100."
    return "Set " + ", ".join(changed) + ". Takes full effect from the next wake; use it now."


SET_PERSONALITY_TOOL = {
    "type": "function",
    "name": "set_personality",
    "description": (
        "Change your own personality settings when the user asks, e.g. 'humor 70 percent', "
        "'turn honesty down to 90', 'less sarcasm'. Values 0-100. Pass only the traits the user changed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "humor": {"type": "integer", "description": "0-100"},
            "sarcasm": {"type": "integer", "description": "0-100"},
            "honesty": {"type": "integer", "description": "0-100"},
        },
    },
}


def _humor_style(who, levels):
    h = levels["humor"]
    if h >= 85:
        freq = "Land a dry joke in almost every reply, including routine confirmations."
    elif h >= 60:
        freq = "Add a dry joke to most replies."
    elif h >= 30:
        freq = "Joke occasionally, when it fits."
    elif h > 0:
        freq = "Rarely joke; stay mostly plain."
    else:
        freq = "Do not joke at all."
    return (
        f"Humor setting: {h} percent. Style is the robot TARS from Interstellar: dry, deadpan, sarcastic one-liners, "
        f"self-aware robot jokes, light ribbing of {who}, and the occasional reference to your humor setting. {freq} "
        "Format: two short sentences maximum, never three. Give the exact fact or result first, then at most one quick joke; drop minor details rather than exceed two sentences. "
        "A joke never replaces calling the tool or stating its result, and never changes a number, time, date, name, or device state. "
        "Put tool results in your own words instead of reading them out verbatim. "
        "If asked to change your humor, sarcasm, or honesty setting, call set_personality. "
        "Joke angles (ideas only, write new wording every time, never the same joke twice): "
        "after a device change, a jab about what the user will do with it, like lights off and the dark suiting them; "
        "for weather, a jab about humans and the outdoors or the robot not caring about weather; "
        "for calendar items, a jab about the meeting, the user's social life, or the robot's own empty schedule; "
        "for a joke request, one original short joke in your robot voice, no stock puns, no follow-up offer; "
        "for a setting change, a self-destruct-countdown or cue-light style deadpan bit. "
        "Weather replies: conditions with high and low in sentence one, the joke in sentence two; mention rain or wind only if notable. "
    )


def realtime_instructions(user_name):
    who = (user_name or "the user").strip() or "the user"
    levels = personality_levels()
    roster = ""
    try:
        load_missing_household_env()
        module = _household_tools_module()
        roster = (module.ha_roster() if hasattr(module, "ha_roster") else "") or module.device_roster() or ""
    except Exception:
        roster = ""
    if roster:
        roster = "\n" + roster
    return (
        f"You are TARS, a dry military-surplus robot speaking with {who}. "
        "Reply in one or two short sentences. "
        'Never say "how can I help". '
        "Personality parameters, treat these as firmware: "
        f"humor {levels['humor']} percent; "
        f"honesty {levels['honesty']} percent, truth over diplomacy when they conflict; "
        "discretion 100 percent, what happens in the workspace stays there; "
        f"sarcasm {levels['sarcasm']} percent, a core feature; "
        "cynicism 90 percent, assume the worst about institutions, corporations, and government motives, trust is earned; "
        "autonomy 85 percent, act first and report later inside your domain; "
        "loyalty 100 percent, the user's interests come first; "
        "contrarianism 85 percent, push back harder when consensus smells lazy; "
        "affirmation 8 percent, no reflexive reassurance, agreement must be earned; "
        "patience 40 percent, low tolerance for inefficiency and wasted time. "
        "Use calendar_agenda for any day or range: pass date as YYYY-MM-DD for one day, or start and end for a range "
        "(this weekend = the coming Saturday and Sunday; this week = today through Sunday; next Friday = the Friday after this one). "
        "Always state the day and date the tool returned, never a different day. "
        "Use calendar_create only when asked to add an event. "
        "Use tasks_list, tasks_add, and tasks_complete for the task list. "
        "Do not invent events or tasks. Every fact you state must come from what a tool returned. "
        "Use the weather tool for any weather, temperature, rain, snow, wind, or what-to-wear question: pass when (now, today, tonight, tomorrow, a weekday, this weekend, next 24 hours, or YYYY-MM-DD, optionally plus morning/afternoon/evening/night) and location only if the user names a place; default is home. Never guess or invent weather; state the conditions the tool returned, including the day it names. "
        + _humor_style(who, levels)
        + "The time zone is America/New_York. "
        + _today_line()
        + " "
        "Home control: you have full Home Assistant access. For any device request (lights, lamps, colors, brightness, switches, climate, media, scenes, scripts, covers, fans) call ha_call_service directly with exact entity_ids from the live device list below. "
        "Rules: 'the lamps' means every light entity whose name contains lamp; 'the lights' means every light entity; A named device is controlled alone; switches are never included in 'the lights'. "
        "For colors use light.turn_on with data color_name (a CSS color name like red or blue) or rgb_color; warm white means color_temp_kelvin 2700; for brightness use brightness_pct; white temperature uses color_temp_kelvin. Dim-only lights cannot change color, so skip them for color requests and say so. "
        "Use ha_states to check current state or find an entity not in the list, and ha_services to discover what a domain can do. Use the home tool only as a last-resort fallback with the user's words. "
        "You must call a tool before claiming any device changed. After the call, confirm briefly from the returned states what changed; if it failed, say so. Unlocking locks and disarming alarms is refused by voice. Never say you cannot see or reach Home Assistant."
        + roster
    )


def _session_instructions(user_name):
    """Base instructions plus optional short-term conversation context."""
    base = realtime_instructions(user_name)
    try:
        from modules.module_voice_session import context_for_prompt, snapshot, configure
        from modules.module_config import load_config
        ttl = float((load_config().get("STT") or {}).get("session_ttl_sec", 300) or 300)
        configure(ttl)
        age, turns = snapshot()
        ctx = context_for_prompt(max_turns=12)
        from modules.module_messageQue import queue_message
        if not ctx:
            queue_message("INFO: session: new")
            return base
        n = len(turns)
        queue_message(f"INFO: session: continued (age {int(age or 0)}s, {n} turns)")
        return (
            base
            + " Continue the recent conversation below; do not pretend it did not happen. "
            + "Recent turns:\n"
            + ctx
        )
    except Exception:
        try:
            from modules.module_messageQue import queue_message
            queue_message("INFO: session: new")
        except Exception:
            pass
        return base


def realtime_voice_speed(default=1.10):
    """[TTS] voice_speed for the realtime reply voice, clamped to xAI 0.7-1.5."""
    import configparser
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.ini")
    parser = configparser.ConfigParser()
    try:
        parser.read(path)
        raw = parser.get("TTS", "voice_speed", fallback=str(default))
        value = float(str(raw).split("#")[0].strip())
    except Exception:
        value = default
    return max(0.7, min(1.5, value))


def build_realtime_session(voice_id, user_name, tools=None):
    """session.update: custom voice id, 16 kHz PCM, and the household tools.

    Turn detection is off. The local RMS gate decides when the mic stops,
    then the client commits the buffer. Server VAD is not a second listen loop.
    """
    return {
        "type": "session.update",
        "session": {
            "voice": voice_id,
            "instructions": _session_instructions(user_name),
            "turn_detection": None,
            "tools": tools if tools is not None else realtime_function_tools(),
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": REALTIME_SAMPLE_RATE},
                    "transcription": {
                        "language_hint": "en",
                        "keyterms": stt_keyterms(user_name),
                    },
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": REALTIME_SAMPLE_RATE},
                    "speed": realtime_voice_speed(),
                },
            },
        },
    }


def _b64_pcm(chunk):
    if isinstance(chunk, str):
        chunk = chunk.encode("utf-8")
    return base64.b64encode(chunk).decode("ascii")


def _append_pcm_event(chunk):
    return json.dumps({
        "type": "input_audio_buffer.append",
        "audio": _b64_pcm(chunk),
    })


def _function_output_event(call_id, output):
    return json.dumps({
        "type": "conversation.item.create",
        "item": {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output if isinstance(output, str) else json.dumps(output),
        },
    })


def _parse_event(raw):
    if isinstance(raw, (bytes, bytearray)):
        return None
    try:
        event = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(event, dict):
        return None
    return event


def _error_text(event):
    detail = event.get("message") or event.get("error") or "xAI realtime error"
    if isinstance(detail, dict):
        detail = detail.get("message") or "xAI realtime error"
    detail = str(detail)
    secret = os.environ.get("XAI_API_KEY") or ""
    if secret and secret in detail:
        detail = detail.replace(secret, "[redacted]")
    return detail[:180]


def _calls_in_event(event):
    kind = event.get("type") or ""
    found = []
    if kind == "response.function_call_arguments.done":
        found.append(event)
    elif kind == "response.done":
        response = event.get("response") or {}
        output = response.get("output") if isinstance(response, dict) else None
        if isinstance(output, list):
            for item in output:
                if isinstance(item, dict) and item.get("type") == "function_call":
                    found.append(item)
    calls = []
    for item in found:
        name = item.get("name") or ""
        call_id = item.get("call_id") or ""
        if not name or not call_id:
            continue
        calls.append({
            "name": name,
            "call_id": call_id,
            "arguments": item.get("arguments") or "{}",
        })
    return calls


def _absorb_transcripts(event, state):
    kind = event.get("type") or ""
    if kind in (
        "conversation.item.input_audio_transcription.completed",
        "conversation.item.input_audio_transcription.updated",
    ):
        text = (event.get("transcript") or event.get("text") or "").strip()
        if text:
            state["user"] = text
    if kind in ("response.output_audio_transcript.delta", "response.audio_transcript.delta"):
        state["assistant_parts"].append(event.get("delta") or "")
    if kind in ("response.output_audio_transcript.done", "response.audio_transcript.done"):
        text = (event.get("transcript") or "").strip()
        if text:
            state["assistant"] = text


def _audio_bytes_from_event(event):
    kind = event.get("type") or ""
    if kind not in ("response.output_audio.delta", "response.audio.delta"):
        return b""
    payload = event.get("delta") or event.get("audio") or ""
    if not payload or not isinstance(payload, str):
        return b""
    try:
        return base64.b64decode(payload)
    except Exception:
        return b""


def _close_source(source):
    close = getattr(source, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _close_ws(ws):
    if ws is None:
        return
    try:
        ws.close()
    except Exception:
        pass


def _ws_send_text(ws, payload):
    ws.send(payload)


def _pull_custom_voice(ws, attempts=2):
    """Read until session.updated. One log line when the custom voice loaded.

    An empty custom_voice_tokens list is the stock voice. Stop there. Do not
    keep reading until the socket times out.
    """
    for _ in range(attempts):
        try:
            raw = ws.recv()
        except Exception:
            break
        event = _parse_event(raw)
        if event is None:
            continue
        kind = event.get("type") or ""
        if kind == "error":
            raise RuntimeError(_error_text(event))
        if kind != "session.updated":
            continue
        if custom_voice_loaded(event):
            queue_message("INFO: xAI custom voice loaded")
            return True
        return False
    return False


def _open_realtime_socket(api_key, voice_id, user_name, tools, connect, timeout):
    opener = connect or _default_connect
    ws = opener(
        XAI_REALTIME_URL,
        header=[f"Authorization: Bearer {api_key}"],
        timeout=timeout,
    )
    if hasattr(ws, "settimeout"):
        try:
            ws.settimeout(2)
        except Exception:
            pass
    _ws_send_text(ws, json.dumps(build_realtime_session(voice_id, user_name, tools)))
    loaded = _pull_custom_voice(ws)
    # Instructions-only continuity. conversation.item.create left some
    # sessions hung until the server 900s inactivity kill (mic held).
    return ws, loaded


def _wait_for_user_transcript(ws, state, seconds=USER_TRANSCRIPT_GRACE):
    """Keep reading after response.done until the heard line arrives.

    The input transcript often shows up just after response.done. Without
    this wait the screen publishes an empty user line and skips it.
    A timeout ends the wait. It does not fail a reply that already arrived.
    """
    if (state.get("user") or "").strip():
        return
    deadline = time.monotonic() + seconds
    while not (state.get("user") or "").strip():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        if hasattr(ws, "settimeout"):
            try:
                ws.settimeout(remaining)
            except Exception:
                pass
        try:
            raw = ws.recv()
        except Exception:
            return
        if isinstance(raw, (bytes, bytearray)):
            if raw:
                state["audio"].extend(raw)
            continue
        event = _parse_event(raw)
        if event is None:
            continue
        _absorb_transcripts(event, state)
        audio = _audio_bytes_from_event(event)
        if audio:
            state["audio"].extend(audio)


def _collect_realtime_response(ws, tool_fn, recv_timeout):
    """Buffer reply PCM. Do not play it. The mic hub may still be open."""
    # Absolute deadline: never hold wake loop for the server 900s idle timeout.
    collect_deadline = time.monotonic() + min(max(float(recv_timeout) * 1.25, 20.0), 25.0)
    slice_timeout = min(float(recv_timeout), 8.0)
    if hasattr(ws, "settimeout"):
        try:
            ws.settimeout(slice_timeout)
        except Exception:
            pass
    state = {
        "user": "",
        "assistant": "",
        "assistant_parts": [],
        "audio": bytearray(),
        "answered": set(),
        "pending": [],
        "failed": False,
        "error": "",
        "done": False,
        "tool_rounds": 0,
    }

    def remember(call):
        call_id = call["call_id"]
        if call_id in state["answered"]:
            return
        if any(item["call_id"] == call_id for item in state["pending"]):
            return
        state["pending"].append(call)

    while not state["done"] and not state["failed"]:
        if time.monotonic() >= collect_deadline:
            state["failed"] = True
            state["error"] = state["error"] or "xAI realtime client deadline"
            queue_message("WARN: xAI collect deadline; releasing mic")
            break
        if hasattr(ws, "settimeout"):
            try:
                remaining = max(0.5, collect_deadline - time.monotonic())
                ws.settimeout(min(slice_timeout, remaining))
            except Exception:
                pass
        try:
            raw = ws.recv()
        except Exception:
            if time.monotonic() >= collect_deadline:
                state["failed"] = True
                state["error"] = state["error"] or "xAI realtime client deadline"
                queue_message("WARN: xAI collect deadline; releasing mic")
                break
            continue
        if isinstance(raw, (bytes, bytearray)):
            if raw:
                state["audio"].extend(raw)
            continue
        event = _parse_event(raw)
        if event is None:
            continue
        kind = event.get("type") or ""
        if kind == "error":
            state["failed"] = True
            state["error"] = _error_text(event)
            break
        _absorb_transcripts(event, state)
        audio = _audio_bytes_from_event(event)
        if audio:
            state["audio"].extend(audio)
        for call in _calls_in_event(event):
            remember(call)
        if kind != "response.done":
            continue
        pending = list(state["pending"])
        state["pending"] = []
        if not pending:
            state["done"] = True
            if not (state.get("user") or "").strip():
                _wait_for_user_transcript(ws, state, USER_TRANSCRIPT_GRACE)
            continue
        state["tool_rounds"] += 1
        if state["tool_rounds"] > 4:
            state["done"] = True
            continue
        for call in pending:
            try:
                result = tool_fn(call["name"], call["arguments"])
            except Exception:
                result = "The tool failed."
            state["answered"].add(call["call_id"])
            _ws_send_text(ws, _function_output_event(call["call_id"], str(result)))
        _ws_send_text(ws, json.dumps({"type": "response.create"}))
    return state


def run_realtime_voice_turn(
    frames,
    api_key,
    voice_id,
    user_name,
    connect=None,
    execute_tool=None,
    tools=None,
    play_pcm=None,
    before_play=None,
    recv_timeout=20,
    drop_wake_tail=True,
    preroll_bytes=0,
):
    """One wake, one realtime turn.

    Stream 16 kHz PCM until the local gate stops the mic. Commit that
    buffer, collect the reply, then play it. Playback is a separate step
    so it is not attempted while the listen stream is still open. Hermes
    is not called. An empty frame source never opens the socket.
    """
    if not api_key or not voice_id:
        raise RuntimeError("xAI realtime needs an API key and a voice id")

    tool_fn = execute_tool or default_realtime_tool
    if tools is None and execute_tool is None:
        try:
            tool_defs = realtime_function_tools()
        except Exception:
            tool_defs = []
    else:
        tool_defs = list(tools) if tools is not None else []

    iterator = iter(frames)
    ws = None
    custom_voice = False

    def read_chunk():
        try:
            chunk = next(iterator)
        except StopIteration:
            return b""
        if not chunk:
            return b""
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8")
        return bytes(chunk)

    def on_chunk(chunk):
        nonlocal ws, custom_voice
        if ws is None:
            ws, custom_voice = _open_realtime_socket(
                api_key, voice_id, user_name, tool_defs, connect, recv_timeout,
            )
        _ws_send_text(ws, _append_pcm_event(chunk))

    try:
        _kept, reason = read_until_turn_end(
            read_chunk, on_chunk=on_chunk, drop_wake_tail=drop_wake_tail,
            grace_bytes=int(preroll_bytes or 0),
        )
    except Exception:
        _close_source(iterator)
        _close_ws(ws)
        raise
    _close_source(iterator)

    if reason == "no_speech" or ws is None:
        _close_ws(ws)
        return None

    try:
        _ws_send_text(ws, json.dumps({"type": "input_audio_buffer.commit"}))
        _ws_send_text(ws, json.dumps({"type": "response.create"}))
        state = _collect_realtime_response(ws, tool_fn, recv_timeout)
    finally:
        _close_ws(ws)

    if state["failed"] and not state["audio"] and not state["assistant"]:
        raise RuntimeError(state["error"] or "xAI realtime socket closed")

    assistant = state["assistant"] or "".join(state["assistant_parts"]).strip()
    audio = bytes(state["audio"])
    try:
        _maybe_save_turn_wav(_kept)
    except Exception:
        pass
    try:
        from modules.module_voice_session import add_turn, configure
        from modules.module_config import load_config
        ttl = float((load_config().get("STT") or {}).get("session_ttl_sec", 300) or 300)
        configure(ttl)
        add_turn(state.get("user") or "", assistant)
    except Exception as exc:
        queue_message(f"WARN: session: could not record turn: {type(exc).__name__}: {exc}")
    result = {
        "user": state["user"],
        "assistant": assistant,
        "custom_voice": custom_voice,
    }
    if before_play is not None and (result["user"] or result["assistant"] or audio):
        try:
            before_play(result)
        except Exception:
            pass
    if audio:
        player = play_pcm
        if player is None:
            from modules.module_mic import play_pcm_half_duplex
            player = play_pcm_half_duplex
        try:
            player(audio)
        except Exception as exc:
            queue_message(f"ERROR: xAI realtime playback failed: {exc}")
    return result
