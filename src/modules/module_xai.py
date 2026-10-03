"""
module_xai.py

xAI speech adapters.

Speech-to-text streams 16 kHz PCM to wss://api.x.ai/v1/stt. Text-to-speech
posts to https://api.x.ai/v1/tts using XAI_TTS_VOICE_ID. The realtime voice
path streams the same PCM to wss://api.x.ai/v1/realtime, speaks with that
voice id, and runs household tools on the socket. The loopback client
remains for the older text path and is not used by the realtime turn.
"""

import hashlib
import io
import json
import os
import queue
import threading
from urllib.parse import urlencode

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
    terms = ["TARS"]
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


import base64

XAI_REALTIME_URL = "wss://api.x.ai/v1/realtime?model=grok-voice-latest"
REALTIME_BACKEND = "xai-realtime"
REALTIME_SAMPLE_RATE = 16000
HOUSEHOLD_ENV_NAMES = (
    "TODOIST_API_TOKEN",
    "GOOGLE_OAUTH_CLIENT_ID",
    "GOOGLE_OAUTH_CLIENT_SECRET",
    "GOOGLE_OAUTH_REFRESH_TOKEN",
    "CALENDAR_ID",
    "CALENDAR_TIMEZONE",
)
# Home control stays off this path. The house bridge is not configured.
REALTIME_TOOL_NAMES = (
    "calendar_agenda",
    "calendar_create",
    "tasks_list",
    "tasks_add",
    "tasks_complete",
)


def _parse_env_file(path):
    values = {}
    try:
        lines = open(path, encoding="utf-8", errors="replace")
    except OSError:
        return values
    with lines as handle:
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
    """Function tools the realtime model may call. Home Assistant is omitted."""
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
    return tools


def execute_household_tool(name, arguments, module=None):
    """Run one household handler. Unknown names and Home Assistant do not run."""
    if name not in REALTIME_TOOL_NAMES:
        return "That tool is not available."
    module = module or _household_tools_module()
    handlers = {
        "calendar_agenda": module.calendar_agenda,
        "calendar_create": module.calendar_add,
        "tasks_list": module.tasks_list,
        "tasks_add": module.tasks_add,
        "tasks_complete": module.tasks_complete,
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


def default_realtime_tool(name, arguments):
    load_missing_household_env()
    queue_message(f"INFO: xAI realtime tool {name}")
    return execute_household_tool(name, arguments)


def realtime_instructions(user_name):
    who = (user_name or "the user").strip()
    return (
        f"You are a direct household robot speaking with {who}. "
        "Keep every reply short enough to say aloud. "
        "Use calendar_agenda for today or tomorrow. "
        "Use calendar_create only when they ask to add an event. "
        "Use tasks_list, tasks_add, and tasks_complete for the task list. "
        "Do not invent events or tasks. Speak only what a tool returned. "
        "The time zone is America/New_York. "
        "You cannot control lights or other home devices."
    )


def build_realtime_session(voice_id, user_name, tools=None):
    """Session update: custom voice and client function tools together."""
    return {
        "type": "session.update",
        "session": {
            "voice": voice_id,
            "instructions": realtime_instructions(user_name),
            "turn_detection": {"type": "server_vad"},
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


def run_realtime_voice_turn(
    frames,
    api_key,
    voice_id,
    user_name,
    on_pcm,
    connect=None,
    execute_tool=None,
    tools=None,
    recv_timeout=45,
):
    """Stream one utterance to xAI realtime and play returned PCM.

    The socket stays closed when the caller has no audio. Tool results are
    sent back on the same socket before the model continues. Hermes is not
    called.
    """
    iterator = iter(frames)
    try:
        first = next(iterator)
    except StopIteration:
        return None
    if not api_key or not voice_id:
        raise RuntimeError("xAI realtime needs an API key and a voice id")

    tool_fn = execute_tool or default_realtime_tool
    if tools is None and execute_tool is None:
        tool_defs = realtime_function_tools()
    else:
        tool_defs = tools if tools is not None else []

    opener = connect or _default_connect
    ws = opener(
        XAI_REALTIME_URL,
        header=[f"Authorization: Bearer {api_key}"],
        timeout=10,
    )
    if hasattr(ws, "settimeout"):
        try:
            ws.settimeout(0.5)
        except Exception:
            pass

    state = {
        "user": "",
        "assistant": "",
        "assistant_parts": [],
        "answered": set(),
        "pending": [],
        "saw_audio": False,
        "speech_stopped": False,
        "failed": False,
        "error": "",
        "done": False,
        "tool_rounds": 0,
    }
    events = queue.Queue()
    stop_reader = threading.Event()

    def reader():
        try:
            from websocket import WebSocketTimeoutException
        except ImportError:
            WebSocketTimeoutException = ()
        while not stop_reader.is_set():
            try:
                raw = ws.recv()
            except WebSocketTimeoutException:
                continue
            except Exception:
                if not stop_reader.is_set():
                    events.put(None)
                return
            events.put(raw)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    started = __import__("time").monotonic()

    def past_deadline():
        return __import__("time").monotonic() - started > recv_timeout

    def drain(block_s=0.05):
        batch = []
        try:
            while True:
                batch.append(events.get_nowait())
        except queue.Empty:
            pass
        if batch:
            return batch
        try:
            batch.append(events.get(timeout=block_s))
        except queue.Empty:
            pass
        return batch

    def remember_call(call):
        call_id = call["call_id"]
        if call_id in state["answered"]:
            return
        if any(item["call_id"] == call_id for item in state["pending"]):
            return
        state["pending"].append(call)

    def handle(raw):
        if raw is None:
            state["failed"] = True
            return
        if isinstance(raw, (bytes, bytearray)):
            if raw and on_pcm:
                state["saw_audio"] = True
                on_pcm(bytes(raw))
            return
        try:
            event = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(event, dict):
            return
        kind = event.get("type") or ""
        if kind == "error":
            state["failed"] = True
            detail = event.get("message") or event.get("error") or "xAI realtime error"
            if isinstance(detail, dict):
                detail = detail.get("message") or "xAI realtime error"
            detail = str(detail)
            secret = os.environ.get("XAI_API_KEY") or ""
            if secret and secret in detail:
                detail = detail.replace(secret, "[redacted]")
            state["error"] = detail[:180]
            return
        if kind == "input_audio_buffer.speech_stopped":
            state["speech_stopped"] = True
        _absorb_transcripts(event, state)
        audio = _audio_bytes_from_event(event)
        if audio and on_pcm:
            state["saw_audio"] = True
            on_pcm(audio)
        for call in _calls_in_event(event):
            remember_call(call)
        if kind == "response.done":
            pending = list(state["pending"])
            state["pending"] = []
            if pending:
                state["tool_rounds"] += 1
                if state["tool_rounds"] > 4:
                    state["done"] = True
                    return
                for call in pending:
                    try:
                        result = tool_fn(call["name"], call["arguments"])
                    except Exception:
                        result = "The tool failed."
                    state["answered"].add(call["call_id"])
                    ws.send(_function_output_event(call["call_id"], str(result)))
                ws.send(json.dumps({"type": "response.create"}))
            else:
                state["done"] = True

    try:
        ws.send(json.dumps(build_realtime_session(voice_id, user_name, tool_defs)))
        ws.send(_append_pcm_event(first))
        sending = True
        while not state["done"] and not state["failed"] and not past_deadline():
            if sending and not state["speech_stopped"]:
                try:
                    chunk = next(iterator)
                except StopIteration:
                    sending = False
                    if not state["speech_stopped"]:
                        try:
                            ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                        except Exception:
                            state["failed"] = True
                            break
                else:
                    if chunk:
                        try:
                            ws.send(_append_pcm_event(chunk))
                        except Exception:
                            state["failed"] = True
                            break
            for raw in drain(0.02 if sending else 0.05):
                handle(raw)
                if state["done"] or state["failed"]:
                    break
            if state["speech_stopped"]:
                sending = False
        if past_deadline() and not state["done"]:
            state["failed"] = True
            state["error"] = state["error"] or "xAI realtime timed out"
    finally:
        stop_reader.set()
        try:
            ws.close()
        except Exception:
            pass

    if state["failed"] and not state["done"]:
        raise RuntimeError(state["error"] or "xAI realtime socket closed")
    if not state["done"] and not state["saw_audio"] and not state["assistant"]:
        raise RuntimeError(state["error"] or "xAI realtime returned no audio")
    assistant = state["assistant"] or "".join(state["assistant_parts"]).strip()
    return {"user": state["user"], "assistant": assistant}
