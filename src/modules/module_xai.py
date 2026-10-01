"""
module_xai.py

xAI speech adapters and the Hermes loopback client for the voice path.

Speech-to-text streams 16 kHz PCM to wss://api.x.ai/v1/stt and ends the
utterance on Smart Turn (speech_final). Text-to-speech posts to
https://api.x.ai/v1/tts using XAI_TTS_VOICE_ID. Answers are a POST to
Hermes /v1/responses on loopback. This module does not call a desktop
agent and does not select Hermes's model.
"""

import hashlib
import io
import json
import os
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
