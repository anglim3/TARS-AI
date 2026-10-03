"""Voice-path checks that do not need API keys or a running Hermes."""

import base64
import json
import os
import queue
import sys
import tempfile
import threading
import unittest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import modules.module_xai as xai


class FakeSocket:
    def __init__(self, final_text):
        self.final_text = final_text
        self.sent = []
        self.closed = False
        self._ready_sent = False
        self._lock = threading.Condition()

    def recv(self):
        with self._lock:
            if not self._ready_sent:
                self._ready_sent = True
                return json.dumps({"type": "transcript.created"})
            while not any(isinstance(item, (bytes, bytearray)) for item in self.sent):
                if self.closed:
                    raise ConnectionError("closed")
                notified = self._lock.wait(timeout=2)
                if not notified and not any(isinstance(item, (bytes, bytearray)) for item in self.sent):
                    raise TimeoutError("no audio")
            return json.dumps({
                "type": "transcript.partial",
                "text": self.final_text,
                "is_final": True,
                "speech_final": True,
            })

    def send(self, data, opcode=None):
        with self._lock:
            self.sent.append(data)
            self._lock.notify_all()

    def send_binary(self, data):
        self.send(data)

    def close(self):
        with self._lock:
            self.closed = True
            self._lock.notify_all()


class VoicePathTests(unittest.TestCase):
    def test_dry_import(self):
        self.assertTrue(callable(xai.text_to_speech_with_pipelining_xai))
        self.assertEqual(xai.XAI_STT_URL, "wss://api.x.ai/v1/stt")
        self.assertEqual(xai.XAI_TTS_URL, "https://api.x.ai/v1/tts")

    def test_stt_url_biases_tars_and_user_and_enables_smart_turn(self):
        url = xai.build_stt_ws_url("Jack")
        self.assertTrue(url.startswith("wss://api.x.ai/v1/stt?"))
        self.assertIn("sample_rate=16000", url)
        self.assertIn("encoding=pcm", url)
        self.assertIn("smart_turn=0.5", url)
        self.assertIn("keyterm=TARS", url)
        self.assertIn("keyterm=Jack", url)
        self.assertNotIn("18789", url)

    def test_partial_transcript_is_not_an_utterance(self):
        kind, text = xai.utterance_from_stt_message(json.dumps({
            "type": "transcript.partial",
            "text": "turn off the",
            "is_final": False,
            "speech_final": False,
        }))
        self.assertEqual(kind, "partial")
        self.assertEqual(text, "turn off the")

        kind, text = xai.utterance_from_stt_message(json.dumps({
            "type": "transcript.partial",
            "text": "turn off the lights",
            "is_final": True,
            "speech_final": True,
        }))
        self.assertEqual(kind, "final")
        self.assertEqual(text, "turn off the lights")

    def test_stream_stops_on_speech_final_and_closes(self):
        socket = FakeSocket("turn off the lights")

        def connect(url, header, timeout):
            self.assertIn("Authorization: Bearer test-key", header)
            return socket

        text = xai.stream_pcm_until_speech_final(
            [b"\x00\x01" * 160, b"\x00\x02" * 160],
            "test-key",
            "Jack",
            connect=connect,
        )
        self.assertEqual(text, "turn off the lights")
        self.assertTrue(socket.closed)
        self.assertTrue(any(isinstance(item, (bytes, bytearray)) for item in socket.sent))

    def test_silence_does_not_open_the_socket(self):
        def connect(url, header, timeout):
            raise AssertionError("socket opened with no audio")

        self.assertIsNone(xai.stream_pcm_until_speech_final(
            iter(()),
            "test-key",
            "Jack",
            connect=connect,
        ))

    def test_hermes_body_has_no_json_schema(self):
        body = xai.hermes_request_body("turn off the kitchen lights")
        encoded = json.dumps(body)
        self.assertEqual(body["conversation"], "tars-voice")
        self.assertEqual(body["input"], "turn off the kitchen lights")
        self.assertNotIn("response_format", body)
        self.assertNotIn("function_calls", encoded)
        self.assertNotIn("json_object", encoded)
        self.assertEqual(
            xai.hermes_responses_url("http://127.0.0.1:8642/v1"),
            "http://127.0.0.1:8642/v1/responses",
        )
        self.assertEqual(
            xai.hermes_health_url("http://127.0.0.1:8642/v1"),
            "http://127.0.0.1:8642/health",
        )

    def test_assistant_sentence_drops_tools_and_reasoning(self):
        payload = {
            "output": [
                {"type": "reasoning", "content": "checking the house"},
                {
                    "type": "function_call",
                    "name": "home",
                    "arguments": "{\"text\": \"turn off the kitchen lights\"}",
                    "status": "completed",
                },
                {
                    "type": "function_call_output",
                    "output": "Turned off the kitchen lights.",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "reasoning_text", "text": "tool finished"},
                        {"type": "output_text", "text": "Kitchen lights are off."},
                    ],
                },
            ]
        }
        self.assertEqual(xai.assistant_text_from_response(payload), "Kitchen lights are off.")



class ScriptedRealtimeSocket:
    """Serve one tool-calling turn when the client appends audio."""

    def __init__(self):
        self.sent = []
        self.closed = False
        self._inbox = queue.Queue()
        self._appends = 0
        self._continued = False
        self._lock = threading.Lock()

    def settimeout(self, _timeout):
        return None

    def recv(self):
        item = self._inbox.get(timeout=3)
        if item is None:
            raise ConnectionError("closed")
        return item

    def send(self, data, opcode=None):
        with self._lock:
            self.sent.append(data)
            if not isinstance(data, str):
                return
            event = json.loads(data)
        kind = event.get("type")
        if kind == "session.update":
            self._inbox.put(json.dumps({"type": "session.updated"}))
        elif kind == "input_audio_buffer.append":
            with self._lock:
                self._appends += 1
                appends = self._appends
            if appends == 2:
                pcm = base64.b64encode(b"\x01\x00" * 8).decode("ascii")
                self._inbox.put(json.dumps({"type": "input_audio_buffer.speech_stopped"}))
                self._inbox.put(json.dumps({
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "what is on today",
                }))
                self._inbox.put(json.dumps({
                    "type": "response.output_audio.delta",
                    "delta": pcm,
                }))
                self._inbox.put(json.dumps({
                    "type": "response.function_call_arguments.done",
                    "name": "calendar_agenda",
                    "call_id": "call_1",
                    "arguments": json.dumps({"when": "today"}),
                }))
                self._inbox.put(json.dumps({"type": "response.done"}))
        elif kind == "conversation.item.create":
            if not self._continued:
                self._continued = True
                pcm = base64.b64encode(b"\x02\x00" * 4).decode("ascii")
                self._inbox.put(json.dumps({
                    "type": "response.output_audio.delta",
                    "delta": pcm,
                }))
                self._inbox.put(json.dumps({
                    "type": "response.output_audio_transcript.done",
                    "transcript": "Nothing on the calendar.",
                }))
                self._inbox.put(json.dumps({"type": "response.done"}))

    def close(self):
        self.closed = True
        self._inbox.put(None)


class RealtimeVoiceTests(unittest.TestCase):
    def test_realtime_tools_skip_home_and_keep_calendar_and_tasks(self):
        tools = xai.realtime_function_tools()
        names = [item["name"] for item in tools]
        self.assertIn("calendar_agenda", names)
        self.assertIn("tasks_list", names)
        self.assertIn("tasks_add", names)
        self.assertIn("tasks_complete", names)
        self.assertIn("calendar_create", names)
        self.assertNotIn("home", names)
        self.assertTrue(all(item["type"] == "function" for item in tools))

    def test_missing_household_env_names_are_filled_without_logging_values(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, ".env")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("CALENDAR_ID=cal-test\n")
                handle.write("CALENDAR_TIMEZONE=America/New_York\n")
                handle.write("HA_TOKEN=should-not-load\n")
            previous_id = os.environ.pop("CALENDAR_ID", None)
            previous_zone = os.environ.pop("CALENDAR_TIMEZONE", None)
            try:
                filled = xai.load_missing_household_env(path)
                self.assertGreaterEqual(filled, 1)
                self.assertEqual(os.environ.get("CALENDAR_ID"), "cal-test")
                self.assertEqual(os.environ.get("CALENDAR_TIMEZONE"), "America/New_York")
                self.assertNotEqual(os.environ.get("HA_TOKEN"), "should-not-load")
            finally:
                if previous_id is None:
                    os.environ.pop("CALENDAR_ID", None)
                else:
                    os.environ["CALENDAR_ID"] = previous_id
                if previous_zone is None:
                    os.environ.pop("CALENDAR_TIMEZONE", None)
                else:
                    os.environ["CALENDAR_TIMEZONE"] = previous_zone

    def test_silence_does_not_open_realtime_socket(self):
        def connect(url, header, timeout):
            raise AssertionError("socket opened with no audio")

        self.assertIsNone(xai.run_realtime_voice_turn(
            iter(()),
            "test-key",
            "voice-test",
            "Ada",
            on_pcm=lambda _chunk: None,
            connect=connect,
            tools=[],
        ))

    def test_realtime_turn_uses_custom_voice_and_returns_tool_audio(self):
        socket = ScriptedRealtimeSocket()
        heard = []
        calls = []

        def connect(url, header, timeout):
            self.assertTrue(url.startswith("wss://api.x.ai/v1/realtime"))
            self.assertIn("model=grok-voice-latest", url)
            self.assertIn("Authorization: Bearer test-key", header)
            self.assertNotIn("8642", url)
            return socket

        def execute_tool(name, arguments):
            calls.append((name, arguments))
            return "Nothing on the calendar for the rest of today."

        result = xai.run_realtime_voice_turn(
            [b"\x00\x01" * 20, b"\x00\x02" * 20],
            "test-key",
            "voice-test",
            "Ada",
            on_pcm=heard.append,
            connect=connect,
            execute_tool=execute_tool,
            tools=xai.realtime_function_tools(),
            recv_timeout=5,
        )
        self.assertTrue(socket.closed)
        self.assertGreaterEqual(len(heard), 2)
        self.assertEqual(calls[0][0], "calendar_agenda")
        self.assertIn("today", calls[0][1])
        self.assertEqual(result["user"], "what is on today")
        self.assertEqual(result["assistant"], "Nothing on the calendar.")
        blob = "\n".join(item for item in socket.sent if isinstance(item, str))
        self.assertNotIn("8642", blob)
        self.assertNotIn("/v1/responses", blob)
        session = json.loads(socket.sent[0])
        self.assertEqual(session["session"]["voice"], "voice-test")
        names = [item["name"] for item in session["session"]["tools"]]
        self.assertIn("calendar_agenda", names)
        self.assertIn("tasks_list", names)
        self.assertIn("tasks_add", names)
        self.assertNotIn("home", names)
        outputs = [
            json.loads(item)
            for item in socket.sent
            if isinstance(item, str) and '"function_call_output"' in item
        ]
        self.assertEqual(outputs[0]["item"]["call_id"], "call_1")
        self.assertTrue(any(
            isinstance(item, str) and json.loads(item).get("type") == "response.create"
            for item in socket.sent
        ))


if __name__ == "__main__":
    unittest.main()
