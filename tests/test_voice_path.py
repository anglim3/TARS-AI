"""Voice-path checks that do not need API keys or a running Hermes."""

import json
import os
import sys
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


if __name__ == "__main__":
    unittest.main()
