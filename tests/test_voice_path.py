"""Voice-path checks that do not need API keys or a running Hermes."""

import base64
import json
import os
import queue
import sys
import threading
import unittest

import numpy as np

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


def _tone(level, samples=1600):
    return np.full(samples, int(level), dtype=np.int16).tobytes()


class PcmSource:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.index = 0
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed or self.index >= len(self.chunks):
            raise StopIteration
        chunk = self.chunks[self.index]
        self.index += 1
        return chunk

    def close(self):
        self.closed = True


class ScriptedRealtimeSocket:
    """One reply after the client commits. Appends do not end the turn."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.sent = []
        self.closed = False
        self.timeout = 3
        self._inbox = queue.Queue()

    def settimeout(self, timeout):
        self.timeout = timeout

    def recv(self):
        try:
            item = self._inbox.get(timeout=self.timeout or 3)
        except queue.Empty as exc:
            raise TimeoutError("recv timed out") from exc
        if item is None:
            raise ConnectionError("closed")
        return item

    def send(self, data, opcode=None):
        self.sent.append(data)
        if not isinstance(data, str):
            return
        event = json.loads(data)
        kind = event.get("type")
        if kind == "session.update":
            self._inbox.put(json.dumps({
                "type": "session.updated",
                "session": {
                    "voice": "xai_ara",
                    "custom_voice_tokens": self.tokens,
                },
            }))
        elif kind == "input_audio_buffer.commit":
            pcm = base64.b64encode(b"\x02\x00" * 8).decode("ascii")
            self._inbox.put(json.dumps({
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "Testing",
            }))
            self._inbox.put(json.dumps({
                "type": "response.output_audio_transcript.done",
                "transcript": "The lab is empty.",
            }))
            self._inbox.put(json.dumps({
                "type": "response.output_audio.delta",
                "delta": pcm,
            }))
            self._inbox.put(json.dumps({"type": "response.done"}))

    def close(self):
        self.closed = True
        self._inbox.put(None)


class _Screen:
    def __init__(self):
        self.lines = []
        self.streamed = []

    def update_data(self, key, value, msg_type="INFO"):
        self.lines.append((key, value, msg_type))

    def update_streaming_data(self, value):
        """The bug: this rewrites the user line in place."""
        self.streamed.append(value)
        if self.lines:
            key, _old, msg_type = self.lines[-1]
            self.lines[-1] = (key, value, msg_type)


class _FakeStream:
    def __init__(self):
        self.stopped = False
        self.closed = False

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class _StuckHub:
    def __init__(self):
        self.input_open = True
        self.paused = False

    def force_release_input(self):
        return None

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False


class RealtimeTurnTests(unittest.TestCase):
    def test_floor_only_falls(self):
        gate = xai.FallingNoiseFloor()
        self.assertFalse(gate.is_speech(100.0))
        self.assertTrue(gate.is_speech(2000.0))
        self.assertLessEqual(gate.floor, 100.0)
        self.assertTrue(gate.is_speech(8000.0))
        self.assertLessEqual(gate.floor, 100.0)
        self.assertFalse(gate.is_speech(40.0))
        self.assertLessEqual(gate.floor, 40.0)

    def test_turn_stops_after_one_second_of_silence(self):
        chunks = (
            [_tone(5000)] * 3
            + [_tone(80)] * 2
            + [_tone(5000)] * 4
            + [_tone(80)] * 15
        )
        source = PcmSource(chunks)
        kept, reason = xai.read_until_turn_end(lambda: next(source))
        self.assertEqual(reason, "silence")
        self.assertEqual(source.index, 3 + 2 + 4 + 10)
        self.assertEqual(len(kept), 2 + 4 + 10)
        self.assertLess(source.index, len(chunks))
        self.assertAlmostEqual(xai.pcm_rms(kept[0]), 80.0, delta=0.1)

    def test_no_speech_ends_at_six_seconds(self):
        chunks = [_tone(5000)] * 3 + [_tone(80)] * 80
        source = PcmSource(chunks)
        _kept, reason = xai.read_until_turn_end(lambda: next(source))
        self.assertEqual(reason, "no_speech")
        self.assertEqual(source.index, 3 + 60)
        self.assertLess(source.index, len(chunks))

    def test_hard_cap_ends_a_turn_that_never_goes_quiet(self):
        chunks = [_tone(5000)] * 3 + [_tone(80)] * 2 + [_tone(5000)] * 130
        source = PcmSource(chunks)
        _kept, reason = xai.read_until_turn_end(lambda: next(source))
        self.assertEqual(reason, "cap")
        self.assertEqual(source.index, 3 + 120)
        self.assertLess(source.index, len(chunks))

    def test_empty_custom_voice_tokens_are_not_tars(self):
        self.assertFalse(xai.custom_voice_loaded({
            "type": "session.updated",
            "session": {"voice": "xai_ara", "custom_voice_tokens": []},
        }))
        self.assertFalse(xai.custom_voice_loaded({
            "type": "session.updated",
            "session": {"voice": "xai_ara"},
        }))

    def test_custom_voice_tokens_are_the_tars_voice(self):
        event = {
            "type": "session.updated",
            "session": {
                "voice": "xai_ara",
                "custom_voice_tokens": [0] * 2893,
            },
        }
        self.assertTrue(xai.custom_voice_loaded(event))
        self.assertTrue(xai.custom_voice_loaded({
            "type": "session.updated",
            "custom_voice_tokens": [1, 2, 3],
        }))

    def test_screen_keeps_the_user_line_and_adds_tars(self):
        screen = _Screen()
        xai.publish_turn_lines(screen, "Jack", "Testing", "The lab is empty.", "TARS")
        self.assertEqual(screen.lines, [
            ("Jack", "Testing", "Jack"),
            ("TARS", "The lab is empty.", "TARS"),
        ])
        self.assertEqual(screen.streamed, [])
        self.assertEqual(screen.lines[0][1], "Testing")

    def test_playback_is_not_attempted_while_the_hub_stream_is_open(self):
        import modules.module_mic as mic

        stuck = _StuckHub()
        calls = []
        played = mic.play_pcm_half_duplex(
            b"\x01\x00" * 8,
            hub=stuck,
            aplay=lambda *args, **kwargs: calls.append("aplay"),
            terminate=lambda: calls.append("terminate"),
            initialize=lambda: calls.append("initialize"),
            cards_text=" 1 [Device]: USB-Audio - USB PnP Sound Device\n",
        )
        self.assertFalse(played)
        self.assertEqual(calls, [])
        self.assertTrue(stuck.input_open)
        self.assertFalse(stuck.paused)

    def test_playback_releases_the_input_before_aplay(self):
        import modules.module_mic as mic

        hub = mic._AudioHub()
        stream = _FakeStream()
        hub._stream = stream
        hub._callbacks[7] = lambda *_args: None
        hub._ensure_stream = lambda: None
        events = []

        def terminate():
            events.append(("terminate", hub.input_open))

        def initialize():
            events.append(("initialize", hub.input_open))

        def aplay(pcm, device, rate):
            events.append(("aplay", hub.input_open, device, rate, pcm))

        cards = (
            " 0 [Headphones     ]: bcm2835 Headphones - bcm2835 Headphones\n"
            " 1 [Device         ]: USB-Audio - USB PnP Sound Device\n"
            "                      C-Media Electronics Inc. USB PnP Sound Device\n"
        )
        played = mic.play_pcm_half_duplex(
            b"\x01\x00" * 8,
            hub=hub,
            aplay=aplay,
            terminate=terminate,
            initialize=initialize,
            cards_text=cards,
        )
        self.assertTrue(played)
        self.assertTrue(stream.stopped)
        self.assertTrue(stream.closed)
        self.assertFalse(hub.input_open)
        self.assertIn(7, hub._callbacks)
        self.assertFalse(hub._paused)
        self.assertEqual([item[0] for item in events], ["terminate", "aplay", "initialize"])
        self.assertFalse(events[0][1])
        self.assertFalse(events[1][1])
        self.assertEqual(events[1][2], "plughw:1,0")
        self.assertEqual(events[1][3], 16000)

    def test_paused_hub_does_not_reopen_the_input(self):
        import modules.module_mic as mic

        hub = mic._AudioHub()
        hub.pause()
        rid = hub.register_callback(lambda *_args: None)
        self.assertFalse(hub.input_open)
        self.assertIn(rid, hub._callbacks)

    def test_realtime_turn_stops_after_silence_and_loads_the_custom_voice(self):
        chunks = (
            [_tone(5000)] * 3
            + [_tone(80)] * 2
            + [_tone(5000)] * 4
            + [_tone(80)] * 15
        )
        source = PcmSource(chunks)
        socket = ScriptedRealtimeSocket([0] * 2893)
        screen = _Screen()
        order = []
        logs = []
        original = xai.queue_message
        xai.queue_message = logs.append

        def connect(url, header, timeout):
            self.assertTrue(url.startswith("wss://api.x.ai/v1/realtime"))
            self.assertIn("model=grok-voice-latest", url)
            self.assertIn("Authorization: Bearer test-key", header)
            self.assertNotIn("8642", url)
            return socket

        def before_play(result):
            self.assertTrue(source.closed)
            order.append("lines")
            xai.publish_turn_lines(
                screen, "Jack", result["user"], result["assistant"], "TARS",
            )

        def play(pcm):
            self.assertTrue(source.closed)
            self.assertTrue(socket.closed)
            order.append(("play", pcm))

        try:
            result = xai.run_realtime_voice_turn(
                source,
                "test-key",
                "tars-voice-id",
                "Jack",
                connect=connect,
                execute_tool=lambda name, arguments: "unused",
                tools=xai.realtime_function_tools(),
                play_pcm=play,
                before_play=before_play,
                recv_timeout=3,
            )
        finally:
            xai.queue_message = original

        self.assertEqual(source.index, 3 + 2 + 4 + 10)
        self.assertTrue(source.closed)
        self.assertTrue(socket.closed)
        self.assertEqual(result["user"], "Testing")
        self.assertEqual(result["assistant"], "The lab is empty.")
        self.assertTrue(result["custom_voice"])
        self.assertEqual(logs.count("INFO: xAI custom voice loaded"), 1)
        self.assertEqual(order[0], "lines")
        self.assertEqual(order[1][0], "play")
        self.assertEqual(screen.lines, [
            ("Jack", "Testing", "Jack"),
            ("TARS", "The lab is empty.", "TARS"),
        ])
        self.assertEqual(screen.streamed, [])
        session = json.loads(socket.sent[0])
        self.assertEqual(session["session"]["voice"], "tars-voice-id")
        self.assertIsNone(session["session"]["turn_detection"])
        self.assertEqual(session["session"]["audio"]["output"]["format"]["rate"], 16000)
        self.assertEqual(session["session"]["audio"]["input"]["format"]["rate"], 16000)
        instructions = session["session"]["instructions"]
        self.assertIn("TARS", instructions)
        self.assertIn("how can I help", instructions)
        self.assertIn("deadpan", instructions)
        names = [item["name"] for item in session["session"]["tools"]]
        self.assertIn("calendar_agenda", names)
        self.assertIn("calendar_create", names)
        self.assertIn("tasks_list", names)
        self.assertIn("tasks_add", names)
        self.assertIn("tasks_complete", names)
        self.assertNotIn("home", names)
        sent_types = [
            json.loads(item).get("type")
            for item in socket.sent
            if isinstance(item, str)
        ]
        self.assertEqual(sent_types.count("input_audio_buffer.append"), 16)
        self.assertIn("input_audio_buffer.commit", sent_types)
        self.assertNotIn("8642", "\n".join(item for item in socket.sent if isinstance(item, str)))
        self.assertNotIn("/v1/responses", "\n".join(item for item in socket.sent if isinstance(item, str)))

    def test_silence_does_not_ask_for_a_reply_or_play(self):
        chunks = [_tone(80)] * 3 + [_tone(80)] * 80
        source = PcmSource(chunks)
        socket = ScriptedRealtimeSocket([])
        played = []

        def connect(url, header, timeout):
            return socket

        result = xai.run_realtime_voice_turn(
            source,
            "test-key",
            "tars-voice-id",
            "Jack",
            connect=connect,
            execute_tool=lambda name, arguments: "unused",
            tools=[],
            play_pcm=played.append,
            recv_timeout=3,
        )
        self.assertIsNone(result)
        self.assertEqual(played, [])
        self.assertEqual(source.index, 3 + 60)
        self.assertTrue(source.closed)
        self.assertTrue(socket.closed)
        sent_types = [
            json.loads(item).get("type")
            for item in socket.sent
            if isinstance(item, str)
        ]
        self.assertNotIn("input_audio_buffer.commit", sent_types)
        self.assertNotIn("response.create", sent_types)

    def test_no_audio_does_not_open_the_realtime_socket(self):
        def connect(url, header, timeout):
            raise AssertionError("socket opened with no audio")

        self.assertIsNone(xai.run_realtime_voice_turn(
            iter(()),
            "test-key",
            "tars-voice-id",
            "Jack",
            connect=connect,
            tools=[],
            play_pcm=lambda _pcm: (_ for _ in ()).throw(AssertionError("played")),
        ))

    def test_home_tool_is_not_available_on_the_realtime_socket(self):
        self.assertEqual(
            xai.execute_household_tool("home", {"text": "turn off the lights"}),
            "That tool is not available.",
        )


if __name__ == "__main__":
    unittest.main()
