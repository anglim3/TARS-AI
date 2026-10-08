"""
openWakeWord wake detector for TARS-AI.

Uses a custom ONNX wake model (default: wakewords/TARS.onnx) with
onnxruntime. Maintains a rolling 16 kHz PCM ring so continuous wake-and-talk
can stream buffered audio into xAI without closing the mic.

The model files are optional and are not stored in git (*.onnx is ignored).
Download the community "TARS" openWakeWord model (phrase "TARS", MIT; see
COLLECTION_LICENSE) plus the openWakeWord feature models melspectrogram.onnx
and embedding_model.onnx into src/wakewords/. Live wake stays Atomik unless
wake_word_processor is set to openwakeword. The upstream wake model is the
TARS entry in the home-assistant-wakewords-collection; the feature models
ship with openWakeWord.
"""
from __future__ import annotations

import os
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np

from modules.module_messageQue import queue_message

CHUNK = 1280  # 80 ms @ 16 kHz — openWakeWord frame size
RATE = 16000


def _default_model_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wakewords")


class OpenWakeWordDetector:
    """Streaming openWakeWord detector with a rolling PCM ring buffer."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.5,
        confirm_frames: int = 2,
        cooldown: float = 2.0,
        ring_seconds: float = 2.0,
        preroll_seconds: float = 0.3,
        melspec_path: Optional[str] = None,
        embedding_path: Optional[str] = None,
    ):
        from openwakeword.model import Model

        base = _default_model_dir()
        self.model_path = model_path or os.path.join(base, "TARS.onnx")
        self.melspec_path = melspec_path or os.path.join(base, "melspectrogram.onnx")
        self.embedding_path = embedding_path or os.path.join(base, "embedding_model.onnx")
        self.threshold = float(threshold)
        self.confirm_frames = max(1, int(confirm_frames))
        self.cooldown = float(cooldown)
        self.preroll_samples = max(0, int(preroll_seconds * RATE))
        ring_samples = max(self.preroll_samples + RATE, int(ring_seconds * RATE))
        # Store whole chunks for simpler byte reconstruction
        self._ring: Deque[np.ndarray] = deque(maxlen=max(1, ring_samples // CHUNK + 2))
        self._ring_samples = 0
        self._consecutive = 0
        self.last_detection_time = 0.0
        self._last_candidate_log = 0.0
        self.model_key = "TARS"

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"openWakeWord model missing: {self.model_path}")

        self.model = Model(
            wakeword_models=[self.model_path],
            inference_framework="onnx",
            melspec_model_path=self.melspec_path,
            embedding_model_path=self.embedding_path,
        )
        # Model name follows filename stem
        keys = list(self.model.models.keys())
        if keys:
            self.model_key = keys[0]

        queue_message(
            f"INFO: openWakeWord loaded model={os.path.basename(self.model_path)} "
            f"key={self.model_key} thr={self.threshold:.2f} confirm={self.confirm_frames} "
            f"cooldown={self.cooldown:.1f}s"
        )

    def reset(self):
        self._ring.clear()
        self._ring_samples = 0
        self._consecutive = 0
        try:
            self.model.preprocessor.reset()
        except Exception:
            pass

    def _push_ring(self, chunk: np.ndarray):
        self._ring.append(np.asarray(chunk, dtype=np.int16).reshape(-1).copy())
        self._ring_samples = sum(len(c) for c in self._ring)

    def preroll_pcm(self, include_latest: bool = True) -> bytes:
        """Return int16 LE PCM from ~preroll_seconds before the latest sample."""
        if not self._ring:
            return b""
        flat = np.concatenate(list(self._ring))
        if self.preroll_samples <= 0:
            return flat.tobytes() if include_latest else b""
        # Start preroll_samples before the end (detection moment)
        start = max(0, len(flat) - self.preroll_samples)
        return flat[start:].tobytes()

    def process_chunk(self, chunk: np.ndarray) -> Tuple[bool, float]:
        """Feed one int16 chunk (preferably 1280 samples). Returns (accepted, score)."""
        flat = np.asarray(chunk, dtype=np.int16).reshape(-1)
        if flat.size == 0:
            return False, 0.0
        if flat.size < CHUNK:
            flat = np.pad(flat, (0, CHUNK - flat.size))
        elif flat.size > CHUNK:
            # Process in CHUNK steps; only last decision matters for accept
            accepted = False
            score = 0.0
            for i in range(0, flat.size, CHUNK):
                part = flat[i:i + CHUNK]
                if part.size < CHUNK:
                    part = np.pad(part, (0, CHUNK - part.size))
                accepted, score = self.process_chunk(part)
                if accepted:
                    return True, score
            return False, score

        if time.time() - self.last_detection_time < self.cooldown:
            self._push_ring(flat)
            return False, 0.0

        self._push_ring(flat)
        pred = self.model.predict(flat)
        score = float(pred.get(self.model_key, 0.0))

        if score >= self.threshold:
            self._consecutive += 1
        else:
            self._consecutive = 0

        if score >= max(0.15, self.threshold * 0.6):
            now = time.time()
            if now - self._last_candidate_log >= 0.5:
                self._last_candidate_log = now
                queue_message(
                    f"INFO: wake candidate score={score:.3f} thr={self.threshold:.2f} "
                    f"streak={self._consecutive}/{self.confirm_frames} engine=openwakeword "
                    f"accepted={'yes' if self._consecutive >= self.confirm_frames else 'no'}"
                )

        if self._consecutive >= self.confirm_frames:
            self.last_detection_time = time.time()
            self._consecutive = 0
            queue_message(
                f"INFO: wake candidate score={score:.3f} thr={self.threshold:.2f} "
                f"gate=ok accepted=yes engine=openwakeword"
            )
            return True, score
        return False, score
