"""Blank and restore the on-device DSI panel.

The 800x480 panel is DRM connector DSI-1 driven by pygame on DISPLAY=:0.
X has no DPMS extension, vcgencmd display_power is not registered, and
/sys/class/graphics/fb0/blank is root-only and does not track this plane.
The panel backlight is /sys/class/backlight/10-0045/brightness (group video).
Writing 0 makes actual_brightness drop to 0; writing the saved level turns
the backlight back on without stopping the app.
"""

import glob
import threading
import time

_IDLE_SEC = 120.0
_BACKLIGHT_GLOB = "/sys/class/backlight/*/brightness"

_instance = None
_init_lock = threading.Lock()


def panel_should_blank(now=None):
    """Night is 8pm through 8am. The panel blanks only then."""
    moment = now if now is not None else time.localtime()
    hour = moment.tm_hour
    return hour >= 20 or hour < 8


def _log(message):
    try:
        from modules.module_messageQue import queue_message
        queue_message(message)
    except Exception:
        pass


class DisplayPower:
    """Turn the panel backlight off after idle, and back on for a wake."""

    def __init__(self, path=None, idle_sec=_IDLE_SEC, start_thread=True):
        self._lock = threading.Lock()
        self._path = path if path is not None else self._find_backlight()
        self._idle_sec = float(idle_sec)
        self._on_level = 255
        self._blanked = False
        # Hold the panel on until the app actually enters wake-word sleep.
        self._suspended = True
        self._idle_from = None
        self._stop = threading.Event()
        if self._path:
            current = self._read()
            if current > 0:
                self._on_level = current
        self._thread = None
        if start_thread:
            self._thread = threading.Thread(
                target=self._run, name="display-blank", daemon=True
            )
            self._thread.start()

    @staticmethod
    def _find_backlight():
        paths = sorted(glob.glob(_BACKLIGHT_GLOB))
        return paths[0] if paths else None

    def _read(self):
        try:
            with open(self._path, "r", encoding="ascii") as handle:
                return int(handle.read().strip() or "0")
        except Exception:
            return 0

    def _write(self, value):
        if not self._path:
            return False
        try:
            with open(self._path, "w", encoding="ascii") as handle:
                handle.write(str(int(value)))
            return True
        except Exception as exc:
            _log(f"WARN: display backlight write failed: {exc}")
            return False

    def _ensure_on(self):
        """Turn the backlight on only if it is currently blanked."""
        if self._blanked or (self._path and self._read() <= 0):
            level = self._on_level if self._on_level > 0 else 255
            if self._write(level):
                if self._blanked:
                    _log("DISPLAY: panel on")
                self._blanked = False

    def enter_sleep(self):
        """Return to wake-word sleep. The 2 minute idle clock starts now."""
        with self._lock:
            self._ensure_on()
            self._suspended = False
            self._idle_from = time.monotonic()

    def wake(self):
        """Wake word accepted. Leave an already-lit panel on and hold it through the turn."""
        with self._lock:
            was_blanked = self._blanked or (self._path and self._read() <= 0)
            self._suspended = True
            self._idle_from = None
            if was_blanked:
                self._ensure_on()

    def _run(self):
        while not self._stop.wait(0.5):
            with self._lock:
                if not panel_should_blank():
                    if self._blanked:
                        self._ensure_on()
                    continue
                if self._suspended or self._blanked or self._idle_from is None:
                    continue
                if time.monotonic() - self._idle_from < self._idle_sec:
                    continue
                if self._write(0):
                    self._blanked = True
                    _log("DISPLAY: panel off (idle)")

    def stop(self):
        self._stop.set()


def get_display_power():
    global _instance
    with _init_lock:
        if _instance is None:
            _instance = DisplayPower()
        return _instance


def note_sleep():
    """Call when the app prints that it is sleeping and is waiting for the wake word."""
    try:
        get_display_power().enter_sleep()
    except Exception as exc:
        _log(f"WARN: display sleep hook failed: {exc}")


def note_wake():
    """Call when the wake word is accepted, before the acknowledgment plays."""
    try:
        get_display_power().wake()
    except Exception as exc:
        _log(f"WARN: display wake hook failed: {exc}")
