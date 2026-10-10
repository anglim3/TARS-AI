"""On-device readout styled after the twin panels in Interstellar.

Black field, small monospace cyan text, a faint glow, and two panels.
Sleep, listen, talk, think, and boot stay visually distinct. Drawing is
cached blits: no per-frame surfaces, no blur filters, no extra copies.
Fonts are free system faces (DejaVu, Liberation, Noto, or Cascadia).
"""

import os
import time
from datetime import datetime

import pygame

BLACK = (0, 0, 0)
DIVIDER = (0, 64, 68)

# Full brightness is the awake panel. Dim is wake-word sleep.
PALETTES = {
    "full": {
        "hi": (158, 246, 238),
        "text": (86, 220, 212),
        "mid": (46, 148, 146),
        "dim": (24, 90, 92),
    },
    "mid": {
        "hi": (118, 196, 190),
        "text": (64, 164, 158),
        "mid": (36, 112, 110),
        "dim": (20, 72, 74),
    },
    "dim": {
        "hi": (96, 176, 170),
        "text": (64, 140, 136),
        "mid": (42, 102, 100),
        "dim": (30, 74, 74),
    },
}

_FONT_PATH = None
_FONT_TRIED = False
_PERSONA = {"humor": 90, "honesty": 95, "sarcasm": 95}
_PERSONA_AT = 0.0


def mono_font(size):
    """A freely licensed monospace face already installed on the system."""
    global _FONT_PATH, _FONT_TRIED
    pygame.font.init()
    if not _FONT_TRIED:
        _FONT_TRIED = True
        for name in (
            "dejavusansmono",
            "liberationmono",
            "notosansmono",
            "notomono",
            "cascadiamono",
        ):
            found = pygame.font.match_font(name)
            if found:
                _FONT_PATH = found
                break
        if _FONT_PATH is None:
            for path in (
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
                "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
                "/usr/share/fonts/truetype/cascadia/CascadiaMono.ttf",
            ):
                if os.path.isfile(path):
                    _FONT_PATH = path
                    break
    if _FONT_PATH:
        return pygame.font.Font(_FONT_PATH, size)
    return pygame.font.Font(None, size)


def read_personality():
    """Humor / honesty / sarcasm, re-read every few seconds."""
    global _PERSONA, _PERSONA_AT
    now = time.monotonic()
    if now - _PERSONA_AT < 4.0:
        return _PERSONA
    _PERSONA_AT = now
    try:
        from modules.module_xai import personality_levels
        got = personality_levels()
        if isinstance(got, dict):
            _PERSONA = got
    except Exception:
        pass
    return _PERSONA


def format_clock(ampm=False, when=None):
    moment = when if when is not None else datetime.now()
    if ampm:
        text = moment.strftime("%I:%M:%S %p")
        if text.startswith("0"):
            text = text[1:]
        return text
    return moment.strftime("%H:%M:%S")


def pair(label, value, cols):
    """Left label, right value, padded so a column of rows lines up."""
    label = str(label)
    value = str(value)
    if len(label) + 1 + len(value) > cols:
        room = cols - len(value) - 1
        if room < 1:
            return (value if cols < 1 else value[:cols])
        label = label[:room]
    gap = cols - len(label) - len(value)
    return label + (" " * max(1, gap)) + value


def _wrap(text, cols):
    text = " ".join(str(text).split())
    if not text:
        return []
    if len(text) <= cols:
        return [text]
    lines = []
    cur = ""
    for word in text.split(" "):
        if len(word) > cols:
            if cur:
                lines.append(cur)
                cur = ""
            while len(word) > cols:
                lines.append(word[:cols])
                word = word[cols:]
        trial = word if not cur else cur + " " + word
        if len(trial) <= cols:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def compose_frame(
    mode,
    columns=46,
    name="TARS",
    wake="hey tars",
    messages=None,
    personality=None,
    time_text=None,
    ampm=False,
    silence=0.0,
    link=None,
    battery=None,
    thermal=None,
):
    """Build the left log and the right status column for one mode.

    mode is SLEEP, LISTEN, TALK, THINK, BOOT, or IDLE.
    messages are (speaker, text) pairs, oldest first.
    """
    key = str(mode or "SLEEP").upper()
    spec = {
        "SLEEP": ("dim", "STANDBY", "SLEEP", "sleep", "WAKE"),
        "LISTEN": ("full", "LISTENING", "RX", "listen", "RX"),
        "TALK": ("full", "TALKING", "TX", "talk", "TX"),
        "THINK": ("mid", "THINKING", "PROC", "think", "PROC"),
        "BOOT": ("full", "BOOTING", "INIT", "boot", "INIT"),
        "IDLE": ("full", "IDLE", "CODE", "idle", "RUN"),
    }.get(key, ("dim", "STANDBY", "SLEEP", "sleep", "WAKE"))
    palette, status, state, meter_style, meter_label = spec
    cols = max(16, int(columns))
    persona = personality or _PERSONA
    humor = _clamp_pct(persona.get("humor"), 90)
    honesty = _clamp_pct(persona.get("honesty"), 95)
    sarcasm = _clamp_pct(persona.get("sarcasm"), 95)
    if time_text is None:
        time_text = format_clock(ampm)

    lines = [("hi", f"{name}//04"), ("rule", "")]
    if key == "SLEEP":
        lines.extend(_diag(cols, [
            ("STATUS", "SLEEP"),
            ("OPTICS", "CLOSED"),
            ("AUDIO", "ARMED"),
            ("WAKE", wake),
            ("LINK", link or "HOLD"),
            ("THERMAL", thermal or "NOM"),
        ]))
        lines.append(("dim", ""))
        lines.extend(_diag(cols, [
            ("servo bus", "park"),
            ("mic gate", "open"),
            ("vision", "off"),
            ("clock", "hold"),
        ], indent=True))
    elif key == "BOOT":
        lines.extend(_diag(cols, [
            ("MEM", "OK"),
            ("BUS", "OK"),
            ("AUDIO", "OK"),
            ("DISPLAY", "OK"),
            ("PERSONA", "OK"),
            ("WAKE", wake),
            ("LINK", link or "SCAN"),
        ]))
    elif key == "LISTEN":
        hold = f"{silence:.1f}s" if silence and silence > 0 else "LIVE"
        lines.extend(_diag(cols, [
            ("CHANNEL", "OPEN"),
            ("VAD", "HOT"),
        ]))
        lines.append(("in", "  " + pair("gate", "rx", cols - 2)))
        lines.append(("in", "  " + pair("hold", hold, cols - 2)))
        lines.append(("dim", ""))
        lines.extend(_message_lines(messages, cols, name, emphasize="user"))
    elif key == "TALK":
        lines.extend(_diag(cols, [
            ("VOICE", "OPEN"),
            ("TX", "HOT"),
        ]))
        lines.append(("dim", ""))
        lines.extend(_message_lines(messages, cols, name, emphasize="self"))
    elif key == "THINK":
        lines.extend(_diag(cols, [
            ("PARSE", "RUN"),
            ("CONTEXT", "HOLD"),
        ]))
        lines.append(("dim", ""))
        lines.extend(_message_lines(messages, cols, name, emphasize="none"))
    else:
        lines.append(("text", pair("STREAM", "CODE", cols)))
        lines.append(("dim", ""))

    lines.extend(_trace(cols, key))

    rows = [
        ("HONESTY", f"{honesty}%"),
        ("HUMOR", f"{humor}%"),
        ("SARCASM", f"{sarcasm}%"),
        None,
        ("MODE", status),
        ("STATE", state),
        ("TIME", time_text),
    ]
    if battery is not None:
        rows.append(("BATT", battery))
    if thermal and key != "SLEEP":
        rows.append(("TEMP", thermal))
    if link and key not in ("SLEEP", "BOOT"):
        rows.append(("LINK", link))

    return {
        "lines": lines,
        "rows": rows,
        "palette": palette,
        "meter": {"style": meter_style, "label": meter_label},
    }


def _clamp_pct(value, default):
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return default


def _diag(cols, pairs, indent=False):
    lines = []
    width = cols - 2 if indent else cols
    prefix = "  " if indent else ""
    kind = "in" if indent else "text"
    for label, value in pairs:
        lines.append((kind, prefix + pair(label, value, width)))
    return lines


_TRACE = {
    "SLEEP": (
        ("loop", "wait"), ("panel", "lit"), ("night", "20-08"),
        ("saver", "arm"), ("mic", "gate"), ("tx", "mute"),
        ("rx", "arm"), ("heat", "low"), ("task", "wake"),
    ),
    "LISTEN": (
        ("loop", "rx"), ("panel", "lit"), ("night", "20-08"),
        ("saver", "off"), ("mic", "open"), ("tx", "mute"),
        ("queue", "0"), ("task", "hear"),
    ),
    "TALK": (
        ("loop", "tx"), ("panel", "lit"), ("saver", "off"),
        ("mic", "shut"), ("rx", "hold"), ("queue", "0"),
        ("task", "speak"),
    ),
    "THINK": (
        ("loop", "run"), ("panel", "lit"), ("saver", "off"),
        ("mic", "hold"), ("tx", "wait"), ("queue", "1"),
        ("task", "parse"),
    ),
    "BOOT": (
        ("loop", "init"), ("panel", "lit"), ("saver", "off"),
        ("mic", "init"), ("tx", "mute"), ("task", "boot"),
    ),
    "IDLE": (
        ("loop", "code"), ("panel", "lit"), ("saver", "run"),
        ("mic", "arm"), ("tx", "mute"), ("task", "idle"),
    ),
}


def _trace(cols, mode):
    """Dim register lines so the left panel stays dense in every state."""
    rows = _TRACE.get(mode) or _TRACE["SLEEP"]
    lines = [("dim", "")]
    width = cols - 2
    for label, value in rows:
        lines.append(("in", "  " + pair(label, value, width)))
    return lines


def _message_lines(messages, cols, name, emphasize):
    if not messages:
        return [("dim", "  awaiting speech")]
    lines = []
    self_name = str(name).upper()
    tail = list(messages)[-5:]
    for speaker, text in tail:
        who = str(speaker or "").strip() or "USER"
        upper = who.upper()
        is_self = upper == self_name
        if emphasize == "self" and is_self:
            kind = "hi"
        elif emphasize == "user" and not is_self:
            kind = "hi"
        else:
            kind = "text"
        lines.append((kind, f"> {upper}"))
        body_kind = "text" if kind == "hi" else "mid"
        wrapped = _wrap(text, max(8, cols - 2)) or [""]
        for chunk in wrapped[:4]:
            lines.append((body_kind, "  " + chunk))
        lines.append(("dim", ""))
    return lines


class FilmScreen:
    """Draw one frame of the two-panel readout onto an existing surface."""

    def __init__(self, width, height):
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self._cache = {}
        self._scan = None
        self._scan_size = None
        self._char_w = {}
        short = min(self.width, self.height)
        self.left_px = max(12, min(15, short // 34))
        self.right_px = max(13, min(17, short // 30))
        self.font = mono_font(self.left_px)
        self.font_right = mono_font(self.right_px)
        self.left_lh = self.left_px + 4
        self.right_lh = self.right_px + 11
        self._layout()

    def _layout(self):
        w, h = self.width, self.height
        margin = max(12, min(w, h) // 38)
        gutter = max(10, min(w, h) // 48)
        self.margin = margin
        self.gutter = gutter
        if w >= h:
            inner_w = w - margin * 2 - gutter
            left_w = int(inner_w * 0.58)
            self.left = pygame.Rect(margin, margin, left_w, h - margin * 2)
            self.right = pygame.Rect(
                margin + left_w + gutter, margin, inner_w - left_w, h - margin * 2
            )
            self.side_by_side = True
        else:
            inner_h = h - margin * 2 - gutter
            top_h = int(inner_h * 0.62)
            self.left = pygame.Rect(margin, margin, w - margin * 2, top_h)
            self.right = pygame.Rect(
                margin, margin + top_h + gutter, w - margin * 2, inner_h - top_h
            )
            self.side_by_side = False

    def columns(self):
        cw = self._advance(self.font)
        pad = 8
        return max(12, (self.left.w - pad) // max(1, cw))

    def _advance(self, font):
        key = id(font)
        cached = self._char_w.get(key)
        if cached is None:
            cached = font.size("M")[0] or self.left_px
            self._char_w[key] = cached
        return cached

    def _glyph(self, text, color, font):
        key = (text, color, id(font))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if len(self._cache) > 480:
            self._cache.clear()
        # Faint glow: one low-alpha copy of the glyph, shifted a pixel.
        sharp = font.render(text, True, color)
        glow = font.render(text, True, color)
        glow.set_alpha(90)
        plate = pygame.Surface((sharp.get_width() + 2, sharp.get_height() + 2), pygame.SRCALPHA)
        plate.blit(glow, (1, 0))
        plate.blit(glow, (0, 1))
        plate.blit(sharp, (0, 0))
        self._cache[key] = plate
        return plate

    def _scanlines(self):
        size = (self.width, self.height)
        if self._scan is not None and self._scan_size == size:
            return self._scan
        scan = pygame.Surface(size, pygame.SRCALPHA)
        shade = (0, 0, 0, 36)
        for y in range(0, self.height, 3):
            pygame.draw.line(scan, shade, (0, y), (self.width, y))
        self._scan = scan
        self._scan_size = size
        return scan

    def render(self, surface, lines, rows, palette="full", meter=None, phase=0.0, level=0.0, reserve_bottom=0):
        colors = PALETTES.get(palette, PALETTES["full"])
        surface.fill(BLACK)
        self._draw_lines(surface, self.left, lines, colors, reserve_bottom)
        self._draw_rows(surface, self.right, rows, colors, meter, phase, level, reserve_bottom)
        self._draw_divider(surface, colors["dim"])
        surface.blit(self._scanlines(), (0, 0))

    def _draw_divider(self, surface, color):
        if self.side_by_side:
            x = self.left.right + self.gutter // 2
            pygame.draw.line(surface, color, (x, self.left.top), (x, self.left.bottom))
        else:
            y = self.left.bottom + self.gutter // 2
            pygame.draw.line(surface, color, (self.left.left, y), (self.left.right, y))

    def _draw_lines(self, surface, rect, lines, colors, reserve_bottom):
        y = rect.y
        limit = rect.bottom - reserve_bottom
        max_w = rect.w - 4
        for item in lines:
            if y + self.left_lh > limit:
                break
            kind, text = item if isinstance(item, tuple) else ("text", item)
            if kind == "rule":
                pygame.draw.line(
                    surface, colors["dim"], (rect.x, y + 4), (rect.right - 2, y + 4)
                )
                y += 10
                continue
            if text:
                color = colors.get(kind, colors["text"])
                if kind == "in":
                    color = colors["mid"]
                cw = self._advance(self.font)
                limit_chars = max(1, max_w // max(1, cw))
                if len(text) > limit_chars:
                    text = text[:limit_chars]
                surface.blit(self._glyph(text, color, self.font), (rect.x, y))
            y += self.left_lh

    def _draw_rows(self, surface, rect, rows, colors, meter, phase, level, reserve_bottom):
        y = rect.y
        limit = rect.bottom - reserve_bottom
        max_w = rect.w - 2
        for row in rows:
            if y + self.right_lh > limit:
                break
            if not row:
                y += self.right_lh // 2
                continue
            label, value = row
            label_plate = self._glyph(str(label), colors["mid"], self.font_right)
            value_plate = self._glyph(str(value), colors["hi"], self.font_right)
            surface.blit(label_plate, (rect.x, y))
            vx = rect.right - value_plate.get_width() - 2
            if vx < rect.x + label_plate.get_width() + 8:
                vx = rect.x + label_plate.get_width() + 8
            if vx + 4 < rect.right:
                surface.blit(value_plate, (vx, y))
            y += self.right_lh
        if meter and y + self.right_lh + 8 < limit:
            self._draw_meter(
                surface, rect, y, meter, colors, phase, level, max_w
            )

    def _draw_meter(self, surface, rect, y, meter, colors, phase, level, max_w):
        label = str(meter.get("label", ""))
        style = meter.get("style", "sleep")
        if label:
            plate = self._glyph(label, colors["text"], self.font_right)
            surface.blit(plate, (rect.x, y))
            y += self.right_lh - 4
        count = 12
        gap = 3
        bw = max(5, min(12, (max_w - gap * (count - 1)) // count))
        bh = max(6, min(11, self.right_px - 4))
        phase = phase % 1.0
        level = max(0.0, min(1.0, float(level or 0.0)))
        for i in range(count):
            on = _meter_on(style, i, count, phase, level)
            color = colors["hi"] if on else colors["dim"]
            x = rect.x + i * (bw + gap)
            pygame.draw.rect(surface, color, (x, y, bw, bh))


def _meter_on(style, index, count, phase, level):
    if style == "sleep":
        return index == 0 and phase < 0.45
    if style == "listen":
        lit = int(round(level * count))
        if lit <= 0:
            return index == int(phase * count) % count
        return index < lit
    if style == "talk":
        width = 4
        head = int(phase * (count - width + 1))
        return head <= index < head + width
    if style == "think":
        return index % 3 == int(phase * 3) % 3
    if style == "boot":
        return index < 9
    if style == "idle":
        return index == int(phase * count) % count
    return False
