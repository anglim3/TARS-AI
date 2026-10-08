"""
module_voice_fx.py

Movie-TARS output voice effect.

The reply PCM (mono S16_LE) is run once through a short sox chain
before it reaches aplay: small-speaker band limit, a presence lift, a
faint metallic comb and a short small-room reverb. The result is then
scaled back to the input RMS (so loudness at the current mixer volume is
unchanged) with the peak held at or below -1 dBFS.

Config lives in [TTS] of config.ini and is re-read when the file
changes, so it can be switched without a restart:

    voice_fx = True            # False plays the dry voice
    voice_fx_preset = tars_movie  # tars (subtle) | tars_movie (big room + echo)
    # optional overrides, any of the preset keys below:
    # voice_fx_highpass = 150
    # voice_fx_reverberance = 20

Any failure (sox missing, timeout, bad value) returns the dry PCM.
"""

import configparser
import os
import shutil
import subprocess
import threading

import numpy as np

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.ini"
)

PRESETS = {
    "tars": {
        "highpass": 150.0,        # Hz, small speaker low cut
        "lowpass": 6500.0,        # Hz, small speaker top end
        "presence_freq": 2000.0,  # Hz
        "presence_q": 1.0,
        "presence_db": 2.5,
        "comb_ms": 3.5,           # metallic comb delay, 0 disables
        "comb_decay": 0.12,       # comb feedback level
        "reverberance": 20.0,     # %
        "hf_damping": 60.0,       # %
        "room_scale": 15.0,       # %
        "predelay_ms": 10.0,
        "wet_db": -7.0,           # reverb wet gain
        "echo_in": 0.8,           # discrete room echo (sox echos), 0 ms taps disable
        "echo_out": 0.9,
        "echo1_ms": 0.0,
        "echo1_decay": 0.0,
        "echo2_ms": 0.0,
        "echo2_decay": 0.0,
        "tempo": 1.0,             # sox tempo, pitch kept. Realtime speed is xAI voice_speed
        "tail_ms": 180.0,         # silence padded so the room tail rings out
        "peak_db": -1.0,          # output peak ceiling
        "match_rms": 1.0,         # 1 = keep input loudness
        "timeout_s": 2.0,
    },
}

# Big, obvious room: long pre-delay, large room, loud wet, two discrete echoes.
PRESETS["tars_movie"] = dict(
    PRESETS["tars"],
    reverberance=65.0,
    hf_damping=50.0,
    room_scale=80.0,
    predelay_ms=30.0,
    wet_db=-1.0,
    echo_in=0.8,
    echo_out=0.9,
    echo1_ms=95.0,
    echo1_decay=0.25,
    echo2_ms=140.0,
    echo2_decay=0.12,
    tail_ms=600.0,
)

_lock = threading.Lock()
_cache = {"mtime": None, "settings": None}


def _read_settings(path=_CONFIG_PATH):
    parser = configparser.ConfigParser()
    try:
        parser.read(path)
    except Exception:
        return None
    if not parser.has_section("TTS"):
        return None
    sec = parser["TTS"]
    enabled = str(sec.get("voice_fx", "True")).strip().lower() in ("1", "true", "yes", "on")
    preset_name = str(sec.get("voice_fx_preset", "tars_movie")).strip().lower() or "tars_movie"
    if not enabled or preset_name not in PRESETS:
        return None
    params = dict(PRESETS[preset_name])
    for key in list(params):
        raw = sec.get(f"voice_fx_{key}")
        if raw is None or str(raw).strip() == "":
            continue
        try:
            params[key] = float(str(raw).split("#")[0].strip())
        except ValueError:
            pass
    params["preset"] = preset_name
    return params


def settings(path=_CONFIG_PATH):
    """Active effect parameters, or None when the effect is off."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    with _lock:
        if _cache["mtime"] != mtime:
            _cache["settings"] = _read_settings(path)
            _cache["mtime"] = mtime
        return _cache["settings"]


def sox_effects(p, sample_rate=16000):
    """sox effect arguments for parameters ``p``."""
    nyq = sample_rate / 2.0
    fx = ["gain", "-8"]  # headroom for the EQ lift, echoes and the reverb sum
    if p["highpass"] > 0:
        fx += ["highpass", f"{p['highpass']:g}"]
    if 0 < p["lowpass"] < nyq * 0.98:
        fx += ["lowpass", f"{p['lowpass']:g}"]
    if p["presence_db"]:
        fx += ["equalizer", f"{p['presence_freq']:g}", f"{p['presence_q']:g}q", f"{p['presence_db']:g}"]
    if p["comb_ms"] > 0 and p["comb_decay"] > 0:
        fx += ["echo", "1", "1", f"{p['comb_ms']:g}", f"{p['comb_decay']:g}"]
    # Speed before the room so echoes and reverb keep their real timing.
    tempo = float(p.get("tempo", 1.0) or 1.0)
    if abs(tempo - 1.0) > 1e-3:
        fx += ["tempo", "-s", f"{tempo:g}"]
    if p["tail_ms"] > 0:
        fx += ["pad", "0", f"{p['tail_ms'] / 1000.0:g}"]
    taps = []
    for n in (1, 2):
        ms, decay = p.get(f"echo{n}_ms", 0), p.get(f"echo{n}_decay", 0)
        if ms > 0 and decay > 0:
            taps += [f"{ms:g}", f"{decay:g}"]
    if taps:
        fx += ["echos", f"{p['echo_in']:g}", f"{p['echo_out']:g}", *taps]
    if p["reverberance"] > 0:
        fx += [
            "reverb",
            f"{p['reverberance']:g}",
            f"{p['hf_damping']:g}",
            f"{p['room_scale']:g}",
            "0",
            f"{p['predelay_ms']:g}",
            f"{p['wet_db']:g}",
        ]
    return fx


def _rms(x):
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def process_pcm(pcm, sample_rate=16000, params=None):
    """Return effected mono S16_LE PCM, or the input unchanged on failure."""
    if not pcm:
        return pcm
    p = params if params is not None else settings()
    if not p:
        return pcm
    sox = shutil.which("sox")
    if not sox:
        return pcm
    raw = ["-t", "raw", "-r", str(int(sample_rate)), "-e", "signed", "-b", "16", "-c", "1"]
    cmd = [sox, "-V1", *raw, "-", *raw, "-", *sox_effects(p, sample_rate)]
    try:
        done = subprocess.run(
            cmd, input=bytes(pcm), capture_output=True,
            timeout=float(p.get("timeout_s", 2.0)), check=False,
        )
    except Exception:
        return pcm
    if done.returncode != 0 or not done.stdout:
        return pcm

    dry = np.frombuffer(bytes(pcm)[: len(pcm) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768.0
    wet = np.frombuffer(done.stdout[: len(done.stdout) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768.0
    if wet.size == 0:
        return pcm
    gain = 1.0
    if p.get("match_rms", 1.0):
        r_wet = _rms(wet[: dry.size] if wet.size > dry.size else wet)
        if r_wet > 1e-6:
            gain = _rms(dry) / r_wet
    peak = float(np.max(np.abs(wet))) * gain
    ceiling = 10 ** (float(p.get("peak_db", -1.0)) / 20.0)
    if peak > ceiling and peak > 0:
        gain *= ceiling / peak
    out = np.clip(wet * gain, -1.0, 1.0)
    return (out * 32767.0).astype("<i2").tobytes()


def process_float(data, sample_rate, params=None):
    """Same effect for a float32 numpy buffer (mono or first channel)."""
    p = params if params is not None else settings()
    if not p:
        return data
    try:
        mono = data if data.ndim == 1 else data[:, 0]
        pcm = (np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        out = process_pcm(pcm, sample_rate, p)
        if out is pcm:
            return data
        return np.frombuffer(out, dtype="<i2").astype(np.float32) / 32768.0
    except Exception:
        return data
