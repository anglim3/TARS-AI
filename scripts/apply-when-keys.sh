#!/usr/bin/env bash
# Flip src/config.ini to the Friday voice block only when the three
# Amelia names are set. Prints one status line and never prints values.
# Desktop autostart stays off. This script does not enable systemd units.
set +x
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
UPSTREAM_ENV="/home/tars/TARS-AI-upstream/.env"

if [[ -f "${UPSTREAM_ENV}" ]]; then
  ENV_FILE="${UPSTREAM_ENV}"
else
  ENV_FILE="${REPO_ROOT}/.env"
fi

export TARS_ENV_FILE="${ENV_FILE}"
export TARS_CONFIG_FILE="${REPO_ROOT}/src/config.ini"

python3 - <<'PY'
import os
import re
import sys
from pathlib import Path

UNCHANGED = "processors left unchanged"
NEEDED = ("XAI_API_KEY", "XAI_TTS_VOICE_ID", "HERMES_API_KEY")

# Friday block from config-switches.md. Wake stays Atomik / hey tars.
# base_url is the loopback Hermes root, not the vision URL.
TARGETS = {
    "STT": {
        "wake_word": "hey tars",
        "wake_word_processor": "atomik",
        "atomik_mode": "auto",
        "stt_processor": "xai",
    },
    "TTS": {
        "ttsoption": "xai",
    },
    "LLM": {
        "llm_backend": "hermes",
        "base_url": "http://127.0.0.1:8642/v1",
        "json_mode": "False",
    },
    "SERVO": {
        "arms_present": "False",
    },
}

SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$")
ASSIGN_RE = re.compile(r"^(\s*)([A-Za-z0-9_]+)(\s*=\s*)(.*)$")


def parse_env_value(raw):
    raw = raw.strip()
    if not raw:
        return ""
    if raw[0] in ("'", '"'):
        quote = raw[0]
        chars = []
        i = 1
        while i < len(raw):
            char = raw[i]
            if char == "\\" and quote == '"' and i + 1 < len(raw):
                chars.append(raw[i + 1])
                i += 2
                continue
            if char == quote:
                return "".join(chars).strip()
            chars.append(char)
            i += 1
        return "".join(chars).strip()
    if " #" in raw:
        raw = raw.split(" #", 1)[0]
    return raw.strip()


def env_values(path):
    values = {}
    file_path = Path(path)
    if not file_path.is_file():
        return values
    text = file_path.read_text(encoding="utf-8", errors="replace")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        name, raw_value = line.split("=", 1)
        name = name.strip()
        if not name:
            continue
        values[name] = parse_env_value(raw_value)
    return values


def keys_ready(values):
    for name in NEEDED:
        if not values.get(name, "").strip():
            return False
    return True


def apply_block(text):
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()
    seen = {section: set() for section in TARGETS}
    entered = set()
    out = []
    current = None

    def flush_missing(section):
        if section not in TARGETS:
            return
        missing = [
            (key, value)
            for key, value in TARGETS[section].items()
            if key not in seen[section]
        ]
        if not missing:
            return
        blanks = 0
        while blanks < len(out) and out[-1 - blanks].strip() == "":
            blanks += 1
        insert_at = len(out) - blanks
        for offset, (key, value) in enumerate(missing):
            out.insert(insert_at + offset, f"{key} = {value}")
            seen[section].add(key)

    for line in lines:
        section_match = SECTION_RE.match(line)
        if section_match:
            flush_missing(current)
            current = section_match.group(1).strip()
            entered.add(current)
            out.append(line)
            continue
        assign_match = ASSIGN_RE.match(line)
        stripped = line.lstrip()
        if (
            assign_match
            and current in TARGETS
            and not stripped.startswith("#")
            and not stripped.startswith(";")
        ):
            key = assign_match.group(2)
            wanted = {name.lower(): name for name in TARGETS[current]}
            canon = wanted.get(key.lower())
            if canon:
                indent = assign_match.group(1)
                equals = assign_match.group(3)
                out.append(f"{indent}{key}{equals}{TARGETS[current][canon]}")
                seen[current].add(canon)
                continue
        out.append(line)

    flush_missing(current)
    for section, keys in TARGETS.items():
        if section in entered:
            continue
        if out and out[-1] != "":
            out.append("")
        out.append(f"[{section}]")
        entered.add(section)
        for key, value in keys.items():
            out.append(f"{key} = {value}")
            seen[section].add(key)

    body = newline.join(out)
    if text.endswith(("\n", "\r\n")) or text == "":
        body += newline
    return body


def main():
    values = env_values(os.environ.get("TARS_ENV_FILE", ""))
    if not keys_ready(values):
        print(UNCHANGED)
        return 0

    config_path = Path(os.environ["TARS_CONFIG_FILE"])
    if not config_path.is_file():
        print(UNCHANGED)
        return 0

    original = config_path.read_text(encoding="utf-8", errors="replace")
    updated = apply_block(original)
    if updated != original:
        tmp_path = config_path.with_name(config_path.name + ".tmp")
        tmp_path.write_text(updated, encoding="utf-8")
        os.chmod(tmp_path, config_path.stat().st_mode)
        os.replace(tmp_path, config_path)
    print("processors set in src/config.ini")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except OSError:
        print(UNCHANGED)
        sys.exit(0)
PY
