# SPDX-License-Identifier: Apache-2.0
"""Shared HTTP benchmark helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

MAX_OUTPUT_TOKENS = 24


def complete(url: str, model: str, prompt: str) -> tuple[float, float, dict]:
    """POST one completion; returns (ttft_s, total_s, {"id", "text"})."""
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "prompt": prompt,
        "stream": True,
    }
    t0 = time.perf_counter()
    text, rid, ttft = "", "", None
    with requests.post(
        f"{url}/v1/completions", json=payload, timeout=600, stream=True
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if not rid:
                rid = chunk.get("id", "")
            piece = chunk["choices"][0].get("text", "")
            if ttft is None and piece:
                ttft = time.perf_counter() - t0
            text += piece
    total = time.perf_counter() - t0
    return ttft or total, total, {"id": rid, "text": text}


def server_hits(server_log: Path | None, response: dict) -> str:
    """Extract the RETRIEVE hit line for this request from the server log."""
    if server_log is None or not server_log.exists():
        return ""
    rid = response.get("id", "")
    ack = ""
    for line in reversed(server_log.read_text(errors="ignore").splitlines()):
        if "RETRIEVE" in line and rid in line:
            if "hit L" in line:
                return line.split("] ", 1)[-1]
            ack = line.split("] ", 1)[-1]
    return ack or "(no RETRIEVE — computed from scratch)"


def chunk_mb(server_log: Path | None) -> float | None:
    """Parse the chunk size (MiB) from the REGISTER line of the server log."""
    if server_log is None or not server_log.exists():
        return None
    for line in server_log.read_text(errors="ignore").splitlines():
        if "REGISTER" in line and "chunk=" in line:
            return float(line.split("chunk=")[1].split(" ")[0])
    return None
