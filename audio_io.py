"""Dependency-light audio decoding for the Voice Integrity demo.

Handles every WAV flavour a phone/browser/contact-centre is likely to emit —
including the G.711 mu-law/A-law telephony payloads described in the research
(Twilio Media Streams fork call audio as 8 kHz mu-law) — with numpy + scipy only.
No ffmpeg, no libsndfile. Non-WAV containers (mp3/m4a/ogg/webm) are transcoded
to 16 kHz mono WAV in the browser before upload; see static/app.js.
"""
from __future__ import annotations

import struct
from fractions import Fraction

import numpy as np
from scipy import signal

TARGET_SR = 16000

WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_FLOAT = 0x0003
WAVE_FORMAT_ALAW = 0x0006
WAVE_FORMAT_MULAW = 0x0007
WAVE_FORMAT_EXTENSIBLE = 0xFFFE

CODEC_NAMES = {
    WAVE_FORMAT_PCM: "PCM",
    WAVE_FORMAT_FLOAT: "IEEE float",
    WAVE_FORMAT_ALAW: "G.711 A-law (telephony)",
    WAVE_FORMAT_MULAW: "G.711 mu-law (telephony)",
}


class AudioError(ValueError):
    """Raised with a message that is safe (and useful) to show a judge."""


def _mulaw_to_linear(u8: np.ndarray) -> np.ndarray:
    u = (~u8.astype(np.int32)) & 0xFF
    t = ((u & 0x0F) << 3) + 0x84
    t = t << ((u & 0x70) >> 4)
    val = t - 0x84
    val = np.where(u & 0x80, -val, val)
    return (val / 32768.0).astype(np.float32)


def _alaw_to_linear(a8: np.ndarray) -> np.ndarray:
    a = a8.astype(np.int32) ^ 0x55
    mant, exp = a & 0x0F, (a & 0x70) >> 4
    val = np.where(exp == 0, (mant << 4) + 8,
                   ((mant << 4) + 0x108) << np.maximum(exp - 1, 0))
    val = np.where(a & 0x80, -val, val)
    return (val / 32768.0).astype(np.float32)


def _decode_pcm(raw: bytes, bits: int, fmt: int) -> np.ndarray:
    if fmt == WAVE_FORMAT_MULAW:
        return _mulaw_to_linear(np.frombuffer(raw, dtype=np.uint8))
    if fmt == WAVE_FORMAT_ALAW:
        return _alaw_to_linear(np.frombuffer(raw, dtype=np.uint8))
    if fmt == WAVE_FORMAT_FLOAT:
        if bits == 32:
            return np.frombuffer(raw, dtype="<f4").astype(np.float32)
        if bits == 64:
            return np.frombuffer(raw, dtype="<f8").astype(np.float32)
        raise AudioError(f"Unsupported float WAV width: {bits}-bit.")
    if fmt == WAVE_FORMAT_PCM:
        if bits == 8:
            return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        if bits == 16:
            return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if bits == 24:
            b = np.frombuffer(raw[: (len(raw) // 3) * 3], dtype=np.uint8).reshape(-1, 3).astype(np.int32)
            v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
            v = np.where(v & 0x800000, v - 0x1000000, v)
            return (v / 8388608.0).astype(np.float32)
        if bits == 32:
            return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        raise AudioError(f"Unsupported PCM width: {bits}-bit.")
    raise AudioError(f"Unsupported WAV codec tag 0x{fmt:04X}. "
                     "Re-record as PCM/mu-law WAV, or let the browser convert it.")


def decode_wav(data: bytes) -> tuple[np.ndarray, int, dict]:
    """Parse a RIFF/WAVE byte string -> (mono float32, sample_rate, meta)."""
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise AudioError("That does not look like an audio file we can read.")
    fmt_chunk, raw = None, None
    pos = 12
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        body = data[pos + 8: pos + 8 + size] if size else b""
        if cid == b"fmt ":
            fmt_chunk = body
        elif cid == b"data":
            raw = body if size and len(body) == size else data[pos + 8:]
        pos += 8 + size + (size & 1)
        if size == 0 and cid not in (b"fmt ", b"data"):
            break
    if fmt_chunk is None or len(fmt_chunk) < 16 or raw is None:
        raise AudioError("The WAV file is missing its format or data chunk (truncated upload?).")

    fmt, channels, sr, _rate, _align, bits = struct.unpack_from("<HHIIHH", fmt_chunk, 0)
    if fmt == WAVE_FORMAT_EXTENSIBLE and len(fmt_chunk) >= 26:
        fmt = struct.unpack_from("<H", fmt_chunk, 24)[0]
    if channels < 1 or sr < 4000:
        raise AudioError(f"Unusable WAV header (channels={channels}, sample rate={sr}).")

    y = _decode_pcm(raw, bits, fmt)
    if channels > 1:
        usable = (len(y) // channels) * channels
        y = y[:usable].reshape(-1, channels).mean(axis=1)
    if y.size == 0:
        raise AudioError("The file contains no audio samples.")
    meta = {
        "codec": CODEC_NAMES.get(fmt, f"tag 0x{fmt:04X}"),
        "bit_depth": 8 if fmt in (WAVE_FORMAT_MULAW, WAVE_FORMAT_ALAW) else bits,
        "channels": channels,
        "source_sample_rate": sr,
    }
    return np.nan_to_num(y.astype(np.float32)), sr, meta


def resample(y: np.ndarray, sr: int, target: int = TARGET_SR) -> np.ndarray:
    if sr == target:
        return y.astype(np.float32)
    f = Fraction(target, sr).limit_denominator(1000)
    return signal.resample_poly(y, f.numerator, f.denominator).astype(np.float32)


def trim_silence(y: np.ndarray, sr: int, pad_ms: int = 120) -> np.ndarray:
    """Energy-gated VAD-lite: drop dead air at the head/tail, keep a small pad."""
    win = max(1, sr // 100)
    n = len(y) // win
    if n < 3:
        return y
    rms = np.sqrt((y[: n * win].reshape(n, win) ** 2).mean(axis=1) + 1e-12)
    thresh = max(rms.max() * 0.04, np.percentile(rms, 10) * 3.0, 1e-4)
    voiced = np.flatnonzero(rms > thresh)
    if voiced.size == 0:
        return y
    pad = int(pad_ms / 10)
    a = max(0, voiced[0] - pad) * win
    b = min(n, voiced[-1] + 1 + pad) * win
    return y[a:b]


def load(data: bytes, max_seconds: int = 60) -> tuple[np.ndarray, dict]:
    """Bytes -> (16 kHz mono float32, meta). Raises AudioError with demo-safe text."""
    y, sr, meta = decode_wav(data)
    meta["source_duration_s"] = round(len(y) / sr, 2)
    y = resample(y, sr)
    y = trim_silence(y, TARGET_SR)
    truncated = len(y) > TARGET_SR * max_seconds
    if truncated:
        y = y[: TARGET_SR * max_seconds]
    peak = float(np.abs(y).max()) if y.size else 0.0
    if peak > 0:
        y = y / max(peak, 0.05) * 0.95  # level-normalise so scores are gain-invariant
    meta.update({
        "sample_rate": TARGET_SR,
        "duration_s": round(len(y) / TARGET_SR, 2),
        "truncated": truncated,
        "upsampled_from_telephony": sr <= 8000,
        "peak_dbfs": round(float(20 * np.log10(max(peak, 1e-6))), 1),
    })
    return y.astype(np.float32), meta
