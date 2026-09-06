"""Generate the consented demo pack - no real voices, no cloning service, no internet.

Everything here is synthesised with a source-filter (Klatt-style) model so the
"genuine" clips carry the things a real larynx produces and a vocoder usually
does not: a moving F0 contour, cycle-to-cycle jitter and shimmer, aspiration
noise, fricative energy above 4 kHz, breathing pauses and a room noise floor.
The "cloned" clips are the same voice rendered the way flat TTS renders it:
smoothed pitch, no micro-variation, clean silence, a hard vocoder band ceiling.

    python scripts/make_samples.py

Writes samples/*.wav + samples/manifest.json (labels, story, expected verdict).
Swap any file for a real recording (with consent) and the app picks it up.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
from scipy import signal

SR = 16000
OUT = Path(__file__).resolve().parent.parent / "samples"

# vowel formant targets (Hz) for a male tract; scaled per voice
VOWELS = {
    "a": (730, 1090, 2440), "e": (530, 1840, 2480), "i": (270, 2290, 3010),
    "o": (570, 840, 2410), "u": (300, 870, 2240),
}
FRICATIVES = {"s": (4200, 7800), "sh": (2400, 5200), "f": (1400, 7000), "h": (800, 4000)}


# ---------------------------------------------------------------- utterance plan

def plan_utterance(rng, seconds=9.0, speech_rate=4.0, pause_scale=1.0, uniform=False):
    """A list of (kind, duration_s, payload) segments: syllables, frications, pauses."""
    segs, t = [], 0.0
    vkeys, fkeys = list(VOWELS), list(FRICATIVES)
    while t < seconds:
        words = rng.integers(3, 6)                       # a phrase between breaths
        for _ in range(int(words)):
            for _ in range(int(rng.integers(1, 4))):     # syllables per word
                if rng.random() < 0.55:
                    d = float(rng.uniform(0.05, 0.11))
                    segs.append(("fric", d, fkeys[int(rng.integers(0, len(fkeys)))]))
                    t += d
                d = float(rng.uniform(0.7, 1.4) / speech_rate)
                stress = float(rng.random() < 0.35)
                segs.append(("vowel", d, (vkeys[int(rng.integers(0, len(vkeys)))], stress)))
                t += d
            d = 0.02 if uniform else float(rng.uniform(0.03, 0.12))   # word gap
            segs.append(("pause", d, None))
            t += d
        # metronomic phrase gaps are a TTS tell; humans vary them a lot
        d = (0.16 if uniform else float(rng.uniform(0.20, 0.70))) * pause_scale
        segs.append(("pause", d, None))
        t += d
    return segs


def f0_contour(segs, rng, base_hz, span_st=2.6, jitter_hz=0.0, drift=True):
    """Frame-rate (1 kHz) F0 track: phrase declination + accent bumps + drift."""
    total = sum(s[1] for s in segs)
    n = int(total * 1000) + 1
    f0 = np.zeros(n)
    pos, phrase_start = 0.0, 0.0
    phrase_len = max(total / 4.0, 1.0)
    for kind, dur, payload in segs:
        a, b = int(pos * 1000), int((pos + dur) * 1000)
        if kind == "vowel":
            stress = payload[1]
            frac = min((pos - phrase_start) / phrase_len, 1.0)
            decl = -3.0 * frac                                    # declination
            accent = (2.6 if stress else 0.6) * np.hanning(max(b - a, 2))
            local = decl + accent + rng.normal(0, 0.35)
            f0[a:b] = base_hz * 2 ** ((local + rng.normal(0, 0.08, max(b - a, 0))) / 12.0)
        pos += dur
        if kind == "pause" and dur > 0.2:
            phrase_start = pos
    if drift:
        slow = np.cumsum(rng.normal(0, 1.0, n))
        slow = signal.savgol_filter(slow, min(2001, (n // 2) * 2 + 1), 2) if n > 2100 else slow * 0
        f0 = f0 * (1 + 0.006 * slow / (np.abs(slow).max() + 1e-9))
    if jitter_hz:
        f0 = f0 + rng.normal(0, jitter_hz, n)
    return f0


def flatten(f0, base_hz, span_st=0.5):
    """TTS-style pitch: one slow, smooth, tiny excursion - no jitter, no drift."""
    t = np.arange(len(f0)) / 1000.0
    smooth = base_hz * 2 ** (span_st / 12.0 * np.sin(2 * np.pi * t / 3.7))
    return np.where(f0 > 0, smooth, 0.0)


# ---------------------------------------------------------------- source-filter

def glottal_flow(u, open_frac=0.45, close_frac=0.16):
    """Rosenberg glottal flow evaluated at fractional phase u in [0, 1)."""
    y = np.zeros_like(u)
    a = u < open_frac
    t = u[a] / open_frac
    y[a] = 3 * t ** 2 - 2 * t ** 3
    b = (u >= open_frac) & (u < open_frac + close_frac)
    t = (u[b] - open_frac) / close_frac
    y[b] = 1 - t ** 2
    return y


def voiced_source(n, f0_track, rng, jitter=0.012, shimmer=0.06, aspiration=0.02):
    """Phase-domain pulse train: fractional periods, per-cycle jitter + shimmer.

    Placing pulses at integer sample offsets (the obvious implementation) rounds
    every period to a whole sample, which injects ~0.7% of fake jitter - enough
    to make a perfectly flat synthetic voice measure as human. Integrating the
    phase instead keeps periods exact, so the only jitter present is the jitter
    we asked for.
    """
    f0s = np.interp(np.arange(n) / SR * 1000.0, np.arange(len(f0_track)), f0_track)
    f0s = np.maximum(f0s, 0.0)
    base = np.cumsum(f0s) / SR
    cyc = np.floor(base).astype(int)
    ncyc = int(cyc.max()) + 2
    if jitter:
        f0s = f0s * (1.0 + rng.normal(0, jitter, ncyc))[cyc]
    phase = np.cumsum(f0s) / SR
    cyc = np.floor(phase).astype(int)
    amp = (1.0 + rng.normal(0, shimmer, int(cyc.max()) + 2))[cyc]
    flow = glottal_flow(phase - cyc) * amp
    out = np.diff(np.concatenate([[0.0], flow]))
    if aspiration:
        out = out + rng.normal(0, aspiration, n) * (np.abs(out).max() + 1e-6)
    return out


def resonator(x, freq, bw, state=None):
    r = np.exp(-np.pi * bw / SR)
    theta = 2 * np.pi * freq / SR
    a = np.array([1.0, -2 * r * np.cos(theta), r * r])
    b = np.array([1 - 2 * r * np.cos(theta) + r * r])
    if state is None:
        state = np.zeros(2)
    y, state = signal.lfilter(b, a, x, zi=state)
    return y, state


def apply_formants(x, tracks, bws=(70, 100, 160, 220), block=160):
    """Time-varying cascade of 2-pole resonators, filtered block by block."""
    y = np.zeros(len(x))
    states = [None] * len(tracks)
    for a in range(0, len(x), block):
        blk = x[a:a + block]
        idx = min(a // block, len(tracks[0]) - 1)
        cur = blk
        for k, tr in enumerate(tracks):
            cur, states[k] = resonator(cur, max(tr[idx], 90.0), bws[k], states[k])
        y[a:a + len(blk)] = cur
    return y


def band_noise(n, lo, hi, rng):
    x = rng.normal(0, 1, n)
    sos = signal.butter(4, [lo / (SR / 2), min(hi, SR / 2 - 200) / (SR / 2)], btype="band", output="sos")
    return signal.sosfilt(sos, x)


# ---------------------------------------------------------------- renderers

def render(segs, f0, rng, voice_scale=1.0, style="real"):
    total = sum(s[1] for s in segs)
    n = int(total * SR)
    nblocks = n // 160 + 2

    genuine = style == "real"
    jitter = 0.018 if genuine else 0.0002
    shimmer = 0.090 if genuine else 0.002
    aspiration = 0.045 if genuine else 0.0005

    f0_samples = np.interp(np.arange(n) / SR * 1000, np.arange(len(f0)), f0)
    src = voiced_source(n, f0, rng, jitter, shimmer, aspiration)

    # formant tracks + amplitude envelope + voicing mask, all at block rate
    tracks = [np.zeros(nblocks) for _ in range(4)]
    amp = np.zeros(nblocks)
    voiced_mask = np.zeros(nblocks)
    fric = np.zeros(n)
    pos = 0.0
    last = [500 * voice_scale, 1500 * voice_scale, 2500 * voice_scale, 3400 * voice_scale]
    for kind, dur, payload in segs:
        a, b = int(pos * SR / 160), int((pos + dur) * SR / 160)
        b = min(max(b, a + 1), nblocks)
        if kind == "vowel":
            v, stress = payload
            f1, f2, f3 = (f * voice_scale for f in VOWELS[v])
            target = [f1, f2, f3, 3400 * voice_scale]
            for k in range(4):
                tracks[k][a:b] = np.linspace(last[k], target[k], b - a)
                last[k] = target[k]
            env = np.hanning(max((b - a) * 2, 4))[: b - a]
            amp[a:b] = (0.85 + 0.35 * stress) * np.clip(env * 1.6, 0, 1)
            voiced_mask[a:b] = 1.0
        elif kind == "fric":
            lo, hi = FRICATIVES[payload]
            if not genuine:
                lo, hi = min(lo, 5200), min(hi, 5800)    # vocoder ceiling
            s, e = int(pos * SR), int((pos + dur) * SR)
            gain = 0.26 if genuine else 0.03             # TTS under-renders fricatives
            seg = band_noise(e - s, lo, hi, rng) * gain
            fric[s:e] += seg * np.hanning(len(seg))
            for k in range(4):
                tracks[k][a:b] = last[k]
        else:
            for k in range(4):
                tracks[k][a:b] = last[k]
        pos += dur

    k = 9
    amp = np.convolve(amp, np.ones(k) / k, mode="same")
    amp_s = np.repeat(amp, 160)[:n]
    voiced_s = np.repeat(np.convolve(voiced_mask, np.ones(5) / 5, mode="same"), 160)[:n]

    voiced = apply_formants(src * amp_s * voiced_s, tracks)
    y = voiced / (np.abs(voiced).max() + 1e-9) * 0.75 + fric

    if genuine:
        room = rng.normal(0, 1, n)
        room = signal.sosfilt(signal.butter(2, 900 / (SR / 2), btype="low", output="sos"), room)
        y = y + room / (np.abs(room).max() + 1e-9) * 0.004        # -48 dBFS room tone
        early = np.zeros(n)                                       # two early reflections
        for delay, gain in ((int(0.011 * SR), 0.13), (int(0.023 * SR), 0.07)):
            early[delay:] += y[:n - delay] * gain
        y = y + early
    else:
        sos = signal.butter(10, 6000 / (SR / 2), btype="low", output="sos")
        y = signal.sosfilt(sos, y)                                # hard synthetic ceiling
    _ = f0_samples
    return y / (np.abs(y).max() + 1e-9) * 0.9


# ---------------------------------------------------------------- wav writing

def write_wav(name, y, sr=SR, fmt="pcm16"):
    y = np.clip(y, -0.98, 0.98)
    if fmt == "pcm16":
        payload = (y * 32767).astype("<i2").tobytes()
        tag, bits = 1, 16
    elif fmt == "mulaw":
        payload = linear_to_mulaw(y).tobytes()
        tag, bits = 7, 8
    else:
        raise ValueError(fmt)
    block = bits // 8
    hdr = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + len(payload), b"WAVE", b"fmt ", 16,
                      tag, 1, sr, sr * block, block, bits, b"data", len(payload))
    (OUT / name).write_bytes(hdr + payload)
    print(f"  wrote {name:26s} {len(y)/sr:5.1f}s  {sr//1000} kHz  {fmt}")


def linear_to_mulaw(y):
    x = np.clip(y, -1, 1) * 32635
    sign = (x < 0).astype(np.uint8) * 0x80
    x = np.abs(x) + 132
    exponent = np.clip(np.floor(np.log2(np.maximum(x, 1))).astype(int) - 7, 0, 7)
    mantissa = ((x.astype(int) >> (exponent + 3)) & 0x0F).astype(np.uint8)
    return (~(sign | (exponent.astype(np.uint8) << 4) | mantissa)).astype(np.uint8)


# ---------------------------------------------------------------- the demo pack

def build():
    OUT.mkdir(exist_ok=True)
    print("synthesising demo pack (source-filter model, no real voices)")

    # --- CEO voice, English ------------------------------------------------
    rng = np.random.default_rng(11)
    segs = plan_utterance(rng, 9.0, speech_rate=4.2, pause_scale=1.40)
    real_en = render(segs, f0_contour(segs, rng, 118.0), rng, 1.0, "real")
    write_wav("real_en.wav", real_en)

    rng = np.random.default_rng(11)
    segs_c = plan_utterance(rng, 9.0, speech_rate=4.2, pause_scale=0.25, uniform=True)
    f0_flat = flatten(f0_contour(segs_c, rng, 118.0, drift=False), 118.0, 0.55)
    clone_en = render(segs_c, f0_flat, rng, 1.0, "clone")
    write_wav("clone_en.wav", clone_en)

    # --- same persona, Hindi-paced (faster syllable rate, longer phrases) ---
    rng = np.random.default_rng(23)
    segs = plan_utterance(rng, 9.0, speech_rate=5.0)
    write_wav("real_hi.wav", render(segs, f0_contour(segs, rng, 132.0), rng, 1.04, "real"))

    rng = np.random.default_rng(23)
    segs_c = plan_utterance(rng, 9.0, speech_rate=5.0, pause_scale=0.25, uniform=True)
    f0_flat = flatten(f0_contour(segs_c, rng, 132.0, drift=False), 132.0, 0.45)
    write_wav("clone_hi.wav", render(segs_c, f0_flat, rng, 1.04, "clone"))

    # --- replay of a genuine call: phone band + room + handset noise --------
    rng = np.random.default_rng(41)
    segs = plan_utterance(rng, 9.0, speech_rate=4.0)
    base = render(segs, f0_contour(segs, rng, 124.0), rng, 1.0, "real")
    sos = signal.butter(6, [300 / (SR / 2), 3400 / (SR / 2)], btype="band", output="sos")
    replay = signal.sosfilt(sos, base)
    replay = replay + rng.normal(0, 0.02, len(replay))
    replay = np.tanh(replay * 1.8) * 0.8                       # loudspeaker clipping
    write_wav("replay_noisy.wav", replay)

    # --- a genuine human who is simply not the enrolled speaker -------------
    rng = np.random.default_rng(57)
    segs = plan_utterance(rng, 9.0, speech_rate=4.4)
    write_wav("wrong_speaker.wav", render(segs, f0_contour(segs, rng, 205.0), rng, 1.18, "real"))

    # --- the telephony leg: 8 kHz G.711 mu-law, exactly what Twilio forks ---
    sos = signal.butter(6, [300 / (SR / 2), 3400 / (SR / 2)], btype="band", output="sos")
    narrow = signal.sosfilt(sos, clone_en)
    narrow = signal.resample_poly(narrow, 1, 2)
    write_wav("clone_phone_8k.wav", narrow / (np.abs(narrow).max() + 1e-9) * 0.9, sr=8000, fmt="mulaw")

    manifest = {
        "note": "Synthetic demo pack - no real voices were recorded or cloned. "
                "Replace any file with a consented recording and the app picks it up.",
        "samples": [
            {"file": "real_en.wav", "label": "CEO - genuine (EN)", "expect_band": "green",
             "story": "The real CEO calls the CFO about a payment.", "kind": "genuine"},
            {"file": "clone_en.wav", "label": "CEO - cloned (EN)", "expect_band": "red",
             "story": "Same script, synthesised voice asking for a Rs 10L transfer.", "kind": "clone"},
            {"file": "real_hi.wav", "label": "CEO - genuine (HI)", "expect_band": "green",
             "story": "Hindi call from the genuine speaker.", "kind": "genuine"},
            {"file": "clone_hi.wav", "label": "CEO - cloned (HI)", "expect_band": "red",
             "story": "Hindi clone - the separation must hold outside English.", "kind": "clone"},
            {"file": "replay_noisy.wav", "label": "Replay over speakerphone", "expect_band": "amber",
             "story": "A genuine recording replayed down a noisy phone line.", "kind": "replay"},
            {"file": "wrong_speaker.wav", "label": "Different real person", "expect_band": "green",
             "story": "A real human, but not the enrolled speaker - use as reference to fail the match leg.",
             "kind": "genuine", "use_as_reference": True},
            {"file": "clone_phone_8k.wav", "label": "Clone on an 8 kHz phone leg", "expect_band": "amber",
             "story": "The same clone as it arrives from Twilio/Asterisk: G.711 mu-law, 8 kHz. Half the "
                      "evidence is gone with the top 4 kHz, so the system escalates instead of "
                      "over-claiming - and the callback still stops the fraud.", "kind": "clone"},
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  wrote manifest.json ({len(manifest['samples'])} entries)")


if __name__ == "__main__":
    build()
