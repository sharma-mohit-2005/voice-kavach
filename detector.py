"""Voice Integrity detection core - explainable heuristic countermeasure (CM).

Design traces to research/voice-integrity-verification-research.md:
  * frame-level front-end, sliding 3 s window / 0.5 s hop  (streaming design, S2/S7)
  * prosody micro-variation as a *side-channel*, never a standalone verdict  (S4)
  * telephony gating: 8 kHz legs lose everything above 4 kHz, so high-frequency
    cues are disabled and confidence is lowered instead of guessing           (S2)
  * calibrated logistic fusion + calibration_id in every response             (S1.1, S8)
  * speaker consistency answers a *different* question than spoof detection   (S3)

Every threshold/weight lives in config.json. Swap the cue stack for
wav2vec2-XLSR + AASIST and `speaker_embedding()` for ECAPA-TDNN without
touching the API or the UI - the contract below is the plug-in seam.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

CONFIG_PATH = Path(__file__).parent / "config.json"
SR = 16000
FRAME = 512          # 32 ms spectral frame
HOP = 160            # 10 ms hop -> 100 fps frame rate
PITCH_FRAME = 1024   # 64 ms - long enough to resolve a 60 Hz male F0
F0_MIN, F0_MAX = 60.0, 400.0
EPS = 1e-10


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


CONFIG = load_config()


# --------------------------------------------------------------------------
# frame-level front-end (computed once per clip, then aggregated per window)
# --------------------------------------------------------------------------

def _frames(y: np.ndarray, size: int, hop: int) -> np.ndarray:
    if len(y) < size:
        y = np.pad(y, (0, size - len(y)))
    n = 1 + (len(y) - size) // hop
    idx = np.arange(size)[None, :] + hop * np.arange(n)[:, None]
    return y[idx]


def _f0_track(y: np.ndarray):
    """Normalised-autocorrelation F0 with parabolic interpolation.

    Returns (f0_hz with NaN where unvoiced, periodicity 0-1, frame rms).
    """
    fr = _frames(y, PITCH_FRAME, HOP)
    rms = np.sqrt((fr ** 2).mean(axis=1) + EPS)
    fr = fr - fr.mean(axis=1, keepdims=True)
    nfft = 1 << int(math.ceil(math.log2(2 * PITCH_FRAME)))
    lo = int(SR / F0_MAX)
    hi = min(int(SR / F0_MIN), PITCH_FRAME - 2)

    f0 = np.full(len(fr), np.nan, dtype=np.float64)
    per = np.zeros(len(fr), dtype=np.float64)
    for a in range(0, len(fr), 512):                     # chunked: bounded memory
        blk = fr[a:a + 512]
        spec = np.fft.rfft(blk, nfft, axis=1)
        ac = np.fft.irfft(spec * np.conj(spec), nfft, axis=1)[:, :hi + 2].real
        norm = ac / (ac[:, :1] + EPS)
        band = norm[:, lo:hi]
        k = np.argmax(band, axis=1)
        rows = np.arange(len(band))
        peak = band[rows, k]
        lag = k + lo
        left = norm[rows, np.maximum(lag - 1, 0)]
        right = norm[rows, np.minimum(lag + 1, norm.shape[1] - 1)]
        denom = left - 2 * peak + right
        safe = np.abs(denom) > EPS
        delta = np.where(safe, 0.5 * (left - right) / np.where(safe, denom, 1.0), 0.0)
        delta = np.clip(delta, -0.5, 0.5)
        per[a:a + 512] = np.clip(peak, 0.0, 1.0)
        f0[a:a + 512] = SR / np.maximum(lag + delta, 1.0)
    # level gate relative to the loudest frame only: a constant-amplitude drone
    # must stay "voiced", otherwise flat TTS would slip past every pitch cue.
    floor = max(float(rms.max()) * 0.06, 1e-4)
    voiced = (per > 0.42) & (rms > floor) & (f0 >= F0_MIN) & (f0 <= F0_MAX)
    f0[~voiced] = np.nan
    # drop octave doubling/halving: a frame more than 5 semitones from its own
    # neighbourhood is a tracker error, and it would masquerade as pitch range.
    if voiced.sum() >= 15:
        filled = np.where(voiced, f0, np.nan)
        pad = np.pad(filled, (7, 7), constant_values=np.nan)
        local = np.nanmedian(np.lib.stride_tricks.sliding_window_view(pad, 15), axis=-1)
        jump = np.abs(12 * np.log2(np.where(voiced, f0, 1.0) / np.where(local > 0, local, 1.0)))
        f0[voiced & (jump > 5.0)] = np.nan
    return f0, per, rms


def frame_features(y: np.ndarray) -> dict:
    """All per-frame arrays the scorer needs. 100 frames per second."""
    f0, per, _ = _f0_track(y)
    raw = _frames(y, FRAME, HOP)
    fr = raw * np.hanning(FRAME)
    mag = np.abs(np.fft.rfft(fr, axis=1)) + EPS
    freqs = np.fft.rfftfreq(FRAME, 1.0 / SR)
    n = min(len(f0), len(mag))

    mag = mag[:n]
    total = mag.sum(axis=1) + EPS
    hf = mag[:, freqs > 4000].sum(axis=1) / total
    gm = np.exp(np.log(mag).mean(axis=1))
    flat = gm / (mag.mean(axis=1) + EPS)
    centroid = (mag * freqs).sum(axis=1) / total
    rms = np.sqrt((raw[:n] ** 2).mean(axis=1) + EPS)
    db = 20 * np.log10(rms + EPS)
    # adaptive speech gate: 30 dB below the loud frames, never below the floor+6
    thr = max(float(np.percentile(db, 90)) - 30.0, float(db.min()) + 6.0)
    speech = db > thr

    return {
        "f0": f0[:n], "periodicity": per[:n], "rms": rms, "db": db, "speech": speech,
        "hf": hf, "flatness": flat, "centroid": centroid,
        "mag": mag, "freqs": freqs, "n": n, "fps": SR / HOP,
    }


# --------------------------------------------------------------------------
# window aggregation -> the scalar feature vector the cues read
# --------------------------------------------------------------------------

def _detrend(x: np.ndarray, k: int = 7) -> np.ndarray:
    """Remove the slow contour so only cycle-scale wobble is left."""
    if len(x) < k + 2:
        return x - x.mean()
    ker = np.ones(k) / k
    trend = np.convolve(x, ker, mode="same")
    edge = k // 2
    trend[:edge] = x[:edge].mean()
    trend[-edge:] = x[-edge:].mean()
    return x - trend


def _runs(mask: np.ndarray):
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def aggregate(ff: dict, a: int = 0, b=None) -> dict:
    b = ff["n"] if b is None else min(b, ff["n"])
    a = max(0, min(a, max(b - 1, 0)))
    f0 = ff["f0"][a:b]
    rms, db, per = ff["rms"][a:b], ff["db"][a:b], ff["periodicity"][a:b]
    speech, hf = ff["speech"][a:b], ff["hf"][a:b]
    mag, freqs = ff["mag"][a:b], ff["freqs"]
    voiced = ~np.isnan(f0)
    nv = int(voiced.sum())

    # --- pitch statistics in semitones (gain / speaker invariant) ----------
    if nv >= 5:
        vals = f0[voiced]
        med = float(np.median(vals))
        st = 12.0 * np.log2(np.clip(vals / med, 0.25, 4.0))
        f0_std_st = float(np.std(st))
    else:
        med, f0_std_st = 0.0, 0.0

    # --- jitter / shimmer over contiguous voiced runs ---------------------
    # Both are detrended first: without that, an intonation glide reads as
    # jitter and a syllable's loudness ramp reads as shimmer, which would
    # score a *natural* contour as suspicious. We want the residual wobble.
    jit, shim = [], []
    for s, e in _runs(voiced):
        if e - s < 6:
            continue
        T = 1.0 / f0[s:e]
        jit.append(np.abs(np.diff(_detrend(T))).mean() / (T.mean() + EPS))
        A = rms[s:e]
        shim.append(np.abs(np.diff(_detrend(A))).mean() / (A.mean() + EPS))
    jitter_pct = float(np.mean(jit) * 100) if jit else 0.0
    shimmer_pct = float(np.mean(shim) * 100) if shim else 0.0

    # --- harmonic-to-noise ratio from periodicity -------------------------
    if nv >= 3:
        r = np.clip(per[voiced], 1e-3, 0.9995)
        hnr_db = float(np.median(10 * np.log10(r / (1 - r))))
    else:
        hnr_db = 0.0

    # --- pause structure --------------------------------------------------
    if len(speech) > 5:
        k = max(1, int(0.08 * ff["fps"]))
        sm = np.convolve(speech.astype(float), np.ones(k) / k, mode="same") > 0.5
        pause_ratio = float(1.0 - sm.mean())
        pauses = [e - s for s, e in _runs(~sm) if (e - s) >= 0.12 * ff["fps"]]
        pauses_per_min = float(len(pauses) / max(len(sm) / ff["fps"], 1e-3) * 60)
        mean_pause_ms = float(np.mean(pauses) / ff["fps"] * 1000) if pauses else 0.0
        # humans vary their pauses; a synthesiser is metronomic
        pause_cv = float(np.std(pauses) / (np.mean(pauses) + EPS)) if len(pauses) >= 3 else 0.5
    else:
        pause_ratio, pauses_per_min, mean_pause_ms, pause_cv = 0.0, 0.0, 0.0, 0.5

    # --- syllabic (2-8 Hz) modulation depth --------------------------------
    # Level-invariant: RMS of the 2-8 Hz component of the *mean-normalised*
    # loudness envelope. Human speech pulses at 3-5 Hz; a TTS drone does not.
    if len(rms) >= 64:
        env = rms / (rms.mean() + EPS) - 1.0
        w = np.hanning(len(env))
        spec = np.abs(np.fft.rfft(env * w)) * (2.0 / max(w.sum(), EPS))
        mf = np.fft.rfftfreq(len(env), 1.0 / ff["fps"])
        band = (mf >= 2.0) & (mf <= 8.0)
        mod_ratio = float(np.sqrt(np.sum(spec[band] ** 2) / 2.0)) if band.any() else 0.0
        syllable_hz = float(mf[band][int(np.argmax(spec[band]))]) if band.any() else 0.0
    else:
        mod_ratio, syllable_hz = 0.0, 0.0

    # --- spectrum: HF energy, bandwidth, cut-off cliff, noise floor -------
    ltas = mag[speech].mean(axis=0) if speech.any() else mag.mean(axis=0)
    ltas_db = 20 * np.log10(ltas + EPS)
    ltas_db = ltas_db - ltas_db.max()
    above = np.flatnonzero(ltas_db > -45.0)
    bandwidth = float(freqs[above[-1]]) if above.size else float(freqs[-1])
    # Band cliff = a *step* in the long-term spectrum between 2 kHz and 7 kHz.
    # Restricted to that range on purpose: every real recording rolls off near
    # Nyquist, and counting that as a cliff flagged genuine audio as synthetic.
    khz = max(1, int(round(1000.0 / (freqs[1] - freqs[0]))))
    band = np.flatnonzero((freqs >= 2000) & (freqs <= 7000 - 1000))
    if band.size and band[-1] + khz < len(ltas_db):
        drops = ltas_db[band] - ltas_db[band + khz]
        cliff_db = float(np.max(drops))
    else:
        cliff_db = 0.0
    quiet = db[~speech] if (~speech).any() else db[db <= np.percentile(db, 10)]
    noise_floor_db = float(np.median(quiet)) if quiet.size else float(db.min())

    return {
        "f0_med_hz": med,
        "f0_std_st": f0_std_st,
        "voiced_ratio": float(nv / max(len(f0), 1)),
        "jitter_pct": jitter_pct,
        "shimmer_pct": shimmer_pct,
        "hnr_db": hnr_db,
        "hf_ratio_pct": float(hf[speech].mean() * 100) if speech.any() else float(hf.mean() * 100),
        "flatness": float(np.median(ff["flatness"][a:b])),
        "centroid_hz": float(np.mean(ff["centroid"][a:b])),
        "bandwidth_hz": bandwidth,
        "cliff_db": cliff_db,
        "noise_floor_db": noise_floor_db,
        "pause_ratio": pause_ratio,
        "pauses_per_min": pauses_per_min,
        "mean_pause_ms": mean_pause_ms,
        "pause_cv": pause_cv,
        "mod_ratio": mod_ratio,
        "syllable_hz": syllable_hz,
        "speech_ratio": float(speech.mean()) if len(speech) else 0.0,
    }


# --------------------------------------------------------------------------
# cue evaluation + calibrated logistic fusion
# --------------------------------------------------------------------------

def _ramp(x: float, lo: float, hi: float, direction: str) -> float:
    """0 = normal, 1 = fully abnormal, linear in between."""
    if hi == lo:
        return 0.0
    t = (x - lo) / (hi - lo)
    e = 1.0 - t if direction == "down" else t
    return float(min(1.0, max(0.0, e)))


def _gate_ok(gate, feats: dict) -> bool:
    if gate is None:
        return True
    if gate == "wideband":            # 8 kHz telephony carries no >4 kHz evidence
        return feats["bandwidth_hz"] > 4300
    if gate == "has_pauses":          # cannot judge silence without silence
        return feats["pause_ratio"] > 0.02
    return True


def evaluate_cues(feats: dict, cfg: dict = CONFIG) -> list:
    out = []
    for name, c in cfg["cues"].items():
        value = float(feats.get(c["feature"], 0.0))
        active = _gate_ok(c.get("gate"), feats)
        e = _ramp(value, c["lo"], c["hi"], c["dir"]) if active else 0.0
        if not active:
            text = "Not assessable on this leg - " + c["ok"].split(" (")[0].lower()
        else:
            text = (c["risk"] if e >= 0.5 else c["ok"]).format(v=value)
        out.append({
            "id": name, "value": round(value, 4), "evidence": round(e, 3),
            "weight": c["weight"], "contribution": round(e * c["weight"], 3),
            "gated": not active,
            "kind": "risk" if e >= 0.5 else ("watch" if e >= 0.25 else "ok"),
            "text": text,
        })
    return out


def spoof_from_cues(cues: list, cfg: dict = CONFIG) -> float:
    z = cfg["logistic"]["bias"] + cfg["logistic"]["scale"] * sum(c["contribution"] for c in cues)
    return float(100.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0))))


def prosody_from_cues(cues: list, cfg: dict = CONFIG) -> float:
    ids = set(cfg["prosody_cues"])
    sel = [c for c in cues if c["id"] in ids and not c["gated"]]
    tot = sum(c["weight"] for c in sel)
    return float(100.0 * sum(c["contribution"] for c in sel) / tot) if tot else 0.0


# --------------------------------------------------------------------------
# speaker consistency (ECAPA-TDNN plug-in seam)
# --------------------------------------------------------------------------

def _mel_filterbank(n_bands: int = 36) -> np.ndarray:
    freqs = np.fft.rfftfreq(FRAME, 1.0 / SR)

    def to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def to_hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    pts = to_hz(np.linspace(to_mel(60.0), to_mel(SR / 2 - 100.0), n_bands + 2))
    fb = np.zeros((n_bands, len(freqs)), dtype=np.float32)
    for i in range(n_bands):
        l, c, r = pts[i], pts[i + 1], pts[i + 2]
        up = (freqs >= l) & (freqs <= c)
        dn = (freqs > c) & (freqs <= r)
        fb[i, up] = (freqs[up] - l) / max(c - l, 1e-6)
        fb[i, dn] = (r - freqs[dn]) / max(r - c, 1e-6)
    return fb


_FB = _mel_filterbank()


def speaker_embedding(ff: dict) -> dict:
    """Voice fingerprint: cepstral mean + within-clip spread + pitch register.

    Cosine on raw log-mel means was useless here - every human vowel spectrum
    correlates at 0.95+, so a different speaker still scored 98/100. Comparing
    cepstral means *in units of the within-clip spread* (a poor-man's
    Mahalanobis distance) restores the dynamic range, and pitch register is
    added because it is the single strongest cheap speaker cue.

    Swap wholesale for speechbrain/spkrec-ecapa-voxceleb (cosine on 192-d
    embeddings) when the weights are available - only this function and
    `speaker_match` need to change.
    """
    m = ff["mag"][ff["speech"]] if ff["speech"].any() else ff["mag"]
    logmel = np.log(m @ _FB.T + EPS)
    cep = fft_dct(logmel)[:, 1:17]              # drop c0: that is loudness
    f0 = ff["f0"][~np.isnan(ff["f0"])]
    return {
        "mean": cep.mean(axis=0),
        "spread": cep.std(axis=0),
        "pitch_st": float(12 * np.log2(np.median(f0) / 100.0)) if f0.size else 0.0,
        "bandwidth": float(ff["freqs"][-1]),
    }


def fft_dct(x: np.ndarray) -> np.ndarray:
    """DCT-II along the last axis via FFT (keeps scipy.fft optional)."""
    n = x.shape[-1]
    ext = np.concatenate([x, x[..., ::-1]], axis=-1)
    spec = np.fft.rfft(ext, axis=-1)[..., :n]
    k = np.arange(n)
    return (spec * np.exp(-1j * np.pi * k / (2 * n))).real * (2.0 / np.sqrt(2 * n))


def speaker_match(test: dict, ref: dict, cfg: dict = CONFIG):
    a, b = speaker_embedding(test), speaker_embedding(ref)
    spread = 0.5 * (a["spread"] + b["spread"]) + EPS
    dist = float(np.mean(np.abs(a["mean"] - b["mean"]) / spread))
    d_pitch = abs(a["pitch_st"] - b["pitch_st"])
    s = cfg["speaker"]
    score = 100.0 * math.exp(-((dist / s["dist_scale"]) ** 2 + (d_pitch / s["pitch_scale_st"]) ** 2))
    return float(np.clip(score, 0.0, 100.0)), {
        "timbre_distance": round(dist, 3),
        "pitch_delta_st": round(d_pitch, 2),
    }


# --------------------------------------------------------------------------
# sliding-window timeline (mirrors the streaming design in the research)
# --------------------------------------------------------------------------

def timeline(ff: dict, cfg: dict = CONFIG) -> dict:
    win = int(cfg["audio"]["window_seconds"] * ff["fps"])
    hop = int(cfg["audio"]["hop_seconds"] * ff["fps"])
    scores, times = [], []
    if ff["n"] >= win:
        for a in range(0, ff["n"] - win + 1, hop):
            f = aggregate(ff, a, a + win)
            scores.append(round(spoof_from_cues(evaluate_cues(f, cfg), cfg), 1))
            times.append(round((a + win / 2) / ff["fps"], 2))
    return {"window_s": cfg["audio"]["window_seconds"], "hop_s": cfg["audio"]["hop_seconds"],
            "t": times, "scores": scores}


# --------------------------------------------------------------------------
# fusion + policy
# --------------------------------------------------------------------------

def fuse(spoof: float, match, prosody: float, context: str, cfg: dict = CONFIG) -> dict:
    f = cfg["fusion"]
    boost = f["context_boost"].get(context, 0)
    if match is None:
        w = f["without_reference"]
        risk = w["spoof"] * spoof + w["prosody"] * prosody + boost
    else:
        w = f["with_reference"]
        risk = w["spoof"] * spoof + w["mismatch"] * (100.0 - match) + w["prosody"] * prosody + boost
    risk = float(np.clip(risk, 0.0, 100.0))
    th = cfg["thresholds"].get(context, cfg["thresholds"]["normal"])
    if risk >= th["red"]:
        band, verdict = "red", "Likely Cloned"
    elif risk >= th["amber"]:
        band, verdict = "amber", "Suspicious"
    else:
        band, verdict = "green", "Likely Genuine"

    # Policy escalations: a weighted average can average away a hard failure.
    # Research S3.3 - on a high-value leg, escalate on EITHER leg (spoof-likely
    # OR voice-mismatch), never on the average of the two.
    escalated = []
    rule = cfg.get("escalations", {}).get("speaker_mismatch")
    if (rule and match is not None and match < rule["match_below"]
            and context in rule["contexts"] and band == "green"):
        band, verdict = rule["min_band"], "Suspicious"
        escalated.append(rule["reason"])

    return {"risk": round(risk, 1), "band": band, "verdict": verdict,
            "action": cfg["actions"][band], "thresholds": th, "escalations": escalated}


def confidence(feats: dict, meta: dict) -> dict:
    notes, score = [], 100
    if meta.get("duration_s", 0) < 3.0:
        score -= 35
        notes.append(f"Only {meta.get('duration_s', 0):.1f}s of audio - 4s or more is far more reliable.")
    if feats["voiced_ratio"] < 0.20:
        score -= 30
        notes.append("Little voiced speech detected - the result is weakly supported.")
    if feats["bandwidth_hz"] <= 4300:
        score -= 20
        notes.append("Narrowband telephony leg (~8 kHz): evidence above 4 kHz is unavailable, "
                     "so high-frequency cues were disabled rather than guessed.")
    if meta.get("upsampled_from_telephony"):
        notes.append("Upsampled from an 8 kHz source; expect ASVspoof-DF-style degradation.")
    if meta.get("truncated"):
        notes.append("Clip truncated to the first 60 seconds.")
    level = "high" if score >= 85 else "medium" if score >= 60 else "low"
    return {"level": level, "score": max(score, 5), "notes": notes}


def waveform_peaks(y: np.ndarray, buckets: int = 360) -> list:
    if len(y) < buckets:
        y = np.pad(y, (0, buckets - len(y)))
    edges = np.linspace(0, len(y), buckets + 1).astype(int)
    return [round(float(np.abs(y[edges[i]:edges[i + 1]]).max()), 3) for i in range(buckets)]


def analyze(y: np.ndarray, meta: dict, reference=None,
            context: str = "high_value", cfg: dict = CONFIG) -> dict:
    ff = frame_features(y)
    feats = aggregate(ff)
    if feats["voiced_ratio"] < 0.02 and feats["speech_ratio"] < 0.10:
        raise ValueError("No speech found in this file - it looks like silence or pure noise.")
    cues = evaluate_cues(feats, cfg)
    spoof = spoof_from_cues(cues, cfg)
    prosody = prosody_from_cues(cues, cfg)

    match, match_detail = None, None
    if reference is not None:
        match, match_detail = speaker_match(ff, frame_features(reference), cfg)

    fused = fuse(spoof, match, prosody, context, cfg)
    return {
        **fused,
        "spoof_score": round(spoof, 1),
        "prosody_flag": round(prosody, 1),
        "match_score": None if match is None else round(match, 1),
        "match_detail": match_detail,
        "cues": sorted(cues, key=lambda c: -c["contribution"]),
        "features": {k: round(v, 3) for k, v in feats.items()},
        "timeline": timeline(ff, cfg),
        "waveform": waveform_peaks(y),
        "confidence": confidence(feats, meta),
    }
