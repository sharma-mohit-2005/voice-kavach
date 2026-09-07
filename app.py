"""Voice Integrity Verification - upload web app (judges demo MVP).

Run:  python -m uvicorn app:app --port 8000     ->  http://127.0.0.1:8000

Stack: FastAPI + numpy/scipy. No ffmpeg, no GPU, no internet, no database.
Audio is decoded in memory, scored, and dropped - nothing is ever written to
disk, which is what lets the UI claim "scores only, no audio stored" (DPDP
data-minimisation, see research S6.1).

Layers:
    audio_io.py   bytes -> 16 kHz mono float32 (PCM/float/G.711 mu-law/A-law)
    detector.py   features -> cues -> calibrated spoof score -> fusion -> policy
    config.json   every threshold, weight and action string
    app.py        this file: HTTP contract only, no scoring logic
"""
from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import audio_io
import detector

BASE = Path(__file__).parent
STATIC = BASE / "static"
SAMPLES = BASE / "samples"
MAX_UPLOAD_MB = 25

app = FastAPI(title="Voice Integrity Verification", version=detector.CONFIG["model_version"])


@app.middleware("http")
async def no_cache(request, call_next):
    """Never let a browser serve a stale page, stylesheet or script.

    StaticFiles sends only etag/last-modified, so browsers fall back to
    heuristic caching and keep serving an old app.js after an edit - which
    looks exactly like "the app is broken". On a demo we always want fresh.
    """
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store, must-revalidate"
    return response


def _cfg() -> dict:
    """Re-read config.json per request so thresholds can be tuned live."""
    try:
        detector.CONFIG = detector.load_config()
    except Exception:                      # keep serving the last good config
        pass
    return detector.CONFIG


def _error(message: str, hint: str = "", code: int = 400):
    return JSONResponse({"error": message, "hint": hint}, status_code=code)


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the page with mtime-stamped asset URLs.

    Browsers (and preview panes) hold app.js/styles.css in memory cache even
    with no-store, so an edit silently does nothing and the app looks broken.
    Stamping each URL with the file's modification time makes a stale asset
    impossible: edit the file, reload, get the new one.
    """
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for asset in ("app.js", "styles.css"):
        path = STATIC / asset
        if path.exists():
            html = html.replace(f"/static/{asset}", f"/static/{asset}?v={int(path.stat().st_mtime)}")
    return html


@app.get("/api/health")
def health():
    cfg = _cfg()
    return {
        "status": "ok",
        "models_loaded": True,
        "model_version": cfg["model_version"],
        "calibration_id": cfg["calibration_id"],
        "engine": "heuristic-cue CM (explainable); plug-in seams for "
                  "wav2vec2-XLSR+AASIST and ECAPA-TDNN",
        "accepts": "wav (PCM 8/16/24/32-bit, IEEE float, G.711 mu-law/A-law); "
                   "other formats are converted to 16 kHz wav in the browser",
        "cues": len(cfg["cues"]),
        "storage": "none - audio is scored in memory and discarded",
    }


@app.get("/api/config")
def get_config():
    cfg = _cfg()
    return {
        "contexts": [
            {"id": "high_value", "label": "High-value transfer",
             "hint": "Strictest thresholds - money is about to move.",
             "thresholds": cfg["thresholds"]["high_value"]},
            {"id": "privileged", "label": "Privileged access",
             "hint": "Credentials, resets, admin actions.",
             "thresholds": cfg["thresholds"]["privileged"]},
            {"id": "normal", "label": "Normal call",
             "hint": "Routine conversation, lenient thresholds.",
             "thresholds": cfg["thresholds"]["normal"]},
        ],
        "languages": [{"id": "auto", "label": "Auto-detect"},
                      {"id": "en", "label": "English"},
                      {"id": "hi", "label": "Hindi"}],
        "cue_labels": {k: v.get("label", k) for k, v in cfg["cues"].items()},
        "model_version": cfg["model_version"],
        "calibration_id": cfg["calibration_id"],
        "max_seconds": cfg["audio"]["max_seconds"],
        "max_upload_mb": MAX_UPLOAD_MB,
    }


@app.get("/api/samples")
def samples():
    import json
    manifest = SAMPLES / "manifest.json"
    known = {}
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        known = {s["file"]: s for s in data.get("samples", [])}
    out = []
    for p in sorted(SAMPLES.glob("*.wav")):
        meta = known.get(p.name, {})
        out.append({
            "name": p.name,
            "url": f"/samples/{p.name}",
            "label": meta.get("label", p.stem.replace("_", " ")),
            "story": meta.get("story", ""),
            "kind": meta.get("kind", "unknown"),
            "expect_band": meta.get("expect_band"),
            "use_as_reference": meta.get("use_as_reference", False),
            "size_kb": round(p.stat().st_size / 1024),
        })
    return {"samples": out,
            "note": "Synthetic demo pack - no real voices were recorded or cloned."}


@app.post("/api/analyze")
async def analyze(audio: UploadFile = File(...),
                  reference: UploadFile = File(default=None),
                  context: str = Form(default="high_value"),
                  language: str = Form(default="auto")):
    cfg = _cfg()
    started = time.perf_counter()

    raw = await audio.read()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        return _error(f"That file is larger than {MAX_UPLOAD_MB} MB.",
                      "Trim the clip - 8 to 15 seconds of speech is plenty.")
    try:
        y, meta = audio_io.load(raw, cfg["audio"]["max_seconds"])
    except audio_io.AudioError as e:
        return _error(str(e), "Drop a wav/mp3/m4a/ogg file; the browser converts it before upload.")

    if meta["duration_s"] < cfg["audio"]["min_seconds"]:
        return _error(f"Only {meta['duration_s']:.1f}s of audio after trimming silence.",
                      "Upload at least 1 second - 4 seconds or more gives a confident result.")

    ref_y, ref_meta = None, None
    if reference is not None and reference.filename:
        try:
            ref_y, ref_meta = audio_io.load(await reference.read(), cfg["audio"]["max_seconds"])
        except audio_io.AudioError as e:
            ref_y, ref_meta = None, {"error": str(e)}

    try:
        result = detector.analyze(y, meta, ref_y, context, cfg)
    except ValueError as e:
        return _error(str(e), "Check that the clip actually contains speech.")

    notes = list(result["confidence"]["notes"])
    if ref_meta and ref_meta.get("error"):
        notes.append(f"Reference ignored: {ref_meta['error']}")
    if ref_meta and not ref_meta.get("error"):
        gap = abs(meta.get("source_sample_rate", 0) - ref_meta.get("source_sample_rate", 0))
        if gap >= cfg["speaker"]["channel_mismatch_hz"] * 2:
            notes.append("Test and reference were captured on very different channels; "
                         "speaker match is depressed by channel mismatch, not necessarily by identity.")
    result["confidence"]["notes"] = notes

    result["meta"] = {
        "model_version": cfg["model_version"],
        "calibration_id": cfg["calibration_id"],
        "file": audio.filename,
        "reference_file": reference.filename if (reference and reference.filename) else None,
        "context": context,
        "language": language,
        "duration_s": meta["duration_s"],
        "source_duration_s": meta["source_duration_s"],
        "sample_rate": meta["sample_rate"],
        "source_sample_rate": meta["source_sample_rate"],
        "codec": meta["codec"],
        "bit_depth": meta["bit_depth"],
        "channels": meta["channels"],
        "peak_dbfs": meta["peak_dbfs"],
        "truncated": meta["truncated"],
        "telephony_leg": meta["upsampled_from_telephony"],
        "window_s": cfg["audio"]["window_seconds"],
        "hop_s": cfg["audio"]["hop_seconds"],
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
        "stored": False,
    }
    return result


app.mount("/samples", StaticFiles(directory=str(SAMPLES)), name="samples")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
