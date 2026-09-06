# PRD: Real-Time Voice Integrity Verification — 1-Week Demo MVP (Upload Web App)

**Version:** 0.2 (1-week judges demo)
**Date:** 2026-09-06
**Source research:** `research/voice-integrity-verification-research.md`
**Goal:** Live demo for judges: upload a voice/audio file in a web app → get spoof risk score + explanation in <30s.

> Change from v0.1: NO live Twilio/Asterisk call tapping. 7-week streaming plan deferred. This is a file-upload prototype that proves the same detection core.

---

## 1. Problem (1 line)

Cloned voices fool caller ID and human ears — we need an upload-and-check tool that flags AI/fake voice before money or secrets move.

## 2. Demo Objective

Build a web app where a judge can:

1. Upload `wav/mp3/m4a` (or record 5–10s on mic).
2. Optionally upload 1 genuine reference sample of the claimed speaker.
3. Click **Analyze** → see in <30s:
   - Risk 0–100 + verdict (Safe / Suspicious / Likely Cloned)
   - Breakdown: spoof score, speaker-match score, prosody flag
   - Waveform + simple reason ("flat pitch, missing micro-variation")
   - Recommended action (Approve / Verify via callback+MFA / Block)
4. Works for English + Hindi. Works offline on a laptop (no Twilio, no cloud key needed for demo).

Success = judge uploads real voice → green, uploads cloned voice → red, understands why.

## 3. Non-Goals (this week)

- No live call tapping (Twilio/Asterisk/SIPREC deferred to post-demo).
- No training from scratch (pretrained models only).
- No gRPC, no Genesys/Cisco/banking CBS integration.
- No auth, no multi-user, no HA/scale.
- Voice never *approves* a transaction — only escalates (RBI rule).

## 4. Users (demo)

| User | What they do |
|---|---|
| Judge | Uploads file, sees verdict |
| Presenter | Picks preloaded samples, narrates story (CXO fraud call) |
| (Future) Bank staff | Same UI embedded as agent banner |

## 5. Success Metrics (judges)

- 6 preloaded demo files all classify correctly: real EN, cloned EN, real HI, cloned HI, replay/noisy, wrong-speaker.
- Analysis time: <30s per file on laptop CPU (target <10s for <15s audio).
- No crash on 5MB mp3, 8kHz phone audio, Hindi code-mix.
- Privacy line on UI: "No audio stored — scores only" + Delete button works.

## 6. Scope — What We Build This Week

### F1. Web App (single page)
- Upload box (drag-drop) + mic record (MediaRecorder → wav) + language dropdown (auto/EN/HI).
- Optional 2nd upload: reference genuine voice.
- Transaction context dropdown: `High-value transfer / Privileged access / Normal call` (changes threshold).
- Results panel: big risk dial, verdict chip, 3 sub-scores with bars, waveform (wavesurfer or plain canvas), reason bullets, action card, JSON expand (`model_version, duration, sr, codec`).
- History list (in-memory + localStorage, Clear button). Preloaded Samples section with 6 one-click files.

### F2. Backend API (FastAPI, 3 endpoints)
- `POST /api/analyze` (multipart: `audio`, optional `reference`, `context`) → `{risk, verdict, spoof_score, match_score, prosody_flag, reasons[], action}`.
- `GET /api/health` → `{models_loaded}`.
- `GET /api/samples` → list of preloaded demo files.
- Audio normalize: ffmpeg → 16kHz mono wav, limit 60s, upsample if 8kHz + tag it.

### F3. Detection Core (pretrained only, no training)
1. **Spoof score (primary):** `clovaai/aasist` pretrained (or AASIST-L if heavy). If AASIST fails to load in time → fallback: wav2vec2-XLSR embeddings + simple logistic head, or even LCNN-LFCC. Pick whichever runs on CPU first — judges care about working demo, not SOTA.
2. **Speaker match (if reference given):** `speechbrain/spkrec-ecapa-voxceleb` cosine similarity → 0–100. No reference → show "skipped".
3. **Prosody flag:** librosa only (pyin F0, jitter/shimmer approx, pause rate, speaking rate). No openSMILE/Praat this week unless 1-line install works. Output: Normal/Flat/Robotic hint.
4. **Fusion:** `risk = 0.7*spoof + 0.2*(1-match) + 0.1*prosody + context_boost`. Thresholds: context-aware (high-value: red ≥65, amber ≥40; normal: red ≥80, amber ≥55). Hardcode, expose in `config.json`.

### F4. Demo Data Pack (must-have)
- `samples/` folder: `real_en.wav, clone_en.wav, real_hi.wav, clone_hi.wav, replay_noisy.wav, wrong_speaker.wav` (record own voices + clone via any TTS/VC tool with consent; keep each 8–15s).
- If cloning tool fails → use ASVspoof 2019 LA eval samples as backup (cite source).

### F5. Privacy/Mock Compliance
- No DB. Temp files deleted after scoring. Scores kept in memory only.
- Banner: "Demo only — feature-only logging, DPDP-friendly" + Delete history button.

## 7. Architecture (1-week)

```
Browser (single page)
 Upload / Record / Preloaded samples
        │  POST /api/analyze (wav, ref?, context)
        ▼
FastAPI
 ffmpeg normalize (→16k mono) → VAD trim (webrtcvad or silero, or skip)
        ├─ AASIST pretrained → spoof 0-100
        ├─ ECAPA (if ref) → match 0-100
        └─ librosa prosody → flag + reasons
        ▼
 Fusion → risk + verdict + action → JSON → UI dial/bars/waveform
```

No Twilio, no streaming WS, no ONNX optimization unless free time.

## 8. Milestones — 7 Days

### Day 1 — Skeleton + Models Load
- FastAPI + static page (upload → dummy score). `ffmpeg` normalize works.
- Load AASIST + ECAPA pretrained, print scores on 2 files in console.
- **Done when:** `curl /api/health` = loaded, 1 real + 1 fake score differ.

### Day 2 — Core Scoring
- Wire `/api/analyze` real: spoof + match (if ref) + librosa prosody + fusion + thresholds.
- **Done when:** 4 local files return sensible risk ordering (real low, clone high).

### Day 3 — Web UI
- Upload/record/context/ref + results (dial, bars, reasons, action, waveform, JSON).
- Preloaded samples one-click. History + Clear.
- **Done when:** end-to-end in browser <30s/file.

### Day 4 — Hindi + Robustness
- Add HI samples, 8kHz phone-audio test, noisy test. Fix crashes (mp3/m4a, long files, silence).
- Tune thresholds + reason strings for judges.
- **Done when:** all 6 demo files pass correctly twice in a row.

### Day 5 — Polish + Story
- CXO-fraud story flow: "₹10L transfer call" context dropdown changes verdict sensitivity.
- UI copy: "Voice matched ≠ safe — always callback+MFA". Privacy banner.
- Record 2-min backup video (in case WiFi/mic fails).
- **Done when:** dry run 5-min script without touching code.

### Day 6 — Buffer + Rehearse
- Edge cases: empty file, >60s file, no-speech file → friendly error.
- Laptop offline test, requirements.txt freeze, README how-to-run.
- **Done when:** fresh clone → `pip install -r requirements.txt; uvicorn app:app` → demo works.

### Day 7 — Judges Demo (no code)
- Only rehearse. Freeze code at noon.

## 9. Dependencies

- Python 3.10+, FastAPI, uvicorn, librosa, soundfile, torch (CPU), transformers/speechbrain, ffmpeg binary.
- Models (download once): `clovaai/aasist` weights, `speechbrain/spkrec-ecapa-voxceleb`, optional Silero VAD.
- 6 demo wavs made by team (consented). Laptop with mic + speakers.

## 10. Risks (1-week)

1. AASIST too heavy/no internet on demo laptop → pre-download weights, keep AASIST-L fallback, test offline Day 1.
2. Pretrained EN model weak on Hindi → set expectations ("Indic fine-tune is roadmap"), ensure HI demo pair still separates (pick clear clone).
3. Mic/file format chaos → ffmpeg normalize + 60s cap + friendly errors.
4. Overclaiming accuracy → UI says "Prototype — not certified; always step-up verify."

## 11. Deferred (tell judges as roadmap)

Live Twilio/Asterisk tap, streaming 0.5s scores, ONNX/edge, IndicWav2Vec A/B, gRPC + bank/SOC integration, full DPDP/RBI audit pack.

---

## Appendix: 5-min Judges Script

1. Story: "CFO gets a call from 'CEO' asking ₹10L transfer."
2. Upload `real_en.wav` + reference → Risk 12 Green → "Proceed with normal MFA."
3. Upload `clone_en.wav` (same script) → Risk 87 Red → "Block + callback to registered number + push approve."
4. Upload `real_hi.wav` vs `clone_hi.wav` → same separation in Hindi.
5. Show breakdown + JSON (`model_version, sr, duration`) → "Scores only, no audio stored." Close with roadmap slide.
