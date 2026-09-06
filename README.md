# Voice Integrity Verification — demo MVP

Drop a call recording, get a **spoof-risk score (0–100)**, the **evidence behind it**, and the
**action to take** — in a couple of seconds, on a laptop, with no internet, no GPU, no ffmpeg
and no database.

Built against `PRD.md` (v0.2, 1-week judges demo) and `research/voice-integrity-verification-research.md`.

```powershell
cd D:\Development\SIH
pip install -r requirements.txt
python -m uvicorn app:app --port 8000
# open http://127.0.0.1:8000
```

---

## What it does

| Leg | Question it answers | How |
|---|---|---|
| **Spoof score** | Is this audio synthetic? | 11 explainable acoustic cues → calibrated logistic fusion |
| **Speaker match** | Is this the enrolled speaker? | cepstral distance in within-clip sigmas + pitch register |
| **Prosody anomaly** | Does the delivery behave like a human? | side-channel only, never a verdict on its own |
| **Policy** | What should the bank/agent do? | context-aware thresholds → proceed / step up / block |

The three legs are fused into one risk number, then compared against thresholds that depend on
what is at stake (`High-value transfer` is stricter than `Normal call`). A **speaker mismatch
escalates on its own** — a weighted average must never be able to average away a hard failure.

### The cues (all visible in the UI, all tunable in `config.json`)

Pitch range · pitch micro-variation (jitter) · amplitude micro-variation (shimmer) · harmonic
purity · energy above 4 kHz · synthetic band ceiling · breathing pauses · syllabic rhythm ·
spectral flatness · channel noise · silence purity.

Each cue reports its measured value, whether it fired, and how much it contributed. Nothing in
the verdict is unexplained.

---

## Two things worth pointing at in a demo

**1. Telephony honesty.** Drop `clone_phone_8k.wav` (a real 8 kHz G.711 µ-law file, exactly what
Twilio Media Streams and Asterisk `externalMedia` hand you). Everything above 4 kHz is gone, so
the high-frequency cues are **disabled rather than guessed**, confidence drops to *medium*, and
the verdict lands on *Suspicious* instead of a confident *Cloned*. The callback still stops the
fraud. This is the research's central warning (ASVspoof 2021 DF: lab EERs collapse ~10× over
real channels) built into the product instead of papered over.

**2. A matching voice is not a safe voice.** Score `clone_en.wav` with `real_en.wav` as the
reference: speaker match ≈ 87/100 — the clone *does* sound like the CEO — while the spoof leg
still says 93. Voice is a signal, never an approval; the RBI 2025 authentication directions
require a dynamically created second factor regardless.

---

## Demo script (5 minutes)

1. **Story** — "The CFO gets a call from the CEO asking for a ₹10 lakh transfer."
2. Drop **CEO — genuine (EN)** → ~15/100, green, *proceed under normal policy*.
3. Drop **CEO — cloned (EN)** → ~93/100, red, *block + callback + push-approve*. Open **Why**:
   flat pitch, no jitter, hard band ceiling at 6 kHz, no breathing pauses.
4. Repeat with the **Hindi** pair → the separation holds outside English.
5. Load **Different real person** as the *reference* against the genuine clip → match 13/100,
   escalated to *Suspicious* even though the audio is a real human.
6. Drop **Clone on an 8 kHz phone leg** → *Suspicious*, confidence *medium*, gated cues shown.
7. Close on **Measurements** + **JSON**: `model_version`, `calibration_id`, codec, and
   `"stored": false`.

---

## Verify it still works

```powershell
python scripts/eval_samples.py
```

Scores every sample, compares against the expected band in `samples/manifest.json`, prints the
speaker-match matrix, and exits non-zero on any regression. Current state:

```
file                     risk  spoof  prosody verdict          expect   conf
clone_en.wav             93.1   92.6     62.8 Likely Cloned    red      high
clone_hi.wav             86.1   83.8     66.3 Likely Cloned    red      medium
clone_phone_8k.wav       51.2   48.3     34.4 Suspicious       amber    medium
real_en.wav              15.1   11.9      0.0 Likely Genuine   green    high
real_hi.wav              15.1   11.9      0.0 Likely Genuine   green    high
replay_noisy.wav         43.3   45.0      0.0 Suspicious       amber    high
wrong_speaker.wav        15.1   11.9      0.0 Likely Genuine   green    high
```

Add `--features` to dump every measured value, or `--context normal` to see the thresholds move.

---

## API

| Endpoint | Returns |
|---|---|
| `GET /api/health` | `models_loaded`, `model_version`, `calibration_id`, accepted formats |
| `GET /api/config` | contexts + thresholds + cue labels (the UI is driven by this) |
| `GET /api/samples` | the demo pack with labels, stories and expected bands |
| `POST /api/analyze` | multipart: `audio`, optional `reference`, `context`, `language` |

`POST /api/analyze` responds with `risk`, `band`, `verdict`, `action`, `escalations[]`,
`spoof_score`, `match_score`, `prosody_flag`, `cues[]` (value, evidence, contribution, gated),
`features{}`, `timeline{}` (3 s window / 0.5 s hop), `waveform[]`, `confidence{}` and `meta{}`.

```powershell
curl.exe -F "audio=@samples/clone_en.wav" -F "context=high_value" http://127.0.0.1:8000/api/analyze
```

**Formats.** The server reads WAV natively — PCM 8/16/24/32-bit, IEEE float, and G.711
µ-law/A-law. Anything else (mp3, m4a, ogg, webm, a microphone recording) is decoded and
re-encoded to 16 kHz mono WAV **in the browser** before upload, so no ffmpeg is needed. WAV files
are passed through untouched, which is what keeps an 8 kHz telephony clip recognisable as one.

---

## Privacy

No database, no temp files, no logging of audio. Clips are decoded in memory, scored, and
dropped; `meta.stored` is `false` on every response. The only thing kept is the score list in
your own browser's `localStorage`, and **Delete session history** removes it. That is the
DPDP-2023 data-minimisation posture from the research, not a slogan on a banner.

---

## Layout

| Path | What |
|---|---|
| `app.py` | FastAPI: HTTP contract only, no scoring logic |
| `audio_io.py` | bytes → 16 kHz mono float32 (PCM / float / G.711), resample, VAD-lite trim |
| `detector.py` | features → cues → calibrated spoof score → fusion → policy |
| `config.json` | every threshold, weight, action string and the `calibration_id` |
| `static/` | `index.html`, `styles.css`, `app.js` — landing is a single drop surface |
| `samples/` | 7 synthetic demo clips + `manifest.json` (labels, stories, expected bands) |
| `scripts/make_samples.py` | source-filter speech synthesiser that builds the demo pack |
| `scripts/eval_samples.py` | regression gate over the whole pack |
| `PRD.md` | product spec and 7-day plan |
| `research/` | primary-source research the design traces to |

### Regenerate the demo pack

```powershell
python scripts/make_samples.py
```

The clips are synthesised, not recorded: the "genuine" ones get a moving F0 contour, cycle-to-cycle
jitter and shimmer, aspiration noise, fricative energy above 4 kHz, breathing pauses and a room
noise floor; the "cloned" ones get the same voice with smoothed pitch, no micro-variation, clean
digital silence and a hard 6 kHz vocoder ceiling. **No real voices were recorded or cloned.**
Drop a consented recording into `samples/` and the app picks it up — add an entry to
`manifest.json` to give it a label and an expected band.

---

## Where the real models plug in

The scoring seams are deliberately narrow:

- `detector.frame_features` / `evaluate_cues` → replace with **wav2vec2-XLSR → AASIST** logits
  (research §1.3: the SSL front-end is what survives channel shift; AASIST-L is 85 K params).
- `detector.speaker_embedding` / `speaker_match` → replace with **ECAPA-TDNN**
  (`speechbrain/spkrec-ecapa-voxceleb`, cosine on 192-d embeddings).

Nothing in `app.py`, `config.json` or the UI changes: the response contract, the cue list, the
thresholds and the policy engine already have the shape those models need. Roadmap beyond that —
live Twilio/Asterisk tap with 0.5 s streaming scores, ONNX/edge inference, IndicWav2Vec A/B for
Indic legs, and a collected Hindi clone-eval set (the research found **no** primary-source Indian
spoof corpus — that gap is real and has to be filled with consented data).

---

## Honest limits

- The detector is an **explainable heuristic baseline**, not a trained countermeasure. It has not
  been evaluated on ASVspoof; no EER or min t-DCF is claimed, and none should be quoted.
- Cue thresholds are calibrated against the synthetic demo pack. Real recordings — especially
  narrowband, noisy or heavily compressed ones — will need re-calibration, which is exactly why
  `calibration_id` ships in every response.
- A determined attacker who adds jitter, breath noise and pauses to a clone defeats these cues.
  That is the argument for the SSL/AASIST upgrade, not a reason to trust the heuristic further.
