# Voice Integrity Verification Framework — Primary-Source Research

> Topic: AI-powered, real-time voice authenticity verification for VoIP/mobile/enterprise calls
> (anti voice-cloning / deepfake fraud; CXO/government impersonation; high-value transaction protection).
> Method: primary sources only (challenge sites + eval plans, arXiv first-author papers,
> IETF RFCs, first-party product docs, official GitHub repos, MeitY/RBI/TRAI official texts).
> Items not verified against a primary source are explicitly flagged **[UNVERIFIED]**.

---

## Executive Summary

- **Detection SOTA moved from hand-crafted cepstra → raw-waveform nets → graph nets → self-supervised (SSL) front-ends.**
  ASVspoof 2019 baselines were LFCC-GMM / CQCC-GMM
  ([eval plan](https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf));
  RawNet2 showed raw-waveform nets learn complementary cues, especially on worst-case attack A17
  ([paper](https://arxiv.org/abs/2011.01108));
  AASIST set the single-system SOTA on 2019 LA at **0.83% EER / 0.0275 min t-DCF** (best seed)
  with a lightweight 85K-param variant at **0.99% EER**
  ([paper](https://arxiv.org/abs/2110.01200), [code](https://github.com/clovaai/aasist));
  wav2vec2-XLSR + augmentation later drove ASVspoof 2021 LA to **0.82% EER** and DF to **2.85% EER**
  ([paper](https://ar5iv.labs.arxiv.org/html/2202.12233)).
- **Telephony reality erases much of that lab accuracy.**
  ASVspoof 2021 (codecs/transmission/compression, no new matched training data) best LA was only
  **0.2177 min t-DCF / 1.32% EER** and best DF **15.64% EER**
  ([overview](https://www.eurecom.edu/publication/6675/download/sec-publi-6675.pdf));
  ASVspoof 5 (crowdsourced MLS data, >4k speakers, attacks optimised against surrogate CMs,
  Malafide/Malacopula adversarial filters, neural codecs) stresses this further
  ([paper](https://arxiv.org/abs/2408.08739)).
  A same-condition re-implementation of AASIST on ASVspoof 5 reports **27.58% EER**,
  dropping to **7.66%** only after a frozen wav2vec2 front-end + learnable fusion
  ([paper](https://arxiv.org/html/2507.11777)) — i.e. **SSL front-end is load-bearing for robustness**.
- **Real-time ingestion is a solved plumbing problem with hard audio constraints.**
  WebRTC mandates Opus + G.711 PCMA/PCMU ([RFC 7874](https://datatracker.ietf.org/doc/html/rfc7874)),
  RTP runs over SRTP/SAVPF ([RFC 8834](https://www.rfc-editor.org/info/rfc8834/)),
  default packetisation is **20 ms** ([RFC 3551](https://www.rfc-editor.org/rfc/rfc3551.html)).
  Twilio Media Streams fork call audio over WebSocket as **base64 `audio/x-mulaw`, 8 kHz**
  ([docs](https://www.twilio.com/docs/voice/media-streams/websocket-messages));
  Asterisk ARI `/channels/externalMedia` emits **RTP/UDP** in a caller-chosen format
  (`ulaw`, `slin16`, `g722`, …) into a bridge ([docs](https://docs.asterisk.org/Development/Reference-Information/Asterisk-Framework-and-API-Examples/External-Media-and-ARI/)),
  with a newer raw-over-WebSocket driver ([docs](https://docs.asterisk.org/Configuration/Channel-Drivers/WebSocket/)).
  **Design consequence: expect 8 kHz µ-law on PSTN paths; upsample to 16 kHz for models, and budget
  for the 8 kHz information loss (no content above 4 kHz).**
- **Speaker-consistency layer has a canonical stack.**
  ECAPA-TDNN (SE-Res2Blocks + channel-attentive pooling) improved ~**19% relative EER**
  over x-vector baselines on VoxCeleb/VoxSRC-2019
  ([paper](https://www.isca-archive.org/interspeech_2020/desplanques20_interspeech.pdf));
  SpeechBrain ships it as `ECAPA_TDNN` ([API docs](https://speechbrain.readthedocs.io/en/latest/API/speechbrain.lobes.models.ECAPA_TDNN.html))
  with a VoxCeleb pretrained verifier ([model card](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)).
  ASVspoof 5's own surrogate ASV is **ECAPA-TDNN + PLDA**
  ([paper](https://arxiv.org/abs/2408.08739)) — use cosine first, PLDA/score-norm per
  SpeechBrain recipes ([repo](https://github.com/speechbrain/speechbrain)).
- **Prosody/behavioural features are measurable with three mature libs, none is a detector by itself.**
  openSMILE eGeMAPSv02 exposes F0, jitter, shimmer, HNR, formants as LLDs/functionals
  ([usage](https://audeering.github.io/opensmile-python/usage.html));
  Parselmouth binds Praat exactly (`To Pitch`, `To PointProcess (periodic, cc)`,
  `Get jitter/shimmer…`) ([repo](https://github.com/YannickJadoul/Parselmouth));
  librosa provides `pyin`/`yin`/`piptrack` pitch APIs ([API](https://librosa.org/doc/main/api/index.html));
  SpeechBrain's voice-analysis tutorial cross-checks all three on jitter/shimmer/HNR
  ([tutorial](https://speechbrain.readthedocs.io/en/latest/tutorials/preprocessing/voice-analysis.html)).
- **Indian-multilingual path exists but is ASR-shaped, not spoof-shaped.**
  IndicWav2Vec covers **40 Indian languages**, SOTA ASR on 9 langs over MUCS/MSR/OpenSLR
  ([project](https://indicnlp.ai4bharat.org/indicwav2vec/), [repo](https://github.com/AI4Bharat/IndicWav2Vec));
  Bhashini/ULCA exposes ASR/NMT/TTS models and pipelines via Search → Config → Compute APIs
  ([API docs](https://bhashini.gitbook.io/bhashini-apis), [ULCA repo](https://github.com/bhashini-dibd/ulca));
  Common Voice Hindi v26 is only **26.67 h (15.6 h validated), 480 speakers**
  ([datasheet](https://mozilladatacollective.com/datasets/cmqiod71900zgnr07uiyw57br)).
  **No primary source found for an Indian-language spoof/deepfake corpus — treat as a build-it gap.**
- **Compliance envelope is now explicit.**
  DPDP Act 2023 (No. 22 of 2023, assent 11 Aug 2023) sets consent/purpose-limitation/minimisation/
  security-safeguards duties with penalties to **₹250 cr**
  ([Act PDF](https://www.meity.gov.in/static/uploads/2024/06/2bf1f0e9f04e6fb4f8fef35e82c42aa5.pdf));
  DPDP Rules 2025 notified **14 Nov 2025**, operationalising the Act
  ([PIB](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2190655));
  RBI's 2025 Authentication Directions require **≥2 factors, ≥1 dynamically created**, compliance by
  **1 Apr 2026** ([RBI](https://www.rbi.org.in/scripts/BS_ViewMasDirections.aspx?id=12898));
  TRAI's CNAP recommendation (23 Feb 2024) would present CAF-based caller name to blunt
  spoofed-identity answering, but notes crowd-sourced ID is unreliable and scammer opt-out is the failure mode
  ([TRAI PDF](https://www.trai.gov.in/sites/default/files/2024-11/Recommendation_23022024_0.pdf)).
- **Recommended architecture (actionable):** tap via Twilio Streams / Asterisk External Media (or SIPREC **[UNVERIFIED]**);
  20 ms-frame VAD-gated streaming front-end; **frozen XLSR/wav2vec2 → AASIST-style back-end** as primary CM
  (distil/quantise to ONNX for edge; ONNX Runtime runs cloud/edge/mobile/web
  ([docs](https://onnxruntime.ai/docs/)));
  parallel **ECAPA-TDNN embedding vs enrolled voiceprint** for cross-session consistency;
  openSMILE/Parselmouth prosody side-channel; score-level fusion → continuous risk score with
  transaction-aware thresholds → step-up (MFA/callback) per RBI two-factor rule; feature-only logging,
  minimal retention per DPDP.

---

## 1. Detection Models & Datasets

### 1.1 Challenges: what they prove

- **ASVspoof 2019** ([site](http://www.asvspoof.org/index2019.html)):
  LA (TTS/VC, 19 attacks A01–A19, VCTK-based, 16 kHz) + PA (simulated replay);
  primary metric **min t-DCF**, secondary **EER**
  ([eval plan](https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf)).
  Baselines (dev): LA LFCC-GMM **0.0663 / 2.71%**, CQCC-GMM **0.0123 / 0.43%**;
  PA LFCC-GMM **0.2554 / 11.96%**, CQCC-GMM **0.1953 / 9.87%** (same source).
  Database paper confirms VCTK source (107 speakers, hemi-anechoic, 96 kHz→16 kHz),
  disjoint train/dev/eval attacks, and human-listening evidence that some spoofs are
  indistinguishable to people ([paper](https://doi.org/10.1016/j.csl.2020.101114)).
- **Top 2019 systems** ([results](https://doi.org/10.1109/tbiom.2021.3059479)):
  LA best primary T05 **0.0692 / 0.22%** (≥5-system fusion, mixed cepstral+spectral + DNN);
  best single T45 **0.1562 / 5.06%** (LFCC + LCNN).
  PA best primary T28 **0.1437 / 0.39%**, best single T28 **0.1470 / 0.52%**
  (spectral + DNN; fusion barely helped PA).
  Worst-case attack dominated: LA **A17**; fusion is what closed the gap.
- **ASVspoof 2021** ([site](http://www.asvspoof.org/index2021.html),
  [eval plan](https://www.asvspoof.org/asvspoof2021/asvspoof2021_evaluation_plan.pdf)):
  no new train/dev; new eval with LA (codec/transmission), PA (real rooms), DF (compression, >100 attacks,
  no ASV → EER metric). Baselines added LFCC-LCNN + RawNet2 alongside GMMs (same eval plan).
  Results ([overview](https://www.eurecom.edu/publication/6675/download/sec-publi-6675.pdf)):
  LA best **0.2177 min t-DCF / 1.32% EER**; DF best T23 **15.64%**, T20 **16.05%**
  vs best baseline B04 **22.38%** — i.e. **channel/compression collapses lab EERs by ~10×**.
- **ASVspoof 5 (2024)** ([paper](https://arxiv.org/abs/2408.08739),
  [ISCA](https://www.isca-archive.org/asvspoof_2024/wang24_asvspoof.html),
  [eval plan](https://www.asvspoof.org/file/ASVspoof5___Evaluation_Plan.pdf),
  [data](https://doi.org/10.5281/zenodo.14498691)):
  source = MLS English (**>4k speakers**, device-diverse) vs VCTK ~100 anechoic;
  attacks A01–A32 (GlowTTS/GradTTS/FastPitch/VITS/ToucanTTS/Tacotron2/YourTTS/XTTS/StarGANv2-VC/DiffVC/…)
  **optimised against surrogate ASV+CM**, plus **Malafide** (anti-CM) and **Malacopula** (anti-ASV)
  adversarial filters, plus codecs incl. neural. Surrogates: **ECAPA-TDNN+PLDA** ASV;
  CMs **AASIST, RawNet2, LFCC-LCNNs, wav2vec2-XLSR-53**. Two tracks: spoof-robust SASV + standalone CM.
  Key quote: baseline CMs (recent SOTA) perform "relatively poorly" but submissions beat them —
  and **score calibration** is flagged as a deployment issue (same paper).

### 1.2 Leading models (features, EERs, code)

| Model | Input / idea | Reported result (2019 LA unless noted) | Primary source |
|---|---|---|---|
| CQCC-GMM / LFCC-GMM baselines | CQCC or LFCC + GMM | CQCC **0.43% EER** dev; LFCC 2.71% dev | [eval plan](https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf) |
| LCNN (T45 single) | LFCC + LCNN | **5.06% EER / 0.1562 t-DCF** | [results](https://doi.org/10.1109/tbiom.2021.3059479) |
| RawNet2 | raw waveform → sinc-conv + ResBlocks + GRU + FMS | pooled worse than LFCC baseline (0.1175 vs 0.09 t-DCF) but **A17 0.181 vs 0.3524**; fusion L+S1 **1.12% EER / 0.0330** | [paper](https://arxiv.org/abs/2011.01108), [code](https://github.com/eurecom-asp/rawnet2-antispoofing) |
| AASIST | raw → RawNet2-like encoder + heterogeneous spectro-temporal graph attention (HS-GAL) + max-graph op | **0.83% EER / 0.0275 t-DCF** best seed (avg 1.13%/0.0347); **AASIST-L 85,306 params: 0.99%/0.0309** | [paper](https://arxiv.org/abs/2110.01200), [code](https://github.com/clovaai/aasist), [IEEE](https://ieeexplore.ieee.org/document/9747766) |
| wav2vec2-XLSR + SA + augmentation | frozen/fine-tuned XLSR front-end + self-attentive pooling + RawBoost-style DA | 2021 LA **0.82% EER** (best; 1.00 avg), 2021 DF **2.85%** (3.69 avg); ~90%/88% relative over sinc baseline | [paper](https://ar5iv.labs.arxiv.org/html/2202.12233), [slides](https://yamagishilab.jp/wp-content/uploads/2022/06/Speaker_odyssey_22_W2V2_anti_spoofing.pdf) |
| XLSR-Mamba (recent) | XLSR + dual-column BiMamba | 2021 LA **0.93%**, DF **1.88%**; In-the-Wild **6.71%**; faster than Transformer | [paper](https://arxiv.org/html/2411.10027v2) |
| AASIST re-examined on ASVspoof 5 | frozen vs trainable SSL + MHA + learnable fusion | vanilla AASIST **27.58%** → frozen SSL **8.76%** → full mods **7.66%** (constrained-training re-implementation, not official SOTA) | [paper](https://arxiv.org/html/2507.11777) |

- **Features lesson:** hand-crafted LFCC/CQCC/MFCC win in clean 2019-LA; raw-waveform and SSL
  front-ends win on worst-case/unseen/channel-shifted attacks (RawNet2 on A17; wav2vec2 on 2021).
  Phase/timing artefacts (the A17 "clicking", hypothesised phase-related) motivate linear-phase
  sinc front-ends ([RawNet2 paper](https://arxiv.org/abs/2011.01108)).
  MFCC-vs-LFCC-vs-CQCC ablations live in the eval-plan baselines and T05/T45 system descriptions —
  cite per-system, not as a universal ranking.
- **Streaming feasibility: [PARTIALLY UNVERIFIED].**
  Verified: AASIST-L is 85K params / ~332 KB ([paper](https://arxiv.org/abs/2110.01200));
  Silero-VAD ONNX runs **<1 ms/chunk**, same order as WebRTC VAD
  ([FAQ](https://github.com/snakers4/silero-vad/wiki/FAQ));
  ONNX Runtime targets cloud/edge/mobile/web ([docs](https://onnxruntime.ai/docs/)).
  Not verified against a primary source: end-to-end AASIST/wav2vec2 frame-wise latency on your
  hardware — **must benchmark**: window (recommend 2–4 s sliding, 0.5 s hop), RTF, and 8 kHz degradation.

### 1.3 What to build on

1. Primary CM: **fine-tuned-or-frozen XLSR/wav2vec2 front-end + AASIST-style graph/Mamba back-end**,
   trained on 2019 LA + 2021 LA/DF augmentations (codec/transmission/compression), evaluated on
   ASVspoof 5-style adversarial splits.
2. Auxiliary fast CM: **AASIST-L / LCNN-LFCC** ONNX for per-chunk gating where the big model is too slow.
3. Always report **pooled + worst-attack EER and min t-DCF**; 2019 showed pooled hides A17-class failures.

---

## 2. Real-Time & Telephony Constraints

- **WebRTC audio (browser/enterprise-collab path):** endpoints REQUIRED: **Opus + PCMA/PCMU (G.711)**,
  Opus preferred above 8 kHz; comfort-noise/DTMF rules
  ([RFC 7874](https://datatracker.ietf.org/doc/html/rfc7874)).
  Media transport is **RTP/RTCP, profile RTP/SAVPF**, multiplexing rules
  ([RFC 8834](https://www.rfc-editor.org/info/rfc8834/)).
- **RTP framing:** default packetisation **20 ms** (or one frame); receivers SHOULD accept 0–200 ms
  ([RFC 3551](https://www.rfc-editor.org/rfc/rfc3551.html)).
  Opus itself spans 8–48 kHz ([same RFC 7874](https://datatracker.ietf.org/doc/html/rfc7874)).
- **PSTN/VoIP ingest (bank/contact-centre path):**
  - Twilio: fork via `<Start><Stream>` (unidirectional, tracks inbound/outbound/both) or
    `<Connect><Stream>` (bidirectional, inbound only); audio over **secure WebSocket as JSON
    `connected/start/media/dtmf/stop/mark`**, media payload **base64 `audio/x-mulaw`, 8 kHz**
    ([overview](https://www.twilio.com/docs/voice/media-streams),
    [messages](https://www.twilio.com/docs/voice/media-streams/websocket-messages),
    [TwiML](https://www.twilio.com/docs/voice/twiml/stream),
    [tutorial](https://www.twilio.com/docs/voice/tutorials/consume-real-time-media-stream-using-websockets-python-and-flask)).
    Unidirectional fork budget shared (max 4 tracks with SIPREC/transcription/AMD) (same overview).
  - Asterisk: ARI `POST /channels/externalMedia` creates a channel in a bridge that forwards
    media as **RTP/UDP** (initial release) in the requested `format` (`ulaw`, `g722`, …) with
    auto-transcode; media server re-encapsulates for the cloud provider
    ([docs](https://docs.asterisk.org/Development/Reference-Information/Asterisk-Framework-and-API-Examples/External-Media-and-ARI/),
    [announcement](https://www.asterisk.org/external-media-a-new-way-to-get-media-in-and-out-of-asterisk/),
    [sample](https://github.com/asterisk/asterisk-external-media)).
    Newer `chan_websocket` sends **raw media in binary WS frames** (e.g. 160 B/20 ms for µ-law),
    no RTP parsing, no self-timing ([docs](https://docs.asterisk.org/Configuration/Channel-Drivers/WebSocket/)).
  - FreeSWITCH `mod_audio_fork` / SIPREC: **[UNVERIFIED]** — not checked against a primary source;
    treat as integration backlog, same pattern as above.
- **8 kHz vs 16 kHz (design-driving):** Twilio path is fixed **8 kHz µ-law** (verified above);
  all ASVspoof training data here is **16 kHz** (2019 DB paper). So: VAD + upsample 8→16 kHz at ingest,
  train/evaluate with an explicit 8 kHz/G.711/opus-codec augmentation arm (2021-LA/DF methodology),
  and expect DF-style degradation on PSTN legs. Wideband (Opus/G.722) legs keep more high-frequency
  artefacts — route model variant by negotiated codec where visible.
- **Latency budget (honest accounting):** verified primitives are 20 ms RTP frames, Twilio
  "near real-time" streams, Asterisk 20 ms µ-law frames. End-to-end alerting budget
  (VAD + window + inference + fusion + policy) is **not specified by any fetched primary source** —
  set your own SLO (e.g. first score ≤2–3 s, refresh every 0.5 s) and measure; streaming-architecture
  latency claims (chunked AASIST, Emformer-style CMs) are **[UNVERIFIED]** for this report.

---

## 3. Speaker Consistency (Verification vs Spoof Detection)

- **Two different questions, both needed:** CM = "is this audio synthetic?" (bona fide vs spoof);
  ASV = "is this the enrolled speaker?" (target vs non-target). ASVspoof's t-DCF exists precisely to
  score the tandem ([2019 eval plan](https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf));
  ASVspoof 5 adds a dedicated SASV track for integrated systems
  ([paper](https://arxiv.org/abs/2408.08739)).
- **ECAPA-TDNN** (Desplanques et al., Interspeech 2020): 1-D SE-Res2Blocks + multi-layer aggregation +
  channel-dependent attentive statistics pooling; **~19% mean relative EER gain** over x-vector baselines
  on VoxCeleb + VoxSRC-2019 ([paper](https://www.isca-archive.org/interspeech_2020/desplanques20_interspeech.pdf),
  arXiv [2005.07143](https://arxiv.org/abs/2005.07143)).
- **First-party implementation:** SpeechBrain `ECAPA_TDNN(...)` (channels/kernels/dilations/attention documented)
  ([API](https://speechbrain.readthedocs.io/en/latest/API/speechbrain.lobes.models.ECAPA_TDNN.html));
  speaker-recognition recipes list ECAPA-TDNN / ResNet / x-vectors / PLDA / score-norm
  ([repo](https://github.com/speechbrain/speechbrain));
  pretrained `spkrec-ecapa-voxceleb` (VoxCeleb1+2, additive-margin softmax, cosine scoring)
  ([model card](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)).
- **Cross-session consistency design (actionable):**
  1. Enrol: ≥3–5 clean utterances per protected persona (CXO/official); store **embeddings only**, never audio.
  2. Runtime: sliding-window ECAPA embeddings → cosine vs enrolment centroid; PLDA + adaptive
     score normalisation per SpeechBrain recipe for the bank-grade path.
  3. Fuse with CM score: `risk = f(CM_spoof, 1−ASV_match, prosody_drift, context)`; high-value
     transactions escalate on either leg (spoof-likely OR voice-mismatch).
  4. Re-enrolment hygiene: time-decayed centroid + drift alerts (new-device/illness vs slow cloning drift).
  WeRTVAF: **[UNVERIFIED]** — named in the brief but no primary source fetched; do not cite it in the build.

---

## 4. Prosody / Behavioral Features

What the libs actually expose (verified):

- **openSMILE (eGeMAPSv02):** Functionals include `F0semitone…`, `jitterLocal_sma3nz`,
  `shimmerLocaldB_sma3nz`, `HNRdBACF_sma3nz`, loudness, spectral flux, MFCC1–4, F1–F3
  frequency/bandwidth/amplitude, alphaRatio, Hammarberg index, slopes, voiced-segment stats;
  LLD level gives the same frame-by-frame ([usage](https://audeering.github.io/opensmile-python/usage.html),
  [repo](https://github.com/audeering/opensmile-python/blob/main/README.rst)).
- **Praat via Parselmouth:** exact Praat algorithms in Python — `To Pitch`, `To PointProcess (periodic, cc)`,
  `Get jitter (local/rap/ppq5/ddp)`, `Get shimmer (local/dB/apq3/apq5/apq11/dda)`, harmonicity, Burg formants
  ([repo](https://github.com/YannickJadoul/Parselmouth)).
- **librosa:** `pyin`, `yin`, `piptrack`, tuning estimators; spectral/rhythm/manipulation stack
  ([features](http://librosa.org/doc/latest/feature.html), [API](https://librosa.org/doc/main/api/index.html)).
- **Cross-validation:** SpeechBrain's voice-analysis tutorial computes F0/jitter/shimmer/HNR three ways
  (native vs Praat vs openSMILE) and plots agreement/disagreement (incl. HNR mismatch caveat)
  ([tutorial](https://speechbrain.readthedocs.io/en/latest/tutorials/preprocessing/voice-analysis.html)).
- **Design use (defensible, narrow):** microvariation stats (jitter/shimmer/HNR distributions), F0
  mean/variance/slope histograms, pause/voiced-segment rates, speaking-rate drift vs enrolment baseline —
  as a **side-channel anomaly score**, not a standalone detector. Cloned speech increasingly matches
  coarse prosody; the residual signal is in fine temporal statistics and cross-session drift.
  Any "prosody detects clones at X%" claim without a cited attack set is **[UNVERIFIED]** — do not make it.

---

## 5. Indian Multilingual Landscape

- **AI4Bharat IndicWav2Vec:** multilingual speech model pretrained on **40 Indian languages**;
  fine-tuned ASR for 9 languages, reported SOTA on **MUCS, MSR, OpenSLR**
  ([project](https://indicnlp.ai4bharat.org/indicwav2vec/),
  [repo](https://github.com/AI4Bharat/IndicWav2Vec)).
  Action: candidate **language-robust front-end** (replace XLSR with IndicWav2Vec in the CM stack;
  test language-agnostic vs language-specific heads) — evidence for ASR transfer, **spoof-transfer
  unproven, must A/B**.
- **Bhashini / ULCA:** mission data+models platform for Indic ASR/MT/TTS/OCR
  ([ULCA repo](https://github.com/bhashini-dibd/ulca));
  integration = Pipeline **Search → Config → Compute** calls with `userID`/`ulcaApiKey`,
  pipeline IDs binding ASR/NMT/TTS service IDs ([API docs](https://bhashini.gitbook.io/bhashini-apis)).
  Action: use for **language ID + transcription context** (what was said, in which language),
  not for spoof scoring; note cloud-API round-trip vs on-device privacy trade-off (§6).
- **Common Voice:** 130+ languages, scripted/spontaneous/code-switch tracks
  ([site](https://commonvoice.mozilla.org/hi), [releases](https://github.com/common-voice/cv-dataset));
  Hindi v26.0: **19,029 clips, 26.67 h total (15.6 h validated), 480 speakers, 42,169 sentences**, CC0
  ([datasheet](https://mozilladatacollective.com/datasets/cmqiod71900zgnr07uiyw57br)).
  Action: fine-tunejebn ASV enrolment and prosody priors per language; **insufficient alone** for CM training.
- **Gaps (explicit):** no primary-source Indian-language **spoof/deepfake** corpus found;
  no verified language-agnostic-vs-specific CM comparison for Indic accents/dialects;
  IndicTTS coverage not verified in this pass (**[UNVERIFIED]** — check AI4Bharat/IndicTTS repo directly
  before citing). **Build plan must include collecting Hindi/Tamil/Bengali/Telugu/Marathi (+ code-mixed)
  bona fide + cloned-voice eval sets with consent, or the "multilingual" claim is untested.**

---

## 6. Privacy / Compliance / Platform Integration

### 6.1 DPDP Act 2023 + Rules 2025 (what the official texts say)

- The **Digital Personal Data Protection Act, 2023 (No. 22 of 2023)**, assent **11 Aug 2023**,
  governs processing of digital personal data balancing individual rights with lawful-purpose processing
  ([official PDF](https://www.meity.gov.in/static/uploads/2024/06/2bf1f0e9f04e6fb4f8fef35e82c42aa5.pdf)).
  Core duties relevant here: consent + transparent notice, **purpose limitation, data minimisation,
  accuracy, storage limitation, reasonable security safeguards, accountability** (PIB summary of the framework
  ([PIB](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2190655)) — verify operative clauses in the Act PDF itself).
  Rights include correction/completion/updating/erasure on request and grievance redressal before the
  Data Protection Board (same Act PDF). Penalties reach **₹250 cr** (security-safeguard failure),
  **₹200 cr** (breach-notification / children's-obligations failures), **₹50 cr** residual
  (same Act PDF, Schedule).
- **DPDP Rules, 2025** notified **14 Nov 2025**, operationalising the Act (Board, consent-manager,
  breach-notification mechanics) ([PIB](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2190655)).
  **Build implication:** consent screens + purpose-scoped enrolment, erasure workflow, breach-notification
  path, and a records posture must be in the MVP — penalties make "log everything" unlawful by default.
- Privacy-by-design mapping: **on-device/edge inference** (embeddings + CM scores never leave the device
  unless risk-escalated); **feature-only logging** (scores, codec, language-ID, no raw audio);
  **minimal retention** with auto-expiry; raw audio retained only on explicit step-up consent with separate TTL.

### 6.2 RBI fraud / authentication constraints

- **RBI (Authentication mechanisms for digital payment transactions) Directions, 2025:**
  all domestic digital payments need **≥2 distinct factors**, **≥1 dynamically created/proven per transaction**;
  issuers may add **behavioural/contextual risk checks** (location, behaviour, device, history) and extra checks
  beyond the minimum; issuers must ensure **robustness/integrity pre-deployment** and compensate losses from
  non-compliant transactions; DPDP adherence required; compliance by **1 Apr 2026**
  ([RBI](https://www.rbi.org.in/scripts/BS_ViewMasDirections.aspx?id=12898)).
  **Build implication:** voice-risk score is the *contextual/behavioural* input; the *second factor* must still be
  dynamic (OTP-independent: push-sign, FIDO, callback to registered device). A "voice says OK, skip MFA" flow
  would violate the Directions — gate high-value transactions the other way (voice-risk → mandatory step-up).
- Fraud-risk reporting (FMR/CFR, LEA reporting ≥₹1 lakh) regime exists under the 2026 Fraud Master Directions
  per secondary write-up — **[UNVERIFIED]** against RBI primary text in this pass; confirm in
  `rbi.org.in` master directions before wiring auto-reporting.

### 6.3 TRAI / telecom call-verification context

- **CNAP recommendations (23 Feb 2024):** present CAF-based calling name on every call; tackles answering
  unknown numbers, UCC/robocall/fraud; explicitly finds **crowd-sourced ID unreliable** and warns scammers
  opting out defeats optional deployment ([TRAI PDF](https://www.trai.gov.in/sites/default/files/2024-11/Recommendation_23022024_0.pdf),
  [press release](https://trai.gov.in/notifications/press-release/trai-releases-recommendations-introduction-calling-name-presentation)).
  DoT back-reference under discussion Oct 2025 (TRAI site).
  **Build implication:** treat CNAP/CLI as **untrusted context features** (they attest the signalling, not the voice);
  voice-integrity must assume CLI/CNAM can be spoofed — hence the product.

### 6.4 Edge / on-device inference (verified options)

- **ONNX Runtime:** cross-platform accelerator (PyTorch/TF/TFLite/scikit inputs), Python/C++/C#/C/Java,
  cloud/edge/mobile/web targets ([docs](https://onnxruntime.ai/docs/), [inference](https://onnxruntime.ai/inference));
  Extensions add audio pre/post-processing custom ops ([extensions](https://onnxruntime.ai/docs/extensions/)).
- **VAD-level precedent:** Silero VAD ONNX supports mobile/edge/ARM (16 kHz), runs **<1 ms/chunk**
  ([FAQ](https://github.com/snakers4/silero-vad/wiki/FAQ)) — use as the gating model; WebRTC VAD specifics
  not re-verified here (py-webrtcvad repo check is backlog).
- TensorFlow Lite / LiteRT: **[UNVERIFIED]** in this pass — confirm at `ai.google.dev/edge` before committing.

### 6.5 Integration APIs (verified patterns + explicit gaps)

- Verified streaming patterns to copy: Twilio WS JSON contract (above), Asterisk ARI externalMedia + raw-WS
  driver (above), Bhashini Search→Config→Compute REST (above). From these, the platform API is:
  `POST /v1/sessions` (enrol/attach policy) → `wss /v1/stream` (20 ms frames in, `{score, risk, action}`
  out every 0.5 s) → `POST /v1/events` (alerts) + webhooks (`risk.high`, `voice.mismatch`) → SIEM/bank-workflow.
  gRPC bidirectional streaming for the same contract: **[UNVERIFIED]** — design mirrors the WS contract;
  confirm against [grpc.io](https://grpc.io/docs/) at build time (not fetched here).
- Contact-centre (Genesys/Cisco/Avaya) and core-banking (CBS/switch) integration constraints:
  **[UNVERIFIED]** — no first-party docs fetched; backlog: SIPREC/RTP-fork support matrices, recording-consent
  APIs, agent-desktop alert widgets, and CBS transaction-hold/approve callback SLAs.

---

## 7. Architecture Recommendations for This Project

**Reference pipeline (each choice traces to §1–§6):**

```
PSTN/SIP/WebRTC ─┬─ Twilio <Stream> (8k mulaw WS) ──────┐
                 └─ Asterisk externalMedia (RTP/WS) ────┤
                                                       ▼
                                              Ingest GW (codec-aware)
                                              8k→16k upsample, 20 ms frames
                                                       ▼
                                              VAD gate (Silero-ONNX, <1ms)
                                                       ▼
            ┌──────────────────────┬───────────────────┴───────────────────┐
            ▼                      ▼                                       ▼
   CM stream (primary)     ASV consistency                        Prosody side-channel
   XLSR/IndicWav2Vec       ECAPA-TDNN vs                          openSMILE eGeMAPS +
   frozen → AASIST-style   enrolment centroid                     Praat jitter/shimmer/
   back-end (ONNX)         (cosine → PLDA)                        HNR/pause stats
            └──────────────────────┴───────────────────┬───────────────────┘
                                                       ▼
                                              Fusion → continuous risk 0–100
                                              (calibrated; cf. ASVspoof5 §1.1)
                                                       ▼
                                              Policy engine (txn-aware thresholds)
                                              low: log ▸ mid: banner+SMS/email ▸
                                              high: block + MFA/callback (RBI §6.2)
                                                       ▼
                                              Feature-only store (DPDP §6.1 TTLs)
```

1. **Ingest:** implement Twilio + Asterisk paths first (both verified); negotiate/record codec per leg;
   tag every frame with `codec/sr` for the model router.
2. **Models:** ship (a) full CM = frozen SSL (XLSR; A/B IndicWav2Vec for Indic legs) + AASIST/Mamba head,
   (b) light CM = AASIST-L/LCNN-LFCC ONNX for fast gating, (c) ECAPA-TDNN verifier,
   (d) prosody anomaly scorer. Train arm 1: 2019 LA; arm 2: +2021 codec/compression DA; arm 3: +Indic
   bona fide/clone eval (to be collected, §5). Report pooled **and** worst-attack metrics always.
3. **Risk engine:** sliding 2–4 s windows, 0.5 s refresh; hysteresis + cooldown to avoid alert fatigue;
   contextual enrichment (CNAP/CLI as untrusted features, txn amount/payee-risk, device/history);
   calibration step mandatory (ASVspoof 5 lesson).
4. **Alerting/UX:** pre-transaction warnings in-app + SMS/email fallback; step-up = dynamic second factor
   or callback to registered number (never the CLI of the suspect call); configurable playbooks per
   vertical (bank / enterprise / government).
5. **Privacy:** edge-first inference; embeddings+scores only; raw audio only with explicit step-up consent
   and short TTL; erasure + grievance endpoints per DPDP; no crowd-sourced voiceprint sharing.
6. **APIs/SDKs:** REST + WS MVP (mirrors Twilio/Asterisk/Bhashini patterns above); gRPC + Genesys/Cisco/Avaya
   connectors after primary-doc verification (backlog §6.5); every event carries `model_version`, `window`,
   `codec`, `calibration_id` for audit.

---

## 8. Open Risks / Gaps

1. **Adversarial + neural-codec fragility:** ASVspoof 5 shows CM-optimised attacks (Malafide/Malacopula) and
   neural codecs blunt SOTA baselines; our full-stack numbers on such data are unmeasured — red-team before pilot.
2. **8 kHz PSTN information loss:** models trained at 16 kHz lose >4 kHz artefacts on Twilio legs; quantify
   per-leg EER before promising uniform accuracy.
3. **Streaming latency unproven:** no verified end-to-end CM latency on target hardware; benchmark RTF vs
   accuracy for window/hop choices; streaming CM architectures **[UNVERIFIED]**.
4. **Indic spoof data absent:** no primary-source Indian-language deepfake corpus; multilingual claims untested
   until Hindi + 4–5 language clone-eval sets are collected with consent.
5. **Calibration:** ASVspoof 5 flags score calibration as a deployment blocker — plan Platt/isotonic per
   deployment leg, not just argmax accuracy.
6. **Unverified integrations:** FreeSWITCH/SIPREC, TFLite/LiteRT, gRPC contract, Genesys/Cisco/Avaya,
   core-banking hold/approve SLAs, RBI fraud-reporting automation, WeRTVAF — each needs a primary-doc pass.
7. **Human factors:** 2019 listening tests show some spoofs fool people; UI must make "voice matched" never
   read as "transaction safe" — always pair with the RBI dynamic second factor.

---

## References

### Detection challenges & datasets
- ASVspoof 2019 site. [http://www.asvspoof.org/index2019.html](http://www.asvspoof.org/index2019.html)
- ASVspoof 2019 evaluation plan (baselines, t-DCF/EER). [https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf](https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf)
- ASVspoof 2019 database paper, CSL 2020. [https://doi.org/10.1016/j.csl.2020.101114](https://doi.org/10.1016/j.csl.2020.101114)
- ASVspoof 2019 results paper, T-BIOM 2021. [https://doi.org/10.1109/tbiom.2021.3059479](https://doi.org/10.1109/tbiom.2021.3059479)
- ASVspoof 2021 site. [http://www.asvspoof.org/index2021.html](http://www.asvspoof.org/index2021.html)
- ASVspoof 2021 evaluation plan. [https://www.asvspoof.org/asvspoof2021/asvspoof2021_evaluation_plan.pdf](https://www.asvspoof.org/asvspoof2021/asvspoof2021_evaluation_plan.pdf)
- ASVspoof 2021 overview paper. [https://www.eurecom.edu/publication/6675/download/sec-publi-6675.pdf](https://www.eurecom.edu/publication/6675/download/sec-publi-6675.pdf)
- ASVspoof 5 paper (arXiv). [https://arxiv.org/abs/2408.08739](https://arxiv.org/abs/2408.08739)
- ASVspoof 5 ISCA page. [https://www.isca-archive.org/asvspoof_2024/wang24_asvspoof.html](https://www.isca-archive.org/asvspoof_2024/wang24_asvspoof.html)
- ASVspoof 5 evaluation plan. [https://www.asvspoof.org/file/ASVspoof5___Evaluation_Plan.pdf](https://www.asvspoof.org/file/ASVspoof5___Evaluation_Plan.pdf)
- ASVspoof 5 dataset (Zenodo). [https://doi.org/10.5281/zenodo.14498691](https://doi.org/10.5281/zenodo.14498691)

### Countermeasure models
- AASIST (arXiv). [https://arxiv.org/abs/2110.01200](https://arxiv.org/abs/2110.01200)
- AASIST official code. [https://github.com/clovaai/aasist](https://github.com/clovaai/aasist)
- AASIST (IEEE Xplore). [https://ieeexplore.ieee.org/document/9747766](https://ieeexplore.ieee.org/document/9747766)
- RawNet2 anti-spoofing (arXiv). [https://arxiv.org/abs/2011.01108](https://arxiv.org/abs/2011.01108)
- RawNet2 anti-spoofing code. [https://github.com/eurecom-asp/rawnet2-antispoofing](https://github.com/eurecom-asp/rawnet2-antispoofing)
- wav2vec2 + augmentation CM. [https://ar5iv.labs.arxiv.org/html/2202.12233](https://ar5iv.labs.arxiv.org/html/2202.12233)
- wav2vec2 CM slides (Odyssey 2022). [https://yamagishilab.jp/wp-content/uploads/2022/06/Speaker_odyssey_22_W2V2_anti_spoofing.pdf](https://yamagishilab.jp/wp-content/uploads/2022/06/Speaker_odyssey_22_W2V2_anti_spoofing.pdf)
- XLSR-Mamba. [https://arxiv.org/html/2411.10027v2](https://arxiv.org/html/2411.10027v2)
- AASIST scaling re-examination (ASVspoof 5). [https://arxiv.org/html/2507.11777](https://arxiv.org/html/2507.11777)

### Speaker verification
- ECAPA-TDNN (Interspeech 2020). [https://www.isca-archive.org/interspeech_2020/desplanques20_interspeech.pdf](https://www.isca-archive.org/interspeech_2020/desplanques20_interspeech.pdf)
- ECAPA-TDNN (arXiv). [https://arxiv.org/abs/2005.07143](https://arxiv.org/abs/2005.07143)
- SpeechBrain ECAPA-TDNN API. [https://speechbrain.readthedocs.io/en/latest/API/speechbrain.lobes.models.ECAPA_TDNN.html](https://speechbrain.readthedocs.io/en/latest/API/speechbrain.lobes.models.ECAPA_TDNN.html)
- SpeechBrain repo (recipes: PLDA, score-norm). [https://github.com/speechbrain/speechbrain](https://github.com/speechbrain/speechbrain)
- Pretrained spkrec-ecapa-voxceleb. [https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)

### Real-time / telephony
- RFC 7874 (WebRTC audio codecs). [https://datatracker.ietf.org/doc/html/rfc7874](https://datatracker.ietf.org/doc/html/rfc7874)
- RFC 8834 (RTP usage in WebRTC). [https://www.rfc-editor.org/info/rfc8834/](https://www.rfc-editor.org/info/rfc8834/)
- RFC 3551 (RTP A/V profile, 20 ms). [https://www.rfc-editor.org/rfc/rfc3551.html](https://www.rfc-editor.org/rfc/rfc3551.html)
- Twilio Media Streams overview. [https://www.twilio.com/docs/voice/media-streams](https://www.twilio.com/docs/voice/media-streams)
- Twilio WS messages (mulaw/8000). [https://www.twilio.com/docs/voice/media-streams/websocket-messages](https://www.twilio.com/docs/voice/media-streams/websocket-messages)
- Twilio `<Stream>` TwiML. [https://www.twilio.com/docs/voice/twiml/stream](https://www.twilio.com/docs/voice/twiml/stream)
- Twilio WS/Flask tutorial. [https://www.twilio.com/docs/voice/tutorials/consume-real-time-media-stream-using-websockets-python-and-flask](https://www.twilio.com/docs/voice/tutorials/consume-real-time-media-stream-using-websockets-python-and-flask)
- Asterisk External Media + ARI. [https://docs.asterisk.org/Development/Reference-Information/Asterisk-Framework-and-API-Examples/External-Media-and-ARI/](https://docs.asterisk.org/Development/Reference-Information/Asterisk-Framework-and-API-Examples/External-Media-and-ARI/)
- Asterisk chan_websocket. [https://docs.asterisk.org/Configuration/Channel-Drivers/WebSocket/](https://docs.asterisk.org/Configuration/Channel-Drivers/WebSocket/)
- Asterisk external-media sample. [https://github.com/asterisk/asterisk-external-media](https://github.com/asterisk/asterisk-external-media)

### Prosody / features
- openSMILE Python usage (eGeMAPS). [https://audeering.github.io/opensmile-python/usage.html](https://audeering.github.io/opensmile-python/usage.html)
- openSMILE Python repo. [https://github.com/audeering/opensmile-python/blob/main/README.rst](https://github.com/audeering/opensmile-python/blob/main/README.rst)
- Parselmouth repo. [https://github.com/YannickJadoul/Parselmouth](https://github.com/YannickJadoul/Parselmouth)
- SpeechBrain voice-analysis tutorial. [https://speechbrain.readthedocs.io/en/latest/tutorials/preprocessing/voice-analysis.html](https://speechbrain.readthedocs.io/en/latest/tutorials/preprocessing/voice-analysis.html)
- librosa features. [http://librosa.org/doc/latest/feature.html](http://librosa.org/doc/latest/feature.html)
- librosa API (pyin/yin/piptrack). [https://librosa.org/doc/main/api/index.html](https://librosa.org/doc/main/api/index.html)

### Indian multilingual
- IndicWav2Vec project. [https://indicnlp.ai4bharat.org/indicwav2vec/](https://indicnlp.ai4bharat.org/indicwav2vec/)
- IndicWav2Vec repo. [https://github.com/AI4Bharat/IndicWav2Vec](https://github.com/AI4Bharat/IndicWav2Vec)
- Bhashini APIs (ULCA pipeline calls). [https://bhashini.gitbook.io/bhashini-apis](https://bhashini.gitbook.io/bhashini-apis)
- ULCA repo. [https://github.com/bhashini-dibd/ulca](https://github.com/bhashini-dibd/ulca)
- Common Voice Hindi dataset. [https://mozilladatacollective.com/datasets/cmqiod71900zgnr07uiyw57br](https://mozilladatacollective.com/datasets/cmqiod71900zgnr07uiyw57br)
- Common Voice site. [https://commonvoice.mozilla.org/hi](https://commonvoice.mozilla.org/hi)
- Common Voice releases repo. [https://github.com/common-voice/cv-dataset](https://github.com/common-voice/cv-dataset)

### Privacy / compliance / edge
- DPDP Act 2023 (MeitY PDF). [https://www.meity.gov.in/static/uploads/2024/06/2bf1f0e9f04e6fb4f8fef35e82c42aa5.pdf](https://www.meity.gov.in/static/uploads/2024/06/2bf1f0e9f04e6fb4f8fef35e82c42aa5.pdf)
- DPDP Rules 2025 (PIB). [https://www.pib.gov.in/PressReleasePage.aspx?PRID=2190655](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2190655)
- RBI Authentication Directions 2025. [https://www.rbi.org.in/scripts/BS_ViewMasDirections.aspx?id=12898](https://www.rbi.org.in/scripts/BS_ViewMasDirections.aspx?id=12898)
- TRAI CNAP recommendations. [https://www.trai.gov.in/sites/default/files/2024-11/Recommendation_23022024_0.pdf](https://www.trai.gov.in/sites/default/files/2024-11/Recommendation_23022024_0.pdf)
- TRAI CNAP press release. [https://trai.gov.in/notifications/press-release/trai-releases-recommendations-introduction-calling-name-presentation](https://trai.gov.in/notifications/press-release/trai-releases-recommendations-introduction-calling-name-presentation)
- ONNX Runtime docs. [https://onnxruntime.ai/docs/](https://onnxruntime.ai/docs/)
- ONNX Runtime inference. [https://onnxruntime.ai/inference](https://onnxruntime.ai/inference)
- ONNX Runtime extensions (audio ops). [https://onnxruntime.ai/docs/extensions/](https://onnxruntime.ai/docs/extensions/)
- Silero VAD FAQ (ONNX/mobile/edge, latency). [https://github.com/snakers4/silero-vad/wiki/FAQ](https://github.com/snakers4/silero-vad/wiki/FAQ)
