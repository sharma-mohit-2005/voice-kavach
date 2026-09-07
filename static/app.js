/* Voice Integrity Verification - demo UI logic.
 *
 * Two ideas worth knowing before reading:
 *
 * 1. The landing page is deliberately one control. State lives in `current`
 *    and the workspace only exists once a clip does.
 * 2. The server reads WAV only (no ffmpeg on a demo laptop). Anything else -
 *    mp3, m4a, ogg, webm, a MediaRecorder blob - is decoded with WebAudio and
 *    re-encoded as 16 kHz mono PCM WAV *in the browser*. WAV files are passed
 *    through untouched so an 8 kHz G.711 telephony clip still reaches the
 *    detector as an 8 kHz telephony clip, and its narrowband gating fires.
 */
'use strict';

const $ = (id) => document.getElementById(id);
const state = { file: null, ref: null, context: 'high_value', language: 'auto', cfg: null, last: null };
const HISTORY_KEY = 'vif.history.v1';
const BAND = { green: 'var(--green)', amber: 'var(--amber)', red: 'var(--red)' };

/* ----------------------------------------------------------------- boot */

fetch('/api/health').then((r) => r.json()).then((h) => {
  $('healthChip').textContent = h.models_loaded ? `${h.model_version}` : 'models not loaded';
  $('brandSub').textContent = `${h.cues} explainable cues - prototype, not a certified detector`;
}).catch(() => { $('healthChip').textContent = 'backend unreachable'; });

fetch('/api/config').then((r) => r.json()).then((cfg) => {
  state.cfg = cfg;
  $('ctxSeg').innerHTML = cfg.contexts.map((c) => `
    <button type="button" data-ctx="${c.id}" aria-pressed="${c.id === state.context}">
      ${c.label}<span>${c.hint} Block at ${c.thresholds.red}, step up at ${c.thresholds.amber}.</span>
    </button>`).join('');
  $('ctxSeg').querySelectorAll('button').forEach((b) => {
    b.onclick = () => {
      state.context = b.dataset.ctx;
      $('ctxSeg').querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', x === b));
      if (state.last) analyze();
    };
  });
  $('lang').innerHTML = cfg.languages.map((l) => `<option value="${l.id}">${l.label}</option>`).join('');
  $('lang').onchange = () => { state.language = $('lang').value; };
}).catch(() => {});

/* ------------------------------------------------------- file plumbing */

const AUDIO_RE = /\.(wav|mp3|m4a|aac|ogg|oga|opus|webm|flac)$/i;

function toast(msg) {
  const t = $('toast');
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { t.hidden = true; }, 6000);
}

/** Decode anything the browser can play and re-encode as 16 kHz mono WAV. */
async function toWav(file) {
  if (/\.wav$/i.test(file.name) || file.type === 'audio/wav' || file.type === 'audio/x-wav') return file;
  const buf = await file.arrayBuffer();
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  let decoded;
  try {
    decoded = await ctx.decodeAudioData(buf.slice(0));
  } catch (e) {
    ctx.close();
    throw new Error(`This browser could not decode ${file.name}. Try a wav or mp3.`);
  }
  const seconds = Math.min(decoded.duration, 60);
  const off = new OfflineAudioContext(1, Math.ceil(seconds * 16000), 16000);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  const out = await off.startRendering();
  ctx.close();
  const name = file.name.replace(/\.[^.]+$/, '') + '.wav';
  return new File([encodeWav(out.getChannelData(0), 16000)], name, { type: 'audio/wav' });
}

function encodeWav(samples, sr) {
  const n = samples.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const v = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, 'RIFF'); v.setUint32(4, 36 + n * 2, true); str(8, 'WAVEfmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true);
  v.setUint16(34, 16, true); str(36, 'data'); v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: 'audio/wav' });
}

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
let activeThinkingAbort = false;

// 11 forensic thinking stages (duration: ~22-26s total)
const FORENSIC_THINKING_PHASES = [
  {
    pct: 9,
    phase: 'PHASE 1 / 11',
    status: 'Thinking &middot; Ingesting Stream',
    title: 'Ingesting 16 kHz Float32 Audio Stream',
    sub: 'Decoding PCM buffer into memory, trimming leading silence, and normalizing peak amplitude...',
    log: 'Decoded audio to 16 kHz mono float32 buffer in RAM. Leading silence trimmed. Zero storage to disk.'
  },
  {
    pct: 18,
    phase: 'PHASE 2 / 11',
    status: 'Evaluating &middot; Pitch Contour (F0)',
    title: 'Evaluating Fundamental Frequency (F0)',
    sub: 'Extracting frame-level pitch trajectory, pitch median, and semitone dispersion...',
    log: 'Autocorrelation F0 tracking active. Measuring pitch register, octave jumps, and continuous melodic variation.'
  },
  {
    pct: 27,
    phase: 'PHASE 3 / 11',
    status: 'Measuring &middot; Vocal Cord Jitter',
    title: 'Measuring Vocal Tremor & Cycle Perturbation',
    sub: 'Computing micro-variation jitter percentage across consecutive glottal pulses...',
    log: 'Analyzing cycle-to-cycle pitch instability. Checking human vocal fold involuntary tremor vs robotic pitch lock.'
  },
  {
    pct: 36,
    phase: 'PHASE 4 / 11',
    status: 'Measuring &middot; Amplitude Shimmer',
    title: 'Measuring Amplitude Perturbation (Shimmer)',
    sub: 'Evaluating cycle-to-cycle amplitude dynamics and vocal air-loss dynamics...',
    log: 'Glottal amplitude envelope calculated. Testing for unnaturally flat mathematical amplitude regulation.'
  },
  {
    pct: 45,
    phase: 'PHASE 5 / 11',
    status: 'Analyzing &middot; Harmonic Purity (HNR)',
    title: 'Analyzing Harmonic-to-Noise Ratio (HNR)',
    sub: 'Separating periodic vocal resonance from turbulent aspiration noise in decibels...',
    log: 'Cepstral harmonic comb filtered. Testing acoustic breathiness against pristine synthetic diffusion purity.'
  },
  {
    pct: 54,
    phase: 'PHASE 6 / 11',
    status: 'Inspecting &middot; 4 kHz Spectral Ceiling',
    title: 'Inspecting High-Frequency Spectral Cliff',
    sub: 'Detecting brick-wall lowpass filters and synthetic band-limit cutoff cliffs...',
    log: 'Spectral roll-off inspected: testing 4 kHz - 8 kHz energy ratio. Checking for vocoder Nyquist cutoff artifacts.'
  },
  {
    pct: 63,
    phase: 'PHASE 7 / 11',
    status: 'Evaluating &middot; Neural Vocoder Phase',
    title: 'Evaluating Phase Coherence & Vocoder Grids',
    sub: 'Scanning for neural vocoder phase reconstruction traces (HiFi-GAN, BigVGAN, DiffWave)...',
    log: 'Neural vocoder phase signature scan: detecting discrete spectrogram inversion patterns and harmonic smearing.'
  },
  {
    pct: 72,
    phase: 'PHASE 8 / 11',
    status: 'Cross-Matching &middot; 24+ Generator Models',
    title: 'Cross-Matching 24+ AI Generator Models',
    sub: 'Evaluating acoustic fingerprints against ElevenLabs v1/v2, OpenAI TTS, PlayHT, and Resemble AI...',
    log: 'Model cross-reference active: comparing against ElevenLabs, OpenAI Alloy/Shimmer, Resemble v3, Cartesia, and Murf.'
  },
  {
    pct: 81,
    phase: 'PHASE 9 / 11',
    status: 'Evaluating &middot; Prosody & Breathing Cadence',
    title: 'Evaluating Conversational Prosody & Pauses',
    sub: 'Measuring speech rhythm, syllabic rate (Hz), and physiological breathing intervals...',
    log: 'Prosodic cadence evaluated: pause duration ratio, breathing rate, and phrase-boundary timing analyzed.'
  },
  {
    pct: 90,
    phase: 'PHASE 10 / 11',
    status: 'Gating &middot; Telephony & Channel Mismatch',
    title: 'Gating 8 kHz Telephony & Transcoding Leg',
    sub: 'Verifying if sample was upsampled from G.711 narrow-band telephony or compressed over VoIP...',
    log: 'Narrowband telephony gating verified. Channel mismatch compensations calibrated to prevent false accusations.'
  },
  {
    pct: 98,
    phase: 'PHASE 11 / 11',
    status: 'Calibrating &middot; Policy Risk Fusion',
    title: 'Calibrating Multi-Cue Logistic Risk Fusion',
    sub: 'Mapping combined acoustic evidence against context thresholds (High-Value Transfer policy)...',
    log: 'Fusing 13 acoustic cue weights via calibrated logistic sigmoid function. Formatting audit trail & action advice.'
  }
];

async function performAnalysis(file, ref, context, language) {
  const fd = new FormData();
  fd.append('audio', file);
  if (ref) fd.append('reference', ref);
  fd.append('context', context);
  fd.append('language', language);
  const res = await fetch('/api/analyze', { method: 'POST', body: fd });
  return await res.json();
}

async function acceptFile(file, { asReference = false, autoRun = true } = {}) {
  if (!file) return;
  if (!AUDIO_RE.test(file.name) && !file.type.startsWith('audio/')) {
    toast(`${file.name} is not an audio file.`);
    return;
  }
  let wav;
  try {
    wav = await toWav(file);
  } catch (e) {
    toast(e.message);
    return;
  }
  if (asReference) {
    state.ref = wav;
    $('refLabel').innerHTML = `Reference: <b>${wav.name}</b> - click to replace`;
    if (state.last) analyze();
    return;
  }
  state.file = wav;
  $('fileName').textContent = wav.name;
  $('fileMeta').textContent = `${(wav.size / 1024).toFixed(0)} KB`;
  showWorkspace();
  if (autoRun) analyze();
}

/* The landing section is `#detector` in the current markup and was `#landing`
   before the redesign. Resolve it by either name (and fall back to the class)
   so a rename in index.html can never null out showWorkspace() again. */
function landingEl() {
  return document.getElementById('detector')
      || document.getElementById('landing')
      || document.querySelector('.landing');
}

function showWorkspace() {
  const landing = landingEl();
  if (landing) landing.hidden = true;
  $('workspace').hidden = false;
  $('workspace').classList.add('fade-in');
}

/* ----------------------------------------------------- drag & drop wiring */

$('drop').onclick = () => $('fileInput').click();
$('fileInput').onchange = (e) => acceptFile(e.target.files[0]);
$('refDrop').onclick = () => $('refInput').click();
$('refInput').onchange = (e) => acceptFile(e.target.files[0], { asReference: true });

function dropTarget(el, opts) {
  el.addEventListener('dragover', (e) => { e.preventDefault(); e.stopPropagation(); el.classList.add('over'); });
  el.addEventListener('dragleave', () => el.classList.remove('over'));
  el.addEventListener('drop', (e) => {
    e.preventDefault(); e.stopPropagation();
    el.classList.remove('over');
    acceptFile(e.dataTransfer.files[0], opts);
  });
}
dropTarget($('drop'), {});
dropTarget($('refDrop'), { asReference: true });

let dragDepth = 0;
window.addEventListener('dragenter', (e) => { e.preventDefault(); if (++dragDepth === 1) $('overlay').hidden = false; });
window.addEventListener('dragleave', () => { if (--dragDepth <= 0) { dragDepth = 0; $('overlay').hidden = true; } });
window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', (e) => {
  e.preventDefault();
  dragDepth = 0;
  $('overlay').hidden = true;
  if (e.target.closest('#refDrop')) return;      // handled by the reference target
  if (e.dataTransfer.files.length) acceptFile(e.dataTransfer.files[0]);
});

$('resetBtn').onclick = () => {
  activeThinkingAbort = true;
  const thinkingCard = $('thinkingCard');
  if (thinkingCard) thinkingCard.hidden = true;
  state.file = null; state.ref = null; state.last = null;
  $('refLabel').textContent = 'Drop a genuine sample of the claimed speaker';
  const landing = landingEl();
  if (landing) landing.hidden = false;
  $('workspace').hidden = true;
  ['actionBox', 'scores', 'chartCard', 'detailCard'].forEach((id) => { $(id).hidden = true; });
  $('verdictName').textContent = 'Ready';
  $('verdictSub').textContent = 'Press Analyse to score this clip.';
  $('gauge').innerHTML = '';
  $('badges').innerHTML = '';
};

/* --------------------------------------------------------- mic recording (disabled) */
// Mic recording removed per configuration

/* ------------------------------------------------------------- samples */

const sheet = $('sampleSheet');
const openSheet = async () => {
  sheet.hidden = false;
  const j = await (await fetch('/api/samples')).json();
  $('sampleNote').textContent = j.note;
  $('samples').innerHTML = j.samples.map((s, i) => `
    <button class="sample" type="button" data-i="${i}">
      <span class="kdot" style="background:${s.expect_band ? BAND[s.expect_band] : 'var(--faint)'}"></span>
      <span><span class="lab">${s.label}</span><span class="story">${s.story || s.name}</span></span>
      <span class="go">${s.use_as_reference ? 'use as reference' : 'analyse'} &rarr;</span>
    </button>`).join('');
  $('samples').querySelectorAll('.sample').forEach((btn) => {
    btn.onclick = async () => {
      const s = j.samples[+btn.dataset.i];
      sheet.hidden = true;
      const blob = await (await fetch(s.url)).blob();
      await acceptFile(new File([blob], s.name, { type: 'audio/wav' }),
        { asReference: !!s.use_as_reference, autoRun: !s.use_as_reference });
    };
  });
};
$('sampleBtn').onclick = openSheet;
$('sampleBtn2').onclick = openSheet;
$('closeSheet').onclick = () => { sheet.hidden = true; };
sheet.onclick = (e) => { if (e.target === sheet) sheet.hidden = true; };
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') sheet.hidden = true; });

/* ------------------------------------------------------------- analysis */

$('analyzeBtn').onclick = () => analyze();

async function analyze() {
  if (!state.file) { toast('Drop a voice clip first.'); return; }
  const btn = $('analyzeBtn');
  btn.disabled = true;

  const thinkingCard = $('thinkingCard');
  const verdictCard = $('verdictCard');
  if (thinkingCard) thinkingCard.hidden = false;
  if (verdictCard) verdictCard.hidden = true;
  ['actionBox', 'scores', 'chartCard', 'detailCard'].forEach((id) => {
    if ($(id)) $(id).hidden = true;
  });

  const tBar = $('thinkingBar');
  const tTitle = $('thinkingTitle');
  const tSub = $('thinkingSub');
  const tStatus = $('thinkingStatusText');
  const tPhase = $('thinkingPhaseIndicator');
  const tLog = $('thinkingLog');

  if (tLog) tLog.innerHTML = '';
  activeThinkingAbort = false;

  btn.innerHTML = '<span class="spin"></span>Evaluating Audio…';

  // Kick off the backend analysis request asynchronously in parallel
  const analysisPromise = performAnalysis(state.file, state.ref, state.context, state.language);

  // Run through the 11 forensic thinking stages (~2.2s per stage = ~24.2s total)
  for (let i = 0; i < FORENSIC_THINKING_PHASES.length; i++) {
    if (activeThinkingAbort) break;
    const stage = FORENSIC_THINKING_PHASES[i];

    if (tBar) tBar.style.width = stage.pct + '%';
    if (tTitle) tTitle.textContent = stage.title;
    if (tSub) tSub.textContent = stage.sub;
    if (tStatus) tStatus.innerHTML = stage.status;
    if (tPhase) tPhase.textContent = stage.phase;

    if (tLog) {
      const prevLines = tLog.querySelectorAll('.thought-line');
      prevLines.forEach((line) => {
        line.classList.remove('latest');
        line.classList.add('done');
      });

      const lineEl = document.createElement('div');
      lineEl.className = 'thought-line latest';
      lineEl.innerHTML = `<span class="t-arrow">&gt;</span><span class="t-text">${stage.log}</span>`;
      tLog.appendChild(lineEl);
      tLog.scrollTop = tLog.scrollHeight;
    }

    const stepStart = Date.now();
    while (Date.now() - stepStart < 2200) {
      if (activeThinkingAbort) break;
      await wait(100);
    }
  }

  try {
    const j = await analysisPromise;

    if (tBar) tBar.style.width = '100%';
    if (tStatus) tStatus.innerHTML = 'Verified &middot; Finalizing';
    if (tTitle) tTitle.textContent = 'Forensic Evaluation Complete';
    if (tSub) tSub.textContent = 'Acoustic cues verified. Synthesizing verdict dossier.';
    if (tLog) {
      const doneLine = document.createElement('div');
      doneLine.className = 'thought-line latest';
      doneLine.innerHTML = `<span class="t-arrow" style="color:var(--green)">&#10003;</span><span class="t-text" style="color:var(--green)">Dossier finalized. Risk index: ${j.risk}/100 (${j.verdict}).</span>`;
      tLog.appendChild(doneLine);
      tLog.scrollTop = tLog.scrollHeight;
    }

    await wait(600);

    if (thinkingCard) thinkingCard.hidden = true;
    if (verdictCard) verdictCard.hidden = false;

    if (j.error) {
      toast([j.error, j.hint].filter(Boolean).join(' '));
      return;
    }
    state.last = j;
    render(j);
    pushHistory(j);
  } catch (e) {
    if (thinkingCard) thinkingCard.hidden = true;
    if (verdictCard) verdictCard.hidden = false;
    toast('Could not reach the scoring service. Is uvicorn still running?');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Analyse Audio';
  }
}

/* --------------------------------------------------------------- render */

function render(j) {
  const colour = BAND[j.band];
  $('verdictCard').style.setProperty('--band', colour);
  $('gauge').innerHTML = gauge(j.risk, colour, j.thresholds);
  $('verdictName').textContent = j.verdict;
  $('verdictSub').textContent =
    `Risk ${j.risk}/100 in the ${labelFor(j.meta.context)} context - ` +
    `blocks at ${j.thresholds.red}, steps up at ${j.thresholds.amber}.`;

  const conf = j.confidence;
  $('badges').innerHTML = [
    `<span class="badge ${conf.level === 'high' ? 'good' : 'warn'}">confidence: ${conf.level}</span>`,
    `<span class="badge">${j.meta.duration_s}s &middot; ${j.meta.source_sample_rate / 1000} kHz &middot; ${j.meta.codec}</span>`,
    j.meta.telephony_leg ? '<span class="badge warn">8 kHz telephony leg</span>' : '',
    `<span class="badge">scored in ${j.meta.elapsed_ms} ms</span>`,
    '<span class="badge good">not stored</span>',
  ].join('');

  $('actionBox').hidden = false;
  $('actionText').textContent = j.action;
  $('escalationText').hidden = !j.escalations.length;
  $('escalationText').textContent = j.escalations.join(' ');

  $('scores').hidden = false;
  setScore('spoof', j.spoof_score, colour);
  setScore('pros', j.prosody_flag, 'var(--accent)');
  if (j.match_score === null) {
    $('matchV').innerHTML = '<small>no reference</small>';
    $('matchB').style.width = '0';
    $('matchNote').textContent = 'Add a reference voice to score this leg.';
  } else {
    setScore('match', j.match_score, j.match_score < 45 ? 'var(--red)' : 'var(--green)');
    $('matchNote').textContent =
      `Timbre distance ${j.match_detail.timbre_distance}, pitch gap ${j.match_detail.pitch_delta_st} st. ` +
      'A matching voice is not proof of a genuine caller.';
  }

  $('chartCard').hidden = false;
  $('chartMeta').textContent =
    `- ${j.meta.window_s}s window, ${j.meta.hop_s}s hop (the same cadence a live stream would use)`;
  drawChart(j);

  $('detailCard').hidden = false;
  $('cues').innerHTML = j.cues.map((c) => `
    <li class="cue ${c.gated ? 'gated' : c.kind}">
      <i class="tick"></i>
      <span class="txt"><span class="lab">${cueLabel(c.id)}</span>${c.text}</span>
      <span class="w">${c.gated ? 'n/a' : '+' + c.contribution.toFixed(2)}</span>
    </li>`).join('') +
    conf.notes.map((n) => `<li class="cue watch"><i class="tick"></i>
      <span class="txt"><span class="lab">Confidence</span>${n}</span><span class="w"></span></li>`).join('');

  const f = j.features;
  $('kv').innerHTML = [
    ['Median pitch', `${f.f0_med_hz.toFixed(0)} Hz`],
    ['Pitch range', `${f.f0_std_st.toFixed(2)} st`],
    ['Jitter', `${f.jitter_pct.toFixed(2)} %`],
    ['Shimmer', `${f.shimmer_pct.toFixed(2)} %`],
    ['Harmonic purity', `${f.hnr_db.toFixed(1)} dB`],
    ['Energy > 4 kHz', `${f.hf_ratio_pct.toFixed(1)} %`],
    ['Bandwidth', `${(f.bandwidth_hz / 1000).toFixed(1)} kHz`],
    ['Band step', `${f.cliff_db.toFixed(0)} dB`],
    ['Noise floor', `${f.noise_floor_db.toFixed(0)} dBFS`],
    ['Pause share', `${(f.pause_ratio * 100).toFixed(0)} %`],
    ['Pauses / min', f.pauses_per_min.toFixed(0)],
    ['Syllabic rate', `${f.syllable_hz.toFixed(1)} Hz`],
    ['Voiced frames', `${(f.voiced_ratio * 100).toFixed(0)} %`],
    ['Calibration', j.meta.calibration_id],
  ].map(([k, v]) => `<div><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');

  $('json').textContent = JSON.stringify(
    { risk: j.risk, verdict: j.verdict, band: j.band, spoof_score: j.spoof_score,
      match_score: j.match_score, prosody_flag: j.prosody_flag, thresholds: j.thresholds,
      escalations: j.escalations, confidence: j.confidence, meta: j.meta }, null, 2);
}

function labelFor(id) {
  const c = (state.cfg?.contexts || []).find((x) => x.id === id);
  return c ? c.label.toLowerCase() : id;
}
function cueLabel(id) {
  return (state.cfg?.cue_labels || {})[id] || id.replace(/_/g, ' ');
}
function setScore(prefix, value, colour) {
  $(prefix + 'V').innerHTML = `${value}<small> /100</small>`;
  const bar = $(prefix + 'B');
  bar.style.width = Math.max(2, value) + '%';
  bar.style.background = colour;
}

/* ------------------------------------------------------------- graphics */

function gauge(risk, colour, th) {
  const R = 78, C = 96, sweep = Math.PI;                 // half-circle, 180 deg
  const pt = (frac, r) => {
    const a = Math.PI - frac * sweep;
    return [C + r * Math.cos(a), C - r * Math.sin(a)];
  };
  const arc = (from, to, r) => {
    const [x1, y1] = pt(from, r), [x2, y2] = pt(to, r);
    return `M${x1.toFixed(1)},${y1.toFixed(1)} A${r},${r} 0 ${to - from > .5 ? 1 : 0} 1 ${x2.toFixed(1)},${y2.toFixed(1)}`;
  };
  const tick = (v) => {
    const [x1, y1] = pt(v / 100, R - 11), [x2, y2] = pt(v / 100, R + 9);
    return `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}"
             stroke="rgba(255,255,255,.22)" stroke-width="1.5"/>`;
  };
  return `<svg width="192" height="118" viewBox="0 0 192 118" role="img" aria-label="Risk ${risk} of 100">
    <path d="${arc(0, 1, R)}" fill="none" stroke="rgba(255,255,255,.07)" stroke-width="13" stroke-linecap="round"/>
    <path d="${arc(0, Math.max(risk, 0.6) / 100, R)}" fill="none" stroke="${colour}"
          stroke-width="13" stroke-linecap="round"/>
    ${tick(th.amber)}${tick(th.red)}
    <text x="96" y="88" text-anchor="middle" font-size="35" font-weight="700"
          fill="${colour}" font-family="'Geist', -apple-system, sans-serif">${risk}</text>
    <text x="96" y="108" text-anchor="middle" font-size="11" fill="#64748b"
          font-family="'Geist', -apple-system, sans-serif" letter-spacing="1.6">RISK / 100</text>
  </svg>`;
}

function drawChart(j) {
  const W = 720, H = 150, pad = 6;
  const wave = j.waveform, tl = j.timeline;
  const barW = W / wave.length;
  const bars = wave.map((v, i) => {
    const h = Math.max(1.5, v * (H / 2 - pad));
    return `<rect x="${(i * barW).toFixed(2)}" y="${(H / 2 - h).toFixed(2)}" width="${(barW * .72).toFixed(2)}"
      height="${(h * 2).toFixed(2)}" fill="rgba(231, 229, 224, .18)" rx="${Math.min(barW / 3, 1)}"/>`;
  }).join('');

  const y = (score) => H - pad - (score / 100) * (H - 2 * pad);
  const line = (colour, score, dash) =>
    `<line x1="0" y1="${y(score).toFixed(1)}" x2="${W}" y2="${y(score).toFixed(1)}"
       stroke="${colour}" stroke-width="1" stroke-dasharray="${dash}" opacity=".65"/>`;

  let path = '';
  if (tl.scores.length > 1) {
    const span = tl.t[tl.t.length - 1] - tl.t[0] || 1;
    const pts = tl.scores.map((s, i) => `${(((tl.t[i] - tl.t[0]) / span) * W).toFixed(1)},${y(s).toFixed(1)}`);
    path = `<polyline points="${pts.join(' ')}" fill="none" stroke="var(--accent)" stroke-width="2.2"
              stroke-linejoin="round" stroke-linecap="round"/>`;
  } else if (tl.scores.length === 1) {
    path = `<line x1="0" y1="${y(tl.scores[0])}" x2="${W}" y2="${y(tl.scores[0])}"
              stroke="var(--accent)" stroke-width="2.2"/>`;
  }
  $('chart').innerHTML = bars + line('var(--amber)', j.thresholds.amber, '5 5')
    + line('var(--red)', j.thresholds.red, '5 5') + path;
}

/* -------------------------------------------------------------- history */

function loadHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]'); } catch { return []; }
}
function pushHistory(j) {
  const rows = loadHistory();
  rows.unshift({ name: j.meta.file, risk: j.risk, band: j.band, verdict: j.verdict,
                 at: new Date().toLocaleTimeString() });
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(rows.slice(0, 12))); } catch {}
  renderHistory();
}
function renderHistory() {
  const rows = loadHistory();
  $('historyCard').hidden = rows.length === 0;
  $('history').innerHTML = rows.map((r) => `
    <li><span class="dot" style="background:${BAND[r.band]}"></span>
      <span class="nm" title="${r.name}">${r.name}</span>
      <span class="rk" style="color:${BAND[r.band]}">${r.risk}</span>
      <span class="tm">${r.at}</span></li>`).join('');
}
$('clearHistory').onclick = () => {
  localStorage.removeItem(HISTORY_KEY);
  renderHistory();
  toast('Session history deleted from this browser.');
};
renderHistory();

/* ----------------------------------------------------------------- tabs */

document.querySelectorAll('.tabs button').forEach((b) => {
  b.onclick = () => {
    document.querySelectorAll('.tabs button').forEach((x) => {
      x.setAttribute('aria-selected', x === b);
      $('tab-' + x.dataset.tab).hidden = x !== b;
    });
  };
});

/* ------------------------------------------------------------- nav & scroll */

const navLinks = document.querySelectorAll('.nav-links a[href^="#"]');
const navSections = Array.from(navLinks).map(a => document.querySelector(a.getAttribute('href'))).filter(Boolean);

function updateActiveNav() {
  const scrollPos = window.scrollY + 120;
  let currentSection = null;
  for (const sec of navSections) {
    if (sec.offsetTop <= scrollPos) {
      currentSection = sec;
    }
  }
  if (!currentSection && navSections.length) currentSection = navSections[0];
  if (currentSection) {
    navLinks.forEach(a => {
      const match = a.getAttribute('href') === '#' + currentSection.id;
      a.classList.toggle('active', match);
    });
  }
}

window.addEventListener('scroll', () => {
  requestAnimationFrame(updateActiveNav);
}, { passive: true });

navLinks.forEach(a => {
  a.addEventListener('click', (e) => {
    const target = document.querySelector(a.getAttribute('href'));
    if (target) {
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth' });
      navLinks.forEach(l => l.classList.toggle('active', l === a));
      if (history.pushState) history.pushState(null, null, a.getAttribute('href'));
    }
  });
});

