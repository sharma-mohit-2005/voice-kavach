"""Regression harness: score every demo sample and check the expected verdict.

    python scripts/eval_samples.py             # table + pass/fail
    python scripts/eval_samples.py --features  # add the raw feature vector

Exit code is non-zero if any sample lands outside its expected band, so this is
the "all 6 demo files classify correctly" gate from the PRD (S5).
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import audio_io  # noqa: E402
import detector  # noqa: E402

SAMPLES = ROOT / "samples"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", action="store_true")
    ap.add_argument("--context", default="high_value")
    args = ap.parse_args()

    manifest = {}
    mf = SAMPLES / "manifest.json"
    if mf.exists():
        manifest = {s["file"]: s for s in json.loads(mf.read_text(encoding="utf-8"))["samples"]}

    header = (f"{'file':22s} {'risk':>6s} {'spoof':>6s} {'prosody':>8s} {'verdict':16s} "
              f"{'expect':8s} {'conf':7s} {'ms':>6s}  result")
    print(header)
    print("-" * len(header))
    failures = 0
    for p in sorted(SAMPLES.glob("*.wav")):
        y, meta = audio_io.load(p.read_bytes())
        t0 = time.perf_counter()
        r = detector.analyze(y, meta, context=args.context)
        ms = (time.perf_counter() - t0) * 1000
        expect = manifest.get(p.name, {}).get("expect_band")
        ok = expect is None or r["band"] == expect
        failures += 0 if ok else 1
        print(f"{p.name:22s} {r['risk']:6.1f} {r['spoof_score']:6.1f} {r['prosody_flag']:8.1f} "
              f"{r['verdict']:16s} {str(expect):8s} {r['confidence']['level']:7s} {ms:6.0f}  "
              f"{'PASS' if ok else 'FAIL'}")
        if args.features:
            for k, v in r["features"].items():
                print(f"      {k:18s} {v}")

    def ff(name):
        y, _ = audio_io.load((SAMPLES / name).read_bytes())
        return detector.frame_features(y)

    pairs = [("real_en.wav", "real_en.wav", "same clip (upper bound)"),
             ("real_en.wav", "real_hi.wav", "same speaker, other language"),
             ("real_en.wav", "wrong_speaker.wav", "different speaker"),
             ("clone_en.wav", "real_en.wav", "clone vs genuine reference")]
    print("\nspeaker match (reference leg)")
    for a, b, label in pairs:
        if not (SAMPLES / a).exists() or not (SAMPLES / b).exists():
            continue
        m, d = detector.speaker_match(ff(a), ff(b))
        print(f"  {a:18s} vs {b:18s} match={m:5.1f}  timbre_d={d['timbre_distance']:.3f} "
              f"dF0={d['pitch_delta_st']:5.2f}st   {label}")

    print(f"\n{'ALL PASS' if failures == 0 else str(failures) + ' FAILURE(S)'}"
          f"   model={detector.CONFIG['model_version']}  calibration={detector.CONFIG['calibration_id']}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
