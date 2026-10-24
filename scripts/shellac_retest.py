"""Re-run Test 1 of the transcription spike with the time-alignment bug fixed.

The first run scored the two shellac transfers at F1 0.029, twenty times worse than
deliberately corrupting a clean recording to 6 dB SNR. That gap was the tell: a
difference in recording quality cannot be worse than destroying the audio outright, so
the fault was in the measurement.

It was. Onsets from transfer B were mapped onto A's clock by one multiplicative scale
taken from the two files' untrimmed durations. A opens with 1.6 s of silence and B with
0.9 s, so after scaling B's content sat about 0.75 s from A's, fifteen times the 50 ms
matching tolerance. Nearly every note missed and the metric called it a transcription
failure.

Fixed by trimming both signals before transcription and fitting the offset explicitly.
With thousands of notes and a 50 ms window, recovering a constant lag is alignment
rather than tuning, and the recovered value is printed so it can be checked against the
trim difference.

Note events are cached, because each transcription costs about nine minutes.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import librosa
import numpy as np

AUDIO = ROOT / "data" / "audio"
OUT = ROOT / "results"
CACHE = OUT / "note_cache"
MODEL_SR = 16000
TOL = 0.05

PAIR = [("A", "ballade1_cortot_1926a.flac"), ("B", "ballade1_cortot_1926b.mp3")]


def load_trimmed(path: Path) -> tuple[np.ndarray, float]:
    y, _ = librosa.load(path, sr=MODEL_SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=35.0)
    return y, len(y) / MODEL_SR


def transcribe_cached(key: str, y: np.ndarray, transcriptor) -> list[dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / f"{key}_trimmed.json"
    if dest.exists():
        return json.loads(dest.read_text())
    t0 = time.time()
    notes = transcriptor.transcribe(y, None)["est_note_events"]
    notes = [{"midi_note": int(n["midi_note"]), "onset_time": float(n["onset_time"]),
              "offset_time": float(n["offset_time"]), "velocity": int(n["velocity"])}
             for n in notes]
    dest.write_text(json.dumps(notes))
    print(f"      {len(notes)} notes in {time.time()-t0:.0f}s")
    return notes


def count_matches(ref, est, scale, offset) -> int:
    by_pitch: dict[int, list[float]] = {}
    for n in ref:
        by_pitch.setdefault(n["midi_note"], []).append(n["onset_time"])
    for v in by_pitch.values():
        v.sort()
    used = {p: set() for p in by_pitch}
    hits = 0
    for n in est:
        t = n["onset_time"] * scale + offset
        cands = by_pitch.get(n["midi_note"], [])
        lo = np.searchsorted(cands, t - TOL)
        best, best_d = None, TOL
        for i in range(lo, len(cands)):
            if cands[i] > t + TOL:
                break
            if i in used[n["midi_note"]]:
                continue
            d = abs(cands[i] - t)
            if d <= best_d:
                best, best_d = i, d
        if best is not None:
            used[n["midi_note"]].add(best)
            hits += 1
    return hits


def main() -> int:
    from piano_transcription_inference import PianoTranscription

    print("loading model...")
    transcriptor = PianoTranscription(device="cpu")

    notes, durs = {}, {}
    for key, filename in PAIR:
        y, d = load_trimmed(AUDIO / filename)
        durs[key] = d
        print(f"   {key}: trimmed {d:.1f}s")
        notes[key] = transcribe_cached(key, y, transcriptor)

    scale = durs["A"] / durs["B"]
    print(f"\n   scale (A/B) = {scale:.4f}")

    # Coarse-to-fine offset search: a constant lag survives trimming, because the
    # two transfers start at slightly different points.
    best = (-1, 0.0)
    for step, span, centre in ((0.02, 3.0, 0.0), (0.005, 0.05, None)):
        centre = best[1] if centre is None else centre
        for off in np.arange(centre - span, centre + span + 1e-9, step):
            h = count_matches(notes["A"], notes["B"], scale, float(off))
            if h > best[0]:
                best = (h, float(off))
    hits, offset = best

    precision = hits / max(len(notes["B"]), 1)
    recall = hits / max(len(notes["A"]), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    print(f"   best offset = {offset:+.3f}s  ({hits} matches)")
    print()
    print("=" * 70)
    print("TEST 1 (corrected)  two transfers of one Cortot 1926 performance")
    print("=" * 70)
    print(f"  notes in A {len(notes['A'])},  notes in B {len(notes['B'])}")
    print(f"  precision {precision:.3f}   recall {recall:.3f}   F1 {f1:.3f}")
    print(f"  (first run reported F1 0.029 with the offset assumed to be zero)")

    result = {"scale": scale, "offset": offset, "matched": hits,
              "n_A": len(notes["A"]), "n_B": len(notes["B"]),
              "precision": precision, "recall": recall, "f1": f1}
    (OUT / "shellac_retest.json").write_text(json.dumps(result, indent=2))

    print()
    if f1 >= 0.8:
        print("  Transcription is consistent across transfers of one shellac")
        print("  performance. Per-note features are viable on the historical roster.")
    elif f1 >= 0.6:
        print("  Partial. A substantial minority of transcribed notes is specific")
        print("  to the transfer, so per-note features carry transfer identity as")
        print("  well as performance identity.")
    else:
        print("  Transcription is not reproducible across transfers of the same")
        print("  shellac performance. Per-note features on pre-1950 material would")
        print("  encode the digitisation as much as the playing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
