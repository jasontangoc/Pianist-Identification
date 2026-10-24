"""Does piano transcription survive shellac? Test before committing to the roster.

Note-level data is needed from recordings that are disproportionately 1920s-40s 78rpm
transfers, because those are the pianists with enough recordings to support a
recording-level split. Everything measured up to here ran on a coarse harmonic summary
of the audio, which is robust to surface noise by construction, and says nothing about
whether individual notes can be recovered.

There is no note list for Cortot's 1926 Ballade to check against, so two tests get
round the missing ground truth. The first takes two independent digitisations of one
performance: whatever the true notes are, both files contain them, so the agreement
between the two transcriptions is an upper bound on how much of a transcription is
signal rather than an artefact of one transfer. The second transcribes a clean
recording, adds shellac-like noise and transcribes again, making the clean
transcription exact ground truth for the noisy one, which isolates noise from
everything else that differs between a 1926 and a 1963 recording.

Test 1 alone would confound noise with every other property of old recordings; test 2
alone would be a simulation. Together they bracket the answer.
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
MIDI_OUT = OUT / "transcriptions"

ONSET_TOLERANCE = 0.05  # seconds; the standard transcription-eval window
MODEL_SR = 16000

CLEAN = "ballade1_michelangeli_1963.flac"
SHELLAC_A = "ballade1_cortot_1926a.flac"
SHELLAC_B = "ballade1_cortot_1926b.mp3"


def transcribe(path: Path, transcriptor) -> list[dict]:
    audio, _ = librosa.load(path, sr=MODEL_SR, mono=True)
    t0 = time.time()
    out = transcriptor.transcribe(audio, None)
    notes = out["est_note_events"]
    print(f"      {len(notes)} notes in {time.time()-t0:.0f}s "
          f"({len(audio)/MODEL_SR:.0f}s audio)")
    return notes


def match(ref: list[dict], est: list[dict], scale: float = 1.0) -> dict:
    """Greedy onset+pitch matching, the standard transcription metric.

    `scale` maps the estimate's time axis onto the reference's, which is needed
    here because the two transfers of one performance differ in playback speed
    by about 5%. Without it every note after the first few seconds falls outside
    the tolerance window and the score is meaningless.
    """
    ref_by_pitch: dict[int, list[float]] = {}
    for n in ref:
        ref_by_pitch.setdefault(int(n["midi_note"]), []).append(float(n["onset_time"]))
    for v in ref_by_pitch.values():
        v.sort()

    used: dict[int, set[int]] = {p: set() for p in ref_by_pitch}
    hits = 0
    for n in sorted(est, key=lambda n: n["onset_time"]):
        pitch = int(n["midi_note"])
        onset = float(n["onset_time"]) * scale
        candidates = ref_by_pitch.get(pitch, [])
        best, best_d = None, ONSET_TOLERANCE
        for i, t in enumerate(candidates):
            if i in used[pitch]:
                continue
            d = abs(t - onset)
            if d <= best_d:
                best, best_d = i, d
        if best is not None:
            used[pitch].add(best)
            hits += 1

    precision = hits / max(len(est), 1)
    recall = hits / max(len(ref), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"matched": hits, "n_ref": len(ref), "n_est": len(est),
            "precision": precision, "recall": recall, "f1": f1}


def shellac_noise(y: np.ndarray, sr: int, snr_db: float = 12.0) -> np.ndarray:
    """Approximate 78rpm surface noise: broadband hiss plus a band-limited
    rumble, with the top octaves rolled off as shellac transfers are."""
    rng = np.random.default_rng(0)
    hiss = rng.normal(0, 1, len(y))
    # rumble
    rumble = np.cumsum(rng.normal(0, 1, len(y)))
    rumble = rumble / (np.abs(rumble).max() + 1e-9)
    noise = hiss + 3.0 * rumble
    noise = noise / (noise.std() + 1e-9)
    sig_rms = y.std()
    noise *= sig_rms / (10 ** (snr_db / 20))
    out = y + noise
    # roll off above ~5 kHz, roughly where 78s stop
    out = librosa.effects.preemphasis(out, coef=-0.75)
    return (out / (np.abs(out).max() + 1e-9) * 0.95).astype(np.float32)


def summary(notes: list[dict], duration: float) -> dict:
    vel = np.array([n.get("velocity", 0) for n in notes], dtype=float)
    pitches = np.array([n["midi_note"] for n in notes], dtype=float)
    return {
        "n_notes": len(notes),
        "notes_per_sec": len(notes) / max(duration, 1e-9),
        "pitch_min": float(pitches.min()) if len(pitches) else None,
        "pitch_max": float(pitches.max()) if len(pitches) else None,
        "velocity_mean": float(vel.mean()) if len(vel) else None,
        "velocity_sd": float(vel.std()) if len(vel) else None,
    }


def main() -> int:
    from piano_transcription_inference import PianoTranscription

    MIDI_OUT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(exist_ok=True)
    print("loading model (downloads ~170 MB checkpoint on first run)...")
    transcriptor = PianoTranscription(device="cpu")

    results: dict[str, object] = {}

    # --- TEST 1: two transfers of one performance -----------------------
    print("\nTEST 1  two digitisations of one Cortot 1926 performance")
    a_y, _ = librosa.load(AUDIO / SHELLAC_A, sr=MODEL_SR, mono=True)
    b_y, _ = librosa.load(AUDIO / SHELLAC_B, sr=MODEL_SR, mono=True)
    dur_a, dur_b = len(a_y) / MODEL_SR, len(b_y) / MODEL_SR
    print(f"   durations {dur_a:.0f}s and {dur_b:.0f}s (speed ratio {dur_b/dur_a:.4f})")

    print("   transcribing transfer A...")
    notes_a = transcribe(AUDIO / SHELLAC_A, transcriptor)
    print("   transcribing transfer B...")
    notes_b = transcribe(AUDIO / SHELLAC_B, transcriptor)

    scale = dur_a / dur_b
    results["test1_shellac_pair"] = {
        "scale_applied": scale,
        "match": match(notes_a, notes_b, scale=scale),
        "A": summary(notes_a, dur_a),
        "B": summary(notes_b, dur_b),
    }
    m = results["test1_shellac_pair"]["match"]
    print(f"   -> F1 {m['f1']:.3f}  (precision {m['precision']:.3f}, "
          f"recall {m['recall']:.3f}, {m['matched']} matched)")

    # --- TEST 2: clean vs clean+noise, exact ground truth ---------------
    print("\nTEST 2  one clean recording, with and without simulated shellac noise")
    clean_y, _ = librosa.load(AUDIO / CLEAN, sr=MODEL_SR, mono=True)
    dur_c = len(clean_y) / MODEL_SR
    print("   transcribing clean...")
    t0 = time.time()
    notes_clean = transcriptor.transcribe(clean_y, None)["est_note_events"]
    print(f"      {len(notes_clean)} notes in {time.time()-t0:.0f}s")

    test2 = {}
    for snr in (18.0, 12.0, 6.0):
        print(f"   transcribing with {snr:.0f} dB SNR noise...")
        noisy = shellac_noise(clean_y, MODEL_SR, snr_db=snr)
        t0 = time.time()
        notes_noisy = transcriptor.transcribe(noisy, None)["est_note_events"]
        print(f"      {len(notes_noisy)} notes in {time.time()-t0:.0f}s")
        scored = match(notes_clean, notes_noisy)
        test2[f"snr_{snr:.0f}db"] = {"match": scored, "summary": summary(notes_noisy, dur_c)}
        print(f"      -> F1 {scored['f1']:.3f} "
              f"(precision {scored['precision']:.3f}, recall {scored['recall']:.3f})")
    results["test2_noise_ladder"] = {"clean": summary(notes_clean, dur_c), "noisy": test2}

    (OUT / "transcription_spike.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    # --- read-out --------------------------------------------------------
    f1_real = results["test1_shellac_pair"]["match"]["f1"]
    print()
    print("=" * 74)
    print("READ-OUT")
    print("=" * 74)
    print(f"  shellac self-agreement (real)      F1 {f1_real:.3f}")
    for k, v in test2.items():
        print(f"  clean vs noise {k:<18}  F1 {v['match']['f1']:.3f}")
    print()
    print(f"  note density: Cortot 1926 {results['test1_shellac_pair']['A']['notes_per_sec']:.1f}/s"
          f"   Michelangeli 1963 {results['test2_noise_ladder']['clean']['notes_per_sec']:.1f}/s")
    print()
    if f1_real >= 0.8:
        print("  Transcription is consistent on shellac. Per-note features can be")
        print("  built on the historical roster.")
    elif f1_real >= 0.6:
        print("  Partial. A meaningful fraction of transcribed notes is transfer-")
        print("  specific. Features must be aggregate/distributional rather than")
        print("  per-note, and the per-note headline features need re-examining.")
    else:
        print("  Transcription is NOT reliable on shellac: two transfers of one")
        print("  performance do not yield the same notes. Either restrict the")
        print("  roster to post-1950 recordings or abandon per-note features.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
