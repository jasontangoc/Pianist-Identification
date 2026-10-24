"""Decide whether two audio files hold the same performance.

Metadata cannot answer it. Cortot's 1926 and 1933 Ballades are 3.6% apart in duration
while two transfers of the 1926 alone are 5.5% apart, so a re-recording can look more
alike than a reissue, and reissues rarely say what they reissue.

Two files of one performance DTW-align to a near-diagonal path, because the only
differences between them are transfer speed and noise. This runs the aligner coarser
than the measurement code does, since it only decides same-or-different: 10.8 fps (hop
2048) is the floor, and at 5.4 fps the identity pair scored worse than three of four
known-different pairs. The statistic is the linear-fit residual as a percentage of
duration, which separates by 2.5x against 1.4x for de-trended tempo spread and does not
amplify staircase noise. Measured: 0.35% for two transfers of one performance, 0.88% for
the closest different pair, so the threshold sits at 0.60%. `calibrate()` re-derives it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

COARSE_SR = 22050
COARSE_HOP = 2048  # 10.8 fps; below this the test stops working
STEP_SIZES = np.array([[1, 1], [1, 2], [2, 1]])

RESIDUAL_THRESHOLD = 0.60  # % of duration; measured band is 0.35 / 0.88
SPREAD_THRESHOLD = 0.09    # secondary check only
WINDOW_SEC = 30.0


@dataclass
class Verdict:
    same_performance: bool
    residual_pct: float
    spread: float
    slope: float
    r_squared: float
    duration_a: float
    duration_b: float
    note: str


def fingerprint(path: str | Path) -> np.ndarray:
    """CENS chroma at ~10.8 frames per second, L2-normalised."""
    y, sr = librosa.load(path, sr=COARSE_SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=35.0)
    chroma = librosa.feature.chroma_cens(
        y=y, sr=sr, hop_length=COARSE_HOP, win_len_smooth=21
    )
    return librosa.util.normalize(chroma, norm=2, axis=0)


SEC_PER_FRAME = COARSE_HOP / COARSE_SR


def _nan_verdict(dur_a: float, dur_b: float, note: str) -> Verdict:
    nan = float("nan")
    return Verdict(False, nan, nan, nan, nan, dur_a, dur_b, note)


def compare(path_a: str | Path, path_b: str | Path) -> Verdict:
    fa, fb = fingerprint(path_a), fingerprint(path_b)
    dur_a = fa.shape[1] * SEC_PER_FRAME
    dur_b = fb.shape[1] * SEC_PER_FRAME

    if min(fa.shape[1], fb.shape[1]) < 20 / SEC_PER_FRAME:
        return _nan_verdict(dur_a, dur_b, "too short to compare")

    # A large length ratio means one file is an excerpt or a single 78rpm side. DTW
    # aligns a fragment to part of the whole and never fails loudly.
    ratio = max(dur_a, dur_b) / max(min(dur_a, dur_b), 1e-6)
    if ratio > 1.35:
        return _nan_verdict(dur_a, dur_b, f"length ratio {ratio:.2f} - fragment vs whole")

    _, path = librosa.sequence.dtw(
        X=fa, Y=fb, metric="cosine", step_sizes_sigma=STEP_SIZES,
        weights_mul=np.array([1.0, 1.0, 1.0]), subseq=False,
    )
    path = path[::-1]
    t_a = path[:, 0] * SEC_PER_FRAME
    t_b = path[:, 1] * SEC_PER_FRAME

    slope, intercept = np.polyfit(t_a, t_b, 1)
    residual = t_b - (slope * t_a + intercept)
    r2 = float(1 - np.var(residual) / max(np.var(t_b), 1e-9))
    residual_pct = float(100 * np.sqrt(np.mean(residual**2)) / max(t_a[-1], 1e-9))

    ratios = []
    start = float(t_a[0])
    while start + WINDOW_SEC <= t_a[-1]:
        i = int(np.searchsorted(t_a, start))
        j = int(np.searchsorted(t_a, start + WINDOW_SEC))
        if j > i and (t_a[j] - t_a[i]) > 1e-6:
            ratios.append((t_b[j] - t_b[i]) / (t_a[j] - t_a[i]))
        start += WINDOW_SEC / 2
    spread = (
        float((np.asarray(ratios) / np.median(ratios)).std())
        if ratios else float("nan")
    )

    # Primary test: playing a disc faster makes it shorter and higher by the same
    # factor, so for one performance at two transfer speeds pitch and duration must
    # agree. This replaced the DTW residual, which is reference-dependent: one Cortot
    # performance against two transfers of another scored 0.241 ("same") and 2.746
    # ("different"), conflating a different performance with a large speed offset. Pitch
    # caught that case, saying the file should be 1% shorter where it was 2% longer.
    pitch_ok, pitch_note, semitones = _pitch_agrees(path_a, path_b, dur_a, dur_b)

    same = pitch_ok and residual_pct < RESIDUAL_THRESHOLD
    if same:
        note = "same performance"
        if abs(slope - 1) > 0.02:
            note += f"; transfer speed differs {100*(slope-1):+.1f}%"
    elif pitch_ok:
        note = f"different performances (shape; {pitch_note})"
    else:
        note = f"different performances ({pitch_note})"

    # A slope far from the duration ratio means one excursion is dragging the fit.
    expected = dur_b / max(dur_a, 1e-9)
    if abs(slope - expected) / expected > 0.03:
        note += f" [slope {slope:.3f} vs length ratio {expected:.3f} - residual unreliable]"
    if ratios and same and spread > SPREAD_THRESHOLD:
        note += " (statistics disagree - inspect)"
    return Verdict(same, residual_pct, spread, float(slope), r2, dur_a, dur_b, note)


PITCH_DURATION_TOLERANCE = 0.015  # measured: 0.009 agree, 0.030+ disagree


def _pitch_agrees(
    path_a: str | Path, path_b: str | Path, dur_a: float, dur_b: float
) -> tuple[bool, str, float]:
    """Check pitch offset against duration ratio.

    Needs a coarse and a fine estimate: librosa's tuning estimate wraps into +/-0.5
    semitones, and a chroma rotation recovers only whole semitones.
    """
    import librosa as _lr

    ya, sra = _lr.load(path_a, sr=COARSE_SR, mono=True)
    yb, srb = _lr.load(path_b, sr=COARSE_SR, mono=True)
    ca = _lr.util.normalize(_lr.feature.chroma_cens(y=ya, sr=sra), norm=2, axis=0).mean(axis=1)
    cb = _lr.util.normalize(_lr.feature.chroma_cens(y=yb, sr=srb), norm=2, axis=0).mean(axis=1)

    scores = [float(np.dot(ca, np.roll(cb, -k))) for k in range(12)]
    coarse = int(np.argmax(scores))
    if coarse > 6:
        coarse -= 12
    fine = float(_lr.estimate_tuning(y=yb, sr=srb)) - float(_lr.estimate_tuning(y=ya, sr=sra))
    semitones = coarse + fine

    implied = 1.0 / float(2 ** (semitones / 12.0))  # lower pitch -> longer
    actual = dur_b / max(dur_a, 1e-9)
    error = abs(implied - actual) / max(actual, 1e-9)
    ok = error <= PITCH_DURATION_TOLERANCE
    note = (
        f"pitch {semitones:+.2f}st implies length x{implied:.3f}, actual x{actual:.3f}"
        f"{'' if ok else f' - off by {100*error:.1f}%'}"
    )
    return ok, note, semitones


def calibrate(known_same: list[tuple], known_different: list[tuple]) -> dict:
    """Re-derive the threshold from labelled pairs, rather than trusting one measured
    on a single pair."""
    same = [compare(*p).residual_pct for p in known_same]
    diff = [compare(*p).residual_pct for p in known_different]
    gap = (max(same), min(diff)) if same and diff else (float("nan"),) * 2
    return {
        "same_spreads": same,
        "different_spreads": diff,
        "separating_band": gap,
        "separable": bool(same and diff and max(same) < min(diff)),
        "suggested_threshold": float(np.mean(gap)) if same and diff else float("nan"),
    }
