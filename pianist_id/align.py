"""Loading, chroma features, and DTW alignment for cross-version comparison.

CENS chroma rather than raw CQT chroma: chroma discards timbre, hall and most of the
mastering, which are the confounds this project controls for, and CENS adds the
quantisation and energy normalisation that make 1920s shellac comparable with a studio
tape. Cosine distance, because after CENS only the direction of the vector carries
harmony. Step sizes {(1,1), (1,2), (2,1)} rather than the default, which allows one
frame of A to match unboundedly many frames of B and reads as infinite rubato.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

SR = 22050
HOP = 512  # 23.2 ms at 22.05 kHz

# Forbids degenerate horizontal/vertical runs; bounds local slope to [1/2, 2].
STEP_SIZES = np.array([[1, 1], [1, 2], [2, 1]])


@dataclass
class Track:
    name: str
    y: np.ndarray
    sr: int
    trim_offset: float  # seconds of leading silence removed
    raw_duration: float  # before trimming

    @property
    def duration(self) -> float:
        return len(self.y) / self.sr


def load(path: str | Path, name: str | None = None, top_db: float = 35.0) -> Track:
    """Load, downmix to mono, resample, and trim leading/trailing silence.

    `top_db` is loose on purpose: a shellac noise floor sits tens of dB above digital
    silence, so a tight threshold trims the studio tape and not the 78, which
    manufactures a duration difference between them.
    """
    path = Path(path)
    y, sr = librosa.load(path, sr=SR, mono=True)
    raw_duration = len(y) / sr
    y_trim, index = librosa.effects.trim(y, top_db=top_db)
    return Track(
        name=name or path.stem,
        y=y_trim,
        sr=sr,
        trim_offset=index[0] / sr,
        raw_duration=raw_duration,
    )


def cens(track: Track, win_len_smooth: int = 41) -> np.ndarray:
    """CENS chroma, L2-normalised per frame, shape (12, n_frames)."""
    chroma = librosa.feature.chroma_cens(
        y=track.y, sr=track.sr, hop_length=HOP, win_len_smooth=win_len_smooth
    )
    return librosa.util.normalize(chroma, norm=2, axis=0)


def frames_to_time(n: np.ndarray | int) -> np.ndarray:
    return librosa.frames_to_time(n, sr=SR, hop_length=HOP)


def semitone_offset(track_a: Track, track_b: Track) -> tuple[float, int, float]:
    """Estimate how many semitones B sounds above A, as (total, coarse, fine).

    The test for a transfer-speed difference, and it needs both parts. A whole-semitone
    shift rotates the mean chroma vector by one bin, which cross-correlation recovers
    (6% speed is almost exactly one semitone). librosa's tuning estimate resolves
    sub-semitone deviation but wraps into [-0.5, +0.5), so alone it reports ~0 for a
    one-semitone shift. A 6% difference is invisible to the fine estimate, a 2% one to
    the coarse.
    """
    mean_a = cens(track_a).mean(axis=1)
    mean_b = cens(track_b).mean(axis=1)

    # Circular cross-correlation over the 12 rotations of B against A.
    scores = [float(np.dot(mean_a, np.roll(mean_b, -k))) for k in range(12)]
    coarse = int(np.argmax(scores))
    if coarse > 6:  # a rotation of 11 is a shift of -1, not +11
        coarse -= 12

    fine_a = float(librosa.estimate_tuning(y=track_a.y, sr=track_a.sr))
    fine_b = float(librosa.estimate_tuning(y=track_b.y, sr=track_b.sr))
    fine = fine_b - fine_a

    return coarse + fine, coarse, fine


def speed_ratio_from_pitch(semitones: float) -> float:
    """Playback speed ratio implied by a pitch shift of `semitones`."""
    return float(2 ** (semitones / 12.0))


def align(track_a: Track, track_b: Track) -> tuple[np.ndarray, np.ndarray]:
    """DTW-align two tracks on CENS chroma.

    Returns (path, cost_matrix) with path as ascending (frame_a, frame_b).
    """
    chroma_a = cens(track_a)
    chroma_b = cens(track_b)
    cost, path = librosa.sequence.dtw(
        X=chroma_a,
        Y=chroma_b,
        metric="cosine",
        step_sizes_sigma=STEP_SIZES,
        weights_mul=np.array([1.0, 1.0, 1.0]),
        subseq=False,
    )
    return path[::-1], cost  # librosa returns the path backwards


def tempo_ratio_curve(
    path: np.ndarray, window_sec: float = 8.0, overlap: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Local tempo ratio of B relative to A along the alignment.

    The path local slope is the relative tempo: where B spends more frames than A per
    unit of music, B is slower.

    The window is in seconds of A, not path steps; the path advances a variable number
    of frames per step, so a fixed step count is a window of unpredictable duration.
    Window length is not cosmetic either, since differentiating a staircase amplifies
    quantisation noise as the window shrinks, and below about 8 s two transfers of one
    performance look as different as two pianists.

    Returns (time_in_a_seconds, ratio), ratio > 1 meaning B is slower.
    """
    t_a = frames_to_time(path[:, 0])
    t_b = frames_to_time(path[:, 1])

    step = window_sec * (1.0 - overlap)
    centres, ratios = [], []
    start = float(t_a[0])
    while start + window_sec <= t_a[-1]:
        i = int(np.searchsorted(t_a, start))
        j = int(np.searchsorted(t_a, start + window_sec))
        if j > i and (t_a[j] - t_a[i]) > 1e-6:
            ratios.append((t_b[j] - t_b[i]) / (t_a[j] - t_a[i]))
            centres.append(start + window_sec / 2)
        start += step
    return np.asarray(centres), np.asarray(ratios)


def tempo_ratio_spread(path: np.ndarray, window_sec: float = 30.0) -> float:
    """Standard deviation of the de-trended local tempo ratio.

    Dividing by the median removes the constant speed offset, whether a transfer speed
    or one pianist being uniformly brisker, leaving how much the two disagree about
    where to spend time. Window-dependent: on the Ballade material the noise floor is
    about 0.05 at 30 s and different pianists sit at 0.11-0.18.
    """
    _, ratios = tempo_ratio_curve(path, window_sec=window_sec)
    if ratios.size == 0:
        return float("nan")
    return float((ratios / np.median(ratios)).std())


def spread_vs_window(
    path: np.ndarray, windows: tuple[float, ...] = (2, 4, 8, 15, 30)
) -> dict[float, float]:
    """De-trended tempo-ratio spread at several window lengths.

    Separates estimator noise from a real difference. Noise falls steeply as the window
    grows, since averaging cancels staircase quantisation, while an interpretive
    difference persists. The shape of the curve is what identifies which one you have:
    at 2 s, two transfers of one performance and two different pianists look alike.
    """
    return {w: tempo_ratio_spread(path, window_sec=w) for w in windows}


def path_linearity(path: np.ndarray) -> dict[str, float]:
    """How close the warping path is to a straight line.

    A straight line means the two recordings differ by one constant factor, so one is a
    re-dub of the other at a different speed. The residual separates same take,
    different transfer from a different take.
    """
    t_a = frames_to_time(path[:, 0])
    t_b = frames_to_time(path[:, 1])
    slope, intercept = np.polyfit(t_a, t_b, 1)
    predicted = slope * t_a + intercept
    residual = t_b - predicted
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "residual_rms": float(np.sqrt(np.mean(residual**2))),
        "residual_max_abs": float(np.max(np.abs(residual))),
        "r_squared": float(1 - np.var(residual) / np.var(t_b)),
    }
