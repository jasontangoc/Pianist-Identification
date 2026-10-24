"""Per-performance sequences: the timing curve itself, not summaries of it.

Whole-piece summaries of timing variability carry no measurable identity while per-chord
melodic quantities do, which suggests a summary discards where the stretching happens.
This builds the representation to test that: four channels per chord event, all
normalised inside the performance, so a faster reading or a faster transfer gives
identical sequences.

    log_ioi      log interval to the next chord, over the performance's own median
    asynchrony   chord spread, in beats
    lead         melody lead, in beats
    velocity     loudness, z-scored within the performance

Sequences are cut into overlapping windows and the model classifies windows, averaging
probabilities to a performance. Position in a Liszt etude means nothing in a Chopin
ballade, so windows plus global pooling force the model onto local gestures, which is
the only thing that can transfer between pieces; 303 performances is also far too few to
train a convolutional net, where the windows number in the thousands.

Windows from one performance are near-duplicates and must never be split across a
train/test boundary. The caller groups by performance, and `window_index` is returned to
make that enforceable.
"""

from __future__ import annotations

import numpy as np

from . import features as F

CHANNELS = ["log_ioi", "asynchrony", "lead", "velocity"]


def chord_series(notes: list[F.Note], window_beats: float = 0.4) -> dict[str, np.ndarray]:
    """Per-chord-event series for one performance."""
    pb = F.provisional_beat(notes)
    chord_window = window_beats * pb if np.isfinite(pb) else 0.05
    groups = F.group_chords(notes, window=chord_window)
    for _ in range(3):
        beat_est = F.local_beat(groups)
        if not np.isfinite(beat_est) or beat_est <= 0:
            break
        new_window = window_beats * beat_est
        if abs(new_window - chord_window) < 1e-4:
            break
        chord_window = new_window
        groups = F.group_chords(notes, window=chord_window)

    beat = F.local_beat(groups)
    if not np.isfinite(beat) or beat <= 0 or len(groups) < 8:
        return {}

    onsets = np.array([g[0].onset for g in groups])
    iois = np.diff(onsets)
    keep = iois > 1e-4
    iois = iois[keep]
    if len(iois) < 8:
        return {}
    groups = [g for g, k in zip(groups[:-1], keep) if k]

    med = np.median(iois)
    log_ioi = np.log(np.clip(iois / med, 0.05, 20.0))

    spread, lead, vel = [], [], []
    for g in groups:
        spread.append((g[-1].onset - g[0].onset) / beat)
        if len(g) >= 2:
            top = max(g, key=lambda n: n.pitch)
            others = [n for n in g if n is not top]
            lead.append((np.median([n.onset for n in others]) - top.onset) / beat)
        else:
            lead.append(0.0)
        vel.append(float(np.mean([n.velocity for n in g])))

    vel = np.array(vel, dtype=float)
    vel = (vel - vel.mean()) / (vel.std() + 1e-9)
    return {
        "log_ioi": log_ioi.astype(np.float32),
        "asynchrony": np.clip(np.array(spread, dtype=np.float32), 0, 3),
        "lead": np.clip(np.array(lead, dtype=np.float32), -3, 3),
        "velocity": vel.astype(np.float32),
    }


def windows(series: dict[str, np.ndarray], length: int = 64, stride: int = 32
            ) -> np.ndarray:
    """Stack channels and cut into overlapping windows, shape (n, C, length)."""
    if not series:
        return np.zeros((0, len(CHANNELS), length), dtype=np.float32)
    mat = np.stack([series[c] for c in CHANNELS], axis=0)
    n = mat.shape[1]
    if n < length:
        return np.zeros((0, len(CHANNELS), length), dtype=np.float32)
    starts = range(0, n - length + 1, stride)
    return np.stack([mat[:, s:s + length] for s in starts], axis=0).astype(np.float32)
