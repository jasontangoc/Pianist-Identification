"""Interpretive features from a note list, computed inside one performance.

Nothing is aligned against another recording, so none of the pairwise-alignment
instability applies here; performances are compared afterwards as feature vectors.
Timing features are a fraction of the performance's own beat, which leaves them
unchanged when a transfer speed rescales the time axis. `absolute_ms` variants are kept
for reporting and the model sees the normalised ones.

Two definitions are judgement calls. Chord grouping uses a fixed window in beats, and
the spread being measured is also what decides the grouping, so `extract` reports the
fraction of chords near the window edge. Melody lead assumes the melody is the top
note, which is wrong wherever the tune is in an inner voice.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Note:
    onset: float
    pitch: int
    velocity: int
    duration: float


def load_midi(path: str | Path | bytes) -> list[Note]:
    """Read a MIDI file into absolute-time note events."""
    import mido

    mid = (
        mido.MidiFile(file=io.BytesIO(path))
        if isinstance(path, bytes)
        else mido.MidiFile(str(path))
    )
    notes: list[Note] = []
    pending: dict[int, list[tuple[float, int]]] = {}
    t = 0.0
    for msg in mid:  # iterating the file yields delta times in seconds
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            pending.setdefault(msg.note, []).append((t, msg.velocity))
        elif msg.type in ("note_off",) or (msg.type == "note_on" and msg.velocity == 0):
            stack = pending.get(msg.note)
            if stack:
                onset, vel = stack.pop(0)
                notes.append(Note(onset, msg.note, vel, t - onset))
    for pitch, stack in pending.items():  # unterminated notes
        for onset, vel in stack:
            notes.append(Note(onset, pitch, vel, 0.0))
    notes.sort(key=lambda n: (n.onset, n.pitch))
    return notes


def provisional_beat(notes: list[Note]) -> float:
    """Grouping-free tempo estimate, used to size the chord window.

    The window should scale with tempo, but the tempo estimate comes from the chord
    grouping, so this breaks the circularity. Gaps below 10 ms are within-chord.
    """
    if len(notes) < 3:
        return float("nan")
    onsets = np.unique(np.array([n.onset for n in notes]))
    gaps = np.diff(onsets)
    gaps = gaps[gaps > 0.01]
    if not len(gaps):
        return float("nan")
    # The distribution is bimodal and only the upper mode is the beat; the median falls
    # between the modes and sizes the window too small to hold the chords being measured.
    return float(np.percentile(gaps, 75))


def group_chords(
    notes: list[Note], window: float | None = None, window_beats: float = 0.4
) -> list[list[Note]]:
    """Cluster notes into simultaneity groups by onset proximity.

    Single-linkage: a note joins the current group if it falls within the window of the
    previous note, not of the group start, so a rolled chord stays together.

    The window is a fraction of the performance's own beat. A fixed millisecond window
    breaks scale invariance, since rescaling the time axis grows every gap while the
    window stays put; validation caught a fixed 50 ms window scaling `beat_sec` by
    1.0397 instead of 1.05. The cost is that a slower performance gets a wider window.
    """
    if not notes:
        return []
    if window is None:
        beat = provisional_beat(notes)
        window = window_beats * beat if np.isfinite(beat) else 0.05
    groups, current = [], [notes[0]]
    for prev, note in zip(notes, notes[1:]):
        if note.onset - prev.onset <= window:
            current.append(note)
        else:
            groups.append(current)
            current = [note]
    groups.append(current)
    return groups


def local_beat(groups: list[list[Note]]) -> float:
    """The performance's own time unit: median interval between chord onsets.

    Used as the denominator for every normalised feature. The median rather
    than the mean because rubato makes the distribution heavy-tailed, and a
    single long pause should not rescale the whole piece.
    """
    onsets = np.array([g[0].onset for g in groups])
    if len(onsets) < 2:
        return float("nan")
    iois = np.diff(onsets)
    iois = iois[iois > 1e-4]
    return float(np.median(iois)) if len(iois) else float("nan")


def chord_asynchrony(groups: list[list[Note]], beat: float, window: float) -> dict:
    """Spread of onsets within chords, absolute and beat-normalised."""
    spreads = [g[-1].onset - g[0].onset for g in groups if len(g) >= 2]
    if not spreads:
        return {"n_chords": 0}
    spreads = np.array(spreads)
    # If many chords sit near the window, the window is deciding the feature.
    near_edge = float(np.mean(spreads > 0.8 * window))
    return {
        "n_chords": len(spreads),
        "asynchrony_ms_mean": float(spreads.mean() * 1000),
        "asynchrony_ms_median": float(np.median(spreads) * 1000),
        "asynchrony_beats_mean": float((spreads / beat).mean()),
        "asynchrony_beats_sd": float((spreads / beat).std()),
        "frac_near_window_edge": near_edge,
    }


def melody_lead(groups: list[list[Note]], beat: float) -> dict:
    """How far the top note precedes the rest of its chord. Positive is melody-first."""
    leads = []
    for g in groups:
        if len(g) < 2:
            continue
        top = max(g, key=lambda n: n.pitch)
        others = [n for n in g if n is not top]
        if not others:
            continue
        leads.append(np.median([n.onset for n in others]) - top.onset)
    if not leads:
        return {"n_melody_chords": 0}
    leads = np.array(leads)
    return {
        "n_melody_chords": len(leads),
        "melody_lead_ms_mean": float(leads.mean() * 1000),
        "melody_lead_ms_median": float(np.median(leads) * 1000),
        "melody_lead_beats_mean": float((leads / beat).mean()),
        "melody_lead_beats_sd": float((leads / beat).std()),
        "frac_melody_first": float(np.mean(leads > 0)),
    }


def timing_variability(groups: list[list[Note]], beat: float) -> dict:
    """How unevenly chord onsets are spaced, over the performance's own median interval."""
    onsets = np.array([g[0].onset for g in groups])
    iois = np.diff(onsets)
    iois = iois[iois > 1e-4]
    if len(iois) < 10:
        return {"n_iois": len(iois)}
    r = iois / beat
    log_r = np.log(r)
    return {
        "n_iois": len(iois),
        "ioi_cv": float(iois.std() / iois.mean()),
        "ioi_log_sd": float(log_r.std()),
        "ioi_ratio_p90_p10": float(np.percentile(r, 90) / max(np.percentile(r, 10), 1e-9)),
    }


def dynamics(notes: list[Note], groups: list[list[Note]]) -> dict:
    """Velocity shaping. Unaffected by the time axis, so no normalisation."""
    vel = np.array([n.velocity for n in notes], dtype=float)
    if len(vel) < 10:
        return {}
    tops = np.array([max(g, key=lambda n: n.pitch).velocity for g in groups if len(g) >= 2],
                    dtype=float)
    rest = np.array([np.median([n.velocity for n in g if n is not max(g, key=lambda x: x.pitch)])
                     for g in groups if len(g) >= 2], dtype=float)
    out = {
        "velocity_mean": float(vel.mean()),
        "velocity_sd": float(vel.std()),
        "velocity_p90_p10": float(np.percentile(vel, 90) - np.percentile(vel, 10)),
    }
    if len(tops) and len(rest):
        # The dynamic counterpart of melody lead.
        out["melody_prominence"] = float(np.mean(tops - rest))
    return out


def extract(notes: list[Note], window_beats: float = 0.4) -> dict:
    """Full feature vector for one performance.

    `window_beats` sizes the chord window as a fraction of the performance's own beat,
    which is what makes the vector scale-invariant.
    """
    # Iterate the window to convergence. The provisional beat comes from raw onset gaps,
    # which include the within-chord gaps asynchrony creates, so a large asynchrony
    # shrinks the window and splits the chords being measured: validation caught an
    # injected 30 ms melody lead coming back as -1.37 ms. Re-estimating from grouped
    # onsets removes the contamination in two or three passes.
    pb = provisional_beat(notes)
    chord_window = window_beats * pb if np.isfinite(pb) else 0.05
    groups = group_chords(notes, window=chord_window)
    for _ in range(3):
        beat_est = local_beat(groups)
        if not np.isfinite(beat_est) or beat_est <= 0:
            break
        new_window = window_beats * beat_est
        if abs(new_window - chord_window) < 1e-4:
            break
        chord_window = new_window
        groups = group_chords(notes, window=chord_window)
    beat = local_beat(groups)
    if not np.isfinite(beat) or beat <= 0:
        return {"n_notes": len(notes), "error": "no usable beat"}
    out = {
        "n_notes": len(notes),
        "n_groups": len(groups),
        "beat_sec": beat,
        "duration_sec": notes[-1].onset - notes[0].onset if notes else 0.0,
        "chord_window_sec": chord_window,
        "window_beats": window_beats,
    }
    out.update(chord_asynchrony(groups, beat, chord_window))
    out.update(melody_lead(groups, beat))
    out.update(timing_variability(groups, beat))
    out.update(dynamics(notes, groups))
    return out


NORMALISED_KEYS = [
    "asynchrony_beats_mean", "asynchrony_beats_sd",
    "melody_lead_beats_mean", "melody_lead_beats_sd", "frac_melody_first",
    "ioi_cv", "ioi_log_sd", "ioi_ratio_p90_p10",
    "velocity_sd", "velocity_p90_p10", "melody_prominence",
]


def vector(features: dict, keys: list[str] | None = None) -> np.ndarray:
    keys = keys or NORMALISED_KEYS
    return np.array([features.get(k, np.nan) for k in keys], dtype=float)


def time_scale(notes: list[Note], factor: float) -> list[Note]:
    """Rescale the time axis, to check that normalised features do not move."""
    return [Note(n.onset * factor, n.pitch, n.velocity, n.duration * factor) for n in notes]
