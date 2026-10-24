"""Kill test on audio: is a pianist closer to themselves than to other pianists?

Timing differences between pianists exist and are measurable above the method's noise
floor, but that is not the claim the project rests on. The claim is that those
differences identify a person, so that one pianist is more consistent with themselves
across separate recordings than with anyone else.

Three distances, all on Chopin's Ballade No. 1 so the music is held constant: two
transfers of one performance (measurement error), one pianist in two recordings years
apart, and two different pianists. The project needs identity < within < between, with
within clearly below between. If within is as large as between, a pianist varies more
between their own takes than pianists vary from each other.

The distance is the de-trended tempo-ratio spread at a 30 s window, not the DTW
residual, which is reference-dependent: it scored the same comparison 0.241 and 2.746
depending only on which transfer was the reference.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pianist_id import align

AUDIO = ROOT / "data" / "audio"
OUT = ROOT / "results"
CACHE = OUT / "dtw_cache_op23"

# (key, file, pianist, year)
TRACKS = [
    ("cortot26a", "ballade1_cortot_1926a.flac", "Cortot", 1926),
    ("cortot26b", "ballade1_cortot_1926b.mp3", "Cortot", 1926),
    ("cortot33", "ballade1_cortot_1933.mp3", "Cortot", 1933),
    ("hofmann37", "ballade1_hofmann_1937.mp3", "Hofmann", 1937),
    ("francois46", "ballade1_francois_1946.flac", "Francois", 1946),
    ("janis52", "ballade1_janis_1952.flac", "Janis", 1952),
    ("rubinstein59", "ballade1_rubinstein_1959.mp3", "Rubinstein", 1959),
    ("richter60", "ballade1_richter_1960.mp3", "Richter", 1960),
    ("michelangeli63", "ballade1_michelangeli_1963.flac", "Michelangeli", 1963),
    ("horowitz65", "ballade1_horowitz_1965.mp3", "Horowitz", 1965),
]

# Established by the transfer-speed check: one performance, two digitisations.
IDENTITY = {("cortot26a", "cortot26b")}
WINDOW = 30.0


def cached_path(ka: str, kb: str, tracks: dict) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / f"{ka}__{kb}.npy"
    if dest.exists():
        return np.load(dest)
    path, _ = align.align(tracks[ka], tracks[kb])
    np.save(dest, path)
    return path


def main() -> int:
    OUT.mkdir(exist_ok=True)
    tracks, meta = {}, {}
    for key, filename, pianist, year in TRACKS:
        path = AUDIO / filename
        if not path.exists():
            print(f"missing {path}")
            return 1
        tracks[key] = align.load(path, name=f"{pianist} {year}")
        meta[key] = (pianist, year)

    rows = []
    keys = [t[0] for t in TRACKS]
    total = len(keys) * (len(keys) - 1) // 2
    for n, (ka, kb) in enumerate(itertools.combinations(keys, 2), 1):
        path = cached_path(ka, kb, tracks)
        spread = align.tempo_ratio_spread(path, window_sec=WINDOW)
        pa, ya = meta[ka]
        pb, yb = meta[kb]
        if (ka, kb) in IDENTITY or (kb, ka) in IDENTITY:
            kind = "identity"
        elif pa == pb:
            kind = "within"
        else:
            kind = "between"
        rows.append(
            {"a": ka, "b": kb, "pianist_a": pa, "pianist_b": pb,
             "year_a": ya, "year_b": yb, "kind": kind, "spread": spread,
             "year_gap": abs(ya - yb)}
        )
        print(f"  [{n:>2}/{total}] {pa} {ya} vs {pb} {yb:<6} {kind:<9} {spread:.4f}")

    groups = {k: [r["spread"] for r in rows if r["kind"] == k]
              for k in ("identity", "within", "between")}

    print()
    print("=" * 78)
    print(f"RESULT  (de-trended tempo spread, {WINDOW:.0f}s window)")
    print("=" * 78)
    for kind in ("identity", "within", "between"):
        vals = groups[kind]
        if not vals:
            print(f"  {kind:<10} (none)")
            continue
        print(f"  {kind:<10} n={len(vals):<3} "
              f"min {min(vals):.4f}  median {np.median(vals):.4f}  max {max(vals):.4f}")

    within, between = groups["within"], groups["between"]

    # Replicate check, before any verdict. Two transfers of one performance measured
    # against the same third recording are the same comparison and must agree. When
    # they do not, the distance measure is too unstable to compare anything.
    replicates = {}
    for row in rows:
        if row["kind"] not in ("within", "between"):
            continue
        pair = tuple(sorted((row["pianist_a"], row["pianist_b"])))
        years = tuple(sorted((row["year_a"], row["year_b"])))
        replicates.setdefault((pair, years), []).append(row["spread"])
    worst = max(
        ((max(v) / min(v), k) for k, v in replicates.items() if len(v) > 1 and min(v) > 0),
        default=(1.0, None),
    )
    print()
    if worst[1] is not None:
        print(f"  replicate check: same comparison measured {worst[0]:.1f}x apart "
              f"via different transfers {worst[1][0]}")

    print()
    if not within:
        print("  INCONCLUSIVE: no within-pianist pair available.")
    elif worst[0] > 2.0:
        print("  INCONCLUSIVE. The distance measure is unstable: the same")
        print("  comparison, measured through two transfers of one performance,")
        print(f"  differs by {worst[0]:.1f}x. Any within-vs-between verdict built on")
        print("  it would be an artefact. Speed-normalise every pair and re-run.")
    else:
        overlap = sum(1 for w in within if w >= min(between))
        print(f"  worst within-pianist pair   {max(within):.4f}")
        print(f"  closest between-pianist pair {min(between):.4f}")
        print(f"  separation                   {min(between)/max(within):.2f}x")
        print()
        if max(within) < min(between):
            print("  PASS. Every within-pianist distance is smaller than every")
            print("  between-pianist distance. A pianist is more consistent with")
            print("  themselves than with anyone else, which is the premise the")
            print("  classifier depends on.")
        elif overlap < len(within):
            print("  PARTIAL. The distributions overlap. Timing shape carries")
            print("  identity information but does not separate cleanly on its own;")
            print("  the per-note features have to do real work.")
        else:
            print("  FAIL. Within-pianist distances are not smaller than")
            print("  between-pianist distances. On this evidence a pianist varies")
            print("  between their own takes as much as pianists differ from each")
            print("  other, and identification from timing shape alone is not")
            print("  supported. Do not build per-note features on these.")

    (OUT / "within_between_audio.json").write_text(
        json.dumps({"window_sec": WINDOW, "pairs": rows}, indent=2), encoding="utf-8"
    )

    # --- figure ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.2))
    colours = {"identity": "C2", "within": "C3", "between": "C0"}
    for i, kind in enumerate(("identity", "within", "between")):
        vals = groups[kind]
        if not vals:
            continue
        ax.scatter(vals, np.full(len(vals), i) + np.random.uniform(-.07, .07, len(vals)),
                   s=70, alpha=.8, color=colours[kind], label=f"{kind} (n={len(vals)})")
    if within and between:
        ax.axvline(max(within), ls="--", lw=1, color="C3")
        ax.axvline(min(between), ls="--", lw=1, color="C0")
    ax.set_yticks(range(3))
    ax.set_yticklabels(["identity\n(same take)", "within\n(same pianist)", "between\n(different)"])
    ax.set_xlabel(f"de-trended tempo-ratio spread, {WINDOW:.0f}s window")
    ax.set_title("Is a pianist closer to themselves than to other pianists?")
    ax.grid(alpha=.3, axis="x")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "within_between_audio.png", dpi=150)
    print(f"\n  -> results/within_between_audio.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
