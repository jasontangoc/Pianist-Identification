"""Kill test: does interpretive timing signal exist, and can it be measured?

"Plot both tempo curves and see whether they look different" cannot pass or fail
honestly, because DTW never fails loudly: a broken alignment also produces two
different-looking curves. So this measures two things instead. The noise floor comes
from two independent transfers of one Cortot performance, where the true interpretive
difference is zero and whatever spread comes back is the method's own error. The signal
comes from pairs of genuinely different pianists.

The project is viable only if signal clears floor by a comfortable margin, and that
margin is the precision budget for every feature downstream: a feature whose
between-pianist variation is smaller than this floor cannot carry information.

    python scripts/fetch_audio.py --killtest
    python scripts/killtest_tempo_curves.py
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
CACHE = OUT / "dtw_cache"

TRACKS = {
    "cortot26a": ("ballade1_cortot_1926a.flac", "Cortot 1926"),
    "cortot26b": ("ballade1_cortot_1926b.mp3", "Cortot 1926 (2nd transfer)"),
    "janis52": ("ballade1_janis_1952.flac", "Janis 1952"),
    "michelangeli63": ("ballade1_michelangeli_1963.flac", "Michelangeli 1963"),
}

IDENTITY_PAIR = ("cortot26a", "cortot26b")
WINDOWS = (2.0, 4.0, 8.0, 15.0, 30.0)
DECISION_WINDOW = 30.0


def cached_align(key_a: str, key_b: str, tracks: dict) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / f"{key_a}__{key_b}.npy"
    if dest.exists():
        return np.load(dest)
    path, _ = align.align(tracks[key_a], tracks[key_b])
    np.save(dest, path)
    return path


def main() -> int:
    OUT.mkdir(exist_ok=True)

    tracks = {}
    for key, (filename, label) in TRACKS.items():
        path = AUDIO / filename
        if not path.exists():
            print(f"missing {path}\nrun: python scripts/fetch_audio.py --killtest")
            return 1
        tracks[key] = align.load(path, name=label)

    pairs = [IDENTITY_PAIR] + [
        p for p in itertools.combinations(TRACKS, 2) if p != IDENTITY_PAIR
        and not (p[0] in IDENTITY_PAIR and p[1] in IDENTITY_PAIR)
    ]

    results = {}
    print("aligning (cached in results/dtw_cache)...")
    for a, b in pairs:
        path = cached_align(a, b, tracks)
        sweep = align.spread_vs_window(path, WINDOWS)
        lin = align.path_linearity(path)
        results[f"{a}|{b}"] = {
            "pair": [tracks[a].name, tracks[b].name],
            "is_identity": (a, b) == IDENTITY_PAIR,
            "global_slope": lin["slope"],
            "r_squared": lin["r_squared"],
            "spread": {str(k): v for k, v in sweep.items()},
            "path_len": int(len(path)),
        }
        print(f"  {tracks[a].name:<28} vs {tracks[b].name:<28} done")

    # --- table -----------------------------------------------------------
    print()
    print("=" * 94)
    print("DE-TRENDED TEMPO-RATIO SPREAD  (global speed offset removed)")
    print("=" * 94)
    print(f"{'pair':<58}" + "".join(f"{w:>7.0f}s" for w in WINDOWS))
    print("-" * 94)
    for key, r in results.items():
        tag = "  [IDENTITY]" if r["is_identity"] else ""
        label = f"{r['pair'][0]} vs {r['pair'][1]}{tag}"
        row = "".join(f"{r['spread'][str(w)]:>8.4f}" for w in WINDOWS)
        print(f"{label:<58}{row}")

    floor = results[f"{IDENTITY_PAIR[0]}|{IDENTITY_PAIR[1]}"]["spread"][str(DECISION_WINDOW)]
    contrasts = {
        k: r["spread"][str(DECISION_WINDOW)]
        for k, r in results.items()
        if not r["is_identity"]
    }
    weakest = min(contrasts.values())
    margin = weakest / floor

    print()
    print("=" * 94)
    print(f"GO / NO-GO   (decision window {DECISION_WINDOW:.0f}s)")
    print("=" * 94)
    print(f"  noise floor (same performance, two transfers)   {floor:.4f}")
    print(f"  weakest between-pianist pair                    {weakest:.4f}")
    print(f"  strongest between-pianist pair                  {max(contrasts.values()):.4f}")
    print(f"  margin (weakest signal / floor)                 {margin:.2f}x")
    print()
    if margin >= 2.0:
        print("  GO. Between-pianist timing difference exceeds the method's own")
        print("  error by a clear margin, on the hardest pair available. The")
        print("  alignment works, and the signal the project depends on is real")
        print("  and measurable.")
    elif margin >= 1.5:
        print("  MARGINAL. Signal clears the floor but not comfortably. Reduce the")
        print("  floor before building on this -- the floor here is dominated by")
        print("  78rpm turntable drift, so a cleaner source pair would raise the")
        print("  margin without changing the question.")
    else:
        print("  NO-GO. Between-pianist difference is not separable from measurement")
        print("  error at this precision. Do not build per-note features on")
        print("  this material.")

    (OUT / "killtest_tempo_curves.json").write_text(
        json.dumps(
            {
                "decision_window_sec": DECISION_WINDOW,
                "noise_floor": floor,
                "weakest_contrast": weakest,
                "margin": margin,
                "pairs": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # --- figure ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))

    # left: tempo curves, identity against contrast, same axes
    for key, r in results.items():
        a, b = key.split("|")
        path = cached_align(a, b, tracks)
        centres, ratios = align.tempo_ratio_curve(path, window_sec=DECISION_WINDOW)
        norm = ratios / np.median(ratios)
        style = dict(lw=2.4, color="k", zorder=5) if r["is_identity"] else dict(lw=1.3)
        axes[0].plot(
            centres, norm,
            label=f"{r['pair'][0]} / {r['pair'][1]}" + (" (identity)" if r["is_identity"] else ""),
            **style,
        )
    axes[0].axhline(1.0, color="grey", lw=0.8, ls=":")
    axes[0].set_xlabel("time through the Ballade (s)")
    axes[0].set_ylabel("local tempo ratio, speed offset removed")
    axes[0].set_title(f"Where pianists disagree about time ({DECISION_WINDOW:.0f}s window)")
    axes[0].legend(fontsize=7.5, loc="upper left")
    axes[0].grid(alpha=0.3)

    # right: spread against window, which is the noise/signal separation
    for key, r in results.items():
        style = dict(lw=2.4, color="k", marker="o", zorder=5) if r["is_identity"] else dict(lw=1.3, marker=".")
        axes[1].plot(
            WINDOWS, [r["spread"][str(w)] for w in WINDOWS],
            label=f"{r['pair'][0]} / {r['pair'][1]}" + (" (identity)" if r["is_identity"] else ""),
            **style,
        )
    axes[1].set_xscale("log")
    axes[1].set_xticks(WINDOWS)
    axes[1].set_xticklabels([f"{w:.0f}" for w in WINDOWS])
    axes[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[1].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axes[1].set_xlabel("analysis window (s)")
    axes[1].set_ylabel("de-trended spread (sd)")
    axes[1].set_title("Noise falls with window; real difference persists")
    axes[1].legend(fontsize=7.5)
    axes[1].grid(alpha=0.3, which="both")

    fig.suptitle(
        f"Kill test -- margin {margin:.2f}x over the method's noise floor", y=0.99
    )
    fig.tight_layout()
    dest = OUT / "killtest_tempo_curves.png"
    fig.savefig(dest, dpi=150)
    print(f"\n  figure -> {dest.relative_to(ROOT)}")
    print(f"  data   -> results/killtest_tempo_curves.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
