"""Is the 29-second gap between the two Cortot 1926 transfers a speed difference?

The two items claim the same performance (Victor, NYC, 27 Dec 1926) but run 7:59 and
8:28. Padding would be harmless. A transfer-speed difference rescales every timing
feature by a constant, which is what the scale-invariant features are built for. Or
they are not the same performance and the manifest is wrong.

Two measurements separate them: a speed change shifts pitch, and 6% is almost exactly a
semitone; and a constant DTW slope means one constant factor while a wobbling path
means the performances differ. Neither is decisive alone. Pitch cannot tell padding
from identity, and path slope cannot tell a speed change from a uniformly faster
performance, which pianists do not do but engineers do.
"""

from __future__ import annotations

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

PAIR = [
    ("ballade1_cortot_1926a.flac", "Cortot 1926 (FLAC transfer)"),
    ("ballade1_cortot_1926b.mp3", "Cortot 1926 (MP3 transfer)"),
]


def main() -> int:
    OUT.mkdir(exist_ok=True)

    tracks = []
    for filename, label in PAIR:
        path = AUDIO / filename
        if not path.exists():
            print(f"missing {path}\nrun: python scripts/fetch_audio.py --killtest")
            return 1
        tracks.append(align.load(path, name=label))

    a, b = tracks

    print("=" * 72)
    print("DURATION")
    print("=" * 72)
    for t in tracks:
        print(
            f"  {t.name:<32} raw {t.raw_duration/60:>5.2f} min   "
            f"trimmed {t.duration/60:>5.2f} min   "
            f"(lead silence {t.trim_offset:.1f}s)"
        )
    raw_ratio = b.raw_duration / a.raw_duration
    trim_ratio = b.duration / a.duration
    print(f"\n  raw duration ratio      B/A = {raw_ratio:.4f}  ({(raw_ratio-1)*100:+.2f}%)")
    print(f"  trimmed duration ratio  B/A = {trim_ratio:.4f}  ({(trim_ratio-1)*100:+.2f}%)")
    print("  -> if padding alone explains the gap, the trimmed ratio is ~1.000")

    print()
    print("=" * 72)
    print("PITCH  (does B sound higher or lower than A?)")
    print("=" * 72)
    total, coarse, fine = align.semitone_offset(a, b)
    implied = align.speed_ratio_from_pitch(total)
    print(f"  coarse (chroma rotation)  {coarse:+d} semitones")
    print(f"  fine   (tuning estimate)  {fine:+.3f} semitones")
    print(f"  total                     {total:+.3f} semitones")
    print(f"  implied speed ratio B/A   {implied:.4f}  ({(implied-1)*100:+.2f}%)")
    print("  -> a speed difference makes B both shorter AND higher, by the same factor")

    print()
    print("=" * 72)
    print("ALIGNMENT  (is the warping path a straight line?)")
    print("=" * 72)
    path, _ = align.align(a, b)
    lin = align.path_linearity(path)
    print(f"  fitted slope              {lin['slope']:.4f}")
    print(f"  R^2 against a line        {lin['r_squared']:.5f}")
    print(f"  residual RMS              {lin['residual_rms']:.3f} s")
    print(f"  residual max              {lin['residual_max_abs']:.3f} s")

    print()
    print("=" * 72)
    print("SPREAD vs WINDOW  (noise, or a real difference?)")
    print("=" * 72)
    sweep = align.spread_vs_window(path)
    print("  window  " + "".join(f"{w:>8.0f}s" for w in sweep))
    print("  spread  " + "".join(f"{v:>9.4f}" for v in sweep.values()))
    decay = sweep[2.0] / sweep[30.0] if sweep[30.0] else float("inf")
    print(f"\n  decay 2s/30s = {decay:.1f}x")
    print("  -> steep decay means the short-window spread was estimator noise")
    print("  -> a real interpretive difference persists at long windows")
    print("  reference: different pianists on this material hold 0.11-0.18 at 30s")

    # --- verdict ---------------------------------------------------------
    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    pitch_says_speed = abs(total) > 0.15  # ~0.9% speed; below this, inaudible
    duration_gap = abs(trim_ratio - 1) > 0.01

    # Absolute residual RMS is the wrong instrument: it scales with the length of the
    # piece and with turntable drift, and on the first pass it put this pair on the
    # wrong side of an arbitrary one-second line while R^2 against a straight line was
    # 0.9999. What separates same take from different take is whether the de-trended
    # spread survives a long window.
    NOISE_FLOOR = 0.08  # 30 s window, measured by the kill test
    same_take = sweep[30.0] < NOISE_FLOOR
    path_is_linear = lin["r_squared"] > 0.999 and same_take

    consistent = (
        pitch_says_speed
        and duration_gap
        and abs(implied - (1 / trim_ratio)) < 0.03
    )

    if consistent and path_is_linear:
        verdict = (
            "TRANSFER SPEED CONFIRMED. Pitch and duration disagree by a\n"
            "  consistent factor and the path is linear. Control 5 is real and\n"
            "  timing features must be made scale-invariant."
        )
    elif not duration_gap:
        verdict = (
            "PADDING ONLY. Musical durations match once silence is trimmed.\n"
            "  This pair is a clean identity control. Control 5 is unproven\n"
            "  here, though it remains a risk for other 78 transfers."
        )
    elif not pitch_says_speed and duration_gap and path_is_linear:
        verdict = (
            "UNIFORM TIMING DIFFERENCE WITHOUT PITCH SHIFT. Consistent with a\n"
            "  digital time-stretch, or with a transfer corrected for pitch.\n"
            "  Still a scale confound; control 5 applies."
        )
    elif not path_is_linear:
        verdict = (
            "NOT THE SAME PERFORMANCE. The warping path is not linear, so these\n"
            "  two items differ in interpretation, not just in transfer. The\n"
            "  manifest claim that both are the 1926 take needs re-checking,\n"
            "  and this pair CANNOT serve as the identity control."
        )
    else:
        verdict = "INCONCLUSIVE. See numbers above."
    print("  " + verdict)

    # --- figure ----------------------------------------------------------
    t_a = align.frames_to_time(path[:, 0])
    t_b = align.frames_to_time(path[:, 1])
    centres, ratios = align.tempo_ratio_curve(path, window_sec=8.0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    axes[0].plot(t_a, t_b, lw=1.6, label="DTW path")
    axes[0].plot(
        t_a, lin["slope"] * t_a + lin["intercept"], "--", lw=1.0,
        label=f"line fit (slope {lin['slope']:.3f})",
    )
    axes[0].set_xlabel(f"{a.name}  (s)")
    axes[0].set_ylabel(f"{b.name}  (s)")
    axes[0].set_title("Warping path: straight = one constant factor")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(centres, ratios, lw=1.2)
    axes[1].axhline(1.0, color="k", lw=0.8, ls=":")
    axes[1].axhline(np.median(ratios), color="C1", lw=0.9, ls="--",
                    label=f"median {np.median(ratios):.3f}")
    axes[1].set_xlabel(f"time in {a.name}  (s)")
    axes[1].set_ylabel("local tempo ratio  B / A")
    axes[1].set_title("Flat = same rubato, only a speed offset")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Identity control: two transfers of one Cortot performance", y=0.99)
    fig.tight_layout()
    dest = OUT / "transfer_speed_check.png"
    fig.savefig(dest, dpi=150)
    print(f"\n  figure -> {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
